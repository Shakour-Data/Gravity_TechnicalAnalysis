"""
Tool Recommendation Service

Provides catalog access, market-context analysis, and dynamic tool
recommendations backed by the configs/tools/catalog.json dataset.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from gravity_tech.config.settings import settings
from gravity_tech.core.contracts.analysis import AnalysisRequest
from gravity_tech.core.domain.entities import Candle
from gravity_tech.core.indicators.momentum import MomentumIndicators
from gravity_tech.core.indicators.trend import TrendIndicators
from gravity_tech.core.indicators.volatility import VolatilityIndicators
from gravity_tech.database.tse_data_source import tse_data_source
from gravity_tech.services.analysis_service import TechnicalAnalysisService


class ToolRecommendationService:
    """Concrete implementation consumed by the API layer."""

    def __init__(self, catalog_path: Path | None = None) -> None:
        self.catalog_path = catalog_path or Path(__file__).resolve().parents[2] / "configs" / "tools" / "catalog.json"
        self.catalog = self._load_catalog()
        self._tool_lookup = {tool["name"].lower(): tool for tool in self.catalog}
        self._cache: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ Catalog

    def list_tools(
        self,
        *,
        category: str | None = None,
        timeframe: str | None = None,
        min_accuracy: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return catalog entries filtered by the requested options."""
        tools = self.catalog

        if category:
            tools = [t for t in tools if t["category"] == category]
        if timeframe:
            tools = [t for t in tools if timeframe in t.get("timeframes", [])]
        if min_accuracy is not None:
            tools = [t for t in tools if t.get("historical_accuracy", 0.0) >= min_accuracy]

        return tools[:limit]

    def category_summary(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for tool in self.catalog:
            counts[tool["category"]] += 1
        return dict(counts)

    def get_tool(self, name: str) -> dict[str, Any] | None:
        return self._tool_lookup.get(name.lower())

    # ------------------------------------------------------------ Recommendations

    async def build_recommendations(
        self,
        *,
        symbol: str,
        timeframe: str,
        analysis_goal: str,
        trading_style: str,
        limit_candles: int,
        top_n: int,
    ) -> dict[str, Any]:
        cache_key = f"{symbol}:{timeframe}:{analysis_goal}:{trading_style}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        candles = await self._fetch_candles(symbol, timeframe, limit_candles)
        market_context = self._analyze_market_context(candles)
        ranked = self._score_tools(market_context, analysis_goal, trading_style, top_n)
        response = {
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis_goal": analysis_goal,
            "trading_style": trading_style,
            "timestamp": datetime.utcnow().isoformat(),
            "market_context": market_context,
            "recommendations": ranked,
            "dynamic_strategy": self._build_dynamic_strategy(ranked, market_context),
            "ml_metadata": {
                "model_type": "rule-based",
                "regime": market_context["regime"],
                "volatility": market_context["volatility"],
                "trend_strength": market_context["trend_strength"],
                "total_tools_analyzed": len(self.catalog),
            },
        }
        self._cache[cache_key] = response
        return response

    # --------------------------------------------------------------- Custom eval

    async def analyze_custom_tools(
        self,
        *,
        symbol: str,
        timeframe: str,
        selected_tools: list[str],
        include_ml_scoring: bool,
        include_patterns: bool,
        limit_candles: int,
    ) -> dict[str, Any]:
        candles = await self._fetch_candles(symbol, timeframe, limit_candles)
        tool_results = self._run_tool_evaluations(selected_tools, candles)

        ml_scoring: dict[str, Any] | None = None
        patterns: list[dict[str, Any]] | None = None
        summary: dict[str, Any] = {}

        if include_ml_scoring or include_patterns:
            analysis = await TechnicalAnalysisService.analyze(
                AnalysisRequest(symbol=symbol, timeframe=timeframe, candles=candles)
            )
            if include_ml_scoring:
                ml_scoring = {
                    "overall_signal": analysis.overall_signal.value if analysis.overall_signal else "NEUTRAL",
                    "trend_signal": analysis.overall_trend_signal.value if analysis.overall_trend_signal else "NEUTRAL",
                    "momentum_signal": analysis.overall_momentum_signal.value
                    if analysis.overall_momentum_signal
                    else "NEUTRAL",
                    "confidence": analysis.overall_confidence,
                }
            if include_patterns and analysis.candlestick_patterns:
                patterns = [
                    {
                        "type": pattern.pattern_type.value if hasattr(pattern.pattern_type, "value") else str(pattern.pattern_type),
                        "signal": pattern.signal.value if hasattr(pattern.signal, "value") else str(pattern.signal),
                        "confidence": pattern.confidence,
                    }
                    for pattern in analysis.candlestick_patterns[:5]
                ]

            summary = {
                "overall_signal": ml_scoring["overall_signal"] if ml_scoring else "UNKNOWN",
                "tools_analyzed": len(selected_tools),
                "bullish_tools": sum(1 for r in tool_results.values() if r["signal"].startswith("BULL")),
                "bearish_tools": sum(1 for r in tool_results.values() if r["signal"].startswith("BEAR")),
                "neutral_tools": sum(1 for r in tool_results.values() if r["signal"] == "NEUTRAL"),
            }
        else:
            summary = {
                "overall_signal": "UNKNOWN",
                "tools_analyzed": len(selected_tools),
                "bullish_tools": sum(1 for r in tool_results.values() if r["signal"].startswith("BULL")),
                "bearish_tools": sum(1 for r in tool_results.values() if r["signal"].startswith("BEAR")),
                "neutral_tools": sum(1 for r in tool_results.values() if r["signal"] == "NEUTRAL"),
            }

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "selected_tools": selected_tools,
            "tool_results": tool_results,
            "ml_scoring": ml_scoring,
            "patterns_detected": patterns,
            "summary": summary,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ---------------------------------------------------------------- Utilities

    def list_categories(self) -> dict[str, Any]:
        categories = {}
        for tool in self.catalog:
            info = categories.setdefault(
                tool["category"],
                {"count": 0, "description": tool.get("category_description", ""), "examples": []},
            )
            info["count"] += 1
            if len(info["examples"]) < 4:
                info["examples"].append(tool["name"])

        return {
            "total_tools": len(self.catalog),
            "total_categories": len(categories),
            "categories": categories,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ---------------------------------------------------------------- Internals

    def _load_catalog(self) -> list[dict[str, Any]]:
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Tool catalog not found at {self.catalog_path}")
        with open(self.catalog_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    async def _fetch_candles(self, symbol: str, timeframe: str, limit: int) -> list[Candle]:
        candles: list[Candle] = []

        if tse_data_source:
            try:
                end_date = datetime.utcnow().date()
                start_date = end_date - timedelta(days=max(limit, 30) * 2)
                raw = tse_data_source.fetch_price_data(
                    ticker=symbol,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                )
                for row in raw[-limit:]:
                    candles.append(
                        Candle(
                            timestamp=row["timestamp"],
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                            volume=float(row["volume"]),
                        )
                    )
            except Exception:
                candles = []

        if not candles:
            candles = self._generate_synthetic_candles(limit)

        return candles

    def _generate_synthetic_candles(self, limit: int) -> list[Candle]:
        candles: list[Candle] = []
        current_price = 100.0
        timestamp = datetime.utcnow() - timedelta(minutes=limit)

        for _ in range(limit):
            drift = random.uniform(-1.5, 1.5)
            volatility = random.uniform(0.5, 2.0)
            open_price = current_price
            close_price = current_price + drift
            high_price = max(open_price, close_price) + volatility
            low_price = min(open_price, close_price) - volatility
            volume = random.uniform(1_000, 5_000)

            candles.append(
                Candle(
                    timestamp=timestamp,
                    open=float(open_price),
                    high=float(high_price),
                    low=float(low_price),
                    close=float(close_price),
                    volume=float(volume),
                )
            )

            current_price = close_price
            timestamp += timedelta(minutes=1)

        return candles

    def _analyze_market_context(self, candles: list[Candle]) -> dict[str, Any]:
        closes = np.array([c.close for c in candles], dtype=float)
        volumes = np.array([c.volume for c in candles], dtype=float)

        price_change = ((closes[-1] - closes[0]) / closes[0]) * 100 if closes[0] else 0.0
        volatility = float(np.std(closes) / np.mean(closes) * 100) if np.mean(closes) else 0.0
        trend_strength = float(abs(price_change))
        volume_ratio = float(np.mean(volumes[-10:]) / (np.mean(volumes[:-10]) or 1.0))

        if trend_strength > 6:
            regime = "trending_bullish" if price_change > 0 else "trending_bearish"
        elif volatility > 2:
            regime = "volatile"
        else:
            regime = "ranging"

        if volume_ratio > 1.2:
            volume_profile = "high"
        elif volume_ratio < 0.8:
            volume_profile = "low"
        else:
            volume_profile = "average"

        return {
            "regime": regime,
            "price_change_pct": round(price_change, 2),
            "volatility": round(volatility, 2),
            "trend_strength": round(trend_strength, 2),
            "volume_ratio": round(volume_ratio, 2),
            "volume_profile": volume_profile,
        }

    def _score_tools(
        self,
        market_context: dict[str, Any],
        analysis_goal: str,
        trading_style: str,
        top_n: int,
    ) -> dict[str, list[dict[str, Any]]]:
        ranked_tools: list[dict[str, Any]] = []
        regime = market_context["regime"]
        for tool in self.catalog:
            base_score = tool.get("historical_accuracy", 0.5) * 100
            if regime in tool.get("ideal_regimes", []):
                base_score += 8
            if analysis_goal in tool.get("best_for", []):
                base_score += 6
            if trading_style in tool.get("supported_styles", []):
                base_score += 4

            base_score += float(tool.get("ml_weight", 0.1)) * 100
            confidence = min(0.98, base_score / 130)

            ranked_tools.append(
                {
                    "name": tool["name"],
                    "category": tool["category"],
                    "ml_weight": tool.get("ml_weight", 0.1),
                    "confidence": round(confidence, 2),
                    "historical_accuracy": f"{tool.get('historical_accuracy', 0.0) * 100:.1f}%",
                    "reason": self._build_reason(tool, market_context, analysis_goal),
                    "priority_score": base_score,
                    "best_for": tool.get("best_for", []),
                }
            )

        ranked_tools.sort(key=lambda item: item["priority_score"], reverse=True)

        buckets = {"must_use": [], "recommended": [], "optional": [], "avoid": []}
        for tool in ranked_tools[: top_n * 2]:
            score = tool["priority_score"]
            if score >= 95:
                buckets["must_use"].append(tool)
            elif score >= 80:
                buckets["recommended"].append(tool)
            elif score >= 60:
                buckets["optional"].append(tool)
            else:
                buckets["avoid"].append(tool)

        return buckets

    def _build_reason(self, tool: dict[str, Any], context: dict[str, Any], analysis_goal: str) -> str:
        pieces = []
        if context["regime"] in tool.get("ideal_regimes", []):
            pieces.append("Aligned with current market regime")
        if analysis_goal in tool.get("best_for", []):
            pieces.append("Matches requested analysis goal")
        if not pieces:
            pieces.append("Provides diversification for the current toolkit")
        return " | ".join(pieces)

    def _build_dynamic_strategy(
        self, buckets: dict[str, list[dict[str, Any]]], market_context: dict[str, Any]
    ) -> dict[str, Any]:
        primaries = [t["name"] for t in buckets["must_use"][:3]]
        supporting = [t["name"] for t in buckets["recommended"][:3]]
        confidence_values = [t["confidence"] for t in buckets["must_use"][:3]]
        confidence = round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else 0.5

        return {
            "primary_tools": primaries,
            "supporting_tools": supporting,
            "confidence": confidence,
            "based_on": "market_context + catalog weights",
            "regime": market_context.get("regime", "mixed"),
            "expected_accuracy": f"{confidence * 100:.1f}%",
        }

    def _run_tool_evaluations(self, tool_names: Iterable[str], candles: list[Candle]) -> dict[str, Any]:
        results: dict[str, Any] = {}

        evaluators = {
            "SMA": lambda: TrendIndicators.sma(candles, 20),
            "EMA": lambda: TrendIndicators.ema(candles, 20),
            "MACD": lambda: TrendIndicators.macd(candles),
            "ADX": lambda: TrendIndicators.adx(candles, 14),
            "RSI": lambda: MomentumIndicators.rsi(candles, 14),
            "Stochastic": lambda: MomentumIndicators.stochastic(candles, 14, 3),
            "Bollinger Bands": lambda: VolatilityIndicators.bollinger_bands(candles, 20, 2.0),
            "ATR": lambda: VolatilityIndicators.atr(candles, 14),
        }

        for tool in tool_names:
            normalized = tool.replace("_", " ").strip()
            evaluator = (
                evaluators.get(tool)
                or evaluators.get(normalized.upper())
                or evaluators.get(normalized.title())
            )
            if not evaluator:
                results[tool] = {
                    "signal": "NEUTRAL",
                    "description": "Tool not supported in backend",
                }
                continue

            try:
                indicator = evaluator()
                results[tool] = {
                    "signal": indicator.signal.value if hasattr(indicator.signal, "value") else str(indicator.signal),
                    "value": indicator.value,
                    "confidence": indicator.confidence,
                    "description": indicator.description,
                }
            except Exception as exc:
                results[tool] = {
                    "signal": "ERROR",
                    "description": f"Failed to evaluate indicator: {exc}",
                }

        return results

