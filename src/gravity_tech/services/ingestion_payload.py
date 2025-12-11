"""
Helper utilities to turn analysis results into a JSON-friendly payload
that can be published as `ANALYSIS_COMPLETED` events for persistence.

The goal is to keep the event lightweight while still carrying the
aggregate scores that the HistoricalScoreManager expects.
"""

from __future__ import annotations

from datetime import UTC, datetime
from statistics import mean
from typing import Any

from gravity_tech.core.contracts.analysis import TechnicalAnalysisResult
from gravity_tech.core.domain.entities import Candle, IndicatorResult, PatternResult, SignalStrength


def _aggregate_category(indicators: list[IndicatorResult]) -> tuple[float, float, str]:
    """Compute normalized score [-1, 1], average confidence, and signal name."""
    if not indicators:
        return 0.0, 0.0, "NEUTRAL"

    weighted = 0.0
    total_conf = 0.0
    for ind in indicators:
        weighted += ind.signal.get_score() * ind.confidence
        total_conf += ind.confidence

    normalized_score = (weighted / total_conf / 2.0) if total_conf else 0.0
    avg_conf = (total_conf / len(indicators)) if indicators else 0.0
    signal = SignalStrength.from_value(normalized_score).name
    return normalized_score, avg_conf, signal


def _normalize_indicators(indicators: list[IndicatorResult]) -> list[dict[str, Any]]:
    """Flatten IndicatorResult objects into JSON-serializable dicts."""
    normalized: list[dict[str, Any]] = []
    for ind in indicators:
        normalized.append(
            {
                "name": ind.indicator_name,
                "category": getattr(ind.category, "name", ind.category),
                "signal": getattr(ind.signal, "name", ind.signal),
                "confidence": ind.confidence,
                "value": ind.value,
                "params": ind.additional_values or {},
                "timestamp": getattr(ind, "timestamp", None),
            }
        )
    return normalized


def _normalize_patterns(patterns: list[PatternResult]) -> list[dict[str, Any]]:
    """Convert pattern results to primitive dictionaries."""
    normalized: list[dict[str, Any]] = []
    for p in patterns:
        normalized.append(
            {
                "pattern_name": p.pattern_name,
                "pattern_type": getattr(p.pattern_type, "name", p.pattern_type),
                "signal": getattr(p.signal, "name", p.signal),
                "confidence": p.confidence,
                "start_time": p.start_time,
                "end_time": p.end_time,
                "price_target": getattr(p, "price_target", None),
                "stop_loss": getattr(p, "stop_loss", None),
                "description": getattr(p, "description", None),
            }
        )
    return normalized


def build_ingestion_payload(
    result: TechnicalAnalysisResult,
    candles: list[Candle],
) -> dict[str, Any]:
    """
    Convert a TechnicalAnalysisResult into the compact payload expected
    by the DataIngestor/Database layer.
    Includes size optimization by limiting large arrays.
    """
    price_at_analysis = candles[-1].close if candles else 0.0

    # Limit candles to last 100 for size control
    limited_candles = candles[-100:] if len(candles) > 100 else candles

    trend_score, trend_conf, trend_signal = _aggregate_category(result.trend_indicators)
    momentum_score, momentum_conf, momentum_signal = _aggregate_category(result.momentum_indicators)
    cycle_score, cycle_conf, cycle_signal = _aggregate_category(result.cycle_indicators)
    volume_score, volume_conf, volume_signal = _aggregate_category(result.volume_indicators)
    volatility_score, vol_conf, vol_signal = _aggregate_category(result.volatility_indicators)
    support_res_score, support_conf, support_signal = _aggregate_category(result.support_resistance_indicators)

    combined_score = (
        (trend_score * 0.30)
        + (momentum_score * 0.25)
        + (cycle_score * 0.25)
        + (volume_score * 0.20)
    )

    combined_confidence = (
        result.overall_confidence
        if result.overall_confidence is not None
        else mean(
            [
                trend_conf,
                momentum_conf,
                cycle_conf,
                volume_conf,
                vol_conf,
                support_conf,
            ]
        )
    )

    # Limit indicator lists to top 50 for size
    limited_trend = result.trend_indicators[:50] if len(result.trend_indicators) > 50 else result.trend_indicators
    limited_momentum = result.momentum_indicators[:50] if len(result.momentum_indicators) > 50 else result.momentum_indicators
    limited_cycle = result.cycle_indicators[:50] if len(result.cycle_indicators) > 50 else result.cycle_indicators
    limited_volume = result.volume_indicators[:50] if len(result.volume_indicators) > 50 else result.volume_indicators
    limited_volatility = result.volatility_indicators[:50] if len(result.volatility_indicators) > 50 else result.volatility_indicators
    limited_support = result.support_resistance_indicators[:50] if len(result.support_resistance_indicators) > 50 else result.support_resistance_indicators

    return {
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "analysis_timestamp": getattr(result, "analysis_timestamp", datetime.now(UTC)),
        "price_at_analysis": price_at_analysis,
        "candles": [{"timestamp": c.timestamp, "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume} for c in limited_candles],
        "trend_score": trend_score,
        "trend_confidence": trend_conf,
        "momentum_score": momentum_score,
        "momentum_confidence": momentum_conf,
        "cycle_score": cycle_score,
        "cycle_confidence": cycle_conf,
        "volume_score": volume_score,
        "volume_confidence": volume_conf,
        "volatility_score": volatility_score,
        "volatility_confidence": vol_conf,
        "support_resistance_score": support_res_score,
        "support_resistance_confidence": support_conf,
        "combined_score": combined_score,
        "combined_confidence": combined_confidence,
        "trend_signal": trend_signal,
        "momentum_signal": momentum_signal,
        "cycle_signal": cycle_signal,
        "volume_signal": volume_signal,
        "volatility_signal": vol_signal,
        "support_resistance_signal": support_signal,
        "indicator_scores": _normalize_indicators(
            limited_trend
            + limited_momentum
            + limited_cycle
            + limited_volume
            + limited_volatility
            + limited_support
        ),
        "classical_patterns": _normalize_patterns(result.classical_patterns[:20] if len(result.classical_patterns) > 20 else result.classical_patterns),
        "candlestick_patterns": _normalize_patterns(result.candlestick_patterns[:20] if len(result.candlestick_patterns) > 20 else result.candlestick_patterns),
    }
