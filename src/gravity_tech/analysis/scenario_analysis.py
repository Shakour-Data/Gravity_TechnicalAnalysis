"""
Three-Scenario Analysis Module
تحلیل سه‌سناریویی (خوشبینانه - خنثی - بدبینانه)

This module provides comprehensive three-scenario analysis for market predictions.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog
from gravity_tech.clients.data_service_client import CandleData, DataServiceClient
from gravity_tech.indicators.momentum import MomentumIndicators
from gravity_tech.indicators.trend import TrendIndicators
from gravity_tech.indicators.volume import VolumeIndicators
from gravity_tech.models.schemas import Candle
from gravity_tech.patterns.classical import ClassicalPatterns
from datetime import timezone

logger = structlog.get_logger()


@dataclass
class ScenarioResult:
    """نتیجه یک سناریو"""
    scenario_type: str  # "optimistic", "neutral", "pessimistic"
    score: float  # 0-100
    probability: float  # 0-100%
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    key_signals: list[str]
    recommendation: str  # "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"
    confidence: str  # "LOW", "MEDIUM", "HIGH"
    timeframe_days: int  # افق زمانی


@dataclass
class ThreeScenarioAnalysis:
    """تحلیل کامل سه‌سناریویی"""
    symbol: str
    timestamp: datetime
    current_price: float

    optimistic: ScenarioResult
    neutral: ScenarioResult
    pessimistic: ScenarioResult

    expected_return: float  # بازدهی مورد انتظار (درصد)
    expected_risk: float  # ریسک مورد انتظار (درصد)
    sharpe_ratio: float

    recommended_scenario: str
    overall_confidence: str


class ScenarioAnalyzer:
    """
    تحلیلگر سه‌سناریویی

    این کلاس برای هر نماد سه سناریو محاسبه می‌کند:
    1. Optimistic (خوشبینانه)
    2. Neutral (خنثی)
    3. Pessimistic (بدبینانه)

    Data Integration:
    - دریافت داده از Data Service Microservice
    - پردازش داده‌های تعدیل‌شده (adjusted prices/volumes)
    - محاسبه سناریوها بر اساس داده‌های clean
    """

    def __init__(self, data_service_client: DataServiceClient | None = None):
        """
        Initialize scenario analyzer.

        Args:
            data_service_client: Client for Data Service (optional for testing)
        """
        self.trend_indicators = TrendIndicators()
        self.momentum_indicators = MomentumIndicators()
        self.volume_indicators = VolumeIndicators()
        self.data_client = data_service_client

    async def analyze_from_service(
        self,
        symbol: str,
        timeframe: str = "1d",
        lookback_days: int = 365
    ) -> ThreeScenarioAnalysis:
        """
        تحلیل سه‌سناریویی با دریافت داده از Data Service

        Args:
            symbol: نماد (مثل "AAPL", "فولاد")
            timeframe: بازه زمانی (1d, 1h, etc.)
            lookback_days: تعداد روزهای گذشته

        Returns:
            ThreeScenarioAnalysis

        Raises:
            ValueError: اگر Data Service client تنظیم نشده باشد
        """
        if not self.data_client:
            raise ValueError("Data service client not configured. Use analyze() with candles instead.")

        logger.info(
            "fetching_data_for_scenario_analysis",
            symbol=symbol,
            timeframe=timeframe,
            lookback_days=lookback_days
        )

        # دریافت داده از Data Service
        from datetime import timedelta
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days)

        candle_data = await self.data_client.get_candles(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )

        # تبدیل CandleData به Candle objects
        candles = [self._convert_to_candle(c) for c in candle_data]

        current_price = candles[-1].close

        logger.info(
            "data_received_for_scenario_analysis",
            symbol=symbol,
            candles_count=len(candles),
            current_price=current_price
        )

        # تحلیل سناریوها
        return self.analyze(symbol, candles, current_price)

    @staticmethod
    def _convert_to_candle(candle_data: CandleData) -> Candle:
        """Convert CandleData from Data Service to internal Candle model."""
        return Candle(
            timestamp=candle_data.timestamp,
            open=candle_data.adjusted_open,
            high=candle_data.adjusted_high,
            low=candle_data.adjusted_low,
            close=candle_data.adjusted_close,
            volume=candle_data.adjusted_volume
        )

    def analyze(
        self,
        symbol: str,
        candles: list[Candle],
        current_price: float = None
    ) -> ThreeScenarioAnalysis:
        """
        تحلیل سه‌سناریویی کامل

        Args:
            symbol: نماد
            candles: شمع‌های قیمتی (adjusted)
            current_price: قیمت فعلی (اختیاری)

        Returns:
            ThreeScenarioAnalysis
        """
        if not candles:
            raise ValueError("No candles provided for scenario analysis.")

        if current_price is None:
            current_price = candles[-1].close

        logger.info(
            "starting_scenario_analysis",
            symbol=symbol,
            candles_count=len(candles),
            current_price=current_price
        )

        # محاسبه ATR برای target و stop loss
        atr = self._calculate_atr(candles)
        atr_percentage = (atr / current_price) * 100

        # تحلیل تکنیکال پایه
        base_analysis = self._base_technical_analysis(candles)

        # سناریو خوشبینانه
        optimistic = self._analyze_optimistic_scenario(
            candles, current_price, atr_percentage, base_analysis
        )

        # سناریو خنثی
        neutral = self._analyze_neutral_scenario(
            candles, current_price, atr_percentage, base_analysis
        )

        # سناریو بدبینانه
        pessimistic = self._analyze_pessimistic_scenario(
            candles, current_price, atr_percentage, base_analysis
        )

        # محاسبه Expected Value
        expected_return, expected_risk, sharpe = self._calculate_expected_values(
            optimistic, neutral, pessimistic
        )

        # تعیین سناریوی پیشنهادی
        recommended = self._determine_recommended_scenario(
            optimistic, neutral, pessimistic
        )

        # محاسبه confidence کلی
        overall_confidence = self._calculate_overall_confidence(
            optimistic, neutral, pessimistic
        )

        return ThreeScenarioAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            current_price=current_price,
            optimistic=optimistic,
            neutral=neutral,
            pessimistic=pessimistic,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe,
            recommended_scenario=recommended,
            overall_confidence=overall_confidence
        )

    def _base_technical_analysis(self, candles: list[Candle]) -> dict[str, Any]:
        """تحلیل تکنیکال پایه"""
        closes = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])

        # Trend
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma_20
        sma_200 = np.mean(closes[-200:]) if len(closes) >= 200 else sma_50

        # Momentum
        rsi = self._calculate_rsi(closes)
        macd_line, signal_line = self._calculate_macd(closes)

        # Volume
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]

        # Patterns
        patterns = ClassicalPatterns.detect_all(candles[-100:])

        return {
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "rsi": rsi,
            "macd_line": macd_line,
            "signal_line": signal_line,
            "avg_volume": avg_volume,
            "current_volume": current_volume,
            "patterns": patterns,
            "trend": "up" if closes[-1] > sma_50 else "down",
        }

    def _analyze_optimistic_scenario(
        self,
        candles: list[Candle],
        current_price: float,
        atr_pct: float,
        base: dict
    ) -> ScenarioResult:
        """تحلیل سناریو خوشبینانه"""

        # محاسبه امتیاز با وزن‌های optimistic
        score = self._calculate_optimistic_score(base)

        # محاسبه احتمال
        probability = self._calculate_optimistic_probability(base)

        # Target و Stop Loss
        target_price = current_price * (1 + (atr_pct * 3) / 100)
        stop_loss = current_price * (1 - (atr_pct * 0.5) / 100)

        denominator = current_price - stop_loss
        risk_reward = (target_price - current_price) / denominator if denominator > 0 else 0.0

        # سیگنال‌های کلیدی
        key_signals = self._identify_bullish_signals(base)

        # توصیه
        if score >= 80:
            recommendation = "STRONG_BUY"
        elif score >= 65:
            recommendation = "BUY"
        else:
            recommendation = "HOLD"

        # Confidence
        confidence = "HIGH" if probability >= 60 else "MEDIUM" if probability >= 40 else "LOW"

        return ScenarioResult(
            scenario_type="optimistic",
            score=score,
            probability=probability,
            target_price=round(target_price, 2),
            stop_loss=round(stop_loss, 2),
            risk_reward_ratio=round(risk_reward, 2),
            key_signals=key_signals,
            recommendation=recommendation,
            confidence=confidence,
            timeframe_days=30
        )

    def _analyze_neutral_scenario(
        self,
        candles: list[Candle],
        current_price: float,
        atr_pct: float,
        base: dict
    ) -> ScenarioResult:
        """تحلیل سناریو خنثی"""

        # امتیاز با وزن‌های متعادل
        score = self._calculate_neutral_score(base)

        probability = self._calculate_neutral_probability(base)

        target_price = current_price * (1 + (atr_pct * 1.5) / 100)
        stop_loss = current_price * (1 - (atr_pct * 1.0) / 100)

        denominator = current_price - stop_loss
        risk_reward = (target_price - current_price) / denominator if denominator > 0 else 0.0

        key_signals = self._identify_neutral_signals(base)

        if score >= 60:
            recommendation = "HOLD"
        elif score >= 45:
            recommendation = "HOLD"
        else:
            recommendation = "AVOID"

        confidence = "MEDIUM"

        return ScenarioResult(
            scenario_type="neutral",
            score=score,
            probability=probability,
            target_price=round(target_price, 2),
            stop_loss=round(stop_loss, 2),
            risk_reward_ratio=round(risk_reward, 2),
            key_signals=key_signals,
            recommendation=recommendation,
            confidence=confidence,
            timeframe_days=60
        )

    def _analyze_pessimistic_scenario(
        self,
        candles: list[Candle],
        current_price: float,
        atr_pct: float,
        base: dict
    ) -> ScenarioResult:
        """تحلیل سناریو بدبینانه"""

        score = self._calculate_pessimistic_score(base)

        probability = self._calculate_pessimistic_probability(base)

        target_price = current_price * (1 + (atr_pct * 0.5) / 100)
        stop_loss = current_price * (1 - (atr_pct * 1.5) / 100)

        denominator = current_price - stop_loss
        risk_reward = (target_price - current_price) / denominator if denominator > 0 else 0.0

        key_signals = self._identify_bearish_signals(base)

        if score <= 30:
            recommendation = "STRONG_SELL"
        elif score <= 45:
            recommendation = "SELL"
        else:
            recommendation = "AVOID"

        confidence = "HIGH" if probability >= 60 else "MEDIUM" if probability >= 40 else "LOW"

        return ScenarioResult(
            scenario_type="pessimistic",
            score=score,
            probability=probability,
            target_price=round(target_price, 2),
            stop_loss=round(stop_loss, 2),
            risk_reward_ratio=round(risk_reward, 2),
            key_signals=key_signals,
            recommendation=recommendation,
            confidence=confidence,
            timeframe_days=90
        )

    def _calculate_optimistic_score(self, base: dict) -> float:
        """محاسبه امتیاز خوشبینانه با وزن‌های بیشتر برای سیگنال‌های مثبت"""
        score = 50.0  # پایه

        # Trend (وزن بیشتر)
        if base["trend"] == "up":
            score += 15

        if base["sma_20"] > base["sma_50"]:
            score += 10  # Golden Cross

        # Momentum
        if 50 <= base["rsi"] <= 70:
            score += 10  # RSI در محدوده مطلوب
        elif base["rsi"] > 70:
            score += 5  # Overbought اما momentum قوی

        if base["macd_line"] > base["signal_line"]:
            score += 10

        # Volume
        if base["current_volume"] > base["avg_volume"] * 1.5:
            score += 10  # حجم بالا

        return min(score, 100.0)

    def _calculate_neutral_score(self, base: dict) -> float:
        """امتیاز خنثی با وزن‌های متعادل"""
        score = 50.0

        if base["trend"] == "up":
            score += 5
        else:
            score -= 5

        if 40 <= base["rsi"] <= 60:
            score += 5

        return max(min(score, 70.0), 30.0)

    def _calculate_pessimistic_score(self, base: dict) -> float:
        """امتیاز بدبینانه با وزن بیشتر برای سیگنال‌های منفی"""
        score = 50.0

        if base["trend"] == "down":
            score -= 15

        if base["sma_20"] < base["sma_50"]:
            score -= 10  # Death Cross

        if base["rsi"] < 30:
            score -= 10  # Oversold

        if base["macd_line"] < base["signal_line"]:
            score -= 10

        if base["current_volume"] < base["avg_volume"] * 0.7:
            score -= 10  # حجم کم

        return max(score, 0.0)

    def _calculate_optimistic_probability(self, base: dict) -> float:
        """محاسبه احتمال سناریو خوشبینانه"""
        prob = 30.0

        bullish_signals = len(self._identify_bullish_signals(base))
        prob += bullish_signals * 5

        return min(prob, 75.0)

    def _calculate_neutral_probability(self, base: dict) -> float:
        """احتمال سناریو خنثی"""
        return 100.0 - self._calculate_optimistic_probability(base) - self._calculate_pessimistic_probability(base)

    def _calculate_pessimistic_probability(self, base: dict) -> float:
        """احتمال سناریو بدبینانه"""
        prob = 20.0

        bearish_signals = len(self._identify_bearish_signals(base))
        prob += bearish_signals * 5

        return min(prob, 50.0)

    def _identify_bullish_signals(self, base: dict) -> list[str]:
        """شناسایی سیگنال‌های صعودی"""
        signals = []

        if base["sma_20"] > base["sma_50"]:
            signals.append("golden_cross")

        if 50 <= base["rsi"] <= 70:
            signals.append("bullish_rsi")

        if base["macd_line"] > base["signal_line"]:
            signals.append("bullish_macd")

        if base["current_volume"] > base["avg_volume"] * 1.5:
            signals.append("high_volume")

        if base["trend"] == "up":
            signals.append("uptrend")

        return signals

    def _identify_neutral_signals(self, base: dict) -> list[str]:
        """سیگنال‌های خنثی"""
        signals = []

        if 40 <= base["rsi"] <= 60:
            signals.append("neutral_rsi")

        if abs(base["macd_line"] - base["signal_line"]) < 0.5:
            signals.append("macd_neutral")

        signals.append("sideways_trend")

        return signals

    def _identify_bearish_signals(self, base: dict) -> list[str]:
        """سیگنال‌های نزولی"""
        signals = []

        if base["sma_20"] < base["sma_50"]:
            signals.append("death_cross")

        if base["rsi"] < 30:
            signals.append("bearish_rsi")

        if base["macd_line"] < base["signal_line"]:
            signals.append("bearish_macd")

        if base["current_volume"] < base["avg_volume"] * 0.7:
            signals.append("low_volume")

        if base["trend"] == "down":
            signals.append("downtrend")

        return signals

    def _calculate_expected_values(
        self,
        optimistic: ScenarioResult,
        neutral: ScenarioResult,
        pessimistic: ScenarioResult
    ) -> tuple[float, float, float]:
        """محاسبه Expected Return, Risk, Sharpe"""

        # Normalize probabilities
        total_prob = optimistic.probability + neutral.probability + pessimistic.probability
        p_opt = optimistic.probability / total_prob
        p_neu = neutral.probability / total_prob
        p_pes = pessimistic.probability / total_prob

        # Expected Return
        expected_return = (
            p_opt * ((optimistic.target_price / optimistic.stop_loss) - 1) * 100 +
            p_neu * ((neutral.target_price / neutral.stop_loss) - 1) * 100 +
            p_pes * ((pessimistic.target_price / pessimistic.stop_loss) - 1) * 100
        )

        # Expected Risk
        expected_risk = (
            p_opt * 10 +
            p_neu * 20 +
            p_pes * 30
        )

        # Sharpe Ratio
        sharpe = expected_return / expected_risk if expected_risk > 0 else 0

        return round(expected_return, 2), round(expected_risk, 2), round(sharpe, 2)

    def _determine_recommended_scenario(
        self,
        optimistic: ScenarioResult,
        neutral: ScenarioResult,
        pessimistic: ScenarioResult
    ) -> str:
        """تعیین سناریوی پیشنهادی"""

        if optimistic.probability > 50 and optimistic.score > 70:
            return "optimistic"
        elif pessimistic.probability > 50 and pessimistic.score < 40:
            return "pessimistic"
        else:
            return "neutral"

    def _calculate_overall_confidence(
        self,
        optimistic: ScenarioResult,
        neutral: ScenarioResult,
        pessimistic: ScenarioResult
    ) -> str:
        """محاسبه confidence کلی"""

        max_prob = max(optimistic.probability, neutral.probability, pessimistic.probability)

        if max_prob >= 60:
            return "HIGH"
        elif max_prob >= 40:
            return "MEDIUM"
        else:
            return "LOW"

    # Helper methods
    def _calculate_atr(self, candles: list[Candle], period: int = 14) -> float:
        """محاسبه Average True Range"""
        if len(candles) < period + 1:
            return 0.0

        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i-1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        return np.mean(true_ranges[-period:])

    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """محاسبه RSI"""
        if len(closes) < period + 1:
            return 50.0

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(
        self,
        closes: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple[float, float]:
        """محاسبه MACD"""
        if len(closes) < slow:
            return 0.0, 0.0

        ema_fast = self._ema(closes, fast)
        ema_slow = self._ema(closes, slow)

        macd_line = ema_fast - ema_slow
        signal_line = self._ema(np.array([macd_line]), signal)

        return macd_line, signal_line

    def _ema(self, data: np.ndarray, period: int) -> float:
        """محاسبه Exponential Moving Average"""
        if len(data) < period:
            return np.mean(data)

        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])

        for price in data[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema
