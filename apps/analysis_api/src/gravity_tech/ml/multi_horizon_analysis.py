"""
Multi-Horizon Trend Analysis System

سیستم تحلیل روند با سه امتیاز مستقل:
- 3-Day Score (کوتاه‌مدت): برای Day Trading
- 7-Day Score (میان‌مدت): برای Swing Trading
- 30-Day Score (بلندمدت): برای Position Trading

با تشخیص الگو: STRONG_UPTREND, BUY_THE_DIP, TREND_REVERSAL, ...

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from gravity_tech.core.domain.entities.signal_strength import SignalStrength
from gravity_tech.ml.multi_horizon_weights import HorizonWeights, MultiHorizonWeightLearner


def score_to_signal(score: float) -> SignalStrength:
    """Map a normalized score [-1, 1] to a SignalStrength value."""
    if score > 0.7:
        return SignalStrength.VERY_BULLISH
    if score > 0.3:
        return SignalStrength.BULLISH
    if score > -0.3:
        return SignalStrength.NEUTRAL
    if score > -0.7:
        return SignalStrength.BEARISH
    return SignalStrength.VERY_BEARISH


class MarketPattern(Enum):
    """
    الگوهای بازار ترکیبی از سه افق
    """

    STRONG_UPTREND = "STRONG_UPTREND"  # همه مثبت
    STRONG_DOWNTREND = "STRONG_DOWNTREND"  # همه منفی
    BUY_THE_DIP = "BUY_THE_DIP"  # کوتاه منفی، میان و بلند مثبت
    SELL_THE_RALLY = "SELL_THE_RALLY"  # کوتاه مثبت، میان و بلند منفی
    TREND_REVERSAL = "TREND_REVERSAL"  # کوتاه و میان مثبت، بلند منفی (یا بالعکس)
    CONSOLIDATION = "CONSOLIDATION"  # همه نزدیک به صفر
    MIXED_SIGNALS = "MIXED_SIGNALS"  # سیگنال‌های مختلط
    UNCERTAIN = "UNCERTAIN"  # اعتماد پایین


@dataclass
class HorizonScore:
    """
    امتیاز یک افق زمانی
    """

    horizon: str  # "3d", "7d", "30d"
    score: float  # [-1, 1] - منفی: نزولی، مثبت: صعودی
    confidence: float  # [0, 1]
    signal: SignalStrength  # VERY_BULLISH, BULLISH, ...

    def get_strength(self) -> str:
        """قدرت سیگنال"""
        abs_score = abs(self.score)
        if abs_score > 0.7:
            return "STRONG"
        elif abs_score > 0.4:
            return "MODERATE"
        elif abs_score > 0.2:
            return "WEAK"
        else:
            return "NEUTRAL"


@dataclass
class TrendScore:
    """Aggregated trend dimension score used by downstream layers."""

    score: float
    confidence: float
    signal: SignalStrength
    pattern: MarketPattern
    recommendation: str

    @property
    def accuracy(self) -> float:
        """Backward compatible alias used by 5D matrix implementation."""
        return self.confidence

    @property
    def strength(self) -> str:
        """Qualitative interpretation of the score."""
        abs_score = abs(self.score)
        if abs_score > 0.7:
            return "STRONG"
        if abs_score > 0.4:
            return "MODERATE"
        if abs_score > 0.2:
            return "WEAK"
        return "NEUTRAL"


@dataclass
class MultiHorizonAnalysis:
    """
    نتیجه تحلیل چند افقی
    """

    timestamp: str

    # امتیازهای سه افق
    score_3d: HorizonScore
    score_7d: HorizonScore
    score_30d: HorizonScore

    # الگوی ترکیبی
    pattern: MarketPattern
    pattern_confidence: float

    # امتیاز ترکیبی هوشمند
    combined_score: float
    combined_confidence: float

    # توصیه‌ها
    recommendation_3d: str  # برای Day Trader
    recommendation_7d: str  # برای Swing Trader
    recommendation_30d: str  # برای Position Trader

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "scores": {
                "3d": {
                    "score": self.score_3d.score,
                    "confidence": self.score_3d.confidence,
                    "signal": self.score_3d.signal.name,
                    "strength": self.score_3d.get_strength(),
                },
                "7d": {
                    "score": self.score_7d.score,
                    "confidence": self.score_7d.confidence,
                    "signal": self.score_7d.signal.name,
                    "strength": self.score_7d.get_strength(),
                },
                "30d": {
                    "score": self.score_30d.score,
                    "confidence": self.score_30d.confidence,
                    "signal": self.score_30d.signal.name,
                    "strength": self.score_30d.get_strength(),
                },
            },
            "pattern": {"type": self.pattern.value, "confidence": self.pattern_confidence},
            "combined": {"score": self.combined_score, "confidence": self.combined_confidence},
            "recommendations": {
                "3d": self.recommendation_3d,
                "7d": self.recommendation_7d,
                "30d": self.recommendation_30d,
            },
        }

    def to_trend_score(self) -> TrendScore:
        """Convert the combined output into a TrendScore for downstream stages."""
        signal = score_to_signal(self.combined_score)
        return TrendScore(
            score=self.combined_score,
            confidence=self.combined_confidence,
            signal=signal,
            pattern=self.pattern,
            recommendation=self.recommendation_7d,
        )


class MultiHorizonTrendAnalyzer:
    """
    تحلیلگر روند چند افقی
    """

    def __init__(self, weight_learner: MultiHorizonWeightLearner):
        """
        Initialize analyzer

        Args:
            weight_learner: مدل آموزش دیده با وزن‌ها
        """
        self.weight_learner = weight_learner
        self.horizons = ["3d", "7d", "30d"]

    def analyze(self, features: dict[str, float]) -> MultiHorizonAnalysis:
        """
        تحلیل چند افقی بر اساس ویژگی‌های فعلی

        Args:
            features: ویژگی‌های استخراج شده (21 ویژگی)

        Returns:
            MultiHorizonAnalysis با سه امتیاز و الگو
        """
        # ایجاد DataFrame برای پیش‌بینی
        X = pd.DataFrame([features])

        # پیش‌بینی امتیازها
        predictions = self.weight_learner.predict_multi_horizon(X)

        # ایجاد HorizonScore برای هر افق
        horizon_scores = {}
        for horizon in self.horizons:
            pred_col = f"pred_{horizon}"
            raw_score = predictions[pred_col].iloc[0]

            # دریافت وزن‌ها و confidence
            horizon_weights = self.weight_learner.get_horizon_weights(horizon)
            confidence = horizon_weights.confidence

            # نرمال‌سازی score به [-1, 1]
            # raw_score معمولاً بازدهی پیش‌بینی شده است (مثلاً 0.05 = 5%)
            # برای نرمال‌سازی، آن را به [-1, 1] تبدیل می‌کنیم
            normalized_score = np.clip(raw_score * 10, -1, 1)  # 0.1 → 1.0

            # تعیین SignalStrength
            signal = self._score_to_signal(normalized_score)

            horizon_scores[horizon] = HorizonScore(
                horizon=horizon, score=normalized_score, confidence=confidence, signal=signal
            )

        # تشخیص الگوی ترکیبی
        pattern, pattern_confidence = self._detect_pattern(horizon_scores)

        # محاسبه امتیاز ترکیبی هوشمند
        combined_score, combined_confidence = self._smart_combination(horizon_scores)

        # ایجاد توصیه‌ها
        rec_3d = self._generate_recommendation(horizon_scores["3d"], pattern)
        rec_7d = self._generate_recommendation(horizon_scores["7d"], pattern)
        rec_30d = self._generate_recommendation(horizon_scores["30d"], pattern)

        return MultiHorizonAnalysis(
            timestamp=pd.Timestamp.now().isoformat(),
            score_3d=horizon_scores["3d"],
            score_7d=horizon_scores["7d"],
            score_30d=horizon_scores["30d"],
            pattern=pattern,
            pattern_confidence=pattern_confidence,
            combined_score=combined_score,
            combined_confidence=combined_confidence,
            recommendation_3d=rec_3d,
            recommendation_7d=rec_7d,
            recommendation_30d=rec_30d,
        )

    def _score_to_signal(self, score: float) -> SignalStrength:
        """تبدیل score به SignalStrength"""
        return score_to_signal(score)

    def _detect_pattern(
        self, horizon_scores: dict[str, HorizonScore]
    ) -> tuple[MarketPattern, float]:
        """
        تشخیص الگوی ترکیبی از سه افق

        Returns:
            (pattern, confidence)
        """
        s3 = horizon_scores["3d"].score
        s7 = horizon_scores["7d"].score
        s30 = horizon_scores["30d"].score

        c3 = horizon_scores["3d"].confidence
        c7 = horizon_scores["7d"].confidence
        c30 = horizon_scores["30d"].confidence

        # اعتماد کلی
        avg_confidence = (c3 + c7 + c30) / 3

        # اعتماد پایین → UNCERTAIN
        if avg_confidence < 0.3:
            return MarketPattern.UNCERTAIN, avg_confidence

        # همه مثبت قوی → STRONG_UPTREND
        if s3 > 0.5 and s7 > 0.5 and s30 > 0.5:
            return MarketPattern.STRONG_UPTREND, avg_confidence

        # همه منفی قوی → STRONG_DOWNTREND
        if s3 < -0.5 and s7 < -0.5 and s30 < -0.5:
            return MarketPattern.STRONG_DOWNTREND, avg_confidence

        # کوتاه منفی، میان و بلند مثبت → BUY_THE_DIP
        if s3 < -0.3 and s7 > 0.2 and s30 > 0.2:
            return MarketPattern.BUY_THE_DIP, avg_confidence

        # کوتاه مثبت، میان و بلند منفی → SELL_THE_RALLY
        if s3 > 0.3 and s7 < -0.2 and s30 < -0.2:
            return MarketPattern.SELL_THE_RALLY, avg_confidence

        # کوتاه و میان مثبت، بلند منفی → TREND_REVERSAL (صعودی کوتاه‌مدت)
        if s3 > 0.3 and s7 > 0.3 and s30 < -0.3:
            return MarketPattern.TREND_REVERSAL, avg_confidence

        # کوتاه و میان منفی، بلند مثبت → TREND_REVERSAL (نزولی کوتاه‌مدت)
        if s3 < -0.3 and s7 < -0.3 and s30 > 0.3:
            return MarketPattern.TREND_REVERSAL, avg_confidence

        # همه نزدیک به صفر → CONSOLIDATION
        if abs(s3) < 0.2 and abs(s7) < 0.2 and abs(s30) < 0.2:
            return MarketPattern.CONSOLIDATION, avg_confidence

        # بقیه → MIXED_SIGNALS
        return MarketPattern.MIXED_SIGNALS, avg_confidence

    def _smart_combination(self, horizon_scores: dict[str, HorizonScore]) -> tuple[float, float]:
        """
        ترکیب هوشمند امتیازها با وزن‌دهی اعتماد

        Returns:
            (combined_score, combined_confidence)
        """
        scores = []
        confidences = []

        for horizon in self.horizons:
            hs = horizon_scores[horizon]
            scores.append(hs.score)
            confidences.append(hs.confidence)

        # وزن‌دهی بر اساس اعتماد
        total_confidence = sum(confidences)

        if total_confidence > 0:
            weighted_score = (
                sum(s * c for s, c in zip(scores, confidences, strict=True)) / total_confidence
            )

            combined_confidence = total_confidence / len(confidences)
        else:
            weighted_score = 0.0
            combined_confidence = 0.0

        return weighted_score, combined_confidence

    def _generate_recommendation(self, horizon_score: HorizonScore, pattern: MarketPattern) -> str:
        """
        ایجاد توصیه برای یک افق
        """
        score = horizon_score.score
        confidence = horizon_score.confidence
        horizon = horizon_score.horizon

        # اعتماد پایین
        if confidence < 0.3:
            return f"⚠️ UNCERTAIN - Low confidence ({confidence:.0%})"

        # بر اساس الگو
        if pattern == MarketPattern.STRONG_UPTREND:
            return "🚀 STRONG BUY - All horizons bullish"

        elif pattern == MarketPattern.STRONG_DOWNTREND:
            return "⛔ STRONG SELL - All horizons bearish"

        elif pattern == MarketPattern.BUY_THE_DIP:
            if horizon == "3d":
                return "💎 BUY THE DIP - Short-term correction, long-term bullish"
            else:
                return "📈 HOLD/BUY - Long-term trend positive"

        elif pattern == MarketPattern.SELL_THE_RALLY:
            if horizon == "3d":
                return "💰 TAKE PROFIT - Short-term rally, long-term bearish"
            else:
                return "📉 SELL - Long-term trend negative"

        elif pattern == MarketPattern.TREND_REVERSAL:
            return "🔄 TREND REVERSAL - Short and long-term divergence"

        elif pattern == MarketPattern.CONSOLIDATION:
            return "⏸️ WAIT - Market consolidating"

        # پیش‌فرض: بر اساس score
        if score > 0.5:
            return f"📈 BUY - {horizon} bullish (confidence: {confidence:.0%})"
        elif score > 0.2:
            return f"↗️ WEAK BUY - {horizon} slightly bullish"
        elif score > -0.2:
            return f"➡️ HOLD - {horizon} neutral"
        elif score > -0.5:
            return f"↘️ WEAK SELL - {horizon} slightly bearish"
        else:
            return f"📉 SELL - {horizon} bearish (confidence: {confidence:.0%})"

    def analyze_batch(self, features_list: list[dict[str, float]]) -> list[MultiHorizonAnalysis]:
        """
        تحلیل دسته‌ای
        """
        return [self.analyze(features) for features in features_list]

    def print_analysis(self, analysis: MultiHorizonAnalysis):
        """
        نمایش زیبای نتیجه تحلیل
        """
        print("\n" + "=" * 70)
        print("🔮 MULTI-HORIZON TREND ANALYSIS")
        print("=" * 70)

        print(f"\n📅 Timestamp: {analysis.timestamp}")

        print(f"\n🎯 Pattern: {analysis.pattern.value}")
        print(f"   Confidence: {analysis.pattern_confidence:.0%}")

        print("\n" + "-" * 70)
        print("📊 HORIZON SCORES")
        print("-" * 70)

        for horizon in ["3d", "7d", "30d"]:
            if horizon == "3d":
                hs = analysis.score_3d
                rec = analysis.recommendation_3d
            elif horizon == "7d":
                hs = analysis.score_7d
                rec = analysis.recommendation_7d
            else:
                hs = analysis.score_30d
                rec = analysis.recommendation_30d

            print(f"\n{horizon.upper()}:")
            print(f"  Score:      {hs.score:+.3f} ({hs.get_strength()})")
            print(f"  Confidence: {hs.confidence:.0%}")
            print(f"  Signal:     {hs.signal.name}")
            print(f"  💡 {rec}")

        print("\n" + "-" * 70)
        print("🧠 COMBINED ANALYSIS")
        print("-" * 70)
        print(f"  Combined Score:      {analysis.combined_score:+.3f}")
        print(f"  Combined Confidence: {analysis.combined_confidence:.0%}")

        print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner

    # ساخت learner مصنوعی برای تست
    learner = MultiHorizonWeightLearner(horizons=["3d", "7d", "30d"])

    # شبیه‌سازی وزن‌های آموخته شده
    learner.feature_names = [f"feature_{i}" for i in range(21)]
    learner.horizon_weights = {
        "3d": HorizonWeights(
            horizon="3d",
            weights={f"feature_{i}": np.random.rand() for i in range(21)},
            metrics={"r2_test": 0.25, "mae_test": 0.04},
            confidence=0.6,
        ),
        "7d": HorizonWeights(
            horizon="7d",
            weights={f"feature_{i}": np.random.rand() for i in range(21)},
            metrics={"r2_test": 0.30, "mae_test": 0.06},
            confidence=0.7,
        ),
        "30d": HorizonWeights(
            horizon="30d",
            weights={f"feature_{i}": np.random.rand() for i in range(21)},
            metrics={"r2_test": 0.35, "mae_test": 0.10},
            confidence=0.75,
        ),
    }

    # ساخت مدل مصنوعی
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor

    learner.model = MultiOutputRegressor(LGBMRegressor())

    # تحلیلگر
    analyzer = MultiHorizonTrendAnalyzer(learner)

    # ویژگی‌های تست (شبیه‌سازی)
    features = {f"feature_{i}": np.random.randn() for i in range(21)}

    # تحلیل
    print("Testing multi-horizon analysis with synthetic data...")
    # analysis = analyzer.analyze(features)
    # analyzer.print_analysis(analysis)


# Alias برای سازگاری با کدهای قبلی
MultiHorizonAnalyzer = MultiHorizonTrendAnalyzer
