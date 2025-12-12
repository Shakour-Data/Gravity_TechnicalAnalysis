"""
Complete Analysis Pipeline - خط لوله تحلیل کامل

این ماژول تمام اجزای سیستم تحلیل تکنیکال را به هم متصل می‌کند:

Layer 1: Base Dimensions (5 بُعد پایه)
    → Trend Analysis
    → Momentum Analysis
    → Volatility Analysis
    → Cycle Analysis
    → Support/Resistance Analysis

Layer 2: Volume-Dimension Matrix
    → 5 تعامل (Volume × هر بُعد)
    → تعدیل امتیازها

Layer 3: 5-Dimensional Decision Matrix
    → ترکیب هوشمند
    → تصمیم نهایی

این فایل orchestrator اصلی سیستم است.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""
from __future__ import annotations

import math
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from gravity_tech.core.domain.entities import Candle
from gravity_tech.ml.five_dimensional_decision_matrix import (
    FiveDimensionalDecision,
    FiveDimensionalDecisionMatrix,
)
from gravity_tech.ml.multi_horizon_analysis import MultiHorizonAnalyzer, TrendScore
from gravity_tech.ml.multi_horizon_cycle_analysis import (
    CycleScore,
    MultiHorizonCycleAnalyzer,
)
from gravity_tech.ml.multi_horizon_feature_extraction import MultiHorizonFeatureExtractor
from gravity_tech.ml.multi_horizon_momentum_analysis import (
    MomentumScore,
    MultiHorizonMomentumAnalyzer,
)
from gravity_tech.ml.multi_horizon_momentum_features import MultiHorizonMomentumFeatureExtractor
from gravity_tech.ml.multi_horizon_support_resistance_analysis import (
    MultiHorizonSupportResistanceAnalyzer,
    SupportResistanceScore,
)
from gravity_tech.ml.multi_horizon_volatility_analysis import (
    MultiHorizonVolatilityAnalyzer,
    VolatilityScore,
)
from gravity_tech.ml.multi_horizon_volatility_features import (
    MultiHorizonVolatilityFeatureExtractor,
)
from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner
from gravity_tech.ml.volume_dimension_matrix import VolumeDimensionMatrix

sys.path.append(str(Path(__file__).parent))

TAnalyzer = TypeVar("TAnalyzer")

MIN_PIPELINE_CANDLES = 120  # حداقل کندل برای پوشش پنجره‌های 100/120 و حجم


class CompleteAnalysisPipeline:
    """
    خط لوله کامل تحلیل تکنیکال

    این کلاس تمام مراحل تحلیل را انجام می‌دهد:
    1. دریافت داده‌های کندل
    2. محاسبه 5 بُعد پایه
    3. اعمال ماتریس حجم (اختیاری)
    4. تصمیم‌گیری 5 بُعدی
    5. خروجی نهایی

    استفاده:
    --------
    >>> candles = load_candles("BTC/USDT", "1h", 100)
    >>> pipeline = CompleteAnalysisPipeline(candles)
    >>> result = pipeline.analyze()
    >>> print(result.decision.final_signal)
    DecisionSignal.STRONG_BUY
    """

    def __init__(
        self,
        candles: list[Candle],
        use_volume_matrix: bool = True,
        custom_weights: dict[str, float] | None = None,
        verbose: bool = True,
        *,
        trend_analyzer: MultiHorizonAnalyzer | None = None,
        trend_learner: MultiHorizonWeightLearner | None = None,
        momentum_analyzer: MultiHorizonMomentumAnalyzer | None = None,
        momentum_learner: MultiHorizonWeightLearner | None = None,
        volatility_analyzer: MultiHorizonVolatilityAnalyzer | None = None,
        volatility_learner: MultiHorizonWeightLearner | None = None,
        cycle_analyzer: MultiHorizonCycleAnalyzer | None = None,
        sr_analyzer: MultiHorizonSupportResistanceAnalyzer | None = None,
        feature_cache: _FeatureCache | None = None,
    ):
        """
        Args:
            candles: لیست کندل‌ها (حداقل 120 کندل الزامی برای پنجره‌های 100/120)
            use_volume_matrix: فعال‌سازی تعدیلات حجم
            custom_weights: وزن‌های سفارشی برای ابعاد
            verbose: نمایش پیام‌های وضعیت
        """
        self.candles = self._sanitize_candles(candles)
        self.use_volume_matrix = use_volume_matrix
        self.custom_weights = custom_weights
        self.verbose = verbose

        # Analyzer wiring (trend/momentum/volatility require trained learners)
        self._trend_analyzer: MultiHorizonAnalyzer | None = self._resolve_analyzer(
            "trend",
            analyzer=trend_analyzer,
            learner=trend_learner,
            factory=MultiHorizonAnalyzer,
        )
        self._momentum_analyzer: MultiHorizonMomentumAnalyzer | None = self._resolve_analyzer(
            "momentum",
            analyzer=momentum_analyzer,
            learner=momentum_learner,
            factory=MultiHorizonMomentumAnalyzer,
        )
        self._volatility_analyzer: MultiHorizonVolatilityAnalyzer | None = self._resolve_analyzer(
            "volatility",
            analyzer=volatility_analyzer,
            learner=volatility_learner,
            factory=MultiHorizonVolatilityAnalyzer,
        )
        self._cycle_analyzer: MultiHorizonCycleAnalyzer = cycle_analyzer or MultiHorizonCycleAnalyzer()
        self._sr_analyzer: MultiHorizonSupportResistanceAnalyzer = sr_analyzer or MultiHorizonSupportResistanceAnalyzer()

        # Cache expensive feature computations for trend/momentum/volatility
        self._feature_cache: _FeatureCache = feature_cache or _FeatureCache(self.candles)

        # نگهداری نتایج واسط
        self._trend_score: TrendScore | None = None
        self._momentum_score: MomentumScore | None = None
        self._volatility_score: VolatilityScore | None = None
        self._cycle_score: CycleScore | None = None
        self._sr_score: SupportResistanceScore | None = None
        self._volume_interactions: dict | None = None
        self._final_decision: FiveDimensionalDecision | None = None

        self._log("✅ Pipeline initialized")
        self._log(f"   Candles: {len(candles)}")
        self._log(f"   Volume Matrix: {'Enabled' if use_volume_matrix else 'Disabled'}")

    @staticmethod
    def _sanitize_candles(candles: list[Candle]) -> list[Candle]:
        """Validate minimum length and finite OHLCV values"""
        if len(candles) < MIN_PIPELINE_CANDLES:
            raise ValueError(f"Pipeline requires at least {MIN_PIPELINE_CANDLES} candles (got {len(candles)})")
        for c in candles:
            values = [c.open, c.high, c.low, c.close, c.volume]
            if not all(math.isfinite(float(v)) for v in values):
                raise ValueError("Candles contain non-finite OHLCV values")
            if c.high < c.low:
                raise ValueError("Candle high must be >= low")
            if c.volume < 0:
                raise ValueError("Candle volume must be non-negative")
        return candles

    def _log(self, message: str):
        """چاپ پیام اگر verbose فعال باشد"""
        if self.verbose:
            print(message)

    def analyze(self) -> PipelineResult:
        """
        اجرای تحلیل کامل

        Returns:
            PipelineResult: شامل تمام نتایج
        """
        self._log("\n" + "=" * 80)
        self._log("🚀 شروع تحلیل کامل (Complete Analysis Pipeline)")
        self._log("=" * 80)

        # Step 1: محاسبه ابعاد پایه
        self._log("\n📊 Step 1: محاسبه 5 بُعد پایه...")
        self._calculate_base_dimensions()

        # Step 2: ماتریس حجم (اختیاری)
        if self.use_volume_matrix:
            self._log("\n📊 Step 2: محاسبه Volume-Dimension Matrix...")
            self._calculate_volume_interactions()
        else:
            self._log("\n⏭️ Step 2: Volume Matrix غیرفعال است")

        # Step 3: تصمیم‌گیری 5 بُعدی
        self._log("\n📊 Step 3: تصمیم‌گیری 5 بُعدی (5D Decision)...")
        self._make_final_decision()

        # ساخت نتیجه
        trend_score, momentum_score, volatility_score, cycle_score, sr_score = self._require_scores()
        decision = self._require_final_decision()

        result = PipelineResult(
            timestamp=datetime.now(),
            candles_count=len(self.candles),
            trend_score=trend_score,
            momentum_score=momentum_score,
            volatility_score=volatility_score,
            cycle_score=cycle_score,
            sr_score=sr_score,
            volume_interactions=self._volume_interactions,
            decision=decision
        )

        self._log("\n" + "=" * 80)
        self._log("✅ تحلیل کامل شد!")
        self._log("=" * 80)

        return result

    def _calculate_base_dimensions(self):
        """?????? 5 ???? ????"""

        trend_analyzer = self._trend_analyzer
        momentum_analyzer = self._momentum_analyzer
        volatility_analyzer = self._volatility_analyzer

        if trend_analyzer is None or momentum_analyzer is None or volatility_analyzer is None:
            raise ValueError(
                "Trend, momentum, and volatility analyzers (or learners) must be provided."
            )

        # Trend
        self._log("   ? Trend Analysis...")
        trend_features = self._feature_cache.trend_features
        trend_result = trend_analyzer.analyze(trend_features)
        trend_score = trend_result.to_trend_score()
        self._trend_score = trend_score
        self._log(f"      Score: {trend_score.score:+.3f}, Signal: {trend_score.signal.value}")

        # Momentum
        self._log("   ? Momentum Analysis...")
        momentum_features = self._feature_cache.momentum_features
        momentum_result = momentum_analyzer.analyze(momentum_features)
        momentum_score = self._select_best_momentum_score(momentum_result)
        self._momentum_score = momentum_score
        self._log(f"      Score: {momentum_score.score:+.3f}, Signal: {momentum_score.signal.value}")

        # Volatility
        self._log("   ? Volatility Analysis...")
        volatility_features = self._feature_cache.volatility_features
        volatility_result = volatility_analyzer.analyze(volatility_features)
        volatility_score = self._select_best_volatility_score(volatility_result)
        self._volatility_score = volatility_score
        self._log(f"      Score: {volatility_score.score:+.3f}, Signal: {volatility_score.signal.value}")

        # Cycle
        self._log("   ? Cycle Analysis...")
        cycle_result = self._cycle_analyzer.analyze(self.candles)
        cycle_score = self._select_best_cycle_score(cycle_result)
        self._cycle_score = cycle_score
        self._log(f"      Score: {cycle_score.score:+.3f}, Phase: {cycle_score.phase}")

        # Support/Resistance
        self._log("   ? Support/Resistance Analysis...")
        sr_result = self._sr_analyzer.analyze(self.candles)
        sr_score = self._select_best_sr_score(sr_result)
        self._sr_score = sr_score
        sr_signal = getattr(sr_score.signal, "value", sr_score.signal)
        self._log(f"      Score: {sr_score.score:+.3f}, Pattern: {sr_signal}")

    def _calculate_volume_interactions(self):
        """محاسبه تعاملات حجم-ابعاد"""

        trend_score, momentum_score, volatility_score, cycle_score, sr_score = self._require_scores()
        volume_matrix = VolumeDimensionMatrix(self.candles)
        self._volume_interactions = volume_matrix.calculate_all_interactions(
            trend_score=trend_score,
            momentum_score=momentum_score,
            volatility_score=volatility_score,
            cycle_score=cycle_score,
            sr_score=sr_score
        )

        # نمایش خلاصه
        for name, interaction in self._volume_interactions.items():
            self._log(f"   → {name}: {interaction.interaction_type.value} "
                      f"({interaction.interaction_score:+.3f})")

    def _make_final_decision(self):
        """تصمیم‌گیری نهایی با 5D Matrix"""

        matrix = FiveDimensionalDecisionMatrix(
            candles=self.candles,
            dimension_weights=self.custom_weights,
            use_volume_matrix=self.use_volume_matrix
        )

        trend_score, momentum_score, volatility_score, cycle_score, sr_score = self._require_scores()
        decision = matrix.analyze(
            trend_score=trend_score,
            momentum_score=momentum_score,
            volatility_score=volatility_score,
            cycle_score=cycle_score,
            sr_score=sr_score
        )
        self._final_decision = decision

        # نمایش خلاصه
        self._log(f"   → Final Score: {decision.final_score:+.3f}")
        self._log(f"   → Final Signal: {decision.final_signal.value}")
        self._log(f"   → Confidence: {decision.final_confidence * 100:.1f}%")
        self._log(f"   → Risk Level: {decision.risk_level.value}")
        self._log(f"   → Agreement: {decision.agreement.overall_agreement * 100:.1f}%")

    def _resolve_analyzer(
        self,
        _name: str,
        analyzer: TAnalyzer | None,
        learner: MultiHorizonWeightLearner | None,
        factory: Callable[[MultiHorizonWeightLearner], TAnalyzer],
    ) -> TAnalyzer | None:
        """Ensure we have a callable analyzer for the requested dimension."""
        if analyzer is not None:
            return analyzer
        if learner is not None:
            return factory(learner)
        return None

    def _select_best_momentum_score(self, analysis) -> MomentumScore:
        candidates = [analysis.momentum_3d, analysis.momentum_7d, analysis.momentum_30d]
        return max(candidates, key=lambda score: score.confidence)

    def _select_best_volatility_score(self, analysis) -> VolatilityScore:
        candidates = [analysis.volatility_3d, analysis.volatility_7d, analysis.volatility_30d]
        return max(candidates, key=lambda score: score.confidence)

    def _select_best_cycle_score(self, analysis) -> CycleScore:
        candidates = [analysis.cycle_3d, analysis.cycle_7d, analysis.cycle_30d]
        return max(candidates, key=lambda score: score.confidence)

    def _select_best_sr_score(self, analysis) -> SupportResistanceScore:
        candidates = [analysis.score_3d, analysis.score_7d, analysis.score_30d]
        return max(candidates, key=lambda score: score.confidence)

    def _require_scores(self) -> tuple[TrendScore, MomentumScore, VolatilityScore, CycleScore, SupportResistanceScore]:
        """Ensure all base scores are available and return them."""
        if (
            self._trend_score is None
            or self._momentum_score is None
            or self._volatility_score is None
            or self._cycle_score is None
            or self._sr_score is None
        ):
            raise RuntimeError("Base dimension scores have not been calculated.")
        return (
            self._trend_score,
            self._momentum_score,
            self._volatility_score,
            self._cycle_score,
            self._sr_score,
        )

    def _require_final_decision(self) -> FiveDimensionalDecision:
        """Ensure the final decision exists before access."""
        if self._final_decision is None:
            raise RuntimeError("Final decision has not been computed.")
        return self._final_decision

    # Properties برای دسترسی آسان به نتایج

    @property
    def trend_score(self) -> TrendScore | None:
        """نتیجه تحلیل روند"""
        return self._trend_score

    @property
    def momentum_score(self) -> MomentumScore | None:
        """نتیجه تحلیل مومنتوم"""
        return self._momentum_score

    @property
    def volatility_score(self) -> VolatilityScore | None:
        """نتیجه تحلیل نوسان"""
        return self._volatility_score

    @property
    def cycle_score(self) -> CycleScore | None:
        """نتیجه تحلیل چرخه"""
        return self._cycle_score

    @property
    def sr_score(self) -> SupportResistanceScore | None:
        """نتیجه تحلیل حمایت/مقاومت"""
        return self._sr_score

    @property
    def volume_interactions(self) -> dict | None:
        """تعاملات حجم-ابعاد"""
        return self._volume_interactions

    @property
    def final_decision(self) -> FiveDimensionalDecision | None:
        """تصمیم نهایی 5 بُعدی"""
        return self._final_decision




class _FeatureCache:
    """Cache expensive feature extractions for pipeline analyzers."""

    def __init__(
        self,
        candles: list[Candle],
        trend_lookback: int = 100,
        momentum_lookback: int = 120,
        volatility_lookback: int = 100,
    ):
        self.candles = candles
        self._trend_lookback = trend_lookback
        self._momentum_lookback = momentum_lookback
        self._volatility_lookback = volatility_lookback

        self._trend_extractor = MultiHorizonFeatureExtractor(lookback_period=trend_lookback)
        self._momentum_extractor = MultiHorizonMomentumFeatureExtractor(
            lookback_period=momentum_lookback
        )
        self._volatility_extractor = MultiHorizonVolatilityFeatureExtractor(
            lookback_period=volatility_lookback
        )

        self._trend_features: dict[str, float] | None = None
        self._momentum_features: dict[str, float] | None = None
        self._volatility_features: dict[str, float] | None = None

    @property
    def trend_features(self) -> dict[str, float]:
        if self._trend_features is None:
            window = self._window(self._trend_lookback)
            self._trend_features = self._trend_extractor.extract_indicator_features(window)
        return self._trend_features

    @property
    def momentum_features(self) -> dict[str, float]:
        if self._momentum_features is None:
            window = self._window(self._momentum_lookback)
            self._momentum_features = self._trend_extractor.extract_indicator_features(window)
        return self._momentum_features

    @property
    def volatility_features(self) -> dict[str, float]:
        if self._volatility_features is None:
            window = self._window(self._volatility_lookback)
            self._volatility_features = self._trend_extractor.extract_indicator_features(window)
        return self._volatility_features

    def _window(self, length: int) -> list[Candle]:
        if len(self.candles) < length:
            raise ValueError(f"Need at least {length} candles for feature extraction")
        return self.candles[-length:]


class PipelineResult:
    """
    نتیجه کامل Pipeline

    شامل تمام نتایج واسط و نهایی
    """

    def __init__(
        self,
        timestamp: datetime,
        candles_count: int,
        trend_score: TrendScore,
        momentum_score: MomentumScore,
        volatility_score: VolatilityScore,
        cycle_score: CycleScore,
        sr_score: SupportResistanceScore,
        volume_interactions: dict | None,
        decision: FiveDimensionalDecision
    ):
        self.timestamp = timestamp
        self.candles_count = candles_count
        self.trend_score = trend_score
        self.momentum_score = momentum_score
        self.volatility_score = volatility_score
        self.cycle_score = cycle_score
        self.sr_score = sr_score
        self.volume_interactions = volume_interactions
        self.decision = decision

    def print_summary(self):
        """چاپ خلاصه نتایج"""
        print("\n" + "=" * 80)
        print("📊 خلاصه نتایج تحلیل")
        print("=" * 80)

        print(f"\n⏰ زمان: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📈 تعداد کندل: {self.candles_count}")

        # ابعاد پایه
        print("\n" + "─" * 80)
        print("📊 ابعاد پایه:")
        print("─" * 80)
        print(f"  Trend:      {self.trend_score.score:+.3f} ({self.trend_score.signal.value})")
        print(f"  Momentum:   {self.momentum_score.score:+.3f} ({self.momentum_score.signal.value})")
        print(f"  Volatility: {self.volatility_score.score:+.3f} ({self.volatility_score.signal.value})")
        print(f"  Cycle:      {self.cycle_score.score:+.3f} ({self.cycle_score.phase})")
        print(f"  S/R:        {self.sr_score.score:+.3f} ({self.sr_score.nearest_level_type})")

        # ماتریس حجم
        if self.volume_interactions:
            print("\n" + "─" * 80)
            print("📊 تعاملات حجم:")
            print("─" * 80)
            for name, interaction in self.volume_interactions.items():
                print(f"  {name}: {interaction.type.value} ({interaction.interaction_score:+.3f})")

        # تصمیم نهایی
        print("\n" + "═" * 80)
        print("🎯 تصمیم نهایی 5 بُعدی:")
        print("═" * 80)
        print(f"  سیگنال: {self.decision.final_signal.value}")
        print(f"  امتیاز: {self.decision.final_score:+.3f}")
        print(f"  اطمینان: {self.decision.final_confidence * 100:.1f}%")
        print(f"  قدرت سیگنال: {self.decision.signal_strength * 100:.1f}%")
        print(f"  توافق: {self.decision.agreement.overall_agreement * 100:.1f}%")
        print(f"  ریسک: {self.decision.risk_level.value}")

        if self.decision.risk_factors:
            print("\n  ⚠️ عوامل ریسک:")
            for factor in self.decision.risk_factors:
                print(f"     - {factor}")

        print("\n  💡 توصیه:")
        print(f"     {self.decision.recommendation}")

        print("\n" + "=" * 80)

    def to_dict(self) -> dict:
        """تبدیل به دیکشنری (برای JSON)"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'candles_count': self.candles_count,
            'base_dimensions': {
                'trend': {
                    'score': self.trend_score.score,
                    'signal': self.trend_score.signal.value,
                    'accuracy': self.trend_score.accuracy
                },
                'momentum': {
                    'score': self.momentum_score.score,
                    'signal': self.momentum_score.signal.value,
                    'accuracy': self.momentum_score.accuracy
                },
                'volatility': {
                    'score': self.volatility_score.score,
                    'signal': self.volatility_score.signal.value,
                    'accuracy': self.volatility_score.accuracy
                },
                'cycle': {
                    'score': self.cycle_score.score,
                    'phase': self.cycle_score.phase,
                    'phase_strength': self.cycle_score.phase_strength,
                    'accuracy': self.cycle_score.accuracy
                },
                'support_resistance': {
                    'score': self.sr_score.score,
                    'signal': getattr(self.sr_score.signal, "value", self.sr_score.signal),
                    'nearest_level_type': self.sr_score.nearest_level_type,
                    'nearest_level_distance': getattr(self.sr_score, "distance_to_key_level", None),
                    'accuracy': self.sr_score.accuracy
                }
            },
            'volume_interactions': {
                name: {
                    'type': interaction.type.value,
                    'score': interaction.interaction_score,
                    'confidence_multiplier': interaction.confidence_multiplier
                }
                for name, interaction in (self.volume_interactions or {}).items()
            },
            'final_decision': {
                'signal': self.decision.final_signal.value,
                'score': self.decision.final_score,
                'confidence': self.decision.final_confidence,
                'signal_strength': self.decision.signal_strength,
                'agreement': self.decision.agreement.overall_agreement,
                'risk_level': self.decision.risk_level.value,
                'risk_factors': self.decision.risk_factors,
                'recommendation': self.decision.recommendation,
                'entry_strategy': self.decision.entry_strategy,
                'exit_strategy': self.decision.exit_strategy,
                'stop_loss': self.decision.stop_loss_suggestion,
                'take_profit': self.decision.take_profit_suggestion,
                'market_condition': self.decision.market_condition,
                'key_insights': self.decision.key_insights
            }
        }


# Convenience Functions
# =====================

def quick_analyze(
    candles: list[Candle],
    verbose: bool = False
) -> PipelineResult:
    """
    تحلیل سریع - با تنظیمات پیش‌فرض

    Args:
        candles: لیست کندل‌ها
        verbose: نمایش جزئیات

    Returns:
        PipelineResult
    """
    pipeline = CompleteAnalysisPipeline(candles, verbose=verbose)
    return pipeline.analyze()


def analyze_with_custom_weights(
    candles: list[Candle],
    weights: dict[str, float],
    verbose: bool = False
) -> PipelineResult:
    """
    تحلیل با وزن‌های سفارشی

    Args:
        candles: لیست کندل‌ها
        weights: وزن‌های سفارشی
        verbose: نمایش جزئیات

    Returns:
        PipelineResult
    """
    pipeline = CompleteAnalysisPipeline(
        candles,
        custom_weights=weights,
        verbose=verbose
    )
    return pipeline.analyze()


def analyze_without_volume(
    candles: list[Candle],
    verbose: bool = False
) -> PipelineResult:
    """
    تحلیل بدون Volume Matrix

    Args:
        candles: لیست کندل‌ها
        verbose: نمایش جزئیات

    Returns:
        PipelineResult
    """
    pipeline = CompleteAnalysisPipeline(
        candles,
        use_volume_matrix=False,
        verbose=verbose
    )
    return pipeline.analyze()


# مثال استفاده
# ==============

if __name__ == "__main__":
    # این فقط برای نمایش ساختار است
    # در استفاده واقعی، باید کندل‌های واقعی لود شوند

    print("🚀 Complete Analysis Pipeline")
    print("=" * 80)
    print("\nبرای استفاده:")
    print("\n1. تحلیل ساده:")
    print("   >>> from ml.complete_analysis_pipeline import quick_analyze")
    print("   >>> candles = load_candles('BTC/USDT', '1h', 100)")
    print("   >>> result = quick_analyze(candles)")
    print("   >>> result.print_summary()")

    print("\n2. تحلیل با وزن‌های سفارشی:")
    print("   >>> weights = {'trend': 0.40, 'momentum': 0.30, ...}")
    print("   >>> result = analyze_with_custom_weights(candles, weights)")

    print("\n3. تحلیل بدون Volume Matrix:")
    print("   >>> result = analyze_without_volume(candles)")

    print("\n4. تحلیل پیشرفته:")
    print("   >>> pipeline = CompleteAnalysisPipeline(")
    print("   ...     candles=candles,")
    print("   ...     use_volume_matrix=True,")
    print("   ...     custom_weights=weights,")
    print("   ...     verbose=True")
    print("   ... )")
    print("   >>> result = pipeline.analyze()")

    print("\n5. دسترسی به نتایج واسط:")
    print("   >>> pipeline.trend_score")
    print("   >>> pipeline.momentum_score")
    print("   >>> pipeline.final_decision")

    print("\n6. خروجی JSON:")
    print("   >>> data = result.to_dict()")
    print("   >>> import json")
    print("   >>> print(json.dumps(data, indent=2, ensure_ascii=False))")

    print("\n" + "=" * 80)
