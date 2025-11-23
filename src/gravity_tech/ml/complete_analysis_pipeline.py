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

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Models
from gravity_tech.models.schemas import (
    Candle,
    TrendScore,
    MomentumScore,
    VolatilityScore,
    CycleScore,
    SupportResistanceScore
)

# Base Dimensions - استفاده از تحلیل‌گرهای Multi-Horizon
from gravity_tech.ml.multi_horizon_analysis import MultiHorizonAnalyzer
from gravity_tech.ml.multi_horizon_momentum_analysis import MultiHorizonMomentumAnalyzer
from gravity_tech.ml.multi_horizon_volatility_analysis import MultiHorizonVolatilityAnalyzer
from gravity_tech.ml.multi_horizon_cycle_analysis import MultiHorizonCycleAnalyzer
from gravity_tech.ml.multi_horizon_support_resistance_analysis import MultiHorizonSupportResistanceAnalyzer

# Volume Matrix
from gravity_tech.ml.volume_dimension_matrix import VolumeDimensionMatrix

# 5D Decision Matrix
from gravity_tech.ml.five_dimensional_decision_matrix import (
    FiveDimensionalDecisionMatrix,
    FiveDimensionalDecision,
    DecisionSignal,
    RiskLevel
)


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
        candles: List[Candle],
        use_volume_matrix: bool = True,
        custom_weights: Optional[Dict[str, float]] = None,
        verbose: bool = True
    ):
        """
        Args:
            candles: لیست کندل‌ها (حداقل 100 کندل توصیه می‌شود)
            use_volume_matrix: فعال‌سازی تعدیلات حجم
            custom_weights: وزن‌های سفارشی برای ابعاد
            verbose: نمایش پیام‌های وضعیت
        """
        self.candles = candles
        self.use_volume_matrix = use_volume_matrix
        self.custom_weights = custom_weights
        self.verbose = verbose
        
        # نگهداری نتایج واسط
        self._trend_score: Optional[TrendScore] = None
        self._momentum_score: Optional[MomentumScore] = None
        self._volatility_score: Optional[VolatilityScore] = None
        self._cycle_score: Optional[CycleScore] = None
        self._sr_score: Optional[SupportResistanceScore] = None
        self._volume_interactions: Optional[Dict] = None
        self._final_decision: Optional[FiveDimensionalDecision] = None
        
        self._log("✅ Pipeline initialized")
        self._log(f"   Candles: {len(candles)}")
        self._log(f"   Volume Matrix: {'Enabled' if use_volume_matrix else 'Disabled'}")
    
    def _log(self, message: str):
        """چاپ پیام اگر verbose فعال باشد"""
        if self.verbose:
            print(message)
    
    def analyze(self) -> 'PipelineResult':
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
        result = PipelineResult(
            timestamp=datetime.now(),
            candles_count=len(self.candles),
            trend_score=self._trend_score,
            momentum_score=self._momentum_score,
            volatility_score=self._volatility_score,
            cycle_score=self._cycle_score,
            sr_score=self._sr_score,
            volume_interactions=self._volume_interactions,
            decision=self._final_decision
        )
        
        self._log("\n" + "=" * 80)
        self._log("✅ تحلیل کامل شد!")
        self._log("=" * 80)
        
        return result
    
    def _calculate_base_dimensions(self):
        """محاسبه 5 بُعد پایه"""
        
        # Trend
        self._log("   → Trend Analysis...")
        trend_analyzer = MultiHorizonAnalyzer()
        trend_result = trend_analyzer.analyze(self.candles)
        self._trend_score = trend_result.combined_score
        self._log(f"      Score: {self._trend_score.score:+.3f}, "
                  f"Signal: {self._trend_score.signal.value}")
        
        # Momentum
        self._log("   → Momentum Analysis...")
        momentum_analyzer = MultiHorizonMomentumAnalyzer()
        momentum_result = momentum_analyzer.analyze(self.candles)
        self._momentum_score = momentum_result.combined_score
        self._log(f"      Score: {self._momentum_score.score:+.3f}, "
                  f"Signal: {self._momentum_score.signal.value}")
        
        # Volatility
        self._log("   → Volatility Analysis...")
        volatility_analyzer = MultiHorizonVolatilityAnalyzer()
        volatility_result = volatility_analyzer.analyze(self.candles)
        self._volatility_score = volatility_result.combined_score
        self._log(f"      Score: {self._volatility_score.score:+.3f}, "
                  f"Signal: {self._volatility_score.signal.value}")
        
        # Cycle
        self._log("   → Cycle Analysis...")
        cycle_analyzer = MultiHorizonCycleAnalyzer()
        cycle_result = cycle_analyzer.analyze(self.candles)
        self._cycle_score = cycle_result.combined_score
        self._log(f"      Score: {self._cycle_score.score:+.3f}, "
                  f"Phase: {cycle_result.pattern.value if hasattr(cycle_result, 'pattern') else 'N/A'}")
        
        # Support/Resistance
        self._log("   → Support/Resistance Analysis...")
        sr_analyzer = MultiHorizonSupportResistanceAnalyzer()
        sr_result = sr_analyzer.analyze(self.candles)
        self._sr_score = sr_result.combined_score
        self._log(f"      Score: {self._sr_score.score:+.3f}, "
                  f"Pattern: {sr_result.pattern.value if hasattr(sr_result, 'pattern') else 'N/A'}")
    
    def _calculate_volume_interactions(self):
        """محاسبه تعاملات حجم-ابعاد"""
        
        volume_matrix = VolumeDimensionMatrix(self.candles)
        self._volume_interactions = volume_matrix.calculate_all_interactions(
            trend_score=self._trend_score,
            momentum_score=self._momentum_score,
            volatility_score=self._volatility_score,
            cycle_score=self._cycle_score,
            sr_score=self._sr_score
        )
        
        # نمایش خلاصه
        for name, interaction in self._volume_interactions.items():
            self._log(f"   → {name}: {interaction.type.value} "
                      f"({interaction.interaction_score:+.3f})")
    
    def _make_final_decision(self):
        """تصمیم‌گیری نهایی با 5D Matrix"""
        
        matrix = FiveDimensionalDecisionMatrix(
            candles=self.candles,
            dimension_weights=self.custom_weights,
            use_volume_matrix=self.use_volume_matrix
        )
        
        self._final_decision = matrix.analyze(
            trend_score=self._trend_score,
            momentum_score=self._momentum_score,
            volatility_score=self._volatility_score,
            cycle_score=self._cycle_score,
            sr_score=self._sr_score
        )
        
        # نمایش خلاصه
        self._log(f"   → Final Score: {self._final_decision.final_score:+.3f}")
        self._log(f"   → Final Signal: {self._final_decision.final_signal.value}")
        self._log(f"   → Confidence: {self._final_decision.final_confidence * 100:.1f}%")
        self._log(f"   → Risk Level: {self._final_decision.risk_level.value}")
        self._log(f"   → Agreement: {self._final_decision.agreement.overall_agreement * 100:.1f}%")
    
    # Properties برای دسترسی آسان به نتایج
    
    @property
    def trend_score(self) -> Optional[TrendScore]:
        """نتیجه تحلیل روند"""
        return self._trend_score
    
    @property
    def momentum_score(self) -> Optional[MomentumScore]:
        """نتیجه تحلیل مومنتوم"""
        return self._momentum_score
    
    @property
    def volatility_score(self) -> Optional[VolatilityScore]:
        """نتیجه تحلیل نوسان"""
        return self._volatility_score
    
    @property
    def cycle_score(self) -> Optional[CycleScore]:
        """نتیجه تحلیل چرخه"""
        return self._cycle_score
    
    @property
    def sr_score(self) -> Optional[SupportResistanceScore]:
        """نتیجه تحلیل حمایت/مقاومت"""
        return self._sr_score
    
    @property
    def volume_interactions(self) -> Optional[Dict]:
        """تعاملات حجم-ابعاد"""
        return self._volume_interactions
    
    @property
    def final_decision(self) -> Optional[FiveDimensionalDecision]:
        """تصمیم نهایی 5 بُعدی"""
        return self._final_decision


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
        volume_interactions: Optional[Dict],
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
            print(f"\n  ⚠️ عوامل ریسک:")
            for factor in self.decision.risk_factors:
                print(f"     - {factor}")
        
        print(f"\n  💡 توصیه:")
        print(f"     {self.decision.recommendation}")
        
        print("\n" + "=" * 80)
    
    def to_dict(self) -> Dict:
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
                    'signal': self.sr_score.signal.value,
                    'nearest_level_type': self.sr_score.nearest_level_type,
                    'nearest_level_distance': self.sr_score.nearest_level_distance,
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
    candles: List[Candle],
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
    candles: List[Candle],
    weights: Dict[str, float],
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
    candles: List[Candle],
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
