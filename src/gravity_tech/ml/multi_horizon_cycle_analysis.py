"""
Multi-Horizon Cycle Analysis System

سیستم تحلیل سیکل با سه امتیاز مستقل:
- 3-Day Cycle Score
- 7-Day Cycle Score
- 30-Day Cycle Score

با پیش‌بینی فاز و دوره سیکل
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime

from gravity_tech.models.schemas import SignalStrength, Candle
from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner, HorizonWeights
from gravity_tech.ml.multi_horizon_cycle_features import MultiHorizonCycleFeatureExtractor


@dataclass
class CycleScore:
    """امتیاز سیکل یک افق"""
    horizon: str
    score: float  # [-1, 1] (negative=نزول سیکلی, positive=صعود سیکلی)
    confidence: float  # [0, 1]
    signal: SignalStrength
    phase: float  # [0, 360] degrees
    cycle_period: float  # تعداد کندل‌های یک سیکل کامل
    
    def get_phase_name(self) -> str:
        """
        نام فاز سیکل
        
        Returns:
            - ACCUMULATION (0-90°): کف سیکل، فاز خرید
            - MARKUP (90-180°): صعود سیکلی
            - DISTRIBUTION (180-270°): سقف سیکل، فاز فروش
            - MARKDOWN (270-360°): نزول سیکلی
        """
        if 0 <= self.phase < 90:
            return "ACCUMULATION"
        elif 90 <= self.phase < 180:
            return "MARKUP"
        elif 180 <= self.phase < 270:
            return "DISTRIBUTION"
        else:
            return "MARKDOWN"
    
    def get_position_in_phase(self) -> str:
        """موقعیت دقیق در فاز"""
        phase_mod = self.phase % 90
        if phase_mod < 30:
            return "EARLY"
        elif phase_mod < 60:
            return "MID"
        else:
            return "LATE"
    
    def get_cycle_speed(self) -> str:
        """سرعت سیکل بر اساس دوره"""
        if self.cycle_period < 12:
            return "VERY_FAST"
        elif self.cycle_period < 18:
            return "FAST"
        elif self.cycle_period <= 28:
            return "NORMAL"
        elif self.cycle_period <= 35:
            return "SLOW"
        else:
            return "VERY_SLOW"


@dataclass
class MultiHorizonCycleAnalysis:
    """نتیجه تحلیل سیکل چند افقی"""
    timestamp: str
    
    # امتیازهای سیکل
    cycle_3d: CycleScore
    cycle_7d: CycleScore
    cycle_30d: CycleScore
    
    # امتیاز ترکیبی
    combined_cycle: float
    combined_confidence: float
    
    # فاز غالب
    dominant_phase: str  # ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN
    
    # توصیه‌ها
    recommendation_3d: str
    recommendation_7d: str
    recommendation_30d: str
    
    # تشخیص alignment
    cycle_alignment: str  # ALIGNED (همه هم‌جهت), MIXED, CONFLICTING
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'cycle_scores': {
                '3d': {
                    'score': self.cycle_3d.score,
                    'confidence': self.cycle_3d.confidence,
                    'signal': self.cycle_3d.signal.name,
                    'phase': self.cycle_3d.phase,
                    'phase_name': self.cycle_3d.get_phase_name(),
                    'position': self.cycle_3d.get_position_in_phase(),
                    'cycle_period': self.cycle_3d.cycle_period,
                    'cycle_speed': self.cycle_3d.get_cycle_speed()
                },
                '7d': {
                    'score': self.cycle_7d.score,
                    'confidence': self.cycle_7d.confidence,
                    'signal': self.cycle_7d.signal.name,
                    'phase': self.cycle_7d.phase,
                    'phase_name': self.cycle_7d.get_phase_name(),
                    'position': self.cycle_7d.get_position_in_phase(),
                    'cycle_period': self.cycle_7d.cycle_period,
                    'cycle_speed': self.cycle_7d.get_cycle_speed()
                },
                '30d': {
                    'score': self.cycle_30d.score,
                    'confidence': self.cycle_30d.confidence,
                    'signal': self.cycle_30d.signal.name,
                    'phase': self.cycle_30d.phase,
                    'phase_name': self.cycle_30d.get_phase_name(),
                    'position': self.cycle_30d.get_position_in_phase(),
                    'cycle_period': self.cycle_30d.cycle_period,
                    'cycle_speed': self.cycle_30d.get_cycle_speed()
                }
            },
            'combined': {
                'cycle_score': self.combined_cycle,
                'confidence': self.combined_confidence,
                'dominant_phase': self.dominant_phase,
                'alignment': self.cycle_alignment
            },
            'recommendations': {
                '3d': self.recommendation_3d,
                '7d': self.recommendation_7d,
                '30d': self.recommendation_30d
            }
        }


class MultiHorizonCycleAnalyzer:
    """
    تحلیل‌گر سیکل چند افقی
    
    محاسبه امتیاز سیکل برای 3 افق زمانی:
    - کوتاه‌مدت (3 روز)
    - میان‌مدت (7 روز)
    - بلندمدت (30 روز)
    """
    
    def __init__(
        self,
        lookback_period: int = 100,
        weights_path: Optional[str] = None
    ):
        """
        Initialize analyzer
        
        Args:
            lookback_period: تعداد کندل‌های گذشته
            weights_path: مسیر فایل وزن‌های آموزش دیده (اختیاری)
        """
        self.lookback_period = lookback_period
        self.feature_extractor = MultiHorizonCycleFeatureExtractor(
            lookback_period=lookback_period
        )
        
        # بارگذاری وزن‌ها
        if weights_path:
            self.weight_learner = MultiHorizonWeightLearner.load(weights_path)
        else:
            # وزن‌های پیش‌فرض
            self.weight_learner = self._create_default_weights()
    
    def _create_default_weights(self) -> MultiHorizonWeightLearner:
        """وزن‌های پیش‌فرض برای اندیکاتورهای سیکل"""
        learner = MultiHorizonWeightLearner()
        
        # وزن‌های پیش‌فرض برای هر اندیکاتور
        default_indicator_weights = {
            'dpo': 0.12,
            'ehlers_cycle': 0.16,
            'dominant_cycle': 0.15,
            'schaff_trend_cycle': 0.14,
            'phase_accumulation': 0.13,
            'hilbert_transform': 0.16,
            'market_cycle_model': 0.14
        }
        
        # وزن‌های یکسان برای همه افق‌ها
        for horizon in ['3d', '7d', '30d']:
            learner.weights[horizon] = HorizonWeights(
                horizon=horizon,
                indicator_weights=default_indicator_weights.copy(),
                feature_importance={}
            )
        
        return learner
    
    def analyze(self, candles: List[Candle]) -> MultiHorizonCycleAnalysis:
        """
        تحلیل سیکل برای همه افق‌ها
        
        Args:
            candles: لیست کندل‌ها
        
        Returns:
            نتیجه تحلیل سیکل چند افقی
        """
        if len(candles) < self.lookback_period:
            return self._get_neutral_analysis()
        
        # استخراج ویژگی‌ها
        features = self.feature_extractor.extract_cycle_features(
            candles[-self.lookback_period:]
        )
        
        # محاسبه امتیاز برای هر افق
        cycle_3d = self._calculate_horizon_score(features, '3d')
        cycle_7d = self._calculate_horizon_score(features, '7d')
        cycle_30d = self._calculate_horizon_score(features, '30d')
        
        # امتیاز ترکیبی (weighted average)
        combined_cycle = (
            cycle_3d.score * cycle_3d.confidence * 0.3 +
            cycle_7d.score * cycle_7d.confidence * 0.4 +
            cycle_30d.score * cycle_30d.confidence * 0.3
        ) / (
            cycle_3d.confidence * 0.3 +
            cycle_7d.confidence * 0.4 +
            cycle_30d.confidence * 0.3
        )
        
        combined_confidence = (
            cycle_3d.confidence * 0.3 +
            cycle_7d.confidence * 0.4 +
            cycle_30d.confidence * 0.3
        )
        
        # تشخیص فاز غالب
        dominant_phase = self._determine_dominant_phase([cycle_3d, cycle_7d, cycle_30d])
        
        # بررسی alignment
        alignment = self._check_alignment([cycle_3d, cycle_7d, cycle_30d])
        
        # توصیه‌ها
        rec_3d = self._generate_recommendation(cycle_3d)
        rec_7d = self._generate_recommendation(cycle_7d)
        rec_30d = self._generate_recommendation(cycle_30d)
        
        return MultiHorizonCycleAnalysis(
            timestamp=datetime.now().isoformat(),
            cycle_3d=cycle_3d,
            cycle_7d=cycle_7d,
            cycle_30d=cycle_30d,
            combined_cycle=combined_cycle,
            combined_confidence=combined_confidence,
            dominant_phase=dominant_phase,
            recommendation_3d=rec_3d,
            recommendation_7d=rec_7d,
            recommendation_30d=rec_30d,
            cycle_alignment=alignment
        )
    
    def _calculate_horizon_score(
        self,
        features: Dict[str, float],
        horizon: str
    ) -> CycleScore:
        """محاسبه امتیاز سیکل برای یک افق"""
        weights = self.weight_learner.weights.get(horizon)
        if not weights:
            weights = self.weight_learner.weights['7d']  # fallback
        
        # اندیکاتورهای سیکل
        indicators = [
            'dpo',
            'ehlers_cycle',
            'dominant_cycle',
            'schaff_trend_cycle',
            'phase_accumulation',
            'hilbert_transform',
            'market_cycle_model'
        ]
        
        # محاسبه weighted score
        total_score = 0.0
        total_confidence = 0.0
        
        for indicator in indicators:
            weight = weights.indicator_weights.get(indicator, 1.0 / len(indicators))
            signal = features.get(f"{indicator}_signal", 0.0)
            confidence = features.get(f"{indicator}_confidence", 0.5)
            
            total_score += signal * confidence * weight
            total_confidence += confidence * weight
        
        # Normalize
        if total_confidence > 0:
            score = total_score / total_confidence
        else:
            score = 0.0
        
        score = np.clip(score, -1, 1)
        
        # محاسبه confidence کلی
        confidences = [
            features.get(f"{ind}_confidence", 0.5)
            for ind in indicators
        ]
        avg_confidence = np.mean(confidences)
        
        # Signal strength
        if score > 0.6:
            signal = SignalStrength.VERY_BULLISH
        elif score > 0.2:
            signal = SignalStrength.BULLISH
        elif score < -0.6:
            signal = SignalStrength.VERY_BEARISH
        elif score < -0.2:
            signal = SignalStrength.BEARISH
        else:
            signal = SignalStrength.NEUTRAL
        
        # فاز و دوره سیکل
        phase = features.get('cycle_avg_phase', 0.0)
        cycle_period = features.get('cycle_avg_period', 20.0)
        
        return CycleScore(
            horizon=horizon,
            score=score,
            confidence=avg_confidence,
            signal=signal,
            phase=phase,
            cycle_period=cycle_period
        )
    
    def _determine_dominant_phase(self, scores: List[CycleScore]) -> str:
        """تشخیص فاز غالب بر اساس confidence"""
        phase_votes = {}
        for score in scores:
            phase_name = score.get_phase_name()
            if phase_name not in phase_votes:
                phase_votes[phase_name] = 0.0
            phase_votes[phase_name] += score.confidence
        
        return max(phase_votes.items(), key=lambda x: x[1])[0]
    
    def _check_alignment(self, scores: List[CycleScore]) -> str:
        """بررسی هم‌جهتی سیکل‌ها"""
        phases = [s.get_phase_name() for s in scores]
        signals = [s.score for s in scores]
        
        # بررسی فاز
        if len(set(phases)) == 1:
            phase_aligned = True
        elif len(set(phases)) == 2:
            phase_aligned = False
        else:
            phase_aligned = False
        
        # بررسی سیگنال
        bullish_count = sum(1 for s in signals if s > 0.2)
        bearish_count = sum(1 for s in signals if s < -0.2)
        
        if bullish_count == 3 or bearish_count == 3:
            signal_aligned = True
        else:
            signal_aligned = False
        
        if phase_aligned and signal_aligned:
            return "ALIGNED"
        elif phase_aligned or signal_aligned:
            return "MIXED"
        else:
            return "CONFLICTING"
    
    def _generate_recommendation(self, score: CycleScore) -> str:
        """تولید توصیه بر اساس امتیاز سیکل"""
        phase_name = score.get_phase_name()
        position = score.get_position_in_phase()
        
        if phase_name == "ACCUMULATION":
            if position == "EARLY":
                return "فرصت خرید عالی - کف سیکل"
            elif position == "MID":
                return "خرید توصیه می‌شود - فاز انباشت"
            else:
                return "آماده شروع صعود - آخرین فرصت خرید"
        
        elif phase_name == "MARKUP":
            if position == "EARLY":
                return "نگهداری - شروع صعود سیکلی"
            elif position == "MID":
                return "نگهداری قوی - قوی‌ترین فاز سیکل"
            else:
                return "نگهداری با احتیاط - نزدیک سقف"
        
        elif phase_name == "DISTRIBUTION":
            if position == "EARLY":
                return "کاهش پوزیشن - شروع توزیع"
            elif position == "MID":
                return "فروش توصیه می‌شود - فاز توزیع"
            else:
                return "فروش - آماده نزول"
        
        else:  # MARKDOWN
            if position == "EARLY":
                return "اجتناب از خرید - شروع نزول"
            elif position == "MID":
                return "صبر برای کف - فاز نزولی"
            else:
                return "آماده ورود - نزدیک کف سیکل"
    
    def _get_neutral_analysis(self) -> MultiHorizonCycleAnalysis:
        """تحلیل خنثی در صورت عدم وجود داده کافی"""
        neutral_score = CycleScore(
            horizon="",
            score=0.0,
            confidence=0.0,
            signal=SignalStrength.NEUTRAL,
            phase=0.0,
            cycle_period=20.0
        )
        
        return MultiHorizonCycleAnalysis(
            timestamp=datetime.now().isoformat(),
            cycle_3d=neutral_score,
            cycle_7d=neutral_score,
            cycle_30d=neutral_score,
            combined_cycle=0.0,
            combined_confidence=0.0,
            dominant_phase="ACCUMULATION",
            recommendation_3d="داده کافی نیست",
            recommendation_7d="داده کافی نیست",
            recommendation_30d="داده کافی نیست",
            cycle_alignment="NEUTRAL"
        )


# ═══════════════════════════════════════════════════════════════
# مثال استفاده
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from utils.sample_data import generate_sample_candles
    
    print("=" * 70)
    print("Multi-Horizon Cycle Analysis")
    print("=" * 70)
    
    # تولید داده نمونه
    candles = generate_sample_candles(
        count=150,
        base_price=50000,
        volatility=0.02,
        trend='sideways'  # سیکل بهتر در رنج دیده می‌شود
    )
    
    # ایجاد analyzer
    analyzer = MultiHorizonCycleAnalyzer(lookback_period=100)
    
    # تحلیل
    print("\n🔄 در حال تحلیل سیکل...")
    analysis = analyzer.analyze(candles)
    
    # نمایش نتایج
    print("\n" + "=" * 70)
    print("نتایج تحلیل سیکل")
    print("=" * 70)
    
    print(f"\n⏰ زمان: {analysis.timestamp}")
    
    # امتیازهای هر افق
    for horizon_name, cycle_score in [
        ('3 روزه', analysis.cycle_3d),
        ('7 روزه', analysis.cycle_7d),
        ('30 روزه', analysis.cycle_30d)
    ]:
        print(f"\n📊 سیکل {horizon_name}:")
        print(f"  امتیاز: {cycle_score.score:.3f}")
        print(f"  اعتماد: {cycle_score.confidence:.2%}")
        print(f"  سیگنال: {cycle_score.signal.name}")
        print(f"  فاز: {cycle_score.phase:.1f}° ({cycle_score.get_phase_name()})")
        print(f"  موقعیت در فاز: {cycle_score.get_position_in_phase()}")
        print(f"  دوره سیکل: {cycle_score.cycle_period:.1f} کندل")
        print(f"  سرعت سیکل: {cycle_score.get_cycle_speed()}")
    
    # امتیاز ترکیبی
    print(f"\n📈 امتیاز ترکیبی:")
    print(f"  سیکل: {analysis.combined_cycle:.3f}")
    print(f"  اعتماد: {analysis.combined_confidence:.2%}")
    print(f"  فاز غالب: {analysis.dominant_phase}")
    print(f"  هم‌جهتی: {analysis.cycle_alignment}")
    
    # توصیه‌ها
    print(f"\n💡 توصیه‌ها:")
    print(f"  3 روزه: {analysis.recommendation_3d}")
    print(f"  7 روزه: {analysis.recommendation_7d}")
    print(f"  30 روزه: {analysis.recommendation_30d}")
    
    # JSON output
    print("\n" + "=" * 70)
    print("خروجی JSON:")
    print("=" * 70)
    import json
    print(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 70)
    print("✅ تحلیل سیکل با موفقیت انجام شد!")
    print("=" * 70)
