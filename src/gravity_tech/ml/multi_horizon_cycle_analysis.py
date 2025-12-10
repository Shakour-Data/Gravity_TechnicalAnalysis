"""
Multi-Horizon Cycle Analysis System

سیستم تحلیل سیکل با سه امتیاز مستقل:
- 3-Day Cycle Score
- 7-Day Cycle Score
- 30-Day Cycle Score

با پیش‌بینی فاز و دوره سیکل
"""

from dataclasses import dataclass
from datetime import datetime


import numpy as np
import pandas as pd
from gravity_tech.ml.multi_horizon_cycle_features import MultiHorizonCycleFeatureExtractor
from gravity_tech.ml.multi_horizon_weights import HorizonWeights, MultiHorizonWeightLearner
from gravity_tech.models.schemas import Candle, SignalStrength


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

    @property
    def accuracy(self) -> float:
        """سازگاری با واسط‌های قدیمی"""
        return self.confidence

    @property
    def phase_strength(self) -> str:
        """Qualitative description of the score magnitude for downstream consumers."""
        magnitude = abs(self.score)
        if magnitude > 0.7:
            return "STRONG"
        if magnitude > 0.4:
            return "MODERATE"
        if magnitude > 0.2:
            return "WEAK"
        return "NEUTRAL"


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

    def to_dict(self) -> dict:
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
        weight_learner: MultiHorizonWeightLearner | None = None,
        weights_path: str | None = None,
        model_path: str | None = None,
        horizons: list[str | None] = None,
    ):
        """
        Initialize analyzer

        Args:
            lookback_period: تعداد کندل‌های گذشته
            weight_learner: نمونه آماده‌ی MultiHorizonWeightLearner
            weights_path: مسیر فایل وزن‌های آموزش دیده (اختیاری)
            model_path: مسیر فایل pickle مدل (اختیاری)
            horizons: فهرست افق‌ها (پیش‌فرض: ['3d','7d','30d'])
        """
        self.lookback_period = lookback_period
        self.horizons = horizons or ['3d', '7d', '30d']
        self.feature_extractor = MultiHorizonCycleFeatureExtractor(
            lookback_period=lookback_period
        )

        if weight_learner is not None:
            self.weight_learner = weight_learner
        elif weights_path:
            self.weight_learner = MultiHorizonWeightLearner.load(weights_path, model_path)
        else:
            self.weight_learner = self._create_default_learner()

    def _create_default_learner(self) -> MultiHorizonWeightLearner:
        """وزن‌های پیش‌فرض برای اندیکاتورهای سیکل"""
        learner = MultiHorizonWeightLearner(horizons=self.horizons)
        feature_names = self.feature_extractor.get_feature_names()
        learner.feature_names = feature_names

        uniform_weights = {name: 1.0 / len(feature_names) for name in feature_names} if feature_names else {}
        metrics = {
            'r2_test': 0.0,
            'mae_test': 0.0,
            'r2_train': 0.0,
            'mae_train': 0.0,
        }

        learner.horizon_weights = {
            horizon: HorizonWeights(
                horizon=horizon,
                weights=uniform_weights.copy(),
                metrics=metrics.copy(),
                confidence=0.25,
            )
            for horizon in self.horizons
        }

        return learner

    def analyze(self, candles: list[Candle]) -> MultiHorizonCycleAnalysis:
        """
        تحلیل سیکل برای همه افق‌ها

        Args:
            candles: لیست کندل‌ها

        Returns:
            نتیجه تحلیل سیکل چند افقی
        """
        if len(candles) < self.lookback_period:
            return self._get_neutral_analysis()

        feature_window = candles[-self.lookback_period:]
        features = self.feature_extractor.extract_cycle_features(feature_window)
        features_df = pd.DataFrame([features])
        predictions = self.weight_learner.predict_multi_horizon(features_df)

        cycle_scores: dict[str, CycleScore] = {}
        for horizon in self.horizons:
            pred_col = f'pred_{horizon}'
            raw_score = float(predictions[pred_col].iloc[0]) if pred_col in predictions else 0.0
            cycle_scores[horizon] = self._build_cycle_score(horizon, raw_score, features)

        cycle_3d = cycle_scores.get('3d', self._neutral_score('3d'))
        cycle_7d = cycle_scores.get('7d', self._neutral_score('7d'))
        cycle_30d = cycle_scores.get('30d', self._neutral_score('30d'))

        weights = {'3d': 0.3, '7d': 0.4, '30d': 0.3}
        weighted_sum = sum(
            cycle_scores[horizon].score * cycle_scores[horizon].confidence * weights[horizon]
            for horizon in weights
        )
        confidence_sum = sum(
            cycle_scores[horizon].confidence * weights[horizon]
            for horizon in weights
        )

        combined_cycle = weighted_sum / confidence_sum if confidence_sum else 0.0
        combined_confidence = confidence_sum

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

    def _build_cycle_score(
        self,
        horizon: str,
        raw_score: float,
        features: dict[str, float],
    ) -> CycleScore:
        """تبدیل خروجی مدل به CycleScore."""
        normalized_score = float(np.clip(raw_score, -1.0, 1.0))
        horizon_weights = self.weight_learner.get_horizon_weights(horizon)
        confidence = horizon_weights.confidence if horizon_weights else 0.0
        signal = self._score_to_signal(normalized_score)
        phase = features.get('cycle_avg_phase', 0.0)
        cycle_period = features.get('cycle_avg_period', 20.0)

        return CycleScore(
            horizon=horizon,
            score=normalized_score,
            confidence=confidence,
            signal=signal,
            phase=phase,
            cycle_period=cycle_period
        )

    def _score_to_signal(self, score: float) -> SignalStrength:
        if score > 0.6:
            return SignalStrength.VERY_BULLISH
        if score > 0.2:
            return SignalStrength.BULLISH
        if score < -0.6:
            return SignalStrength.VERY_BEARISH
        if score < -0.2:
            return SignalStrength.BEARISH
        return SignalStrength.NEUTRAL

    def _neutral_score(self, horizon: str) -> CycleScore:
        return CycleScore(
            horizon=horizon,
            score=0.0,
            confidence=0.0,
            signal=SignalStrength.NEUTRAL,
            phase=0.0,
            cycle_period=20.0,
        )

    def _determine_dominant_phase(self, scores: list[CycleScore]) -> str:
        """تشخیص فاز غالب بر اساس confidence"""
        phase_votes = {}
        for score in scores:
            phase_name = score.get_phase_name()
            if phase_name not in phase_votes:
                phase_votes[phase_name] = 0.0
            phase_votes[phase_name] += score.confidence

        return max(phase_votes.items(), key=lambda x: x[1])[0]

    def _check_alignment(self, scores: list[CycleScore]) -> str:
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
        return MultiHorizonCycleAnalysis(
            timestamp=datetime.now().isoformat(),
            cycle_3d=self._neutral_score('3d'),
            cycle_7d=self._neutral_score('7d'),
            cycle_30d=self._neutral_score('30d'),
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
    print("\n📈 امتیاز ترکیبی:")
    print(f"  سیکل: {analysis.combined_cycle:.3f}")
    print(f"  اعتماد: {analysis.combined_confidence:.2%}")
    print(f"  فاز غالب: {analysis.dominant_phase}")
    print(f"  هم‌جهتی: {analysis.cycle_alignment}")

    # توصیه‌ها
    print("\n💡 توصیه‌ها:")
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
