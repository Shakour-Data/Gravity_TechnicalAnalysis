"""
Integrated Multi-Horizon Analysis with Volume-Dimension Matrix
===============================================================

این ماژول همه 5 dimension اصلی را با volume adjustments ترکیب می‌کند.

معماری:
1. محاسبه 5 dimension score (Trend, Momentum, Volatility, Cycle, S/R)
2. محاسبه 5 volume interaction از ماتریس
3. اعمال adjustments: adjusted_score = base_score + interaction_score
4. ترکیب نهایی با وزن‌های داینامیک

وزن‌های پایه:
- Trend: 30%
- Momentum: 25%
- Volatility: 15%
- Cycle: 20%
- S/R: 10%

وزن‌ها به صورت داینامیک بر اساس confidence تعدیل می‌شوند.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from gravity_tech.ml.volume_dimension_matrix import (
    InteractionType,
    VolumeDimensionInteraction,
    VolumeDimensionMatrix,
)
from gravity_tech.models.schemas import (
    Candle,
    CycleScore,
    MomentumScore,
    SupportResistanceScore,
    TrendScore,
    VolatilityScore,
)


class MarketSignal(Enum):
    """سیگنال نهایی بازار"""

    VERY_BULLISH = "VERY_BULLISH"  # بسیار صعودی
    BULLISH = "BULLISH"  # صعودی
    SLIGHTLY_BULLISH = "SLIGHTLY_BULLISH"  # کمی صعودی
    NEUTRAL = "NEUTRAL"  # خنثی
    SLIGHTLY_BEARISH = "SLIGHTLY_BEARISH"  # کمی نزولی
    BEARISH = "BEARISH"  # نزولی
    VERY_BEARISH = "VERY_BEARISH"  # بسیار نزولی


@dataclass
class DimensionAnalysis:
    """تحلیل یک dimension شامل base و adjusted scores"""

    name: str

    # امتیاز پایه (قبل از volume adjustment)
    base_score: float  # [-1, +1]
    base_confidence: float  # [0, 1]

    # volume interaction
    volume_interaction: VolumeDimensionInteraction

    # امتیاز تعدیل شده (بعد از volume adjustment)
    adjusted_score: float  # [-1, +1]
    adjusted_confidence: float  # [0, 1]

    # وزن در ترکیب نهایی
    weight: float  # [0, 1]

    # توضیحات
    signal: MarketSignal
    explanation: str


@dataclass
class IntegratedAnalysis:
    """نتیجه نهایی تحلیل یکپارچه"""

    # تحلیل هر dimension
    dimensions: dict[str, DimensionAnalysis]

    # امتیازهای کلی
    overall_base_score: float  # قبل از volume
    overall_adjusted_score: float  # بعد از volume
    overall_confidence: float

    # سیگنال نهایی
    final_signal: MarketSignal
    signal_strength: float  # [0, 1] - قدرت سیگنال

    # توافق بین dimensions
    dimensions_agreement: float  # [0, 1]
    conflicting_signals: list[str]  # dimensions مخالف

    # توصیه
    recommendation: str
    risk_level: str  # LOW, MEDIUM, HIGH

    # اطلاعات تکمیلی
    dominant_dimension: str  # قوی‌ترین dimension
    weakest_dimension: str  # ضعیف‌ترین dimension
    volume_impact: float  # [-1, +1] - تاثیر کلی حجم


class IntegratedMultiHorizonAnalyzer:
    """
    تحلیل‌گر یکپارچه با volume matrix

    این کلاس همه 5 dimension را دریافت کرده و با استفاده از
    volume-dimension matrix، امتیازها را تعدیل و ترکیب می‌کند.
    """

    # وزن‌های پایه برای هر dimension
    BASE_WEIGHTS = {
        "Trend": 0.30,
        "Momentum": 0.25,
        "Volatility": 0.15,
        "Cycle": 0.20,
        "SupportResistance": 0.10,
    }

    def __init__(self, candles: list[Candle]):
        """
        Args:
            candles: لیست کندل‌ها (حداقل 50 کندل)
        """
        self.candles = candles
        self.volume_matrix = VolumeDimensionMatrix(candles)

    def analyze(
        self,
        trend_score: TrendScore,
        momentum_score: MomentumScore,
        volatility_score: VolatilityScore,
        cycle_score: CycleScore,
        sr_score: SupportResistanceScore,
    ) -> IntegratedAnalysis:
        """
        تحلیل یکپارچه با volume adjustments

        Returns:
            IntegratedAnalysis با نتیجه نهایی
        """

        # ═══ گام 1: محاسبه volume interactions ═══
        interactions = self.volume_matrix.calculate_all_interactions(
            trend_score=trend_score,
            momentum_score=momentum_score,
            volatility_score=volatility_score,
            cycle_score=cycle_score,
            sr_score=sr_score,
        )

        # ═══ گام 2: تحلیل هر dimension ═══
        dimensions_analysis = {
            "Trend": self._analyze_dimension(
                name="Trend",
                base_score=trend_score.score,
                base_confidence=trend_score.confidence,
                interaction=interactions["Trend"],
            ),
            "Momentum": self._analyze_dimension(
                name="Momentum",
                base_score=momentum_score.score,
                base_confidence=momentum_score.confidence,
                interaction=interactions["Momentum"],
            ),
            "Volatility": self._analyze_dimension(
                name="Volatility",
                base_score=volatility_score.score,
                base_confidence=volatility_score.confidence,
                interaction=interactions["Volatility"],
            ),
            "Cycle": self._analyze_dimension(
                name="Cycle",
                base_score=cycle_score.score,
                base_confidence=cycle_score.confidence,
                interaction=interactions["Cycle"],
            ),
            "SupportResistance": self._analyze_dimension(
                name="SupportResistance",
                base_score=sr_score.score,
                base_confidence=sr_score.confidence,
                interaction=interactions["SupportResistance"],
            ),
        }

        # ═══ گام 3: محاسبه وزن‌های داینامیک ═══
        dynamic_weights = self._calculate_dynamic_weights(dimensions_analysis)

        # اعمال وزن‌های داینامیک
        for dim_name, weight in dynamic_weights.items():
            dimensions_analysis[dim_name].weight = weight

        # ═══ گام 4: محاسبه امتیاز کلی ═══
        overall_base_score = self._calculate_overall_score(dimensions_analysis, use_adjusted=False)

        overall_adjusted_score = self._calculate_overall_score(
            dimensions_analysis, use_adjusted=True
        )

        # ═══ گام 5: محاسبه توافق بین dimensions ═══
        agreement, conflicting = self._calculate_agreement(dimensions_analysis)

        # ═══ گام 6: تعیین سیگنال نهایی ═══
        final_signal, signal_strength = self._determine_final_signal(
            overall_adjusted_score, agreement
        )

        # ═══ گام 7: محاسبه confidence کلی ═══
        overall_confidence = self._calculate_overall_confidence(dimensions_analysis, agreement)

        # ═══ گام 8: تعیین dimension غالب و ضعیف ═══
        dominant = max(
            dimensions_analysis.items(),
            key=lambda x: abs(x[1].adjusted_score) * x[1].adjusted_confidence,
        )[0]

        weakest = min(
            dimensions_analysis.items(),
            key=lambda x: abs(x[1].adjusted_score) * x[1].adjusted_confidence,
        )[0]

        # ═══ گام 9: محاسبه تاثیر کلی حجم ═══
        volume_impact = overall_adjusted_score - overall_base_score

        # ═══ گام 10: تولید توصیه ═══
        recommendation = self._generate_recommendation(
            final_signal=final_signal,
            signal_strength=signal_strength,
            confidence=overall_confidence,
            agreement=agreement,
            conflicting=conflicting,
            volume_impact=volume_impact,
        )

        # ═══ گام 11: تعیین سطح ریسک ═══
        risk_level = self._determine_risk_level(
            agreement=agreement, confidence=overall_confidence, conflicting=conflicting
        )

        return IntegratedAnalysis(
            dimensions=dimensions_analysis,
            overall_base_score=overall_base_score,
            overall_adjusted_score=overall_adjusted_score,
            overall_confidence=overall_confidence,
            final_signal=final_signal,
            signal_strength=signal_strength,
            dimensions_agreement=agreement,
            conflicting_signals=conflicting,
            recommendation=recommendation,
            risk_level=risk_level,
            dominant_dimension=dominant,
            weakest_dimension=weakest,
            volume_impact=volume_impact,
        )

    def _analyze_dimension(
        self,
        name: str,
        base_score: float,
        base_confidence: float,
        interaction: VolumeDimensionInteraction,
    ) -> DimensionAnalysis:
        """
        تحلیل یک dimension با volume adjustment
        """

        # اعمال volume adjustment
        adjusted_score = base_score + interaction.interaction_score

        # محدود کردن به [-1, +1]
        adjusted_score = np.clip(adjusted_score, -1.0, 1.0)

        # تعدیل confidence بر اساس interaction
        if interaction.interaction_type == InteractionType.STRONG_CONFIRM:
            adjusted_confidence = min(0.95, base_confidence * 1.15)
        elif interaction.interaction_type == InteractionType.CONFIRM:
            adjusted_confidence = min(0.95, base_confidence * 1.08)
        elif interaction.interaction_type == InteractionType.DIVERGENCE:
            adjusted_confidence = base_confidence * 0.75
        elif interaction.interaction_type == InteractionType.WARN:
            adjusted_confidence = base_confidence * 0.85
        elif interaction.interaction_type == InteractionType.FAKE:
            adjusted_confidence = base_confidence * 0.60
        else:  # NEUTRAL
            adjusted_confidence = base_confidence

        # تعیین سیگنال
        signal = self._score_to_signal(adjusted_score)

        # توضیحات
        explanation = self._generate_dimension_explanation(
            name=name, base_score=base_score, adjusted_score=adjusted_score, interaction=interaction
        )

        return DimensionAnalysis(
            name=name,
            base_score=base_score,
            base_confidence=base_confidence,
            volume_interaction=interaction,
            adjusted_score=adjusted_score,
            adjusted_confidence=adjusted_confidence,
            weight=0.0,  # خواهد شد با dynamic weights
            signal=signal,
            explanation=explanation,
        )

    def _calculate_dynamic_weights(
        self, dimensions: dict[str, DimensionAnalysis]
    ) -> dict[str, float]:
        """
        محاسبه وزن‌های داینامیک بر اساس confidence هر dimension

        منطق:
        - dimensions با confidence بالاتر وزن بیشتری می‌گیرند
        - وزن‌ها جمعاً باید 1.0 شوند
        """

        # وزن اولیه = base_weight × confidence
        weighted_confidences = {}

        for name, analysis in dimensions.items():
            base_weight = self.BASE_WEIGHTS[name]
            weighted_conf = base_weight * analysis.adjusted_confidence
            weighted_confidences[name] = weighted_conf

        # نرمال‌سازی به 1.0
        total = sum(weighted_confidences.values())

        if total > 0:
            return {name: w / total for name, w in weighted_confidences.items()}
        else:
            # fallback: استفاده از وزن‌های پایه
            return self.BASE_WEIGHTS.copy()

    def _calculate_overall_score(
        self, dimensions: dict[str, DimensionAnalysis], use_adjusted: bool = True
    ) -> float:
        """
        محاسبه امتیاز کلی (weighted average)

        Args:
            use_adjusted: اگر True، از adjusted_score استفاده می‌کند
                         اگر False، از base_score (برای مقایسه)
        """

        weighted_sum = 0.0
        total_weight = 0.0

        for analysis in dimensions.values():
            score = analysis.adjusted_score if use_adjusted else analysis.base_score
            weight = analysis.weight

            weighted_sum += score * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0

    def _calculate_agreement(
        self, dimensions: dict[str, DimensionAnalysis]
    ) -> tuple[float, list[str]]:
        """
        محاسبه میزان توافق بین dimensions

        Returns:
            (agreement [0,1], conflicting_dimensions)
        """

        scores = [d.adjusted_score for d in dimensions.values()]

        # 1. محاسبه انحراف معیار
        std = np.std(scores)

        # 2. agreement = 1 - CV (coefficient of variation)
        mean_abs = np.mean(np.abs(scores))
        if mean_abs > 0.1:
            cv = std / mean_abs
            agreement = max(0.0, 1.0 - cv)
        else:
            agreement = 0.5  # نمی‌توان تعیین کرد

        # 3. شناسایی dimensions مخالف
        # اگر score یک dimension خیلی با میانگین فرق دارد
        mean_score = np.mean(scores)
        conflicting = []

        for name, analysis in dimensions.items():
            if mean_score > 0.3 and analysis.adjusted_score < -0.3:
                conflicting.append(name)
            elif mean_score < -0.3 and analysis.adjusted_score > 0.3:
                conflicting.append(name)

        return agreement, conflicting

    def _determine_final_signal(
        self, overall_score: float, agreement: float
    ) -> tuple[MarketSignal, float]:
        """
        تعیین سیگنال نهایی و قدرت آن

        Returns:
            (signal, strength [0,1])
        """

        signal = self._score_to_signal(overall_score)

        # قدرت سیگنال = |score| × agreement
        strength = abs(overall_score) * agreement
        strength = np.clip(strength, 0.0, 1.0)

        return signal, strength

    def _score_to_signal(self, score: float) -> MarketSignal:
        """تبدیل امتیاز به سیگنال"""
        if score > 0.7:
            return MarketSignal.VERY_BULLISH
        elif score > 0.4:
            return MarketSignal.BULLISH
        elif score > 0.15:
            return MarketSignal.SLIGHTLY_BULLISH
        elif score > -0.15:
            return MarketSignal.NEUTRAL
        elif score > -0.4:
            return MarketSignal.SLIGHTLY_BEARISH
        elif score > -0.7:
            return MarketSignal.BEARISH
        else:
            return MarketSignal.VERY_BEARISH

    def _calculate_overall_confidence(
        self, dimensions: dict[str, DimensionAnalysis], agreement: float
    ) -> float:
        """
        محاسبه confidence کلی

        Formula:
        - 60% از agreement
        - 40% از میانگین confidences
        """

        avg_confidence = np.mean([d.adjusted_confidence for d in dimensions.values()])

        overall = (agreement * 0.6) + (avg_confidence * 0.4)

        return np.clip(overall, 0.0, 1.0)

    def _generate_dimension_explanation(
        self,
        name: str,
        base_score: float,
        adjusted_score: float,
        interaction: VolumeDimensionInteraction,
    ) -> str:
        """تولید توضیحات برای یک dimension"""

        name_persian = {
            "Trend": "روند",
            "Momentum": "مومنتوم",
            "Volatility": "نوسان",
            "Cycle": "سیکل",
            "SupportResistance": "حمایت/مقاومت",
        }

        base_signal = self._score_to_signal(base_score)
        adjusted_signal = self._score_to_signal(adjusted_score)

        change = adjusted_score - base_score

        explanation = f"{name_persian[name]}: "

        if abs(change) < 0.05:
            explanation += f"حجم تاثیر کمی دارد - {adjusted_signal.value}"
        elif change > 0:
            explanation += f"حجم تقویت می‌کند: {base_signal.value} → {adjusted_signal.value}"
        else:
            explanation += f"حجم تضعیف می‌کند: {base_signal.value} → {adjusted_signal.value}"

        # اضافه کردن interaction explanation
        explanation += f" | {interaction.explanation}"

        return explanation

    def _generate_recommendation(
        self,
        final_signal: MarketSignal,
        signal_strength: float,
        confidence: float,
        agreement: float,
        conflicting: list[str],
        volume_impact: float,
    ) -> str:
        """تولید توصیه نهایی"""

        recommendations = []

        # 1. توصیه اصلی بر اساس سیگنال
        if final_signal == MarketSignal.VERY_BULLISH:
            if signal_strength > 0.8:
                recommendations.append("🟢 **خرید قوی** - همه شرایط مساعد است")
            else:
                recommendations.append("🟢 خرید - سیگنال صعودی قوی")

        elif final_signal == MarketSignal.BULLISH:
            if confidence > 0.75:
                recommendations.append("🟢 خرید - سیگنال صعودی با اطمینان بالا")
            else:
                recommendations.append("🟡 خرید محتاطانه - سیگنال صعودی با اطمینان متوسط")

        elif final_signal == MarketSignal.SLIGHTLY_BULLISH:
            recommendations.append("🟡 نگهداری یا خرید کم‌ریسک - سیگنال کمی صعودی")

        elif final_signal == MarketSignal.NEUTRAL:
            recommendations.append("⚪ انتظار - بازار خنثی، صبر برای سیگنال واضح‌تر")

        elif final_signal == MarketSignal.SLIGHTLY_BEARISH:
            recommendations.append("🟡 کاهش پوزیشن - سیگنال کمی نزولی")

        elif final_signal == MarketSignal.BEARISH:
            if confidence > 0.75:
                recommendations.append("🔴 فروش - سیگنال نزولی با اطمینان بالا")
            else:
                recommendations.append("🟡 فروش محتاطانه - سیگنال نزولی با اطمینان متوسط")

        elif final_signal == MarketSignal.VERY_BEARISH:
            if signal_strength > 0.8:
                recommendations.append("🔴 **فروش قوی** - همه شرایط نزولی")
            else:
                recommendations.append("🔴 فروش - سیگنال نزولی قوی")

        # 2. هشدار اگر توافق پایین است
        if agreement < 0.5:
            recommendations.append(f"⚠️ توافق پایین ({agreement:.0%}) - سیگنال‌های متناقض")

        # 3. هشدار اگر dimensions مخالف دارد
        if conflicting:
            dim_names = {
                "Trend": "روند",
                "Momentum": "مومنتوم",
                "Volatility": "نوسان",
                "Cycle": "سیکل",
                "SupportResistance": "S/R",
            }
            conflicting_persian = [dim_names[d] for d in conflicting]
            recommendations.append(f"⚠️ تناقض در: {', '.join(conflicting_persian)}")

        # 4. تاثیر حجم
        if abs(volume_impact) > 0.15:
            if volume_impact > 0:
                recommendations.append(f"📊 حجم تقویت‌کننده (+{volume_impact:.2f})")
            else:
                recommendations.append(f"📊 حجم تضعیف‌کننده ({volume_impact:.2f})")

        # 5. توصیه مدیریت ریسک
        if confidence < 0.6:
            recommendations.append("⚠️ استفاده از استاپ لاس محکم توصیه می‌شود")

        return " | ".join(recommendations)

    def _determine_risk_level(
        self, agreement: float, confidence: float, conflicting: list[str]
    ) -> str:
        """تعیین سطح ریسک"""

        # محاسبه risk score
        risk_score = 0

        # کم‌ترین ریسک: توافق بالا، اطمینان بالا، بدون تناقض
        if agreement > 0.75 and confidence > 0.75 and not conflicting:
            return "LOW"

        # ریسک متوسط
        if agreement > 0.5 and confidence > 0.6:
            return "MEDIUM"

        # ریسک بالا: توافق پایین یا اطمینان پایین یا تناقضات زیاد
        return "HIGH"


# ═══════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════


def example_integrated_analysis():
    """
    مثال استفاده از تحلیل یکپارچه
    """
    import random

    from models.schemas import Candle

    # شبیه‌سازی کندل‌ها
    candles = []
    base_price = 50000

    for i in range(100):
        open_price = base_price + random.uniform(-500, 500)
        close_price = open_price + random.uniform(-300, 300)
        high_price = max(open_price, close_price) + random.uniform(0, 200)
        low_price = min(open_price, close_price) - random.uniform(0, 200)
        volume = random.uniform(1000, 2000)

        candles.append(
            Candle(
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                timestamp=1700000000 + i * 3600,
            )
        )

        base_price = close_price

    # شبیه‌سازی scores
    trend_score = TrendScore(score=0.75, confidence=0.85, signal="BULLISH", strength=0.80)

    momentum_score = MomentumScore(score=0.60, confidence=0.75, signal="BULLISH", strength=0.65)

    volatility_score = VolatilityScore(
        score=0.40, confidence=0.70, signal="EXPANDING", strength=0.50
    )

    cycle_score = CycleScore(score=0.55, confidence=0.72, phase="MARKUP", strength=0.60)

    sr_score = SupportResistanceScore(
        score=0.65,
        confidence=0.78,
        signal="NEAR_SUPPORT",
        bounce_probability=0.72,
        breakout_probability=0.28,
    )

    # تحلیل یکپارچه
    analyzer = IntegratedMultiHorizonAnalyzer(candles)

    result = analyzer.analyze(
        trend_score=trend_score,
        momentum_score=momentum_score,
        volatility_score=volatility_score,
        cycle_score=cycle_score,
        sr_score=sr_score,
    )

    # نمایش نتایج
    print("=" * 80)
    print("📊 INTEGRATED MULTI-HORIZON ANALYSIS")
    print("=" * 80)

    print(f"\n🎯 Final Signal: {result.final_signal.value}")
    print(f"   Signal Strength: {result.signal_strength:.2%}")
    print(f"   Overall Confidence: {result.overall_confidence:.2%}")
    print(f"   Dimensions Agreement: {result.dimensions_agreement:.2%}")
    print(f"   Risk Level: {result.risk_level}")

    print("\n📈 Overall Scores:")
    print(f"   Before Volume: {result.overall_base_score:+.3f}")
    print(f"   After Volume:  {result.overall_adjusted_score:+.3f}")
    print(f"   Volume Impact: {result.volume_impact:+.3f}")

    print("\n🔍 Dimension Analysis:")
    for name, analysis in result.dimensions.items():
        print(f"\n   {name}:")
        print(f"      Base Score:     {analysis.base_score:+.3f}")
        print(f"      Adjusted Score: {analysis.adjusted_score:+.3f}")
        print(f"      Confidence:     {analysis.adjusted_confidence:.2%}")
        print(f"      Weight:         {analysis.weight:.2%}")
        print(f"      Signal:         {analysis.signal.value}")
        print(f"      Volume Effect:  {analysis.volume_interaction.interaction_type.value}")

    print("\n💡 Key Insights:")
    print(f"   Dominant Dimension: {result.dominant_dimension}")
    print(f"   Weakest Dimension:  {result.weakest_dimension}")
    if result.conflicting_signals:
        print(f"   Conflicting:        {', '.join(result.conflicting_signals)}")

    print("\n📋 Recommendation:")
    print(f"   {result.recommendation}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    example_integrated_analysis()
