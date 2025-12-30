"""
5-Dimensional Decision Matrix

این ماژول یک سیستم تصمیم‌گیری 5 بُعدی برای تحلیل جامع بازار ارائه می‌دهد.

5 Dimensions:
1. Trend (روند): جهت حرکت بازار
2. Momentum (مومنتوم): قدرت و سرعت حرکت
3. Volatility (نوسان): میزان نوسانات قیمت
4. Cycle (سیکل): فاز بازار (Accumulation, Markup, Distribution, Markdown)
5. Support/Resistance (حمایت/مقاومت): سطوح کلیدی قیمتی

این ماژول تمام 5 dimension را با هم ترکیب می‌کند و یک سیگنال نهایی تولید می‌کند.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from gravity_tech.core.domain.entities import Candle
from gravity_tech.ml.multi_horizon_analysis import TrendScore
from gravity_tech.ml.multi_horizon_cycle_analysis import CycleScore
from gravity_tech.ml.multi_horizon_momentum_analysis import MomentumScore
from gravity_tech.ml.multi_horizon_support_resistance_analysis import SupportResistanceScore
from gravity_tech.ml.multi_horizon_volatility_analysis import VolatilityScore
from gravity_tech.models.schemas import SignalStrength

MIN_CANDLES = 120  # برای پوشش پنجره‌های 100/120


class DecisionSignal(Enum):
    """سیگنال نهایی 5D"""

    VERY_STRONG_BUY = "VERY_STRONG_BUY"  # خرید بسیار قوی (همه ابعاد موافق)
    STRONG_BUY = "STRONG_BUY"  # خرید قوی (اکثریت قاطع موافق)
    BUY = "BUY"  # خرید (اکثریت ساده موافق)
    WEAK_BUY = "WEAK_BUY"  # خرید ضعیف (کمی مثبت)
    NEUTRAL = "NEUTRAL"  # خنثی (بدون سیگنال واضح)
    WEAK_SELL = "WEAK_SELL"  # فروش ضعیف (کمی منفی)
    SELL = "SELL"  # فروش (اکثریت ساده موافق)
    STRONG_SELL = "STRONG_SELL"  # فروش قوی (اکثریت قاطع موافق)
    VERY_STRONG_SELL = "VERY_STRONG_SELL"  # فروش بسیار قوی (همه ابعاد موافق)


class RiskLevel(Enum):
    """سطح ریسک تصمیم"""

    VERY_LOW = "VERY_LOW"  # ریسک بسیار کم (همه ابعاد هماهنگ)
    LOW = "LOW"  # ریسک کم (توافق بالا)
    MODERATE = "MODERATE"  # ریسک متوسط (توافق متوسط)
    HIGH = "HIGH"  # ریسک بالا (عدم توافق)
    VERY_HIGH = "VERY_HIGH"  # ریسک بسیار بالا (تناقض شدید)


@dataclass
class DimensionState:
    """وضعیت یک dimension"""

    name: str  # نام dimension
    score: float  # امتیاز [-1, +1]
    confidence: float  # اطمینان [0, 1]
    signal: SignalStrength  # سیگنال
    weight: float  # وزن در ترکیب نهایی
    volume_adjusted_score: float  # امتیاز بعد از تعدیل حجم
    volume_adjustment: float  # میزان تعدیل حجم
    description: str  # توضیحات


@dataclass
class DimensionAgreement:
    """توافق بین dimensions"""

    overall_agreement: float  # توافق کلی [0, 1]
    bullish_dimensions: list[str]  # dimensions صعودی
    bearish_dimensions: list[str]  # dimensions نزولی
    neutral_dimensions: list[str]  # dimensions خنثی
    strongest_dimension: str  # قوی‌ترین dimension
    weakest_dimension: str  # ضعیف‌ترین dimension
    conflicting: bool  # آیا تناقض وجود دارد؟


@dataclass
class FiveDimensionalDecision:
    """تصمیم نهایی 5 بُعدی"""

    timestamp: datetime

    # وضعیت هر dimension
    dimensions: dict[str, DimensionState]

    # امتیاز نهایی
    final_score: float  # [-1, +1]
    final_confidence: float  # [0, 1]
    final_signal: DecisionSignal
    signal_strength: float  # [0, 1] قدرت سیگنال

    # تحلیل توافق
    agreement: DimensionAgreement

    # ریسک
    risk_level: RiskLevel
    risk_factors: list[str]  # عوامل ریسک

    # توصیه‌ها
    recommendation: str  # توصیه اصلی
    entry_strategy: str  # استراتژی ورود
    exit_strategy: str  # استراتژی خروج
    stop_loss_suggestion: str  # پیشنهاد استاپ لاس
    take_profit_suggestion: str  # پیشنهاد حد سود

    # اطلاعات اضافی
    market_condition: str  # شرایط کلی بازار
    key_insights: list[str]  # نکات کلیدی


class FiveDimensionalDecisionMatrix:
    """
    ماتریس تصمیم‌گیری 5 بُعدی

    این کلاس تمام 5 dimension را تحلیل و ترکیب می‌کند و یک تصمیم نهایی ارائه می‌دهد.
    """

    @staticmethod
    def _scale_score_100(value: float) -> float:
        """Scale/clip any score to [-100, 100]. Keeps internal logic on [-1, 1] but returns 100-based."""
        return float(np.clip(value * 100.0, -100.0, 100.0))

    def _scale_dimensions_100(
        self, dimensions: dict[str, DimensionState]
    ) -> dict[str, DimensionState]:
        """Return dimensions with scores scaled to [-100, 100] for output."""
        for dim in dimensions.values():
            dim.score = self._scale_score_100(dim.score)
            dim.volume_adjusted_score = self._scale_score_100(dim.volume_adjusted_score)
        return dimensions

    # وزن‌های پایه هر dimension (قابل یادگیری با ML)
    DEFAULT_WEIGHTS = {
        "trend": 0.30,
        "momentum": 0.25,
        "volatility": 0.15,
        "cycle": 0.20,
        "support_resistance": 0.10,
    }

    def __init__(
        self,
        candles: list[Candle],
        dimension_weights: dict[str, float] | None = None,
        use_volume_matrix: bool = True,
    ):
        """
        Args:
            candles: لیست کندل‌ها (حداقل 120 کندل برای پوشش پنجره‌های 100/120)
            dimension_weights: وزن‌های سفارشی برای dimensions
            use_volume_matrix: استفاده از Volume-Dimension Matrix
        """
        self.candles = self._validate_candles(candles)
        self.weights = dimension_weights or self.DEFAULT_WEIGHTS
        self.use_volume_matrix = use_volume_matrix

        # نرمال‌سازی وزن‌ها
        total_weight = sum(self.weights.values())
        if total_weight <= 0:
            self.weights = self.DEFAULT_WEIGHTS
            total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

    @staticmethod
    def _validate_candles(candles: list[Candle]) -> list[Candle]:
        """Ensure minimum length and finite OHLCV"""
        if len(candles) < MIN_CANDLES:
            raise ValueError(
                f"FiveDimensionalDecisionMatrix requires at least {MIN_CANDLES} candles (got {len(candles)})"
            )
        for c in candles:
            vals = np.array([c.open, c.high, c.low, c.close, c.volume], dtype=float)
            if not np.all(np.isfinite(vals)):
                raise ValueError("Non-finite OHLCV detected in candles")
            if c.high < c.low:
                raise ValueError("High must be >= Low for all candles")
            if c.volume < 0:
                raise ValueError("Volume must be non-negative")
        return candles

    def analyze(
        self,
        trend_score: TrendScore,
        momentum_score: MomentumScore,
        volatility_score: VolatilityScore,
        cycle_score: CycleScore,
        sr_score: SupportResistanceScore,
    ) -> FiveDimensionalDecision:
        """
        تحلیل جامع و تولید تصمیم نهایی 5D

        Args:
            trend_score: نتیجه تحلیل روند
            momentum_score: نتیجه تحلیل مومنتوم
            volatility_score: نتیجه تحلیل نوسان
            cycle_score: نتیجه تحلیل سیکل
            sr_score: نتیجه تحلیل حمایت/مقاومت

        Returns:
            FiveDimensionalDecision: تصمیم نهایی با تمام جزئیات
        """

        # گام 1: جمع‌آوری state هر dimension
        dimensions = self._collect_dimension_states(
            trend_score, momentum_score, volatility_score, cycle_score, sr_score
        )

        # گام 2: اعمال تعدیلات حجم (اگر فعال باشد)
        if self.use_volume_matrix:
            dimensions = self._apply_volume_adjustments(
                dimensions, trend_score, momentum_score, volatility_score, cycle_score, sr_score
            )

        # گام 3: محاسبه وزن‌های داینامیک بر اساس confidence
        dimensions = self._calculate_dynamic_weights(dimensions)

        # گام 4: محاسبه امتیاز و اطمینان نهایی
        final_score, final_confidence = self._calculate_final_score(dimensions)

        # گام 5: تحلیل توافق بین dimensions
        agreement = self._analyze_agreement(dimensions)

        # گام 6: تعیین سیگنال نهایی
        final_signal = self._determine_signal(final_score, agreement)

        # گام 7: محاسبه قدرت سیگنال
        signal_strength = self._calculate_signal_strength(final_score, final_confidence, agreement)

        # گام 8: ارزیابی ریسک
        risk_level, risk_factors = self._assess_risk(dimensions, agreement, final_confidence)

        # گام 9: تولید توصیه‌ها
        recommendation = self._generate_recommendation(
            final_signal, signal_strength, risk_level, agreement, dimensions
        )

        entry_strategy = self._generate_entry_strategy(final_signal, dimensions, risk_level)

        exit_strategy = self._generate_exit_strategy(final_signal, dimensions, risk_level)

        stop_loss = self._suggest_stop_loss(final_signal, dimensions, risk_level)

        take_profit = self._suggest_take_profit(final_signal, dimensions, signal_strength)

        # گام 10: تحلیل شرایط بازار
        market_condition = self._analyze_market_condition(dimensions)

        # گام 11: استخراج نکات کلیدی
        key_insights = self._extract_key_insights(dimensions, agreement, final_signal)

        # خروجی امتیازها در مقیاس [-100, 100]
        dimensions = self._scale_dimensions_100(dimensions)
        final_score_out = self._scale_score_100(final_score)

        return FiveDimensionalDecision(
            timestamp=datetime.now(),
            dimensions=dimensions,
            final_score=final_score_out,
            final_confidence=final_confidence,
            final_signal=final_signal,
            signal_strength=signal_strength,
            agreement=agreement,
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommendation=recommendation,
            entry_strategy=entry_strategy,
            exit_strategy=exit_strategy,
            stop_loss_suggestion=stop_loss,
            take_profit_suggestion=take_profit,
            market_condition=market_condition,
            key_insights=key_insights,
        )

    def _collect_dimension_states(
        self,
        trend: TrendScore,
        momentum: MomentumScore,
        volatility: VolatilityScore,
        cycle: CycleScore,
        sr: SupportResistanceScore,
    ) -> dict[str, DimensionState]:
        """جمع‌آوری state هر dimension"""

        return {
            "trend": DimensionState(
                name="Trend",
                score=trend.score,
                confidence=trend.accuracy,
                signal=trend.signal,
                weight=self.weights["trend"],
                volume_adjusted_score=trend.score,
                volume_adjustment=0.0,
                description=f"روند {self._translate_signal(trend.signal)}",
            ),
            "momentum": DimensionState(
                name="Momentum",
                score=momentum.score,
                confidence=momentum.accuracy,
                signal=momentum.signal,
                weight=self.weights["momentum"],
                volume_adjusted_score=momentum.score,
                volume_adjustment=0.0,
                description=f"مومنتوم {self._translate_signal(momentum.signal)}",
            ),
            "volatility": DimensionState(
                name="Volatility",
                score=volatility.score,
                confidence=volatility.accuracy,
                signal=volatility.signal,
                weight=self.weights["volatility"],
                volume_adjusted_score=volatility.score,
                volume_adjustment=0.0,
                description=f"نوسان {self._translate_signal(volatility.signal)}",
            ),
            "cycle": DimensionState(
                name="Cycle",
                score=cycle.score,
                confidence=cycle.accuracy,
                signal=cycle.signal,
                weight=self.weights["cycle"],
                volume_adjusted_score=cycle.score,
                volume_adjustment=0.0,
                description=f"سیکل در فاز {cycle.phase}",
            ),
            "support_resistance": DimensionState(
                name="Support/Resistance",
                score=sr.score,
                confidence=sr.accuracy,
                signal=self._map_sr_signal_to_signal_strength(sr.signal),
                weight=self.weights["support_resistance"],
                volume_adjusted_score=sr.score,
                volume_adjustment=0.0,
                description=f"نزدیک {sr.nearest_level_type if sr.nearest_level_type else 'سطح'}",
            ),
        }

    def _map_sr_signal_to_signal_strength(self, sr_signal: str) -> SignalStrength:
        """Map S/R signal string to SignalStrength enum"""
        mapping = {
            "NEAR_SUPPORT": SignalStrength.BULLISH,
            "NEAR_RESISTANCE": SignalStrength.BEARISH,
            "AT_SUPPORT": SignalStrength.VERY_BULLISH,
            "AT_RESISTANCE": SignalStrength.VERY_BEARISH,
            "NEUTRAL": SignalStrength.NEUTRAL,
        }
        return mapping.get(sr_signal, SignalStrength.NEUTRAL)

    def _apply_volume_adjustments(
        self,
        dimensions: dict[str, DimensionState],
        trend: TrendScore,
        momentum: MomentumScore,
        volatility: VolatilityScore,
        cycle: CycleScore,
        sr: SupportResistanceScore,
    ) -> dict[str, DimensionState]:
        """
        اعمال تعدیلات حجم از Volume-Dimension Matrix

        این متد از ml/volume_dimension_matrix.py استفاده می‌کند
        """
        try:
            from gravity_tech.ml.volume_dimension_matrix import VolumeDimensionMatrix

            vol_matrix = VolumeDimensionMatrix(self.candles)

            # محاسبه interactions برای هر dimension
            interactions = vol_matrix.calculate_all_interactions(
                trend, momentum, volatility, cycle, sr
            )

            # اعمال adjustments
            for dim_name, interaction in interactions.items():
                if dim_name in dimensions:
                    dim = dimensions[dim_name]

                    # تعدیل score
                    dim.volume_adjusted_score = np.clip(
                        dim.score + interaction.interaction_score, -1.0, 1.0
                    )
                    dim.volume_adjustment = interaction.interaction_score

                    # تعدیل confidence بر اساس نوع interaction
                    confidence_multiplier = self._get_confidence_multiplier(
                        interaction.interaction_type
                    )
                    dim.confidence = np.clip(
                        max(0.1, dim.confidence) * confidence_multiplier, 0.0, 1.0
                    )

                    # به‌روزرسانی description
                    dim.description += f" (حجم: {interaction.interaction_type.value})"

        except Exception as exc:
            # اگر ماتریس حجم دردسترس نباشد یا اعتبارسنجی شکست بخورد، یادداشت کن
            for dim in dimensions.values():
                dim.description += f" (حجم غیرفعال: {exc})"

        return dimensions

    def _get_confidence_multiplier(self, interaction_type) -> float:
        """محاسبه ضریب confidence بر اساس نوع interaction"""
        from gravity_tech.ml.volume_dimension_matrix import InteractionType

        multipliers = {
            InteractionType.STRONG_CONFIRM: 1.15,
            InteractionType.CONFIRM: 1.08,
            InteractionType.NEUTRAL: 1.0,
            InteractionType.WARN: 0.92,
            InteractionType.DIVERGENCE: 0.75,
            InteractionType.FAKE: 0.60,
        }
        return multipliers.get(interaction_type, 1.0)

    def _calculate_dynamic_weights(
        self, dimensions: dict[str, DimensionState]
    ) -> dict[str, DimensionState]:
        """محاسبه وزن‌های داینامیک بر اساس confidence"""

        # محاسبه weighted confidence با کف 0.1 برای جلوگیری از حذف کامل یک بعد
        weighted_confidences = {
            name: dim.weight * max(0.1, dim.confidence) for name, dim in dimensions.items()
        }

        total_weighted = sum(weighted_confidences.values())

        # تعدیل وزن‌ها
        if total_weighted > 0:
            for name, dim in dimensions.items():
                dim.weight = weighted_confidences[name] / total_weighted

        return dimensions

    def _calculate_final_score(self, dimensions: dict[str, DimensionState]) -> tuple[float, float]:
        """محاسبه امتیاز و اطمینان نهایی"""

        # امتیاز نهایی = میانگین وزن‌دار
        final_score = sum(dim.volume_adjusted_score * dim.weight for dim in dimensions.values())

        # اطمینان نهایی = ترکیب agreement + accuracy
        scores = [dim.volume_adjusted_score for dim in dimensions.values()]
        confidences = [dim.confidence for dim in dimensions.values()]

        # Agreement: چقدر dimensions هماهنگ هستند؟ (در حالت خنثی تنبیه سنگین نمی‌شود)
        if len(scores) > 1:
            mean_abs = abs(np.mean(scores))
            agreement = 1.0 - (np.std(scores) / (mean_abs + 0.5))
        else:
            agreement = 1.0
        agreement = float(np.clip(agreement, 0.0, 1.0))

        # Accuracy: میانگین confidence
        avg_accuracy = float(np.clip(np.mean(confidences), 0.0, 1.0))

        # ترکیب (60% agreement + 40% accuracy)
        final_confidence = (agreement * 0.6) + (avg_accuracy * 0.4)

        return np.clip(final_score, -1.0, 1.0), np.clip(final_confidence, 0.0, 1.0)

    def _analyze_agreement(self, dimensions: dict[str, DimensionState]) -> DimensionAgreement:
        """تحلیل توافق بین dimensions"""

        bullish = []
        bearish = []
        neutral = []

        for name, dim in dimensions.items():
            if dim.volume_adjusted_score > 0.2:
                bullish.append(name)
            elif dim.volume_adjusted_score < -0.2:
                bearish.append(name)
            else:
                neutral.append(name)

        # محاسبه overall agreement
        scores = [dim.volume_adjusted_score for dim in dimensions.values()]
        if len(scores) > 1:
            cv = np.std(scores) / (abs(np.mean(scores)) + 0.01)  # Coefficient of Variation
            overall_agreement = max(0, 1 - cv)
        else:
            overall_agreement = 1.0

        # قوی‌ترین و ضعیف‌ترین dimension
        sorted_dims = sorted(
            dimensions.items(),
            key=lambda x: abs(x[1].volume_adjusted_score * x[1].weight),
            reverse=True,
        )
        strongest = sorted_dims[0][0] if sorted_dims else ""
        weakest = sorted_dims[-1][0] if sorted_dims else ""

        # آیا تناقض وجود دارد؟
        conflicting = len(bullish) > 0 and len(bearish) > 0

        return DimensionAgreement(
            overall_agreement=float(overall_agreement),
            bullish_dimensions=bullish,
            bearish_dimensions=bearish,
            neutral_dimensions=neutral,
            strongest_dimension=strongest,
            weakest_dimension=weakest,
            conflicting=conflicting,
        )

    def _determine_signal(
        self, final_score: float, agreement: DimensionAgreement
    ) -> DecisionSignal:
        """تعیین سیگنال نهایی بر اساس score و agreement"""

        # بررسی توافق کامل
        # Very Strong Signals: همه dimensions موافق
        if final_score > 0.7 and agreement.overall_agreement > 0.9:
            return DecisionSignal.VERY_STRONG_BUY
        elif final_score < -0.7 and agreement.overall_agreement > 0.9:
            return DecisionSignal.VERY_STRONG_SELL

        # Strong Signals: اکثریت قاطع
        elif final_score > 0.5 and agreement.overall_agreement > 0.75:
            return DecisionSignal.STRONG_BUY
        elif final_score < -0.5 and agreement.overall_agreement > 0.75:
            return DecisionSignal.STRONG_SELL

        # Regular Signals: اکثریت ساده
        elif final_score > 0.3 and agreement.overall_agreement > 0.6:
            return DecisionSignal.BUY
        elif final_score < -0.3 and agreement.overall_agreement > 0.6:
            return DecisionSignal.SELL

        # Weak Signals: کمی مثبت/منفی
        elif final_score > 0.1:
            return DecisionSignal.WEAK_BUY
        elif final_score < -0.1:
            return DecisionSignal.WEAK_SELL

        # Neutral: بدون سیگنال واضح
        else:
            return DecisionSignal.NEUTRAL

    def _calculate_signal_strength(
        self, final_score: float, final_confidence: float, agreement: DimensionAgreement
    ) -> float:
        """محاسبه قدرت سیگنال [0, 1]"""

        # ترکیب 3 عامل
        score_strength = abs(final_score)  # [0, 1]
        confidence_strength = final_confidence  # [0, 1]
        agreement_strength = agreement.overall_agreement  # [0, 1]

        # میانگین وزن‌دار (score مهم‌تر است)
        signal_strength = (
            score_strength * 0.5 + confidence_strength * 0.3 + agreement_strength * 0.2
        )

        return np.clip(signal_strength, 0.0, 1.0)

    def _assess_risk(
        self,
        dimensions: dict[str, DimensionState],
        agreement: DimensionAgreement,
        final_confidence: float,
    ) -> tuple[RiskLevel, list[str]]:
        """ارزیابی سطح ریسک و عوامل ریسک"""

        risk_factors = []

        # عامل 1: عدم توافق
        if agreement.conflicting:
            risk_factors.append("تناقض بین dimensions")

        if agreement.overall_agreement < 0.5:
            risk_factors.append("عدم توافق قوی بین dimensions")

        # عامل 2: confidence پایین
        if final_confidence < 0.6:
            risk_factors.append("اطمینان پایین در تحلیل")

        # عامل 3: نوسان بالا
        vol_dim = dimensions.get("volatility")
        if vol_dim and vol_dim.score > 0.5:
            risk_factors.append("نوسان بالای بازار")

        # عامل 4: فاز بازار
        cycle_dim = dimensions.get("cycle")
        if cycle_dim and "distribution" in cycle_dim.description.lower():
            risk_factors.append("فاز توزیع - احتمال ریزش")

        # عامل 5: واگرایی حجم
        volume_divergences = [
            dim.name
            for dim in dimensions.values()
            if "DIVERGENCE" in dim.description or "FAKE" in dim.description
        ]
        if volume_divergences:
            risk_factors.append(f"واگرایی حجم در {', '.join(volume_divergences)}")

        # تعیین سطح ریسک
        risk_score = len(risk_factors)

        if risk_score == 0 and agreement.overall_agreement > 0.9:
            risk_level = RiskLevel.VERY_LOW
        elif risk_score <= 1 and agreement.overall_agreement > 0.75:
            risk_level = RiskLevel.LOW
        elif risk_score <= 2 and agreement.overall_agreement > 0.6:
            risk_level = RiskLevel.MODERATE
        elif risk_score <= 3:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.VERY_HIGH

        return risk_level, risk_factors

    def _generate_recommendation(
        self,
        signal: DecisionSignal,
        strength: float,
        risk: RiskLevel,
        agreement: DimensionAgreement,
        dimensions: dict[str, DimensionState],
    ) -> str:
        """تولید توصیه اصلی"""

        recommendations = []

        # بر اساس سیگنال
        if signal in [DecisionSignal.VERY_STRONG_BUY, DecisionSignal.STRONG_BUY]:
            recommendations.append("🟢 **خرید قوی توصیه می‌شود**")
            recommendations.append(f"قدرت سیگنال: {strength * 100:.1f}%")
            recommendations.append(f"توافق dimensions: {agreement.overall_agreement * 100:.1f}%")

            if len(agreement.bullish_dimensions) == 5:
                recommendations.append("✅ همه 5 dimension سیگنال صعودی می‌دهند!")

        elif signal == DecisionSignal.BUY:
            recommendations.append("🟢 خرید توصیه می‌شود")
            recommendations.append(f"قدرت سیگنال: {strength * 100:.1f}%")

        elif signal == DecisionSignal.WEAK_BUY:
            recommendations.append("🟡 خرید محتاطانه یا انتظار برای تایید بیشتر")

        elif signal in [DecisionSignal.VERY_STRONG_SELL, DecisionSignal.STRONG_SELL]:
            recommendations.append("🔴 **فروش قوی توصیه می‌شود**")
            recommendations.append(f"قدرت سیگنال: {strength * 100:.1f}%")

        elif signal == DecisionSignal.SELL:
            recommendations.append("🔴 فروش توصیه می‌شود")

        elif signal == DecisionSignal.WEAK_SELL:
            recommendations.append("🟡 فروش محتاطانه یا کاهش پوزیشن")

        else:  # NEUTRAL
            recommendations.append("⚪ خنثی - بهتر است صبر کنید")
            recommendations.append("انتظار برای سیگنال واضح‌تر")

        # بر اساس ریسک
        recommendations.append(f"\nسطح ریسک: {risk.value}")

        if risk in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            recommendations.append("⚠️ **هشدار: ریسک بالا! با احتیاط عمل کنید**")

        return "\n".join(recommendations)

    def _generate_entry_strategy(
        self, signal: DecisionSignal, dimensions: dict[str, DimensionState], risk: RiskLevel
    ) -> str:
        """تولید استراتژی ورود"""

        if signal in [
            DecisionSignal.VERY_STRONG_BUY,
            DecisionSignal.STRONG_BUY,
            DecisionSignal.BUY,
        ]:
            if risk in [RiskLevel.VERY_LOW, RiskLevel.LOW]:
                return "ورود فوری با 50-70% سرمایه، بقیه در اصلاحات"
            else:
                return "ورود تدریجی: 30% الان، 40% در اصلاح، 30% در تایید"

        elif signal == DecisionSignal.WEAK_BUY:
            return "انتظار برای تایید بیشتر یا ورود با مقدار کم (10-20%)"

        elif signal in [
            DecisionSignal.VERY_STRONG_SELL,
            DecisionSignal.STRONG_SELL,
            DecisionSignal.SELL,
        ]:
            if risk in [RiskLevel.VERY_LOW, RiskLevel.LOW]:
                return "خروج فوری 50-70% یا Short با حجم متوسط"
            else:
                return "خروج تدریجی یا Short با حجم کم"

        else:
            return "بدون ورود - انتظار برای سیگنال واضح"

    def _generate_exit_strategy(
        self, signal: DecisionSignal, dimensions: dict[str, DimensionState], risk: RiskLevel
    ) -> str:
        """تولید استراتژی خروج"""

        if signal in [DecisionSignal.VERY_STRONG_BUY, DecisionSignal.STRONG_BUY]:
            return "Trailing Stop 5-7% یا خروج در نشانه‌های ضعف روند"

        elif signal == DecisionSignal.BUY:
            return "Trailing Stop 3-5% یا خروج در رسیدن به هدف"

        elif signal in [DecisionSignal.VERY_STRONG_SELL, DecisionSignal.STRONG_SELL]:
            return "خروج سریع از لانگ‌ها یا Trailing Stop برای شورت 5-7%"

        else:
            return "خروج در صورت شکست سطح حمایت/مقاومت کلیدی"

    def _suggest_stop_loss(
        self, signal: DecisionSignal, dimensions: dict[str, DimensionState], risk: RiskLevel
    ) -> str:
        """پیشنهاد استاپ لاس"""

        sr_dim = dimensions.get("support_resistance")

        if signal in [
            DecisionSignal.VERY_STRONG_BUY,
            DecisionSignal.STRONG_BUY,
            DecisionSignal.BUY,
        ]:
            if sr_dim and sr_dim.score > 0:
                return "استاپ 2-3% زیر نزدیک‌ترین سطح حمایت"
            else:
                return "استاپ 3-5% زیر قیمت ورود"

        elif signal in [
            DecisionSignal.VERY_STRONG_SELL,
            DecisionSignal.STRONG_SELL,
            DecisionSignal.SELL,
        ]:
            if sr_dim and sr_dim.score < 0:
                return "استاپ 2-3% بالای نزدیک‌ترین سطح مقاومت"
            else:
                return "استاپ 3-5% بالای قیمت ورود"

        else:
            return "استاپ سفت 5-7% (با توجه به عدم قطعیت)"

    def _suggest_take_profit(
        self, signal: DecisionSignal, dimensions: dict[str, DimensionState], strength: float
    ) -> str:
        """پیشنهاد حد سود"""

        if signal in [DecisionSignal.VERY_STRONG_BUY, DecisionSignal.STRONG_BUY]:
            return "TP1: +5%, TP2: +10%, TP3: +15% (بسته به قدرت)"

        elif signal == DecisionSignal.BUY:
            return "TP1: +3%, TP2: +6%, TP3: +10%"

        elif signal in [DecisionSignal.VERY_STRONG_SELL, DecisionSignal.STRONG_SELL]:
            return "TP1: -5%, TP2: -10%, TP3: -15%"

        elif signal == DecisionSignal.SELL:
            return "TP1: -3%, TP2: -6%, TP3: -10%"

        else:
            return "هدف مشخص نیست - بر اساس R/R حداقل 1:2"

    def _analyze_market_condition(self, dimensions: dict[str, DimensionState]) -> str:
        """تحلیل شرایط کلی بازار"""

        conditions = []

        # Trend
        trend = dimensions.get("trend")
        if trend:
            if trend.volume_adjusted_score > 0.5:
                conditions.append("روند صعودی قوی")
            elif trend.volume_adjusted_score < -0.5:
                conditions.append("روند نزولی قوی")
            else:
                conditions.append("روند ضعیف یا خنثی")

        # Cycle
        cycle = dimensions.get("cycle")
        if cycle:
            conditions.append(cycle.description)

        # Volatility
        vol = dimensions.get("volatility")
        if vol and vol.score > 0.5:
            conditions.append("نوسان بالا")
        elif vol and vol.score < -0.5:
            conditions.append("نوسان پایین")

        return ", ".join(conditions)

    def _extract_key_insights(
        self,
        dimensions: dict[str, DimensionState],
        agreement: DimensionAgreement,
        signal: DecisionSignal,
    ) -> list[str]:
        """استخراج نکات کلیدی"""

        insights = []

        # نکته 1: قوی‌ترین dimension
        strongest = dimensions.get(agreement.strongest_dimension)
        if strongest:
            insights.append(
                f"💪 قوی‌ترین بُعد: {strongest.name} با امتیاز {strongest.volume_adjusted_score:.2f}"
            )

        # نکته 2: تناقض
        if agreement.conflicting:
            insights.append(
                f"⚠️ تناقض: {len(agreement.bullish_dimensions)} بُعد صعودی، "
                f"{len(agreement.bearish_dimensions)} بُعد نزولی"
            )

        # نکته 3: تاثیر حجم
        volume_impacts = [
            f"{dim.name}: {dim.volume_adjustment:+.2f}"
            for dim in dimensions.values()
            if abs(dim.volume_adjustment) > 0.05
        ]
        if volume_impacts:
            insights.append(f"📊 تاثیر حجم: {', '.join(volume_impacts)}")

        # نکته 4: فاز بازار
        cycle = dimensions.get("cycle")
        if cycle:
            insights.append(f"🔄 {cycle.description}")

        # نکته 5: نزدیکی به S/R
        sr = dimensions.get("support_resistance")
        if sr:
            insights.append(f"📍 {sr.description}")

        return insights

    @staticmethod
    def _translate_signal(signal: SignalStrength) -> str:
        """ترجمه signal به فارسی"""
        translations = {
            SignalStrength.VERY_BULLISH: "بسیار صعودی",
            SignalStrength.BULLISH: "صعودی",
            SignalStrength.BULLISH_BROKEN: "صعودی شکسته",
            SignalStrength.NEUTRAL: "خنثی",
            SignalStrength.BEARISH_BROKEN: "نزولی شکسته",
            SignalStrength.BEARISH: "نزولی",
            SignalStrength.VERY_BEARISH: "بسیار نزولی",
        }
        return translations.get(signal, str(signal))
