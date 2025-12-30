"""
Multi-Horizon Support/Resistance Analysis با ML

این ماژول تحلیل Support & Resistance را برای 3 افق زمانی انجام می‌دهد:
- کوتاه‌مدت (3 روز)
- میان‌مدت (7 روز)
- بلندمدت (30 روز)

برای هر افق، اسکور و احتمالات زیر محاسبه می‌شود:
- Bounce Probability (احتمال برگشت از سطح)
- Breakout Probability (احتمال شکست سطح)
- Level Strength (قدرت سطح)
- Support/Resistance Score (اسکور کلی S/R)
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gravity_tech.ml.multi_horizon_support_resistance_features import (
    MultiHorizonSupportResistanceFeatureExtractor,
)
from gravity_tech.models.schemas import Candle


@dataclass
class SupportResistanceScore:
    """اسکور Support & Resistance برای یک افق زمانی"""

    horizon: str  # '3d', '7d', '30d'
    score: float  # [-1, +1]: -1=strong resistance nearby, +1=strong support nearby
    confidence: float  # [0, 1]

    # Probabilities
    bounce_probability: float  # احتمال برگشت از سطح نزدیک [0, 1]
    breakout_probability: float  # احتمال شکست سطح نزدیک [0, 1]

    # Levels
    nearest_support: float  # قیمت نزدیک‌ترین حمایت
    nearest_resistance: float  # قیمت نزدیک‌ترین مقاومت
    support_strength: float  # قدرت حمایت [0, 1]
    resistance_strength: float  # قدرت مقاومت [0, 1]

    # Position
    sr_position: float  # موقعیت در محدوده [0, 1]: 0=at support, 1=at resistance
    distance_to_key_level: float  # فاصله تا سطح کلیدی (%)

    # Signal
    signal: str  # NEAR_SUPPORT, NEAR_RESISTANCE, NEUTRAL, AT_SUPPORT, AT_RESISTANCE
    recommendation: str  # توصیه معاملاتی

    def get_position_label(self) -> str:
        """برچسب موقعیت"""
        if self.sr_position < 0.2:
            return "NEAR_SUPPORT"
        elif self.sr_position < 0.4:
            return "BELOW_MIDRANGE"
        elif self.sr_position < 0.6:
            return "MIDRANGE"
        elif self.sr_position < 0.8:
            return "ABOVE_MIDRANGE"
        else:
            return "NEAR_RESISTANCE"

    def get_action_recommendation(self) -> str:
        """توصیه اقدام"""
        if self.signal == "NEAR_SUPPORT" and self.bounce_probability > 0.6:
            return "CONSIDER_BUY"
        elif self.signal == "NEAR_RESISTANCE" and self.bounce_probability > 0.6:
            return "CONSIDER_SELL"
        elif self.breakout_probability > 0.7:
            return "WATCH_FOR_BREAKOUT"
        else:
            return "WAIT"

    @property
    def accuracy(self) -> float:
        """سازگاری با موتور تصمیم‌گیری"""
        return self.confidence

    @property
    def nearest_level_type(self) -> str:
        """
        برچسب سطح غالب نزدیک.

        Returns:
            'support', 'resistance' یا 'level' در صورت عدم تشخیص.
        """
        signal = (self.signal or "").upper()
        if "SUPPORT" in signal:
            return "Support"
        if "RESISTANCE" in signal:
            return "Resistance"
        if self.sr_position <= 0.5:
            return "Support"
        if self.sr_position > 0.5:
            return "Resistance"
        return "Level"


@dataclass
class MultiHorizonSupportResistanceAnalysis:
    """تحلیل Multi-Horizon کامل"""

    score_3d: SupportResistanceScore
    score_7d: SupportResistanceScore
    score_30d: SupportResistanceScore

    # Combined metrics
    overall_sr_score: float  # میانگین وزن‌دار اسکورها
    overall_confidence: float  # میانگین confidence

    # Alignment
    horizons_agreement: float  # توافق بین افق‌ها [0, 1]
    alignment: str  # ALIGNED, MIXED, CONFLICTING

    # Key levels (از همه افق‌ها)
    critical_support: float
    critical_resistance: float

    # Overall recommendation
    overall_signal: str
    overall_recommendation: str


class MultiHorizonSupportResistanceAnalyzer:
    """تحلیل‌گر ML-based Support & Resistance"""

    def __init__(self, weights_path: str | None = None, model_path: str | None = None):
        """
        Initialize analyzer

        Args:
            weights_path: مسیر فایل وزن‌های ML (اختیاری)
        """
        self.feature_extractor = MultiHorizonSupportResistanceFeatureExtractor()
        self.weights = self._load_weights(weights_path)
        self.model_state = self._load_model_state(model_path)

    def analyze(self, candles: list[Candle]) -> MultiHorizonSupportResistanceAnalysis:
        """
        تحلیل کامل Support & Resistance برای همه افق‌ها

        Args:
            candles: لیست کندل‌ها

        Returns:
            MultiHorizonSupportResistanceAnalysis
        """
        if len(candles) < 50:
            raise ValueError("حداقل 50 کندل نیاز است")

        current_price = candles[-1].close

        # استخراج ویژگی‌ها
        all_features = self.feature_extractor.extract_all_horizons(candles)

        # تحلیل هر افق
        score_3d = self._calculate_horizon_score(candles, "3d", all_features, current_price)
        score_7d = self._calculate_horizon_score(candles, "7d", all_features, current_price)
        score_30d = self._calculate_horizon_score(candles, "30d", all_features, current_price)

        # ترکیب اسکورها
        overall_sr_score = score_3d.score * 0.4 + score_7d.score * 0.35 + score_30d.score * 0.25

        overall_confidence = (
            score_3d.confidence * 0.4 + score_7d.confidence * 0.35 + score_30d.confidence * 0.25
        )

        # محاسبه توافق
        horizons_agreement = self._calculate_horizons_agreement(score_3d, score_7d, score_30d)

        alignment = self._determine_alignment(score_3d, score_7d, score_30d)

        # سطوح بحرانی
        critical_support = max(
            score_3d.nearest_support, score_7d.nearest_support, score_30d.nearest_support
        )
        critical_resistance = min(
            score_3d.nearest_resistance, score_7d.nearest_resistance, score_30d.nearest_resistance
        )

        # سیگنال و توصیه کلی
        overall_signal = self._determine_overall_signal(score_3d, score_7d, score_30d)
        overall_recommendation = self._generate_overall_recommendation(
            overall_sr_score, overall_signal, horizons_agreement, score_3d, score_7d, score_30d
        )

        return MultiHorizonSupportResistanceAnalysis(
            score_3d=score_3d,
            score_7d=score_7d,
            score_30d=score_30d,
            overall_sr_score=overall_sr_score,
            overall_confidence=overall_confidence,
            horizons_agreement=horizons_agreement,
            alignment=alignment,
            critical_support=critical_support,
            critical_resistance=critical_resistance,
            overall_signal=overall_signal,
            overall_recommendation=overall_recommendation,
        )

    def _calculate_horizon_score(
        self,
        candles: list[Candle],
        horizon: str,
        all_features: dict[str, float],
        current_price: float,
    ) -> SupportResistanceScore:
        """محاسبه اسکور یک افق زمانی"""

        # دریافت وزن‌ها
        weights = self.weights[horizon]

        # استخراج ویژگی‌های این افق
        prefix = f"{horizon}_"
        horizon_features = {
            k.replace(prefix, ""): v for k, v in all_features.items() if k.startswith(prefix)
        }

        # محاسبه اسکور وزن‌دار
        score = self._predict_score(horizon, horizon_features, weights)

        # محاسبه confidence
        confidence = self._calculate_confidence(horizon_features)

        # محاسبه احتمالات
        bounce_prob = self._calculate_bounce_probability(horizon_features, weights)
        breakout_prob = self._calculate_breakout_probability(horizon_features, weights)

        # استخراج سطوح
        resistance_dist_pct = horizon_features.get("nearest_resistance_dist", 5.0)
        support_dist_pct = horizon_features.get("nearest_support_dist", 5.0)

        nearest_resistance = current_price * (1 + resistance_dist_pct / 100)
        nearest_support = current_price * (1 - support_dist_pct / 100)

        resistance_strength = horizon_features.get("resistance_strength", 0.5)
        support_strength = horizon_features.get("support_strength", 0.5)

        sr_position = horizon_features.get("sr_position", 0.5)
        distance_to_key_level = min(resistance_dist_pct, support_dist_pct)

        # تعیین سیگنال
        signal = self._determine_signal(sr_position, distance_to_key_level)

        # توصیه
        recommendation = self._generate_recommendation(
            signal, bounce_prob, breakout_prob, sr_position, horizon
        )

        return SupportResistanceScore(
            horizon=horizon,
            score=score,
            confidence=confidence,
            bounce_probability=bounce_prob,
            breakout_probability=breakout_prob,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            support_strength=support_strength,
            resistance_strength=resistance_strength,
            sr_position=sr_position,
            distance_to_key_level=distance_to_key_level,
            signal=signal,
            recommendation=recommendation,
        )

    def _predict_score(
        self,
        horizon: str,
        features: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        """Use a trained regression bundle when available."""
        if self.model_state and horizon in self.model_state:
            bundle = self.model_state[horizon]
            feature_names = bundle.get("feature_names", [])
            coef = bundle.get("weights", [])
            intercept = bundle.get("intercept", 0.0)
            total = intercept
            for name, weight in zip(feature_names, coef, strict=False):
                total += features.get(name, 0.0) * weight
            return float(np.clip(total, -1.0, 1.0))
        return self._calculate_weighted_score(features, weights)

    def _calculate_weighted_score(
        self, features: dict[str, float], weights: dict[str, float]
    ) -> float:
        """محاسبه اسکور وزن‌دار از ویژگی‌ها"""
        score = 0.0
        total_weight = 0.0

        for feature_name, weight in weights.items():
            if feature_name in features:
                score += features[feature_name] * weight
                total_weight += abs(weight)

        if total_weight > 0:
            score = score / total_weight

        # Normalize to [-1, +1]
        return np.clip(score, -1.0, 1.0)

    def _calculate_confidence(self, features: dict[str, float]) -> float:
        """محاسبه confidence"""
        # Confidence بالاتر وقتی:
        # 1. تعداد سطوح بیشتر
        # 2. قدرت سطوح بیشتر
        # 3. فاصله به سطح کمتر

        level_count = features.get("resistance_count", 0) + features.get("support_count", 0)
        level_strength = (
            features.get("resistance_strength", 0) + features.get("support_strength", 0)
        ) / 2
        level_density = features.get("level_density", 0)
        distance = features.get("nearest_level_dist", 10.0)

        # نزدیکی به سطح → confidence بیشتر
        distance_factor = max(0, 1.0 - distance / 10.0)

        # ترکیب
        confidence = (
            min(level_count / 5.0, 1.0) * 0.3
            + level_strength * 0.3
            + level_density * 0.2
            + distance_factor * 0.2
        )

        return np.clip(confidence, 0.0, 1.0)

    def _calculate_bounce_probability(
        self, features: dict[str, float], weights: dict[str, float]
    ) -> float:
        """محاسبه احتمال برگشت از سطح"""
        # احتمال bounce بالاتر وقتی:
        # 1. نزدیک سطح قوی هستیم
        # 2. چند سطح در همان ناحیه (clustering)
        # 3. سطح چند بار تست شده

        distance = features.get("nearest_level_dist", 10.0)
        level_strength = max(
            features.get("resistance_strength", 0), features.get("support_strength", 0)
        )
        level_density = features.get("level_density", 0)
        sr_position = features.get("sr_position", 0.5)

        # نزدیک به support یا resistance
        near_level = sr_position < 0.2 or sr_position > 0.8

        # فاکتور فاصله (نزدیک‌تر = احتمال بیشتر)
        distance_factor = max(0, 1.0 - distance / 5.0)

        # محاسبه احتمال
        bounce_prob = (
            distance_factor * 0.4
            + level_strength * 0.3
            + level_density * 0.2
            + (1.0 if near_level else 0.0) * 0.1
        )

        return np.clip(bounce_prob, 0.0, 1.0)

    def _calculate_breakout_probability(
        self, features: dict[str, float], weights: dict[str, float]
    ) -> float:
        """محاسبه احتمال شکست سطح"""
        # احتمال breakout بالاتر وقتی:
        # 1. سطح ضعیف
        # 2. momentum قوی
        # 3. چند بار سطح تست شده (آماده شکست)

        level_strength = max(
            features.get("resistance_strength", 0), features.get("support_strength", 0)
        )
        sr_bias = features.get("sr_bias", 0)

        # سطح ضعیف → احتمال شکست بیشتر
        weak_level = 1.0 - level_strength

        # momentum قوی (از sr_bias)
        momentum_strong = abs(sr_bias)

        # محاسبه احتمال
        breakout_prob = weak_level * 0.5 + momentum_strong * 0.5

        return np.clip(breakout_prob, 0.0, 1.0)

    def _determine_signal(self, sr_position: float, distance_to_level: float) -> str:
        """تعیین سیگنال بر اساس موقعیت"""
        # در سطح (فاصله < 1%)
        if distance_to_level < 1.0:
            if sr_position < 0.3:
                return "AT_SUPPORT"
            elif sr_position > 0.7:
                return "AT_RESISTANCE"

        # نزدیک سطح (فاصله < 2%)
        if distance_to_level < 2.0:
            if sr_position < 0.3:
                return "NEAR_SUPPORT"
            elif sr_position > 0.7:
                return "NEAR_RESISTANCE"

        return "NEUTRAL"

    def _generate_recommendation(
        self,
        signal: str,
        bounce_prob: float,
        breakout_prob: float,
        sr_position: float,
        horizon: str,
    ) -> str:
        """تولید توصیه برای یک افق"""

        if signal == "AT_SUPPORT":
            if bounce_prob > 0.7:
                return f"فرصت خرید قوی - حمایت قوی ({horizon})"
            elif bounce_prob > 0.5:
                return f"فرصت خرید متوسط - حمایت ({horizon})"
            elif breakout_prob > 0.6:
                return f"احتمال شکست حمایت - احتیاط ({horizon})"
            else:
                return f"در حمایت - منتظر تایید ({horizon})"

        elif signal == "NEAR_SUPPORT":
            if bounce_prob > 0.6:
                return f"نزدیک حمایت - آماده خرید ({horizon})"
            else:
                return f"نزدیک حمایت - مراقب شکست ({horizon})"

        elif signal == "AT_RESISTANCE":
            if bounce_prob > 0.7:
                return f"فرصت فروش قوی - مقاومت قوی ({horizon})"
            elif bounce_prob > 0.5:
                return f"فرصت فروش متوسط - مقاومت ({horizon})"
            elif breakout_prob > 0.6:
                return f"احتمال شکست مقاومت - فرصت خرید ({horizon})"
            else:
                return f"در مقاومت - منتظر تایید ({horizon})"

        elif signal == "NEAR_RESISTANCE":
            if bounce_prob > 0.6:
                return f"نزدیک مقاومت - آماده فروش ({horizon})"
            else:
                return f"نزدیک مقاومت - مراقب شکست ({horizon})"

        else:  # NEUTRAL
            if sr_position < 0.5:
                return f"بین سطوح - گرایش به حمایت ({horizon})"
            else:
                return f"بین سطوح - گرایش به مقاومت ({horizon})"

    def _calculate_horizons_agreement(
        self,
        score_3d: SupportResistanceScore,
        score_7d: SupportResistanceScore,
        score_30d: SupportResistanceScore,
    ) -> float:
        """محاسبه توافق بین افق‌ها"""
        scores = [score_3d.score, score_7d.score, score_30d.score]
        positions = [score_3d.sr_position, score_7d.sr_position, score_30d.sr_position]

        # توافق در score
        score_std = np.std(scores)
        score_agreement = max(0, 1.0 - score_std)

        # توافق در position
        position_std = np.std(positions)
        position_agreement = max(0, 1.0 - position_std)

        # میانگین
        return (score_agreement + position_agreement) / 2

    def _determine_alignment(
        self,
        score_3d: SupportResistanceScore,
        score_7d: SupportResistanceScore,
        score_30d: SupportResistanceScore,
    ) -> str:
        """تعیین alignment بین افق‌ها"""
        agreement = self._calculate_horizons_agreement(score_3d, score_7d, score_30d)

        if agreement > 0.7:
            return "ALIGNED"
        elif agreement > 0.4:
            return "MIXED"
        else:
            return "CONFLICTING"

    def _determine_overall_signal(
        self,
        score_3d: SupportResistanceScore,
        score_7d: SupportResistanceScore,
        score_30d: SupportResistanceScore,
    ) -> str:
        """تعیین سیگنال کلی"""
        signals = [score_3d.signal, score_7d.signal, score_30d.signal]

        # اکثریت
        signal_counts = {}
        for sig in signals:
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

        # پیدا کردن رایج‌ترین
        most_common = max(signal_counts, key=signal_counts.get)

        # اگر توافق بالا
        if signal_counts[most_common] >= 2:
            return most_common

        # در غیر اینصورت NEUTRAL
        return "NEUTRAL"

    def _generate_overall_recommendation(
        self,
        overall_score: float,
        overall_signal: str,
        agreement: float,
        score_3d: SupportResistanceScore,
        score_7d: SupportResistanceScore,
        score_30d: SupportResistanceScore,
    ) -> str:
        """تولید توصیه کلی"""

        # توافق بالا
        if agreement > 0.7:
            if overall_signal == "AT_SUPPORT" or overall_signal == "NEAR_SUPPORT":
                if score_3d.bounce_probability > 0.6:
                    return "🟢 فرصت خرید قوی - همه افق‌ها نزدیک حمایت قوی"
                else:
                    return "🟡 نزدیک حمایت - منتظر تایید باشید"

            elif overall_signal == "AT_RESISTANCE" or overall_signal == "NEAR_RESISTANCE":
                if score_3d.bounce_probability > 0.6:
                    return "🔴 فرصت فروش قوی - همه افق‌ها نزدیک مقاومت قوی"
                else:
                    return "🟡 نزدیک مقاومت - منتظر تایید باشید"

        # توافق متوسط
        elif agreement > 0.4:
            return "⚪ سیگنال‌های مختلط - احتیاط کنید و منتظر شفاف‌تر شدن وضعیت باشید"

        # بدون توافق
        else:
            return "⚫ تعارض بین افق‌ها - از معامله خودداری کنید"

        return "⚪ خنثی - منتظر شرایط بهتر"

    def _load_weights(self, weights_path: str | None) -> dict[str, dict[str, float]]:
        """بارگذاری وزن‌های ML"""
        if weights_path and Path(weights_path).exists():
            with open(weights_path) as f:
                return json.load(f)

        # وزن‌های پیش‌فرض
        return {
            "3d": {
                "nearest_resistance_dist": -0.3,
                "resistance_strength": -0.2,
                "nearest_support_dist": 0.3,
                "support_strength": 0.2,
                "sr_position": -0.4,  # negative: نزدیک به 1 (resistance) = سیگنال فروش
                "sr_bias": 0.3,
                "level_density": 0.15,
                "fib_signal": 0.2,
                "camarilla_signal": 0.15,
            },
            "7d": {
                "nearest_resistance_dist": -0.25,
                "resistance_strength": -0.2,
                "nearest_support_dist": 0.25,
                "support_strength": 0.2,
                "sr_position": -0.35,
                "sr_bias": 0.25,
                "level_density": 0.15,
                "fib_signal": 0.2,
                "camarilla_signal": 0.15,
            },
            "30d": {
                "nearest_resistance_dist": -0.2,
                "resistance_strength": -0.2,
                "nearest_support_dist": 0.2,
                "support_strength": 0.2,
                "sr_position": -0.3,
                "sr_bias": 0.2,
                "level_density": 0.15,
                "fib_signal": 0.2,
                "camarilla_signal": 0.15,
            },
        }

    def _load_model_state(self, model_path: str | None) -> dict[str, dict[str, list[float]]]:
        """Load pickled regression bundles if available."""
        if model_path and Path(model_path).exists():
            with open(model_path, "rb") as fh:
                return pickle.load(fh)
        return {}


# ═══════════════════════════════════════════════════════════════
# مثال استفاده
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from utils.sample_data import generate_sample_candles

    print("=" * 70)
    print("Multi-Horizon Support/Resistance Analysis")
    print("=" * 70)

    # تولید داده نمونه
    candles = generate_sample_candles(
        count=800, base_price=50000, volatility=0.02, trend="sideways"
    )

    # ایجاد analyzer
    analyzer = MultiHorizonSupportResistanceAnalyzer()

    # تحلیل
    print("\n🔍 در حال تحلیل Support & Resistance...")
    analysis = analyzer.analyze(candles)

    current_price = candles[-1].close
    print(f"\nقیمت فعلی: ${current_price:,.2f}")

    # نمایش نتایج هر افق
    print("\n" + "=" * 70)
    print("تحلیل به تفکیک افق‌های زمانی:")
    print("=" * 70)

    for score in [analysis.score_3d, analysis.score_7d, analysis.score_30d]:
        print(f"\n📊 {score.horizon}:")
        print(f"   اسکور: {score.score:+.2f} (Confidence: {score.confidence:.1%})")
        print(f"   موقعیت: {score.get_position_label()} ({score.sr_position:.1%})")
        print(f"   سیگنال: {score.signal}")
        print(f"   احتمال Bounce: {score.bounce_probability:.1%}")
        print(f"   احتمال Breakout: {score.breakout_probability:.1%}")
        print(
            f"   نزدیک‌ترین حمایت: ${score.nearest_support:,.2f} ({((score.nearest_support - current_price) / current_price * 100):+.2f}%)"
        )
        print(
            f"   نزدیک‌ترین مقاومت: ${score.nearest_resistance:,.2f} ({((score.nearest_resistance - current_price) / current_price * 100):+.2f}%)"
        )
        print(f"   قدرت حمایت: {score.support_strength:.1%}")
        print(f"   قدرت مقاومت: {score.resistance_strength:.1%}")
        print(f"   📝 {score.recommendation}")

    # نمایش تحلیل کلی
    print("\n" + "=" * 70)
    print("تحلیل کلی:")
    print("=" * 70)
    print(f"\nاسکور کلی: {analysis.overall_sr_score:+.2f}")
    print(f"Confidence کلی: {analysis.overall_confidence:.1%}")
    print(f"توافق بین افق‌ها: {analysis.horizons_agreement:.1%}")
    print(f"Alignment: {analysis.alignment}")
    print(f"\nحمایت بحرانی: ${analysis.critical_support:,.2f}")
    print(f"مقاومت بحرانی: ${analysis.critical_resistance:,.2f}")
    print(f"\nسیگنال کلی: {analysis.overall_signal}")
    print("\n💡 توصیه نهایی:")
    print(f"   {analysis.overall_recommendation}")

    print("\n" + "=" * 70)
    print("✅ تحلیل کامل شد!")
    print("=" * 70)
