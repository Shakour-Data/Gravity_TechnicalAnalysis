"""
Multi-Horizon Support/Resistance Feature Extraction

این ماژول ویژگی‌های Support & Resistance را برای 3 افق زمانی استخراج می‌کند:
- کوتاه‌مدت (3 روز)
- میان‌مدت (7 روز)
- بلندمدت (30 روز)

ویژگی‌ها شامل:
- تعداد سطوح حمایت/مقاومت
- قدرت سطوح
- فاصله از قیمت فعلی
- Clustering (توافق بین روش‌ها)
- Zone Width (پهنای ناحیه)
- Touch Count (تعداد تماس)
"""

from dataclasses import dataclass

import numpy as np
from gravity_tech.core.indicators.support_resistance import SupportResistanceIndicators
from gravity_tech.models.schemas import Candle


@dataclass
class SRFeatures:
    """ویژگی‌های استخراج شده از سطوح S/R"""
    # Pivot Points features
    pivot_distance: float  # فاصله تا pivot (%)
    pivot_signal_strength: float  # قدرت سیگنال pivot
    above_pivot: int  # 1 if above, 0 if below

    # Resistance features
    nearest_resistance_distance: float  # فاصله تا نزدیک‌ترین مقاومت (%)
    resistance_count_nearby: int  # تعداد مقاومت‌های نزدیک (±5%)
    resistance_strength_avg: float  # میانگین قدرت مقاومت‌ها

    # Support features
    nearest_support_distance: float  # فاصله تا نزدیک‌ترین حمایت (%)
    support_count_nearby: int  # تعداد حمایت‌های نزدیک
    support_strength_avg: float  # میانگین قدرت حمایت‌ها

    # Position features
    sr_range_position: float  # موقعیت در محدوده S/R [0, 1]
    distance_to_nearest_level: float  # فاصله کوچکترین (%)

    # Fibonacci features
    fib_nearest_level: str  # نزدیک‌ترین سطح فیبوناچی
    fib_distance: float  # فاصله تا سطح فیبوناچی (%)
    fib_signal_strength: float  # قدرت سیگنال فیبوناچی

    # Camarilla features
    camarilla_signal_strength: float  # قدرت سیگنال کاماریلا

    # Combined features
    overall_sr_bias: float  # گرایش کلی: +1 (نزدیک مقاومت) تا -1 (نزدیک حمایت)
    level_density: float  # تراکم سطوح در محدوده نزدیک


class MultiHorizonSupportResistanceFeatureExtractor:
    """استخراج ویژگی‌های Multi-Horizon برای Support & Resistance"""

    def __init__(self):
        """Initialize feature extractor"""
        self.sr_indicators = SupportResistanceIndicators()

    def extract_sr_features(
        self,
        candles: list[Candle]
    ) -> SRFeatures:
        """
        استخراج ویژگی‌های S/R از کندل‌ها

        Args:
            candles: لیست کندل‌ها

        Returns:
            SRFeatures: ویژگی‌های استخراج شده
        """
        if len(candles) < 10:
            return self._get_empty_features()

        current_price = candles[-1].close

        # محاسبه اندیکاتورها
        pivot_result = self.sr_indicators.pivot_points(candles)
        fib_result = self.sr_indicators.fibonacci_retracement(candles, lookback=50)
        camarilla_result = self.sr_indicators.camarilla_pivots(candles)
        sr_levels_result = self.sr_indicators.support_resistance_levels(candles, window=50)

        # === Pivot Features ===
        pivot_price = pivot_result.value
        pivot_distance = ((pivot_price - current_price) / current_price) * 100
        pivot_signal_strength = self._signal_to_numeric(pivot_result.signal)
        above_pivot = 1 if current_price > pivot_price else 0

        # === Resistance Features ===
        # از Pivot Points
        r1 = pivot_result.additional_values['r1']
        r2 = pivot_result.additional_values['r2']
        r3 = pivot_result.additional_values['r3']
        resistances_pivot = [r for r in [r1, r2, r3] if r > current_price]

        # از Camarilla
        cam_r1 = camarilla_result.additional_values['R1']
        cam_r2 = camarilla_result.additional_values['R2']
        cam_r3 = camarilla_result.additional_values['R3']
        cam_r4 = camarilla_result.additional_values['R4']
        resistances_camarilla = [r for r in [cam_r1, cam_r2, cam_r3, cam_r4] if r > current_price]

        # از Dynamic S/R
        dynamic_resistance = sr_levels_result.additional_values['resistance']
        if dynamic_resistance > current_price:
            resistances_dynamic = [dynamic_resistance]
        else:
            resistances_dynamic = []

        # ترکیب همه مقاومت‌ها
        all_resistances = resistances_pivot + resistances_camarilla + resistances_dynamic

        if all_resistances:
            nearest_resistance = min(all_resistances)
            nearest_resistance_distance = ((nearest_resistance - current_price) / current_price) * 100
            # شمارش مقاومت‌های نزدیک (±5%)
            resistance_count_nearby = sum(1 for r in all_resistances
                                         if abs((r - current_price) / current_price * 100) < 5.0)
            # میانگین قدرت (فرض: نزدیک‌تر = قوی‌تر)
            resistance_strength_avg = np.mean([1.0 / (1.0 + abs((r - current_price) / current_price))
                                              for r in all_resistances])
        else:
            nearest_resistance_distance = 10.0  # مقدار پیش‌فرض
            resistance_count_nearby = 0
            resistance_strength_avg = 0.0

        # === Support Features ===
        s1 = pivot_result.additional_values['s1']
        s2 = pivot_result.additional_values['s2']
        s3 = pivot_result.additional_values['s3']
        supports_pivot = [s for s in [s1, s2, s3] if s < current_price]

        cam_s1 = camarilla_result.additional_values['S1']
        cam_s2 = camarilla_result.additional_values['S2']
        cam_s3 = camarilla_result.additional_values['S3']
        cam_s4 = camarilla_result.additional_values['S4']
        supports_camarilla = [s for s in [cam_s1, cam_s2, cam_s3, cam_s4] if s < current_price]

        dynamic_support = sr_levels_result.additional_values['support']
        if dynamic_support < current_price:
            supports_dynamic = [dynamic_support]
        else:
            supports_dynamic = []

        all_supports = supports_pivot + supports_camarilla + supports_dynamic

        if all_supports:
            nearest_support = max(all_supports)
            nearest_support_distance = ((current_price - nearest_support) / current_price) * 100
            support_count_nearby = sum(1 for s in all_supports
                                      if abs((s - current_price) / current_price * 100) < 5.0)
            support_strength_avg = np.mean([1.0 / (1.0 + abs((s - current_price) / current_price))
                                           for s in all_supports])
        else:
            nearest_support_distance = 10.0
            support_count_nearby = 0
            support_strength_avg = 0.0

        # === Position Features ===
        # موقعیت در محدوده S/R
        if all_supports and all_resistances:
            sr_range = max(all_resistances) - min(all_supports)
            if sr_range > 0:
                sr_range_position = (current_price - min(all_supports)) / sr_range
            else:
                sr_range_position = 0.5
        else:
            sr_range_position = 0.5

        # کوچکترین فاصله
        distance_to_nearest_level = min(nearest_resistance_distance, nearest_support_distance)

        # === Fibonacci Features ===
        fib_nearest_level = fib_result.description.split()[-1] if 'فیبوناچی' in fib_result.description else "0.5"
        fib_value = fib_result.value
        fib_distance = abs((fib_value - current_price) / current_price) * 100
        fib_signal_strength = self._signal_to_numeric(fib_result.signal)

        # === Camarilla Features ===
        camarilla_signal_strength = self._signal_to_numeric(camarilla_result.signal)

        # === Combined Features ===
        # گرایش کلی: نزدیکی به مقاومت (+) یا حمایت (-)
        if nearest_resistance_distance < nearest_support_distance:
            overall_sr_bias = nearest_resistance_distance / (nearest_resistance_distance + nearest_support_distance)
        else:
            overall_sr_bias = -nearest_support_distance / (nearest_resistance_distance + nearest_support_distance)

        # تراکم سطوح
        total_levels_nearby = resistance_count_nearby + support_count_nearby
        level_density = total_levels_nearby / 10.0  # Normalize by max expected

        return SRFeatures(
            pivot_distance=pivot_distance,
            pivot_signal_strength=pivot_signal_strength,
            above_pivot=above_pivot,
            nearest_resistance_distance=nearest_resistance_distance,
            resistance_count_nearby=resistance_count_nearby,
            resistance_strength_avg=resistance_strength_avg,
            nearest_support_distance=nearest_support_distance,
            support_count_nearby=support_count_nearby,
            support_strength_avg=support_strength_avg,
            sr_range_position=sr_range_position,
            distance_to_nearest_level=distance_to_nearest_level,
            fib_nearest_level=fib_nearest_level,
            fib_distance=fib_distance,
            fib_signal_strength=fib_signal_strength,
            camarilla_signal_strength=camarilla_signal_strength,
            overall_sr_bias=overall_sr_bias,
            level_density=level_density
        )

    def extract_horizon_features(
        self,
        candles: list[Candle],
        horizon: str  # '3d', '7d', '30d'
    ) -> dict[str, float]:
        """
        استخراج ویژگی‌های یک افق زمانی خاص

        Args:
            candles: لیست کندل‌ها
            horizon: افق زمانی

        Returns:
            Dict: ویژگی‌ها به صورت dictionary
        """
        # تعداد کندل برای هر افق (فرض: 1h candles)
        horizon_periods = {
            '3d': 72,    # 3 days * 24 hours
            '7d': 168,   # 7 days * 24 hours
            '30d': 720   # 30 days * 24 hours
        }

        period = horizon_periods.get(horizon, 72)
        recent_candles = candles[-period:] if len(candles) > period else candles

        # استخراج ویژگی‌ها
        features = self.extract_sr_features(recent_candles)

        # تبدیل به dictionary با prefix افق
        return {
            f'{horizon}_pivot_distance': features.pivot_distance,
            f'{horizon}_pivot_signal': features.pivot_signal_strength,
            f'{horizon}_above_pivot': float(features.above_pivot),
            f'{horizon}_nearest_resistance_dist': features.nearest_resistance_distance,
            f'{horizon}_resistance_count': float(features.resistance_count_nearby),
            f'{horizon}_resistance_strength': features.resistance_strength_avg,
            f'{horizon}_nearest_support_dist': features.nearest_support_distance,
            f'{horizon}_support_count': float(features.support_count_nearby),
            f'{horizon}_support_strength': features.support_strength_avg,
            f'{horizon}_sr_position': features.sr_range_position,
            f'{horizon}_nearest_level_dist': features.distance_to_nearest_level,
            f'{horizon}_fib_distance': features.fib_distance,
            f'{horizon}_fib_signal': features.fib_signal_strength,
            f'{horizon}_camarilla_signal': features.camarilla_signal_strength,
            f'{horizon}_sr_bias': features.overall_sr_bias,
            f'{horizon}_level_density': features.level_density
        }

    def extract_all_horizons(
        self,
        candles: list[Candle]
    ) -> dict[str, float]:
        """
        استخراج ویژگی‌های همه افق‌های زمانی

        Args:
            candles: لیست کندل‌ها

        Returns:
            Dict: همه ویژگی‌ها
        """
        all_features = {}

        # ویژگی‌های هر افق
        for horizon in ['3d', '7d', '30d']:
            horizon_features = self.extract_horizon_features(candles, horizon)
            all_features.update(horizon_features)

        # ویژگی‌های ترکیبی بین افق‌ها
        combined = self._extract_cross_horizon_features(candles)
        all_features.update(combined)

        return all_features

    def _extract_cross_horizon_features(
        self,
        candles: list[Candle]
    ) -> dict[str, float]:
        """استخراج ویژگی‌های ترکیبی بین افق‌ها"""
        features_3d = self.extract_sr_features(candles[-72:] if len(candles) > 72 else candles)
        features_7d = self.extract_sr_features(candles[-168:] if len(candles) > 168 else candles)
        features_30d = self.extract_sr_features(candles[-720:] if len(candles) > 720 else candles)

        return {
            # توافق بین افق‌ها در نزدیکی به مقاومت
            'resistance_agreement': self._calculate_agreement([
                features_3d.nearest_resistance_distance,
                features_7d.nearest_resistance_distance,
                features_30d.nearest_resistance_distance
            ]),

            # توافق بین افق‌ها در نزدیکی به حمایت
            'support_agreement': self._calculate_agreement([
                features_3d.nearest_support_distance,
                features_7d.nearest_support_distance,
                features_30d.nearest_support_distance
            ]),

            # میانگین موقعیت در محدوده S/R
            'avg_sr_position': np.mean([
                features_3d.sr_range_position,
                features_7d.sr_range_position,
                features_30d.sr_range_position
            ]),

            # انحراف معیار موقعیت (ثبات)
            'sr_position_std': np.std([
                features_3d.sr_range_position,
                features_7d.sr_range_position,
                features_30d.sr_range_position
            ]),

            # میانگین تراکم سطوح
            'avg_level_density': np.mean([
                features_3d.level_density,
                features_7d.level_density,
                features_30d.level_density
            ]),

            # میانگین SR Bias
            'avg_sr_bias': np.mean([
                features_3d.overall_sr_bias,
                features_7d.overall_sr_bias,
                features_30d.overall_sr_bias
            ]),

            # توافق در Signal Strength
            'pivot_signal_agreement': self._calculate_agreement([
                features_3d.pivot_signal_strength,
                features_7d.pivot_signal_strength,
                features_30d.pivot_signal_strength
            ])
        }

    def _calculate_agreement(self, values: list[float]) -> float:
        """محاسبه توافق (1 - CV)"""
        if not values or all(v == 0 for v in values):
            return 0.0

        mean = np.mean(values)
        if mean == 0:
            return 0.0

        std = np.std(values)
        cv = std / abs(mean)

        # توافق = 1 - CV (محدود به [0, 1])
        agreement = max(0.0, min(1.0, 1.0 - cv))
        return agreement

    def _signal_to_numeric(self, signal) -> float:
        """تبدیل SignalStrength به عدد"""
        signal_map = {
            'VERY_BULLISH': 1.0,
            'BULLISH': 0.6,
            'BULLISH_BROKEN': 0.3,
            'NEUTRAL': 0.0,
            'BEARISH_BROKEN': -0.3,
            'BEARISH': -0.6,
            'VERY_BEARISH': -1.0
        }

        signal_str = str(signal).split('.')[-1] if '.' in str(signal) else str(signal)
        return signal_map.get(signal_str, 0.0)

    def _get_empty_features(self) -> SRFeatures:
        """ویژگی‌های خالی برای داده ناکافی"""
        return SRFeatures(
            pivot_distance=0.0,
            pivot_signal_strength=0.0,
            above_pivot=0,
            nearest_resistance_distance=5.0,
            resistance_count_nearby=0,
            resistance_strength_avg=0.0,
            nearest_support_distance=5.0,
            support_count_nearby=0,
            support_strength_avg=0.0,
            sr_range_position=0.5,
            distance_to_nearest_level=5.0,
            fib_nearest_level="0.5",
            fib_distance=2.0,
            fib_signal_strength=0.0,
            camarilla_signal_strength=0.0,
            overall_sr_bias=0.0,
            level_density=0.0
        )


# ═══════════════════════════════════════════════════════════════
# مثال استفاده
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from utils.sample_data import generate_sample_candles

    print("=" * 70)
    print("Multi-Horizon Support/Resistance Feature Extraction")
    print("=" * 70)

    # تولید داده نمونه
    candles = generate_sample_candles(
        count=800,  # حداقل 720 برای 30d
        base_price=50000,
        volatility=0.02,
        trend='sideways'
    )

    # ایجاد feature extractor
    extractor = MultiHorizonSupportResistanceFeatureExtractor()

    # استخراج ویژگی‌های یک افق
    print("\n📊 ویژگی‌های افق 7 روزه:")
    print("=" * 70)
    features_7d = extractor.extract_horizon_features(candles, '7d')

    for key, value in sorted(features_7d.items()):
        print(f"{key:40s}: {value:8.4f}")

    # استخراج همه افق‌ها
    print("\n\n📊 همه افق‌های زمانی:")
    print("=" * 70)
    all_features = extractor.extract_all_horizons(candles)

    print(f"\nتعداد کل ویژگی‌ها: {len(all_features)}")

    # ویژگی‌های ترکیبی
    print("\n🔗 ویژگی‌های ترکیبی:")
    print("=" * 70)
    combined_keys = [k for k in all_features.keys() if not k.startswith(('3d', '7d', '30d'))]
    for key in combined_keys:
        print(f"{key:40s}: {all_features[key]:8.4f}")

    # تحلیل Support vs Resistance
    print("\n\n📈 تحلیل Support vs Resistance:")
    print("=" * 70)

    for horizon in ['3d', '7d', '30d']:
        print(f"\n{horizon}:")
        res_dist = all_features[f'{horizon}_nearest_resistance_dist']
        sup_dist = all_features[f'{horizon}_nearest_support_dist']
        sr_pos = all_features[f'{horizon}_sr_position']
        sr_bias = all_features[f'{horizon}_sr_bias']

        print(f"  نزدیک‌ترین مقاومت: {res_dist:+.2f}%")
        print(f"  نزدیک‌ترین حمایت: {sup_dist:+.2f}%")
        print(f"  موقعیت در محدوده: {sr_pos*100:.1f}%")
        print(f"  SR Bias: {sr_bias:+.2f} ({'نزدیک مقاومت' if sr_bias > 0 else 'نزدیک حمایت'})")

    print("\n" + "=" * 70)
    print("✅ استخراج ویژگی‌ها کامل شد!")
    print("=" * 70)
