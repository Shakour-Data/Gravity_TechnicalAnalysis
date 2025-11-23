"""
Multi-Horizon Feature Extraction for Cycle

استخراج ویژگی‌های سیکل برای چندین افق زمانی:
- 7 اندیکاتور سیکل (DPO, Ehler's, Dominant, STC, Phase, Hilbert, Market Model)
- Multi-Horizon Learning: 3d, 7d, 30d
- Phase detection [0-360°]
- Cycle period estimation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime

from gravity_tech.models.schemas import Candle
from gravity_tech.indicators.cycle import CycleIndicators


class MultiHorizonCycleFeatureExtractor:
    """
    استخراج ویژگی‌های سیکل برای چندین افق زمانی
    """
    
    HORIZONS = [3, 7, 30]
    
    # اندیکاتورهای سیکل
    CYCLE_INDICATORS = [
        'dpo',
        'ehlers_cycle',
        'dominant_cycle',
        'schaff_trend_cycle',
        'phase_accumulation',
        'hilbert_transform',
        'market_cycle_model'
    ]
    
    def __init__(
        self,
        lookback_period: int = 100,
        horizons: List = None
    ):
        """
        Initialize feature extractor
        
        Args:
            lookback_period: تعداد کندل‌های گذشته
            horizons: لیست افق‌های زمانی (پیش‌فرض: [3, 7, 30])
                     می‌تواند int باشد [3, 7, 30] یا string ['3d', '7d', '30d']
        """
        self.lookback_period = lookback_period
        
        # تبدیل horizons به int اگر string است
        if horizons is None:
            self.horizons = self.HORIZONS
        else:
            self.horizons = []
            for h in horizons:
                if isinstance(h, str):
                    self.horizons.append(int(h.replace('d', '')))
                else:
                    self.horizons.append(int(h))
        
        self.max_horizon = max(self.horizons)
    
    def extract_cycle_features(
        self,
        candles: List[Candle]
    ) -> Dict[str, float]:
        """
        استخراج ویژگی‌های سیکل
        
        Returns:
            ویژگی‌ها شامل:
            - {indicator}_signal: سیگنال نرمال شده [-1, 1]
            - {indicator}_confidence: دقت [0, 1]
            - {indicator}_weighted: signal × confidence
            - {indicator}_normalized: مقدار نرمال شده [-1, 1]
            - {indicator}_phase: فاز سیکل [0, 360]
            - {indicator}_cycle_period: طول سیکل (تعداد کندل)
        """
        if len(candles) < self.lookback_period:
            return self._get_empty_features()
        
        features = {}
        
        # محاسبه همه اندیکاتورهای سیکل
        cycle_results = CycleIndicators.calculate_all(candles[-self.lookback_period:])
        
        # استخراج ویژگی از هر اندیکاتور
        for indicator_key, result in cycle_results.items():
            # Signal (normalized to [-1, 1])
            signal_value = self._signal_to_numeric(result.signal)
            features[f"{indicator_key}_signal"] = signal_value
            
            # Confidence [0, 1]
            features[f"{indicator_key}_confidence"] = result.confidence
            
            # Weighted signal
            features[f"{indicator_key}_weighted"] = signal_value * result.confidence
            
            # Normalized value [-1, 1]
            features[f"{indicator_key}_normalized"] = result.normalized
            
            # Phase [0, 360]
            features[f"{indicator_key}_phase"] = result.phase
            
            # Cycle period
            features[f"{indicator_key}_cycle_period"] = float(result.cycle_period)
        
        # ویژگی‌های ترکیبی
        features.update(self._extract_combined_features(cycle_results))
        
        # ویژگی‌های فاز
        features.update(self._extract_phase_features(cycle_results))
        
        # ویژگی‌های دوره سیکل
        features.update(self._extract_period_features(cycle_results))
        
        return features
    
    def _signal_to_numeric(self, signal) -> float:
        """تبدیل SignalStrength به مقدار عددی"""
        signal_map = {
            'VERY_BULLISH': 1.0,
            'BULLISH': 0.5,
            'NEUTRAL': 0.0,
            'BEARISH': -0.5,
            'VERY_BEARISH': -1.0
        }
        signal_str = str(signal).split('.')[-1] if hasattr(signal, 'name') else str(signal)
        return signal_map.get(signal_str, 0.0)
    
    def _extract_combined_features(self, cycle_results: Dict) -> Dict[str, float]:
        """ویژگی‌های ترکیبی از چندین اندیکاتور"""
        features = {}
        
        # میانگین سیگنال همه اندیکاتورها
        signals = [self._signal_to_numeric(r.signal) for r in cycle_results.values()]
        features['cycle_avg_signal'] = np.mean(signals)
        features['cycle_signal_std'] = np.std(signals)
        
        # میانگین confidence
        confidences = [r.confidence for r in cycle_results.values()]
        features['cycle_avg_confidence'] = np.mean(confidences)
        features['cycle_confidence_std'] = np.std(confidences)
        
        # Weighted average signal
        weighted_signals = [
            self._signal_to_numeric(r.signal) * r.confidence 
            for r in cycle_results.values()
        ]
        total_confidence = sum(confidences)
        if total_confidence > 0:
            features['cycle_weighted_signal'] = sum(weighted_signals) / total_confidence
        else:
            features['cycle_weighted_signal'] = 0.0
        
        # Agreement (چند درصد اندیکاتورها هماهنگ هستند)
        bullish = sum(1 for s in signals if s > 0.2)
        bearish = sum(1 for s in signals if s < -0.2)
        total = len(signals)
        features['cycle_agreement'] = max(bullish, bearish) / total if total > 0 else 0.0
        
        return features
    
    def _extract_phase_features(self, cycle_results: Dict) -> Dict[str, float]:
        """ویژگی‌های مربوط به فاز سیکل"""
        features = {}
        
        phases = [r.phase for r in cycle_results.values()]
        
        # میانگین فاز
        # از sin/cos استفاده می‌کنیم چون فاز دورانی است (0=360)
        phase_rads = np.radians(phases)
        avg_sin = np.mean(np.sin(phase_rads))
        avg_cos = np.mean(np.cos(phase_rads))
        avg_phase = np.degrees(np.arctan2(avg_sin, avg_cos))
        if avg_phase < 0:
            avg_phase += 360
        
        features['cycle_avg_phase'] = avg_phase
        
        # Phase quadrant distribution
        # 0-90 (Accumulation), 90-180 (Markup), 180-270 (Distribution), 270-360 (Markdown)
        q1 = sum(1 for p in phases if 0 <= p < 90) / len(phases)
        q2 = sum(1 for p in phases if 90 <= p < 180) / len(phases)
        q3 = sum(1 for p in phases if 180 <= p < 270) / len(phases)
        q4 = sum(1 for p in phases if 270 <= p < 360) / len(phases)
        
        features['cycle_phase_q1_accumulation'] = q1
        features['cycle_phase_q2_markup'] = q2
        features['cycle_phase_q3_distribution'] = q3
        features['cycle_phase_q4_markdown'] = q4
        
        # Dominant quadrant
        quadrants = [q1, q2, q3, q4]
        dominant_quadrant = quadrants.index(max(quadrants)) + 1
        features['cycle_dominant_quadrant'] = float(dominant_quadrant)
        
        # Phase dispersion (اختلاف فازها - نشان‌دهنده توافق)
        phase_dispersion = np.std(np.sin(phase_rads))**2 + np.std(np.cos(phase_rads))**2
        features['cycle_phase_dispersion'] = phase_dispersion
        
        return features
    
    def _extract_period_features(self, cycle_results: Dict) -> Dict[str, float]:
        """ویژگی‌های مربوط به دوره سیکل"""
        features = {}
        
        periods = [r.cycle_period for r in cycle_results.values()]
        
        # میانگین دوره سیکل
        features['cycle_avg_period'] = np.mean(periods)
        features['cycle_period_std'] = np.std(periods)
        features['cycle_min_period'] = float(min(periods))
        features['cycle_max_period'] = float(max(periods))
        
        # Period consistency (یکنواختی دوره‌ها)
        if features['cycle_avg_period'] > 0:
            features['cycle_period_cv'] = features['cycle_period_std'] / features['cycle_avg_period']
        else:
            features['cycle_period_cv'] = 0.0
        
        # دسته‌بندی دوره
        # Fast: < 15, Normal: 15-30, Slow: > 30
        fast_count = sum(1 for p in periods if p < 15)
        normal_count = sum(1 for p in periods if 15 <= p <= 30)
        slow_count = sum(1 for p in periods if p > 30)
        total = len(periods)
        
        features['cycle_fast_ratio'] = fast_count / total
        features['cycle_normal_ratio'] = normal_count / total
        features['cycle_slow_ratio'] = slow_count / total
        
        return features
    
    def extract_horizon_features(
        self,
        candles: List[Candle],
        horizon: int
    ) -> Dict[str, float]:
        """
        استخراج ویژگی برای یک افق زمانی خاص
        
        Args:
            candles: لیست کندل‌ها
            horizon: افق زمانی (3, 7, 30)
        
        Returns:
            ویژگی‌ها برای افق مشخص شده
        """
        if len(candles) < self.lookback_period + horizon:
            return self._get_empty_features()
        
        # ویژگی‌های فعلی
        current_features = self.extract_cycle_features(candles)
        
        # برچسب برای آینده (target)
        future_candles = candles[-(self.lookback_period - horizon):]
        future_features = self.extract_cycle_features(future_candles)
        
        # Feature suffix
        suffix = f"_{horizon}d"
        
        features = {}
        for key, value in current_features.items():
            features[f"{key}{suffix}"] = value
        
        # هدف پیش‌بینی: سیگنال و فاز آینده
        features[f"target_signal{suffix}"] = future_features.get('cycle_weighted_signal', 0.0)
        features[f"target_phase{suffix}"] = future_features.get('cycle_avg_phase', 0.0)
        
        return features
    
    def extract_all_horizons(
        self,
        candles: List[Candle]
    ) -> Dict[str, float]:
        """
        استخراج ویژگی برای همه افق‌های زمانی
        
        Returns:
            ویژگی‌های ترکیب شده از همه افق‌ها
        """
        if len(candles) < self.lookback_period + self.max_horizon:
            return {}
        
        all_features = {}
        
        for horizon in self.horizons:
            horizon_features = self.extract_horizon_features(candles, horizon)
            all_features.update(horizon_features)
        
        return all_features
    
    def _get_empty_features(self) -> Dict[str, float]:
        """ویژگی‌های خالی در صورت عدم وجود داده کافی"""
        features = {}
        
        # ویژگی‌های پایه برای هر اندیکاتور
        for indicator in self.CYCLE_INDICATORS:
            features[f"{indicator}_signal"] = 0.0
            features[f"{indicator}_confidence"] = 0.0
            features[f"{indicator}_weighted"] = 0.0
            features[f"{indicator}_normalized"] = 0.0
            features[f"{indicator}_phase"] = 0.0
            features[f"{indicator}_cycle_period"] = 20.0  # دوره پیش‌فرض
        
        # ویژگی‌های ترکیبی
        features['cycle_avg_signal'] = 0.0
        features['cycle_signal_std'] = 0.0
        features['cycle_avg_confidence'] = 0.0
        features['cycle_confidence_std'] = 0.0
        features['cycle_weighted_signal'] = 0.0
        features['cycle_agreement'] = 0.0
        
        # ویژگی‌های فاز
        features['cycle_avg_phase'] = 0.0
        features['cycle_phase_q1_accumulation'] = 0.25
        features['cycle_phase_q2_markup'] = 0.25
        features['cycle_phase_q3_distribution'] = 0.25
        features['cycle_phase_q4_markdown'] = 0.25
        features['cycle_dominant_quadrant'] = 1.0
        features['cycle_phase_dispersion'] = 0.0
        
        # ویژگی‌های دوره
        features['cycle_avg_period'] = 20.0
        features['cycle_period_std'] = 0.0
        features['cycle_min_period'] = 20.0
        features['cycle_max_period'] = 20.0
        features['cycle_period_cv'] = 0.0
        features['cycle_fast_ratio'] = 0.0
        features['cycle_normal_ratio'] = 1.0
        features['cycle_slow_ratio'] = 0.0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """لیست نام همه ویژگی‌ها"""
        dummy_features = self._get_empty_features()
        return list(dummy_features.keys())
    
    def get_feature_count(self) -> int:
        """تعداد کل ویژگی‌ها"""
        return len(self.get_feature_names())


# ═══════════════════════════════════════════════════════════════
# مثال استفاده
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from utils.sample_data import generate_sample_candles
    
    print("=" * 70)
    print("Multi-Horizon Cycle Feature Extraction")
    print("=" * 70)
    
    # تولید داده نمونه
    candles = generate_sample_candles(
        count=200,
        base_price=50000,
        volatility=0.02,
        trend='sideways'  # سیکل بهتر در رنج دیده می‌شود
    )
    
    # ایجاد extractor
    extractor = MultiHorizonCycleFeatureExtractor(
        lookback_period=100,
        horizons=[3, 7, 30]
    )
    
    # استخراج ویژگی‌های فعلی
    print("\n1. استخراج ویژگی‌های فعلی:")
    features = extractor.extract_cycle_features(candles)
    
    print(f"\nتعداد ویژگی‌ها: {len(features)}")
    print("\nنمونه ویژگی‌ها:")
    for key in list(features.keys())[:15]:
        print(f"  {key}: {features[key]:.4f}")
    
    # استخراج برای افق 7 روزه
    print("\n2. استخراج برای افق 7 روزه:")
    horizon_features = extractor.extract_horizon_features(candles, 7)
    print(f"تعداد ویژگی‌های افق 7d: {len(horizon_features)}")
    
    # Target values
    target_signal = horizon_features.get('target_signal_7d', 0)
    target_phase = horizon_features.get('target_phase_7d', 0)
    print(f"\nTarget Signal (7d): {target_signal:.4f}")
    print(f"Target Phase (7d): {target_phase:.2f}°")
    
    # استخراج برای همه افق‌ها
    print("\n3. استخراج برای همه افق‌ها:")
    all_features = extractor.extract_all_horizons(candles)
    print(f"تعداد کل ویژگی‌ها: {len(all_features)}")
    
    # نمایش ویژگی‌های کلیدی
    print("\n4. ویژگی‌های کلیدی سیکل:")
    key_features = [
        'cycle_avg_signal',
        'cycle_avg_confidence',
        'cycle_weighted_signal',
        'cycle_agreement',
        'cycle_avg_phase',
        'cycle_dominant_quadrant',
        'cycle_avg_period',
        'cycle_fast_ratio',
        'cycle_normal_ratio',
        'cycle_slow_ratio'
    ]
    
    for key in key_features:
        if key in features:
            value = features[key]
            if 'phase' in key:
                print(f"  {key}: {value:.2f}°")
            elif 'period' in key:
                print(f"  {key}: {value:.1f} candles")
            else:
                print(f"  {key}: {value:.4f}")
    
    # Phase quadrant analysis
    print("\n5. توزیع فاز سیکل:")
    print(f"  Q1 (Accumulation 0-90°): {features['cycle_phase_q1_accumulation']:.2%}")
    print(f"  Q2 (Markup 90-180°): {features['cycle_phase_q2_markup']:.2%}")
    print(f"  Q3 (Distribution 180-270°): {features['cycle_phase_q3_distribution']:.2%}")
    print(f"  Q4 (Markdown 270-360°): {features['cycle_phase_q4_markdown']:.2%}")
    
    dominant_q = int(features['cycle_dominant_quadrant'])
    phases = ['Accumulation', 'Markup', 'Distribution', 'Markdown']
    print(f"\n  → فاز غالب: Q{dominant_q} ({phases[dominant_q-1]})")
    
    print("\n" + "=" * 70)
    print("✅ Feature extraction completed successfully!")
    print("=" * 70)
