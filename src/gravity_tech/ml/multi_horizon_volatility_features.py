"""
Multi-Horizon Feature Extraction for Volatility

استخراج ویژگی‌های نوسان برای چندین افق زمانی:
- 8 اندیکاتور نوسان (ATR, Bollinger, Keltner, Donchian, StdDev, HV, ATR%, Chaikin)
- Multi-Horizon Learning: 3d, 7d, 30d
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime

from gravity_tech.models.schemas import Candle
from gravity_tech.indicators.volatility import VolatilityIndicators


class MultiHorizonVolatilityFeatureExtractor:
    """
    استخراج ویژگی‌های نوسان برای چندین افق زمانی
    """
    
    HORIZONS = [3, 7, 30]
    
    # اندیکاتورهای نوسان
    VOLATILITY_INDICATORS = [
        'atr',
        'bollinger_bands',
        'keltner_channel',
        'donchian_channel',
        'standard_deviation',
        'historical_volatility',
        'atr_percentage',
        'chaikin_volatility'
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
    
    def extract_volatility_features(
        self,
        candles: List[Candle]
    ) -> Dict[str, float]:
        """
        استخراج ویژگی‌های نوسان
        
        Returns:
            ویژگی‌ها شامل:
            - {indicator}_signal: سیگنال نرمال شده [-1, 1]
            - {indicator}_confidence: دقت [0, 1]
            - {indicator}_weighted: signal × confidence
            - {indicator}_normalized: مقدار نرمال شده [-1, 1]
            - {indicator}_percentile: صدک تاریخی [0, 100]
        """
        if len(candles) < self.lookback_period:
            raise ValueError(f"Need at least {self.lookback_period} candles")
        
        features = {}
        
        # ═══════════════════════════════════════════════════════
        # 1. ATR (Average True Range)
        # ═══════════════════════════════════════════════════════
        try:
            atr_result = VolatilityIndicators.atr(candles, period=14)
            
            features['atr_signal'] = atr_result.signal.get_score() / 2.0  # نرمال به [-1, 1]
            features['atr_confidence'] = atr_result.confidence
            features['atr_weighted'] = features['atr_signal'] * features['atr_confidence']
            features['atr_normalized'] = atr_result.normalized
            features['atr_percentile'] = atr_result.percentile
        except Exception as e:
            print(f"Warning: ATR calculation error: {e}")
            features.update({
                'atr_signal': 0.0, 'atr_confidence': 0.5,
                'atr_weighted': 0.0, 'atr_normalized': 0.0, 'atr_percentile': 50.0
            })
        
        # ═══════════════════════════════════════════════════════
        # 2. Bollinger Bands
        # ═══════════════════════════════════════════════════════
        try:
            bb_result = VolatilityIndicators.bollinger_bands(candles, period=20)
            
            features['bollinger_bands_signal'] = bb_result.signal.get_score() / 2.0
            features['bollinger_bands_confidence'] = bb_result.confidence
            features['bollinger_bands_weighted'] = features['bollinger_bands_signal'] * features['bollinger_bands_confidence']
            features['bollinger_bands_normalized'] = bb_result.normalized
            features['bollinger_bands_percentile'] = bb_result.percentile
        except Exception as e:
            print(f"Warning: Bollinger Bands calculation error: {e}")
            features.update({
                'bollinger_bands_signal': 0.0, 'bollinger_bands_confidence': 0.5,
                'bollinger_bands_weighted': 0.0, 'bollinger_bands_normalized': 0.0,
                'bollinger_bands_percentile': 50.0
            })
        
        # ═══════════════════════════════════════════════════════
        # 3. Keltner Channel
        # ═══════════════════════════════════════════════════════
        try:
            keltner_result = VolatilityIndicators.keltner_channel(candles, period=20)
            
            features['keltner_channel_signal'] = keltner_result.signal.get_score() / 2.0
            features['keltner_channel_confidence'] = keltner_result.confidence
            features['keltner_channel_weighted'] = features['keltner_channel_signal'] * features['keltner_channel_confidence']
            features['keltner_channel_normalized'] = keltner_result.normalized
            features['keltner_channel_percentile'] = keltner_result.percentile
        except Exception as e:
            print(f"Warning: Keltner Channel calculation error: {e}")
            features.update({
                'keltner_channel_signal': 0.0, 'keltner_channel_confidence': 0.5,
                'keltner_channel_weighted': 0.0, 'keltner_channel_normalized': 0.0,
                'keltner_channel_percentile': 50.0
            })
        
        # ═══════════════════════════════════════════════════════
        # 4. Donchian Channel
        # ═══════════════════════════════════════════════════════
        try:
            donchian_result = VolatilityIndicators.donchian_channel(candles, period=20)
            
            features['donchian_channel_signal'] = donchian_result.signal.get_score() / 2.0
            features['donchian_channel_confidence'] = donchian_result.confidence
            features['donchian_channel_weighted'] = features['donchian_channel_signal'] * features['donchian_channel_confidence']
            features['donchian_channel_normalized'] = donchian_result.normalized
            features['donchian_channel_percentile'] = donchian_result.percentile
        except Exception as e:
            print(f"Warning: Donchian Channel calculation error: {e}")
            features.update({
                'donchian_channel_signal': 0.0, 'donchian_channel_confidence': 0.5,
                'donchian_channel_weighted': 0.0, 'donchian_channel_normalized': 0.0,
                'donchian_channel_percentile': 50.0
            })
        
        # ═══════════════════════════════════════════════════════
        # 5. Standard Deviation
        # ═══════════════════════════════════════════════════════
        try:
            std_result = VolatilityIndicators.standard_deviation(candles, period=20)
            
            features['standard_deviation_signal'] = std_result.signal.get_score() / 2.0
            features['standard_deviation_confidence'] = std_result.confidence
            features['standard_deviation_weighted'] = features['standard_deviation_signal'] * features['standard_deviation_confidence']
            features['standard_deviation_normalized'] = std_result.normalized
            features['standard_deviation_percentile'] = std_result.percentile
        except Exception as e:
            print(f"Warning: Standard Deviation calculation error: {e}")
            features.update({
                'standard_deviation_signal': 0.0, 'standard_deviation_confidence': 0.5,
                'standard_deviation_weighted': 0.0, 'standard_deviation_normalized': 0.0,
                'standard_deviation_percentile': 50.0
            })
        
        # ═══════════════════════════════════════════════════════
        # 6. Historical Volatility
        # ═══════════════════════════════════════════════════════
        try:
            hv_result = VolatilityIndicators.historical_volatility(candles, period=20)
            
            features['historical_volatility_signal'] = hv_result.signal.get_score() / 2.0
            features['historical_volatility_confidence'] = hv_result.confidence
            features['historical_volatility_weighted'] = features['historical_volatility_signal'] * features['historical_volatility_confidence']
            features['historical_volatility_normalized'] = hv_result.normalized
            features['historical_volatility_percentile'] = hv_result.percentile
        except Exception as e:
            print(f"Warning: Historical Volatility calculation error: {e}")
            features.update({
                'historical_volatility_signal': 0.0, 'historical_volatility_confidence': 0.5,
                'historical_volatility_weighted': 0.0, 'historical_volatility_normalized': 0.0,
                'historical_volatility_percentile': 50.0
            })
        
        # ═══════════════════════════════════════════════════════
        # 7. ATR Percentage
        # ═══════════════════════════════════════════════════════
        try:
            atr_pct_result = VolatilityIndicators.atr_percentage(candles, period=14)
            
            features['atr_percentage_signal'] = atr_pct_result.signal.get_score() / 2.0
            features['atr_percentage_confidence'] = atr_pct_result.confidence
            features['atr_percentage_weighted'] = features['atr_percentage_signal'] * features['atr_percentage_confidence']
            features['atr_percentage_normalized'] = atr_pct_result.normalized
            features['atr_percentage_percentile'] = atr_pct_result.percentile
        except Exception as e:
            print(f"Warning: ATR Percentage calculation error: {e}")
            features.update({
                'atr_percentage_signal': 0.0, 'atr_percentage_confidence': 0.5,
                'atr_percentage_weighted': 0.0, 'atr_percentage_normalized': 0.0,
                'atr_percentage_percentile': 50.0
            })
        
        # ═══════════════════════════════════════════════════════
        # 8. Chaikin Volatility
        # ═══════════════════════════════════════════════════════
        try:
            chaikin_result = VolatilityIndicators.chaikin_volatility(candles, period=10)
            
            features['chaikin_volatility_signal'] = chaikin_result.signal.get_score() / 2.0
            features['chaikin_volatility_confidence'] = chaikin_result.confidence
            features['chaikin_volatility_weighted'] = features['chaikin_volatility_signal'] * features['chaikin_volatility_confidence']
            features['chaikin_volatility_normalized'] = chaikin_result.normalized
            features['chaikin_volatility_percentile'] = chaikin_result.percentile
        except Exception as e:
            print(f"Warning: Chaikin Volatility calculation error: {e}")
            features.update({
                'chaikin_volatility_signal': 0.0, 'chaikin_volatility_confidence': 0.5,
                'chaikin_volatility_weighted': 0.0, 'chaikin_volatility_normalized': 0.0,
                'chaikin_volatility_percentile': 50.0
            })
        
        return features
    
    def calculate_future_volatility_change(
        self,
        candles: List[Candle],
        horizon: int
    ) -> float:
        """
        محاسبه تغییر نوسان در آینده
        
        برای آموزش ML، باید بدانیم نوسان در آینده افزایش یا کاهش می‌یابد.
        
        از ATR استفاده می‌کنیم به عنوان معیار نوسان:
        - اگر ATR افزایش یابد → نوسان در حال افزایش (مثبت)
        - اگر ATR کاهش یابد → نوسان در حال کاهش (منفی)
        
        Args:
            candles: لیست کندل‌ها
            horizon: افق زمانی (چند روز آینده)
            
        Returns:
            تغییر نوسان نرمال شده [-1, +1]
        """
        if len(candles) < horizon + 20:
            return 0.0
        
        try:
            # محاسبه ATR فعلی
            current_candles = candles[:-horizon]
            tr_current = VolatilityIndicators.true_range(current_candles)
            atr_current = pd.Series(tr_current).ewm(span=14, adjust=False).mean().iloc[-1]
            
            # محاسبه ATR آینده
            future_candles = candles
            tr_future = VolatilityIndicators.true_range(future_candles)
            atr_future = pd.Series(tr_future).ewm(span=14, adjust=False).mean().iloc[-1]
            
            # درصد تغییر
            pct_change = ((atr_future - atr_current) / atr_current) * 100
            
            # نرمال سازی به [-1, +1]
            # +50% = +1, -50% = -1
            normalized_change = np.clip(pct_change / 50.0, -1.0, 1.0)
            
            return normalized_change
            
        except Exception as e:
            print(f"Warning: Future volatility calculation error: {e}")
            return 0.0
    
    def extract_features_with_target(
        self,
        candles: List[Candle],
        horizon: int
    ) -> Tuple[Dict[str, float], float]:
        """
        استخراج ویژگی‌ها به همراه target برای آموزش
        
        Args:
            candles: لیست کندل‌ها
            horizon: افق زمانی
            
        Returns:
            (features, target)
            - features: ویژگی‌های نوسان
            - target: تغییر نوسان در آینده [-1, +1]
        """
        # استخراج ویژگی‌ها از کندل‌های قبل از horizon
        training_candles = candles[:-horizon] if horizon > 0 else candles
        features = self.extract_volatility_features(training_candles)
        
        # محاسبه target
        target = self.calculate_future_volatility_change(candles, horizon)
        
        return features, target
    
    def create_training_dataset(
        self,
        candles: List[Candle],
        horizons: List[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        ایجاد دیتاست آموزشی برای چند افق زمانی
        
        Args:
            candles: لیست کندل‌ها
            horizons: لیست افق‌های زمانی (پیش‌فرض: [3, 7, 30])
            
        Returns:
            (X, y)
            - X: DataFrame ویژگی‌ها
            - y: DataFrame اهداف (یک ستون برای هر horizon)
        """
        if horizons is None:
            horizons = self.horizons
        
        X_list = []
        y_dict = {f'target_{h}d': [] for h in horizons}
        
        # حداکثر افق
        max_h = max(horizons)
        
        # برای هر نقطه زمانی
        for i in range(self.lookback_period, len(candles) - max_h):
            window_candles = candles[:i+1]
            
            try:
                # استخراج ویژگی‌ها
                features = self.extract_volatility_features(window_candles)
                X_list.append(features)
                
                # محاسبه target برای هر horizon
                for h in horizons:
                    future_candles = candles[:i+h+1]
                    target = self.calculate_future_volatility_change(future_candles, h)
                    y_dict[f'target_{h}d'].append(target)
                    
            except Exception as e:
                print(f"Warning: Error at index {i}: {e}")
                continue
        
        X = pd.DataFrame(X_list)
        y = pd.DataFrame(y_dict)
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """
        لیست نام ویژگی‌ها
        """
        feature_names = []
        for indicator in self.VOLATILITY_INDICATORS:
            feature_names.extend([
                f'{indicator}_signal',
                f'{indicator}_confidence',
                f'{indicator}_weighted',
                f'{indicator}_normalized',
                f'{indicator}_percentile'
            ])
        return feature_names
