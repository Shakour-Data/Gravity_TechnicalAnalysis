"""
Multi-Horizon Feature Extraction for Momentum

استخراج ویژگی‌های مومنتوم برای چندین افق زمانی:
- 8 اندیکاتور مومنتوم (RSI, Stochastic, CCI, Williams %R, ROC, Momentum, OBV, CMF)
- تشخیص واگرایی برای هر اندیکاتور
- Multi-Horizon Learning: 3d, 7d, 30d
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime

from gravity_tech.models.schemas import Candle
from gravity_tech.indicators.momentum import MomentumIndicators
from gravity_tech.patterns.divergence import DivergenceDetector


class MultiHorizonMomentumFeatureExtractor:
    """
    استخراج ویژگی‌های مومنتوم برای چندین افق زمانی
    """
    
    HORIZONS = [3, 7, 30]
    
    # اندیکاتورهای مومنتوم
    MOMENTUM_INDICATORS = [
        'rsi',
        'stochastic',
        'cci',
        'williams_r',
        'roc',
        'momentum',
        'obv',
        'cmf'
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
        
        # Divergence detector
        self.divergence_detector = DivergenceDetector(lookback=20)
    
    def extract_momentum_features(
        self,
        candles: List[Candle]
    ) -> Dict[str, float]:
        """
        استخراج ویژگی‌های مومنتوم
        
        Returns:
            ویژگی‌ها شامل:
            - {indicator}_signal: سیگنال نرمال شده [-1, 1]
            - {indicator}_confidence: دقت [0, 1]
            - {indicator}_weighted: signal × confidence
            - {indicator}_divergence: امتیاز واگرایی [-2, 2]
        """
        if len(candles) < self.lookback_period:
            raise ValueError(f"Need at least {self.lookback_period} candles")
        
        features = {}
        
        # ═══════════════════════════════════════════════════════
        # 1. RSI
        # ═══════════════════════════════════════════════════════
        try:
            rsi_result = MomentumIndicators.rsi(candles, period=14)
            rsi_values = [self._calculate_rsi_value(candles[max(0, i-14):i+1]) 
                         for i in range(len(candles))]
            rsi_divergence = self.divergence_detector.detect(candles, rsi_values, "RSI")
            
            features['rsi_signal'] = rsi_result.signal.get_score() / 2.0  # نرمال به [-1, 1]
            features['rsi_confidence'] = rsi_result.confidence
            features['rsi_weighted'] = features['rsi_signal'] * features['rsi_confidence']
            features['rsi_divergence'] = rsi_divergence.get_signal_score() / 2.0  # نرمال به [-1, 1]
        except Exception as e:
            print(f"Warning: RSI calculation error: {e}")
            features.update({'rsi_signal': 0.0, 'rsi_confidence': 0.5, 
                           'rsi_weighted': 0.0, 'rsi_divergence': 0.0})
        
        # ═══════════════════════════════════════════════════════
        # 2. Stochastic
        # ═══════════════════════════════════════════════════════
        try:
            stoch_result = MomentumIndicators.stochastic(candles, k_period=14)
            stoch_values = [self._calculate_stoch_value(candles[max(0, i-14):i+1]) 
                           for i in range(len(candles))]
            stoch_divergence = self.divergence_detector.detect(candles, stoch_values, "Stochastic")
            
            features['stochastic_signal'] = stoch_result.signal.get_score() / 2.0
            features['stochastic_confidence'] = stoch_result.confidence
            features['stochastic_weighted'] = features['stochastic_signal'] * features['stochastic_confidence']
            features['stochastic_divergence'] = stoch_divergence.get_signal_score() / 2.0
        except Exception as e:
            print(f"Warning: Stochastic calculation error: {e}")
            features.update({'stochastic_signal': 0.0, 'stochastic_confidence': 0.5,
                           'stochastic_weighted': 0.0, 'stochastic_divergence': 0.0})
        
        # ═══════════════════════════════════════════════════════
        # 3. CCI
        # ═══════════════════════════════════════════════════════
        try:
            cci_result = MomentumIndicators.cci(candles, period=20)
            cci_values = [self._calculate_cci_value(candles[max(0, i-20):i+1]) 
                         for i in range(len(candles))]
            cci_divergence = self.divergence_detector.detect(candles, cci_values, "CCI")
            
            features['cci_signal'] = cci_result.signal.get_score() / 2.0
            features['cci_confidence'] = cci_result.confidence
            features['cci_weighted'] = features['cci_signal'] * features['cci_confidence']
            features['cci_divergence'] = cci_divergence.get_signal_score() / 2.0
        except Exception as e:
            print(f"Warning: CCI calculation error: {e}")
            features.update({'cci_signal': 0.0, 'cci_confidence': 0.5,
                           'cci_weighted': 0.0, 'cci_divergence': 0.0})
        
        # ═══════════════════════════════════════════════════════
        # 4. Williams %R
        # ═══════════════════════════════════════════════════════
        try:
            williams_result = MomentumIndicators.williams_r(candles, period=14)
            williams_values = [self._calculate_williams_value(candles[max(0, i-14):i+1]) 
                              for i in range(len(candles))]
            williams_divergence = self.divergence_detector.detect(candles, williams_values, "Williams%R")
            
            features['williams_r_signal'] = williams_result.signal.get_score() / 2.0
            features['williams_r_confidence'] = williams_result.confidence
            features['williams_r_weighted'] = features['williams_r_signal'] * features['williams_r_confidence']
            features['williams_r_divergence'] = williams_divergence.get_signal_score() / 2.0
        except Exception as e:
            print(f"Warning: Williams %R calculation error: {e}")
            features.update({'williams_r_signal': 0.0, 'williams_r_confidence': 0.5,
                           'williams_r_weighted': 0.0, 'williams_r_divergence': 0.0})
        
        # ═══════════════════════════════════════════════════════
        # 5-8. ROC, Momentum, OBV, CMF (به صورت مشابه)
        # ═══════════════════════════════════════════════════════
        for indicator in ['roc', 'momentum', 'obv', 'cmf']:
            try:
                if indicator == 'roc':
                    result = MomentumIndicators.roc(candles, period=12)
                elif indicator == 'momentum':
                    result = MomentumIndicators.momentum(candles, period=10)
                elif indicator == 'obv':
                    result = MomentumIndicators.obv(candles)
                else:  # cmf
                    result = MomentumIndicators.cmf(candles, period=20)
                
                features[f'{indicator}_signal'] = result.signal.get_score() / 2.0
                features[f'{indicator}_confidence'] = result.confidence
                features[f'{indicator}_weighted'] = features[f'{indicator}_signal'] * features[f'{indicator}_confidence']
                # واگرایی فقط برای اندیکاتورهای اصلی
                features[f'{indicator}_divergence'] = 0.0
                
            except Exception as e:
                print(f"Warning: {indicator.upper()} calculation error: {e}")
                features.update({
                    f'{indicator}_signal': 0.0,
                    f'{indicator}_confidence': 0.5,
                    f'{indicator}_weighted': 0.0,
                    f'{indicator}_divergence': 0.0
                })
        
        return features
    
    def _calculate_rsi_value(self, candles: List[Candle]) -> float:
        """محاسبه مقدار RSI برای یک نقطه"""
        if len(candles) < 2:
            return 50.0
        result = MomentumIndicators.rsi(candles, period=min(14, len(candles)-1))
        return result.value
    
    def _calculate_stoch_value(self, candles: List[Candle]) -> float:
        """محاسبه مقدار Stochastic برای یک نقطه"""
        if len(candles) < 2:
            return 50.0
        result = MomentumIndicators.stochastic(candles, k_period=min(14, len(candles)-1))
        return result.value
    
    def _calculate_cci_value(self, candles: List[Candle]) -> float:
        """محاسبه مقدار CCI برای یک نقطه"""
        if len(candles) < 2:
            return 0.0
        result = MomentumIndicators.cci(candles, period=min(20, len(candles)-1))
        return result.value
    
    def _calculate_williams_value(self, candles: List[Candle]) -> float:
        """محاسبه مقدار Williams %R برای یک نقطه"""
        if len(candles) < 2:
            return -50.0
        result = MomentumIndicators.williams_r(candles, period=min(14, len(candles)-1))
        return result.value
    
    def calculate_multi_horizon_returns(
        self,
        candles: List[Candle],
        current_idx: int
    ) -> Dict[str, float]:
        """
        محاسبه بازدهی برای چندین افق زمانی
        
        Returns:
            {'return_3d': 0.025, 'return_7d': 0.058, 'return_30d': 0.152}
        """
        current_price = candles[current_idx].close
        returns = {}
        
        for horizon in self.horizons:
            future_idx = current_idx + horizon
            
            if future_idx < len(candles):
                future_price = candles[future_idx].close
                return_pct = (future_price - current_price) / current_price
                returns[f'return_{horizon}d'] = return_pct
            else:
                returns[f'return_{horizon}d'] = None
        
        return returns
    
    def extract_training_dataset(
        self,
        candles: List[Candle]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        استخراج دیتاست آموزشی با چندین هدف
        
        Returns:
            X: DataFrame ویژگی‌ها
            Y: DataFrame اهداف (return_3d, return_7d, return_30d)
        """
        print(f"\n📊 Extracting multi-horizon momentum features...")
        print(f"   Horizons: {self.horizons} days")
        print(f"   Indicators: {len(self.MOMENTUM_INDICATORS)}")
        
        all_features = []
        all_targets = []
        
        for i in range(len(candles) - self.lookback_period - self.max_horizon):
            window = candles[i : i + self.lookback_period]
            
            try:
                # استخراج ویژگی‌ها
                features = self.extract_momentum_features(window)
                
                # محاسبه بازدهی‌های آینده
                targets = self.calculate_multi_horizon_returns(
                    candles,
                    i + self.lookback_period
                )
                
                # فقط نمونه‌های کامل
                if None not in targets.values():
                    all_features.append(features)
                    all_targets.append(targets)
                    
            except Exception as e:
                print(f"Warning: Error processing sample {i}: {e}")
                continue
        
        X = pd.DataFrame(all_features)
        Y = pd.DataFrame(all_targets)
        
        print(f"✅ Extracted {len(X)} complete training samples")
        print(f"   Features: {X.shape[1]} columns")
        print(f"   Targets: {Y.shape[1]} horizons")
        
        # نمایش آمار
        for horizon in self.horizons:
            col = f'return_{horizon}d'
            mean_return = Y[col].mean()
            std_return = Y[col].std()
            print(f"   {col}: mean={mean_return:.4f} ({mean_return*100:.2f}%), std={std_return:.4f}")
        
        return X, Y
    
    def create_summary_statistics(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame
    ) -> Dict:
        """آمار خلاصه"""
        return {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_horizons': Y.shape[1],
            'horizons': self.horizons,
            'feature_names': list(X.columns),
            'target_statistics': {
                f'return_{h}d': {
                    'mean': Y[f'return_{h}d'].mean(),
                    'std': Y[f'return_{h}d'].std(),
                    'min': Y[f'return_{h}d'].min(),
                    'max': Y[f'return_{h}d'].max(),
                    'positive_ratio': (Y[f'return_{h}d'] > 0).mean()
                }
                for h in self.horizons
            }
        }
