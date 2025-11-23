"""
Training Pipeline برای Multi-Horizon Cycle System

آموزش مدل سیکل برای سه افق مستقل
"""

import numpy as np
import pandas as pd
from typing import Optional, List
import os
from datetime import datetime

from gravity_tech.models.schemas import Candle
from gravity_tech.ml.multi_horizon_cycle_features import MultiHorizonCycleFeatureExtractor
from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner


def create_realistic_cycle_data(
    num_samples: int = 2000,
    cycle_regime: str = 'mixed'  # 'fast', 'slow', 'mixed', 'range'
) -> List[Candle]:
    """
    ایجاد داده‌های واقعی با سیکل‌های مختلف
    
    Args:
        num_samples: تعداد کندل‌ها
        cycle_regime: رژیم سیکل
            - 'fast': سیکل‌های سریع (8-15 کندل)
            - 'slow': سیکل‌های کند (30-50 کندل)
            - 'mixed': ترکیب سیکل‌های مختلف
            - 'range': رنج بدون روند (سیکل واضح)
    """
    np.random.seed(42)
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_samples, freq='1h')
    
    base_price = 50000
    candles = []
    
    for i in range(num_samples):
        # تعیین cycle period بر اساس regime
        if cycle_regime == 'fast':
            cycle_period = 12  # 12 کندل
            amplitude = base_price * 0.02  # 2%
        elif cycle_regime == 'slow':
            cycle_period = 40  # 40 کندل
            amplitude = base_price * 0.05  # 5%
        elif cycle_regime == 'range':
            cycle_period = 20
            amplitude = base_price * 0.03  # 3%
        else:  # mixed
            # تغییر دوره سیکل در طول زمان
            if i % 400 < 200:
                cycle_period = 15  # fast
                amplitude = base_price * 0.02
            else:
                cycle_period = 35  # slow
                amplitude = base_price * 0.04
        
        # محاسبه موقعیت در سیکل (phase)
        phase = (i % cycle_period) / cycle_period * 2 * np.pi
        
        # موج سینوسی برای سیکل
        cycle_component = amplitude * np.sin(phase)
        
        # trend (کم یا صفر برای range)
        if cycle_regime == 'range':
            trend_component = 0
        else:
            # trend خیلی ملایم
            trend_component = base_price * 0.0001 * i
        
        # noise
        noise = np.random.normal(0, base_price * 0.005)
        
        # قیمت پایانی
        close_price = base_price + trend_component + cycle_component + noise
        
        # قیمت باز شدن (از close قبلی)
        if i == 0:
            open_price = base_price
        else:
            open_price = candles[-1].close
        
        # High/Low با توجه به volatility داخل کندل
        intracandle_volatility = amplitude * 0.3
        high_change = abs(np.random.normal(0, intracandle_volatility))
        low_change = abs(np.random.normal(0, intracandle_volatility))
        
        high_price = max(open_price, close_price) + high_change
        low_price = min(open_price, close_price) - low_change
        
        # Volume (بیشتر در turning points)
        # Volume بالاتر در فازهای 0-90 و 180-270 (turning points)
        phase_deg = (phase * 180 / np.pi) % 360
        if 315 <= phase_deg or phase_deg < 45 or 135 <= phase_deg < 225:
            # نزدیک کف یا سقف
            volume_multiplier = 1.5
        else:
            volume_multiplier = 1.0
        
        base_volume = 1000000
        volume = base_volume * volume_multiplier * (1 + np.random.normal(0, 0.2))
        volume = max(volume, 100000)
        
        # ایجاد Candle
        candle = Candle(
            timestamp=dates[i],
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        )
        candles.append(candle)
    
    return candles


class MultiHorizonCycleTrainer:
    """
    آموزش‌دهنده مدل سیکل چند افقی
    """
    
    def __init__(
        self,
        lookback_period: int = 100,
        horizons: List = None
    ):
        """
        Initialize trainer
        
        Args:
            lookback_period: تعداد کندل‌های گذشته
            horizons: لیست افق‌های زمانی
        """
        self.lookback_period = lookback_period
        self.horizons = horizons or [3, 7, 30]
        self.feature_extractor = MultiHorizonCycleFeatureExtractor(
            lookback_period=lookback_period,
            horizons=self.horizons
        )
        self.weight_learner = MultiHorizonWeightLearner()
    
    def prepare_training_data(
        self,
        candles: List[Candle],
        horizon: int
    ) -> tuple:
        """
        آماده‌سازی داده برای آموزش یک افق
        
        Returns:
            (features_list, targets_list)
        """
        features_list = []
        targets_list = []
        
        max_idx = len(candles) - self.lookback_period - horizon
        
        for i in range(self.lookback_period, max_idx):
            # استخراج ویژگی‌ها
            candles_window = candles[:i]
            features = self.feature_extractor.extract_cycle_features(
                candles_window[-self.lookback_period:]
            )
            
            # محاسبه target: تغییر فاز در آینده
            current_phase = features.get('cycle_avg_phase', 0.0)
            
            # فاز آینده
            future_candles = candles[:i + horizon]
            future_features = self.feature_extractor.extract_cycle_features(
                future_candles[-self.lookback_period:]
            )
            future_phase = future_features.get('cycle_avg_phase', 0.0)
            
            # محاسبه تغییر فاز
            phase_change = future_phase - current_phase
            # Normalize به [-180, 180]
            while phase_change > 180:
                phase_change -= 360
            while phase_change < -180:
                phase_change += 360
            
            # Target: سیگنال بر اساس تغییر فاز
            # صعودی اگر فاز در حال پیشرفت به سمت Markup/Peak
            if 45 <= future_phase < 135:  # Markup phase
                target = 1.0  # صعودی
            elif 135 <= future_phase < 225:  # Distribution phase
                target = -0.5  # نزدیک سقف
            elif 225 <= future_phase < 315:  # Markdown phase
                target = -1.0  # نزولی
            else:  # 315-45: Accumulation phase
                target = 0.5  # نزدیک کف
            
            features_list.append(features)
            targets_list.append(target)
        
        return features_list, targets_list
    
    def train(
        self,
        train_candles: List[Candle],
        validation_candles: Optional[List[Candle]] = None,
        save_path: str = "models/cycle"
    ):
        """
        آموزش مدل برای همه افق‌ها
        
        Args:
            train_candles: داده‌های آموزش
            validation_candles: داده‌های اعتبارسنجی
            save_path: مسیر ذخیره مدل
        """
        print("=" * 70)
        print("Multi-Horizon Cycle Training")
        print("=" * 70)
        
        print(f"\nتعداد کندل‌های آموزش: {len(train_candles)}")
        if validation_candles:
            print(f"تعداد کندل‌های اعتبارسنجی: {len(validation_candles)}")
        
        # آموزش برای هر افق
        for horizon in self.horizons:
            print(f"\n{'='*70}")
            print(f"آموزش برای افق {horizon} روزه")
            print(f"{'='*70}")
            
            # آماده‌سازی داده
            print("\n1. آماده‌سازی داده آموزش...")
            train_features, train_targets = self.prepare_training_data(
                train_candles, horizon
            )
            print(f"   تعداد نمونه‌های آموزش: {len(train_features)}")
            
            if validation_candles:
                print("\n2. آماده‌سازی داده اعتبارسنجی...")
                val_features, val_targets = self.prepare_training_data(
                    validation_candles, horizon
                )
                print(f"   تعداد نمونه‌های اعتبارسنجی: {len(val_features)}")
            else:
                val_features, val_targets = None, None
            
            # آموزش
            print("\n3. آموزش مدل...")
            self.weight_learner.train_horizon(
                horizon=f"{horizon}d",
                features_list=train_features,
                targets=train_targets
            )
            
            # ارزیابی
            if val_features:
                print("\n4. ارزیابی مدل...")
                predictions = []
                for features in val_features:
                    # محاسبه امتیاز با وزن‌های یادگرفته شده
                    score = self._calculate_weighted_score(features, horizon)
                    predictions.append(score)
                
                predictions = np.array(predictions)
                val_targets_arr = np.array(val_targets)
                
                # محاسبه معیارهای ارزیابی
                mae = np.mean(np.abs(predictions - val_targets_arr))
                rmse = np.sqrt(np.mean((predictions - val_targets_arr) ** 2))
                
                # Accuracy (با threshold)
                correct = np.sum(np.sign(predictions) == np.sign(val_targets_arr))
                accuracy = correct / len(val_targets_arr)
                
                print(f"\n   MAE: {mae:.4f}")
                print(f"   RMSE: {rmse:.4f}")
                print(f"   Direction Accuracy: {accuracy:.2%}")
        
        # ذخیره مدل
        print(f"\n{'='*70}")
        print("ذخیره مدل...")
        os.makedirs(save_path, exist_ok=True)
        model_file = os.path.join(save_path, "cycle_weights.json")
        self.weight_learner.save(model_file)
        print(f"✅ مدل در {model_file} ذخیره شد")
        
        print("\n" + "=" * 70)
        print("✅ آموزش با موفقیت تکمیل شد!")
        print("=" * 70)
    
    def _calculate_weighted_score(
        self,
        features: dict,
        horizon: int
    ) -> float:
        """محاسبه امتیاز با وزن‌های یادگرفته شده"""
        horizon_key = f"{horizon}d"
        weights = self.weight_learner.weights.get(horizon_key)
        
        if not weights:
            return 0.0
        
        indicators = [
            'dpo',
            'ehlers_cycle',
            'dominant_cycle',
            'schaff_trend_cycle',
            'phase_accumulation',
            'hilbert_transform',
            'market_cycle_model'
        ]
        
        total_score = 0.0
        total_weight = 0.0
        
        for indicator in indicators:
            weight = weights.indicator_weights.get(indicator, 1.0 / len(indicators))
            signal = features.get(f"{indicator}_signal", 0.0)
            confidence = features.get(f"{indicator}_confidence", 0.5)
            
            total_score += signal * confidence * weight
            total_weight += confidence * weight
        
        if total_weight > 0:
            return total_score / total_weight
        return 0.0


# ═══════════════════════════════════════════════════════════════
# اجرای آموزش
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Horizon Cycle Training Pipeline")
    print("=" * 70)
    
    # 1. تولید داده آموزش (ترکیب رژیم‌های مختلف)
    print("\n1. تولید داده آموزش...")
    train_data = []
    
    # Fast cycles
    print("   - تولید داده با سیکل‌های سریع...")
    train_data.extend(create_realistic_cycle_data(500, 'fast'))
    
    # Slow cycles
    print("   - تولید داده با سیکل‌های کند...")
    train_data.extend(create_realistic_cycle_data(500, 'slow'))
    
    # Range-bound
    print("   - تولید داده رنج...")
    train_data.extend(create_realistic_cycle_data(500, 'range'))
    
    # Mixed
    print("   - تولید داده ترکیبی...")
    train_data.extend(create_realistic_cycle_data(500, 'mixed'))
    
    print(f"   ✅ مجموع داده آموزش: {len(train_data)} کندل")
    
    # 2. تولید داده اعتبارسنجی
    print("\n2. تولید داده اعتبارسنجی...")
    val_data = create_realistic_cycle_data(600, 'mixed')
    print(f"   ✅ مجموع داده اعتبارسنجی: {len(val_data)} کندل")
    
    # 3. آموزش مدل
    print("\n3. شروع آموزش...")
    trainer = MultiHorizonCycleTrainer(
        lookback_period=100,
        horizons=[3, 7, 30]
    )
    
    trainer.train(
        train_candles=train_data,
        validation_candles=val_data,
        save_path="models/cycle"
    )
    
    # 4. نمایش وزن‌های یادگرفته شده
    print("\n" + "=" * 70)
    print("وزن‌های یادگرفته شده:")
    print("=" * 70)
    
    for horizon, weights in trainer.weight_learner.weights.items():
        print(f"\nافق {horizon}:")
        for indicator, weight in weights.indicator_weights.items():
            print(f"  {indicator}: {weight:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Pipeline کامل شد!")
    print("=" * 70)
    print("\nفایل مدل: models/cycle/cycle_weights.json")
    print("برای استفاده:")
    print("  analyzer = MultiHorizonCycleAnalyzer(weights_path='models/cycle/cycle_weights.json')")
