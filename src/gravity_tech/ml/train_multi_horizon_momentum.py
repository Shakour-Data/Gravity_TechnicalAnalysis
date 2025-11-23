"""
Training Pipeline برای Multi-Horizon Momentum System

آموزش مدل مومنتوم برای سه افق مستقل
"""

import numpy as np
import pandas as pd
from typing import Optional
import os

from gravity_tech.ml.multi_horizon_momentum_features import MultiHorizonMomentumFeatureExtractor
from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner


def create_realistic_market_data(
    num_samples: int = 2000,
    trend: str = 'mixed'  # 'uptrend', 'downtrend', 'mixed'
) -> pd.DataFrame:
    """
    ایجاد داده‌های واقعی بازار برای تست
    """
    np.random.seed(42)
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_samples, freq='1h')
    
    # قیمت پایه
    base_price = 30000
    prices = [base_price]
    volumes = []
    
    for i in range(1, num_samples):
        if trend == 'uptrend':
            drift = 0.002
        elif trend == 'downtrend':
            drift = -0.002
        else:  # mixed
            if i < num_samples // 3:
                drift = 0.003
            elif i < 2 * num_samples // 3:
                drift = -0.002
            else:
                drift = 0.001
        
        volatility = 0.01
        change = drift + np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
        
        base_volume = 1000000
        volume = base_volume * (1 + np.random.normal(0, 0.3))
        volumes.append(max(volume, 100000))
    
    # یک حجم اضافه برای index 0
    volumes.insert(0, 1000000)
    
    # OHLC
    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices
    })
    
    df['high'] = df['close'] * (1 + np.abs(np.random.normal(0, 0.005, len(df))))
    df['low'] = df['close'] * (1 - np.abs(np.random.normal(0, 0.005, len(df))))
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    df['volume'] = volumes
    
    return df


def train_momentum_model(
    candles: pd.DataFrame,
    horizons: list[str] = None,
    test_size: float = 0.2,
    output_dir: str = 'models/momentum',
    verbose: bool = True
) -> MultiHorizonWeightLearner:
    """
    آموزش مدل مومنتوم چند افقی
    
    Args:
        candles: داده‌های کندل (OHLCV)
        horizons: لیست افق‌ها (پیش‌فرض: ['3d', '7d', '30d'])
        test_size: درصد داده تست
        output_dir: مسیر ذخیره مدل
        verbose: نمایش جزئیات
    """
    if horizons is None:
        horizons = ['3d', '7d', '30d']
    
    if verbose:
        print("="*70)
        print("🎯 TRAINING MULTI-HORIZON MOMENTUM MODEL")
        print("="*70)
        print(f"\n📊 Dataset: {len(candles)} candles")
        print(f"⏱️  Horizons: {horizons}")
        print(f"✂️  Test Size: {test_size:.0%}")
    
    # استخراج ویژگی‌ها
    if verbose:
        print("\n🔍 Extracting momentum features...")
    
    extractor = MultiHorizonMomentumFeatureExtractor(horizons=horizons)
    X, Y = extractor.extract_training_dataset(candles)
    
    if verbose:
        print(f"   ✅ Features: {X.shape[1]} columns")
        print(f"   ✅ Samples: {X.shape[0]} rows")
        print(f"   ✅ Targets: {Y.shape[1]} horizons")
    
    # آموزش مدل
    if verbose:
        print("\n🤖 Training models...")
    
    learner = MultiHorizonWeightLearner(
        horizons=horizons,
        test_size=test_size,
        random_state=42,
        lgbm_params={
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'n_estimators': 100,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6
        }
    )
    
    learner.train(X, Y, verbose=verbose)
    
    # ذخیره مدل
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'momentum_weights.json')
    learner.save_weights(model_path)
    
    if verbose:
        print(f"\n💾 Model saved: {model_path}")
    
    # نمایش نتایج
    if verbose:
        print("\n" + "="*70)
        print("📈 TRAINING RESULTS")
        print("="*70)
        
        for horizon in horizons:
            weights_info = learner.get_horizon_weights(horizon)
            print(f"\n{horizon.upper()}:")
            print(f"  R² Score:   {weights_info.metrics['r2_score']:.3f}")
            print(f"  MAE:        {weights_info.metrics['mae']:.4f}")
            print(f"  Confidence: {weights_info.confidence:.0%}")
    
    return learner


def main():
    """تابع اصلی آموزش"""
    print("\n🚀 Starting Momentum Training Pipeline\n")
    
    # ایجاد داده
    print("📦 Generating market data...")
    candles = create_realistic_market_data(num_samples=3000, trend='mixed')
    print(f"   ✅ Generated {len(candles)} candles\n")
    
    # آموزش مدل
    learner = train_momentum_model(
        candles=candles,
        horizons=['3d', '7d', '30d'],
        test_size=0.2,
        output_dir='models/momentum',
        verbose=True
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print("\nModel ready for inference!")
    print("Use: learner.predict_multi_horizon(X)")
    
    return learner


if __name__ == '__main__':
    learner = main()
