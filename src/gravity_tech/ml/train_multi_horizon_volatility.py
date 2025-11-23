"""
Training Pipeline برای Multi-Horizon Volatility System

آموزش مدل نوسان برای سه افق مستقل
"""

import numpy as np
import pandas as pd
from typing import Optional, List
import os
from datetime import datetime

from gravity_tech.models.schemas import Candle
from gravity_tech.ml.multi_horizon_volatility_features import MultiHorizonVolatilityFeatureExtractor
from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner


def create_realistic_volatility_data(
    num_samples: int = 2000,
    volatility_regime: str = 'mixed'  # 'low', 'high', 'mixed', 'squeeze'
) -> List[Candle]:
    """
    ایجاد داده‌های واقعی با رژیم‌های مختلف نوسان
    
    Args:
        num_samples: تعداد کندل‌ها
        volatility_regime: رژیم نوسان
            - 'low': نوسان پایین
            - 'high': نوسان بالا
            - 'mixed': ترکیب رژیم‌های مختلف
            - 'squeeze': فشردگی و شکست
    """
    np.random.seed(42)
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_samples, freq='1h')
    
    base_price = 30000
    prices = [base_price]
    
    candles = []
    
    for i in range(num_samples):
        # تعیین volatility بر اساس regime
        if volatility_regime == 'low':
            volatility = 0.005  # 0.5%
        elif volatility_regime == 'high':
            volatility = 0.03  # 3%
        elif volatility_regime == 'squeeze':
            # فشردگی در نیمه اول، شکست در نیمه دوم
            if i < num_samples // 2:
                volatility = 0.003  # بسیار پایین
            else:
                volatility = 0.04  # انفجار نوسان
        else:  # mixed
            # تغییر رژیم در طول زمان
            cycle = i % 300
            if cycle < 100:
                volatility = 0.005  # low
            elif cycle < 200:
                volatility = 0.015  # medium
            else:
                volatility = 0.03  # high
        
        # تولید قیمت
        drift = np.random.normal(0, 0.0005)  # drift خیلی کم
        change = drift + np.random.normal(0, volatility)
        
        if i == 0:
            open_price = base_price
            close_price = base_price * (1 + change)
        else:
            open_price = prices[-1]
            close_price = open_price * (1 + change)
        
        prices.append(close_price)
        
        # High/Low با توجه به volatility
        high_change = abs(np.random.normal(0, volatility))
        low_change = abs(np.random.normal(0, volatility))
        
        high_price = max(open_price, close_price) * (1 + high_change)
        low_price = min(open_price, close_price) * (1 - low_change)
        
        # Volume
        base_volume = 1000000
        volume = base_volume * (1 + np.random.normal(0, 0.3))
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


def train_volatility_model(
    candles: List[Candle],
    horizons: List[str] = None,
    test_size: float = 0.2,
    output_dir: str = 'models/volatility',
    verbose: bool = True
) -> MultiHorizonWeightLearner:
    """
    آموزش مدل نوسان چند افقی
    
    Args:
        candles: لیست کندل‌ها
        horizons: لیست افق‌ها (پیش‌فرض: ['3d', '7d', '30d'])
        test_size: درصد داده تست
        output_dir: مسیر ذخیره مدل
        verbose: نمایش جزئیات
        
    Returns:
        مدل آموزش‌دیده
    """
    if horizons is None:
        horizons = ['3d', '7d', '30d']
    
    if verbose:
        print("="*70)
        print("🎯 TRAINING MULTI-HORIZON VOLATILITY MODEL")
        print("="*70)
        print(f"\n📊 Dataset: {len(candles)} candles")
        print(f"⏱️  Horizons: {horizons}")
        print(f"✂️  Test Size: {test_size:.0%}")
    
    # استخراج ویژگی‌ها
    if verbose:
        print("\n🔍 Extracting volatility features...")
    
    extractor = MultiHorizonVolatilityFeatureExtractor(horizons=horizons)
    X, Y = extractor.create_training_dataset(candles, horizons=[int(h.replace('d', '')) for h in horizons])
    
    if verbose:
        print(f"   ✅ Features: {X.shape[1]} columns")
        print(f"   ✅ Samples: {X.shape[0]} rows")
        print(f"   ✅ Targets: {Y.shape[1]} horizons")
        print(f"\n   📋 Feature columns:")
        for i, col in enumerate(X.columns[:5], 1):
            print(f"      {i}. {col}")
        if len(X.columns) > 5:
            print(f"      ... and {len(X.columns) - 5} more")
    
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
            'n_estimators': 150,  # بیشتر از momentum
            'learning_rate': 0.03,  # کمتر برای دقت بیشتر
            'num_leaves': 31,
            'max_depth': 7,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 0.1   # L2 regularization
        }
    )
    
    learner.train(X, Y, verbose=verbose)
    
    # ذخیره مدل
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'volatility_weights.json')
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
            print(f"  RMSE:       {weights_info.metrics['rmse']:.4f}")
            print(f"  Confidence: {weights_info.confidence:.2f}")
            
            # نمایش top features
            top_features = sorted(
                weights_info.weights.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            
            print(f"\n  🔝 Top 5 Features:")
            for feat, weight in top_features:
                print(f"     {feat:30s}: {weight:+.4f}")
    
    # تحلیل اهمیت اندیکاتورها
    if verbose:
        print("\n" + "="*70)
        print("📊 INDICATOR IMPORTANCE ANALYSIS")
        print("="*70)
        
        indicators = [
            'atr', 'bollinger_bands', 'keltner_channel', 'donchian_channel',
            'standard_deviation', 'historical_volatility', 'atr_percentage', 'chaikin_volatility'
        ]
        
        for horizon in horizons:
            weights_info = learner.get_horizon_weights(horizon)
            
            print(f"\n{horizon.upper()}:")
            indicator_importance = {}
            
            for indicator in indicators:
                # جمع وزن‌های absolute برای همه ویژگی‌های این اندیکاتور
                indicator_weights = [
                    abs(weight) for feat, weight in weights_info.weights.items()
                    if feat.startswith(indicator)
                ]
                if indicator_weights:
                    indicator_importance[indicator] = sum(indicator_weights) / len(indicator_weights)
            
            # مرتب‌سازی
            sorted_importance = sorted(
                indicator_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for indicator, importance in sorted_importance:
                bar = "█" * int(importance * 50)
                print(f"  {indicator:25s}: {bar} {importance:.3f}")
    
    return learner


def main():
    """
    اجرای کامل pipeline آموزش
    """
    print("\n" + "="*70)
    print("🚀 VOLATILITY MODEL TRAINING PIPELINE")
    print("="*70)
    
    # 1. ایجاد داده‌های آموزشی
    print("\n📦 Creating training data...")
    print("   Generating mixed volatility regime data...")
    
    candles = create_realistic_volatility_data(
        num_samples=2000,
        volatility_regime='mixed'
    )
    
    print(f"   ✅ Generated {len(candles)} candles")
    print(f"   📅 Date range: {candles[0].timestamp} to {candles[-1].timestamp}")
    
    # آمار اولیه
    closes = [c.close for c in candles]
    print(f"\n   📊 Price statistics:")
    print(f"      Min:  ${min(closes):,.2f}")
    print(f"      Max:  ${max(closes):,.2f}")
    print(f"      Mean: ${np.mean(closes):,.2f}")
    print(f"      Std:  ${np.std(closes):,.2f}")
    
    # 2. آموزش مدل
    print("\n" + "="*70)
    learner = train_volatility_model(
        candles=candles,
        horizons=['3d', '7d', '30d'],
        test_size=0.2,
        output_dir='models/volatility',
        verbose=True
    )
    
    # 3. تست مدل
    print("\n" + "="*70)
    print("🧪 TESTING MODEL ON RECENT DATA")
    print("="*70)
    
    # استخراج ویژگی‌های آخرین داده
    extractor = MultiHorizonVolatilityFeatureExtractor()
    features = extractor.extract_volatility_features(candles)
    
    print("\n📊 Sample features:")
    for i, (key, value) in enumerate(list(features.items())[:5], 1):
        print(f"   {i}. {key:30s}: {value:.4f}")
    
    # پیش‌بینی
    X_test = pd.DataFrame([features])
    predictions = learner.predict_multi_horizon(X_test)
    
    print("\n🎯 Predictions:")
    for horizon in ['3d', '7d', '30d']:
        pred_value = predictions[f'pred_{horizon}'].iloc[0]
        print(f"   {horizon}: {pred_value:+.4f} ({'افزایش نوسان' if pred_value > 0 else 'کاهش نوسان'})")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\n💡 Next steps:")
    print("   1. Test the model with real market data")
    print("   2. Integrate with MultiHorizonVolatilityAnalyzer")
    print("   3. Compare predictions with actual volatility changes")
    print("   4. Fine-tune hyperparameters if needed")
    print()


if __name__ == '__main__':
    main()
