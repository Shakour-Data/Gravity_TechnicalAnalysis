"""
Multi-Horizon Training Pipeline

Pipeline کامل برای آموزش سیستم چند افقی:
1. دریافت داده (Bitcoin)
2. استخراج ویژگی‌ها (3d, 7d, 30d)
3. آموزش مدل Multi-Output
4. ذخیره وزن‌ها
5. ارزیابی و گزارش
"""

import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gravity_tech.models.schemas import Candle
from gravity_tech.ml.multi_horizon_feature_extraction import MultiHorizonFeatureExtractor
from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner


def create_realistic_market_data(
    base_price: float = 50000,
    candles_count: int = 500,
    trend: str = "mixed"
) -> List[Candle]:
    """
    ساخت داده واقع‌گرایانه بازار برای آموزش
    
    Args:
        base_price: قیمت شروع
        candles_count: تعداد کندل‌ها
        trend: 'bullish', 'bearish', 'mixed'
    """
    candles = []
    base_time = datetime.now() - timedelta(days=candles_count)
    current_price = base_price
    
    for i in range(candles_count):
        # شبیه‌سازی حرکت واقعی قیمت
        if trend == "bullish":
            trend_component = i * 10
            volatility = 0.02
        elif trend == "bearish":
            trend_component = -i * 10
            volatility = 0.02
        else:  # mixed
            # ترکیب از چندین سیکل
            trend_component = (
                np.sin(i / 30) * 1000 +  # سیکل کوتاه
                np.sin(i / 100) * 3000 +  # سیکل میان
                i * 5  # روند کلی صعودی ملایم
            )
            volatility = 0.025
        
        # قیمت close
        close_price = current_price + trend_component + np.random.normal(0, current_price * volatility)
        
        # high و low
        daily_range = abs(np.random.normal(0, current_price * volatility * 1.5))
        high = close_price + daily_range * np.random.uniform(0.3, 0.7)
        low = close_price - daily_range * np.random.uniform(0.3, 0.7)
        
        # open
        if i > 0:
            open_price = candles[-1].close + np.random.normal(0, current_price * volatility * 0.5)
        else:
            open_price = current_price
        
        # volume
        volume = abs(np.random.normal(1000000, 200000))
        
        candle = Candle(
            timestamp=(base_time + timedelta(days=i)).isoformat(),
            open=max(0, open_price),
            high=max(0, high),
            low=max(0, low),
            close=max(0, close_price),
            volume=volume
        )
        
        candles.append(candle)
        current_price = close_price
    
    return candles


def train_multi_horizon_system(
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    lookback_days: int = 365,
    horizons: list = None,
    output_dir: str = "ml_models/multi_horizon"
):
    """
    آموزش سیستم چند افقی
    
    Args:
        symbol: نماد (مثلاً BTCUSDT)
        interval: بازه زمانی (1d)
        lookback_days: تعداد روزهای گذشته
        horizons: لیست افق‌ها (پیش‌فرض: [3, 7, 30])
        output_dir: مسیر ذخیره مدل‌ها
    """
    horizons = horizons or [3, 7, 30]
    
    print("\n" + "="*70)
    print("🚀 MULTI-HORIZON TRAINING PIPELINE")
    print("="*70)
    print(f"Symbol: {symbol}")
    print(f"Interval: {interval}")
    print(f"Lookback: {lookback_days} days")
    print(f"Horizons: {horizons} days")
    print("="*70)
    
    # ═══════════════════════════════════════════════════════════
    # Step 1: دریافت داده
    # ═══════════════════════════════════════════════════════════
    print("\n📊 Step 1: Generating market data...")
    
    candles = create_realistic_market_data(
        base_price=50000,
        candles_count=lookback_days + 60,  # اضافه برای lookback و horizon
        trend="mixed"
    )
    
    print(f"✅ Generated {len(candles)} candles")
    print(f"   Date range: {candles[0].timestamp} → {candles[-1].timestamp}")
    print(f"   Price range: ${candles[0].close:.2f} → ${candles[-1].close:.2f}")
    
    # ═══════════════════════════════════════════════════════════
    # Step 2: استخراج ویژگی‌ها - سطح 1 (Indicators)
    # ═══════════════════════════════════════════════════════════
    print("\n🔬 Step 2a: Extracting Level 1 Features (Indicators)...")
    
    extractor = MultiHorizonFeatureExtractor(
        lookback_period=100,
        horizons=horizons
    )
    
    X_indicators, Y = extractor.extract_training_dataset(
        candles,
        level="indicators"
    )
    
    print(f"✅ Level 1 Features: {X_indicators.shape}")
    print(f"   Targets: {Y.shape}")
    
    # آمار
    stats_indicators = extractor.create_summary_statistics(X_indicators, Y)
    
    # ═══════════════════════════════════════════════════════════
    # Step 3: آموزش مدل - سطح 1
    # ═══════════════════════════════════════════════════════════
    print("\n🎓 Step 3a: Training Level 1 Model (Indicator Weights)...")
    
    learner_indicators = MultiHorizonWeightLearner(
        horizons=[f'{h}d' for h in horizons],
        test_size=0.2,
        random_state=42
    )
    
    learner_indicators.train(X_indicators, Y, verbose=True)
    
    # ذخیره وزن‌های سطح 1
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    weights_file_l1 = output_path / f"indicator_weights_{symbol.lower()}.json"
    learner_indicators.save_weights(str(weights_file_l1))
    
    # ═══════════════════════════════════════════════════════════
    # Step 4: استخراج ویژگی‌ها - سطح 2 (Dimensions)
    # ═══════════════════════════════════════════════════════════
    print("\n🔬 Step 4a: Extracting Level 2 Features (Dimensions)...")
    
    X_dimensions, Y_dim = extractor.extract_training_dataset(
        candles,
        level="dimensions"
    )
    
    print(f"✅ Level 2 Features: {X_dimensions.shape}")
    
    stats_dimensions = extractor.create_summary_statistics(X_dimensions, Y_dim)
    
    # ═══════════════════════════════════════════════════════════
    # Step 5: آموزش مدل - سطح 2
    # ═══════════════════════════════════════════════════════════
    print("\n🎓 Step 5a: Training Level 2 Model (Dimension Weights)...")
    
    learner_dimensions = MultiHorizonWeightLearner(
        horizons=[f'{h}d' for h in horizons],
        test_size=0.2,
        random_state=42
    )
    
    learner_dimensions.train(X_dimensions, Y_dim, verbose=True)
    
    # ذخیره وزن‌های سطح 2
    weights_file_l2 = output_path / f"dimension_weights_{symbol.lower()}.json"
    learner_dimensions.save_weights(str(weights_file_l2))
    
    # ═══════════════════════════════════════════════════════════
    # Step 6: گزارش نهایی
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("📋 FINAL REPORT")
    print("="*70)
    
    # مقایسه دو سطح
    print("\n🔍 Level 1 (Indicators) vs Level 2 (Dimensions):")
    print("-" * 70)
    
    summary_l1 = learner_indicators.get_summary()
    summary_l2 = learner_dimensions.get_summary()
    
    for horizon in [f'{h}d' for h in horizons]:
        print(f"\n{horizon.upper()}:")
        
        l1_details = summary_l1['horizon_details'][horizon]
        l2_details = summary_l2['horizon_details'][horizon]
        
        print(f"  Level 1 (Indicators):")
        print(f"    R²:         {l1_details['r2_test']:+.4f}")
        print(f"    MAE:        {l1_details['mae_test']:.4f} ({l1_details['mae_test']*100:.2f}%)")
        print(f"    Confidence: {l1_details['confidence']:.2f}")
        
        print(f"  Level 2 (Dimensions):")
        print(f"    R²:         {l2_details['r2_test']:+.4f}")
        print(f"    MAE:        {l2_details['mae_test']:.4f} ({l2_details['mae_test']*100:.2f}%)")
        print(f"    Confidence: {l2_details['confidence']:.2f}")
        
        # بهتر کدام؟
        if l1_details['r2_test'] > l2_details['r2_test']:
            print(f"    ✅ Level 1 performs better (R² difference: {l1_details['r2_test'] - l2_details['r2_test']:+.4f})")
        else:
            print(f"    ✅ Level 2 performs better (R² difference: {l2_details['r2_test'] - l1_details['r2_test']:+.4f})")
    
    # ═══════════════════════════════════════════════════════════
    # Step 7: ذخیره تنظیمات
    # ═══════════════════════════════════════════════════════════
    config = {
        'symbol': symbol,
        'interval': interval,
        'training_date': datetime.now().isoformat(),
        'lookback_days': lookback_days,
        'horizons': horizons,
        'n_samples': len(X_indicators),
        'level1': {
            'n_features': X_indicators.shape[1],
            'feature_names': list(X_indicators.columns),
            'weights_file': str(weights_file_l1)
        },
        'level2': {
            'n_features': X_dimensions.shape[1],
            'feature_names': list(X_dimensions.columns),
            'weights_file': str(weights_file_l2)
        },
        'statistics': {
            'level1': stats_indicators,
            'level2': stats_dimensions
        }
    }
    
    config_file = output_path / f"config_{symbol.lower()}.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Configuration saved: {config_file}")
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED!")
    print("="*70)
    print(f"Output directory: {output_path}")
    print(f"Files created:")
    print(f"  - {weights_file_l1.name}")
    print(f"  - {weights_file_l2.name}")
    print(f"  - {config_file.name}")
    print("="*70)
    
    return {
        'learner_indicators': learner_indicators,
        'learner_dimensions': learner_dimensions,
        'config': config,
        'output_dir': output_path
    }


def load_trained_model(
    symbol: str = "BTCUSDT",
    level: str = "indicators",  # "indicators" or "dimensions"
    model_dir: str = "ml_models/multi_horizon"
):
    """
    بارگذاری مدل آموزش دیده
    
    Args:
        symbol: نماد
        level: سطح (indicators یا dimensions)
        model_dir: مسیر مدل‌ها
        
    Returns:
        MultiHorizonWeightLearner
    """
    model_path = Path(model_dir)
    
    if level == "indicators":
        weights_file = model_path / f"indicator_weights_{symbol.lower()}.json"
    else:
        weights_file = model_path / f"dimension_weights_{symbol.lower()}.json"
    
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")
    
    learner = MultiHorizonWeightLearner()
    learner.load_weights(str(weights_file))
    
    return learner


# ═══════════════════════════════════════════════════════════
# CLI Interface
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Horizon Training Pipeline"
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading symbol (default: BTCUSDT)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='1d',
        help='Candle interval (default: 1d)'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=365,
        help='Lookback days (default: 365)'
    )
    parser.add_argument(
        '--horizons',
        type=int,
        nargs='+',
        default=[3, 7, 30],
        help='Horizons in days (default: 3 7 30)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ml_models/multi_horizon',
        help='Output directory (default: ml_models/multi_horizon)'
    )
    
    args = parser.parse_args()
    
    # آموزش
    result = train_multi_horizon_system(
        symbol=args.symbol,
        interval=args.interval,
        lookback_days=args.lookback,
        horizons=args.horizons,
        output_dir=args.output
    )
    
    print("\n✅ Training pipeline finished successfully!")
