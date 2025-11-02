"""
مثال استفاده از سیستم تحلیل نوسان چند افقی

این مثال نشان می‌دهد:
1. ایجاد داده‌های واقعی بازار
2. استخراج ویژگی‌های نوسان
3. آموزش مدل ML
4. تحلیل چند افقی
5. دریافت توصیه‌های معاملاتی
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

from models.schemas import Candle
from indicators.volatility import VolatilityIndicators
from ml.multi_horizon_volatility_features import MultiHorizonVolatilityFeatureExtractor
from ml.multi_horizon_volatility_analysis import MultiHorizonVolatilityAnalyzer
from ml.train_multi_horizon_volatility import create_realistic_volatility_data, train_volatility_model


def create_sample_candles(num_candles: int = 500) -> list:
    """ایجاد داده‌های نمونه"""
    print("\n📦 Creating sample data...")
    candles = create_realistic_volatility_data(
        num_samples=num_candles,
        volatility_regime='mixed'
    )
    print(f"   ✅ Created {len(candles)} candles")
    return candles


def example_1_basic_volatility_indicators():
    """
    مثال 1: استفاده از اندیکاتورهای پایه نوسان
    """
    print("\n" + "="*70)
    print("📊 EXAMPLE 1: Basic Volatility Indicators")
    print("="*70)
    
    # ایجاد داده
    candles = create_sample_candles(200)
    
    print("\n🔍 Calculating volatility indicators...")
    
    # ═══════════════════════════════════════════════════════
    # 1. ATR (Average True Range)
    # ═══════════════════════════════════════════════════════
    print("\n1️⃣ ATR (Average True Range):")
    atr_result = VolatilityIndicators.atr(candles, period=14)
    print(f"   Value:       {atr_result.value:.2f}")
    print(f"   Normalized:  {atr_result.normalized:+.3f}")
    print(f"   Percentile:  {atr_result.percentile:.1f}th")
    print(f"   Signal:      {atr_result.signal.name}")
    print(f"   Confidence:  {atr_result.confidence:.2f}")
    print(f"   📝 {atr_result.description}")
    
    # ═══════════════════════════════════════════════════════
    # 2. Bollinger Bands
    # ═══════════════════════════════════════════════════════
    print("\n2️⃣ Bollinger Bands:")
    bb_result = VolatilityIndicators.bollinger_bands(candles, period=20)
    print(f"   Bandwidth:   {bb_result.value:.2f}%")
    print(f"   Percentile:  {bb_result.percentile:.1f}th")
    print(f"   Signal:      {bb_result.signal.name}")
    print(f"   📝 {bb_result.description}")
    
    # ═══════════════════════════════════════════════════════
    # 3. Historical Volatility
    # ═══════════════════════════════════════════════════════
    print("\n3️⃣ Historical Volatility:")
    hv_result = VolatilityIndicators.historical_volatility(candles, period=20)
    print(f"   HV:          {hv_result.value:.2f}%")
    print(f"   Percentile:  {hv_result.percentile:.1f}th")
    print(f"   Signal:      {hv_result.signal.name}")
    print(f"   📝 {hv_result.description}")
    
    # ═══════════════════════════════════════════════════════
    # 4. Chaikin Volatility
    # ═══════════════════════════════════════════════════════
    print("\n4️⃣ Chaikin Volatility:")
    chaikin_result = VolatilityIndicators.chaikin_volatility(candles, period=10)
    print(f"   Value:       {chaikin_result.value:+.2f}%")
    print(f"   Signal:      {chaikin_result.signal.name}")
    print(f"   📝 {chaikin_result.description}")
    
    # ═══════════════════════════════════════════════════════
    # 5. همه اندیکاتورها یکجا
    # ═══════════════════════════════════════════════════════
    print("\n" + "-"*70)
    print("📊 ALL INDICATORS SUMMARY:")
    print("-"*70)
    
    all_results = VolatilityIndicators.calculate_all(candles)
    
    for name, result in all_results.items():
        direction = "↗️" if result.normalized > 0 else "↘️" if result.normalized < 0 else "→"
        print(f"   {name:25s}: {result.signal.name:15s} {direction} ({result.confidence:.2f})")


def example_2_feature_extraction():
    """
    مثال 2: استخراج ویژگی‌ها برای ML
    """
    print("\n" + "="*70)
    print("🔍 EXAMPLE 2: Feature Extraction for ML")
    print("="*70)
    
    # ایجاد داده
    candles = create_sample_candles(200)
    
    # استخراج ویژگی‌ها
    print("\n📊 Extracting features...")
    extractor = MultiHorizonVolatilityFeatureExtractor(horizons=['3d', '7d', '30d'])
    features = extractor.extract_volatility_features(candles)
    
    print(f"   ✅ Extracted {len(features)} features")
    
    # نمایش ویژگی‌ها
    print("\n🔢 Sample features:")
    feature_groups = {}
    for key, value in features.items():
        indicator = key.rsplit('_', 1)[0]
        if indicator not in feature_groups:
            feature_groups[indicator] = {}
        feature_groups[indicator][key] = value
    
    for i, (indicator, feats) in enumerate(list(feature_groups.items())[:3], 1):
        print(f"\n   {i}. {indicator.upper()}:")
        for feat_name, feat_value in feats.items():
            print(f"      {feat_name:35s}: {feat_value:+.4f}")


def example_3_training_model():
    """
    مثال 3: آموزش مدل ML
    """
    print("\n" + "="*70)
    print("🤖 EXAMPLE 3: Training ML Model")
    print("="*70)
    
    # ایجاد داده‌های آموزشی
    candles = create_sample_candles(1000)
    
    # آموزش مدل
    print("\n🎯 Training volatility model...")
    learner = train_volatility_model(
        candles=candles,
        horizons=['3d', '7d', '30d'],
        test_size=0.2,
        output_dir='models/volatility',
        verbose=True
    )
    
    print("\n✅ Model trained successfully!")


def example_4_full_analysis():
    """
    مثال 4: تحلیل کامل چند افقی
    """
    print("\n" + "="*70)
    print("📈 EXAMPLE 4: Full Multi-Horizon Analysis")
    print("="*70)
    
    # ایجاد و آموزش مدل
    print("\n1️⃣ Preparing model...")
    candles = create_sample_candles(1000)
    
    learner = train_volatility_model(
        candles=candles,
        horizons=['3d', '7d', '30d'],
        test_size=0.2,
        output_dir='models/volatility',
        verbose=False
    )
    
    print("   ✅ Model ready")
    
    # استخراج ویژگی‌ها
    print("\n2️⃣ Extracting features from recent data...")
    extractor = MultiHorizonVolatilityFeatureExtractor()
    features = extractor.extract_volatility_features(candles)
    print("   ✅ Features extracted")
    
    # تحلیل چند افقی
    print("\n3️⃣ Performing multi-horizon analysis...")
    analyzer = MultiHorizonVolatilityAnalyzer(learner)
    analysis = analyzer.analyze(features)
    
    # نمایش نتایج
    print("\n" + "="*70)
    print("📊 ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\n⏰ Timestamp: {analysis.timestamp}")
    
    print("\n📈 Volatility Scores:")
    print(f"   3-Day:  {analysis.volatility_3d.score:+.3f} ({analysis.volatility_3d.get_strength()}) - {analysis.volatility_3d.get_direction()}")
    print(f"   7-Day:  {analysis.volatility_7d.score:+.3f} ({analysis.volatility_7d.get_strength()}) - {analysis.volatility_7d.get_direction()}")
    print(f"   30-Day: {analysis.volatility_30d.score:+.3f} ({analysis.volatility_30d.get_strength()}) - {analysis.volatility_30d.get_direction()}")
    
    print(f"\n📊 Combined:")
    print(f"   Score:      {analysis.combined_volatility:+.3f}")
    print(f"   Confidence: {analysis.combined_confidence:.2f}")
    
    print(f"\n🎯 Volatility Phase: {analysis.volatility_phase}")
    
    print("\n💡 Recommendations:")
    print(f"   3d:  {analysis.recommendation_3d}")
    print(f"   7d:  {analysis.recommendation_7d}")
    print(f"   30d: {analysis.recommendation_30d}")
    
    # مشاوره معاملاتی
    print("\n" + "="*70)
    print("💼 TRADING ADVICE")
    print("="*70)
    
    advice = analyzer.get_trading_advice(analysis)
    
    for trader_type, recommendation in advice.items():
        print(f"\n{trader_type.upper()}:")
        print(f"   {recommendation}")


def example_5_volatility_scenarios():
    """
    مثال 5: سناریوهای مختلف نوسان
    """
    print("\n" + "="*70)
    print("🎬 EXAMPLE 5: Volatility Scenarios")
    print("="*70)
    
    scenarios = {
        'low': 'Low Volatility (Calm Market)',
        'high': 'High Volatility (Active Market)',
        'squeeze': 'Squeeze & Breakout',
        'mixed': 'Mixed Volatility Regimes'
    }
    
    for regime, description in scenarios.items():
        print(f"\n{'─'*70}")
        print(f"📊 Scenario: {description}")
        print(f"{'─'*70}")
        
        # ایجاد داده با رژیم خاص
        candles = create_realistic_volatility_data(
            num_samples=500,
            volatility_regime=regime
        )
        
        # محاسبه اندیکاتورها
        atr = VolatilityIndicators.atr(candles)
        bb = VolatilityIndicators.bollinger_bands(candles)
        hv = VolatilityIndicators.historical_volatility(candles)
        
        print(f"\n   ATR:        {atr.signal.name:15s} (Percentile: {atr.percentile:.0f}th)")
        print(f"   Bollinger:  {bb.signal.name:15s} (Percentile: {bb.percentile:.0f}th)")
        print(f"   Hist Vol:   {hv.signal.name:15s} (Value: {hv.value:.1f}%)")
        
        # تفسیر
        if regime == 'squeeze':
            print("\n   💡 Interpretation:")
            print("      - All indicators show low volatility")
            print("      - Market is consolidating")
            print("      - ⚠️ Expect breakout soon!")
        elif regime == 'high':
            print("\n   💡 Interpretation:")
            print("      - High volatility regime")
            print("      - Active market with opportunities")
            print("      - ⚠️ Higher risk - use smaller positions")
        elif regime == 'low':
            print("\n   💡 Interpretation:")
            print("      - Low volatility environment")
            print("      - Calm market conditions")
            print("      - ✅ Good for swing trading")


def main():
    """اجرای همه مثال‌ها"""
    print("\n" + "="*70)
    print("🚀 VOLATILITY ANALYSIS EXAMPLES")
    print("="*70)
    print("\nThis will demonstrate:")
    print("  1. Basic volatility indicators")
    print("  2. Feature extraction for ML")
    print("  3. Training ML model")
    print("  4. Full multi-horizon analysis")
    print("  5. Different volatility scenarios")
    
    try:
        # مثال 1
        example_1_basic_volatility_indicators()
        
        # مثال 2
        example_2_feature_extraction()
        
        # مثال 3
        example_3_training_model()
        
        # مثال 4
        example_4_full_analysis()
        
        # مثال 5
        example_5_volatility_scenarios()
        
        print("\n" + "="*70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\n📚 For more information, see:")
        print("   - VOLATILITY_ANALYSIS_GUIDE.md")
        print("   - indicators/volatility.py")
        print("   - ml/multi_horizon_volatility_analysis.py")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
