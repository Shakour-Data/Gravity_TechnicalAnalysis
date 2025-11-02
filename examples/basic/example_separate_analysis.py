"""
مثال: تحلیل جداگانه Trend و Momentum

نشان می‌دهد چطور دو تحلیل مستقل انجام می‌شود
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from ml.multi_horizon_feature_extraction import MultiHorizonFeatureExtractor
from ml.multi_horizon_momentum_features import MultiHorizonMomentumFeatureExtractor
from ml.multi_horizon_weights import MultiHorizonWeightLearner
from ml.multi_horizon_analysis import MultiHorizonAnalyzer
from ml.multi_horizon_momentum_analysis import MultiHorizonMomentumAnalyzer
from ml.combined_trend_momentum_analysis import CombinedTrendMomentumAnalyzer


def create_market_data(num_samples: int = 1500) -> pd.DataFrame:
    """ایجاد داده بازار"""
    np.random.seed(42)
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_samples, freq='1h')
    base_price = 30000
    prices = [base_price]
    
    for i in range(1, num_samples):
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
    
    volumes = [1000000 * (1 + np.random.normal(0, 0.3)) for _ in range(num_samples)]
    
    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': volumes
    })
    
    df['high'] = df['close'] * (1 + np.abs(np.random.normal(0, 0.005, len(df))))
    df['low'] = df['close'] * (1 - np.abs(np.random.normal(0, 0.005, len(df))))
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    
    return df


def main():
    """مثال اصلی"""
    print("\n" + "🔷"*40)
    print("مثال: تحلیل جداگانه TREND و MOMENTUM")
    print("🔷"*40)
    
    # ایجاد داده
    print("\n📦 ایجاد داده بازار...")
    candles = create_market_data(1500)
    print(f"   ✅ {len(candles)} کندل ایجاد شد")
    
    # ════════════════════════════════════════════════════════
    # 1️⃣ تحلیل TREND (10 اندیکاتور روند)
    # ════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("1️⃣  تحلیل TREND (فقط اندیکاتورهای روند)")
    print("="*80)
    
    print("\n🔍 استخراج ویژگی‌های روند...")
    print("   اندیکاتورها: SMA, EMA, WMA, DEMA, TEMA, MACD, ADX, SAR, Supertrend, Ichimoku")
    
    trend_extractor = MultiHorizonFeatureExtractor(horizons=['3d', '7d', '30d'])
    X_trend, Y_trend = trend_extractor.extract_training_dataset(candles)
    
    print(f"   ✅ {X_trend.shape[1]} ویژگی روند استخراج شد")
    print(f"   ✅ {X_trend.shape[0]} نمونه آماده")
    
    print("\n🤖 آموزش مدل روند...")
    trend_learner = MultiHorizonWeightLearner(
        horizons=['3d', '7d', '30d'],
        test_size=0.2,
        random_state=42
    )
    trend_learner.train(X_trend, Y_trend, verbose=False)
    print("   ✅ مدل روند آموزش داده شد")
    
    print("\n📊 تحلیل روند...")
    trend_analyzer = MultiHorizonAnalyzer(trend_learner)
    trend_features = X_trend.iloc[-1].to_dict()
    trend_analysis = trend_analyzer.analyze(trend_features)
    
    print("\n📈 نتایج تحلیل TREND:")
    print(f"   3d:  امتیاز = {trend_analysis.trend_3d.score:+.3f}, اعتماد = {trend_analysis.trend_3d.confidence:.0%}")
    print(f"   7d:  امتیاز = {trend_analysis.trend_7d.score:+.3f}, اعتماد = {trend_analysis.trend_7d.confidence:.0%}")
    print(f"   30d: امتیاز = {trend_analysis.trend_30d.score:+.3f}, اعتماد = {trend_analysis.trend_30d.confidence:.0%}")
    print(f"\n   💡 توصیه 3d: {trend_analysis.recommendation_3d}")
    print(f"   💡 توصیه 7d: {trend_analysis.recommendation_7d}")
    print(f"   💡 توصیه 30d: {trend_analysis.recommendation_30d}")
    
    # ════════════════════════════════════════════════════════
    # 2️⃣ تحلیل MOMENTUM (8 اندیکاتور مومنتوم)
    # ════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("2️⃣  تحلیل MOMENTUM (فقط اندیکاتورهای مومنتوم)")
    print("="*80)
    
    print("\n🔍 استخراج ویژگی‌های مومنتوم...")
    print("   اندیکاتورها: RSI, Stochastic, CCI, Williams %R, ROC, Momentum, OBV, CMF")
    print("   + تشخیص Divergence (Regular/Hidden Bullish/Bearish)")
    
    momentum_extractor = MultiHorizonMomentumFeatureExtractor(horizons=['3d', '7d', '30d'])
    X_momentum, Y_momentum = momentum_extractor.extract_training_dataset(candles)
    
    print(f"   ✅ {X_momentum.shape[1]} ویژگی مومنتوم استخراج شد")
    print(f"   ✅ {X_momentum.shape[0]} نمونه آماده")
    
    print("\n🤖 آموزش مدل مومنتوم...")
    momentum_learner = MultiHorizonWeightLearner(
        horizons=['3d', '7d', '30d'],
        test_size=0.2,
        random_state=42
    )
    momentum_learner.train(X_momentum, Y_momentum, verbose=False)
    print("   ✅ مدل مومنتوم آموزش داده شد")
    
    print("\n📊 تحلیل مومنتوم...")
    momentum_analyzer = MultiHorizonMomentumAnalyzer(momentum_learner)
    momentum_features = X_momentum.iloc[-1].to_dict()
    momentum_analysis = momentum_analyzer.analyze(momentum_features)
    
    print("\n📈 نتایج تحلیل MOMENTUM:")
    print(f"   3d:  امتیاز = {momentum_analysis.momentum_3d.score:+.3f}, اعتماد = {momentum_analysis.momentum_3d.confidence:.0%}")
    print(f"   7d:  امتیاز = {momentum_analysis.momentum_7d.score:+.3f}, اعتماد = {momentum_analysis.momentum_7d.confidence:.0%}")
    print(f"   30d: امتیاز = {momentum_analysis.momentum_30d.score:+.3f}, اعتماد = {momentum_analysis.momentum_30d.confidence:.0%}")
    print(f"\n   💡 توصیه 3d: {momentum_analysis.recommendation_3d}")
    print(f"   💡 توصیه 7d: {momentum_analysis.recommendation_7d}")
    print(f"   💡 توصیه 30d: {momentum_analysis.recommendation_30d}")
    
    # ════════════════════════════════════════════════════════
    # 3️⃣ ترکیب هوشمند (اختیاری)
    # ════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("3️⃣  ترکیب هوشمند TREND + MOMENTUM (اختیاری)")
    print("="*80)
    
    print("\n🧠 ترکیب دو تحلیل...")
    combined_analyzer = CombinedTrendMomentumAnalyzer(
        trend_analyzer=trend_analyzer,
        momentum_analyzer=momentum_analyzer,
        trend_weight=0.5,
        momentum_weight=0.5
    )
    
    combined_analysis = combined_analyzer.analyze(trend_features, momentum_features)
    
    print("\n📊 نتایج ترکیبی:")
    print(f"   3d:  Trend={trend_analysis.trend_3d.score:+.3f}, Momentum={momentum_analysis.momentum_3d.score:+.3f} → Combined={combined_analysis.combined_score_3d:+.3f}")
    print(f"   7d:  Trend={trend_analysis.trend_7d.score:+.3f}, Momentum={momentum_analysis.momentum_7d.score:+.3f} → Combined={combined_analysis.combined_score_7d:+.3f}")
    print(f"   30d: Trend={trend_analysis.trend_30d.score:+.3f}, Momentum={momentum_analysis.momentum_30d.score:+.3f} → Combined={combined_analysis.combined_score_30d:+.3f}")
    
    print(f"\n   🎯 توصیه نهایی: {combined_analysis.final_action.value}")
    print(f"   🎯 اعتماد نهایی: {combined_analysis.final_confidence:.0%}")
    
    # ════════════════════════════════════════════════════════
    # 📋 خلاصه
    # ════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("📋 خلاصه")
    print("="*80)
    
    print("\n✅ دو تحلیل کاملاً مستقل:")
    print("   1. تحلیل TREND: بر اساس 10 اندیکاتور روند")
    print("      → امتیاز روند برای 3 افق (3d, 7d, 30d)")
    print()
    print("   2. تحلیل MOMENTUM: بر اساس 8 اندیکاتور مومنتوم + Divergence")
    print("      → امتیاز مومنتوم برای 3 افق (3d, 7d, 30d)")
    print()
    print("   3. ترکیب (اختیاری): برای تصمیم‌گیری نهایی")
    print("      → امتیاز ترکیبی برای 3 افق")
    
    print("\n✅ هر تحلیل می‌تواند به صورت جداگانه استفاده شود:")
    print("   - فقط TREND → برای تحلیل روند بلندمدت")
    print("   - فقط MOMENTUM → برای نقاط ورود/خروج کوتاه‌مدت")
    print("   - ترکیب → برای تصمیم‌گیری جامع")
    
    print("\n" + "="*80)
    print("✅ مثال به پایان رسید")
    print("="*80)


if __name__ == '__main__':
    main()
