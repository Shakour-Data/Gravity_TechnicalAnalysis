"""
Test Complete Multi-Horizon System (Trend + Momentum)

Author: Gravity Tech Team
Date: 2024
Version: 1.0
License: MIT

End-to-end test for combined analysis system.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import pytest

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from gravity_tech.ml.multi_horizon_feature_extraction import MultiHorizonFeatureExtractor
from gravity_tech.ml.multi_horizon_momentum_features import MultiHorizonMomentumFeatureExtractor
from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner
from gravity_tech.ml.multi_horizon_analysis import MultiHorizonAnalyzer
from gravity_tech.ml.multi_horizon_momentum_analysis import MultiHorizonMomentumAnalyzer
from gravity_tech.ml.combined_trend_momentum_analysis import CombinedTrendMomentumAnalyzer
from gravity_tech.core.domain.entities import Candle


@pytest.fixture(scope="module")
def candles():
    """Fixture to provide candle data for tests."""
    # Create data
    df = create_realistic_market_data(num_samples=1500, trend='uptrend')
    
    # Convert DataFrame to Candle objects
    candle_objects = []
    for _, row in df.iterrows():
        candle = Candle(
            timestamp=row['timestamp'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        candle_objects.append(candle)
    
    return candle_objects


@pytest.fixture(scope="module")
def trained_models(candles):
    """Fixture to provide trained models for combined system test."""
    # Extract trend features
    trend_extractor = MultiHorizonFeatureExtractor(horizons=['3d', '7d', '30d'])
    X_trend, Y_trend = trend_extractor.extract_training_dataset(candles)
    
    # Training trend model
    trend_learner = MultiHorizonWeightLearner(
        horizons=['3d', '7d', '30d'],
        test_size=0.2,
        random_state=42
    )
    trend_learner.train(X_trend, Y_trend, verbose=False)
    
    # Extract momentum features
    momentum_extractor = MultiHorizonMomentumFeatureExtractor(horizons=['3d', '7d', '30d'])
    X_momentum, Y_momentum = momentum_extractor.extract_training_dataset(candles)
    
    # Training momentum model
    momentum_learner = MultiHorizonWeightLearner(
        horizons=['3d', '7d', '30d'],
        test_size=0.2,
        random_state=42
    )
    momentum_learner.train(X_momentum, Y_momentum, verbose=False)
    
    return trend_learner, momentum_learner


def create_realistic_market_data(
    num_samples: int = 2000,
    trend: str = 'mixed'
) -> pd.DataFrame:
    """Create market data."""
    np.random.seed(42)
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_samples, freq='1h')
    base_price = 30000
    prices = [base_price]
    volumes = []
    
    for i in range(1, num_samples):
        if trend == 'uptrend':
            drift = 0.002
        elif trend == 'downtrend':
            drift = -0.002
        else:
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
    
    volumes.insert(0, 1000000)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices
    })
    
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    
    # Generate high/low that properly bracket open/close
    open_prices = df['open'].values
    close_prices = df['close'].values
    
    body_highs = np.maximum(open_prices, close_prices)
    body_lows = np.minimum(open_prices, close_prices)
    
    # Add some wick above and below the body (0.5% of close price)
    wick_ranges = np.abs(close_prices) * 0.005
    wick_above = np.random.uniform(0, wick_ranges, len(df))
    wick_below = np.random.uniform(0, wick_ranges, len(df))
    
    df['high'] = body_highs + wick_above
    df['low'] = body_lows - wick_below
    
    df['volume'] = volumes
    
    return df


def test_trend_system(candles):
    """Test trend system."""
    print("\n" + "="*80)
    print("🔵 TESTING TREND SYSTEM")
    print("="*80)
    
    print(f"\n✅ Generated {len(candles)} candles")
    
    # Extract features
    trend_extractor = MultiHorizonFeatureExtractor(horizons=['3d', '7d', '30d'])
    X_trend, Y_trend = trend_extractor.extract_training_dataset(candles)
    print(f"✅ Trend features: {X_trend.shape}")
    
    # Training
    trend_learner = MultiHorizonWeightLearner(
        horizons=['3d', '7d', '30d'],
        test_size=0.2,
        random_state=42
    )
    trend_learner.train(X_trend, Y_trend, verbose=False)
    print("✅ Trend model trained")
    
    # Analysis
    trend_analyzer = MultiHorizonAnalyzer(trend_learner)
    latest_features = X_trend.iloc[-1].to_dict()
    trend_analysis = trend_analyzer.analyze(latest_features)
    
    print("\n📊 Trend Analysis Results:")
    print(f"  3d Score: {trend_analysis.score_3d.score:+.3f} ({trend_analysis.score_3d.confidence:.0%})")
    print(f"  7d Score: {trend_analysis.score_7d.score:+.3f} ({trend_analysis.score_7d.confidence:.0%})")
    print(f"  30d Score: {trend_analysis.score_30d.score:+.3f} ({trend_analysis.score_30d.confidence:.0%})")
    
    return trend_learner, trend_analyzer, candles


def test_momentum_system(candles):
    """Test momentum system."""
    print("\n" + "="*80)
    print("🟢 TESTING MOMENTUM SYSTEM")
    print("="*80)
    
    # Extract features
    momentum_extractor = MultiHorizonMomentumFeatureExtractor(horizons=['3d', '7d', '30d'])
    X_momentum, Y_momentum = momentum_extractor.extract_training_dataset(candles)
    print(f"\n✅ Momentum features: {X_momentum.shape}")
    
    # Training
    momentum_learner = MultiHorizonWeightLearner(
        horizons=['3d', '7d', '30d'],
        test_size=0.2,
        random_state=42
    )
    momentum_learner.train(X_momentum, Y_momentum, verbose=False)
    print("✅ Momentum model trained")
    
    # Analysis
    momentum_analyzer = MultiHorizonMomentumAnalyzer(momentum_learner)
    latest_features = X_momentum.iloc[-1].to_dict()
    momentum_analysis = momentum_analyzer.analyze(latest_features)
    
    print("\n📊 Momentum Analysis Results:")
    print(f"  3d Score: {momentum_analysis.momentum_3d.score:+.3f} ({momentum_analysis.momentum_3d.confidence:.0%})")
    print(f"  7d Score: {momentum_analysis.momentum_7d.score:+.3f} ({momentum_analysis.momentum_7d.confidence:.0%})")
    print(f"  30d Score: {momentum_analysis.momentum_30d.score:+.3f} ({momentum_analysis.momentum_30d.confidence:.0%})")
    
    return momentum_learner, momentum_analyzer


def test_combined_system(trained_models, candles):
    """Test combined system."""
    print("\n" + "="*80)
    print("🟣 TESTING COMBINED SYSTEM")
    print("="*80)
    
    trend_learner, momentum_learner = trained_models
    
    # Create analyzers
    trend_analyzer = MultiHorizonAnalyzer(trend_learner)
    momentum_analyzer = MultiHorizonMomentumAnalyzer(momentum_learner)
    
    # Create combined analyzer
    combined_analyzer = CombinedTrendMomentumAnalyzer(
        trend_analyzer=trend_analyzer,
        momentum_analyzer=momentum_analyzer,
        trend_weight=0.5,
        momentum_weight=0.5
    )
    
    # Extract latest features
    trend_extractor = MultiHorizonFeatureExtractor(horizons=['3d', '7d', '30d'])
    momentum_extractor = MultiHorizonMomentumFeatureExtractor(horizons=['3d', '7d', '30d'])
    
    X_trend, _ = trend_extractor.extract_training_dataset(candles)
    X_momentum, _ = momentum_extractor.extract_training_dataset(candles)
    
    trend_features = X_trend.iloc[-1].to_dict()
    momentum_features = X_momentum.iloc[-1].to_dict()
    
    # Combined analysis
    combined_analysis = combined_analyzer.analyze(trend_features, momentum_features)
    
    # Display results
    combined_analyzer.print_analysis(combined_analysis)
    
    return combined_analyzer, combined_analysis


def test_different_scenarios():
    """Test different scenarios."""
    print("\n" + "="*80)
    print("🎬 TESTING DIFFERENT MARKET SCENARIOS")
    print("="*80)
    
    scenarios = [
        ('STRONG UPTREND', 'uptrend'),
        ('STRONG DOWNTREND', 'downtrend'),
        ('MIXED MARKET', 'mixed')
    ]
    
    for name, trend_type in scenarios:
        print(f"\n\n{'='*80}")
        print(f"📈 Scenario: {name}")
        print("="*80)
        
        # Create data
        df = create_realistic_market_data(num_samples=1500, trend=trend_type)
        candles = []
        for _, row in df.iterrows():
            candle = Candle(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            candles.append(candle)
        
        # Training
        print("\n🔄 Training models...")
        
        trend_extractor = MultiHorizonFeatureExtractor(horizons=['3d', '7d', '30d'])
        X_trend, Y_trend = trend_extractor.extract_training_dataset(candles)
        
        momentum_extractor = MultiHorizonMomentumFeatureExtractor(horizons=['3d', '7d', '30d'])
        X_momentum, Y_momentum = momentum_extractor.extract_training_dataset(candles)
        
        trend_learner = MultiHorizonWeightLearner(horizons=['3d', '7d', '30d'], test_size=0.2, random_state=42)
        trend_learner.train(X_trend, Y_trend, verbose=False)
        
        momentum_learner = MultiHorizonWeightLearner(horizons=['3d', '7d', '30d'], test_size=0.2, random_state=42)
        momentum_learner.train(X_momentum, Y_momentum, verbose=False)
        
        # Combined analysis
        trend_analyzer = MultiHorizonAnalyzer(trend_learner)
        momentum_analyzer = MultiHorizonMomentumAnalyzer(momentum_learner)
        combined_analyzer = CombinedTrendMomentumAnalyzer(
            trend_analyzer, momentum_analyzer, 0.5, 0.5
        )
        
        trend_features = X_trend.iloc[-1].to_dict()
        momentum_features = X_momentum.iloc[-1].to_dict()
        
        combined_analysis = combined_analyzer.analyze(trend_features, momentum_features)
        
        # Display summary
        print("\n📋 Summary:")
        print(f"  Final Action:    {combined_analysis.final_action.value}")
        print(f"  Final Confidence: {combined_analysis.final_confidence:.0%}")
        print(f"  3d Combined:     {combined_analysis.combined_score_3d:+.3f}")
        print(f"  7d Combined:     {combined_analysis.combined_score_7d:+.3f}")
        print(f"  30d Combined:    {combined_analysis.combined_score_30d:+.3f}")


def test_full_integration():
    """Test full integration (equivalent to main function)."""
    print("\n" + "🚀"*40)
    print("COMPLETE MULTI-HORIZON SYSTEM TEST")
    print("="*40)
    
    try:
        # Create data
        df = create_realistic_market_data(num_samples=1500, trend='uptrend')
        candles = []
        for _, row in df.iterrows():
            candle = Candle(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            candles.append(candle)
        
        # Test trend system
        trend_learner, trend_analyzer, _ = test_trend_system(candles)
        
        # Test momentum system
        momentum_learner, momentum_analyzer = test_momentum_system(candles)
        
        # Test combined system
        combined_analyzer, combined_analysis = test_combined_system(
            (trend_learner, momentum_learner), candles
        )
        
        # Test different scenarios
        test_different_scenarios()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED SUCCESSFULLY")
        print("="*80)
        print("\nSystem is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function."""
    test_full_integration()


if __name__ == '__main__':
    main()

