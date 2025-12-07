"""
Fast Integration Tests for Multi-Horizon System

Uses real data but optimized for speed:
- Smaller datasets (100 samples vs 1500)
- Pre-computed fixtures
- Minimal horizons (3d only)
- Mock objects where appropriate

Author: Gravity Tech Team
Date: 2024
Version: 1.0
License: MIT
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gravity_tech.core.domain.entities import Candle
from gravity_tech.ml.combined_trend_momentum_analysis import CombinedTrendMomentumAnalyzer
from gravity_tech.ml.multi_horizon_analysis import MultiHorizonTrendAnalyzer
from gravity_tech.ml.multi_horizon_feature_extraction import MultiHorizonFeatureExtractor
from gravity_tech.ml.multi_horizon_momentum_analysis import MultiHorizonMomentumAnalyzer
from gravity_tech.ml.multi_horizon_momentum_features import MultiHorizonMomentumFeatureExtractor
from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))


def create_realistic_market_data(num_samples: int = 100, trend: str = 'mixed') -> pd.DataFrame:
    """Create minimal realistic market data for fast testing."""
    np.random.seed(42)

    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_samples, freq='1h')
    base_price = 30000.0
    prices = [base_price]
    volumes = []

    for _ in range(1, num_samples):
        if trend == 'uptrend':
            drift = 0.001  # Smaller drift for stability
        elif trend == 'downtrend':
            drift = -0.001
        else:
            drift = 0.0005 * np.random.choice([-1, 1])

        # Add volatility
        volatility = 0.005
        shock = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + drift + shock)

        prices.append(max(new_price, 1000.0))  # Floor price
        volumes.append(np.random.randint(100, 1000))

    # Create OHLC from close prices
    df_data = []
    for i, (timestamp, close) in enumerate(zip(dates, prices, strict=True)):
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1]

        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.002)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.002)))

        df_data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volumes[i] if i < len(volumes) else 100
        })

    return pd.DataFrame(df_data)


@pytest.fixture(scope="session")
def fast_candles():
    """Fast fixture with minimal realistic data."""
    df = create_realistic_market_data(num_samples=500, trend='uptrend')  # Increased to 500 samples for all horizons

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


@pytest.fixture(scope="session")
def fast_trained_models(fast_candles):
    """Pre-trained models for fast testing."""
    # Use all horizons but with smaller data for speed
    horizons = ['3d', '7d', '30d']

    # Extract features
    trend_extractor = MultiHorizonFeatureExtractor(horizons=horizons)
    X_trend, Y_trend = trend_extractor.extract_training_dataset(fast_candles)

    momentum_extractor = MultiHorizonMomentumFeatureExtractor(horizons=horizons)
    X_momentum, Y_momentum = momentum_extractor.extract_training_dataset(fast_candles)

    # Train models
    trend_learner = MultiHorizonWeightLearner(
        horizons=horizons,
        test_size=0.3,  # Smaller training set
        random_state=42
    )
    trend_learner.train(X_trend, Y_trend, verbose=False)

    momentum_learner = MultiHorizonWeightLearner(
        horizons=horizons,
        test_size=0.3,
        random_state=42
    )
    momentum_learner.train(X_momentum, Y_momentum, verbose=False)

    return trend_learner, momentum_learner


def test_trend_analysis_fast(fast_candles, fast_trained_models):
    """Fast test for trend analysis."""
    trend_learner, _ = fast_trained_models

    # Create analyzer
    trend_analyzer = MultiHorizonTrendAnalyzer(trend_learner)

    # Get latest features
    trend_extractor = MultiHorizonFeatureExtractor(horizons=['3d'])
    X_trend, _ = trend_extractor.extract_training_dataset(fast_candles)
    latest_features = X_trend.iloc[-1].to_dict()

    # Analyze
    analysis = trend_analyzer.analyze(latest_features)

    # Basic assertions - check that we have the expected structure
    assert hasattr(analysis, 'score_3d')
    assert hasattr(analysis, 'score_7d')
    assert hasattr(analysis, 'score_30d')
    assert hasattr(analysis, 'pattern')
    assert hasattr(analysis, 'combined_score')

    # Check 3d score specifically
    assert analysis.score_3d.score is not None
    assert -1.0 <= analysis.score_3d.score <= 1.0
    assert 0.0 <= analysis.score_3d.confidence <= 1.0


def test_momentum_analysis_fast(fast_candles, fast_trained_models):
    """Fast test for momentum analysis."""
    _, momentum_learner = fast_trained_models

    # Create analyzer
    momentum_analyzer = MultiHorizonMomentumAnalyzer(momentum_learner)

    # Get latest features
    momentum_extractor = MultiHorizonMomentumFeatureExtractor(horizons=['3d'])
    X_momentum, _ = momentum_extractor.extract_training_dataset(fast_candles)
    latest_features = X_momentum.iloc[-1].to_dict()

    # Analyze
    analysis = momentum_analyzer.analyze(latest_features)

    # Basic assertions - check that we have the expected structure
    assert hasattr(analysis, 'momentum_3d')
    assert hasattr(analysis, 'momentum_7d')
    assert hasattr(analysis, 'momentum_30d')

    # Check 3d momentum specifically
    assert analysis.momentum_3d.score is not None
    assert -1.0 <= analysis.momentum_3d.score <= 1.0
    assert 0.0 <= analysis.momentum_3d.confidence <= 1.0


def test_combined_analysis_fast(fast_candles, fast_trained_models):
    """Fast test for combined trend+momentum analysis."""
    trend_learner, momentum_learner = fast_trained_models

    # Create analyzers
    trend_analyzer = MultiHorizonTrendAnalyzer(trend_learner)
    momentum_analyzer = MultiHorizonMomentumAnalyzer(momentum_learner)
    combined_analyzer = CombinedTrendMomentumAnalyzer(
        trend_analyzer, momentum_analyzer, 0.6, 0.4
    )

    # Get latest features
    trend_extractor = MultiHorizonFeatureExtractor(horizons=['3d'])
    momentum_extractor = MultiHorizonMomentumFeatureExtractor(horizons=['3d'])

    X_trend, _ = trend_extractor.extract_training_dataset(fast_candles)
    X_momentum, _ = momentum_extractor.extract_training_dataset(fast_candles)

    trend_features = X_trend.iloc[-1].to_dict()
    momentum_features = X_momentum.iloc[-1].to_dict()

    # Analyze
    analysis = combined_analyzer.analyze(trend_features, momentum_features)

    # Basic assertions
    assert hasattr(analysis, 'final_action')
    assert hasattr(analysis, 'final_confidence')
    assert analysis.final_action in ['BUY', 'SELL', 'HOLD']
    assert 0.0 <= analysis.final_confidence <= 1.0


def test_feature_extraction_shapes(fast_candles):
    """Test that feature extraction produces correct shapes."""
    horizons = ['3d']

    trend_extractor = MultiHorizonFeatureExtractor(horizons=horizons)
    X_trend, Y_trend = trend_extractor.extract_training_dataset(fast_candles)

    momentum_extractor = MultiHorizonMomentumFeatureExtractor(horizons=horizons)
    X_momentum, Y_momentum = momentum_extractor.extract_training_dataset(fast_candles)

    # Check shapes
    assert X_trend.shape[0] > 0
    assert X_momentum.shape[0] > 0
    assert Y_trend.shape[0] > 0
    assert Y_momentum.shape[0] > 0

    # Check that we have features for each horizon
    expected_features = len(horizons) * 2  # trend + momentum per horizon
    assert X_trend.shape[1] >= expected_features


def test_model_training_convergence(fast_trained_models):
    """Test that models train and produce reasonable results."""
    trend_learner, momentum_learner = fast_trained_models

    # Check that models have been trained
    assert hasattr(trend_learner, 'models')
    assert hasattr(momentum_learner, 'models')
    assert len(trend_learner.models) > 0
    assert len(momentum_learner.models) > 0


@pytest.mark.parametrize("trend_type", ["uptrend", "downtrend", "mixed"])
def test_different_market_conditions(trend_type):
    """Test analysis under different market conditions."""
    # Create small dataset for specific trend
    df = create_realistic_market_data(num_samples=30, trend=trend_type)

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

    # Quick training
    horizons = ['3d']
    trend_extractor = MultiHorizonFeatureExtractor(horizons=horizons)
    X_trend, Y_trend = trend_extractor.extract_training_dataset(candles)

    trend_learner = MultiHorizonWeightLearner(horizons=horizons, test_size=0.5, random_state=42)
    trend_learner.train(X_trend, Y_trend, verbose=False)

    # Analysis
    trend_analyzer = MultiHorizonTrendAnalyzer(trend_learner)
    latest_features = X_trend.iloc[-1].to_dict()
    analysis = trend_analyzer.analyze(latest_features)

    # Should produce valid results regardless of market condition
    assert analysis.score_3d.score is not None
    assert -1.0 <= analysis.score_3d.score <= 1.0
    assert 0.0 <= analysis.score_3d.confidence <= 1.0
