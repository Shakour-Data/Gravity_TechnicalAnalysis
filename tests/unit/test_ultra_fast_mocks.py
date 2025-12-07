"""
Ultra-Fast Unit Tests for Multi-Horizon System

Uses mock objects and minimal data for sub-second execution:
- Mock trained models
- Pre-computed features
- No actual ML training
- Focus on integration logic

Author: Gravity Tech Team
Date: 2024
Version: 1.0
License: MIT
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from gravity_tech.core.domain.entities import Candle
from gravity_tech.ml.combined_trend_momentum_analysis import CombinedTrendMomentumAnalyzer
from gravity_tech.ml.multi_horizon_analysis import MultiHorizonTrendAnalyzer
from gravity_tech.ml.multi_horizon_momentum_analysis import MultiHorizonMomentumAnalyzer
from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))


@pytest.fixture(scope="session")
def mock_candles():
    """Mock candles for ultra-fast testing."""
    candles = []
    base_time = pd.Timestamp.now()

    for i in range(10):  # Minimal data
        candle = Candle(
            timestamp=base_time + pd.Timedelta(hours=i),
            open=30000.0 + i,
            high=30010.0 + i,
            low=29990.0 + i,
            close=30005.0 + i,
            volume=1000 + i * 10,
            symbol='TEST',
            timeframe='1h'
        )
        candles.append(candle)

    return candles


@pytest.fixture(scope="session")
def mock_trained_models():
    """Mock trained models for ultra-fast testing."""
    # Mock MultiHorizonWeightLearner
    mock_trend_learner = Mock(spec=MultiHorizonWeightLearner)
    mock_momentum_learner = Mock(spec=MultiHorizonWeightLearner)

    # Mock horizons
    mock_trend_learner.horizons = ['3d', '7d', '30d']
    mock_momentum_learner.horizons = ['3d', '7d', '30d']

    # Mock predict_multi_horizon method
    def mock_predict_trend(X):
        # Return mock predictions for all horizons
        return pd.DataFrame({
            'pred_3d': [0.02] * len(X),  # 2% return
            'pred_7d': [0.01] * len(X),  # 1% return
            'pred_30d': [-0.005] * len(X)  # -0.5% return
        })

    def mock_predict_momentum(X):
        return pd.DataFrame({
            'pred_3d': [0.015] * len(X),  # Positive momentum
            'pred_7d': [0.008] * len(X),
            'pred_30d': [-0.002] * len(X)
        })

    mock_trend_learner.predict_multi_horizon = mock_predict_trend
    mock_momentum_learner.predict_multi_horizon = mock_predict_momentum

    # Mock get_horizon_weights method
    def mock_get_weights(horizon):
        mock_weights = Mock()
        mock_weights.confidence = 0.8
        return mock_weights

    mock_trend_learner.get_horizon_weights = mock_get_weights
    mock_momentum_learner.get_horizon_weights = mock_get_weights

    return mock_trend_learner, mock_momentum_learner


@pytest.fixture(scope="session")
def mock_analyzers(mock_trained_models):
    """Mock analyzers for testing."""
    trend_learner, momentum_learner = mock_trained_models

    trend_analyzer = MultiHorizonTrendAnalyzer(trend_learner)
    momentum_analyzer = MultiHorizonMomentumAnalyzer(momentum_learner)

    return trend_analyzer, momentum_analyzer


@pytest.fixture(scope="session")
def mock_combined_analyzer(mock_analyzers):
    """Mock combined analyzer with default weights."""
    trend_analyzer, momentum_analyzer = mock_analyzers
    return CombinedTrendMomentumAnalyzer(trend_analyzer, momentum_analyzer, 0.5, 0.5)


@pytest.fixture(scope="session")
def mock_features():
    """Mock features for testing."""
    return {
        'sma_20': 29950.0,
        'sma_50': 29800.0,
        'rsi': 65.0,
        'macd': 25.0,
        'macd_signal': 20.0,
        'bb_upper': 30200.0,
        'bb_lower': 29700.0,
        'stoch_k': 75.0,
        'stoch_d': 70.0,
        'volume_sma': 1500.0,
        'price_change': 0.005,
        'volatility': 0.02,
        'trend_strength': 0.7,
        'momentum': 0.6,
        'support_level': 29500.0,
        'resistance_level': 30500.0,
        'fib_236': 29750.0,
        'fib_382': 29900.0,
        'fib_618': 30100.0,
        'pivot_point': 30000.0,
        'r1': 30100.0,
        's1': 29900.0
    }


def test_mock_trend_analysis_ultra_fast(mock_trained_models, mock_features):
    """Ultra-fast test for trend analysis using mocks."""
    trend_learner, _ = mock_trained_models

    # Create analyzer
    analyzer = MultiHorizonTrendAnalyzer(trend_learner)

    # Analyze with mock features
    analysis = analyzer.analyze(mock_features)

    # Basic assertions
    assert hasattr(analysis, 'score_3d')
    assert hasattr(analysis, 'score_7d')
    assert hasattr(analysis, 'score_30d')
    assert hasattr(analysis, 'pattern')
    assert hasattr(analysis, 'combined_score')

    # Check scores are reasonable
    assert -1.0 <= analysis.score_3d.score <= 1.0
    assert -1.0 <= analysis.score_7d.score <= 1.0
    assert -1.0 <= analysis.score_30d.score <= 1.0
    assert 0.0 <= analysis.score_3d.confidence <= 1.0


def test_mock_momentum_analysis_ultra_fast(mock_trained_models, mock_features):
    """Ultra-fast test for momentum analysis using mocks."""
    _, momentum_learner = mock_trained_models

    # Create analyzer
    analyzer = MultiHorizonMomentumAnalyzer(momentum_learner)

    # Analyze with mock features
    analysis = analyzer.analyze(mock_features)

    # Basic assertions
    assert hasattr(analysis, 'momentum_3d')
    assert hasattr(analysis, 'momentum_7d')
    assert hasattr(analysis, 'momentum_30d')

    # Check scores are reasonable
    assert -1.0 <= analysis.momentum_3d.score <= 1.0
    assert -1.0 <= analysis.momentum_7d.score <= 1.0
    assert -1.0 <= analysis.momentum_30d.score <= 1.0
    assert 0.0 <= analysis.momentum_3d.confidence <= 1.0


def test_mock_combined_analysis_ultra_fast(mock_trained_models, mock_features):
    """Ultra-fast test for combined analysis using mocks."""
    trend_learner, momentum_learner = mock_trained_models

    # Create analyzers
    trend_analyzer = MultiHorizonTrendAnalyzer(trend_learner)
    momentum_analyzer = MultiHorizonMomentumAnalyzer(momentum_learner)
    combined_analyzer = CombinedTrendMomentumAnalyzer(
        trend_analyzer, momentum_analyzer, 0.6, 0.4
    )

    # Analyze
    analysis = combined_analyzer.analyze(mock_features, mock_features)

    # Basic assertions
    assert hasattr(analysis, 'final_action')
    assert hasattr(analysis, 'final_confidence')
    assert analysis.final_action.value in ['BUY', 'SELL', 'HOLD']  # Use enum value
    assert 0.0 <= analysis.final_confidence <= 1.0


def test_mock_feature_shapes(mock_candles):
    """Test that mock data has correct structure."""
    assert len(mock_candles) > 0
    assert all(isinstance(c, Candle) for c in mock_candles)

    # Check first candle has required attributes
    candle = mock_candles[0]
    assert hasattr(candle, 'timestamp')
    assert hasattr(candle, 'open')
    assert hasattr(candle, 'high')
    assert hasattr(candle, 'low')
    assert hasattr(candle, 'close')
    assert hasattr(candle, 'volume')


def test_mock_model_structure(mock_trained_models):
    """Test that mock models have expected structure."""
    trend_learner, momentum_learner = mock_trained_models

    assert hasattr(trend_learner, 'horizons')
    assert hasattr(momentum_learner, 'horizons')
    assert len(trend_learner.horizons) == 3
    assert len(momentum_learner.horizons) == 3
    assert '3d' in trend_learner.horizons
    assert '7d' in trend_learner.horizons
    assert '30d' in trend_learner.horizons


@pytest.mark.parametrize("trend_weight,momentum_weight", [
    (0.5, 0.5),
    (0.7, 0.3),
    (0.3, 0.7),
    (1.0, 0.0),
    (0.0, 1.0)
])
def test_combined_weights_variations(mock_trained_models, mock_features, trend_weight, momentum_weight):
    """Test combined analysis with different weight combinations."""
    trend_learner, momentum_learner = mock_trained_models

    trend_analyzer = MultiHorizonTrendAnalyzer(trend_learner)
    momentum_analyzer = MultiHorizonMomentumAnalyzer(momentum_learner)
    combined_analyzer = CombinedTrendMomentumAnalyzer(
        trend_analyzer, momentum_analyzer, trend_weight, momentum_weight
    )

    analysis = combined_analyzer.analyze(mock_features, mock_features)

    assert analysis.final_action.value in ['BUY', 'SELL', 'HOLD']  # Use enum value
    assert 0.0 <= analysis.final_confidence <= 1.0


def test_edge_case_all_zeros(mock_trained_models):
    """Test analysis with zero features (edge case)."""
    trend_learner, momentum_learner = mock_trained_models

    # Zero features
    zero_features = {f'feature_{k}': 0.0 for k in range(21)}  # 21 features

    trend_analyzer = MultiHorizonTrendAnalyzer(trend_learner)
    analysis = trend_analyzer.analyze(zero_features)

    # Should still produce valid results
    assert hasattr(analysis, 'score_3d')
    assert isinstance(analysis.score_3d.score, int | float)
    assert isinstance(analysis.score_3d.confidence, int | float)


def test_edge_case_extreme_values(mock_trained_models):
    """Test analysis with extreme feature values."""
    trend_learner, momentum_learner = mock_trained_models

    # Extreme features
    extreme_features = {f'feature_{k}': 1000000.0 if k % 2 == 0 else -1000000.0 for k in range(21)}

    trend_analyzer = MultiHorizonTrendAnalyzer(trend_learner)
    analysis = trend_analyzer.analyze(extreme_features)
    # Should still produce valid results (clipped to reasonable ranges)
    assert hasattr(analysis, 'score_3d')
    assert -1.0 <= analysis.score_3d.score <= 1.0
    assert 0.0 <= analysis.score_3d.confidence <= 1.0


def test_performance_ultra_fast(mock_combined_analyzer, mock_features):
    """Performance test to ensure ultra-fast execution."""
    import time

    start_time = time.time()
    analysis = mock_combined_analyzer.analyze(mock_features, mock_features)
    end_time = time.time()

    execution_time = end_time - start_time

    # Should complete in under 0.1 seconds
    assert execution_time < 0.1, f"Test took {execution_time:.3f}s, should be < 0.1s"
    assert analysis.final_action.value in ['BUY', 'SELL', 'HOLD']
