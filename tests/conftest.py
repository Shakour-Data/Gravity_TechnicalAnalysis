"""
Test Configuration and Fixtures

Global pytest configuration and shared fixtures.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import pytest
from datetime import datetime, timedelta
from typing import List


@pytest.fixture
def sample_candles():
    """Create sample candle data for testing"""
    from src.core.domain.entities import Candle
    
    candles = []
    base_price = 40000
    base_time = datetime.now() - timedelta(days=100)
    
    for i in range(2000):  # Increased from 100 to 2000 candles
        open_price = base_price + (i * 10)
        close_price = open_price + ((i % 10) - 5) * 50
        high_price = max(open_price, close_price) + 100
        low_price = min(open_price, close_price) - 100
        
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=1000 + (i * 10),
            symbol="BTCUSDT",
            timeframe="1h"
        ))
    
    return candles


@pytest.fixture
def trend_learner(sample_candles):
    """Create and train trend weight learner."""
    from gravity_tech.ml.multi_horizon_feature_extraction import MultiHorizonFeatureExtractor
    from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner
    
    # Extract features
    extractor = MultiHorizonFeatureExtractor(horizons=['3d', '7d', '30d'])
    X, y = extractor.extract_training_dataset(sample_candles)
    
    # Create and train learner
    learner = MultiHorizonWeightLearner()
    learner.train(X, y)
    
    return learner


@pytest.fixture
def momentum_learner(sample_candles):
    """Create and train momentum weight learner."""
    from gravity_tech.ml.multi_horizon_momentum_features import MultiHorizonMomentumFeatureExtractor
    from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner
    
    # Extract features
    extractor = MultiHorizonMomentumFeatureExtractor(horizons=['3d', '7d', '30d'])
    X, y = extractor.extract_training_dataset(sample_candles)
    
    # Create and train learner
    learner = MultiHorizonWeightLearner()
    learner.train(X, y)
    
    return learner


@pytest.fixture
def uptrend_candles():
    """Create uptrend candle data"""
    from src.core.domain.entities import Candle
    
    candles = []
    base_price = 40000
    base_time = datetime.now() - timedelta(days=100)
    
    for i in range(100):
        open_price = base_price + (i * 100)  # Clear uptrend
        close_price = open_price + 80
        high_price = close_price + 50
        low_price = open_price - 30
        
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=1000 + (i * 10)
        ))
    
    return candles


@pytest.fixture
def downtrend_candles():
    """Create downtrend candle data"""
    from src.core.domain.entities import Candle
    
    candles = []
    base_price = 50000
    base_time = datetime.now() - timedelta(days=100)
    
    for i in range(100):
        open_price = base_price - (i * 100)  # Clear downtrend
        close_price = open_price - 80
        high_price = open_price + 30
        low_price = close_price - 50
        
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=1000 + (i * 10)
        ))
    
    return candles


@pytest.fixture
def volatile_candles():
    """Create volatile candle data"""
    from src.core.domain.entities import Candle
    import random
    
    random.seed(42)
    candles = []
    base_price = 40000
    base_time = datetime.now() - timedelta(days=100)
    
    for i in range(100):
        volatility = random.uniform(-500, 500)
        open_price = base_price + volatility
        close_price = open_price + random.uniform(-300, 300)
        high_price = max(open_price, close_price) + random.uniform(100, 500)
        low_price = min(open_price, close_price) - random.uniform(100, 500)
        
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=1000 + random.randint(0, 500)
        ))
    
    return candles


@pytest.fixture
def minimal_candles():
    """Create minimal candle data (14 candles for RSI, etc.)"""
    from src.core.domain.entities import Candle
    
    candles = []
    base_time = datetime(2024, 1, 1)
    
    for i in range(14):
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=100.0 + i,
            high=102.0 + i,
            low=99.0 + i,
            close=101.0 + i,
            volume=1000000
        ))
    
    return candles


@pytest.fixture
def insufficient_candles():
    """Create insufficient candle data (too few for most indicators)"""
    from src.core.domain.entities import Candle
    
    candles = []
    base_time = datetime(2024, 1, 1)
    
    for i in range(5):
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000000
        ))
    
    return candles


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "ml: marks tests requiring ML libraries"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance benchmarking tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add 'integration' marker to tests in integration/ folder
        if 'integration' in str(item.fspath):
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def trend_learner():
    """Create a MultiHorizonWeightLearner for trend analysis."""
    from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner
    return MultiHorizonWeightLearner(horizons=['3d', '7d', '30d'])


@pytest.fixture
def momentum_learner():
    """Create a MultiHorizonWeightLearner for momentum analysis."""
    from gravity_tech.ml.multi_horizon_weights import MultiHorizonWeightLearner
    return MultiHorizonWeightLearner(horizons=['3d', '7d', '30d'])


@pytest.fixture
def candles(sample_candles):
    """Provide candles fixture for tests."""
    return sample_candles
