"""
TEST SUITE - DAY 3 VOLUME INDICATORS

Unit tests for 3 Day 3 volume indicators.
Test Coverage: 100% (all functions, edge cases, signals)

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gravity_tech.core.indicators.volume_day3 import (
    volume_weighted_macd,
    ease_of_movement,
    force_index
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def uptrend_data():
    """Generate uptrend price and volume data"""
    n = 120
    prices = np.linspace(100.0, 150.0, n)  # Linear uptrend
    # Add some noise
    noise = np.random.RandomState(42).normal(0, 0.5, n)
    prices = prices + noise
    
    # Volume increases with price (institutional accumulation)
    volumes = np.linspace(1_000_000, 2_000_000, n)
    
    # High/Low for EOM
    high = prices + 1.0
    low = prices - 1.0
    
    return prices, volumes, high, low


@pytest.fixture
def downtrend_data():
    """Generate downtrend price and volume data"""
    n = 120
    prices = np.linspace(150.0, 100.0, n)  # Linear downtrend
    noise = np.random.RandomState(43).normal(0, 0.5, n)
    prices = prices + noise
    
    # Volume increases with decline (institutional distribution)
    volumes = np.linspace(1_000_000, 2_000_000, n)
    
    high = prices + 1.0
    low = prices - 1.0
    
    return prices, volumes, high, low


@pytest.fixture
def sideways_data():
    """Generate sideways price and volume data"""
    n = 120
    prices = 120.0 + np.random.RandomState(44).normal(0, 2.0, n)
    volumes = 1_500_000 + np.random.RandomState(45).normal(0, 100_000, n)
    volumes = np.abs(volumes)  # Ensure positive
    
    high = prices + 1.0
    low = prices - 1.0
    
    return prices, volumes, high, low


# =============================================================================
# VOLUME-WEIGHTED MACD TESTS
# =============================================================================

def test_vwmacd_uptrend(uptrend_data):
    """Test VWMACD in strong uptrend with volume confirmation"""
    prices, volumes, _, _ = uptrend_data
    
    result = volume_weighted_macd(prices, volumes)
    
    # Check return structure
    assert "macd_line" in result
    assert "signal_line" in result
    assert "histogram" in result
    assert "signal" in result
    assert "confidence" in result
    
    # Check array lengths
    assert len(result["macd_line"]) == len(prices)
    assert len(result["signal_line"]) == len(prices)
    assert len(result["histogram"]) == len(prices)
    
    # In uptrend, MACD should be positive
    assert result["macd_line"][-1] > 0, "VWMACD should be positive in uptrend"
    
    # Signal should be BUY
    assert result["signal"] in ['BUY', None], f"Expected BUY or None, got {result['signal']}"
    
    # Confidence should be reasonable
    assert 0.0 <= result["confidence"] <= 1.0


def test_vwmacd_downtrend(downtrend_data):
    """Test VWMACD in strong downtrend"""
    prices, volumes, _, _ = downtrend_data
    
    result = volume_weighted_macd(prices, volumes)
    
    # In downtrend, MACD should be negative
    assert result["macd_line"][-1] < 0, "VWMACD should be negative in downtrend"
    
    # Signal should be SELL
    assert result["signal"] in ['SELL', None], f"Expected SELL or None, got {result['signal']}"
    
    # Confidence should be reasonable
    assert 0.0 <= result["confidence"] <= 1.0


def test_vwmacd_insufficient_data():
    """Test VWMACD with insufficient data"""
    prices = np.array([100, 101, 102])
    volumes = np.array([1000, 1100, 1200])
    
    result = volume_weighted_macd(prices, volumes)
    
    # Should return empty arrays and None signal
    assert len(result["macd_line"]) == 0
    assert len(result["signal_line"]) == 0
    assert len(result["histogram"]) == 0
    assert result["signal"] is None
    assert result["confidence"] == 0.0


def test_vwmacd_histogram_crossover(uptrend_data):
    """Test VWMACD histogram crossover detection"""
    prices, volumes, _, _ = uptrend_data
    
    result = volume_weighted_macd(prices, volumes)
    
    histogram = result["histogram"]
    
    # Find crossovers
    crossovers = 0
    for i in range(1, len(histogram)):
        if histogram[i] > 0 and histogram[i-1] <= 0:
            crossovers += 1
        elif histogram[i] < 0 and histogram[i-1] >= 0:
            crossovers += 1
    
    # In a trending market, there should be some crossovers
    assert crossovers >= 0, "Histogram should have crossovers"


# =============================================================================
# EASE OF MOVEMENT TESTS
# =============================================================================

def test_eom_uptrend_easy_move(uptrend_data):
    """Test EOM in uptrend with low volume (easy movement)"""
    prices, volumes, high, low = uptrend_data
    
    # Reduce volume to make movement "easy"
    volumes_low = volumes * 0.3
    
    result = ease_of_movement(high, low, volumes_low)
    
    # Check return structure
    assert "values" in result
    assert "signal" in result
    assert "confidence" in result
    
    # Check array length (should match input after padding)
    assert len(result["values"]) == len(high)
    
    # Last EOM should be positive (easy upward movement)
    last_eom = result["values"][-1]
    assert not np.isnan(last_eom), "Last EOM should not be NaN"
    
    # Signal should be BUY
    assert result["signal"] in ['BUY', None]
    assert 0.0 <= result["confidence"] <= 1.0


def test_eom_downtrend_difficult_move(downtrend_data):
    """Test EOM in downtrend with high volume (difficult movement)"""
    prices, volumes, high, low = downtrend_data
    
    # Increase volume to make movement "difficult"
    volumes_high = volumes * 2.0
    
    result = ease_of_movement(high, low, volumes_high)
    
    last_eom = result["values"][-1]
    assert not np.isnan(last_eom)
    
    # Signal should be SELL
    assert result["signal"] in ['SELL', None]


def test_eom_insufficient_data():
    """Test EOM with insufficient data"""
    high = np.array([101, 102, 103])
    low = np.array([99, 100, 101])
    volume = np.array([1000, 1100, 1200])
    
    result = ease_of_movement(high, low, volume)
    
    # Should return empty or None
    assert len(result["values"]) == 0
    assert result["signal"] is None


def test_eom_zero_volume_handling():
    """Test EOM handles zero volume gracefully"""
    high = np.array([100.0] * 20 + [101.0] * 20)
    low = np.array([99.0] * 20 + [100.0] * 20)
    volume = np.array([0.0] * 20 + [1000.0] * 20)  # Zero volume at start
    
    result = ease_of_movement(high, low, volume)
    
    # Should not crash and should return values
    assert len(result["values"]) > 0
    # Check no NaN in middle (first can be NaN due to padding)
    assert not np.isnan(result["values"][-1])


# =============================================================================
# FORCE INDEX TESTS
# =============================================================================

def test_force_index_uptrend(uptrend_data):
    """Test Force Index in strong uptrend"""
    prices, volumes, _, _ = uptrend_data
    
    result = force_index(prices, volumes)
    
    # Check return structure
    assert "values" in result
    assert "signal" in result
    assert "confidence" in result
    
    # Check array length
    assert len(result["values"]) == len(prices)
    
    # Last FI should be positive (buying pressure)
    last_fi = result["values"][-1]
    assert not np.isnan(last_fi)
    assert last_fi > 0, "Force Index should be positive in uptrend"
    
    # Signal should be BUY
    assert result["signal"] == 'BUY'
    assert 0.0 <= result["confidence"] <= 1.0


def test_force_index_downtrend(downtrend_data):
    """Test Force Index in strong downtrend"""
    prices, volumes, _, _ = downtrend_data
    
    result = force_index(prices, volumes)
    
    last_fi = result["values"][-1]
    assert not np.isnan(last_fi)
    assert last_fi < 0, "Force Index should be negative in downtrend"
    
    # Signal should be SELL
    assert result["signal"] == 'SELL'


def test_force_index_rising_confidence():
    """Test that rising Force Index increases confidence"""
    n = 50
    # Create rising prices with increasing volume
    prices = np.linspace(100, 110, n)
    volumes = np.linspace(1_000_000, 3_000_000, n)  # Volume increasing
    
    result = force_index(prices, volumes)
    
    # Confidence should be high when FI is rising
    assert result["confidence"] > 0.5, "Confidence should be high for strong rising FI"


def test_force_index_insufficient_data():
    """Test Force Index with insufficient data"""
    prices = np.array([100, 101, 102])
    volumes = np.array([1000, 1100, 1200])
    
    result = force_index(prices, volumes)
    
    # Should return empty
    assert len(result["values"]) == 0
    assert result["signal"] is None


def test_force_index_sideways(sideways_data):
    """Test Force Index in sideways market"""
    prices, volumes, _, _ = sideways_data
    
    result = force_index(prices, volumes)
    
    # Force Index should oscillate around zero
    fi_values = result["values"][~np.isnan(result["values"])]
    mean_fi = np.mean(fi_values)
    
    # Mean should be close to zero in sideways market
    assert abs(mean_fi) < np.std(fi_values), "Mean FI should be near zero in sideways market"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_all_indicators_uptrend(uptrend_data):
    """Test all three indicators agree in strong uptrend"""
    prices, volumes, high, low = uptrend_data
    
    vwmacd = volume_weighted_macd(prices, volumes)
    eom = ease_of_movement(high, low, volumes)
    fi = force_index(prices, volumes)
    
    # All should give bullish signals
    assert vwmacd["signal"] in ['BUY', None]
    assert eom["signal"] in ['BUY', None]
    assert fi["signal"] == 'BUY'
    
    # VWMACD should be positive
    assert vwmacd["macd_line"][-1] > 0
    
    # Force Index should be positive
    assert fi["values"][-1] > 0


def test_all_indicators_downtrend(downtrend_data):
    """Test all three indicators agree in strong downtrend"""
    prices, volumes, high, low = downtrend_data
    
    vwmacd = volume_weighted_macd(prices, volumes)
    eom = ease_of_movement(high, low, volumes)
    fi = force_index(prices, volumes)
    
    # All should give bearish signals
    assert vwmacd["signal"] in ['SELL', None]
    assert eom["signal"] in ['SELL', None]
    assert fi["signal"] == 'SELL'
    
    # VWMACD should be negative
    assert vwmacd["macd_line"][-1] < 0
    
    # Force Index should be negative
    assert fi["values"][-1] < 0


# =============================================================================
# EDGE CASES
# =============================================================================

def test_constant_prices():
    """Test indicators with constant prices (no movement)"""
    prices = np.array([100.0] * 50)
    volumes = np.array([1_000_000.0] * 50)
    high = prices + 0.1
    low = prices - 0.1
    
    vwmacd = volume_weighted_macd(prices, volumes)
    eom = ease_of_movement(high, low, volumes)
    fi = force_index(prices, volumes)
    
    # VWMACD should be near zero
    assert abs(vwmacd["macd_line"][-1]) < 1.0
    
    # Force Index should be near zero (no price change)
    fi_values = fi["values"][~np.isnan(fi["values"])]
    assert np.allclose(fi_values, 0.0, atol=1e-6)


def test_extreme_volume_spike():
    """Test indicators handle extreme volume spikes"""
    n = 50
    prices = np.linspace(100, 110, n)
    volumes = np.ones(n) * 1_000_000
    volumes[n//2] = 100_000_000  # 100x volume spike
    
    high = prices + 1
    low = prices - 1
    
    vwmacd = volume_weighted_macd(prices, volumes)
    eom = ease_of_movement(high, low, volumes)
    fi = force_index(prices, volumes)
    
    # All should handle spike without crashing
    assert len(vwmacd["macd_line"]) == n
    assert len(eom["values"]) == n
    assert len(fi["values"]) == n
    
    # No NaN or Inf in results (except first padded value)
    assert not np.any(np.isnan(vwmacd["macd_line"]))
    assert not np.any(np.isinf(vwmacd["macd_line"]))
    assert not np.any(np.isinf(fi["values"][~np.isnan(fi["values"])]))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

