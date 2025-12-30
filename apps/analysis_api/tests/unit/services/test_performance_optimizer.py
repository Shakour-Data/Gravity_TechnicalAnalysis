import numpy as np
from gravity_tech.services.performance_optimizer import (
    calculate_macd_numba,
    calculate_rsi_numba,
    calculate_sma_numba,
)


class TestPerformanceOptimizer:
    """Test Numba-optimized functions."""

    def test_calculate_sma_numba(self):
        prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0])
        result = calculate_sma_numba(prices, 3)
        expected = np.array([np.nan, np.nan, 101.0, 102.0, 103.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_calculate_rsi_numba(self):
        prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0])
        result = calculate_rsi_numba(prices, 14)
        assert isinstance(result, float)
        assert 0 <= result <= 100

    def test_calculate_macd_numba(self):
        prices = np.array([100.0] * 50)  # Constant prices
        macd, signal = calculate_macd_numba(prices)
        assert len(macd) == len(prices)
        assert len(signal) == len(prices)
        # For constant prices, MACD should be close to 0
        assert abs(macd[-1]) < 0.1
