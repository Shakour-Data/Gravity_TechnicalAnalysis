"""Unit tests for support and resistance indicators."""

import numpy as np
import pytest
from gravity_tech.core.indicators.support_resistance import SupportResistanceIndicators


class TestSupportResistanceIndicators:
    """Test Support and Resistance Indicators class."""

    @pytest.fixture
    def sample_candles(self):
        """Create sample candle data for testing."""
        from datetime import datetime, timedelta

        from gravity_tech.core.domain.entities import Candle

        # Create 100 sample candles with trending data
        candles = []
        base_date = datetime(2023, 1, 1)
        base_price = 100.0
        for i in range(100):
            current_date = base_date + timedelta(days=i)
            open_price = base_price + np.sin(i * 0.1) * 5
            close_price = open_price + np.random.normal(0, 1)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.5))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.5))
            volume = 1000000 + np.random.normal(0, 100000)

            candle = Candle(
                timestamp=current_date,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                symbol="BTCUSDT",
                timeframe="1h"
            )
            candles.append(candle)
            base_price += 0.1  # Slight upward trend

        return candles

    def test_calculate_all_basic(self, sample_candles):
        """Test calculate_all method."""
        results = SupportResistanceIndicators.calculate_all(sample_candles)

        assert isinstance(results, list)
        assert len(results) > 0
        for result in results:
            assert hasattr(result, 'indicator_name')
            assert hasattr(result, 'value')
            assert hasattr(result, 'signal')
            assert hasattr(result, 'confidence')

    def test_pivot_points_basic(self, sample_candles):
        """Test basic pivot points calculation."""
        result = SupportResistanceIndicators.pivot_points(sample_candles, method='standard')

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'Pivot Points (Standard)'

    def test_pivot_points_camarilla(self, sample_candles):
        """Test Camarilla pivot points calculation."""
        result = SupportResistanceIndicators.pivot_points(sample_candles, method='camarilla')

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'Pivot Points (Camarilla)'

    def test_fibonacci_retracement_basic(self, sample_candles):
        """Test basic Fibonacci retracement calculation."""
        result = SupportResistanceIndicators.fibonacci_retracement(sample_candles, lookback=50)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'Fibonacci Retracement(50)'

    def test_camarilla_pivots_basic(self, sample_candles):
        """Test basic Camarilla pivots calculation."""
        result = SupportResistanceIndicators.camarilla_pivots(sample_candles)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'Camarilla Pivots'

    def test_support_resistance_levels_basic(self, sample_candles):
        """Test basic support/resistance levels calculation."""
        result = SupportResistanceIndicators.support_resistance_levels(sample_candles, window=50, num_touches=2)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'Support/Resistance(50)'

    def test_dynamic_support_resistance_basic(self, sample_candles):
        """Test basic dynamic support/resistance calculation."""
        result = SupportResistanceIndicators.dynamic_support_resistance(sample_candles, short_period=10, long_period=20)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'Dynamic S/R(10,20)'

    def test_identify_key_levels_basic(self, sample_candles):
        """Test basic key levels identification."""
        result = SupportResistanceIndicators.identify_key_levels(sample_candles, lookback=50)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'Key Levels(50)'

    def test_detect_breakout_basic(self, sample_candles):
        """Test basic breakout detection."""
        result = SupportResistanceIndicators.detect_breakout(sample_candles, lookback=20, window=20)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'Breakout Detection(20)'

    def test_detect_zones_basic(self, sample_candles):
        """Test basic zone detection."""
        result = SupportResistanceIndicators.detect_zones(sample_candles, zone_width=0.005)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'Zone Detection(0.5%)'

    def test_price_action_at_level_basic(self, sample_candles):
        """Test basic price action at level."""
        level = 105.0  # A level within the price range
        result = SupportResistanceIndicators.price_action_at_level(sample_candles, level, tolerance=0.01)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'Price Action at 105.00'