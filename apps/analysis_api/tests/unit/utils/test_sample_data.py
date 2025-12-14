"""
Unit tests for src/gravity_tech/utils/sample_data.py

Tests sample data generation utilities.
"""

from src.gravity_tech.utils.sample_data import generate_sample_candles


class TestSampleData:
    """Test sample data generation functionality."""

    def test_generate_sample_candles_default(self):
        """Test generating sample candles with default parameters."""
        candles = generate_sample_candles()

        assert len(candles) == 100
        assert all(hasattr(candle, 'timestamp') for candle in candles)
        assert all(hasattr(candle, 'open') for candle in candles)
        assert all(hasattr(candle, 'close') for candle in candles)
        assert all(hasattr(candle, 'volume') for candle in candles)

    def test_generate_sample_candles_custom_count(self):
        """Test generating sample candles with custom count."""
        candles = generate_sample_candles(num_candles=50)

        assert len(candles) == 50

    def test_generate_sample_candles_uptrend(self):
        """Test generating uptrend candles."""
        candles = generate_sample_candles(num_candles=20, trend="uptrend")

        assert len(candles) == 20

        # Check that prices generally increase
        first_close = candles[0].close
        last_close = candles[-1].close
        assert last_close > first_close

    def test_generate_sample_candles_downtrend(self):
        """Test generating downtrend candles."""
        candles = generate_sample_candles(num_candles=20, trend="downtrend")

        assert len(candles) == 20

        # Check that prices generally decrease
        first_close = candles[0].close
        last_close = candles[-1].close
        assert last_close < first_close

    def test_candle_price_relationships(self):
        """Test that candle prices have proper relationships."""
        candles = generate_sample_candles(num_candles=10)

        for candle in candles:
            # High should be >= max(open, close)
            assert candle.high >= candle.open
            assert candle.high >= candle.close

            # Low should be <= min(open, close)
            assert candle.low <= candle.open
            assert candle.low <= candle.close

            # Volume should be positive
            assert candle.volume > 0
