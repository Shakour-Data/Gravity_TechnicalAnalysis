"""Unit tests for candle entity."""

from datetime import datetime

import pytest
from gravity_tech.core.domain.entities.candle import Candle, CandleType


class TestCandle:
    """Test Candle entity."""

    def test_candle_creation(self):
        """Test basic candle creation."""
        candle = Candle(
            timestamp=datetime.fromisoformat("2023-01-01T00:00:00Z"),
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=1000,
        )

        assert candle.open == 100.0
        assert candle.high == 110.0
        assert candle.low == 90.0
        assert candle.close == 105.0
        assert candle.volume == 1000
        assert candle.symbol == "UNKNOWN"
        assert candle.timeframe == "1h"

    def test_candle_properties(self):
        """Test candle computed properties."""
        candle = Candle(
            timestamp=datetime.fromisoformat("2023-01-01T00:00:00Z"),
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=1000,
        )

        assert candle.body_size == 5.0  # |105 - 100|
        assert candle.upper_shadow == 5.0  # 110 - 105
        assert candle.lower_shadow == 10.0  # 100 - 90
        assert candle.total_range == 20.0  # 110 - 90
        assert candle.body_percent == 25.0  # 5/20 * 100
        assert candle.is_bullish is True
        assert candle.is_bearish is False

    def test_candle_type_bullish(self):
        """Test bullish candle type."""
        candle = Candle(
            timestamp=datetime.fromisoformat("2023-01-01T00:00:00Z"),
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=1000,
        )

        assert candle.candle_type == CandleType.BULLISH

    def test_candle_type_bearish(self):
        """Test bearish candle type."""
        candle = Candle(
            timestamp=datetime.fromisoformat("2023-01-01T00:00:00Z"),
            open=105.0,
            high=110.0,
            low=90.0,
            close=100.0,
            volume=1000,
        )

        assert candle.candle_type == CandleType.BEARISH

    def test_candle_type_doji(self):
        """Test doji candle type."""
        candle = Candle(
            timestamp=datetime.fromisoformat("2023-01-01T00:00:00Z"),
            open=100.0,
            high=110.0,
            low=90.0,
            close=100.1,
            volume=1000,
        )

        assert candle.candle_type == CandleType.DOJI

    def test_candle_validation_high_low(self):
        """Test candle validation for high/low consistency."""
        with pytest.raises(ValueError, match="Low.*must be <= high"):
            Candle(
                timestamp=datetime.fromisoformat("2023-01-01T00:00:00Z"),
                open=100.0,
                high=100.0,
                low=110.0,  # low > high
                close=105.0,
                volume=1000,
            )

    def test_candle_validation_negative_volume(self):
        """Test candle validation for negative volume."""
        with pytest.raises(ValueError, match="Volume.*cannot be negative"):
            Candle(
                timestamp=datetime.fromisoformat("2023-01-01T00:00:00Z"),
                open=100.0,
                high=110.0,
                low=90.0,
                close=105.0,
                volume=-1000,
            )
