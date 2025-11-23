"""
Unit tests for src/core/domain/entities/candle.py

Tests Candle dataclass, properties, and methods.
Covers validation, properties, and true_range calculations.
"""

import pytest
import numpy as np
from datetime import datetime
from src.core.domain.entities.candle import Candle, CandleType


class TestCandleCreation:
    """Test suite for Candle creation and validation"""

    def test_candle_creation_valid(self):
        """Test creating a valid candle"""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        candle = Candle(
            timestamp=timestamp,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
            symbol="BTCUSDT",
            timeframe="1h"
        )
        assert candle.timestamp == timestamp
        assert candle.open == 100.0
        assert candle.high == 105.0
        assert candle.low == 95.0
        assert candle.close == 102.0
        assert candle.volume == 1000.0
        assert candle.symbol == "BTCUSDT"
        assert candle.timeframe == "1h"

    def test_candle_creation_defaults(self):
        """Test candle creation with default values"""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        candle = Candle(
            timestamp=timestamp,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0
        )
        assert candle.symbol == "UNKNOWN"
        assert candle.timeframe == "1h"

    def test_candle_validation_high_too_low(self):
        """Test validation when high is less than max(open, close)"""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        with pytest.raises(ValueError, match="High .* must be >= max"):
            Candle(
                timestamp=timestamp,
                open=100.0,
                high=101.0,  # Less than close
                low=95.0,
                close=102.0,
                volume=1000.0
            )

    def test_candle_validation_low_too_high(self):
        """Test validation when low is greater than min(open, close)"""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        with pytest.raises(ValueError, match="Low .* must be <= min"):
            Candle(
                timestamp=timestamp,
                open=100.0,
                high=105.0,
                low=101.0,  # Greater than open
                close=102.0,
                volume=1000.0
            )

    def test_candle_validation_negative_volume(self):
        """Test validation for negative volume"""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        with pytest.raises(ValueError, match="Volume .* cannot be negative"):
            Candle(
                timestamp=timestamp,
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=-100.0
            )

    def test_candle_immutable(self):
        """Test that Candle is immutable (frozen dataclass)"""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        candle = Candle(
            timestamp=timestamp,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0
        )
        with pytest.raises(AttributeError):
            candle.open = 101.0


class TestCandleProperties:
    """Test suite for Candle properties"""

    @pytest.fixture
    def bullish_candle(self):
        """Fixture for a bullish candle"""
        return Candle(
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0
        )

    @pytest.fixture
    def bearish_candle(self):
        """Fixture for a bearish candle"""
        return Candle(
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            open=102.0,
            high=105.0,
            low=95.0,
            close=100.0,
            volume=1000.0
        )

    @pytest.fixture
    def doji_candle(self):
        """Fixture for a doji candle"""
        return Candle(
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.05,  # Very small body
            volume=1000.0
        )

    def test_candle_type_bullish(self, bullish_candle):
        """Test candle type for bullish candle"""
        assert bullish_candle.candle_type == CandleType.BULLISH

    def test_candle_type_bearish(self, bearish_candle):
        """Test candle type for bearish candle"""
        assert bearish_candle.candle_type == CandleType.BEARISH

    def test_candle_type_doji(self, doji_candle):
        """Test candle type for doji candle"""
        assert doji_candle.candle_type == CandleType.DOJI

    def test_body_size_bullish(self, bullish_candle):
        """Test body size for bullish candle"""
        assert bullish_candle.body_size == 2.0

    def test_body_size_bearish(self, bearish_candle):
        """Test body size for bearish candle"""
        assert bearish_candle.body_size == 2.0

    def test_upper_shadow(self, bullish_candle):
        """Test upper shadow calculation"""
        assert bullish_candle.upper_shadow == 3.0  # high - close

    def test_lower_shadow(self, bullish_candle):
        """Test lower shadow calculation"""
        assert bullish_candle.lower_shadow == 5.0  # open - low

    def test_total_range(self, bullish_candle):
        """Test total range calculation"""
        assert bullish_candle.total_range == 10.0  # high - low

    def test_body_percent(self, bullish_candle):
        """Test body percentage calculation"""
        assert bullish_candle.body_percent == 20.0  # (body_size / total_range) * 100

    def test_is_bullish(self, bullish_candle, bearish_candle):
        """Test is_bullish property"""
        assert bullish_candle.is_bullish is True
        assert bearish_candle.is_bullish is False

    def test_is_bearish(self, bullish_candle, bearish_candle):
        """Test is_bearish property"""
        assert bullish_candle.is_bearish is False
        assert bearish_candle.is_bearish is True

    def test_is_doji_default(self, bullish_candle):
        """Test is_doji with default threshold"""
        assert bullish_candle.is_doji() is False

    def test_is_doji_custom_threshold(self, doji_candle):
        """Test is_doji with custom threshold"""
        assert doji_candle.is_doji(threshold=0.01) is True

    def test_typical_price(self, bullish_candle):
        """Test typical price calculation"""
        expected = (105.0 + 95.0 + 102.0) / 3  # (H + L + C) / 3
        assert bullish_candle.typical_price == pytest.approx(expected)


class TestTrueRange:
    """Test suite for True Range calculations"""

    @pytest.fixture
    def current_candle(self):
        """Fixture for current candle"""
        return Candle(
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0
        )

    @pytest.fixture
    def previous_candle(self):
        """Fixture for previous candle"""
        return Candle(
            timestamp=datetime(2023, 1, 1, 11, 0, 0),
            open=98.0,
            high=103.0,
            low=93.0,
            close=101.0,
            volume=1000.0
        )

    def test_true_range_no_previous(self, current_candle):
        """Test true range without previous candle"""
        tr = current_candle.true_range()
        assert tr == 10.0  # high - low

    def test_true_range_with_previous_case1(self, current_candle, previous_candle):
        """Test true range case 1: current high - current low"""
        tr = current_candle.true_range(previous_candle)
        assert tr == 10.0  # max(high-low, |high-prev_close|, |low-prev_close|)

    def test_true_range_with_previous_case2(self):
        """Test true range case 2: |current high - previous close|"""
        current = Candle(
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            open=100.0,
            high=110.0,  # High gap up
            low=105.0,
            close=108.0,
            volume=1000.0
        )
        previous = Candle(
            timestamp=datetime(2023, 1, 1, 11, 0, 0),
            open=98.0,
            high=103.0,
            low=93.0,
            close=95.0,  # Previous close much lower
            volume=1000.0
        )
        tr = current.true_range(previous)
        assert tr == 15.0  # |110 - 95|

    def test_true_range_with_previous_case3(self):
        """Test true range case 3: |current low - previous close|"""
        current = Candle(
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            open=100.0,
            high=105.0,
            low=85.0,  # Low gap down
            close=90.0,
            volume=1000.0
        )
        previous = Candle(
            timestamp=datetime(2023, 1, 1, 11, 0, 0),
            open=98.0,
            high=103.0,
            low=93.0,
            close=110.0,  # Previous close much higher
            volume=1000.0
        )
        tr = current.true_range(previous)
        assert tr == 25.0  # |85 - 110|

    def test_true_range_edge_case_equal_prices(self):
        """Test true range with equal high/low/close"""
        current = Candle(
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1000.0
        )
        tr = current.true_range()
        assert tr == 0.0

    @pytest.mark.parametrize("current_high,current_low,prev_close,expected", [
        (105.0, 95.0, 102.0, 10.0),  # Normal range
        (110.0, 100.0, 95.0, 15.0),  # High gap
        (105.0, 90.0, 110.0, 20.0),  # Low gap
        (100.0, 100.0, 100.0, 0.0),  # No range
    ])
    def test_true_range_parametrized(self, current_high, current_low, prev_close, expected):
        """Parametrized test for true range calculations"""
        current = Candle(
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            open=100.0,
            high=current_high,
            low=current_low,
            close=102.0,
            volume=1000.0
        )
        previous = Candle(
            timestamp=datetime(2023, 1, 1, 11, 0, 0),
            open=98.0,
            high=103.0,
            low=93.0,
            close=prev_close,
            volume=1000.0
        )
        tr = current.true_range(previous)
        assert tr == expected


class TestCandleTypeEnum:
    """Test suite for CandleType enum"""

    def test_candle_type_values(self):
        """Test CandleType enum values"""
        assert CandleType.BULLISH.value == "BULLISH"
        assert CandleType.BEARISH.value == "BEARISH"
        assert CandleType.DOJI.value == "DOJI"

    def test_candle_type_str_representation(self):
        """Test string representation of CandleType"""
        assert str(CandleType.BULLISH) == "BULLISH"
        assert str(CandleType.BEARISH) == "BEARISH"
        assert str(CandleType.DOJI) == "DOJI"
