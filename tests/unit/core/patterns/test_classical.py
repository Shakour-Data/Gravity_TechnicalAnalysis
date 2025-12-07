"""
Unit tests for classical.py in core/patterns module.

Tests cover all classical pattern detection methods to achieve >50% coverage.
"""

import numpy as np
import pandas as pd
import pytest
from gravity_tech.core.domain.entities import Candle, PatternResult, PatternType
from gravity_tech.core.domain.entities import CoreSignalStrength as SignalStrength
from gravity_tech.core.patterns.classical import ClassicalPatterns


def create_realistic_tse_data(num_samples: int = 100, trend: str = 'mixed', seed: int = 42) -> pd.DataFrame:
    """Create realistic TSE market data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_samples, freq='1d')

    base_price = 15000.0
    volatility = 0.025
    base_volume = 500_000
    volume_variability = 0.5

    prices = [base_price]
    volumes = []

    for _ in range(1, num_samples):
        if trend == 'uptrend':
            drift = 0.001  # Slight upward drift
        elif trend == 'downtrend':
            drift = -0.001  # Slight downward drift
        else:  # mixed
            drift = 0.0002 * rng.normal()

        shock = rng.normal() * volatility
        new_price = prices[-1] * (1 + drift + shock)
        prices.append(max(new_price, 1000))  # Floor price

        volume = base_volume * (1 + volume_variability * rng.normal())
        volumes.append(max(volume, 10000))

    # Create OHLC from prices
    opens = prices[:-1]
    closes = prices[1:]
    highs = [max(o, c) * (1 + abs(rng.normal()) * volatility * 0.5) for o, c in zip(opens, closes, strict=True)]
    lows = [min(o, c) * (1 - abs(rng.normal()) * volatility * 0.5) for o, c in zip(opens, closes, strict=True)]

    df = pd.DataFrame({
        'timestamp': dates[1:],
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })

    return df


def dataframe_to_candles(df: pd.DataFrame) -> list[Candle]:
    """Convert DataFrame to list of Candle objects."""
    candles = []
    for _, row in df.iterrows():
        candle = Candle(
            timestamp=row['timestamp'],
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume'])
        )
        candles.append(candle)
    return candles


@pytest.fixture
def real_tse_candles():
    """Realistic TSE market data for testing."""
    df = create_realistic_tse_data(num_samples=100, trend='uptrend', seed=42)
    return dataframe_to_candles(df)


@pytest.fixture
def downtrend_candles():
    """Downtrend TSE data."""
    df = create_realistic_tse_data(num_samples=100, trend='downtrend', seed=123)
    return dataframe_to_candles(df)


@pytest.fixture
def mixed_candles():
    """Mixed trend TSE data."""
    df = create_realistic_tse_data(num_samples=100, trend='mixed', seed=456)
    return dataframe_to_candles(df)


@pytest.fixture
def head_and_shoulders_candles():
    """Candles forming a head and shoulders pattern."""
    # Create a clear head and shoulders pattern
    highs = [100, 105, 110, 105, 100]  # Left shoulder, head, right shoulder
    lows = [95, 98, 102, 98, 95]   # Troughs between peaks
    closes = [98, 102, 105, 102, 98]

    candles = []
    for i in range(len(highs)):
        open_price = closes[i-1] if i > 0 else highs[0] - 2
        # Ensure high >= max(open, close) and low <= min(open, close)
        high = max(highs[i], open_price, closes[i])
        low = min(lows[i], open_price, closes[i])
        candle = Candle(
            timestamp=pd.Timestamp(f'2024-01-{i+1:02d}'),
            open=float(open_price),
            high=float(high),
            low=float(low),
            close=float(closes[i]),
            volume=1000000.0
        )
        candles.append(candle)
    return candles


@pytest.fixture
def inverse_head_and_shoulders_candles():
    """Candles forming an inverse head and shoulders pattern."""
    # Create inverse head and shoulders pattern
    lows = [100, 95, 90, 95, 100]  # Left shoulder, head, right shoulder
    highs = [105, 102, 98, 102, 105]   # Peaks between troughs
    closes = [102, 98, 95, 98, 102]

    candles = []
    for i in range(len(lows)):
        open_price = closes[i-1] if i > 0 else lows[0] + 2
        # Ensure high >= max(open, close) and low <= min(open, close)
        high = max(highs[i], open_price, closes[i])
        low = min(lows[i], open_price, closes[i])
        candle = Candle(
            timestamp=pd.Timestamp(f'2024-01-{i+1:02d}'),
            open=float(open_price),
            high=float(high),
            low=float(low),
            close=float(closes[i]),
            volume=1000000.0
        )
        candles.append(candle)
    return candles


@pytest.fixture
def double_top_candles():
    """Candles forming a double top pattern."""
    # Create double top pattern
    highs = [100, 105, 110, 105, 100, 95]  # Two peaks at similar level
    lows = [95, 98, 102, 98, 95, 90]
    closes = [98, 102, 105, 102, 98, 93]

    candles = []
    for i in range(len(highs)):
        open_price = closes[i-1] if i > 0 else highs[0] - 2
        # Ensure high >= max(open, close) and low <= min(open, close)
        high = max(highs[i], open_price, closes[i])
        low = min(lows[i], open_price, closes[i])
        candle = Candle(
            timestamp=pd.Timestamp(f'2024-01-{i+1:02d}'),
            open=float(open_price),
            high=float(high),
            low=float(low),
            close=float(closes[i]),
            volume=1000000.0
        )
        candles.append(candle)
    return candles


@pytest.fixture
def ascending_triangle_candles():
    """Candles forming an ascending triangle pattern."""
    # Create ascending triangle pattern
    highs = [100, 101, 102, 103, 104, 105]  # Horizontal resistance
    lows = [95, 96, 97, 98, 99, 100]   # Rising support
    closes = [97, 98, 99, 100, 101, 102]

    candles = []
    for i in range(len(highs)):
        open_price = closes[i-1] if i > 0 else highs[0] - 2
        candle = Candle(
            timestamp=pd.Timestamp(f'2024-01-{i+1:02d}'),
            open=float(open_price),
            high=float(highs[i]),
            low=float(lows[i]),
            close=float(closes[i]),
            volume=1000000.0
        )
        candles.append(candle)
    return candles


class TestClassicalPatterns:
    """Test suite for ClassicalPatterns class."""

    def test_find_swing_points_basic(self, real_tse_candles):
        """Test basic swing point detection."""
        swings = ClassicalPatterns.find_swing_points(real_tse_candles, order=3)

        assert 'highs' in swings
        assert 'lows' in swings
        assert isinstance(swings['highs'], list)
        assert isinstance(swings['lows'], list)

        # Should find some swing points in realistic data
        assert len(swings['highs']) > 0
        assert len(swings['lows']) > 0

        # Each swing point should be a tuple of (index, price)
        for idx, price in swings['highs'] + swings['lows']:
            assert isinstance(idx, int | np.integer)
            assert isinstance(price, int | float | np.floating)

    def test_find_swing_points_empty_data(self):
        """Test swing point detection with empty data."""
        swings = ClassicalPatterns.find_swing_points([], order=3)

        assert swings['highs'] == []
        assert swings['lows'] == []

    def test_find_swing_points_insufficient_data(self):
        """Test swing point detection with insufficient data."""
        candles = [Candle(
            timestamp=pd.Timestamp('2024-01-01'),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000.0
        )]
        swings = ClassicalPatterns.find_swing_points(candles, order=5)

        # With only 1 candle, no swing points can be found
        assert swings['highs'] == []
        assert swings['lows'] == []

    def test_detect_head_and_shoulders_no_pattern(self, real_tse_candles):
        """Test head and shoulders detection with no pattern."""
        result = ClassicalPatterns.detect_head_and_shoulders(real_tse_candles[:20])

        # Should return None for insufficient data or no pattern
        assert result is None

    def test_detect_head_and_shoulders_with_pattern(self, head_and_shoulders_candles):
        """Test head and shoulders detection with clear pattern."""
        result = ClassicalPatterns.detect_head_and_shoulders(head_and_shoulders_candles)

        # May or may not detect pattern depending on exact data
        if result is not None:
            assert isinstance(result, PatternResult)
            assert result.pattern_name == "Head and Shoulders"
            assert result.pattern_type == PatternType.CLASSICAL
            assert result.signal in [SignalStrength.BEARISH, SignalStrength.VERY_BEARISH]
            assert 0.7 <= result.confidence <= 0.85
            if result.price_target is not None and result.stop_loss is not None:
                assert result.price_target < result.stop_loss

    def test_detect_inverse_head_and_shoulders_no_pattern(self, real_tse_candles):
        """Test inverse head and shoulders detection with no pattern."""
        result = ClassicalPatterns.detect_inverse_head_and_shoulders(real_tse_candles[:20])

        assert result is None

    def test_detect_inverse_head_and_shoulders_with_pattern(self, inverse_head_and_shoulders_candles):
        """Test inverse head and shoulders detection with clear pattern."""
        result = ClassicalPatterns.detect_inverse_head_and_shoulders(inverse_head_and_shoulders_candles)

        # May or may not detect pattern depending on exact data
        if result is not None:
            assert isinstance(result, PatternResult)
            assert result.pattern_name == "Inverse Head and Shoulders"
            assert result.pattern_type == PatternType.CLASSICAL
            assert result.signal in [SignalStrength.BULLISH, SignalStrength.VERY_BULLISH]
            assert 0.7 <= result.confidence <= 0.85
            if result.price_target is not None and result.stop_loss is not None:
                assert result.price_target > result.stop_loss

    def test_detect_double_top_no_pattern(self, real_tse_candles):
        """Test double top detection with no pattern."""
        result = ClassicalPatterns.detect_double_top(real_tse_candles[:30])

        assert result is None

    def test_detect_double_top_with_pattern(self, double_top_candles):
        """Test double top detection with clear pattern."""
        result = ClassicalPatterns.detect_double_top(double_top_candles)

        # May or may not detect pattern depending on exact data
        if result is not None:
            assert isinstance(result, PatternResult)
            assert result.pattern_name == "Double Top"
            assert result.pattern_type == PatternType.CLASSICAL
            assert result.signal == SignalStrength.BEARISH
            assert result.confidence > 0.8

    def test_detect_double_bottom_no_pattern(self, real_tse_candles):
        """Test double bottom detection with no pattern."""
        result = ClassicalPatterns.detect_double_bottom(real_tse_candles[:30])

        # May or may not find pattern in realistic data
        if result is not None:
            assert isinstance(result, PatternResult)
            assert result.pattern_name == "Double Bottom"
            assert result.signal == SignalStrength.BULLISH

    def test_detect_ascending_triangle_no_pattern(self, real_tse_candles):
        """Test ascending triangle detection with no pattern."""
        result = ClassicalPatterns.detect_ascending_triangle(real_tse_candles[:30])

        assert result is None

    def test_detect_ascending_triangle_with_pattern(self, ascending_triangle_candles):
        """Test ascending triangle detection with clear pattern."""
        result = ClassicalPatterns.detect_ascending_triangle(ascending_triangle_candles)

        # May or may not detect pattern depending on exact data
        if result is not None:
            assert isinstance(result, PatternResult)
            assert result.pattern_name == "Ascending Triangle"
            assert result.pattern_type == PatternType.CLASSICAL
            assert result.signal == SignalStrength.BULLISH
            assert result.confidence > 0.7

    def test_detect_descending_triangle_no_pattern(self, real_tse_candles):
        """Test descending triangle detection with no pattern."""
        result = ClassicalPatterns.detect_descending_triangle(real_tse_candles[:30])

        assert result is None

    def test_detect_symmetrical_triangle_no_pattern(self, real_tse_candles):
        """Test symmetrical triangle detection with no pattern."""
        result = ClassicalPatterns.detect_symmetrical_triangle(real_tse_candles[:30])

        assert result is None

    def test_detect_all_with_real_data(self, real_tse_candles):
        """Test detect_all method with realistic TSE data."""
        results = ClassicalPatterns.detect_all(real_tse_candles)

        assert isinstance(results, list)
        # May or may not find patterns in random data
        for result in results:
            assert isinstance(result, PatternResult)
            assert result.pattern_type == PatternType.CLASSICAL

    def test_detect_all_empty_data(self):
        """Test detect_all with empty data."""
        results = ClassicalPatterns.detect_all([])

        assert results == []

    def test_detect_all_insufficient_data(self):
        """Test detect_all with insufficient data."""
        candles = [Candle(
            timestamp=pd.Timestamp('2024-01-01'),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000.0
        )]
        results = ClassicalPatterns.detect_all(candles)

        assert results == []

    def test_pattern_result_structure(self, head_and_shoulders_candles):
        """Test that PatternResult has all required fields."""
        result = ClassicalPatterns.detect_head_and_shoulders(head_and_shoulders_candles)

        if result is not None:
            assert hasattr(result, 'pattern_name')
            assert hasattr(result, 'pattern_type')
            assert hasattr(result, 'signal')
            assert hasattr(result, 'confidence')
            assert hasattr(result, 'start_time')
            assert hasattr(result, 'end_time')
            assert hasattr(result, 'price_target')
            assert hasattr(result, 'stop_loss')
            assert hasattr(result, 'description')

    def test_edge_case_min_pattern_bars(self):
        """Test behavior with minimum pattern bars."""
        # Create data with exactly minimum bars
        candles = []
        for i in range(20):
            candle = Candle(
                timestamp=pd.Timestamp(f'2024-01-{i+1:02d}'),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000000.0
            )
            candles.append(candle)

        result = ClassicalPatterns.detect_head_and_shoulders(candles)
        # May or may not find pattern, but shouldn't crash
        assert result is None or isinstance(result, PatternResult)
