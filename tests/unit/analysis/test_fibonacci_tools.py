"""
Unit tests for fibonacci_tools.py in analysis module.

Tests cover all methods in FibonacciTools class to achieve >50% coverage.
"""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from gravity_tech.analysis.fibonacci_tools import FibonacciTools
from gravity_tech.core.domain.entities import Candle, FibonacciLevel, FibonacciResult
from gravity_tech.core.domain.entities.signal_strength import SignalStrength


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

    for i in range(1, num_samples):
        if trend == 'uptrend':
            drift = 0.0015
        elif trend == 'downtrend':
            drift = -0.0015
        else:  # mixed
            if i < num_samples // 3:
                drift = 0.002
            elif i < 2 * num_samples // 3:
                drift = -0.001
            else:
                drift = 0.0005

        change = drift + rng.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 100))

        volume = base_volume * (1 + rng.normal(0, volume_variability))
        volumes.append(int(max(volume, 1000)))

    volumes.insert(0, base_volume)

    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices
    })

    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    open_prices = df['open'].to_numpy()
    close_prices = df['close'].to_numpy()
    body_highs = np.maximum(open_prices, close_prices)
    body_lows = np.minimum(open_prices, close_prices)
    wick_ranges = np.abs(close_prices) * 0.008
    wick_above = rng.uniform(0, wick_ranges)
    wick_below = rng.uniform(0, wick_ranges)

    df['high'] = body_highs + wick_above
    df['low'] = body_lows - wick_below
    df['volume'] = volumes

    return df


def dataframe_to_candles(df: pd.DataFrame) -> list[Candle]:
    """Convert DataFrame to Candle list."""
    candles = []
    for _, row in df.iterrows():
        candles.append(Candle(
            timestamp=row['timestamp'],
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=int(row['volume'])
        ))
    return candles


class TestFibonacciTools:
    """Test suite for FibonacciTools class."""

    @pytest.fixture
    def fib_tools(self):
        """Fixture for FibonacciTools instance."""
        return FibonacciTools()

    def test_calculate_retracements_basic(self, fib_tools):
        """Test basic retracement calculation."""
        high = Decimal('100.0')
        low = Decimal('50.0')

        result = fib_tools.calculate_retracements(high, low)

        assert isinstance(result, list)
        assert len(result) == 5  # Standard retracement levels
        assert all(isinstance(level, FibonacciLevel) for level in result)

        # Check specific levels
        levels = {level.ratio: level.price for level in result}
        assert Decimal('0.236') in levels
        assert Decimal('0.382') in levels
        assert Decimal('0.5') in levels
        assert Decimal('0.618') in levels
        assert Decimal('0.786') in levels

    def test_calculate_retracements_values(self, fib_tools):
        """Test retracement values are calculated correctly."""
        high = Decimal('100.0')
        low = Decimal('0.0')

        result = fib_tools.calculate_retracements(high, low)

        # Find 0.618 level
        level_618 = next(level for level in result if level.ratio == Decimal('0.618'))
        expected_price = Decimal('38.2')  # 100 - (100-0) * 0.618
        assert abs(level_618.price - expected_price) < Decimal('0.01')

    def test_calculate_extensions_uptrend(self, fib_tools):
        """Test extension calculation for uptrend."""
        high = Decimal('100.0')
        low = Decimal('50.0')

        result = fib_tools.calculate_extensions(high, low, direction="up")

        assert isinstance(result, list)
        assert len(result) == 8  # Standard extension levels
        assert all(isinstance(level, FibonacciLevel) for level in result)

    def test_calculate_extensions_downtrend(self, fib_tools):
        """Test extension calculation for downtrend."""
        high = Decimal('100.0')
        low = Decimal('50.0')

        result = fib_tools.calculate_extensions(high, low, direction="down")

        assert isinstance(result, list)
        assert len(result) == 8

    def test_calculate_arcs_basic(self, fib_tools):
        """Test arc calculation."""
        center = Decimal('75.0')
        radius = Decimal('25.0')

        result = fib_tools.calculate_arcs(center, radius)

        assert isinstance(result, list)
        assert len(result) == 4  # Standard arc levels
        assert all(isinstance(level, FibonacciLevel) for level in result)

    def test_calculate_fans_basic(self, fib_tools):
        """Test fan calculation."""
        start_point = (Decimal('0.0'), Decimal('50.0'))
        end_point = (Decimal('100.0'), Decimal('100.0'))

        result = fib_tools.calculate_fans(start_point, end_point)

        assert isinstance(result, list)
        assert len(result) == 4  # Standard fan levels
        assert all(isinstance(level, dict) for level in result)

    def test_calculate_projections_basic(self, fib_tools):
        """Test projection calculation."""
        move1_high = Decimal('100.0')
        move1_low = Decimal('50.0')
        move2_high = Decimal('150.0')
        move2_low = Decimal('100.0')

        result = fib_tools.calculate_projections(move1_high, move1_low, move2_high, move2_low)

        assert isinstance(result, list)
        assert len(result) == 5  # Standard projection levels
        assert all(isinstance(level, FibonacciLevel) for level in result)

    @pytest.fixture
    def sample_candles(self, real_tse_candles):
        """Fixture for sample candle data."""
        return real_tse_candles

    def test_find_fibonacci_levels_basic(self, fib_tools, sample_candles):
        """Test finding fibonacci levels from candles."""
        result = fib_tools.find_fibonacci_levels(sample_candles, lookback=50)

        assert isinstance(result, list)
        # Depending on data, may return levels or empty list
        if result:
            assert all(isinstance(level, FibonacciLevel) for level in result)

    def test_analyze_fibonacci_confluence_basic(self, fib_tools, real_tse_candles):
        """Test confluence analysis with real TSE data."""
        current_price = 15000.0

        result = fib_tools.analyze_fibonacci_confluence(real_tse_candles, current_price)

        # May return None if no confluence found
        if result:
            assert isinstance(result, FibonacciResult)
            # Check if it has the expected attributes
            assert hasattr(result, 'levels')

    def test_calculate_retracements_edge_case_equal_high_low(self, fib_tools):
        """Test retracement calculation when high equals low."""
        high = low = Decimal('100.0')

        result = fib_tools.calculate_retracements(high, low)

        assert isinstance(result, list)
        # Should handle zero range gracefully

    def test_calculate_extensions_invalid_direction(self, fib_tools):
        """Test extension calculation with invalid direction."""
        high = Decimal('100.0')
        low = Decimal('50.0')

        with pytest.raises(ValueError):
            fib_tools.calculate_extensions(high, low, direction="invalid")

    def test_calculate_arcs_zero_radius(self, fib_tools):
        """Test arc calculation with zero radius."""
        center = Decimal('75.0')
        radius = Decimal('0.0')

        result = fib_tools.calculate_arcs(center, radius)

        assert isinstance(result, list)
        # Should handle zero radius

    def test_calculate_fans_same_points(self, fib_tools):
        """Test fan calculation with same start and end points."""
        point = (Decimal('50.0'), Decimal('50.0'))

        result = fib_tools.calculate_fans(point, point)

        assert isinstance(result, list)
        # Should handle degenerate case

    def test_calculate_projections_zero_moves(self, fib_tools):
        """Test projection calculation with zero moves."""
        move1_high = move1_low = Decimal('100.0')
        move2_high = move2_low = Decimal('100.0')

        result = fib_tools.calculate_projections(move1_high, move1_low, move2_high, move2_low)

        assert isinstance(result, list)
        # Should handle zero moves

    def test_find_fibonacci_levels_empty_candles(self, fib_tools):
        """Test finding levels with empty candle list."""
        result = fib_tools.find_fibonacci_levels([])

        assert isinstance(result, list)
        assert len(result) == 0

    def test_analyze_fibonacci_confluence_empty_candles(self, fib_tools):
        """Test confluence analysis with empty candles."""
        result = fib_tools.analyze_fibonacci_confluence([], Decimal('100.0'))

        assert result is None

    # Additional tests to reach >50% coverage
    def test_calculate_retracements_precision(self, fib_tools):
        """Test retracement precision."""
        high = Decimal('100.12345678')
        low = Decimal('50.87654321')

        result = fib_tools.calculate_retracements(high, low, precision=4)

        assert all(level.price.as_tuple().exponent >= -4 for level in result)

    def test_calculate_extensions_custom_ratios(self, fib_tools):
        """Test extensions with custom ratios."""
        high = Decimal('100.0')
        low = Decimal('50.0')
        custom_ratios = [Decimal('0.5'), Decimal('1.0')]

        result = fib_tools.calculate_extensions(high, low, custom_ratios=custom_ratios)

        assert len(result) == 2
        assert result[0].ratio == Decimal('0.5')
        assert result[1].ratio == Decimal('1.0')

    def test_calculate_arcs_custom_angles(self, fib_tools):
        """Test arcs with custom angle range."""
        center = Decimal('75.0')
        radius = Decimal('25.0')

        result = fib_tools.calculate_arcs(center, radius, angle_range=(45, 135))

        assert isinstance(result, list)

    def test_calculate_fans_custom_ratios(self, fib_tools):
        """Test fans with custom ratios."""
        start_point = (Decimal('0.0'), Decimal('50.0'))
        end_point = (Decimal('100.0'), Decimal('100.0'))
        custom_ratios = [Decimal('0.5')]

        result = fib_tools.calculate_fans(start_point, end_point, custom_ratios=custom_ratios)

        assert len(result) == 1

    def test_calculate_projections_custom_ratios(self, fib_tools):
        """Test projections with custom ratios."""
        move1_high = Decimal('100.0')
        move1_low = Decimal('50.0')
        move2_high = Decimal('150.0')
        move2_low = Decimal('100.0')
        custom_ratios = [Decimal('1.0')]

        result = fib_tools.calculate_projections(move1_high, move1_low, move2_high, move2_low, custom_ratios=custom_ratios)

        assert len(result) == 1
        assert result[0].ratio == Decimal('1.0')

    def test_find_fibonacci_levels_different_lookback(self, fib_tools, sample_candles):
        """Test finding levels with different lookback periods."""
        for lookback in [10, 25, 100]:
            result = fib_tools.find_fibonacci_levels(sample_candles, lookback=lookback)
            assert isinstance(result, list)

    def test_analyze_fibonacci_confluence_different_prices(self, fib_tools, sample_candles):
        """Test confluence analysis with different current prices."""
        for price in [Decimal('90.0'), Decimal('105.0'), Decimal('120.0')]:
            result = fib_tools.analyze_fibonacci_confluence(sample_candles, price)
            # Result may be None or FibonacciResult
            if result:
                assert isinstance(result, FibonacciResult)

    # Tests for private methods (if needed for coverage)
    def test_private_find_swings(self, fib_tools, sample_candles):
        """Test private _find_swings method."""
        swings = fib_tools._find_swings(sample_candles)
        assert isinstance(swings, list)

    def test_private_remove_duplicate_levels(self, fib_tools):
        """Test private _remove_duplicate_levels method."""
        levels = [
            FibonacciLevel(price=100.0, ratio=0.5, level_type='retracement', strength=1.0, touches=1, description='Test level'),
            FibonacciLevel(price=100.0, ratio=0.618, level_type='retracement', strength=1.0, touches=1, description='Test level'),
            FibonacciLevel(price=105.0, ratio=0.5, level_type='extension', strength=1.0, touches=1, description='Test level'),
        ]
        result = fib_tools._remove_duplicate_levels(levels)
        assert isinstance(result, list)
        assert len(result) <= len(levels)

    def test_private_calculate_confidence(self, fib_tools):
        """Test private _calculate_confidence method."""
        levels = [
            FibonacciLevel(price=100.0, ratio=0.5, level_type='retracement', strength=1.0, touches=1, description='Test level'),
            FibonacciLevel(price=105.0, ratio=0.618, level_type='retracement', strength=1.0, touches=1, description='Test level'),
        ]
        confidence = fib_tools._calculate_confidence(levels, Decimal('102.0'))
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_private_determine_signal(self, fib_tools):
        """Test private _determine_signal method."""
        signal = fib_tools._determine_signal(Decimal('102.0'), Decimal('100.0'), Decimal('105.0'))
        assert isinstance(signal, SignalStrength)
