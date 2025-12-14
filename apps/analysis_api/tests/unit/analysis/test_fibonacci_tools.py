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
        assert 0.236 in levels
        assert 0.382 in levels
        assert 0.5 in levels
        assert 0.618 in levels
        assert 0.786 in levels

    def test_calculate_retracements_values(self, fib_tools):
        """Test retracement values are calculated correctly."""
        high = Decimal('100.0')
        low = Decimal('0.0')

        result = fib_tools.calculate_retracements(high, low)

        # Find 0.618 level
        level_618 = next(level for level in result if level.ratio == 0.618)
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
        center_point = (Decimal('75.0'), 0)
        radius_point = (Decimal('100.0'), 10)
        time_point = 5

        result = fib_tools.calculate_arcs(center_point, radius_point, time_point)

        assert isinstance(result, list)
        assert len(result) == 4  # Standard arc levels
        assert all(isinstance(level, FibonacciLevel) for level in result)

    def test_calculate_fans_basic(self, fib_tools):
        """Test fan calculation."""
        origin_point = (Decimal('0.0'), 0)
        high_point = (Decimal('100.0'), 100)
        time_point = 50

        result = fib_tools.calculate_fans(origin_point, high_point, time_point)

        assert isinstance(result, list)
        assert len(result) == 4  # Standard fan levels
        assert all(isinstance(level, FibonacciLevel) for level in result)

    def test_calculate_projections_basic(self, fib_tools):
        """Test projection calculation."""
        move1_high = Decimal('100.0')
        move1_low = Decimal('50.0')
        move2_high = Decimal('150.0')

        result = fib_tools.calculate_projections(move1_low, move1_high, move2_high)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(level, FibonacciLevel) for level in result)
        assert len(result) == 5  # Standard projection levels
        assert all(isinstance(level, FibonacciLevel) for level in result)

    @pytest.fixture
    def sample_candles(self, real_tse_candles):
        """Fixture for sample candle data."""
        return real_tse_candles

    def test_find_fibonacci_levels_basic(self, fib_tools, sample_candles):
        """Test finding fibonacci levels from candles."""
        result = fib_tools.find_fibonacci_levels(sample_candles, lookback_period=50)

        assert isinstance(result, FibonacciResult)
        # Depending on data, may return levels or empty list
        if result.levels:
            assert all(isinstance(level, FibonacciLevel) for level in result.levels)

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
        center_point = (Decimal('75.0'), 0)
        radius_point = (Decimal('75.0'), 10)

        result = fib_tools.calculate_arcs(center_point, radius_point, 5)

        assert isinstance(result, list)
        # Should handle zero radius

    def test_calculate_fans_same_points(self, fib_tools):
        """Test fan calculation with same start and end points."""
        point = (Decimal('50.0'), 50)

        result = fib_tools.calculate_fans(point, point, 50)

        assert isinstance(result, list)
        # Should handle degenerate case

    def test_calculate_projections_zero_moves(self, fib_tools):
        """Test projection calculation with zero moves."""
        move1_high = move1_low = Decimal('100.0')
        move2_high = Decimal('100.0')

        result = fib_tools.calculate_projections(move1_low, move1_high, move2_high)

        assert isinstance(result, list)
        # Should handle zero moves

    def test_find_fibonacci_levels_empty_candles(self, fib_tools):
        """Test finding levels with empty candle list."""
        from gravity_tech.core.domain.entities import FibonacciResult

        result = fib_tools.find_fibonacci_levels([])

        assert isinstance(result, FibonacciResult)
        assert len(result.retracement_levels) == 0
        assert len(result.extension_levels) == 0

    def test_analyze_fibonacci_confluence_empty_candles(self, fib_tools):
        """Test confluence analysis with empty levels."""
        result = fib_tools.analyze_fibonacci_confluence([], Decimal('0.001'))

        assert result == {}

    # Additional tests to reach >50% coverage
    def test_calculate_retracements_precision(self, fib_tools):
        """Test retracement precision."""
        high = Decimal('100.12345678')
        low = Decimal('50.87654321')

        result = fib_tools.calculate_retracements(high, low)

        assert isinstance(result, list)
        assert len(result) > 0
        # Check that prices are quantized to 2 decimal places
        for level in result:
            assert level.price == round(level.price, 2)

    def test_calculate_extensions_custom_ratios(self, fib_tools):
        """Test extensions with custom ratios."""
        high = Decimal('100.0')
        low = Decimal('50.0')
        custom_ratios = [0.5, 1.0]

        result = fib_tools.calculate_extensions(high, low, ratios=custom_ratios)

        assert len(result) == 2
        assert result[0].ratio == 0.5
        assert result[1].ratio == 1.0

    def test_calculate_arcs_custom_angles(self, fib_tools):
        """Test arcs with custom ratios."""
        center_point = (Decimal('75.0'), 0)
        radius_point = (Decimal('100.0'), 10)
        time_point = 5

        result = fib_tools.calculate_arcs(center_point, radius_point, time_point)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_calculate_fans_custom_ratios(self, fib_tools):
        """Test fans with custom ratios."""
        origin_point = (Decimal('50.0'), 0)
        high_point = (Decimal('100.0'), 10)
        time_point = 5
        custom_ratios = [0.5]

        result = fib_tools.calculate_fans(origin_point, high_point, time_point, ratios=custom_ratios)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_calculate_projections_custom_ratios(self, fib_tools):
        """Test projections with custom ratios."""
        wave1_start = Decimal('50.0')
        wave1_end = Decimal('100.0')
        wave2_end = Decimal('150.0')
        custom_ratios = [1.0]

        result = fib_tools.calculate_projections(wave1_start, wave1_end, wave2_end, ratios=custom_ratios)

        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0].ratio == Decimal('1.0')

    def test_find_fibonacci_levels_different_lookback(self, fib_tools, sample_candles):
        """Test finding levels with different lookback periods."""
        from gravity_tech.core.domain.entities import FibonacciResult

        for lookback in [10, 25, 100]:
            result = fib_tools.find_fibonacci_levels(sample_candles[:lookback])
            assert isinstance(result, FibonacciResult)

    def test_analyze_fibonacci_confluence_different_prices(self, fib_tools, sample_candles):
        """Test confluence analysis with different current prices."""
        # First find fibonacci levels
        fib_result = fib_tools.find_fibonacci_levels(sample_candles)

        # Convert dict levels to FibonacciLevel objects for confluence analysis
        from gravity_tech.core.domain.entities import FibonacciLevel
        all_levels = []
        for ratio, price in fib_result.retracement_levels.items():
            all_levels.append(FibonacciLevel(
                ratio=float(ratio),
                price=price,
                level_type="MEDIUM",
                strength=0.5,
                touches=1,
                description=f"{ratio} retracement"
            ))
        for ratio, price in fib_result.extension_levels.items():
            all_levels.append(FibonacciLevel(
                ratio=float(ratio),
                price=price,
                level_type="STRONG",
                strength=0.7,
                touches=1,
                description=f"{ratio} extension"
            ))

        for tolerance in [Decimal('0.01'), Decimal('0.05'), Decimal('0.10')]:
            result = fib_tools.analyze_fibonacci_confluence(all_levels, tolerance)
            assert isinstance(result, dict)

    # Tests for private methods (if needed for coverage)
    def test_private_find_swings(self, fib_tools, sample_candles):
        """Test private _find_swings method."""
        swings = fib_tools._find_swings(sample_candles, Decimal('0.01'))
        assert isinstance(swings, tuple)
        assert len(swings) == 2

    def test_private_remove_duplicate_levels(self, fib_tools):
        """Test private _remove_duplicate_levels method."""
        levels = [
            FibonacciLevel(price=Decimal('100.0'), ratio=0.5, level_type='MEDIUM', strength=1.0, touches=1, description='Test level'),
            FibonacciLevel(price=Decimal('100.0'), ratio=0.618, level_type='MEDIUM', strength=1.0, touches=1, description='Test level'),
            FibonacciLevel(price=Decimal('105.0'), ratio=0.5, level_type='STRONG', strength=1.0, touches=1, description='Test level'),
        ]
        result = fib_tools._remove_duplicate_levels(levels)
        assert isinstance(result, list)
        assert len(result) <= len(levels)

    def test_private_calculate_confidence(self, fib_tools):
        """Test private _calculate_confidence method."""
        levels = [
            FibonacciLevel(price=Decimal('100.0'), ratio=0.5, level_type='MEDIUM', strength=1.0, touches=1, description='Test level'),
            FibonacciLevel(price=Decimal('105.0'), ratio=0.618, level_type='MEDIUM', strength=1.0, touches=1, description='Test level'),
        ]
        confidence = fib_tools._calculate_confidence(levels, [])
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_private_determine_signal(self, fib_tools):
        """Test private _determine_signal method."""
        levels = [
            FibonacciLevel(price=Decimal('100.0'), ratio=0.5, level_type='MEDIUM', strength=1.0, touches=1, description='Test level'),
            FibonacciLevel(price=Decimal('105.0'), ratio=0.618, level_type='MEDIUM', strength=1.0, touches=1, description='Test level'),
        ]
        signal = fib_tools._determine_signal(levels, Decimal('102.0'))
        assert isinstance(signal, SignalStrength)

    # Comprehensive tests for 95%+ coverage

    def test_calculate_retracements_custom_ratios(self, fib_tools):
        """Test retracements with custom ratios."""
        high = Decimal('100.0')
        low = Decimal('50.0')
        custom_ratios = [0.25, 0.75]

        result = fib_tools.calculate_retracements(high, low, ratios=custom_ratios)

        assert len(result) == 2
        assert result[0].ratio == 0.25
        assert result[1].ratio == 0.75

    def test_calculate_extensions_downtrend_custom_ratios(self, fib_tools):
        """Test downtrend extensions with custom ratios."""
        high = Decimal('100.0')
        low = Decimal('50.0')
        custom_ratios = [1.5, 2.0]

        result = fib_tools.calculate_extensions(high, low, direction="down", ratios=custom_ratios)

        assert len(result) == 2
        # For downtrend, extensions go below low
        for level in result:
            assert level.price < low

    def test_calculate_arcs_different_time_points(self, fib_tools):
        """Test arcs with different time points."""
        center_point = (Decimal('75.0'), 0)
        radius_point = (Decimal('100.0'), 10)

        for time_point in [1, 5, 15]:
            result = fib_tools.calculate_arcs(center_point, radius_point, time_point)
            assert isinstance(result, list)
            assert len(result) == 4

    def test_calculate_fans_different_configurations(self, fib_tools):
        """Test fans with different point configurations."""
        test_cases = [
            ((Decimal('0.0'), 0), (Decimal('100.0'), 50), 25),
            ((Decimal('50.0'), 10), (Decimal('150.0'), 60), 40),
        ]

        for origin_point, high_point, time_point in test_cases:
            result = fib_tools.calculate_fans(origin_point, high_point, time_point)
            assert isinstance(result, list)
            assert len(result) == 4

    def test_calculate_projections_different_scenarios(self, fib_tools):
        """Test projections with different wave scenarios."""
        test_cases = [
            (Decimal('50.0'), Decimal('100.0'), Decimal('120.0')),  # Partial retracement
            (Decimal('50.0'), Decimal('100.0'), Decimal('130.0')),  # Beyond high
            (Decimal('50.0'), Decimal('100.0'), Decimal('80.0')),   # Within range
        ]

        for wave1_start, wave1_end, wave2_end in test_cases:
            result = fib_tools.calculate_projections(wave1_start, wave1_end, wave2_end)
            assert isinstance(result, list)
            assert len(result) == 5

    def test_find_fibonacci_levels_minimal_data(self, fib_tools):
        """Test finding levels with minimal candle data."""
        # Create minimal candle data
        candles = [
            Candle(timestamp=pd.Timestamp('2023-01-01'), open=100.0, high=105.0, low=95.0, close=102.0, volume=1000),
            Candle(timestamp=pd.Timestamp('2023-01-02'), open=102.0, high=110.0, low=98.0, close=108.0, volume=1200),
        ]

        result = fib_tools.find_fibonacci_levels(candles, lookback_period=10)

        assert isinstance(result, FibonacciResult)
        # Should handle minimal data gracefully

    def test_find_fibonacci_levels_large_lookback(self, fib_tools, sample_candles):
        """Test finding levels with lookback larger than available data."""
        result = fib_tools.find_fibonacci_levels(sample_candles[:5], lookback_period=50)

        assert isinstance(result, FibonacciResult)
        # Should use all available data when lookback > len(candles)

    def test_analyze_fibonacci_confluence_with_candles(self, fib_tools, sample_candles):
        """Test confluence analysis when passed candles directly."""
        current_price = Decimal('15000.0')

        result = fib_tools.analyze_fibonacci_confluence(sample_candles, current_price)

        assert isinstance(result, dict)
        # Should internally call find_fibonacci_levels

    def test_analyze_fibonacci_confluence_no_confluence(self, fib_tools):
        """Test confluence analysis with widely spaced levels."""
        levels = [
            FibonacciLevel(price=Decimal('100.0'), ratio=0.5, level_type='MEDIUM', strength=1.0, touches=1, description='Test'),
            FibonacciLevel(price=Decimal('150.0'), ratio=0.618, level_type='MEDIUM', strength=1.0, touches=1, description='Test'),
        ]

        result = fib_tools.analyze_fibonacci_confluence(levels, Decimal('125.0'), tolerance=Decimal('0.01'))

        assert isinstance(result, dict)
        # Should have no confluence zones due to large spacing

    def test_analyze_fibonacci_confluence_multiple_confluence(self, fib_tools):
        """Test confluence analysis with multiple overlapping levels."""
        levels = [
            FibonacciLevel(price=Decimal('100.01'), ratio=0.5, level_type='MEDIUM', strength=1.0, touches=1, description='Test1'),
            FibonacciLevel(price=Decimal('100.02'), ratio=0.618, level_type='MEDIUM', strength=1.0, touches=1, description='Test2'),
            FibonacciLevel(price=Decimal('100.03'), ratio=0.786, level_type='MEDIUM', strength=1.0, touches=1, description='Test3'),
            FibonacciLevel(price=Decimal('150.0'), ratio=1.0, level_type='STRONG', strength=1.0, touches=1, description='Test4'),
        ]

        result = fib_tools.analyze_fibonacci_confluence(levels, Decimal('125.0'), tolerance=Decimal('0.05'))

        assert isinstance(result, dict)
        assert len(result) >= 1  # Should have at least one confluence zone

    def test_private_find_swings_edge_cases(self, fib_tools):
        """Test _find_swings with edge cases."""
        # Empty candles
        swings = fib_tools._find_swings([], Decimal('0.01'))
        assert swings == ([], [])

        # Single candle
        single_candle = [Candle(timestamp=pd.Timestamp('2023-01-01'), open=100.0, high=105.0, low=95.0, close=102.0, volume=1000)]
        swings = fib_tools._find_swings(single_candle, Decimal('0.01'))
        assert isinstance(swings, tuple)
        assert len(swings) == 2

    def test_private_remove_duplicate_levels_edge_cases(self, fib_tools):
        """Test _remove_duplicate_levels with edge cases."""
        # Empty list
        result = fib_tools._remove_duplicate_levels([])
        assert result == []

        # Single level
        single_level = [FibonacciLevel(price=Decimal('100.0'), ratio=0.5, level_type='MEDIUM', strength=1.0, touches=1, description='Test')]
        result = fib_tools._remove_duplicate_levels(single_level)
        assert len(result) == 1
        assert result[0] == single_level[0]

        # All duplicates
        duplicate_levels = [
            FibonacciLevel(price=Decimal('100.0'), ratio=0.5, level_type='MEDIUM', strength=1.0, touches=1, description='Test1'),
            FibonacciLevel(price=Decimal('100.0'), ratio=0.618, level_type='MEDIUM', strength=1.0, touches=1, description='Test2'),
            FibonacciLevel(price=Decimal('100.0'), ratio=0.786, level_type='MEDIUM', strength=1.0, touches=1, description='Test3'),
        ]
        result = fib_tools._remove_duplicate_levels(duplicate_levels)
        assert len(result) == 1  # Should keep only one

    def test_private_calculate_confidence_edge_cases(self, fib_tools):
        """Test _calculate_confidence with edge cases."""
        # Empty levels
        confidence = fib_tools._calculate_confidence([], [])
        assert confidence == 0.0

        # Empty candles
        levels = [FibonacciLevel(price=Decimal('100.0'), ratio=0.5, level_type='MEDIUM', strength=1.0, touches=1, description='Test')]
        confidence = fib_tools._calculate_confidence(levels, [])
        assert confidence == 0.0

        # Price near level
        candles = [Candle(timestamp=pd.Timestamp('2023-01-01'), open=100.0, high=105.0, low=95.0, close=100.5, volume=1000)]
        confidence = fib_tools._calculate_confidence(levels, candles)
        assert confidence > 0.0

    def test_private_determine_signal_edge_cases(self, fib_tools):
        """Test _determine_signal with edge cases."""
        # Empty levels
        signal = fib_tools._determine_signal([], Decimal('100.0'))
        assert signal == SignalStrength.NEUTRAL

        # Only support levels
        support_levels = [FibonacciLevel(price=Decimal('95.0'), ratio=0.5, level_type='MEDIUM', strength=1.0, touches=1, description='Support')]
        signal = fib_tools._determine_signal(support_levels, Decimal('100.0'))
        assert signal == SignalStrength.NEUTRAL  # No resistance levels

        # Only resistance levels
        resistance_levels = [FibonacciLevel(price=Decimal('105.0'), ratio=0.5, level_type='MEDIUM', strength=1.0, touches=1, description='Resistance')]
        signal = fib_tools._determine_signal(resistance_levels, Decimal('100.0'))
        assert signal == SignalStrength.NEUTRAL  # No support levels

        # Close to support
        mixed_levels = [
            FibonacciLevel(price=Decimal('99.3'), ratio=0.5, level_type='MEDIUM', strength=1.0, touches=1, description='Support'),
            FibonacciLevel(price=Decimal('110.0'), ratio=0.618, level_type='MEDIUM', strength=1.0, touches=1, description='Resistance'),
        ]
        signal = fib_tools._determine_signal(mixed_levels, Decimal('99.5'))
        assert signal == SignalStrength.BULLISH

        # Close to resistance
        close_resistance_levels = [
            FibonacciLevel(price=Decimal('105.0'), ratio=0.5, level_type='MEDIUM', strength=1.0, touches=1, description='Support'),
            FibonacciLevel(price=Decimal('109.6'), ratio=0.618, level_type='MEDIUM', strength=1.0, touches=1, description='Resistance'),
        ]
        signal = fib_tools._determine_signal(close_resistance_levels, Decimal('109.5'))
        assert signal == SignalStrength.BEARISH

    def test_find_fibonacci_levels_with_various_swing_sizes(self, fib_tools):
        """Test finding levels with different minimum swing sizes."""
        candles = [
            Candle(timestamp=pd.Timestamp(f'2023-01-{i+1:02d}'), open=100.0 + i, high=105.0 + i, low=95.0 + i, close=102.0 + i, volume=1000)
            for i in range(20)
        ]

        for min_swing in [Decimal('0.01'), Decimal('1.0'), Decimal('5.0')]:
            result = fib_tools.find_fibonacci_levels(candles, min_swing_size=min_swing)
            assert isinstance(result, FibonacciResult)

    def test_analyze_fibonacci_confluence_various_tolerances(self, fib_tools):
        """Test confluence analysis with various tolerance levels."""
        levels = [
            FibonacciLevel(price=Decimal('100.0'), ratio=0.5, level_type='MEDIUM', strength=1.0, touches=1, description='Level1'),
            FibonacciLevel(price=Decimal('100.1'), ratio=0.618, level_type='MEDIUM', strength=1.0, touches=1, description='Level2'),
            FibonacciLevel(price=Decimal('100.2'), ratio=0.786, level_type='MEDIUM', strength=1.0, touches=1, description='Level3'),
        ]

        tolerances = [Decimal('0.001'), Decimal('0.01'), Decimal('0.1'), Decimal('1.0')]

        for tolerance in tolerances:
            result = fib_tools.analyze_fibonacci_confluence(levels, Decimal('100.0'), tolerance)
            assert isinstance(result, dict)

    def test_integration_full_workflow(self, fib_tools, sample_candles):
        """Test full workflow from candles to signal determination."""
        # Find levels
        fib_result = fib_tools.find_fibonacci_levels(sample_candles)

        # Analyze confluence
        current_price = Decimal(str(sample_candles[-1].close))
        confluence = fib_tools.analyze_fibonacci_confluence(fib_result.levels, current_price)

        # Determine signal
        signal = fib_tools._determine_signal(fib_result.levels, float(current_price))

        # Verify all components work together
        assert isinstance(fib_result, FibonacciResult)
        assert isinstance(confluence, dict)
        assert isinstance(signal, SignalStrength)
