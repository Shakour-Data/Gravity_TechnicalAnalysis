"""
Comprehensive Tests for Fibonacci Tools

Tests cover:
- Fibonacci retracements calculation
- Fibonacci extensions calculation
- Fibonacci projections
- Integration with TSE database data
- Edge cases and error handling

Author: Gravity Tech Team
Date: December 4, 2025
Version: 1.0.0
License: MIT
"""

import pytest
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import List
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from gravity_tech.core.domain.entities import Candle
from gravity_tech.analysis.fibonacci_tools import FibonacciTools


class TestFibonacciTools:
    """Test suite for Fibonacci tools with TSE data integration."""

    @pytest.fixture
    def tse_db_connection(self):
        """Fixture to provide TSE database connection."""
        db_path = project_root / "data" / "tse_data.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        yield conn
        conn.close()

    @pytest.fixture
    def sample_tse_candles(self, tse_db_connection) -> List[Candle]:
        """Load real TSE candle data for testing."""
        cursor = tse_db_connection.cursor()
        cursor.execute("""
            SELECT * FROM candles
            WHERE symbol = 'شستا'
            ORDER BY timestamp ASC
            LIMIT 100
        """)

        candles = []
        for row in cursor.fetchall():
            candles.append(Candle(
                timestamp=datetime.fromisoformat(row['timestamp']),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume'])
            ))

        return candles

    @pytest.fixture
    def fibonacci_tools(self):
        """Fixture to provide FibonacciTools instance."""
        return FibonacciTools()

    def test_fibonacci_tools_initialization(self, fibonacci_tools):
        """Test FibonacciTools initialization."""
        assert fibonacci_tools is not None
        assert hasattr(fibonacci_tools, 'calculate_retracements')
        assert hasattr(fibonacci_tools, 'calculate_extensions')

    def test_calculate_retracements_basic(self, fibonacci_tools):
        """Test basic Fibonacci retracement calculation."""
        high = 100.0
        low = 50.0

        retracements = fibonacci_tools.calculate_retracements(high, low)

        assert isinstance(retracements, dict)
        assert 'levels' in retracements
        assert 'description' in retracements

        levels = retracements['levels']
        expected_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

        for level in expected_levels:
            assert level in levels
            # Check that retracement values are between high and low
            assert low <= levels[level] <= high

    def test_calculate_retracements_with_tse_data(self, fibonacci_tools, sample_tse_candles):
        """Test retracements calculation with real TSE data."""
        # Find swing high and low in the data
        highs = [c.high for c in sample_tse_candles]
        lows = [c.low for c in sample_tse_candles]

        swing_high = max(highs)
        swing_low = min(lows)

        retracements = fibonacci_tools.calculate_retracements(swing_high, swing_low)

        assert retracements['levels'][0.0] == swing_high  # 0% retracement
        assert retracements['levels'][1.0] == swing_low   # 100% retracement

        # Check golden ratio level
        golden_ratio_level = retracements['levels'][0.618]
        expected_golden = swing_high - (swing_high - swing_low) * 0.618
        assert abs(golden_ratio_level - expected_golden) < 0.01

    def test_calculate_extensions_basic(self, fibonacci_tools):
        """Test basic Fibonacci extension calculation."""
        start_point = 50.0
        end_point = 100.0

        extensions = fibonacci_tools.calculate_extensions(start_point, end_point)

        assert isinstance(extensions, dict)
        assert 'levels' in extensions
        assert 'description' in extensions

        levels = extensions['levels']
        expected_levels = [1.0, 1.236, 1.382, 1.5, 1.618, 2.0, 2.618]

        for level in expected_levels:
            assert level in levels

    def test_calculate_extensions_with_tse_trend(self, fibonacci_tools, sample_tse_candles):
        """Test extensions with TSE trend data."""
        # Find a clear uptrend segment
        trend_start = sample_tse_candles[0].close
        trend_end = sample_tse_candles[50].close

        extensions = fibonacci_tools.calculate_extensions(trend_start, trend_end)

        # 100% extension should equal the end point
        assert extensions['levels'][1.0] == trend_end

        # 161.8% extension should be higher than end point
        assert extensions['levels'][1.618] > trend_end

    def test_fibonacci_projections(self, fibonacci_tools, sample_tse_candles):
        """Test Fibonacci projections with TSE data."""
        # Use first 20 candles for wave analysis
        wave1_data = sample_tse_candles[:20]
        wave2_data = sample_tse_candles[20:40]

        wave1_high = max(c.high for c in wave1_data)
        wave1_low = min(c.low for c in wave1_data)
        wave2_high = max(c.high for c in wave2_data)
        wave2_low = min(c.low for c in wave2_data)

        # Calculate projection
        projection = fibonacci_tools.calculate_projection(
            wave1_high, wave1_low, wave2_high, wave2_low
        )

        assert isinstance(projection, dict)
        assert 'projection_levels' in projection
        assert 'description' in projection

    def test_fibonacci_time_zones(self, fibonacci_tools, sample_tse_candles):
        """Test Fibonacci time zones calculation."""
        start_date = sample_tse_candles[0].timestamp
        end_date = sample_tse_candles[-1].timestamp

        time_zones = fibonacci_tools.calculate_time_zones(start_date, end_date)

        assert isinstance(time_zones, dict)
        assert 'time_zones' in time_zones
        assert 'description' in time_zones

        zones = time_zones['time_zones']
        assert len(zones) > 0

        # Check that time zones are in chronological order
        zone_times = [zone['timestamp'] for zone in zones]
        assert zone_times == sorted(zone_times)

    def test_fibonacci_arcs(self, fibonacci_tools, sample_tse_candles):
        """Test Fibonacci arcs calculation."""
        center_point = sample_tse_candles[25]  # Middle point
        radius_end = sample_tse_candles[50]   # End point for radius

        arcs = fibonacci_tools.calculate_arcs(center_point.close, radius_end.close)

        assert isinstance(arcs, dict)
        assert 'arcs' in arcs
        assert 'description' in arcs

    def test_fibonacci_fans(self, fibonacci_tools, sample_tse_candles):
        """Test Fibonacci fan lines."""
        anchor_point = sample_tse_candles[0].close
        trend_end = sample_tse_candles[30].close

        fans = fibonacci_tools.calculate_fans(anchor_point, trend_end)

        assert isinstance(fans, dict)
        assert 'fan_lines' in fans
        assert 'description' in fans

    def test_edge_cases_zero_range(self, fibonacci_tools):
        """Test edge case with zero price range."""
        high = 100.0
        low = 100.0  # Same as high

        retracements = fibonacci_tools.calculate_retracements(high, low)

        # All retracement levels should be the same
        for level_value in retracements['levels'].values():
            assert level_value == high

    def test_edge_cases_negative_prices(self, fibonacci_tools):
        """Test with negative price values."""
        high = -50.0
        low = -100.0

        retracements = fibonacci_tools.calculate_retracements(high, low)

        assert retracements['levels'][0.0] == high
        assert retracements['levels'][1.0] == low

    def test_multiple_symbols_tse_data(self, fibonacci_tools, tse_db_connection):
        """Test Fibonacci calculations across multiple TSE symbols."""
        cursor = tse_db_connection.cursor()

        symbols = ['شستا', 'فملی', 'وبملт']
        results = {}

        for symbol in symbols:
            cursor.execute("""
                SELECT open, high, low, close FROM candles
                WHERE symbol = ?
                ORDER BY timestamp ASC
                LIMIT 50
            """, (symbol,))

            prices = cursor.fetchall()
            if prices:
                highs = [row['high'] for row in prices]
                lows = [row['low'] for row in prices]

                swing_high = max(highs)
                swing_low = min(lows)

                retracements = fibonacci_tools.calculate_retracements(swing_high, swing_low)
                results[symbol] = retracements

        assert len(results) == len(symbols)
        for symbol, retracements in results.items():
            assert 'levels' in retracements
            assert len(retracements['levels']) > 0

    def test_fibonacci_integration_with_indicators(self, fibonacci_tools, sample_tse_candles):
        """Test Fibonacci tools integration with technical indicators."""
        # Calculate retracements for the entire dataset
        highs = [c.high for c in sample_tse_candles]
        lows = [c.low for c in sample_tse_candles]

        swing_high = max(highs)
        swing_low = min(lows)

        retracements = fibonacci_tools.calculate_retracements(swing_high, swing_low)

        # Check that retracement levels make mathematical sense
        assert retracements['levels'][0.5] == (swing_high + swing_low) / 2
        assert retracements['levels'][0.618] < retracements['levels'][0.5]

    def test_performance_large_dataset(self, fibonacci_tools, tse_db_connection):
        """Test performance with large TSE dataset."""
        import time

        cursor = tse_db_connection.cursor()
        cursor.execute("""
            SELECT high, low FROM candles
            WHERE symbol = 'شستا'
            ORDER BY timestamp ASC
        """)

        prices = cursor.fetchall()
        highs = [row['high'] for row in prices]
        lows = [row['low'] for row in prices]

        swing_high = max(highs)
        swing_low = min(lows)

        # Measure performance
        start_time = time.time()
        retracements = fibonacci_tools.calculate_retracements(swing_high, swing_low)
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete in reasonable time
        assert execution_time < 1.0  # Less than 1 second
        assert len(retracements['levels']) > 0

    def test_fibonacci_levels_consistency(self, fibonacci_tools):
        """Test that Fibonacci levels are consistent across different calculations."""
        high = 1000.0
        low = 500.0

        retracements1 = fibonacci_tools.calculate_retracements(high, low)
        retracements2 = fibonacci_tools.calculate_retracements(high, low)

        # Results should be identical
        assert retracements1['levels'] == retracements2['levels']

        # Check specific mathematical relationships
        levels = retracements1['levels']
        assert abs(levels[0.5] - 750.0) < 0.01  # Midpoint
        assert abs(levels[0.618] - (high - (high - low) * 0.618)) < 0.01  # Golden ratio

    def test_error_handling_invalid_inputs(self, fibonacci_tools):
        """Test error handling for invalid inputs."""
        # Test with None values
        with pytest.raises((TypeError, AttributeError)):
            fibonacci_tools.calculate_retracements(None, 100)

        with pytest.raises((TypeError, AttributeError)):
            fibonacci_tools.calculate_retracements(100, None)

        # Test with non-numeric values
        with pytest.raises((TypeError, ValueError)):
            fibonacci_tools.calculate_retracements("high", "low")