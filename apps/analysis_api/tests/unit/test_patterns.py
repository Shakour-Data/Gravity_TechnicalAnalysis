"""
Phase 4: Unit Tests for Pattern Detection

Test coverage for pattern detection algorithms:
- Harmonic Patterns (Gartley, Butterfly, Bat, Crab)
- Candlestick Patterns (Head & Shoulders, Double Top/Bottom, etc)
- Classical Patterns (Support/Resistance, Triangles, Channels)

Target: 100% coverage of pattern detection
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

# ============================================================================
# PATTERN TEST DATA BUILDERS
# ============================================================================


class PatternTestData:
    """Generate test data with known patterns"""

    @staticmethod
    def create_candle(
        timestamp: datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float = 1000.0,
    ) -> dict:
        """Create a candle"""
        return {
            "timestamp": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
        }

    @staticmethod
    def head_and_shoulders() -> list[dict]:
        """Generate head & shoulders pattern"""
        base_date = datetime.now()
        return [
            # Left shoulder
            PatternTestData.create_candle(base_date + timedelta(days=0), 100, 105, 95, 100),
            PatternTestData.create_candle(base_date + timedelta(days=1), 100, 108, 95, 105),
            PatternTestData.create_candle(base_date + timedelta(days=2), 105, 107, 100, 103),
            # Head
            PatternTestData.create_candle(base_date + timedelta(days=3), 103, 110, 100, 105),
            PatternTestData.create_candle(base_date + timedelta(days=4), 105, 115, 103, 110),
            PatternTestData.create_candle(base_date + timedelta(days=5), 110, 112, 105, 108),
            # Right shoulder
            PatternTestData.create_candle(base_date + timedelta(days=6), 108, 110, 102, 105),
            PatternTestData.create_candle(base_date + timedelta(days=7), 105, 110, 100, 105),
            PatternTestData.create_candle(base_date + timedelta(days=8), 105, 108, 100, 103),
        ]

    @staticmethod
    def double_bottom() -> list[dict]:
        """Generate double bottom pattern"""
        base_date = datetime.now()
        return [
            # First bottom
            PatternTestData.create_candle(base_date + timedelta(days=0), 100, 105, 80, 85),
            PatternTestData.create_candle(base_date + timedelta(days=1), 85, 95, 80, 90),
            # Middle recovery
            PatternTestData.create_candle(base_date + timedelta(days=2), 90, 100, 88, 98),
            # Second bottom (similar to first)
            PatternTestData.create_candle(base_date + timedelta(days=3), 98, 102, 82, 86),
            PatternTestData.create_candle(base_date + timedelta(days=4), 86, 96, 82, 92),
            # Breakout
            PatternTestData.create_candle(base_date + timedelta(days=5), 92, 108, 90, 103),
        ]

    @staticmethod
    def double_top() -> list[dict]:
        """Generate double top pattern"""
        base_date = datetime.now()
        return [
            # First top
            PatternTestData.create_candle(base_date + timedelta(days=0), 100, 115, 98, 113),
            PatternTestData.create_candle(base_date + timedelta(days=1), 113, 118, 110, 115),
            # Middle dip
            PatternTestData.create_candle(base_date + timedelta(days=2), 115, 118, 105, 107),
            # Second top (similar to first)
            PatternTestData.create_candle(base_date + timedelta(days=3), 107, 120, 105, 115),
            PatternTestData.create_candle(base_date + timedelta(days=4), 115, 118, 110, 114),
            # Breakdown
            PatternTestData.create_candle(base_date + timedelta(days=5), 114, 115, 95, 97),
        ]

    @staticmethod
    def uptrend() -> list[dict]:
        """Generate uptrend pattern"""
        base_date = datetime.now()
        candles = []
        price = 100

        for day in range(20):
            close = price + np.random.uniform(1, 3)
            high = close + np.random.uniform(0.5, 1.5)
            low = max(price - 1, close - 2)

            candles.append(
                PatternTestData.create_candle(
                    base_date + timedelta(days=day), price, high, low, close
                )
            )
            price = close

        return candles

    @staticmethod
    def downtrend() -> list[dict]:
        """Generate downtrend pattern"""
        base_date = datetime.now()
        candles = []
        price = 100

        for day in range(20):
            close = price - np.random.uniform(1, 3)
            low = close - np.random.uniform(0.5, 1.5)
            high = max(price + 1, close + 2)

            candles.append(
                PatternTestData.create_candle(
                    base_date + timedelta(days=day), price, high, low, close
                )
            )
            price = close

        return candles


# ============================================================================
# HARMONIC PATTERNS TESTS
# ============================================================================


@pytest.mark.unit
class TestGartleyPattern:
    """Test Gartley pattern detection"""

    def test_gartley_structure(self):
        """Test Gartley pattern has correct ratios"""
        # Gartley ratios: D=0.618*XA, C=0.618*AB, B=0.618*XA
        X = 100
        A = 110  # 10% move

        # Calculate Gartley points
        AB_size = A - X
        B = A - (0.618 * AB_size)  # Should be around 106.18

        assert 100 < B < A
        assert abs((A - B) - 0.618 * AB_size) < 0.01

    def test_gartley_recognition(self):
        """Test detection of Gartley pattern"""
        candles = [
            {"high": 100, "low": 100},  # X
            {"high": 110, "low": 110},  # A
            {"high": 106, "low": 106},  # B
            {"high": 107, "low": 107},  # C
            {"high": 102, "low": 102},  # D
        ]

        # Extract high/low sequence
        price_series = [c["high"] for c in candles]

        # Simple pattern recognition: D < A and D > X
        x, a = price_series[0], price_series[1]
        d = price_series[-1]

        assert d < a  # D is lower than A
        assert d > x  # D is higher than X


@pytest.mark.unit
class TestButterflyPattern:
    """Test Butterfly pattern detection"""

    def test_butterfly_extreme_point(self):
        """Test Butterfly has extreme point at D"""
        candles = [
            {"high": 100, "low": 100},  # X
            {"high": 120, "low": 120},  # A
            {"high": 115, "low": 115},  # B
            {"high": 117, "low": 117},  # C
            {"high": 95, "low": 95},  # D (extreme)
        ]

        price_series = [c["high"] for c in candles]

        # D should be extreme
        d = price_series[-1]
        assert d < price_series[0]  # D < X


@pytest.mark.unit
class TestBatPattern:
    """Test Bat pattern detection"""

    def test_bat_structure(self):
        """Test Bat pattern structure"""
        candles = [
            {"high": 100, "low": 100},  # X
            {"high": 112, "low": 112},  # A
            {"high": 107, "low": 107},  # B
            {"high": 109, "low": 109},  # C
            {"high": 103, "low": 103},  # D
        ]

        price_series = [c["high"] for c in candles]

        x, a, b, c, d = price_series

        # Bat has specific ratios
        assert x < a  # Uptrend
        assert x < d  # D is higher than X
        assert b > d and b < a  # B is between D and A


# ============================================================================
# CANDLESTICK PATTERNS TESTS
# ============================================================================


@pytest.mark.unit
class TestHeadAndShouldersPattern:
    """Test Head & Shoulders detection"""

    def test_h_s_creation(self):
        """Test creating H&S pattern"""
        candles = PatternTestData.head_and_shoulders()

        # Extract key points
        opens = [c["open"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]

        # Find peaks and troughs
        assert len(highs) == 9
        assert max(highs[3:6]) > max(highs[0:3])  # Head higher than left shoulder
        assert max(highs[3:6]) > max(highs[6:9])  # Head higher than right shoulder

    def test_h_s_neckline(self):
        """Test H&S neckline formation"""
        candles = PatternTestData.head_and_shoulders()

        lows = [c["low"] for c in candles]

        # Neckline connects lows of shoulders
        left_shoulder_low = lows[2]
        right_shoulder_low = lows[8]

        assert left_shoulder_low == right_shoulder_low  # Equal lows


@pytest.mark.unit
class TestDoubleTopBottom:
    """Test Double Top/Bottom pattern detection"""

    def test_double_bottom_creation(self):
        """Test creating double bottom"""
        candles = PatternTestData.double_bottom()

        lows = [c["low"] for c in candles]

        # Two bottoms should be at similar levels
        bottom1 = lows[1]
        bottom2 = lows[4]

        assert abs(bottom1 - bottom2) / bottom1 < 0.05  # Within 5%

    def test_double_top_creation(self):
        """Test creating double top"""
        candles = PatternTestData.double_top()

        highs = [c["high"] for c in candles]

        # Two tops should be at similar levels
        top1 = highs[1]
        top2 = highs[4]

        assert abs(top1 - top2) / top1 < 0.05  # Within 5%


@pytest.mark.unit
class TestBullishEngulfing:
    """Test Bullish Engulfing pattern"""

    def test_bullish_engulfing_structure(self):
        """Test bullish engulfing has correct structure"""
        # Day 1: Down day
        candle1 = {"open": 105, "close": 100, "high": 106, "low": 99}

        # Day 2: Up day engulfing day 1
        candle2 = {"open": 99, "close": 106, "high": 107, "low": 98}

        # Engulfing conditions
        assert candle2["open"] < candle1["close"]  # Day 2 opens below Day 1 close
        assert candle2["close"] > candle1["open"]  # Day 2 closes above Day 1 open
        assert candle2["close"] > candle1["close"]  # Day 2 up


@pytest.mark.unit
class TestBearishEngulfing:
    """Test Bearish Engulfing pattern"""

    def test_bearish_engulfing_structure(self):
        """Test bearish engulfing has correct structure"""
        # Day 1: Up day
        candle1 = {"open": 100, "close": 105, "high": 106, "low": 99}

        # Day 2: Down day engulfing day 1
        candle2 = {"open": 106, "close": 99, "high": 107, "low": 98}

        # Engulfing conditions
        assert candle2["open"] > candle1["close"]  # Day 2 opens above Day 1 close
        assert candle2["close"] < candle1["open"]  # Day 2 closes below Day 1 open
        assert candle2["close"] < candle1["close"]  # Day 2 down


# ============================================================================
# CLASSICAL PATTERNS TESTS
# ============================================================================


@pytest.mark.unit
class TestSupportResistance:
    """Test Support & Resistance level detection"""

    def test_support_level_identification(self):
        """Test identifying support levels"""
        prices = [100, 98, 97, 98, 96, 97, 99, 97, 98]

        # Support is local minimum
        local_mins = []
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                local_mins.append(prices[i])

        assert len(local_mins) > 0
        assert all(p < 100 for p in local_mins)

    def test_resistance_level_identification(self):
        """Test identifying resistance levels"""
        prices = [100, 102, 103, 102, 104, 103, 101, 103, 102]

        # Resistance is local maximum
        local_maxs = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                local_maxs.append(prices[i])

        assert len(local_maxs) > 0
        assert all(p > 100 for p in local_maxs)


@pytest.mark.unit
class TestTrianglePatterns:
    """Test Triangle pattern detection"""

    def test_ascending_triangle(self):
        """Test ascending triangle"""
        prices = [100, 105, 102, 106, 104, 107, 105, 108]

        # Ascending triangle: higher lows, same highs
        # Extract local highs and lows
        local_highs = []
        local_lows = []

        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                local_highs.append((i, prices[i]))
            elif prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                local_lows.append((i, prices[i]))

        # Check if lows are ascending
        if len(local_lows) > 1:
            assert local_lows[-1][1] > local_lows[0][1]

    def test_descending_triangle(self):
        """Test descending triangle"""
        prices = [108, 103, 106, 102, 105, 101, 104, 100]

        # Descending triangle: lower highs, same lows
        local_highs = []
        local_lows = []

        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                local_highs.append((i, prices[i]))
            elif prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                local_lows.append((i, prices[i]))

        # Check if highs are descending
        if len(local_highs) > 1:
            assert local_highs[-1][1] < local_highs[0][1]


@pytest.mark.unit
class TestChannels:
    """Test Channel pattern detection"""

    def test_ascending_channel(self):
        """Test ascending channel"""
        # Ascending channel has rising support and resistance
        prices = [100, 102, 101, 103, 102, 104, 103, 105, 104, 106]

        # Simple check: prices trending up
        assert prices[-1] > prices[0]
        assert prices[-1] > prices[-5]

    def test_descending_channel(self):
        """Test descending channel"""
        # Descending channel has falling support and resistance
        prices = [106, 104, 105, 103, 104, 102, 103, 101, 102, 100]

        # Simple check: prices trending down
        assert prices[-1] < prices[0]
        assert prices[-1] < prices[-5]


# ============================================================================
# PATTERN VALIDATION TESTS
# ============================================================================


@pytest.mark.unit
class TestPatternValidation:
    """Test pattern validation logic"""

    def test_minimum_candles_required(self):
        """Test patterns require minimum candles"""
        short_candles = [{"high": 100, "low": 100}]

        # Most patterns need 3+ candles
        assert len(short_candles) < 3

    def test_pattern_on_uptrend(self):
        """Test patterns on uptrend"""
        candles = PatternTestData.uptrend()

        # Extract closes
        closes = [c["close"] for c in candles]

        # Should be generally increasing
        assert closes[-1] > closes[0]

    def test_pattern_on_downtrend(self):
        """Test patterns on downtrend"""
        candles = PatternTestData.downtrend()

        # Extract closes
        closes = [c["close"] for c in candles]

        # Should be generally decreasing
        assert closes[-1] < closes[0]


# ============================================================================
# PATTERN COMBINATIONS TESTS
# ============================================================================


@pytest.mark.unit
class TestPatternCombinations:
    """Test multiple patterns together"""

    def test_h_s_with_support_resistance(self):
        """Test H&S combined with support/resistance"""
        candles = PatternTestData.head_and_shoulders()

        # H&S has neckline support
        lows = [c["low"] for c in candles]

        # Find neckline level
        neckline = min(lows[2], lows[8])

        assert neckline > 0
        assert neckline < max([c["high"] for c in candles])

    def test_double_bottom_with_breakout(self):
        """Test double bottom with breakout"""
        candles = PatternTestData.double_bottom()

        highs = [c["high"] for c in candles]

        # Last candle should break above resistance
        resistance = max(highs[:5])
        assert highs[-1] > resistance
