"""
Unit tests for Support and Resistance indicators (src/core/indicators/support_resistance.py)

Tests cover:
- All 4 support/resistance indicators (Pivot Points, Fibonacci Retracement, Camarilla Pivots, Support/Resistance Levels)
- Signal generation and confidence levels
- Edge cases and error handling
- Price position calculations
- Fibonacci level calculations
- Pivot point calculations
- Backward compatibility methods
- Helper functions
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from src.core.indicators.support_resistance import (
    SupportResistanceIndicators
)
from src.core.domain.entities import (
    Candle,
    IndicatorResult,
    CoreSignalStrength as SignalStrength,
    IndicatorCategory
)


class TestSupportResistanceIndicators:
    """Test SupportResistanceIndicators class methods"""

    @pytest.fixture
    def sample_candles(self):
        """Create sample candles for testing"""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        # Create trending up pattern with clear highs/lows
        prices = []
        for i in range(60):
            # Upward trend with some volatility
            base_price = 100 + i * 0.5
            price = base_price + 2 * np.sin(2 * np.pi * i / 10)  # Add some oscillation
            prices.append(price)

        candles = []
        for i, price in enumerate(prices):
            candle = Candle(
                timestamp=base_time,
                open=price - 0.5,
                high=price + 1.5,
                low=price - 1.5,
                close=price,
                volume=1000 + i * 10
            )
            candles.append(candle)

        return candles

    @pytest.fixture
    def minimal_candles(self):
        """Create minimal candles for edge case testing"""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        return [
            Candle(timestamp=base_time, open=100, high=101, low=99, close=100.5, volume=1000),
            Candle(timestamp=base_time, open=100.5, high=102, low=100, close=101.5, volume=1100),
            Candle(timestamp=base_time, open=101.5, high=103, low=101, close=102.5, volume=1200),
        ]

    def test_pivot_points_normal_operation(self, sample_candles):
        """Test Pivot Points with normal data"""
        result = SupportResistanceIndicators.pivot_points(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.SUPPORT_RESISTANCE
        assert result.indicator_name == "Pivot Points"
        assert isinstance(result.signal, SignalStrength)
        assert 0 <= result.confidence <= 1
        assert isinstance(result.description, str)
        assert "پیوت" in result.description

        # Check additional values
        assert "R1" in result.additional_values
        assert "R2" in result.additional_values
        assert "R3" in result.additional_values
        assert "S1" in result.additional_values
        assert "S2" in result.additional_values
        assert "S3" in result.additional_values

        # Check pivot levels are reasonable
        pivot = result.value
        r1 = result.additional_values["R1"]
        s1 = result.additional_values["S1"]
        assert r1 > pivot > s1

    def test_pivot_points_signal_generation(self, sample_candles):
        """Test Pivot Points signal generation logic"""
        result = SupportResistanceIndicators.pivot_points(sample_candles)

        # Signal should be one of the expected values
        assert result.signal in [
            SignalStrength.VERY_BULLISH, SignalStrength.BULLISH,
            SignalStrength.BULLISH_BROKEN, SignalStrength.NEUTRAL,
            SignalStrength.BEARISH_BROKEN, SignalStrength.BEARISH, SignalStrength.VERY_BEARISH
        ]

        # Confidence should be reasonable
        assert result.confidence >= 0.7

    def test_pivot_points_price_position(self, sample_candles):
        """Test Pivot Points price position calculation"""
        from dataclasses import replace

        # Test with price above R2
        high_price_candle = replace(sample_candles[-1], close=sample_candles[-1].high + 10, high=sample_candles[-1].high + 10)
        test_candles = sample_candles[:-1] + [high_price_candle]

        result = SupportResistanceIndicators.pivot_points(test_candles)
        assert result.signal == SignalStrength.VERY_BULLISH

        # Test with price below S2
        low_price_candle = replace(sample_candles[-1], close=sample_candles[-1].low - 10, low=sample_candles[-1].low - 10)
        test_candles = sample_candles[:-1] + [low_price_candle]

        result = SupportResistanceIndicators.pivot_points(test_candles)
        assert result.signal == SignalStrength.VERY_BEARISH

    def test_fibonacci_retracement_normal(self, sample_candles):
        """Test Fibonacci Retracement with normal data"""
        result = SupportResistanceIndicators.fibonacci_retracement(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.SUPPORT_RESISTANCE
        assert "Fibonacci Retracement" in result.indicator_name
        assert isinstance(result.signal, SignalStrength)
        assert 0 <= result.confidence <= 1

        # Check Fibonacci levels
        assert "0.0" in result.additional_values
        assert "1.0" in result.additional_values
        assert result.additional_values["0.0"] > result.additional_values["1.0"]  # High > Low

    def test_fibonacci_retracement_levels(self, sample_candles):
        """Test Fibonacci Retracement level calculations"""
        result = SupportResistanceIndicators.fibonacci_retracement(sample_candles)

        levels = result.additional_values
        high = levels["0.0"]
        low = levels["1.0"]

        # Check standard Fibonacci levels
        expected_levels = ["0.0", "0.236", "0.382", "0.5", "0.618", "0.786", "1.0"]
        for level in expected_levels:
            assert level in levels
            assert low <= levels[level] <= high

        # Check level ordering
        assert levels["0.0"] >= levels["0.236"] >= levels["0.382"] >= levels["0.5"] >= levels["0.618"] >= levels["0.786"] >= levels["1.0"]

    def test_fibonacci_retracement_signals(self, sample_candles):
        """Test Fibonacci Retracement signal generation"""
        # Test in uptrend
        result = SupportResistanceIndicators.fibonacci_retracement(sample_candles)
        assert result.signal in [
            SignalStrength.BULLISH, SignalStrength.BULLISH_BROKEN,
            SignalStrength.NEUTRAL, SignalStrength.BEARISH_BROKEN,
            SignalStrength.VERY_BULLISH
        ]

    def test_fibonacci_retracement_different_lookbacks(self, sample_candles):
        """Test Fibonacci Retracement with different lookback periods"""
        for lookback in [20, 30, 50]:
            result = SupportResistanceIndicators.fibonacci_retracement(sample_candles, lookback)
            assert isinstance(result, IndicatorResult)
            assert f"({lookback})" in result.indicator_name

    def test_camarilla_pivots_normal(self, sample_candles):
        """Test Camarilla Pivots with normal data"""
        result = SupportResistanceIndicators.camarilla_pivots(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.SUPPORT_RESISTANCE
        assert result.indicator_name == "Camarilla Pivots"
        assert isinstance(result.signal, SignalStrength)

        # Check Camarilla levels
        expected_levels = ["R4", "R3", "R2", "R1", "S1", "S2", "S3", "S4"]
        for level in expected_levels:
            assert level in result.additional_values

        # Check level ordering
        levels = result.additional_values
        assert levels["R4"] >= levels["R3"] >= levels["R2"] >= levels["R1"] >= levels["S1"] >= levels["S2"] >= levels["S3"] >= levels["S4"]

    def test_camarilla_pivots_signals(self, sample_candles):
        """Test Camarilla Pivots signal generation"""
        result = SupportResistanceIndicators.camarilla_pivots(sample_candles)

        assert result.signal in [
            SignalStrength.VERY_BULLISH, SignalStrength.BULLISH, SignalStrength.BULLISH_BROKEN,
            SignalStrength.NEUTRAL,
            SignalStrength.BEARISH_BROKEN, SignalStrength.BEARISH, SignalStrength.VERY_BEARISH
        ]

    def test_support_resistance_levels_normal(self, sample_candles):
        """Test Support/Resistance Levels with normal data"""
        result = SupportResistanceIndicators.support_resistance_levels(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.SUPPORT_RESISTANCE
        assert "Support/Resistance" in result.indicator_name
        assert isinstance(result.signal, SignalStrength)

        # Check resistance and support levels
        assert "resistance" in result.additional_values
        assert "support" in result.additional_values
        assert result.additional_values["resistance"] > result.additional_values["support"]

    def test_support_resistance_levels_calculation(self, sample_candles):
        """Test Support/Resistance Levels calculation logic"""
        result = SupportResistanceIndicators.support_resistance_levels(sample_candles)

        resistance = result.additional_values["resistance"]
        support = result.additional_values["support"]
        current_price = sample_candles[-1].close

        # Price should be between support and resistance
        assert support <= current_price <= resistance

        # Check position percentage in description
        assert "موقعیت:" in result.description
        assert "% بین حمایت و مقاومت" in result.description

    def test_calculate_all_normal(self, sample_candles):
        """Test calculate_all method"""
        results = SupportResistanceIndicators.calculate_all(sample_candles)

        assert isinstance(results, list)
        assert len(results) == 4  # All 4 indicators

        for result in results:
            assert isinstance(result, IndicatorResult)
            assert result.category == IndicatorCategory.SUPPORT_RESISTANCE

    def test_calculate_all_insufficient_data(self, minimal_candles):
        """Test calculate_all with insufficient data"""
        results = SupportResistanceIndicators.calculate_all(minimal_candles)

        # Should only return pivot points and camarilla (need >=10 candles for others)
        assert isinstance(results, list)
        assert len(results) == 2

        indicator_names = [r.indicator_name for r in results]
        assert "Pivot Points" in indicator_names
        assert "Camarilla Pivots" in indicator_names

    # Edge cases and error handling
    def test_all_indicators_with_empty_candles(self):
        """Test all indicators with empty candle list"""
        empty_candles = []

        # Should handle gracefully or raise appropriate errors
        with pytest.raises((IndexError, ZeroDivisionError, ValueError)):
            SupportResistanceIndicators.pivot_points(empty_candles)

        with pytest.raises((IndexError, ZeroDivisionError, ValueError)):
            SupportResistanceIndicators.fibonacci_retracement(empty_candles)

        with pytest.raises((IndexError, ZeroDivisionError, ValueError)):
            SupportResistanceIndicators.camarilla_pivots(empty_candles)

        with pytest.raises((IndexError, ZeroDivisionError, ValueError)):
            SupportResistanceIndicators.support_resistance_levels(empty_candles)

    def test_all_indicators_with_single_candle(self):
        """Test all indicators with single candle"""
        single_candle = [Candle(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            open=100, high=101, low=99, close=100.5, volume=1000
        )]

        # Pivot points should work with minimal data
        result_pp = SupportResistanceIndicators.pivot_points(single_candle)
        assert isinstance(result_pp, IndicatorResult)

        # Fibonacci retracement should handle insufficient data gracefully
        result_fr = SupportResistanceIndicators.fibonacci_retracement(single_candle)
        assert isinstance(result_fr, IndicatorResult)

        result_cp = SupportResistanceIndicators.camarilla_pivots(single_candle)
        assert isinstance(result_cp, IndicatorResult)

        # Support resistance levels needs more data for local extrema but handles gracefully
        result_sr = SupportResistanceIndicators.support_resistance_levels(single_candle)
        assert isinstance(result_sr, IndicatorResult)

    def test_fibonacci_with_insufficient_lookback(self, minimal_candles):
        """Test Fibonacci with insufficient lookback"""
        # Should handle gracefully
        result = SupportResistanceIndicators.fibonacci_retracement(minimal_candles, lookback=50)
        assert isinstance(result, IndicatorResult)

    def test_support_resistance_with_no_local_extrema(self, sample_candles):
        """Test Support/Resistance with flat price action"""
        # Create flat candles
        flat_candles = []
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(60):
            candle = Candle(
                timestamp=base_time,
                open=100, high=100.5, low=99.5, close=100,
                volume=1000
            )
            flat_candles.append(candle)

        result = SupportResistanceIndicators.support_resistance_levels(flat_candles)
        assert isinstance(result, IndicatorResult)

    def test_pivot_points_with_extreme_price(self, sample_candles):
        """Test Pivot Points with extreme price movements"""
        from dataclasses import replace

        # Test with very high price
        extreme_high = replace(sample_candles[-1], close=sample_candles[-1].high * 2, high=sample_candles[-1].high * 2)
        test_candles = sample_candles[:-1] + [extreme_high]

        result = SupportResistanceIndicators.pivot_points(test_candles)
        assert result.signal == SignalStrength.VERY_BULLISH

        # Test with very low price
        extreme_low = replace(sample_candles[-1], close=sample_candles[-1].low * 0.5, low=sample_candles[-1].low * 0.5)
        test_candles = sample_candles[:-1] + [extreme_low]

        result = SupportResistanceIndicators.pivot_points(test_candles)
        assert result.signal == SignalStrength.VERY_BEARISH

    def test_camarilla_levels_calculation(self, sample_candles):
        """Test Camarilla levels calculation accuracy"""
        result = SupportResistanceIndicators.camarilla_pivots(sample_candles)

        levels = result.additional_values
        last_candle = sample_candles[-2] if len(sample_candles) > 1 else sample_candles[-1]

        close = last_candle.close
        high = last_candle.high
        low = last_candle.low
        range_hl = high - low

        # Verify level calculations
        expected_r1 = close + (range_hl * 1.1 / 12)
        expected_s1 = close - (range_hl * 1.1 / 12)

        assert abs(levels["R1"] - expected_r1) < 0.01
        assert abs(levels["S1"] - expected_s1) < 0.01

    def test_fibonacci_level_accuracy(self, sample_candles):
        """Test Fibonacci level calculation accuracy"""
        result = SupportResistanceIndicators.fibonacci_retracement(sample_candles)

        levels = result.additional_values
        high = levels["0.0"]
        low = levels["1.0"]
        diff = high - low

        # Check specific Fibonacci ratios
        assert abs(levels["0.5"] - (high - 0.5 * diff)) < 0.01
        assert abs(levels["0.618"] - (high - 0.618 * diff)) < 0.01
        assert abs(levels["0.786"] - (high - 0.786 * diff)) < 0.01

    def test_signal_confidence_levels(self, sample_candles):
        """Test that confidence levels are appropriate for signals"""
        indicators = [
            SupportResistanceIndicators.pivot_points,
            SupportResistanceIndicators.fibonacci_retracement,
            SupportResistanceIndicators.camarilla_pivots,
            SupportResistanceIndicators.support_resistance_levels,
        ]

        for indicator_func in indicators:
            result = indicator_func(sample_candles)
            assert 0 <= result.confidence <= 1

            # Strong signals should have higher confidence
            if result.signal in [SignalStrength.VERY_BULLISH, SignalStrength.VERY_BEARISH]:
                assert result.confidence >= 0.7

    def test_indicator_names(self, sample_candles):
        """Test that indicator names are properly formatted"""
        result_pp = SupportResistanceIndicators.pivot_points(sample_candles)
        assert result_pp.indicator_name == "Pivot Points"

        result_fr = SupportResistanceIndicators.fibonacci_retracement(sample_candles, 50)
        assert result_fr.indicator_name == "Fibonacci Retracement(50)"

        result_cp = SupportResistanceIndicators.camarilla_pivots(sample_candles)
        assert result_cp.indicator_name == "Camarilla Pivots"

        result_sr = SupportResistanceIndicators.support_resistance_levels(sample_candles, 50)
        assert "Support/Resistance(50)" in result_sr.indicator_name

    def test_description_content(self, sample_candles):
        """Test that descriptions contain relevant information"""
        result_pp = SupportResistanceIndicators.pivot_points(sample_candles)
        assert "پیوت" in result_pp.description

        result_fr = SupportResistanceIndicators.fibonacci_retracement(sample_candles)
        assert "فیبوناچی" in result_fr.description

        result_sr = SupportResistanceIndicators.support_resistance_levels(sample_candles)
        assert "حمایت" in result_sr.description and "مقاومت" in result_sr.description

    def test_additional_values_completeness(self, sample_candles):
        """Test that additional_values contain all expected keys"""
        result_pp = SupportResistanceIndicators.pivot_points(sample_candles)
        required_pp_keys = ["R1", "R2", "R3", "S1", "S2", "S3"]
        for key in required_pp_keys:
            assert key in result_pp.additional_values

        result_fr = SupportResistanceIndicators.fibonacci_retracement(sample_candles)
        required_fr_keys = ["0.0", "0.236", "0.382", "0.5", "0.618", "0.786", "1.0"]
        for key in required_fr_keys:
            assert key in result_fr.additional_values

        result_cp = SupportResistanceIndicators.camarilla_pivots(sample_candles)
        required_cp_keys = ["R4", "R3", "R2", "R1", "S1", "S2", "S3", "S4"]
        for key in required_cp_keys:
            assert key in result_cp.additional_values

        result_sr = SupportResistanceIndicators.support_resistance_levels(sample_candles)
        required_sr_keys = ["resistance", "support"]
        for key in required_sr_keys:
            assert key in result_sr.additional_values
