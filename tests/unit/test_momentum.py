"""
Unit tests for Momentum indicators (src/core/indicators/momentum.py)

Tests cover:
- All 7 momentum indicators (RSI, Stochastic, CCI, ROC, Williams %R, MFI, Ultimate Oscillator)
- Signal generation and confidence levels
- Momentum calculations and oscillator logic
- Edge cases and error handling
- Backward compatibility methods
- Helper functions
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from src.core.indicators.momentum import (
    MomentumIndicators,
    _ema,
    _rsi_from_changes,
    tsi,
    schaff_trend_cycle,
    connors_rsi
)
from src.core.domain.entities import (
    Candle,
    IndicatorResult,
    CoreSignalStrength as SignalStrength,
    IndicatorCategory
)


class TestMomentumIndicators:
    """Test MomentumIndicators class methods"""

    @pytest.fixture
    def sample_candles(self):
        """Create sample candles for testing"""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        # Create trending up pattern with momentum changes
        prices = []
        for i in range(60):
            # Upward trend with momentum shifts
            base_price = 100 + i * 0.5
            # Add momentum changes
            if i < 20:
                momentum = 1  # Slow momentum
            elif i < 40:
                momentum = 3  # Strong momentum
            else:
                momentum = 0.5  # Weakening momentum

            price = base_price + momentum * np.sin(2 * np.pi * i / 10)
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

    def test_ema_helper_function(self):
        """Test _ema helper function"""
        values = np.array([1, 2, 3, 4, 5])
        result = _ema(values, 3)

        assert len(result) == len(values)
        assert result[0] == values[0]  # First value unchanged
        assert result[-1] >= 1.0  # EMA should be at least the first value
        # EMA is a weighted average, so it should be between min and max
        assert np.min(values) <= result[-1] <= np.max(values)

        # Test with different periods
        result2 = _ema(values, 2)
        assert len(result2) == len(values)
        # Different periods should produce different results
        assert not np.allclose(result, result2)

    def test_rsi_from_changes_helper(self):
        """Test _rsi_from_changes helper function"""
        changes = np.array([1, -0.5, 2, -1, 0.5])
        result = _rsi_from_changes(changes, 3)

        assert len(result) == len(changes)
        assert 0 <= result[-1] <= 100  # RSI should be 0-100

    def test_tsi_function(self):
        """Test TSI function"""
        # TSI needs at least r + s + 2 = 40 prices
        prices = np.array([100 + i * 0.5 for i in range(50)])
        result = tsi(prices)

        assert isinstance(result, dict)
        assert "values" in result
        assert "signal" in result
        assert "confidence" in result
        assert len(result["values"]) == len(prices)
        assert result["signal"] in ['BUY', 'SELL', None]
        assert 0 <= result["confidence"] <= 1

    def test_schaff_trend_cycle_function(self):
        """Test Schaff Trend Cycle function"""
        # STC needs at least slow + cycle = 36 prices
        prices = np.array([100 + i * 0.3 for i in range(50)])
        result = schaff_trend_cycle(prices)

        assert isinstance(result, dict)
        assert "values" in result
        assert "signal" in result
        assert "confidence" in result
        assert len(result["values"]) == len(prices)
        assert result["signal"] in ['BUY', 'SELL', None]
        assert 0 <= result["confidence"] <= 1

    def test_connors_rsi_function(self):
        """Test Connors RSI function"""
        prices = np.array([100, 101, 102, 103, 102, 101, 102, 103, 104, 105, 106, 107, 108])
        result = connors_rsi(prices)

        assert isinstance(result, dict)
        assert "values" in result
        assert "signal" in result
        assert "confidence" in result
        assert len(result["values"]) == len(prices)
        assert result["signal"] in ['BUY', 'SELL', None]
        assert 0 <= result["confidence"] <= 1

    def test_rsi_normal_operation(self, sample_candles):
        """Test RSI with normal data"""
        result = MomentumIndicators.rsi(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.MOMENTUM
        assert "RSI" in result.indicator_name
        assert isinstance(result.signal, SignalStrength)
        assert 0 <= result.confidence <= 1
        assert isinstance(result.description, str)
        assert "RSI در سطح" in result.description

    def test_rsi_calculation_accuracy(self, sample_candles):
        """Test RSI calculation accuracy"""
        result = MomentumIndicators.rsi(sample_candles, period=14)

        closes = pd.Series([c.close for c in sample_candles])
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        expected_rsi = 100 - (100 / (1 + rs))
        expected_rsi = expected_rsi.iloc[-1]

        assert abs(result.value - expected_rsi) < 0.01

    def test_rsi_signal_generation(self, sample_candles):
        """Test RSI signal generation"""
        result = MomentumIndicators.rsi(sample_candles)

        rsi_value = result.value

        # Signal based on RSI levels
        if rsi_value > 80:
            assert result.signal == SignalStrength.VERY_BEARISH
        elif rsi_value > 70:
            assert result.signal == SignalStrength.BEARISH
        elif rsi_value < 20:
            assert result.signal == SignalStrength.VERY_BULLISH
        elif rsi_value < 30:
            assert result.signal == SignalStrength.BULLISH

    def test_stochastic_normal_operation(self, sample_candles):
        """Test Stochastic Oscillator with normal data"""
        result = MomentumIndicators.stochastic(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.MOMENTUM
        assert "Stochastic" in result.indicator_name
        assert isinstance(result.signal, SignalStrength)

        # Check additional values
        assert "D" in result.additional_values

    def test_stochastic_calculation_accuracy(self, sample_candles):
        """Test Stochastic calculation accuracy"""
        result = MomentumIndicators.stochastic(sample_candles, k_period=14, d_period=3)

        df = pd.DataFrame({
            'high': [c.high for c in sample_candles],
            'low': [c.low for c in sample_candles],
            'close': [c.close for c in sample_candles]
        })

        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        expected_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        expected_d = expected_k.rolling(window=3).mean()

        assert abs(result.value - expected_k.iloc[-1]) < 0.01
        assert abs(result.additional_values["D"] - expected_d.iloc[-1]) < 0.01

    def test_stochastic_signal_generation(self, sample_candles):
        """Test Stochastic signal generation"""
        result = MomentumIndicators.stochastic(sample_candles)

        k_value = result.value
        d_value = result.additional_values["D"]

        # Signal based on levels and crossover
        if k_value > 80 and d_value > 80:
            assert result.signal == SignalStrength.VERY_BEARISH
        elif k_value < 20 and d_value < 20:
            assert result.signal == SignalStrength.VERY_BULLISH

    def test_cci_normal_operation(self, sample_candles):
        """Test CCI with normal data"""
        result = MomentumIndicators.cci(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.MOMENTUM
        assert "CCI" in result.indicator_name
        assert isinstance(result.signal, SignalStrength)

    def test_cci_calculation_accuracy(self, sample_candles):
        """Test CCI calculation accuracy"""
        result = MomentumIndicators.cci(sample_candles, period=20)

        typical_prices = pd.Series([c.typical_price for c in sample_candles])
        sma = typical_prices.rolling(window=20).mean()
        mad = typical_prices.rolling(window=20).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        expected_cci = (typical_prices - sma) / (0.015 * mad)
        expected_cci = expected_cci.iloc[-1]

        assert abs(result.value - expected_cci) < 0.01

    def test_cci_signal_generation(self, sample_candles):
        """Test CCI signal generation"""
        result = MomentumIndicators.cci(sample_candles)

        cci_value = result.value

        if cci_value > 200:
            assert result.signal == SignalStrength.VERY_BULLISH
        elif cci_value < -200:
            assert result.signal == SignalStrength.VERY_BEARISH

    def test_roc_normal_operation(self, sample_candles):
        """Test ROC with normal data"""
        result = MomentumIndicators.roc(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.MOMENTUM
        assert "ROC" in result.indicator_name
        assert isinstance(result.signal, SignalStrength)

    def test_roc_calculation_accuracy(self, sample_candles):
        """Test ROC calculation accuracy"""
        result = MomentumIndicators.roc(sample_candles, period=12)

        closes = pd.Series([c.close for c in sample_candles])
        expected_roc = ((closes - closes.shift(12)) / closes.shift(12)) * 100
        expected_roc = expected_roc.iloc[-1]

        assert abs(result.value - expected_roc) < 0.01

    def test_williams_r_normal_operation(self, sample_candles):
        """Test Williams %R with normal data"""
        result = MomentumIndicators.williams_r(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.MOMENTUM
        assert "Williams %R" in result.indicator_name
        assert isinstance(result.signal, SignalStrength)

    def test_williams_r_calculation_accuracy(self, sample_candles):
        """Test Williams %R calculation accuracy"""
        result = MomentumIndicators.williams_r(sample_candles, period=14)

        df = pd.DataFrame({
            'high': [c.high for c in sample_candles],
            'low': [c.low for c in sample_candles],
            'close': [c.close for c in sample_candles]
        })

        high_max = df['high'].rolling(window=14).max()
        low_min = df['low'].rolling(window=14).min()
        expected_wr = -100 * ((high_max - df['close']) / (high_max - low_min))
        expected_wr = expected_wr.iloc[-1]

        assert abs(result.value - expected_wr) < 0.01

    def test_williams_r_signal_generation(self, sample_candles):
        """Test Williams %R signal generation"""
        result = MomentumIndicators.williams_r(sample_candles)

        wr_value = result.value

        # Inverted RSI logic
        if wr_value > -20:
            assert result.signal == SignalStrength.VERY_BEARISH
        elif wr_value < -80:
            assert result.signal == SignalStrength.VERY_BULLISH

    def test_mfi_normal_operation(self, sample_candles):
        """Test MFI with normal data"""
        result = MomentumIndicators.mfi(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.MOMENTUM
        assert "MFI" in result.indicator_name
        assert isinstance(result.signal, SignalStrength)

    def test_mfi_calculation_accuracy(self, sample_candles):
        """Test MFI calculation accuracy"""
        result = MomentumIndicators.mfi(sample_candles, period=14)

        df = pd.DataFrame({
            'high': [c.high for c in sample_candles],
            'low': [c.low for c in sample_candles],
            'close': [c.close for c in sample_candles],
            'volume': [c.volume for c in sample_candles],
            'typical': [c.typical_price for c in sample_candles]
        })

        money_flow = df['typical'] * df['volume']
        positive_flow = money_flow.where(df['typical'] > df['typical'].shift(1), 0)
        negative_flow = money_flow.where(df['typical'] < df['typical'].shift(1), 0)
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        expected_mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        expected_mfi = expected_mfi.iloc[-1]

        assert abs(result.value - expected_mfi) < 0.01

    def test_ultimate_oscillator_normal_operation(self, sample_candles):
        """Test Ultimate Oscillator with normal data"""
        result = MomentumIndicators.ultimate_oscillator(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.MOMENTUM
        assert "Ultimate Oscillator" in result.indicator_name
        assert isinstance(result.signal, SignalStrength)

    def test_calculate_all_normal(self, sample_candles):
        """Test calculate_all method"""
        results = MomentumIndicators.calculate_all(sample_candles)

        assert isinstance(results, list)
        assert len(results) >= 5  # Should return multiple indicators

        for result in results:
            assert isinstance(result, IndicatorResult)
            assert result.category == IndicatorCategory.MOMENTUM

    def test_calculate_all_insufficient_data(self, minimal_candles):
        """Test calculate_all with insufficient data"""
        results = MomentumIndicators.calculate_all(minimal_candles)

        # Should return basic indicators that work with minimal data
        assert isinstance(results, list)
        # May return fewer indicators with minimal data

    # Edge cases and error handling
    def test_all_indicators_with_empty_candles(self):
        """Test all indicators with empty candle list"""
        empty_candles = []

        # Should handle gracefully or raise appropriate errors
        with pytest.raises((IndexError, ZeroDivisionError, ValueError)):
            MomentumIndicators.rsi(empty_candles)

        with pytest.raises((IndexError, ZeroDivisionError, ValueError)):
            MomentumIndicators.stochastic(empty_candles)

    def test_all_indicators_with_single_candle(self):
        """Test all indicators with single candle"""
        single_candle = [Candle(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            open=100, high=101, low=99, close=100.5, volume=1000
        )]

        # RSI, Stochastic, CCI, ROC, Williams %R, MFI, Ultimate Oscillator باید ValueError بدهند
        with pytest.raises(ValueError):
            MomentumIndicators.rsi(single_candle)
        with pytest.raises(ValueError):
            MomentumIndicators.stochastic(single_candle)
        with pytest.raises(ValueError):
            MomentumIndicators.cci(single_candle)
        with pytest.raises(ValueError):
            MomentumIndicators.roc(single_candle)
        with pytest.raises(ValueError):
            MomentumIndicators.williams_r(single_candle)
        with pytest.raises(ValueError):
            MomentumIndicators.mfi(single_candle)
        with pytest.raises(ValueError):
            MomentumIndicators.ultimate_oscillator(single_candle)

    def test_indicators_with_zero_volume(self, sample_candles):
        """Test indicators with zero volume candles"""
        # Create candle with zero volume
        zero_volume_candle = sample_candles[-1]._replace(volume=0)
        test_candles = sample_candles[:-1] + [zero_volume_candle]

        # Should handle zero volume gracefully
        result_mfi = MomentumIndicators.mfi(test_candles)
        assert isinstance(result_mfi, IndicatorResult)

    def test_rsi_with_extreme_values(self, sample_candles):
        """Test RSI with extreme price movements"""
        # Create RSI extreme conditions
        extreme_up_candles = sample_candles[:14] + [sample_candles[-1]._replace(close=sample_candles[-1].close * 2)]
        result = MomentumIndicators.rsi(extreme_up_candles)
        assert isinstance(result, IndicatorResult)

        extreme_down_candles = sample_candles[:14] + [sample_candles[-1]._replace(close=sample_candles[-1].close * 0.5)]
        result = MomentumIndicators.rsi(extreme_down_candles)
        assert isinstance(result, IndicatorResult)

    def test_stochastic_with_flat_range(self, sample_candles):
        """Test Stochastic with flat high-low range"""
        # Create candles with same high/low
        flat_candles = []
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(20):
            candle = Candle(
                timestamp=base_time,
                open=100, high=100, low=100, close=100,
                volume=1000
            )
            flat_candles.append(candle)

        result = MomentumIndicators.stochastic(flat_candles)
        assert isinstance(result, IndicatorResult)
        # Should handle division by zero

    def test_cci_with_zero_mad(self, sample_candles):
        """Test CCI with zero MAD (constant prices)"""
        # Create constant price candles
        constant_candles = []
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(25):
            candle = Candle(
                timestamp=base_time,
                open=100, high=100, low=100, close=100,
                volume=1000
            )
            constant_candles.append(candle)

        result = MomentumIndicators.cci(constant_candles)
        assert isinstance(result, IndicatorResult)

    def test_roc_with_insufficient_history(self, minimal_candles):
        """Test ROC with insufficient lookback"""
        with pytest.raises(ValueError):
            MomentumIndicators.roc(minimal_candles, period=10)

    def test_signal_confidence_levels(self, sample_candles):
        """Test that confidence levels are appropriate for signals"""
        indicators = [
            lambda: MomentumIndicators.rsi(sample_candles),
            lambda: MomentumIndicators.stochastic(sample_candles),
            lambda: MomentumIndicators.cci(sample_candles),
            lambda: MomentumIndicators.roc(sample_candles),
            lambda: MomentumIndicators.williams_r(sample_candles),
            lambda: MomentumIndicators.mfi(sample_candles),
        ]

        for indicator_func in indicators:
            result = indicator_func()
            assert 0 <= result.confidence <= 1

            # Strong signals should have reasonable confidence
            if result.signal in [SignalStrength.VERY_BULLISH, SignalStrength.VERY_BEARISH]:
                assert result.confidence >= 0.6

    def test_indicator_names_formatting(self, sample_candles):
        """Test that indicator names are properly formatted"""
        assert MomentumIndicators.rsi(sample_candles, 21).indicator_name == "RSI(21)"
        assert MomentumIndicators.stochastic(sample_candles, 21, 5).indicator_name == "Stochastic(21,5)"
        assert MomentumIndicators.cci(sample_candles, 25).indicator_name == "CCI(25)"
        assert MomentumIndicators.roc(sample_candles, 15).indicator_name == "ROC(15)"
        assert MomentumIndicators.williams_r(sample_candles, 21).indicator_name == "Williams %R(21)"
        assert MomentumIndicators.mfi(sample_candles, 21).indicator_name == "MFI(21)"
        assert "Ultimate Oscillator" in MomentumIndicators.ultimate_oscillator(sample_candles, 10, 15, 30).indicator_name

    def test_description_content(self, sample_candles):
        """Test that descriptions contain relevant information"""
        result_rsi = MomentumIndicators.rsi(sample_candles)
        assert "RSI در سطح" in result_rsi.description and "اشباع" in result_rsi.description

        result_stoch = MomentumIndicators.stochastic(sample_candles)
        assert "Stochastic:" in result_stoch.description

        result_cci = MomentumIndicators.cci(sample_candles)
        assert "CCI در سطح" in result_cci.description

        result_roc = MomentumIndicators.roc(sample_candles)
        assert "نرخ تغییر:" in result_roc.description and "%" in result_roc.description

    def test_additional_values_completeness(self, sample_candles):
        """Test that additional_values contain all expected keys when applicable"""
        result_stoch = MomentumIndicators.stochastic(sample_candles)
        assert "D" in result_stoch.additional_values

        # Other indicators may not have additional values
        result_rsi = MomentumIndicators.rsi(sample_candles)
        # RSI doesn't have additional values

    def test_momentum_indicators_range_bounds(self, sample_candles):
        """Test that momentum indicators stay within expected ranges"""
        result_rsi = MomentumIndicators.rsi(sample_candles)
        assert 0 <= result_rsi.value <= 100

        result_stoch = MomentumIndicators.stochastic(sample_candles)
        assert 0 <= result_stoch.value <= 100
        assert 0 <= result_stoch.additional_values["D"] <= 100

        result_wr = MomentumIndicators.williams_r(sample_candles)
        assert -100 <= result_wr.value <= 0

        result_mfi = MomentumIndicators.mfi(sample_candles)
        assert 0 <= result_mfi.value <= 100

    def test_helper_functions_edge_cases(self):
        """Test helper functions with edge cases"""
        # Empty arrays
        with pytest.raises((IndexError, ZeroDivisionError)):
            _ema(np.array([]), 3)

        with pytest.raises((IndexError, ZeroDivisionError)):
            _rsi_from_changes(np.array([]), 3)

        # Single values
        single_value = np.array([100.0])
        result_ema = _ema(single_value, 3)
        assert len(result_ema) == 1
        assert result_ema[0] == 100.0

        result_rsi = _rsi_from_changes(single_value, 3)
        assert len(result_rsi) == 1

    def test_tsi_edge_cases(self):
        """Test TSI with edge cases"""
        # Insufficient data
        short_prices = np.array([100, 101])
        with pytest.raises(ValueError):
            tsi(short_prices)

        # Constant prices - need enough data
        constant_prices = np.array([100] * 50)
        result = tsi(constant_prices)
        assert isinstance(result, dict)

    def test_schaff_edge_cases(self):
        """Test Schaff Trend Cycle with edge cases"""
        # Insufficient data
        short_prices = np.array([100, 101, 102])
        with pytest.raises(ValueError):
            schaff_trend_cycle(short_prices)

    def test_connors_rsi_edge_cases(self):
        """Test Connors RSI with edge cases"""
        # Insufficient data
        short_prices = np.array([100, 101])
        with pytest.raises(ValueError):
            connors_rsi(short_prices)

        # Constant prices
        constant_prices = np.array([100] * 15)
        result = connors_rsi(constant_prices)
        # سیگنال معتبر (مثلاً 'SELL' یا 'BUY' یا 'HOLD') قابل قبول است
        assert result["signal"] in ["SELL", "BUY", "HOLD", None]
