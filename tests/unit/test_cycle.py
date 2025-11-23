"""
Unit tests for Cycle indicators (src/core/indicators/cycle.py)

Tests cover:
- All 7 cycle indicators (DPO, Ehlers Cycle, Dominant Cycle, STC, Phase Accumulation, Hilbert Transform, Market Cycle Model)
- CycleResult dataclass validation
- Signal generation and confidence levels
- Phase calculations (0-360 degrees)
- Edge cases and error handling
- Backward compatibility methods
- Helper functions
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from src.core.indicators.cycle import (
    CycleIndicators,
    CycleResult,
    convert_cycle_to_indicator_result
)
from src.core.domain.entities import (
    Candle,
    IndicatorResult,
    CoreSignalStrength as SignalStrength,
    IndicatorCategory
)


class TestCycleResult:
    """Test CycleResult dataclass"""

    def test_cycle_result_creation(self):
        """Test creating a valid CycleResult"""
        result = CycleResult(
            indicator_name="Test Cycle",
            value=10.5,
            normalized=0.8,
            phase=90.0,
            cycle_period=20,
            signal=SignalStrength.BULLISH,
            confidence=0.85,
            description="Test cycle result"
        )

        assert result.value == 10.5
        assert result.normalized == 0.8
        assert result.phase == 90.0
        assert result.cycle_period == 20
        assert result.signal == SignalStrength.BULLISH
        assert result.confidence == 0.85
        assert result.description == "Test cycle result"

    def test_cycle_result_immutability(self):
        """Test that CycleResult is immutable"""
        result = CycleResult(
            indicator_name="Test Cycle Immutable",
            value=10.0,
            normalized=0.5,
            phase=180.0,
            cycle_period=25,
            signal=SignalStrength.NEUTRAL,
            confidence=0.7,
            description="Immutable test"
        )

        with pytest.raises(AttributeError):
            result.value = 15.0

        with pytest.raises(AttributeError):
            result.signal = SignalStrength.BEARISH


class TestCycleIndicators:
    """Test CycleIndicators class methods"""

    @pytest.fixture
    def sample_candles(self):
        """Create sample candles for testing"""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        # Create a sine wave pattern with 50 candles
        prices = []
        for i in range(50):
            # Sine wave with period of ~20 candles
            price = 100 + 5 * np.sin(2 * np.pi * i / 20)
            prices.append(price)

        candles = []
        for i, price in enumerate(prices):
            candle = Candle(
                timestamp=base_time,
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
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

    def test_dpo_normal_operation(self, sample_candles):
        """Test DPO with normal data"""
        result = CycleIndicators.dpo(sample_candles)

        assert isinstance(result, CycleResult)
        assert isinstance(result.value, (int, float))
        assert -1 <= result.normalized <= 1
        assert 0 <= result.phase <= 360
        assert result.cycle_period > 0
        assert isinstance(result.signal, SignalStrength)
        assert 0 <= result.confidence <= 1
        assert isinstance(result.description, str)
        assert "DPO=" in result.description

    def test_dpo_insufficient_data(self, minimal_candles):
        """Test DPO with insufficient data"""
        with pytest.raises(ValueError):
            CycleIndicators.dpo(minimal_candles)

    def test_dpo_signal_generation(self, sample_candles):
        """Test DPO signal generation logic"""
        result = CycleIndicators.dpo(sample_candles)

        # Signal should be one of the expected values
        assert result.signal in [
            SignalStrength.VERY_BULLISH, SignalStrength.BULLISH,
            SignalStrength.NEUTRAL,
            SignalStrength.BEARISH, SignalStrength.VERY_BEARISH
        ]

        # Confidence should be reasonable
        assert result.confidence >= 0.6

    @pytest.mark.parametrize("period", [10, 20, 30])
    def test_dpo_different_periods(self, sample_candles, period):
        """Test DPO with different periods"""
        result = CycleIndicators.dpo(sample_candles, period)

        assert isinstance(result, CycleResult)
        assert result.cycle_period > 0

    def test_ehlers_cycle_period_normal(self, sample_candles):
        """Test Ehlers cycle period detection"""
        result = CycleIndicators.ehlers_cycle_period(sample_candles)

        assert isinstance(result, CycleResult)
        assert 10 <= result.cycle_period <= 50  # Within expected bounds
        assert 0 <= result.phase <= 360
        assert -1 <= result.normalized <= 1

    def test_ehlers_cycle_period_minimal_data(self, minimal_candles):
        """Test Ehlers with minimal data"""
        with pytest.raises(ValueError):
            CycleIndicators.ehlers_cycle_period(minimal_candles)

    def test_dominant_cycle_normal(self, sample_candles):
        """Test dominant cycle detection"""
        result = CycleIndicators.dominant_cycle(sample_candles)

        assert isinstance(result, CycleResult)
        assert 8 <= result.cycle_period <= 50  # Within bounds
        assert 0 <= result.phase <= 360
        assert -1 <= abs(result.normalized) <= 1

    def test_dominant_cycle_minimal_data(self, minimal_candles):
        """Test dominant cycle with minimal data"""
        with pytest.raises(ValueError):
            CycleIndicators.dominant_cycle(minimal_candles)

    def test_schaff_trend_cycle_normal(self, sample_candles):
        """Test Schaff Trend Cycle"""
        result = CycleIndicators.schaff_trend_cycle(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.CYCLE
        assert "STC(" in result.indicator_name
        assert 0 <= result.value <= 100  # STC is 0-100
        assert isinstance(result.signal, SignalStrength)

    def test_schaff_trend_cycle_signals(self, sample_candles):
        """Test STC signal generation"""
        result = CycleIndicators.schaff_trend_cycle(sample_candles)

        # Check signal based on STC value
        if result.value > 85:
            assert result.signal == SignalStrength.VERY_BEARISH
        elif result.value > 75:
            assert result.signal == SignalStrength.BEARISH
        elif result.value < 15:
            assert result.signal == SignalStrength.VERY_BULLISH
        elif result.value < 25:
            assert result.signal == SignalStrength.BULLISH
        else:
            assert result.signal == SignalStrength.NEUTRAL

    def test_phase_accumulation_normal(self, sample_candles):
        """Test phase accumulation"""
        result = CycleIndicators.phase_accumulation(sample_candles)

        assert isinstance(result, CycleResult)
        assert 0 <= result.phase <= 360
        assert -1 <= result.normalized <= 1
        assert result.cycle_period > 0

    def test_hilbert_transform_phase_normal(self, sample_candles):
        """Test Hilbert transform phase detection"""
        result = CycleIndicators.hilbert_transform_phase(sample_candles)

        assert isinstance(result, CycleResult)
        assert 0 <= result.phase <= 360
        assert -1 <= result.normalized <= 1
        assert 6 <= result.cycle_period <= 50

    def test_market_cycle_model_normal(self, sample_candles):
        """Test market cycle model"""
        result = CycleIndicators.market_cycle_model(sample_candles)

        assert isinstance(result, CycleResult)
        assert 0 <= result.phase <= 360
        assert -1 <= result.normalized <= 1
        assert result.cycle_period == 40  # Fixed for this model

    def test_sine_wave_normal(self, sample_candles):
        """Test sine wave indicator"""
        result = CycleIndicators.sine_wave(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.CYCLE
        assert "Sine Wave(" in result.indicator_name
        assert -1 <= result.value <= 1
        assert isinstance(result.signal, SignalStrength)

    def test_calculate_all_normal(self, sample_candles):
        """Test calculate_all method"""
        results = CycleIndicators.calculate_all(sample_candles)

        assert isinstance(results, list)
        assert len(results) == 8  # 7 cycle indicators + sine wave + STC

        for result in results:
            assert isinstance(result, IndicatorResult)
            assert result.category == IndicatorCategory.CYCLE

    def test_detrended_price_oscillator_backward_compat(self, sample_candles):
        """Test backward compatibility method"""
        result = CycleIndicators.detrended_price_oscillator(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.CYCLE
        assert "DPO(" in result.indicator_name

    def test_convert_cycle_to_indicator_result(self, sample_candles):
        """Test conversion function"""
        cycle_result = CycleIndicators.dpo(sample_candles)
        indicator_result = convert_cycle_to_indicator_result(cycle_result, "Test DPO")

        assert isinstance(indicator_result, IndicatorResult)
        assert indicator_result.indicator_name == "Test DPO"
        assert indicator_result.category == IndicatorCategory.CYCLE
        assert indicator_result.signal == cycle_result.signal
        assert indicator_result.value == cycle_result.value
        assert indicator_result.confidence == cycle_result.confidence
        assert indicator_result.description == cycle_result.description

    def test_estimate_cycle_period_normal(self):
        """Test cycle period estimation helper"""
        # Create oscillator data with known peaks
        oscillator = np.sin(np.linspace(0, 4*np.pi, 100))  # 25 samples per cycle

        period = CycleIndicators._estimate_cycle_period(oscillator)
        assert isinstance(period, int)
        assert period > 0

    def test_estimate_cycle_period_insufficient_data(self):
        """Test cycle period estimation with insufficient data"""
        oscillator = np.array([1, 2, 3])

        period = CycleIndicators._estimate_cycle_period(oscillator)
        assert period == 20  # Default fallback

    def test_calculate_phase_from_oscillator_normal(self):
        """Test phase calculation helper"""
        oscillator = np.sin(np.linspace(0, 2*np.pi, 50))  # Full cycle

        phase = CycleIndicators._calculate_phase_from_oscillator(oscillator)
        assert isinstance(phase, float)
        assert 0 <= phase <= 360

    def test_calculate_phase_from_oscillator_flat(self):
        """Test phase calculation with flat oscillator"""
        oscillator = np.full(20, 5.0)  # Flat line

        phase = CycleIndicators._calculate_phase_from_oscillator(oscillator)
        assert phase == 0.0

    def test_calculate_phase_from_oscillator_insufficient_data(self):
        """Test phase calculation with insufficient data"""
        oscillator = np.array([1, 2])

        phase = CycleIndicators._calculate_phase_from_oscillator(oscillator)
        assert phase == 0.0

    # Edge cases and error handling
    def test_all_indicators_with_empty_candles(self):
        """Test all indicators with empty candle list"""
        empty_candles = []

        with pytest.raises(ValueError):
            CycleIndicators.dpo(empty_candles)
        with pytest.raises(ValueError):
            CycleIndicators.ehlers_cycle_period(empty_candles)
        with pytest.raises(ValueError):
            CycleIndicators.dominant_cycle(empty_candles)
        with pytest.raises((IndexError, ZeroDivisionError, ValueError)):
            CycleIndicators.schaff_trend_cycle(empty_candles)
        with pytest.raises((IndexError, ZeroDivisionError, ValueError)):
            CycleIndicators.phase_accumulation(empty_candles)
        with pytest.raises((IndexError, ZeroDivisionError, ValueError)):
            CycleIndicators.hilbert_transform_phase(empty_candles)
        with pytest.raises((IndexError, ZeroDivisionError, ValueError)):
            CycleIndicators.market_cycle_model(empty_candles)
        with pytest.raises((IndexError, ZeroDivisionError, ValueError)):
            CycleIndicators.sine_wave(empty_candles)

    def test_all_indicators_with_single_candle(self):
        """Test all indicators with single candle"""
        single_candle = [Candle(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            open=100, high=101, low=99, close=100.5, volume=1000
        )]

        with pytest.raises(ValueError):
            CycleIndicators.dpo(single_candle)
        with pytest.raises(ValueError):
            CycleIndicators.ehlers_cycle_period(single_candle)
        with pytest.raises(ValueError):
            CycleIndicators.dominant_cycle(single_candle)
        with pytest.raises((IndexError, ZeroDivisionError, ValueError)):
            CycleIndicators.schaff_trend_cycle(single_candle)

    @pytest.mark.parametrize("invalid_period", [0, -1, -10])
    def test_dpo_invalid_periods(self, sample_candles, invalid_period):
        """Test DPO with invalid periods"""
        # Should raise ValueError for invalid periods
        with pytest.raises(ValueError):
            CycleIndicators.dpo(sample_candles, invalid_period)

    def test_phase_wrapping(self, sample_candles):
        """Test that phases are properly wrapped to 0-360 range"""
        indicators = [
            CycleIndicators.dpo,
            CycleIndicators.ehlers_cycle_period,
            CycleIndicators.dominant_cycle,
            CycleIndicators.phase_accumulation,
            CycleIndicators.hilbert_transform_phase,
            CycleIndicators.market_cycle_model,
        ]

        for indicator_func in indicators:
            result = indicator_func(sample_candles)
            assert 0 <= result.phase <= 360, f"Phase {result.phase} out of range for {indicator_func.__name__}"

    def test_normalized_values_range(self, sample_candles):
        """Test that normalized values are in [-1, 1] range"""
        indicators = [
            CycleIndicators.dpo,
            CycleIndicators.ehlers_cycle_period,
            CycleIndicators.dominant_cycle,
            CycleIndicators.phase_accumulation,
            CycleIndicators.hilbert_transform_phase,
            CycleIndicators.market_cycle_model,
        ]

        for indicator_func in indicators:
            result = indicator_func(sample_candles)
            assert -1 <= result.normalized <= 1, f"Normalized {result.normalized} out of range for {indicator_func.__name__}"

    def test_confidence_values_range(self, sample_candles):
        """Test that confidence values are in [0, 1] range"""
        indicators = [
            CycleIndicators.dpo,
            CycleIndicators.ehlers_cycle_period,
            CycleIndicators.dominant_cycle,
            CycleIndicators.phase_accumulation,
            CycleIndicators.hilbert_transform_phase,
            CycleIndicators.market_cycle_model,
        ]

        for indicator_func in indicators:
            result = indicator_func(sample_candles)
            assert 0 <= result.confidence <= 1, f"Confidence {result.confidence} out of range for {indicator_func.__name__}"

    def test_cycle_period_positive(self, sample_candles):
        """Test that cycle periods are positive"""
        indicators = [
            CycleIndicators.dpo,
            CycleIndicators.ehlers_cycle_period,
            CycleIndicators.dominant_cycle,
            CycleIndicators.phase_accumulation,
            CycleIndicators.hilbert_transform_phase,
            CycleIndicators.market_cycle_model,
        ]

        for indicator_func in indicators:
            result = indicator_func(sample_candles)
            assert result.cycle_period > 0, f"Cycle period {result.cycle_period} not positive for {indicator_func.__name__}"

    def test_signal_types(self, sample_candles):
        """Test that signals are valid SignalStrength enums"""
        indicators = [
            CycleIndicators.dpo,
            CycleIndicators.ehlers_cycle_period,
            CycleIndicators.dominant_cycle,
            CycleIndicators.phase_accumulation,
            CycleIndicators.hilbert_transform_phase,
            CycleIndicators.market_cycle_model,
        ]

        for indicator_func in indicators:
            result = indicator_func(sample_candles)
            assert isinstance(result.signal, SignalStrength)
            assert result.signal in SignalStrength

    def test_description_presence(self, sample_candles):
        """Test that descriptions are present and meaningful"""
        indicators = [
            CycleIndicators.dpo,
            CycleIndicators.ehlers_cycle_period,
            CycleIndicators.dominant_cycle,
            CycleIndicators.phase_accumulation,
            CycleIndicators.hilbert_transform_phase,
            CycleIndicators.market_cycle_model,
        ]

        for indicator_func in indicators:
            result = indicator_func(sample_candles)
            assert isinstance(result.description, str)
            assert len(result.description) > 0
