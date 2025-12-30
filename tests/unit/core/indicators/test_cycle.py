"""Unit tests for cycle indicators."""

import numpy as np
import pytest
from gravity_tech.core.indicators.cycle import (
    CycleIndicators,
    convert_cycle_to_indicator_result,
)


class TestCycleIndicators:
    """Test Cycle Indicators class."""

    @pytest.fixture
    def sample_candles(self):
        """Create sample candle data for testing."""
        from datetime import datetime, timedelta

        from gravity_tech.core.domain.entities import Candle

        # Create 100 sample candles with trending data
        candles = []
        base_date = datetime(2023, 1, 1)
        base_price = 100.0
        for i in range(100):
            current_date = base_date + timedelta(days=i)
            open_price = base_price + np.sin(i * 0.1) * 5
            close_price = open_price + np.random.normal(0, 1)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.5))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.5))
            volume = 1000000 + np.random.normal(0, 100000)

            candle = Candle(
                timestamp=current_date,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                symbol="BTCUSDT",
                timeframe="1h",
            )
            candles.append(candle)
            base_price += 0.1  # Slight upward trend

        return candles

    def test_dpo_basic(self, sample_candles):
        """Test basic DPO calculation."""
        result = CycleIndicators.dpo(sample_candles, period=20)

        assert hasattr(result, "value")
        assert hasattr(result, "phase")
        assert hasattr(result, "confidence")  # Instead of strength
        assert hasattr(result, "cycle_period")
        assert isinstance(result.value, (int, float))
        assert isinstance(result.phase, (int, float))
        assert 0 <= result.phase <= 360
        assert isinstance(result.confidence, (int, float))
        assert 0 <= result.confidence <= 1

    def test_dpo_insufficient_data(self, sample_candles):
        """Test DPO with insufficient data."""
        short_candles = sample_candles[:10]  # Less than period + 1
        with pytest.raises(ValueError, match="Not enough candles"):
            CycleIndicators.dpo(short_candles, period=20)

    def test_ehlers_cycle_period_basic(self, sample_candles):
        """Test basic Ehler's cycle period calculation."""
        result = CycleIndicators.ehlers_cycle_period(sample_candles, smooth_period=5)

        assert hasattr(result, "value")
        assert hasattr(result, "phase")
        assert hasattr(result, "confidence")
        assert hasattr(result, "cycle_period")
        assert isinstance(result.cycle_period, int)
        assert result.cycle_period > 0

    def test_dominant_cycle_basic(self, sample_candles):
        """Test basic dominant cycle calculation."""
        result = CycleIndicators.dominant_cycle(sample_candles, min_period=8, max_period=50)

        assert hasattr(result, "value")
        assert hasattr(result, "phase")
        assert hasattr(result, "confidence")
        assert hasattr(result, "cycle_period")
        assert isinstance(result.cycle_period, int)
        assert 8 <= result.cycle_period <= 50

    def test_schaff_trend_cycle_basic(self, sample_candles):
        """Test basic Schaff Trend Cycle calculation."""
        result = CycleIndicators.schaff_trend_cycle(sample_candles, fast=23, slow=50, cycle=10)

        assert hasattr(result, "indicator_name")
        assert hasattr(result, "value")
        assert hasattr(result, "signal")
        assert hasattr(result, "confidence")
        assert result.indicator_name == "STC(23,50,10)"

    def test_phase_accumulation_basic(self, sample_candles):
        """Test basic phase accumulation calculation."""
        result = CycleIndicators.phase_accumulation(sample_candles, period=14)

        assert hasattr(result, "value")
        assert hasattr(result, "phase")
        assert hasattr(result, "confidence")
        assert hasattr(result, "cycle_period")
        assert 0 <= result.phase <= 360

    def test_hilbert_transform_phase_basic(self, sample_candles):
        """Test basic Hilbert transform phase calculation."""
        result = CycleIndicators.hilbert_transform_phase(sample_candles, period=7)

        assert hasattr(result, "value")
        assert hasattr(result, "phase")
        assert hasattr(result, "confidence")
        assert hasattr(result, "cycle_period")
        assert 0 <= result.phase <= 360

    def test_market_cycle_model_basic(self, sample_candles):
        """Test basic market cycle model calculation."""
        result = CycleIndicators.market_cycle_model(sample_candles, lookback=50)

        assert hasattr(result, "value")
        assert hasattr(result, "phase")
        assert hasattr(result, "confidence")
        assert hasattr(result, "cycle_period")
        assert isinstance(result.phase, (int, float))
        assert 0 <= result.phase <= 360

    def test_calculate_all_basic(self, sample_candles):
        """Test calculate_all method."""
        results = CycleIndicators.calculate_all(sample_candles)

        assert isinstance(results, list)
        assert len(results) > 0
        for result in results:
            assert hasattr(result, "indicator_name")
            assert hasattr(result, "value")
            assert hasattr(result, "signal")
            assert hasattr(result, "confidence")

    def test_detrended_price_oscillator_basic(self, sample_candles):
        """Test detrended price oscillator calculation."""
        result = CycleIndicators.dpo(sample_candles, period=20)  # DPO is detrended price oscillator

        assert hasattr(result, "value")
        assert hasattr(result, "phase")
        assert hasattr(result, "confidence")
        assert hasattr(result, "cycle_period")
        assert isinstance(result.value, (int, float))

    def test_convert_cycle_to_indicator_result(self, sample_candles):
        """Test conversion from CycleResult to IndicatorResult."""
        cycle_result = CycleIndicators.dpo(sample_candles, period=20)
        indicator_result = convert_cycle_to_indicator_result(cycle_result, "DPO")

        assert hasattr(indicator_result, "indicator_name")
        assert hasattr(indicator_result, "value")
        assert hasattr(indicator_result, "signal")
        assert hasattr(indicator_result, "confidence")
        assert indicator_result.indicator_name == "DPO"
