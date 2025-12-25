"""Unit tests for volatility indicators."""

import numpy as np
import pytest
from gravity_tech.core.domain.entities import Candle
from gravity_tech.core.domain.entities.indicator_result import IndicatorResult
from gravity_tech.core.indicators.volatility import VolatilityIndicators


class TestVolatilityIndicators:
    """Test VolatilityIndicators class."""

    @pytest.fixture
    def sample_candles(self):
        """Create sample candles for testing."""
        return [
            Candle(
                timestamp="2023-01-01T00:00:00Z",
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000
            ),
            Candle(
                timestamp="2023-01-02T00:00:00Z",
                open=102.0,
                high=108.0,
                low=98.0,
                close=105.0,
                volume=1100
            ),
            Candle(
                timestamp="2023-01-03T00:00:00Z",
                open=105.0,
                high=110.0,
                low=100.0,
                close=108.0,
                volume=1200
            ),
            Candle(
                timestamp="2023-01-04T00:00:00Z",
                open=108.0,
                high=112.0,
                low=105.0,
                close=110.0,
                volume=1300
            ),
            Candle(
                timestamp="2023-01-05T00:00:00Z",
                open=110.0,
                high=115.0,
                low=108.0,
                close=112.0,
                volume=1400
            ),
        ] * 5  # Repeat to have enough data

    def test_true_range(self, sample_candles):
        """Test true range calculation."""
        tr = VolatilityIndicators.true_range(sample_candles)

        assert isinstance(tr, np.ndarray)
        assert len(tr) == len(sample_candles)
        assert tr[0] == sample_candles[0].high - sample_candles[0].low  # First TR is just high-low
        assert all(tr >= 0)  # All true ranges should be non-negative

    def test_atr_basic(self, sample_candles):
        """Test basic ATR calculation."""
        result = VolatilityIndicators.atr(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "ATR(14)"
        assert result.category.name == "VOLATILITY"
        assert result.signal is not None
        assert 0 <= result.confidence <= 1
        assert "atr" in result.additional_values
        assert "atr_percent" in result.additional_values
        assert "percentile" in result.additional_values

    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data."""
        candles = [
            Candle(
                timestamp="2023-01-01T00:00:00Z",
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000
            )
        ]
        # ATR should handle single candle gracefully or raise appropriate error
        try:
            result = VolatilityIndicators.atr(candles)
            assert isinstance(result, IndicatorResult)
        except (IndexError, ValueError, ZeroDivisionError):
            pass  # Expected for insufficient data

    def test_bollinger_bands_basic(self, sample_candles):
        """Test basic Bollinger Bands calculation."""
        result = VolatilityIndicators.bollinger_bands(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "Bollinger Bands(20,2.0)"
        assert result.category.name == "VOLATILITY"
        assert result.signal is not None
        assert 0 <= result.confidence <= 1
        assert "upper" in result.additional_values
        assert "lower" in result.additional_values
        assert "middle" in result.additional_values
        assert "bandwidth" in result.additional_values

    def test_bollinger_bands_insufficient_data(self):
        """Test Bollinger Bands with insufficient data."""
        candles = [
            Candle(
                timestamp="2023-01-01T00:00:00Z",
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000
            )
        ]
        # BB should handle insufficient data gracefully
        try:
            result = VolatilityIndicators.bollinger_bands(candles)
            assert isinstance(result, IndicatorResult)
        except (IndexError, ValueError, ZeroDivisionError):
            pass  # Expected for insufficient data

    def test_keltner_channel_basic(self, sample_candles):
        """Test basic Keltner Channel calculation."""
        from gravity_tech.core.indicators.volatility import convert_volatility_to_indicator_result

        vol_result = VolatilityIndicators.keltner_channel(sample_candles)
        result = convert_volatility_to_indicator_result(vol_result, "KC(20,2.0)")

        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "KC(20,2.0)"
        assert result.category.name == "VOLATILITY"
        assert result.signal is not None
        assert 0 <= result.confidence <= 1

    def test_donchian_channel_basic(self, sample_candles):
        """Test basic Donchian Channel calculation."""
        from gravity_tech.core.indicators.volatility import convert_volatility_to_indicator_result

        vol_result = VolatilityIndicators.donchian_channel(sample_candles)
        result = convert_volatility_to_indicator_result(vol_result, "DC(20)")

        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "DC(20)"
        assert result.category.name == "VOLATILITY"
        assert result.signal is not None
        assert 0 <= result.confidence <= 1

    def test_standard_deviation_basic(self, sample_candles):
        """Test basic Standard Deviation calculation."""
        from gravity_tech.core.indicators.volatility import convert_volatility_to_indicator_result

        vol_result = VolatilityIndicators.standard_deviation(sample_candles)
        result = convert_volatility_to_indicator_result(vol_result, "STD(20)")

        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "STD(20)"
        assert result.category.name == "VOLATILITY"
        assert result.signal is not None
        assert 0 <= result.confidence <= 1

    def test_historical_volatility_basic(self, sample_candles):
        """Test basic Historical Volatility calculation."""
        from gravity_tech.core.indicators.volatility import convert_volatility_to_indicator_result

        vol_result = VolatilityIndicators.historical_volatility(sample_candles)
        result = convert_volatility_to_indicator_result(vol_result, "HV(20)")

        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "HV(20)"
        assert result.category.name == "VOLATILITY"
        assert result.signal is not None
        assert 0 <= result.confidence <= 1

    def test_atr_percentage_basic(self, sample_candles):
        """Test basic ATR Percentage calculation."""
        from gravity_tech.core.indicators.volatility import convert_volatility_to_indicator_result

        vol_result = VolatilityIndicators.atr_percentage(sample_candles)
        result = convert_volatility_to_indicator_result(vol_result, "ATR%(14)")

        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "ATR%(14)"
        assert result.category.name == "VOLATILITY"
        assert result.signal is not None
        assert 0 <= result.confidence <= 1

    def test_chaikin_volatility_basic(self, sample_candles):
        """Test basic Chaikin Volatility calculation."""
        from gravity_tech.core.indicators.volatility import convert_volatility_to_indicator_result

        vol_result = VolatilityIndicators.chaikin_volatility(sample_candles)
        result = convert_volatility_to_indicator_result(vol_result, "CV(10,10)")

        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "CV(10,10)"
        assert result.category.name == "VOLATILITY"
        assert result.signal is not None
        assert 0 <= result.confidence <= 1

    def test_calculate_all_basic(self, sample_candles):
        """Test calculate_all method."""
        from gravity_tech.core.indicators.volatility import VolatilityResult

        results = VolatilityIndicators.calculate_all(sample_candles)

        assert isinstance(results, dict)
        assert len(results) > 0
        for key, result in results.items():
            assert isinstance(result, (IndicatorResult, VolatilityResult))
            if isinstance(result, IndicatorResult):
                assert result.category.name == "VOLATILITY"
                assert 0 <= result.confidence <= 1
            elif isinstance(result, VolatilityResult):
                assert hasattr(result, 'value')
                assert hasattr(result, 'normalized')
                assert hasattr(result, 'percentile')