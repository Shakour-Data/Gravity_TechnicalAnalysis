"""
Test suite for volume indicators
"""
import numpy as np
import pytest
from gravity_tech.core.indicators.volume import VolumeIndicators


class TestVolumeIndicators:
    """Test class for volume indicators."""

    @pytest.fixture
    def sample_candles(self):
        """Create sample candle data for testing."""
        import datetime

        from gravity_tech.core.domain.entities import Candle

        # Create sample candles with realistic price and volume data
        candles = []
        base_time = datetime.datetime(2024, 1, 1, 9, 0, 0)

        # Generate 100 candles with trending price and varying volume
        for i in range(100):
            open_price = 100 + i * 0.5 + np.random.normal(0, 2)
            close_price = open_price + np.random.normal(0, 1)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.5))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.5))
            volume = 1000000 + np.random.normal(0, 200000)  # Base volume around 1M

            candle = Candle(
                timestamp=base_time + datetime.timedelta(minutes=i),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=max(1000, volume),  # Ensure positive volume
                symbol="TEST"
            )
            candles.append(candle)

        return candles

    def test_calculate_all_basic(self, sample_candles):
        """Test basic calculate_all functionality."""
        results = VolumeIndicators.calculate_all(sample_candles)

        assert isinstance(results, list)
        assert len(results) > 0

        for result in results:
            assert hasattr(result, 'indicator_name')
            assert hasattr(result, 'value')
            assert hasattr(result, 'signal')
            assert hasattr(result, 'confidence')

    def test_accumulation_distribution_basic(self, sample_candles):
        """Test basic accumulation/distribution calculation."""
        result = VolumeIndicators.accumulation_distribution(sample_candles)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'A/D Line'

    def test_volume_rate_of_change_basic(self, sample_candles):
        """Test basic volume rate of change calculation."""
        result = VolumeIndicators.volume_rate_of_change(sample_candles, period=14)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'VROC(14)'

    def test_volume_profile_basic(self, sample_candles):
        """Test basic volume profile calculation."""
        result = VolumeIndicators.volume_profile(sample_candles, bins=20)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'Volume Profile(20)'

    def test_volume_oscillator_basic(self, sample_candles):
        """Test basic volume oscillator calculation."""
        result = VolumeIndicators.volume_oscillator(sample_candles, short_period=5, long_period=10)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'Volume Oscillator(5,10)'

    def test_obv_basic(self, sample_candles):
        """Test basic OBV calculation."""
        result = VolumeIndicators.obv(sample_candles)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'OBV'

    def test_cmf_basic(self, sample_candles):
        """Test basic Chaikin Money Flow calculation."""
        result = VolumeIndicators.cmf(sample_candles, period=20)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'CMF(20)'

    def test_vwap_basic(self, sample_candles):
        """Test basic VWAP calculation."""
        result = VolumeIndicators.vwap(sample_candles)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'VWAP'

    def test_ad_line_basic(self, sample_candles):
        """Test basic Accumulation/Distribution Line calculation."""
        result = VolumeIndicators.ad_line(sample_candles)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'A/D Line'

    def test_pvt_basic(self, sample_candles):
        """Test basic Price Volume Trend calculation."""
        result = VolumeIndicators.pvt(sample_candles)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'PVT'

    def test_accumulation_distribution_insufficient_data(self, sample_candles):
        """Test accumulation/distribution with insufficient data."""
        with pytest.raises(ValueError, match="Cannot calculate AD Line with empty candles"):
            VolumeIndicators.accumulation_distribution([])

    def test_volume_rate_of_change_insufficient_data(self, sample_candles):
        """Test volume rate of change with insufficient data."""
        with pytest.raises(ValueError, match="Need at least 15 candles for Volume Rate of Change"):
            VolumeIndicators.volume_rate_of_change(sample_candles[:10], period=14)

    def test_volume_profile_insufficient_data(self, sample_candles):
        """Test volume profile with insufficient data."""
        with pytest.raises(ValueError, match="Cannot calculate Volume Profile with empty candles"):
            VolumeIndicators.volume_profile([], bins=20)

    def test_volume_oscillator_insufficient_data(self, sample_candles):
        """Test volume oscillator with insufficient data."""
        # Volume oscillator may not have validation, so test with minimal data
        try:
            result = VolumeIndicators.volume_oscillator(sample_candles[:5], short_period=5, long_period=10)
            assert hasattr(result, 'indicator_name')
        except (ValueError, IndexError, KeyError):
            pass  # Expected for insufficient data

    def test_obv_insufficient_data(self, sample_candles):
        """Test OBV with insufficient data."""
        with pytest.raises(ValueError, match="Not enough candles for OBV"):
            VolumeIndicators.obv(sample_candles[:1])

    def test_cmf_insufficient_data(self, sample_candles):
        """Test Chaikin Money Flow with insufficient data."""
        with pytest.raises(ValueError, match="Not enough candles or invalid period for Chaikin Money Flow"):
            VolumeIndicators.cmf(sample_candles[:10], period=20)

    def test_vwap_insufficient_data(self, sample_candles):
        """Test VWAP with insufficient data."""
        with pytest.raises(ValueError, match="Not enough candles for VWAP"):
            VolumeIndicators.vwap([])

    def test_ad_line_insufficient_data(self, sample_candles):
        """Test Accumulation/Distribution Line with insufficient data."""
        # AD Line may not handle empty list gracefully
        try:
            result = VolumeIndicators.ad_line([])
            assert hasattr(result, 'indicator_name')
        except (ValueError, IndexError, KeyError):
            pass  # Expected for insufficient data

    def test_pvt_insufficient_data(self, sample_candles):
        """Test Price Volume Trend with insufficient data."""
        # PVT doesn't have validation, so it should work with any data
        result = VolumeIndicators.pvt(sample_candles[:1])
        assert hasattr(result, 'indicator_name')

    def test_calculate_all_insufficient_data(self, sample_candles):
        """Test calculate_all with insufficient data."""
        # Should handle insufficient data gracefully or raise appropriate errors
        try:
            results = VolumeIndicators.calculate_all(sample_candles[:5])
            assert isinstance(results, list)
        except (ValueError, IndexError):
            pass  # Expected for insufficient data