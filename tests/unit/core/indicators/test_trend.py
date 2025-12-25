import sys
from pathlib import Path

import pytest

# Add src to path
src_path = Path(__file__).resolve().parents[5] / "apps" / "analysis_api" / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from gravity_tech.core.domain.entities import Candle
from gravity_tech.core.indicators.trend import TrendIndicators


@pytest.fixture
def sample_candles():
    """Sample candles for testing"""
    return [
        Candle(open=100, high=105, low=95, close=102, volume=1000, timestamp="2023-01-01"),
        Candle(open=102, high=107, low=97, close=104, volume=1000, timestamp="2023-01-02"),
        Candle(open=104, high=109, low=99, close=106, volume=1000, timestamp="2023-01-03"),
        Candle(open=106, high=111, low=101, close=108, volume=1000, timestamp="2023-01-04"),
        Candle(open=108, high=113, low=103, close=110, volume=1000, timestamp="2023-01-05"),
        Candle(open=110, high=115, low=105, close=112, volume=1000, timestamp="2023-01-06"),
        Candle(open=112, high=117, low=107, close=114, volume=1000, timestamp="2023-01-07"),
        Candle(open=114, high=119, low=109, close=116, volume=1000, timestamp="2023-01-08"),
        Candle(open=116, high=121, low=111, close=118, volume=1000, timestamp="2023-01-09"),
        Candle(open=118, high=123, low=113, close=120, volume=1000, timestamp="2023-01-10"),
        Candle(open=120, high=125, low=115, close=122, volume=1000, timestamp="2023-01-11"),
        Candle(open=122, high=127, low=117, close=124, volume=1000, timestamp="2023-01-12"),
        Candle(open=124, high=129, low=119, close=126, volume=1000, timestamp="2023-01-13"),
        Candle(open=126, high=131, low=121, close=128, volume=1000, timestamp="2023-01-14"),
        Candle(open=128, high=133, low=123, close=130, volume=1000, timestamp="2023-01-15"),
        Candle(open=130, high=135, low=125, close=132, volume=1000, timestamp="2023-01-16"),
        Candle(open=132, high=137, low=127, close=134, volume=1000, timestamp="2023-01-17"),
        Candle(open=134, high=139, low=129, close=136, volume=1000, timestamp="2023-01-18"),
        Candle(open=136, high=141, low=131, close=138, volume=1000, timestamp="2023-01-19"),
        Candle(open=138, high=143, low=133, close=140, volume=1000, timestamp="2023-01-20"),
        Candle(open=140, high=145, low=135, close=142, volume=1000, timestamp="2023-01-21"),
        Candle(open=142, high=147, low=137, close=144, volume=1000, timestamp="2023-01-22"),
        Candle(open=144, high=149, low=139, close=146, volume=1000, timestamp="2023-01-23"),
        Candle(open=146, high=151, low=141, close=148, volume=1000, timestamp="2023-01-24"),
        Candle(open=148, high=153, low=143, close=150, volume=1000, timestamp="2023-01-25"),
        Candle(open=150, high=155, low=145, close=152, volume=1000, timestamp="2023-01-26"),
        Candle(open=152, high=157, low=147, close=154, volume=1000, timestamp="2023-01-27"),
        Candle(open=154, high=159, low=149, close=156, volume=1000, timestamp="2023-01-28"),
        Candle(open=156, high=161, low=151, close=158, volume=1000, timestamp="2023-01-29"),
        Candle(open=158, high=163, low=153, close=160, volume=1000, timestamp="2023-01-30"),
    ]


class TestTrendIndicators:
    """Test class for TrendIndicators"""

    def test_sma_basic(self, sample_candles):
        result = TrendIndicators.sma(sample_candles, period=5)
        assert result is not None
        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == "SMA(5)"
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_sma_insufficient_data(self, sample_candles):
        with pytest.raises(ValueError, match="Need at least 50 candles for SMA"):
            TrendIndicators.sma(sample_candles[:10], period=50)

    def test_ema_basic(self, sample_candles):
        result = TrendIndicators.ema(sample_candles, period=5)
        assert result is not None
        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == "EMA(5)"
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_ema_insufficient_data(self, sample_candles):
        with pytest.raises(ValueError, match="Need at least 50 candles for EMA"):
            TrendIndicators.ema(sample_candles[:10], period=50)

    def test_wma_basic(self, sample_candles):
        result = TrendIndicators.wma(sample_candles, period=5)
        assert result is not None
        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == "WMA(5)"
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_wma_insufficient_data(self, sample_candles):
        with pytest.raises(ValueError, match="Need at least 50 candles for WMA"):
            TrendIndicators.wma(sample_candles[:10], period=50)

    def test_dema_basic(self, sample_candles):
        result = TrendIndicators.dema(sample_candles, period=5)
        assert result is not None
        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == "DEMA(5)"
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_dema_insufficient_data(self, sample_candles):
        with pytest.raises(ValueError, match="Need at least 100 candles for DEMA"):
            TrendIndicators.dema(sample_candles[:10], period=50)

    def test_tema_basic(self, sample_candles):
        result = TrendIndicators.tema(sample_candles, period=5)
        assert result is not None
        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == "TEMA(5)"
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_tema_insufficient_data(self, sample_candles):
        with pytest.raises(ValueError, match="Need at least 150 candles for TEMA"):
            TrendIndicators.tema(sample_candles[:10], period=50)

    def test_macd_basic(self, sample_candles):
        result = TrendIndicators.macd(sample_candles, fast=12, slow=26, signal_period=9)
        assert result is not None
        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == "MACD"
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_macd_insufficient_data(self, sample_candles):
        with pytest.raises(ValueError, match="Need at least 26 candles for MACD"):
            TrendIndicators.macd(sample_candles[:10], fast=12, slow=26, signal_period=9)

    def test_adx_basic(self, sample_candles):
        result = TrendIndicators.adx(sample_candles, period=14)
        assert result is not None
        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == "ADX(14)"
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_adx_insufficient_data(self, sample_candles):
        with pytest.raises(ValueError, match="Need at least 28 candles for ADX"):
            TrendIndicators.adx(sample_candles[:10], period=14)

    def test_donchian_channels_basic(self, sample_candles):
        result = TrendIndicators.donchian_channels(sample_candles, period=20)
        assert result is not None
        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == "Donchian Channels(20)"
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_donchian_channels_insufficient_data(self, sample_candles):
        with pytest.raises(ValueError, match="Need at least 20 candles for Donchian Channels"):
            TrendIndicators.donchian_channels(sample_candles[:10], period=20)

    def test_aroon_basic(self, sample_candles):
        result = TrendIndicators.aroon(sample_candles, period=25)
        assert result is not None
        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == "Aroon(25)"
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_aroon_insufficient_data(self, sample_candles):
        with pytest.raises(ValueError, match="Need at least 25 candles for Aroon"):
            TrendIndicators.aroon(sample_candles[:10], period=25)

    def test_vortex_indicator_basic(self, sample_candles):
        result = TrendIndicators.vortex_indicator(sample_candles, period=14)
        assert result is not None
        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == "Vortex Indicator(14)"
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_vortex_indicator_insufficient_data(self, sample_candles):
        with pytest.raises(ValueError, match="Need at least 15 candles for Vortex Indicator"):
            TrendIndicators.vortex_indicator(sample_candles[:10], period=14)

    def test_mcginley_dynamic_basic(self, sample_candles):
        result = TrendIndicators.mcginley_dynamic(sample_candles, period=20, k_factor=0.6)
        assert result is not None
        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == "McGinley Dynamic(20)"
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_mcginley_dynamic_insufficient_data(self, sample_candles):
        with pytest.raises(ValueError, match="Need at least 20 candles for McGinley Dynamic"):
            TrendIndicators.mcginley_dynamic(sample_candles[:10], period=20, k_factor=0.6)

    def test_calculate_all_basic(self, sample_candles):
        results = TrendIndicators.calculate_all(sample_candles)
        assert isinstance(results, list)
        assert len(results) > 0
        for result in results:
            assert hasattr(result, 'indicator_name')
            assert hasattr(result, 'signal')
            assert hasattr(result, 'confidence')
            assert result.confidence >= 0.0
            assert result.confidence <= 1.0