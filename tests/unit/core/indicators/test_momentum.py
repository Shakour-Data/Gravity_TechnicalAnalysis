"""
Unit tests for momentum indicators.
"""
import numpy as np
import pytest
from gravity_tech.core.indicators.momentum import (
    MomentumIndicators,
    connors_rsi,
    schaff_trend_cycle,
    tsi,
)


class TestTSI:
    """Test True Strength Index."""

    def test_tsi_basic(self):
        """Test basic TSI calculation."""
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 10)
        result = tsi(prices)

        assert isinstance(result, dict)
        assert 'values' in result
        assert 'signal' in result
        assert 'confidence' in result

        assert len(result['values']) == len(prices)
        assert isinstance(result['values'], np.ndarray)
        assert result['signal'] in [None, 'BUY', 'SELL']
        assert 0 <= result['confidence'] <= 1

    def test_tsi_insufficient_data(self):
        """Test TSI with insufficient data."""
        prices = np.array([100, 101])
        with pytest.raises(ValueError, match="insufficient data"):
            tsi(prices)

    def test_tsi_zero_prices(self):
        """Test TSI with zero prices."""
        prices = np.zeros(50)
        result = tsi(prices)

        assert len(result['values']) == len(prices)
        # All zeros should give TSI of 0
        assert np.allclose(result['values'], 0, atol=1e-10)


class TestSchaffTrendCycle:
    """Test Schaff Trend Cycle."""

    def test_schaff_basic(self):
        """Test basic Schaff Trend Cycle calculation."""
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 10)
        result = schaff_trend_cycle(prices)

        assert isinstance(result, dict)
        assert 'values' in result
        assert 'signal' in result
        assert 'confidence' in result

        assert len(result['values']) == len(prices)
        assert isinstance(result['values'], np.ndarray)
        assert result['signal'] in [None, 'BUY', 'SELL']
        assert 0 <= result['confidence'] <= 1

    def test_schaff_insufficient_data(self):
        """Test Schaff with insufficient data."""
        prices = np.array([100, 101])
        with pytest.raises(ValueError, match="insufficient data"):
            schaff_trend_cycle(prices)


class TestConnorsRSI:
    """Test Connors RSI."""

    def test_connors_rsi_basic(self):
        """Test basic Connors RSI calculation."""
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 10)
        result = connors_rsi(prices)

        assert isinstance(result, dict)
        assert 'values' in result
        assert 'signal' in result
        assert 'confidence' in result

        assert len(result['values']) == len(prices)
        assert isinstance(result['values'], np.ndarray)
        assert result['signal'] in [None, 'BUY', 'SELL']
        assert 0 <= result['confidence'] <= 1

    def test_connors_rsi_insufficient_data(self):
        """Test Connors RSI with insufficient data."""
        prices = np.array([100, 101])
        with pytest.raises(ValueError, match="insufficient data"):
            connors_rsi(prices)

    def test_connors_rsi_constant_prices(self):
        """Test Connors RSI with constant prices."""
        prices = np.full(200, 100.0)
        result = connors_rsi(prices)

        assert len(result['values']) == len(prices)
        # Constant prices should give CRSI of 0 (no momentum)
        assert np.allclose(result['values'][-10:], 0, atol=1e-10)


class TestMomentumIndicators:
    """Test MomentumIndicators class methods."""

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
        results = MomentumIndicators.calculate_all(sample_candles)

        assert isinstance(results, list)
        assert len(results) > 0

        for result in results:
            assert hasattr(result, 'indicator_name')
            assert hasattr(result, 'value')
            assert hasattr(result, 'signal')
            assert hasattr(result, 'confidence')

    def test_rsi_basic(self, sample_candles):
        """Test basic RSI calculation."""
        result = MomentumIndicators.rsi(sample_candles, period=14)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'RSI(14)'

    def test_stochastic_basic(self, sample_candles):
        """Test basic Stochastic calculation."""
        result = MomentumIndicators.stochastic(sample_candles, k_period=14, d_period=3)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'Stochastic(14,3)'

    def test_cci_basic(self, sample_candles):
        """Test basic CCI calculation."""
        result = MomentumIndicators.cci(sample_candles, period=20)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'CCI(20)'

    def test_roc_basic(self, sample_candles):
        """Test basic ROC calculation."""
        result = MomentumIndicators.roc(sample_candles, period=12)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'ROC(12)'

    def test_williams_r_basic(self, sample_candles):
        """Test basic Williams %R calculation."""
        result = MomentumIndicators.williams_r(sample_candles, period=14)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'Williams %R(14)'

    def test_mfi_basic(self, sample_candles):
        """Test basic MFI calculation."""
        result = MomentumIndicators.mfi(sample_candles, period=14)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'MFI(14)'

    def test_ultimate_oscillator_basic(self, sample_candles):
        """Test basic Ultimate Oscillator calculation."""
        result = MomentumIndicators.ultimate_oscillator(sample_candles, period1=7, period2=14, period3=28)

        assert hasattr(result, 'indicator_name')
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
        assert hasattr(result, 'confidence')
        assert result.indicator_name == 'Ultimate Oscillator(7,14,28)'
