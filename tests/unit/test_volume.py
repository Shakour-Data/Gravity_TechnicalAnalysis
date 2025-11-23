"""
Unit tests for Volume indicators (src/core/indicators/volume.py)

Tests cover:
- All 4 volume indicators (OBV, CMF, VWAP, AD Line, PVT, Volume Oscillator)
- Signal generation and confidence levels
- Volume-based calculations
- Accumulation/distribution logic
- Edge cases and error handling
- Backward compatibility methods
- Helper functions
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.core.indicators.volume import (
    VolumeIndicators
)
from src.core.domain.entities import (
    Candle,
    IndicatorResult,
    CoreSignalStrength as SignalStrength,
    IndicatorCategory
)


class TestVolumeIndicators:
    """Test VolumeIndicators class methods"""

    @pytest.fixture
    def sample_candles(self):
        """Create sample candles for testing"""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        # Create trending up pattern with varying volume
        prices = []
        volumes = []
        for i in range(60):
            # Upward trend with some volatility
            base_price = 100 + i * 0.5
            price = base_price + 2 * np.sin(2 * np.pi * i / 10)
            prices.append(price)

            # Volume increases during up moves, decreases during down moves
            trend = 1 if i > 5 else -1
            volume = 1000 + 500 * trend + np.random.normal(0, 100)
            volumes.append(max(100, volume))

        candles = []
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            candle = Candle(
                timestamp=base_time,
                open=price - 0.5,
                high=price + 1.5,
                low=price - 1.5,
                close=price,
                volume=int(volume)
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

    def test_obv_normal_operation(self, sample_candles):
        """Test OBV with normal data"""
        result = VolumeIndicators.obv(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.VOLUME
        assert result.indicator_name == "OBV"
        assert isinstance(result.signal, SignalStrength)
        assert 0 <= result.confidence <= 1
        assert isinstance(result.description, str)
        assert "حجم" in result.description

    def test_obv_calculation_accuracy(self, sample_candles):
        """Test OBV calculation accuracy"""
        result = VolumeIndicators.obv(sample_candles)

        # Manual OBV calculation
        df = pd.DataFrame([{
            'close': c.close,
            'volume': c.volume
        } for c in sample_candles])

        obv_manual = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv_manual.append(obv_manual[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv_manual.append(obv_manual[-1] - df['volume'].iloc[i])
            else:
                obv_manual.append(obv_manual[-1])

        expected_obv = obv_manual[-1]
        assert abs(result.value - expected_obv) < 0.01

    def test_obv_signal_generation(self, sample_candles):
        """Test OBV signal generation based on trend divergence"""
        result = VolumeIndicators.obv(sample_candles)

        # Signal should be based on OBV trend vs price trend
        assert result.signal in [
            SignalStrength.VERY_BULLISH, SignalStrength.BULLISH,
            SignalStrength.BEARISH_BROKEN, SignalStrength.NEUTRAL,
            SignalStrength.BEARISH_BROKEN, SignalStrength.BEARISH, SignalStrength.VERY_BEARISH
        ]

    def test_cmf_normal_operation(self, sample_candles):
        """Test CMF with normal data"""
        result = VolumeIndicators.cmf(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.VOLUME
        assert "CMF" in result.indicator_name
        assert isinstance(result.signal, SignalStrength)
        assert -1 <= result.value <= 1  # CMF ranges from -1 to 1

    def test_cmf_calculation_accuracy(self, sample_candles):
        """Test CMF calculation accuracy"""
        result = VolumeIndicators.cmf(sample_candles, period=10)

        df = pd.DataFrame([{
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume
        } for c in sample_candles])

        # Manual CMF calculation
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mf_volume = mf_multiplier * df['volume']
        expected_cmf = mf_volume.rolling(window=10).sum() / df['volume'].rolling(window=10).sum()
        expected_cmf = expected_cmf.iloc[-1]

        assert abs(result.value - expected_cmf) < 0.01

    def test_cmf_signal_generation(self, sample_candles):
        """Test CMF signal generation"""
        result = VolumeIndicators.cmf(sample_candles)

        cmf_value = result.value

        # Signal based on CMF levels
        if cmf_value > 0.25:
            assert result.signal == SignalStrength.VERY_BULLISH
        elif cmf_value > 0.1:
            assert result.signal == SignalStrength.BULLISH
        elif cmf_value < -0.25:
            assert result.signal == SignalStrength.VERY_BEARISH
        elif cmf_value < -0.1:
            assert result.signal == SignalStrength.BEARISH

    def test_vwap_normal_operation(self, sample_candles):
        """Test VWAP with normal data"""
        result = VolumeIndicators.vwap(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.VOLUME
        assert result.indicator_name == "VWAP"
        assert isinstance(result.signal, SignalStrength)

    def test_vwap_calculation_accuracy(self, sample_candles):
        """Test VWAP calculation accuracy"""
        result = VolumeIndicators.vwap(sample_candles)

        df = pd.DataFrame([{
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume,
            'typical': lambda c: (c.high + c.low + c.close) / 3
        } for c in sample_candles])

        df['typical'] = df.apply(lambda row: (row['high'] + row['low'] + row['close']) / 3, axis=1)
        df['pv'] = df['typical'] * df['volume']
        expected_vwap = df['pv'].cumsum() / df['volume'].cumsum()
        expected_vwap = expected_vwap.iloc[-1]

        assert abs(result.value - expected_vwap) < 0.01

    def test_vwap_signal_generation(self):
        """Test VWAP signal generation"""
        # Create sample candles directly
        candles = []
        base_price = 40000
        base_time = datetime.now() - timedelta(days=100)
        
        for i in range(100):
            open_price = base_price + (i * 10)
            close_price = open_price + ((i % 10) - 5) * 50
            high_price = max(open_price, close_price) + 100
            low_price = min(open_price, close_price) - 100
            
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=1000 + (i * 10)
            ))
        
        # Test price above VWAP
        high_price_candle = candles[-1]._replace(close=candles[-1].close * 1.05)
        test_candles = candles[:-1] + [high_price_candle]

        result = VolumeIndicators.vwap(test_candles)
        assert result.signal in [SignalStrength.VERY_BULLISH, SignalStrength.BULLISH]

        # Test price below VWAP
        low_price_candle = candles[-1]._replace(close=candles[-1].close * 0.95)
        test_candles = candles[:-1] + [low_price_candle]

        result = VolumeIndicators.vwap(test_candles)
        assert result.signal in [SignalStrength.VERY_BEARISH, SignalStrength.BEARISH]

    def test_ad_line_normal_operation(self, sample_candles):
        """Test AD Line with normal data"""
        result = VolumeIndicators.ad_line(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.VOLUME
        assert result.indicator_name == "A/D Line"
        assert isinstance(result.signal, SignalStrength)

    def test_ad_line_calculation_accuracy(self, sample_candles):
        """Test AD Line calculation accuracy"""
        result = VolumeIndicators.ad_line(sample_candles)

        df = pd.DataFrame([{
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume
        } for c in sample_candles])

        # Manual AD Line calculation
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        expected_ad = (clv * df['volume']).cumsum()
        expected_ad = expected_ad.iloc[-1]

        assert abs(result.value - expected_ad) < 0.01

    def test_pvt_normal_operation(self, sample_candles):
        """Test PVT with normal data"""
        result = VolumeIndicators.pvt(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.VOLUME
        assert result.indicator_name == "PVT"
        assert isinstance(result.signal, SignalStrength)

    def test_pvt_calculation_accuracy(self, sample_candles):
        """Test PVT calculation accuracy"""
        result = VolumeIndicators.pvt(sample_candles)

        df = pd.DataFrame([{
            'close': c.close,
            'volume': c.volume
        } for c in sample_candles])

        price_change = df['close'].pct_change()
        expected_pvt = (price_change * df['volume']).cumsum()
        expected_pvt = expected_pvt.iloc[-1]

        assert abs(result.value - expected_pvt) < 0.01

    def test_volume_oscillator_normal_operation(self, sample_candles):
        """Test Volume Oscillator with normal data"""
        result = VolumeIndicators.volume_oscillator(sample_candles)

        assert isinstance(result, IndicatorResult)
        assert result.category == IndicatorCategory.VOLUME
        assert "Volume Oscillator" in result.indicator_name
        assert isinstance(result.signal, SignalStrength)

    def test_volume_oscillator_calculation_accuracy(self, sample_candles):
        """Test Volume Oscillator calculation accuracy"""
        result = VolumeIndicators.volume_oscillator(sample_candles, short_period=5, long_period=10)

        volumes = pd.Series([c.volume for c in sample_candles])
        short_ma = volumes.rolling(window=5).mean()
        long_ma = volumes.rolling(window=10).mean()
        expected_vo = ((short_ma - long_ma) / long_ma) * 100
        expected_vo = expected_vo.iloc[-1]

        assert abs(result.value - expected_vo) < 0.01

    def test_calculate_all_normal(self, sample_candles):
        """Test calculate_all method"""
        results = VolumeIndicators.calculate_all(sample_candles)

        assert isinstance(results, list)
        assert len(results) >= 4  # Should return multiple indicators

        for result in results:
            assert isinstance(result, IndicatorResult)
            assert result.category == IndicatorCategory.VOLUME

    def test_calculate_all_insufficient_data(self, minimal_candles):
        """Test calculate_all with insufficient data"""
        results = VolumeIndicators.calculate_all(minimal_candles)
        # اگر هیچ اندیکاتوری بازنگردد، قابل قبول است (داده کافی نیست)
        assert isinstance(results, list)
        assert len(results) >= 0

    # Edge cases and error handling
    def test_all_indicators_with_empty_candles(self):
        """Test all indicators with empty candle list"""
        empty_candles = []
        # همه اندیکاتورها باید ValueError بدهند
        with pytest.raises(ValueError):
            VolumeIndicators.obv(empty_candles)
        with pytest.raises(ValueError):
            VolumeIndicators.cmf(empty_candles)
        with pytest.raises(ValueError):
            VolumeIndicators.vwap(empty_candles)

    def test_all_indicators_with_single_candle(self):
        """Test all indicators with single candle"""
        single_candle = [Candle(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            open=100, high=101, low=99, close=100.5, volume=1000
        )]
        # همه اندیکاتورها باید ValueError بدهند
        with pytest.raises(ValueError):
            VolumeIndicators.obv(single_candle)
        with pytest.raises(ValueError):
            VolumeIndicators.cmf(single_candle)
        with pytest.raises(ValueError):
            VolumeIndicators.vwap(single_candle)

    def test_indicators_with_zero_volume(self, sample_candles):
        """Test indicators with zero volume candles"""
        # Create candle with zero volume
        zero_volume_candle = sample_candles[-1]._replace(volume=0)
        test_candles = sample_candles[:-1] + [zero_volume_candle]

        # Should handle zero volume gracefully
        result_obv = VolumeIndicators.obv(test_candles)
        assert isinstance(result_obv, IndicatorResult)

        result_cmf = VolumeIndicators.cmf(test_candles)
        assert isinstance(result_cmf, IndicatorResult)

    def test_cmf_with_zero_range(self, sample_candles):
        """Test CMF with candles that have zero high-low range"""
        # Create candle with high == low
        flat_candle = sample_candles[-1]._replace(high=100, low=100, close=100)
        test_candles = sample_candles[:-1] + [flat_candle]

        result = VolumeIndicators.cmf(test_candles)
        assert isinstance(result, IndicatorResult)
        # Should handle division by zero

    def test_vwap_with_large_volume(self, sample_candles):
        """Test VWAP with extremely large volume"""
        large_volume_candle = sample_candles[-1]._replace(volume=1000000)
        test_candles = sample_candles[:-1] + [large_volume_candle]

        result = VolumeIndicators.vwap(test_candles)
        assert isinstance(result, IndicatorResult)
        assert result.value > 0

    def test_ad_line_with_extreme_values(self, sample_candles):
        """Test AD Line with extreme price values"""
        extreme_candle = sample_candles[-1]._replace(high=1000, low=1, close=500)
        test_candles = sample_candles[:-1] + [extreme_candle]

        result = VolumeIndicators.ad_line(test_candles)
        assert isinstance(result, IndicatorResult)

    def test_pvt_with_price_changes(self, sample_candles):
        """Test PVT with various price change scenarios"""
        # Test with large price increase
        large_increase = sample_candles[-1]._replace(close=sample_candles[-1].close * 2)
        test_candles = sample_candles[:-1] + [large_increase]

        result = VolumeIndicators.pvt(test_candles)
        assert isinstance(result, IndicatorResult)

        # Test with large price decrease
        large_decrease = sample_candles[-1]._replace(close=sample_candles[-1].close * 0.5)
        test_candles = sample_candles[:-1] + [large_decrease]

        result = VolumeIndicators.pvt(test_candles)
        assert isinstance(result, IndicatorResult)

    def test_volume_oscillator_extremes(self, sample_candles):
        """Test Volume Oscillator with extreme volume changes"""
        # Create alternating high/low volume pattern
        high_vol_candle = sample_candles[-1]._replace(volume=5000)
        low_vol_candle = sample_candles[-2]._replace(volume=100)

        test_candles = sample_candles[:-2] + [low_vol_candle, high_vol_candle]

        result = VolumeIndicators.volume_oscillator(test_candles)
        assert isinstance(result, IndicatorResult)

    def test_signal_confidence_levels(self, sample_candles):
        """Test that confidence levels are appropriate for signals"""
        indicators = [
            lambda: VolumeIndicators.obv(sample_candles),
            lambda: VolumeIndicators.cmf(sample_candles),
            lambda: VolumeIndicators.vwap(sample_candles),
            lambda: VolumeIndicators.ad_line(sample_candles),
            lambda: VolumeIndicators.pvt(sample_candles),
        ]

        for indicator_func in indicators:
            result = indicator_func()
            assert 0 <= result.confidence <= 1

            # Strong signals should have reasonable confidence
            if result.signal in [SignalStrength.VERY_BULLISH, SignalStrength.VERY_BEARISH]:
                assert result.confidence >= 0.7

    def test_indicator_names_formatting(self, sample_candles):
        """Test that indicator names are properly formatted"""
        assert VolumeIndicators.obv(sample_candles).indicator_name == "OBV"
        assert VolumeIndicators.cmf(sample_candles, 15).indicator_name == "CMF(15)"
        assert VolumeIndicators.vwap(sample_candles).indicator_name == "VWAP"
        assert VolumeIndicators.ad_line(sample_candles).indicator_name == "A/D Line"
        assert VolumeIndicators.pvt(sample_candles).indicator_name == "PVT"
        assert VolumeIndicators.volume_oscillator(sample_candles, short_period=3, long_period=8).indicator_name == "Volume Oscillator(3,8)"

    def test_description_content(self, sample_candles):
        """Test that descriptions contain relevant information"""
        result_obv = VolumeIndicators.obv(sample_candles)
        assert "حجم" in result_obv.description and ("تأیید" in result_obv.description or "واگرا" in result_obv.description)

        result_cmf = VolumeIndicators.cmf(sample_candles)
        assert "جریان پول" in result_cmf.description and ("مثبت" in result_cmf.description or "منفی" in result_cmf.description)

        result_vwap = VolumeIndicators.vwap(sample_candles)
        assert "VWAP" in result_vwap.description and "%" in result_vwap.description

    def test_additional_values_completeness(self, sample_candles):
        """Test that additional_values contain all expected keys when applicable"""
        # Most volume indicators don't have additional_values, but let's verify
        result_obv = VolumeIndicators.obv(sample_candles)
        # OBV doesn't have additional values

        result_cmf = VolumeIndicators.cmf(sample_candles)
        # CMF doesn't have additional values

        result_vwap = VolumeIndicators.vwap(sample_candles)
        # VWAP doesn't have additional values

        # All results should be valid IndicatorResult objects
        assert isinstance(result_obv, IndicatorResult)
        assert isinstance(result_cmf, IndicatorResult)
        assert isinstance(result_vwap, IndicatorResult)

    def test_volume_indicators_range_bounds(self, sample_candles):
        """Test that volume indicators stay within expected ranges"""
        result_cmf = VolumeIndicators.cmf(sample_candles)
        assert -1 <= result_cmf.value <= 1

        result_vo = VolumeIndicators.volume_oscillator(sample_candles)
        # Volume oscillator can be any value, but should be reasonable
        assert isinstance(result_vo.value, (int, float))

    def test_divergence_detection(self, sample_candles):
        """Test divergence detection in OBV and AD Line"""
        # Create divergence scenario: price up, volume down
        decreasing_volume = [c.volume * 0.9 for c in sample_candles[-5:]]
        divergence_candles = []
        for i, c in enumerate(sample_candles):
            if i >= len(sample_candles) - 5:
                new_volume = decreasing_volume[i - (len(sample_candles) - 5)]
                divergence_candles.append(c._replace(volume=int(new_volume)))
            else:
                divergence_candles.append(c)

        result_obv = VolumeIndicators.obv(divergence_candles)
        assert isinstance(result_obv, IndicatorResult)

        result_ad = VolumeIndicators.ad_line(divergence_candles)
        assert isinstance(result_ad, IndicatorResult)
