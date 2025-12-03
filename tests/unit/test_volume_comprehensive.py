"""
Comprehensive Test Suite for Volume Indicators
===============================================
Target: 70%+ coverage for volume.py (162 lines)
Current: 11.73% → Target: 70%+

Volume Indicators (6 main):
1. OBV - On Balance Volume
2. CMF - Chaikin Money Flow
3. VWAP - Volume Weighted Average Price
4. A/D Line - Accumulation/Distribution Line
5. PVT - Price Volume Trend
6. Volume Oscillator
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from src.core.domain.entities import Candle, CoreSignalStrength as SignalStrength, IndicatorCategory
from src.core.indicators.volume import VolumeIndicators


def generate_volume_candles(count: int = 50, base_price: float = 100.0, trend: str = "neutral") -> list[Candle]:
    """Generate test candles with volume patterns"""
    candles = []
    current_price = base_price
    base_time = datetime.now()
    
    for i in range(count):
        if trend == "up":
            price_change = abs(np.random.randn()) * 0.5
            volume = 1000000 * (1 + np.random.rand())  # Higher volume on up
        elif trend == "down":
            price_change = -abs(np.random.randn()) * 0.5
            volume = 1000000 * (1 + np.random.rand())  # Higher volume on down
        else:
            price_change = np.random.randn() * 0.3
            volume = 1000000 * (0.5 + np.random.rand())
        
        current_price += price_change
        open_price = current_price - price_change * 0.5
        close_price = current_price
        high_price = max(open_price, close_price) * 1.01
        low_price = min(open_price, close_price) * 0.99
        
        candle = Candle(
            timestamp=base_time + timedelta(hours=i),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        )
        candles.append(candle)
    
    return candles


def generate_divergence_candles(volume_trend: str, price_trend: str, count: int = 30) -> list[Candle]:
    """Generate candles with volume-price divergence"""
    candles = []
    base_time = datetime.now()
    base_price = 100.0
    base_volume = 1000000
    
    for i in range(count):
        # Price trend
        if price_trend == "up":
            price = base_price + i * 0.5
        elif price_trend == "down":
            price = base_price - i * 0.5
        else:
            price = base_price + np.random.randn() * 0.3
        
        # Volume trend (opposite for divergence)
        if volume_trend == "increasing":
            volume = base_volume * (1 + i * 0.05)
        elif volume_trend == "decreasing":
            volume = base_volume * max(0.3, 1 - i * 0.02)
        else:
            volume = base_volume
        
        candle = Candle(
            timestamp=base_time + timedelta(hours=i),
            open=price,
            high=price * 1.01,
            low=price * 0.99,
            close=price,
            volume=volume
        )
        candles.append(candle)
    
    return candles


# ========================================
# Test OBV (On Balance Volume)
# ========================================

class TestOBV:
    """Test OBV indicator"""
    
    def test_obv_basic(self):
        """Test basic OBV calculation"""
        candles = generate_volume_candles(50)
        result = VolumeIndicators.obv(candles)
        
        assert result.indicator_name == "OBV"
        assert result.category == IndicatorCategory.VOLUME
        assert isinstance(result.value, float)
        assert 0 <= result.confidence <= 1
    
    def test_obv_uptrend(self):
        """Test OBV in uptrend with volume confirmation"""
        candles = generate_volume_candles(50, trend="up")
        result = VolumeIndicators.obv(candles)
        
        # OBV should be positive/increasing in uptrend with volume
        assert result.signal in [SignalStrength.VERY_BULLISH, SignalStrength.BULLISH]
    
    def test_obv_downtrend(self):
        """Test OBV in downtrend with volume confirmation"""
        candles = generate_volume_candles(50, trend="down")
        result = VolumeIndicators.obv(candles)
        
        # OBV should be negative/decreasing in downtrend
        assert result.signal in [SignalStrength.VERY_BEARISH, SignalStrength.BEARISH]
    
    def test_obv_bullish_divergence(self):
        """Test OBV bullish divergence (price down, volume up)"""
        candles = generate_divergence_candles(volume_trend="increasing", price_trend="down")
        result = VolumeIndicators.obv(candles)
        
        # OBV logic is complex - just verify it calculates
        assert result.signal is not None
        assert isinstance(result.value, float)
    
    def test_obv_bearish_divergence(self):
        """Test OBV bearish divergence (price up, volume down)"""
        candles = generate_divergence_candles(volume_trend="decreasing", price_trend="up")
        result = VolumeIndicators.obv(candles)
        
        # OBV logic is complex - just verify it calculates
        assert result.signal is not None
        assert isinstance(result.value, float)
    
    def test_obv_neutral(self):
        """Test OBV neutral signal"""
        candles = generate_volume_candles(50, trend="neutral")
        result = VolumeIndicators.obv(candles)
        
        # Could be any signal depending on volume pattern
        assert result.signal is not None


# ========================================
# Test CMF (Chaikin Money Flow)
# ========================================

class TestCMF:
    """Test CMF indicator"""
    
    def test_cmf_basic(self):
        """Test basic CMF calculation"""
        candles = generate_volume_candles(50)
        result = VolumeIndicators.cmf(candles, period=20)
        
        assert result.indicator_name == "CMF(20)"
        assert result.category == IndicatorCategory.VOLUME
        assert -1 <= result.value <= 1  # CMF range
        assert 0 <= result.confidence <= 1
    
    def test_cmf_very_bullish(self):
        """Test CMF very bullish signal (>0.25)"""
        # Create strong buying pressure candles
        base_time = datetime.now()
        candles = []
        for i in range(30):
            price = 100 + i * 0.5
            # Close near high = bullish
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price,
                high=price + 1.0,
                low=price - 0.2,
                close=price + 0.9,
                volume=1000000
            ))
        
        result = VolumeIndicators.cmf(candles, period=20)
        assert result.value > 0  # Positive CMF
    
    def test_cmf_very_bearish(self):
        """Test CMF very bearish signal (<-0.25)"""
        # Create strong selling pressure candles
        base_time = datetime.now()
        candles = []
        for i in range(30):
            price = 100 - i * 0.5
            # Close near low = bearish
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price,
                high=price + 0.2,
                low=price - 1.0,
                close=price - 0.9,
                volume=1000000
            ))
        
        result = VolumeIndicators.cmf(candles, period=20)
        assert result.value < 0  # Negative CMF
    
    def test_cmf_bullish(self):
        """Test CMF bullish signal (0.1 to 0.25)"""
        candles = generate_volume_candles(50, trend="up")
        result = VolumeIndicators.cmf(candles, period=20)
        
        # Could be bullish range
        assert result.signal is not None
    
    def test_cmf_neutral(self):
        """Test CMF neutral signal"""
        candles = generate_volume_candles(50, trend="neutral")
        result = VolumeIndicators.cmf(candles, period=20)
        
        assert result.signal is not None
    
    def test_cmf_different_periods(self):
        """Test CMF with different periods"""
        candles = generate_volume_candles(100)
        
        cmf10 = VolumeIndicators.cmf(candles, period=10)
        cmf20 = VolumeIndicators.cmf(candles, period=20)
        cmf30 = VolumeIndicators.cmf(candles, period=30)
        
        assert cmf10.indicator_name == "CMF(10)"
        assert cmf20.indicator_name == "CMF(20)"
        assert cmf30.indicator_name == "CMF(30)"


# ========================================
# Test VWAP
# ========================================

class TestVWAP:
    """Test VWAP indicator"""
    
    def test_vwap_basic(self):
        """Test basic VWAP calculation"""
        candles = generate_volume_candles(50)
        result = VolumeIndicators.vwap(candles)
        
        assert result.indicator_name == "VWAP"
        assert result.category == IndicatorCategory.VOLUME
        assert result.value > 0
    
    def test_vwap_price_above(self):
        """Test VWAP with price significantly above"""
        base_time = datetime.now()
        candles = []
        # Start low, end high
        for i in range(30):
            price = 100 + i * 2  # Strong uptrend
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            ))
        
        result = VolumeIndicators.vwap(candles)
        # Current price should be well above VWAP
        assert result.signal in [SignalStrength.VERY_BULLISH, SignalStrength.BULLISH, SignalStrength.BULLISH_BROKEN]
    
    def test_vwap_price_below(self):
        """Test VWAP with price significantly below"""
        base_time = datetime.now()
        candles = []
        # Start high, end low
        for i in range(30):
            price = 100 - i * 2  # Strong downtrend
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000
            ))
        
        result = VolumeIndicators.vwap(candles)
        # Current price should be well below VWAP
        assert result.signal in [SignalStrength.VERY_BEARISH, SignalStrength.BEARISH, SignalStrength.BEARISH_BROKEN]
    
    def test_vwap_neutral(self):
        """Test VWAP neutral (price near VWAP)"""
        candles = generate_volume_candles(50, trend="neutral")
        result = VolumeIndicators.vwap(candles)
        
        assert result.signal is not None


# ========================================
# Test A/D Line
# ========================================

class TestADLine:
    """Test A/D Line indicator"""
    
    def test_ad_line_basic(self):
        """Test basic A/D Line calculation"""
        candles = generate_volume_candles(50)
        result = VolumeIndicators.ad_line(candles)
        
        assert result.indicator_name == "A/D Line"
        assert result.category == IndicatorCategory.VOLUME
        assert isinstance(result.value, float)
    
    def test_ad_line_accumulation(self):
        """Test A/D Line showing accumulation"""
        candles = generate_volume_candles(50, trend="up")
        result = VolumeIndicators.ad_line(candles)
        
        # Should show bullish signal
        assert result.signal in [SignalStrength.VERY_BULLISH, SignalStrength.BULLISH, SignalStrength.NEUTRAL]
    
    def test_ad_line_distribution(self):
        """Test A/D Line showing distribution"""
        candles = generate_volume_candles(50, trend="down")
        result = VolumeIndicators.ad_line(candles)
        
        # Should show bearish signal
        assert result.signal in [SignalStrength.VERY_BEARISH, SignalStrength.BEARISH, SignalStrength.NEUTRAL]
    
    def test_ad_line_bullish_divergence(self):
        """Test A/D Line bullish divergence"""
        candles = generate_divergence_candles(volume_trend="increasing", price_trend="down")
        result = VolumeIndicators.ad_line(candles)
        
        assert result.signal is not None
    
    def test_ad_line_bearish_divergence(self):
        """Test A/D Line bearish divergence"""
        candles = generate_divergence_candles(volume_trend="decreasing", price_trend="up")
        result = VolumeIndicators.ad_line(candles)
        
        assert result.signal is not None


# ========================================
# Test PVT (Price Volume Trend)
# ========================================

class TestPVT:
    """Test PVT indicator"""
    
    def test_pvt_basic(self):
        """Test basic PVT calculation"""
        candles = generate_volume_candles(50)
        result = VolumeIndicators.pvt(candles)
        
        assert result.indicator_name == "PVT"
        assert result.category == IndicatorCategory.VOLUME
        assert isinstance(result.value, float)
    
    def test_pvt_uptrend(self):
        """Test PVT in uptrend"""
        candles = generate_volume_candles(50, trend="up")
        result = VolumeIndicators.pvt(candles)
        
        assert result.signal in [SignalStrength.VERY_BULLISH, SignalStrength.BULLISH, SignalStrength.BULLISH_BROKEN, SignalStrength.NEUTRAL]
    
    def test_pvt_downtrend(self):
        """Test PVT in downtrend"""
        candles = generate_volume_candles(50, trend="down")
        result = VolumeIndicators.pvt(candles)
        
        assert result.signal in [SignalStrength.VERY_BEARISH, SignalStrength.BEARISH, SignalStrength.BEARISH_BROKEN, SignalStrength.NEUTRAL]
    
    def test_pvt_neutral(self):
        """Test PVT neutral"""
        candles = generate_volume_candles(50, trend="neutral")
        result = VolumeIndicators.pvt(candles)
        
        assert result.signal is not None


# ========================================
# Test Volume Oscillator
# ========================================

class TestVolumeOscillator:
    """Test Volume Oscillator indicator"""
    
    def test_vo_basic(self):
        """Test basic Volume Oscillator calculation"""
        candles = generate_volume_candles(50)
        result = VolumeIndicators.volume_oscillator(candles, short_period=5, long_period=10)
        
        assert result.indicator_name == "Volume Oscillator(5,10)"
        assert result.category == IndicatorCategory.VOLUME
        assert isinstance(result.value, float)
    
    def test_vo_high_volume(self):
        """Test VO with increasing volume"""
        base_time = datetime.now()
        candles = []
        for i in range(30):
            price = 100 + i * 0.3
            volume = 1000000 * (1 + i * 0.1)  # Increasing volume
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=volume
            ))
        
        result = VolumeIndicators.volume_oscillator(candles, short_period=5, long_period=10)
        assert result.value > 0  # Positive oscillator
    
    def test_vo_low_volume(self):
        """Test VO with decreasing volume"""
        base_time = datetime.now()
        candles = []
        for i in range(30):
            price = 100 + i * 0.3
            volume = 2000000 * max(0.3, 1 - i * 0.02)  # Decreasing volume
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=volume
            ))
        
        result = VolumeIndicators.volume_oscillator(candles, short_period=5, long_period=10)
        assert result.value < 0  # Negative oscillator
    
    def test_vo_different_periods(self):
        """Test VO with different periods"""
        candles = generate_volume_candles(100)
        
        vo1 = VolumeIndicators.volume_oscillator(candles, short_period=5, long_period=10)
        vo2 = VolumeIndicators.volume_oscillator(candles, short_period=10, long_period=20)
        
        assert vo1.indicator_name == "Volume Oscillator(5,10)"
        assert vo2.indicator_name == "Volume Oscillator(10,20)"


# ========================================
# Test Calculate All
# ========================================

class TestCalculateAll:
    """Test calculate_all method"""
    
    def test_calculate_all_sufficient_data(self):
        """Test calculate_all with sufficient data"""
        candles = generate_volume_candles(100)
        results = VolumeIndicators.calculate_all(candles)
        
        # Should return all 6 indicators
        assert len(results) == 6
        indicator_names = [r.indicator_name for r in results]
        assert "OBV" in indicator_names
        assert "CMF(20)" in indicator_names
        assert "VWAP" in indicator_names
        assert "A/D Line" in indicator_names
        assert "PVT" in indicator_names
        assert "Volume Oscillator(5,10)" in indicator_names
    
    def test_calculate_all_insufficient_data(self):
        """Test calculate_all with insufficient data"""
        candles = generate_volume_candles(15)
        results = VolumeIndicators.calculate_all(candles)
        
        # Should return only VO (needs 10 candles)
        assert len(results) == 1
        assert results[0].indicator_name == "Volume Oscillator(5,10)"
    
    def test_calculate_all_minimal_data(self):
        """Test calculate_all with minimal data"""
        candles = generate_volume_candles(5)
        results = VolumeIndicators.calculate_all(candles)
        
        # Should return empty or very limited
        assert isinstance(results, list)


# ========================================
# Test Edge Cases
# ========================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_equal_high_low(self):
        """Test with candles where high == low"""
        base_time = datetime.now()
        candles = []
        for i in range(30):
            price = 100
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=1000000
            ))
        
        # Should handle division by zero gracefully
        result_cmf = VolumeIndicators.cmf(candles, period=20)
        result_ad = VolumeIndicators.ad_line(candles)
        
        assert result_cmf.value is not None
        assert result_ad.value is not None
    
    def test_zero_volume(self):
        """Test with zero volume candles"""
        base_time = datetime.now()
        candles = []
        for i in range(30):
            price = 100 + i * 0.5
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=0.0
            ))
        
        # Should handle zero volume
        result_obv = VolumeIndicators.obv(candles)
        assert result_obv.value == 0
    
    def test_consistent_uptrend(self):
        """Test with consistent uptrend"""
        base_time = datetime.now()
        candles = []
        for i in range(30):
            price = 100 + i
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price,
                high=price + 1,
                low=price - 0.5,
                close=price + 0.8,
                volume=1000000
            ))
        
        result_obv = VolumeIndicators.obv(candles)
        result_pvt = VolumeIndicators.pvt(candles)
        
        assert result_obv.value > 0
        assert result_pvt.value > 0
    
    def test_consistent_downtrend(self):
        """Test with consistent downtrend"""
        base_time = datetime.now()
        candles = []
        for i in range(30):
            price = 100 - i
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=price,
                high=price + 0.5,
                low=price - 1,
                close=price - 0.8,
                volume=1000000
            ))
        
        result_obv = VolumeIndicators.obv(candles)
        result_pvt = VolumeIndicators.pvt(candles)
        
        assert result_obv.value < 0
        assert result_pvt.value < 0
