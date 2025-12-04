"""
Unit Tests for New Trend Indicators (v1.1.0)

Tests for 4 new trend indicators:
1. Donchian Channels
2. Aroon Indicator
3. Vortex Indicator (VI)
4. McGinley Dynamic

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
Created: November 7, 2025
"""

import pytest
import numpy as np
from gravity_tech.core.domain.entities import Candle, CoreSignalStrength as SignalStrength
from gravity_tech.core.indicators.trend import TrendIndicators


@pytest.fixture
def sample_candles_uptrend():
    """Generate uptrend sample data"""
    candles = []
    base_price = 100.0
    for i in range(50):
        price = base_price + i * 0.5  # Steady uptrend
        candles.append(Candle(
            open=price - 0.2,
            high=price + 0.3,
            low=price - 0.3,
            close=price,
            volume=1000000 + i * 1000,
            timestamp=1699920000 + i * 300
        ))
    return candles


@pytest.fixture
def sample_candles_downtrend():
    """Generate downtrend sample data"""
    candles = []
    base_price = 150.0
    for i in range(50):
        price = base_price - i * 0.5  # Steady downtrend
        candles.append(Candle(
            open=price + 0.2,
            high=price + 0.3,
            low=price - 0.3,
            close=price,
            volume=1000000 + i * 1000,
            timestamp=1699920000 + i * 300
        ))
    return candles


@pytest.fixture
def sample_candles_sideways():
    """Generate sideways/ranging market data"""
    candles = []
    base_price = 100.0
    for i in range(50):
        # Oscillate around base price
        price = base_price + np.sin(i * 0.3) * 2
        candles.append(Candle(
            open=price - 0.1,
            high=price + 0.2,
            low=price - 0.2,
            close=price,
            volume=1000000,
            timestamp=1699920000 + i * 300
        ))
    return candles


class TestDonchianChannels:
    """Test Donchian Channels indicator"""
    
    def test_donchian_uptrend_breakout(self, sample_candles_uptrend):
        """Test Donchian Channels during uptrend (should show bullish signal)"""
        result = TrendIndicators.donchian_channels(sample_candles_uptrend, period=20)
        
        assert result.indicator_name == "Donchian Channels(20)"
        assert result.signal in [SignalStrength.BULLISH, SignalStrength.VERY_BULLISH]
        assert result.confidence > 0.6
        assert "upper_band" in result.additional_values
        assert "lower_band" in result.additional_values
        assert "channel_width_pct" in result.additional_values
        assert result.additional_values["upper_band"] > result.additional_values["lower_band"]
    
    def test_donchian_downtrend_breakout(self, sample_candles_downtrend):
        """Test Donchian Channels during downtrend (should show bearish signal)"""
        result = TrendIndicators.donchian_channels(sample_candles_downtrend, period=20)
        
        assert result.signal in [SignalStrength.BEARISH, SignalStrength.VERY_BEARISH]
        assert result.confidence > 0.6
    
    def test_donchian_sideways(self, sample_candles_sideways):
        """Test Donchian Channels in sideways market"""
        result = TrendIndicators.donchian_channels(sample_candles_sideways, period=20)
        
        # In sideways market, signals can vary but price should be within channel
        assert result.signal in [SignalStrength.NEUTRAL, SignalStrength.BULLISH_BROKEN, 
                                  SignalStrength.BEARISH_BROKEN, SignalStrength.BULLISH, 
                                  SignalStrength.BEARISH]
        # Price position should indicate it's within the channel
        assert 0 < result.additional_values["price_position_pct"] < 100
    
    def test_donchian_insufficient_data(self):
        """Test Donchian with insufficient candles"""
        # Candle(open, high, low, close, volume, timestamp)
        candles = [Candle(100, 101, 99, 100, 1000000, 1699920000 + i*300) for i in range(10)]
        
        with pytest.raises(ValueError, match="Need at least 20 candles"):
            TrendIndicators.donchian_channels(candles, period=20)
    
    def test_donchian_custom_period(self, sample_candles_uptrend):
        """Test Donchian with custom period"""
        result = TrendIndicators.donchian_channels(sample_candles_uptrend, period=10)
        
        assert result.indicator_name == "Donchian Channels(10)"
        assert result.confidence > 0.5


class TestAroon:
    """Test Aroon Indicator"""
    
    def test_aroon_strong_uptrend(self, sample_candles_uptrend):
        """Test Aroon during strong uptrend"""
        result = TrendIndicators.aroon(sample_candles_uptrend, period=25)
        
        assert result.indicator_name == "Aroon(25)"
        assert result.signal in [SignalStrength.BULLISH, SignalStrength.VERY_BULLISH]
        assert result.additional_values["aroon_up"] > result.additional_values["aroon_down"]
        assert result.additional_values["aroon_oscillator"] > 0
        assert result.confidence > 0.6
    
    def test_aroon_strong_downtrend(self, sample_candles_downtrend):
        """Test Aroon during strong downtrend"""
        result = TrendIndicators.aroon(sample_candles_downtrend, period=25)
        
        assert result.signal in [SignalStrength.BEARISH, SignalStrength.VERY_BEARISH]
        assert result.additional_values["aroon_down"] > result.additional_values["aroon_up"]
        assert result.additional_values["aroon_oscillator"] < 0
    
    def test_aroon_sideways(self, sample_candles_sideways):
        """Test Aroon in sideways market"""
        result = TrendIndicators.aroon(sample_candles_sideways, period=25)
        
        # In sideways, Aroon values can vary but oscillator should not be extreme
        assert -80 <= result.additional_values["aroon_oscillator"] <= 80
    
    def test_aroon_periods_since_extreme(self, sample_candles_uptrend):
        """Test that Aroon correctly tracks periods since highs/lows"""
        result = TrendIndicators.aroon(sample_candles_uptrend, period=25)
        
        # In uptrend, recent high should be very recent
        assert result.additional_values["periods_since_high"] < 5
        # Recent low should be much earlier
        assert result.additional_values["periods_since_low"] > 15
    
    def test_aroon_insufficient_data(self):
        """Test Aroon with insufficient candles"""
        candles = [Candle(100, 101, 99, 100, 1000000, 1699920000 + i*300) for i in range(15)]
        
        with pytest.raises(ValueError, match="Need at least 25 candles"):
            TrendIndicators.aroon(candles, period=25)


class TestVortexIndicator:
    """Test Vortex Indicator (VI)"""
    
    def test_vortex_uptrend(self, sample_candles_uptrend):
        """Test Vortex during uptrend (VI+ > VI-)"""
        result = TrendIndicators.vortex_indicator(sample_candles_uptrend, period=14)
        
        assert result.indicator_name == "Vortex Indicator(14)"
        assert result.signal in [SignalStrength.BULLISH, SignalStrength.VERY_BULLISH, SignalStrength.BULLISH_BROKEN]
        assert result.additional_values["vi_plus"] > result.additional_values["vi_minus"]
        assert result.additional_values["vi_difference"] > 0
        assert result.confidence > 0.5
    
    def test_vortex_downtrend(self, sample_candles_downtrend):
        """Test Vortex during downtrend (VI- > VI+)"""
        result = TrendIndicators.vortex_indicator(sample_candles_downtrend, period=14)
        
        assert result.signal in [SignalStrength.BEARISH, SignalStrength.VERY_BEARISH, SignalStrength.BEARISH_BROKEN]
        assert result.additional_values["vi_minus"] > result.additional_values["vi_plus"]
        assert result.additional_values["vi_difference"] < 0
    
    def test_vortex_values_positive(self, sample_candles_uptrend):
        """Test that VI+ and VI- are positive"""
        result = TrendIndicators.vortex_indicator(sample_candles_uptrend, period=14)
        
        assert result.additional_values["vi_plus"] > 0
        assert result.additional_values["vi_minus"] > 0
    
    def test_vortex_strong_trend_confidence(self, sample_candles_uptrend):
        """Test that strong trends have higher confidence"""
        result = TrendIndicators.vortex_indicator(sample_candles_uptrend, period=14)
        
        # Strong divergence should increase confidence
        if abs(result.additional_values["vi_difference"]) > 0.2:
            assert result.confidence > 0.7
    
    def test_vortex_insufficient_data(self):
        """Test Vortex with insufficient candles"""
        candles = [Candle(100, 101, 99, 100, 1000000, 1699920000 + i*300) for i in range(10)]
        
        with pytest.raises(ValueError, match="Need at least 15 candles"):
            TrendIndicators.vortex_indicator(candles, period=14)


class TestMcGinleyDynamic:
    """Test McGinley Dynamic indicator"""
    
    def test_mcginley_uptrend(self, sample_candles_uptrend):
        """Test McGinley Dynamic during uptrend"""
        result = TrendIndicators.mcginley_dynamic(sample_candles_uptrend, period=20)
        
        assert result.indicator_name == "McGinley Dynamic(20)"
        assert result.signal in [SignalStrength.BULLISH, SignalStrength.VERY_BULLISH, SignalStrength.BULLISH_BROKEN]
        assert result.additional_values["current_price"] > result.additional_values["md_value"]
        assert result.additional_values["deviation_pct"] > 0
        assert result.confidence > 0.6
    
    def test_mcginley_downtrend(self, sample_candles_downtrend):
        """Test McGinley Dynamic during downtrend"""
        result = TrendIndicators.mcginley_dynamic(sample_candles_downtrend, period=20)
        
        assert result.signal in [SignalStrength.BEARISH, SignalStrength.VERY_BEARISH, SignalStrength.BEARISH_BROKEN]
        assert result.additional_values["current_price"] < result.additional_values["md_value"]
        assert result.additional_values["deviation_pct"] < 0
    
    def test_mcginley_adaptive_nature(self, sample_candles_uptrend):
        """Test that McGinley adapts to price movements"""
        result = TrendIndicators.mcginley_dynamic(sample_candles_uptrend, period=20)
        
        # MD should be less than price in uptrend but following it
        md_value = result.additional_values["md_value"]
        current_price = result.additional_values["current_price"]
        
        assert md_value > 0
        assert md_value < current_price  # Should lag price in uptrend
        assert abs(current_price - md_value) / current_price < 0.1  # But not too far
    
    def test_mcginley_slope_calculation(self, sample_candles_uptrend):
        """Test slope calculation in McGinley"""
        result = TrendIndicators.mcginley_dynamic(sample_candles_uptrend, period=20)
        
        # In uptrend, slope should be positive
        assert result.additional_values["slope_pct"] > 0
    
    def test_mcginley_custom_k_factor(self, sample_candles_uptrend):
        """Test McGinley with custom k_factor"""
        result1 = TrendIndicators.mcginley_dynamic(sample_candles_uptrend, period=20, k_factor=0.6)
        result2 = TrendIndicators.mcginley_dynamic(sample_candles_uptrend, period=20, k_factor=1.2)
        
        # Different k_factors should produce different MD values
        # Higher k_factor = slower adaptation
        assert result1.additional_values["md_value"] != result2.additional_values["md_value"]
    
    def test_mcginley_insufficient_data(self):
        """Test McGinley with insufficient candles"""
        candles = [Candle(100, 101, 99, 100, 1000000, 1699920000 + i*300) for i in range(10)]
        
        with pytest.raises(ValueError, match="Need at least 20 candles"):
            TrendIndicators.mcginley_dynamic(candles, period=20)
        
        with pytest.raises(ValueError, match="Need at least 20 candles"):
            TrendIndicators.mcginley_dynamic(candles, period=20)


class TestNewIndicatorsIntegration:
    """Integration tests for all new indicators"""
    
    def test_all_new_indicators_calculate_all(self, sample_candles_uptrend):
        """Test that calculate_all includes new indicators"""
        results = TrendIndicators.calculate_all(sample_candles_uptrend)
        
        # Should have original indicators + 4 new ones
        indicator_names = [r.indicator_name for r in results]
        
        assert any("Donchian Channels" in name for name in indicator_names)
        assert any("Aroon" in name for name in indicator_names)
        assert any("Vortex Indicator" in name for name in indicator_names)
        assert any("McGinley Dynamic" in name for name in indicator_names)
    
    def test_all_new_indicators_return_valid_results(self, sample_candles_uptrend):
        """Test that all new indicators return valid IndicatorResult"""
        indicators = [
            TrendIndicators.donchian_channels(sample_candles_uptrend, 20),
            TrendIndicators.aroon(sample_candles_uptrend, 25),
            TrendIndicators.vortex_indicator(sample_candles_uptrend, 14),
            TrendIndicators.mcginley_dynamic(sample_candles_uptrend, 20)
        ]
        
        for result in indicators:
            assert result.indicator_name is not None
            assert result.signal in SignalStrength
            assert 0 <= result.confidence <= 1
            assert result.description is not None
            assert isinstance(result.additional_values, dict)
            assert len(result.additional_values) > 0
    
    def test_performance_benchmark(self, sample_candles_uptrend):
        """Basic performance test - should be fast"""
        import time
        
        # Each indicator should calculate in < 10ms (will be optimized to <0.1ms later)
        for indicator_func, period in [
            (TrendIndicators.donchian_channels, 20),
            (TrendIndicators.aroon, 25),
            (TrendIndicators.vortex_indicator, 14),
            (TrendIndicators.mcginley_dynamic, 20)
        ]:
            start = time.time()
            result = indicator_func(sample_candles_uptrend, period)
            duration = time.time() - start
            
            assert duration < 0.01, f"{result.indicator_name} took {duration*1000:.2f}ms (should be <10ms)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

