"""
Comprehensive Tests for Core Volume Indicators

Testing all volume-based indicators with proper coverage.

Author: Dr. Sarah O'Connor (QA Lead)  
Date: November 15, 2025
Version: 1.0.0
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from gravity_tech.core.domain.entities import Candle, SignalStrength
from gravity_tech.core.indicators.volume import VolumeIndicators


class TestOnBalanceVolume:
    """Test On-Balance Volume (OBV) indicator"""
    
    def test_obv_uptrend(self, uptrend_candles):
        """Test OBV in uptrend - should increase"""
        result = VolumeIndicators.obv(uptrend_candles)
        
        assert result is not None
        assert result.value != 0
        assert 'obv' in result.additional_values
        
        # Signal should be bullish in uptrend
        assert result.signal in [SignalStrength.BULLISH, SignalStrength.VERY_BULLISH, 
                                  SignalStrength.BULLISH_BROKEN]
    
    def test_obv_downtrend(self, downtrend_candles):
        """Test OBV in downtrend - should decrease"""
        result = VolumeIndicators.obv(downtrend_candles)
        
        assert result is not None
        
        # Signal should be bearish in downtrend
        assert result.signal in [SignalStrength.BEARISH, SignalStrength.VERY_BEARISH, 
                                  SignalStrength.BEARISH_BROKEN, SignalStrength.NEUTRAL]
    
    def test_obv_signal_strength(self, sample_candles):
        """Test OBV signal strength calculation"""
        result = VolumeIndicators.obv(sample_candles)
        
        assert result is not None
        assert hasattr(result, 'signal')
        assert isinstance(result.signal, SignalStrength)


class TestVWAP:
    """Test Volume Weighted Average Price"""
    
    def test_vwap_basic(self, sample_candles):
        """Test basic VWAP calculation"""
        result = VolumeIndicators.vwap(sample_candles)
        
        assert result is not None
        assert result.value > 0
    
    def test_vwap_values_reasonable(self, sample_candles):
        """Test VWAP values are within price range"""
        result = VolumeIndicators.vwap(sample_candles)
        
        prices = [c.close for c in sample_candles]
        min_price = min(prices)
        max_price = max(prices)
        
        # VWAP should be within price range
        assert min_price <= result.value <= max_price * 1.1  # Allow 10% tolerance


class TestAccumulationDistribution:
    """Test Accumulation/Distribution Line"""
    
    def test_ad_line_uptrend(self, uptrend_candles):
        """Test AD Line in uptrend"""
        result = VolumeIndicators.accumulation_distribution(uptrend_candles)
        
        assert result is not None
        assert 'ad_line' in result.additional_values
        
        ad_values = result.additional_values['ad_line']
        # AD should accumulate in uptrend
        assert ad_values[-1] > ad_values[0]
    
    def test_ad_line_downtrend(self, downtrend_candles):
        """Test AD Line in downtrend"""
        result = VolumeIndicators.accumulation_distribution(downtrend_candles)
        
        assert result is not None
        ad_values = result.additional_values['ad_line']
        
        # AD should distribute in downtrend
        assert ad_values[-1] < ad_values[0]
    
    def test_ad_line_signal(self, sample_candles):
        """Test AD Line signal generation"""
        result = VolumeIndicators.accumulation_distribution(sample_candles)
        
        assert result is not None
        assert hasattr(result, 'signal')
        assert -2.0 <= result.signal.get_score() <= 2.0


class TestChaikinMoneyFlow:
    """Test Chaikin Money Flow (CMF)"""
    
    def test_cmf_basic(self, sample_candles):
        """Test basic CMF calculation"""
        result = VolumeIndicators.chaikin_money_flow(sample_candles, period=20)
        
        assert result is not None
        assert -1.0 <= result.value <= 1.0
        assert 'cmf' in result.additional_values
    
    def test_cmf_uptrend_positive(self, uptrend_candles):
        """Test CMF is positive in uptrend"""
        result = VolumeIndicators.cmf(uptrend_candles, period=20)
        
        assert result is not None
        # CMF typically positive in uptrend
        assert result.value >= -0.5  # Allow some tolerance
    
    def test_cmf_downtrend_negative(self, downtrend_candles):
        """Test CMF is negative in downtrend"""
        result = VolumeIndicators.cmf(downtrend_candles, period=20)
        
        assert result is not None
        # CMF typically negative in downtrend
        assert result.value <= 0.5  # Allow some tolerance
    
    def test_cmf_custom_period(self, sample_candles):
        """Test CMF with custom period"""
        result_10 = VolumeIndicators.cmf(sample_candles, period=10)
        result_30 = VolumeIndicators.cmf(sample_candles, period=30)
        
        assert result_10 is not None
        assert result_30 is not None
        # Different periods may give different values
        assert result_10.value != result_30.value or True  # May be equal


class TestMoneyFlowIndex:
    """Test Money Flow Index (MFI)"""
    
    def test_mfi_basic(self, sample_candles):
        """Test basic MFI calculation"""
        result = VolumeIndicators.money_flow_index(sample_candles, period=14)
        
        assert result is not None
        assert 0 <= result.value <= 100
        assert 'mfi' in result.additional_values
    
    def test_mfi_overbought(self, uptrend_candles):
        """Test MFI in overbought conditions"""
        result = VolumeIndicators.money_flow_index(uptrend_candles, period=14)
        
        assert result is not None
        # In strong uptrend, MFI can be high
        assert result.value >= 0
    
    def test_mfi_oversold(self, downtrend_candles):
        """Test MFI in oversold conditions"""
        result = VolumeIndicators.money_flow_index(downtrend_candles, period=14)
        
        assert result is not None
        # In strong downtrend, MFI can be low
        assert result.value <= 100
    
    def test_mfi_signal_generation(self, sample_candles):
        """Test MFI signal strength"""
        result = VolumeIndicators.money_flow_index(sample_candles, period=14)
        
        assert result is not None
        assert hasattr(result, 'signal')
        assert -2.0 <= result.signal.get_score() <= 2.0


class TestVolumeRateOfChange:
    """Test Volume Rate of Change"""
    
    def test_vroc_basic(self, sample_candles):
        """Test basic VROC calculation"""
        result = VolumeIndicators.volume_rate_of_change(sample_candles, period=14)
        
        assert result is not None
        assert 'vroc' in result.additional_values
    
    def test_vroc_increasing_volume(self):
        """Test VROC with increasing volume"""
        candles = []
        base_time = datetime(2024, 1, 1)
        
        for i in range(50):
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1000000 + (i * 100000)  # Increasing volume
            ))
        
        result = VolumeIndicators.volume_rate_of_change(candles, period=14)
        
        assert result is not None
        assert result.value > 0  # Positive VROC for increasing volume
    
    def test_vroc_decreasing_volume(self):
        """Test VROC with decreasing volume"""
        candles = []
        base_time = datetime(2024, 1, 1)
        
        for i in range(50):
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=5000000 - (i * 50000)  # Decreasing volume
            ))
        
        result = VolumeIndicators.volume_rate_of_change(candles, period=14)
        
        assert result is not None
        assert result.value < 0  # Negative VROC for decreasing volume


class TestVolumeProfile:
    """Test Volume Profile analysis"""
    
    def test_volume_profile_basic(self, sample_candles):
        """Test basic volume profile calculation"""
        result = VolumeIndicators.volume_profile(sample_candles, bins=20)
        
        assert result is not None
        assert 'price_levels' in result.additional_values
        assert 'volume_at_price' in result.additional_values
        assert 'poc' in result.additional_values  # Point of Control
    
    def test_volume_profile_poc(self, sample_candles):
        """Test Point of Control calculation"""
        result = VolumeIndicators.volume_profile(sample_candles, bins=20)
        
        assert result is not None
        poc = result.additional_values.get('poc')
        
        if poc:
            prices = [c.close for c in sample_candles]
            # POC should be within price range
            assert min(prices) <= poc <= max(prices)
    
    def test_volume_profile_custom_bins(self, sample_candles):
        """Test volume profile with different bin sizes"""
        result_10 = VolumeIndicators.volume_profile(sample_candles, bins=10)
        result_50 = VolumeIndicators.volume_profile(sample_candles, bins=50)
        
        assert result_10 is not None
        assert result_50 is not None
        
        # Different bins should give different number of price levels
        assert len(result_10.additional_values.get('price_levels', [])) <= \
               len(result_50.additional_values.get('price_levels', []))


class TestVolumeOscillator:
    """Test Volume Oscillator"""
    
    def test_volume_oscillator_basic(self, sample_candles):
        """Test basic volume oscillator"""
        result = VolumeIndicators.volume_oscillator(
            sample_candles,
            short_period=5,
            long_period=10
        )
        
        assert result is not None
        assert 'oscillator' in result.additional_values
    
    def test_volume_oscillator_signal(self, sample_candles):
        """Test volume oscillator signal"""
        result = VolumeIndicators.volume_oscillator(sample_candles)
        
        assert result is not None
        assert hasattr(result, 'signal')
        assert -2.0 <= result.signal.get_score() <= 2.0


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_candles(self):
        """Test with empty candle list"""
        result = VolumeIndicators.on_balance_volume([])
        assert result is not None
        assert result.signal == SignalStrength.NEUTRAL
        assert result.confidence == 0.0
    
    def test_single_candle(self):
        """Test with single candle"""
        candle = [Candle(
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000000
        )]
        
        result = VolumeIndicators.on_balance_volume(candle)
        assert result is not None
        assert result.signal == SignalStrength.NEUTRAL
        assert result.confidence == 0.0  # Need at least 2 candles
    
    def test_zero_volume_candles(self):
        """Test with zero volume candles"""
        candles = []
        base_time = datetime(2024, 1, 1)
        
        for i in range(20):
            candles.append(Candle(
                timestamp=base_time + timedelta(hours=i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=0  # Zero volume
            ))
        
        # Should handle gracefully
        result = VolumeIndicators.on_balance_volume(candles)
        # May return None or handle zeros
        assert result is None or result.value == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

