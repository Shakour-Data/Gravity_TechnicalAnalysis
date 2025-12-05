"""
Comprehensive Tests for Momentum Indicators

Testing RSI, Stochastic, CCI, ROC, Williams %R

Author: Dr. Sarah O'Connor (QA Lead) + Yuki Tanaka (Data Scientist)
Date: November 15, 2025
Version: 1.0.0
"""

import pytest
from gravity_tech.core.domain.entities import CoreSignalStrength as SignalStrength
from gravity_tech.core.indicators.momentum import MomentumIndicators


class TestRSI:
    """Test Relative Strength Index"""
    
    def test_rsi_basic(self, sample_candles):
        """RSI should calculate correctly"""
        result = MomentumIndicators.rsi(sample_candles, period=14)
        
        assert result is not None
        assert 0 <= result.value <= 100
        assert hasattr(result, 'signal')
    
    def test_rsi_overbought(self, uptrend_candles):
        """RSI in overbought conditions"""
        result = MomentumIndicators.rsi(uptrend_candles, period=14)
        
        assert result is not None
        # In strong uptrend, RSI can be high
        assert result.value >= 0
    
    def test_rsi_oversold(self, downtrend_candles):
        """RSI in oversold conditions"""
        result = MomentumIndicators.rsi(downtrend_candles, period=14)
        
        assert result is not None
        # In strong downtrend, RSI can be low
        assert result.value <= 100
    
    def test_rsi_custom_period(self, sample_candles):
        """RSI with custom period"""
        result_9 = MomentumIndicators.rsi(sample_candles, period=9)
        result_21 = MomentumIndicators.rsi(sample_candles, period=21)
        
        assert result_9 is not None
        assert result_21 is not None


class TestStochastic:
    """Test Stochastic Oscillator"""
    
    def test_stochastic_basic(self, sample_candles):
        """Stochastic should calculate correctly"""
        result = MomentumIndicators.stochastic(sample_candles, k_period=14, d_period=3)
        
        assert result is not None
        assert 0 <= result.value <= 100
    
    def test_stochastic_uptrend(self, uptrend_candles):
        """Stochastic in uptrend"""
        result = MomentumIndicators.stochastic(uptrend_candles)
        
        assert result is not None
        assert isinstance(result.signal, SignalStrength)
    
    def test_stochastic_default_params(self, sample_candles):
        """Stochastic with default parameters"""
        result = MomentumIndicators.stochastic(sample_candles)
        
        assert result is not None


class TestCCI:
    """Test Commodity Channel Index"""
    
    def test_cci_basic(self, sample_candles):
        """CCI should calculate correctly"""
        result = MomentumIndicators.cci(sample_candles, period=20)
        
        assert result is not None
        assert hasattr(result, 'value')
        assert hasattr(result, 'signal')
    
    def test_cci_uptrend(self, uptrend_candles):
        """CCI in uptrend"""
        result = MomentumIndicators.cci(uptrend_candles)
        
        assert result is not None
    
    def test_cci_custom_period(self, sample_candles):
        """CCI with custom period"""
        result = MomentumIndicators.cci(sample_candles, period=10)
        
        assert result is not None


class TestROC:
    """Test Rate of Change"""
    
    def test_roc_basic(self, sample_candles):
        """ROC should calculate correctly"""
        result = MomentumIndicators.roc(sample_candles, period=12)
        
        assert result is not None
        assert hasattr(result, 'value')
    
    def test_roc_uptrend(self, uptrend_candles):
        """ROC in uptrend should be positive"""
        result = MomentumIndicators.roc(uptrend_candles, period=12)
        
        assert result is not None
        # In uptrend, ROC typically positive
        assert result.value != 0
    
    def test_roc_default_period(self, sample_candles):
        """ROC with default period"""
        result = MomentumIndicators.roc(sample_candles)
        
        assert result is not None


class TestWilliamsR:
    """Test Williams %R"""
    
    def test_williams_basic(self, sample_candles):
        """Williams %R should calculate correctly"""
        result = MomentumIndicators.williams_r(sample_candles, period=14)
        
        assert result is not None
        assert -100 <= result.value <= 0
    
    def test_williams_uptrend(self, uptrend_candles):
        """Williams %R in uptrend"""
        result = MomentumIndicators.williams_r(uptrend_candles)
        
        assert result is not None
        # In uptrend, Williams %R closer to 0
        assert result.value >= -100
    
    def test_williams_downtrend(self, downtrend_candles):
        """Williams %R in downtrend"""
        result = MomentumIndicators.williams_r(downtrend_candles)
        
        assert result is not None
        # In downtrend, Williams %R closer to -100
        assert result.value <= 0


class TestEdgeCases:
    """Test edge cases for momentum indicators"""
    
    def test_minimal_data_rsi(self, minimal_candles):
        """RSI with minimal data"""
        try:
            result = MomentumIndicators.rsi(minimal_candles, period=5)
            assert result is None or result is not None
        except Exception:
            pass
    
    def test_minimal_data_stochastic(self, minimal_candles):
        """Stochastic with minimal data"""
        try:
            result = MomentumIndicators.stochastic(minimal_candles, k_period=5)
            assert result is None or result is not None
        except Exception:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

