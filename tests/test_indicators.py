"""
Test Suite for Technical Analysis Service

Comprehensive tests for technical indicators and analysis services.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import pytest
from src.core.domain.entities import Candle
from datetime import datetime, timedelta


@pytest.fixture
def sample_candles():
    """Create sample candle data for testing"""
    candles = []
    base_price = 40000
    base_time = datetime.now() - timedelta(days=100)
    
    for i in range(100):
        # Simulate some price movement
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
    
    return candles


def test_candle_properties(sample_candles):
    """Test candle property calculations"""
    candle = sample_candles[50]
    
    assert candle.typical_price == (candle.high + candle.low + candle.close) / 3
    assert isinstance(candle.is_bullish, bool)
    assert isinstance(candle.is_bearish, bool)
    assert candle.body_size >= 0
    assert candle.upper_shadow >= 0
    assert candle.lower_shadow >= 0


def test_trend_indicators(sample_candles):
    """Test trend indicator calculations"""
    from indicators.trend import TrendIndicators
    
    # Test SMA
    sma_result = TrendIndicators.sma(sample_candles, 20)
    assert sma_result.indicator_name == "SMA(20)"
    assert sma_result.value > 0
    assert 0 <= sma_result.confidence <= 1
    
    # Test EMA
    ema_result = TrendIndicators.ema(sample_candles, 20)
    assert ema_result.indicator_name == "EMA(20)"
    assert ema_result.value > 0
    
    # Test MACD
    macd_result = TrendIndicators.macd(sample_candles)
    assert macd_result.indicator_name == "MACD"
    assert "signal" in macd_result.additional_values
    assert "histogram" in macd_result.additional_values


def test_momentum_indicators(sample_candles):
    """Test momentum indicator calculations"""
    from indicators.momentum import MomentumIndicators
    
    # Test RSI
    rsi_result = MomentumIndicators.rsi(sample_candles, 14)
    assert rsi_result.indicator_name == "RSI(14)"
    assert 0 <= rsi_result.value <= 100
    
    # Test Stochastic
    stoch_result = MomentumIndicators.stochastic(sample_candles, 14, 3)
    assert stoch_result.indicator_name == "Stochastic(14,3)"
    assert 0 <= stoch_result.value <= 100


def test_cycle_indicators(sample_candles):
    """Test cycle indicator calculations"""
    from indicators.cycle import CycleIndicators
    
    # Test Sine Wave
    sine_result = CycleIndicators.sine_wave(sample_candles, 20)
    assert sine_result.indicator_name == "Sine Wave(20)"
    assert -1 <= sine_result.value <= 1
    
    # Test DPO
    dpo_result = CycleIndicators.detrended_price_oscillator(sample_candles, 20)
    assert dpo_result.indicator_name == "DPO(20)"
    
    # Test STC
    if len(sample_candles) >= 50:
        stc_result = CycleIndicators.schaff_trend_cycle(sample_candles, 23, 50, 10)
        assert stc_result.indicator_name == "STC(23,50,10)"
        assert 0 <= stc_result.value <= 100


def test_volume_indicators(sample_candles):
    """Test volume indicator calculations"""
    from indicators.volume import VolumeIndicators
    
    # Test OBV
    obv_result = VolumeIndicators.obv(sample_candles)
    assert obv_result.indicator_name == "OBV"
    
    # Test VWAP
    vwap_result = VolumeIndicators.vwap(sample_candles)
    assert vwap_result.indicator_name == "VWAP"
    assert vwap_result.value > 0


def test_volatility_indicators(sample_candles):
    """Test volatility indicator calculations"""
    from indicators.volatility import VolatilityIndicators
    
    # Test Bollinger Bands
    bb_result = VolatilityIndicators.bollinger_bands(sample_candles, 20, 2.0)
    assert bb_result.indicator_name == "Bollinger Bands(20,2.0)"
    assert "upper" in bb_result.additional_values
    assert "lower" in bb_result.additional_values
    
    # Test ATR
    atr_result = VolatilityIndicators.atr(sample_candles, 14)
    assert atr_result.indicator_name == "ATR(14)"
    assert atr_result.value > 0


def test_support_resistance(sample_candles):
    """Test support/resistance indicators"""
    from indicators.support_resistance import SupportResistanceIndicators
    
    # Test Pivot Points
    pivot_result = SupportResistanceIndicators.pivot_points(sample_candles)
    assert pivot_result.indicator_name == "Pivot Points"
    assert "R1" in pivot_result.additional_values
    assert "S1" in pivot_result.additional_values


def test_candlestick_patterns(sample_candles):
    """Test candlestick pattern detection"""
    from patterns.candlestick import CandlestickPatterns
    
    patterns = CandlestickPatterns.detect_patterns(sample_candles)
    assert isinstance(patterns, list)


def test_elliott_wave(sample_candles):
    """Test Elliott Wave analysis"""
    from patterns.elliott_wave import analyze_elliott_waves
    
    result = analyze_elliott_waves(sample_candles)
    # Result may be None if no pattern detected
    if result:
        assert result.wave_pattern in ["IMPULSIVE", "CORRECTIVE"]
        assert 1 <= result.current_wave <= 5
        assert len(result.waves) >= 4
        assert result.confidence > 0


@pytest.mark.asyncio
async def test_complete_analysis(sample_candles):
    """Test complete analysis"""
    from services.analysis_service import TechnicalAnalysisService
    from models.schemas import AnalysisRequest
    
    request = AnalysisRequest(
        symbol="BTCUSDT",
        timeframe="1h",
        candles=sample_candles
    )
    
    result = await TechnicalAnalysisService.analyze(request)
    
    assert result.symbol == "BTCUSDT"
    assert result.timeframe == "1h"
    assert len(result.trend_indicators) > 0
    assert len(result.momentum_indicators) > 0
    assert len(result.cycle_indicators) > 0
    assert len(result.volume_indicators) > 0
    assert result.overall_signal is not None
    assert result.overall_confidence is not None
    # Elliott wave may or may not be detected
    assert result.elliott_wave_analysis is None or result.elliott_wave_analysis.wave_pattern in ["IMPULSIVE", "CORRECTIVE"]
