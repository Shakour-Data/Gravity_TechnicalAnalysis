"""
Test Suite for Technical Analysis Service

Comprehensive tests for technical indicators and analysis services.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gravity_tech.core.domain.entities import Candle
from src.gravity_tech.indicators.trend import TrendIndicators
from src.gravity_tech.indicators.momentum import MomentumIndicators
from src.gravity_tech.indicators.cycle import CycleIndicators
from src.gravity_tech.indicators.volume import VolumeIndicators
from src.gravity_tech.indicators.volatility import VolatilityIndicators
from src.gravity_tech.indicators.support_resistance import SupportResistanceIndicators
from src.gravity_tech.patterns.candlestick import CandlestickPatterns
from src.gravity_tech.patterns.elliott_wave import analyze_elliott_waves
from src.gravity_tech.services.analysis_service import TechnicalAnalysisService
from src.gravity_tech.models.schemas import AnalysisRequest


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
    # Test SMA
    sma_result = TrendIndicators.sma(sample_candles, 20)
    assert sma_result.indicator_name == "SMA(20)"
    assert sma_result.value is not None
    assert 0 <= sma_result.confidence <= 1
    
    # Test EMA
    ema_result = TrendIndicators.ema(sample_candles, 20)
    assert ema_result.indicator_name == "EMA(20)"
    assert ema_result.value is not None
    
    # Test MACD
    macd_result = TrendIndicators.macd(sample_candles)
    assert macd_result.indicator_name == "MACD"
    assert macd_result.value is not None


def test_momentum_indicators(sample_candles):
    """Test momentum indicator calculations"""
    # Test RSI
    rsi_result = MomentumIndicators.rsi(sample_candles, 14)
    assert rsi_result.indicator_name == "RSI(14)"
    assert rsi_result.value is not None
    assert 0 <= rsi_result.confidence <= 1
    
    # Test CCI
    cci_result = MomentumIndicators.cci(sample_candles, 20)
    assert cci_result.indicator_name == "CCI(20)"
    assert cci_result.value is not None


def test_cycle_indicators(sample_candles):
    """Test cycle indicator calculations"""
    # Test Hilbert Transform
    ht_result = CycleIndicators.hilbert_transform_phase(sample_candles)
    assert ht_result.indicator_name == "Hilbert Transform Phase(7)"
    assert ht_result.value is not None
    assert 0 <= ht_result.confidence <= 1
    
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
    # Test OBV
    obv_result = VolumeIndicators.obv(sample_candles)
    assert obv_result.indicator_name == "OBV"
    assert obv_result.value is not None
    
    # Test VWAP
    vwap_result = VolumeIndicators.vwap(sample_candles)
    assert vwap_result.indicator_name == "VWAP"
    assert vwap_result.value is not None


def test_volatility_indicators(sample_candles):
    """Test volatility indicator calculations"""
    # Test ATR
    atr_result = VolatilityIndicators.atr(sample_candles, 14)
    assert atr_result.indicator_name == "ATR(14)"
    assert atr_result.value is not None
    assert 0 <= atr_result.confidence <= 1
    
    # Test Standard Deviation
    std_result = VolatilityIndicators.standard_deviation(sample_candles, 20)
    assert hasattr(std_result, 'signal')  # VolatilityResult has different structure
    assert std_result.value is not None


def test_support_resistance(sample_candles):
    """Test support/resistance indicators"""
    # Test Pivot Points
    pivot_result = SupportResistanceIndicators.pivot_points(sample_candles)
    assert pivot_result.indicator_name == "Pivot Points"
    assert pivot_result.value is not None
    assert 0 <= pivot_result.confidence <= 1


def test_candlestick_patterns(sample_candles):
    """Test candlestick pattern detection"""
    # Test Doji detection
    result = CandlestickPatterns.is_doji(sample_candles[-1])
    assert isinstance(result, bool)
    
    # Test Hammer detection
    result = CandlestickPatterns.is_hammer(sample_candles[-1])
    assert isinstance(result, bool)


def test_elliott_wave(sample_candles):
    """Test Elliott Wave analysis"""
    result = analyze_elliott_waves(sample_candles)
    # Result may be None if no pattern detected
    if result:
        assert hasattr(result, 'wave_pattern')
        assert hasattr(result, 'confidence')
        assert result.confidence >= 0


def test_complete_analysis(sample_candles):
    """Test complete analysis"""
    import asyncio
    
    # Create analysis request
    request = AnalysisRequest(
        symbol="BTCUSDT",
        timeframe="1h",
        candles=sample_candles
    )
    
    # Run async analysis
    result = asyncio.run(TechnicalAnalysisService.analyze(request))
    
    # Verify result structure
    assert result.symbol == "BTCUSDT"
    assert result.timeframe == "1h"
    assert hasattr(result, 'overall_signal')
    assert 0 <= result.overall_confidence <= 1
    assert 0 <= result.overall_confidence <= 1
    assert len(result.trend_indicators) > 0
    assert len(result.momentum_indicators) > 0
    assert len(result.cycle_indicators) > 0
    assert len(result.volume_indicators) > 0
    assert result.overall_signal is not None
    assert result.overall_confidence is not None
    # Elliott wave may or may not be detected
    assert result.elliott_wave_analysis is None or result.elliott_wave_analysis.wave_pattern in ["IMPULSIVE", "CORRECTIVE"]

