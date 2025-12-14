"""
Test script to demonstrate accuracy weighting in overall signal calculation

Author: Gravity Tech Team
Date: 2024
Version: 1.0
License: MIT
"""
import pytest
from datetime import datetime, timezone
from gravity_tech.core.contracts.analysis import TechnicalAnalysisResult
from gravity_tech.core.domain.entities import (
    IndicatorCategory,
    IndicatorResult,
    CoreSignalStrength as SignalStrength,
)
from gravity_tech.services.signal_engine import compute_overall_signals
from datetime import timezone


def create_test_indicator(name: str, category: IndicatorCategory, 
                         signal: SignalStrength, confidence: float) -> IndicatorResult:
    """Helper to create test indicator"""
    return IndicatorResult(
        indicator_name=name,
        category=category,
        signal=signal,
        value=0.0,
        confidence=confidence,
        timestamp=datetime.now(timezone.utc)
    )


@pytest.mark.parametrize("name,trend_conf,momentum_conf,cycle_conf,volume_conf", [
    ("All categories have high accuracy", 0.9, 0.9, 0.9, 0.9),
    ("Trend low accuracy, rest high accuracy", 0.3, 0.9, 0.9, 0.9),
    ("Momentum low accuracy, rest high accuracy", 0.9, 0.3, 0.9, 0.9),
    ("Various accuracies", 0.8, 0.6, 0.4, 0.7),
    ("All categories have low accuracy", 0.3, 0.3, 0.3, 0.3),
])
def test_scenario(name: str, trend_conf: float, momentum_conf: float, 
                 cycle_conf: float, volume_conf: float):
    """Test a specific scenario"""
    print(f"\n{'='*60}")
    print(f"سناریو: {name}")
    print(f"{'='*60}")
    
    # Create analysis with same signals but different confidences
    analysis = TechnicalAnalysisResult(
        symbol="BTCUSDT",
        timeframe="1h",
        trend_indicators=[
            create_test_indicator("MA", IndicatorCategory.TREND, SignalStrength.BULLISH, trend_conf),
            create_test_indicator("MACD", IndicatorCategory.TREND, SignalStrength.BULLISH, trend_conf),
        ],
        momentum_indicators=[
            create_test_indicator("RSI", IndicatorCategory.MOMENTUM, SignalStrength.BULLISH, momentum_conf),
            create_test_indicator("Stoch", IndicatorCategory.MOMENTUM, SignalStrength.BULLISH, momentum_conf),
        ],
        cycle_indicators=[
            create_test_indicator("Sine", IndicatorCategory.CYCLE, SignalStrength.BULLISH, cycle_conf),
            create_test_indicator("Phase", IndicatorCategory.CYCLE, SignalStrength.BULLISH, cycle_conf),
        ],
        volume_indicators=[
            create_test_indicator("OBV", IndicatorCategory.VOLUME, SignalStrength.BULLISH, volume_conf),
        ]
    )
    
    # Calculate signals
    compute_overall_signals(analysis)
    
    # Display results
    print(f"\nConfidence (Accuracy) per category:")
    print(f"  - Trend: {trend_conf:.2f}")
    print(f"  - Momentum: {momentum_conf:.2f}")
    print(f"  - Cycle: {cycle_conf:.2f}")
    print(f"  - Volume: {volume_conf:.2f}")
    
    print(f"\nResults:")
    print(f"  - Overall signal: {analysis.overall_signal.value}")
    print(f"  - Overall confidence: {analysis.overall_confidence:.3f}")
    
    # Show how weights were adjusted
    print(f"\nAccuracy impact on weights:")
    print(f"  Categories with higher accuracy have more weight in final decision")
    """Test a specific scenario"""
    print(f"\n{'='*60}")
    print(f"سناریو: {name}")
    print(f"{'='*60}")
    
    # Create analysis with same signals but different confidences
    analysis = TechnicalAnalysisResult(
        symbol="BTCUSDT",
        timeframe="1h",
        trend_indicators=[
            create_test_indicator("MA", IndicatorCategory.TREND, SignalStrength.BULLISH, trend_conf),
            create_test_indicator("MACD", IndicatorCategory.TREND, SignalStrength.BULLISH, trend_conf),
        ],
        momentum_indicators=[
            create_test_indicator("RSI", IndicatorCategory.MOMENTUM, SignalStrength.BULLISH, momentum_conf),
            create_test_indicator("Stoch", IndicatorCategory.MOMENTUM, SignalStrength.BULLISH, momentum_conf),
        ],
        cycle_indicators=[
            create_test_indicator("Sine", IndicatorCategory.CYCLE, SignalStrength.BULLISH, cycle_conf),
            create_test_indicator("Phase", IndicatorCategory.CYCLE, SignalStrength.BULLISH, cycle_conf),
        ],
        volume_indicators=[
            create_test_indicator("OBV", IndicatorCategory.VOLUME, SignalStrength.BULLISH, volume_conf),
        ]
    )
    
    # Calculate signals
    compute_overall_signals(analysis)
    
    # Display results
    print(f"\nConfidence (Accuracy) per category:")
    print(f"  - Trend: {trend_conf:.2f}")
    print(f"  - Momentum: {momentum_conf:.2f}")
    print(f"  - Cycle: {cycle_conf:.2f}")
    print(f"  - Volume: {volume_conf:.2f}")
    
    print(f"\nResults:")
    print(f"  - Overall signal: {analysis.overall_signal.value}")
    print(f"  - Overall confidence: {analysis.overall_confidence:.3f}")
    
    # Show how weights were adjusted
    print(f"\nAccuracy impact on weights:")
    print(f"  Categories with higher accuracy have more weight in final decision")

