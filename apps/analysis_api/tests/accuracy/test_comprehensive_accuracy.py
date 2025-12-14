"""
Comprehensive test showing how accuracy affects decision making

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


def create_indicator(name: str, category: IndicatorCategory, 
                    signal: SignalStrength, confidence: float) -> IndicatorResult:
    return IndicatorResult(
        indicator_name=name,
        category=category,
        signal=signal,
        value=0.0,
        confidence=confidence,
        timestamp=datetime.now(timezone.utc)
    )


@pytest.mark.parametrize("title,indicators", [
    ("Ideal conditions - Strong and confident signal", {
        'trend': [
            create_indicator("SMA", IndicatorCategory.TREND, SignalStrength.BULLISH, 0.9),
            create_indicator("EMA", IndicatorCategory.TREND, SignalStrength.BULLISH, 0.9),
        ],
        'momentum': [
            create_indicator("RSI", IndicatorCategory.MOMENTUM, SignalStrength.BULLISH, 0.9),
            create_indicator("Stoch", IndicatorCategory.MOMENTUM, SignalStrength.BULLISH, 0.9),
        ],
        'cycle': [
            create_indicator("Sine", IndicatorCategory.CYCLE, SignalStrength.BULLISH, 0.9),
        ],
        'volume': [
            create_indicator("OBV", IndicatorCategory.VOLUME, SignalStrength.BULLISH, 0.9),
        ]
    }),
    ("Uncertainty - Accurate indicators with different signals", {
        'trend': [
            create_indicator("SMA", IndicatorCategory.TREND, SignalStrength.BULLISH, 0.9),
            create_indicator("EMA", IndicatorCategory.TREND, SignalStrength.BEARISH, 0.9),
        ],
        'momentum': [
            create_indicator("RSI", IndicatorCategory.MOMENTUM, SignalStrength.BULLISH, 0.9),
            create_indicator("Stoch", IndicatorCategory.MOMENTUM, SignalStrength.BEARISH, 0.9),
        ],
        'cycle': [
            create_indicator("Sine", IndicatorCategory.CYCLE, SignalStrength.NEUTRAL, 0.9),
        ],
        'volume': [
            create_indicator("OBV", IndicatorCategory.VOLUME, SignalStrength.NEUTRAL, 0.9),
        ]
    }),
    ("Rely on trend - Other indicators uncertain", {
        'trend': [
            create_indicator("SMA", IndicatorCategory.TREND, SignalStrength.BULLISH, 0.95),
            create_indicator("EMA", IndicatorCategory.TREND, SignalStrength.BULLISH, 0.95),
            create_indicator("MACD", IndicatorCategory.TREND, SignalStrength.BULLISH, 0.95),
        ],
        'momentum': [
            create_indicator("RSI", IndicatorCategory.MOMENTUM, SignalStrength.NEUTRAL, 0.3),
            create_indicator("Stoch", IndicatorCategory.MOMENTUM, SignalStrength.BEARISH, 0.2),
        ],
        'cycle': [
            create_indicator("Sine", IndicatorCategory.CYCLE, SignalStrength.NEUTRAL, 0.3),
        ],
        'volume': [
            create_indicator("OBV", IndicatorCategory.VOLUME, SignalStrength.NEUTRAL, 0.4),
        ]
    }),
    ("Possible trend change - Momentum and cycle giving change signals", {
        'trend': [
            create_indicator("SMA", IndicatorCategory.TREND, SignalStrength.BEARISH, 0.4),
            create_indicator("EMA", IndicatorCategory.TREND, SignalStrength.BEARISH, 0.3),
        ],
        'momentum': [
            create_indicator("RSI", IndicatorCategory.MOMENTUM, SignalStrength.BULLISH, 0.95),
            create_indicator("Stoch", IndicatorCategory.MOMENTUM, SignalStrength.BULLISH, 0.9),
        ],
        'cycle': [
            create_indicator("Sine", IndicatorCategory.CYCLE, SignalStrength.BULLISH, 0.9),
            create_indicator("Phase", IndicatorCategory.CYCLE, SignalStrength.BULLISH, 0.95),
        ],
        'volume': [
            create_indicator("OBV", IndicatorCategory.VOLUME, SignalStrength.BULLISH, 0.8),
        ]
    }),
    ("Uncertain market - All indicators uncertain", {
        'trend': [
            create_indicator("SMA", IndicatorCategory.TREND, SignalStrength.BULLISH, 0.3),
            create_indicator("EMA", IndicatorCategory.TREND, SignalStrength.BULLISH, 0.3),
        ],
        'momentum': [
            create_indicator("RSI", IndicatorCategory.MOMENTUM, SignalStrength.BULLISH, 0.3),
            create_indicator("Stoch", IndicatorCategory.MOMENTUM, SignalStrength.BULLISH, 0.3),
        ],
        'cycle': [
            create_indicator("Sine", IndicatorCategory.CYCLE, SignalStrength.BULLISH, 0.3),
        ],
        'volume': [
            create_indicator("OBV", IndicatorCategory.VOLUME, SignalStrength.BULLISH, 0.3),
        ]
    }),
])
def test_scenario(title: str, indicators: dict):
    """Test a specific scenario"""
    print(f"\n{'='*70}")
    print(f"📊 {title}")
    print(f"{'='*70}")
    
    analysis = TechnicalAnalysisResult(
        symbol="BTCUSDT",
        timeframe="1h",
        trend_indicators=indicators['trend'],
        momentum_indicators=indicators['momentum'],
        cycle_indicators=indicators['cycle'],
        volume_indicators=indicators['volume']
    )
    
    # Calculate signals
    compute_overall_signals(analysis)
    
    # Calculate accuracies
    def get_avg_confidence(inds):
        if not inds:
            return 0.0
        return sum(i.confidence for i in inds) / len(inds)
    
    trend_acc = get_avg_confidence(indicators['trend'])
    momentum_acc = get_avg_confidence(indicators['momentum'])
    cycle_acc = get_avg_confidence(indicators['cycle'])
    volume_acc = get_avg_confidence(indicators['volume'])
    
    print(f"\nIndicators:")
    print(f"  Trend:    {len(indicators['trend'])} indicators, avg accuracy: {trend_acc:.2f}")
    print(f"  Momentum:  {len(indicators['momentum'])} indicators, avg accuracy: {momentum_acc:.2f}")
    print(f"  Cycle:    {len(indicators['cycle'])} indicators, avg accuracy: {cycle_acc:.2f}")
    print(f"  Volume:     {len(indicators['volume'])} indicators, avg accuracy: {volume_acc:.2f}")
    
    print(f"\nCategory signals:")
    print(f"  Trend:    {analysis.overall_trend_signal.value}")
    print(f"  Momentum:  {analysis.overall_momentum_signal.value}")
    print(f"  Cycle:    {analysis.overall_cycle_signal.value}")
    
    print(f"\n🎯 Final result:")
    print(f"  Overall signal: {analysis.overall_signal.value}")
    print(f"  Overall confidence: {analysis.overall_confidence:.1%}")
    
    return analysis

