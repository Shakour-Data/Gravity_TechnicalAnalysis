"""
Test script to demonstrate accuracy weighting in overall signal calculation

Author: Gravity Tech Team
Date: 2024
Version: 1.0
License: MIT
"""
from datetime import datetime
from src.core.domain.entities import (
    IndicatorResult,
    IndicatorCategory,
    CoreSignalStrength as SignalStrength
)
from gravity_tech.models.schemas import TechnicalAnalysisResult


def create_test_indicator(name: str, category: IndicatorCategory, 
                         signal: SignalStrength, confidence: float) -> IndicatorResult:
    """Helper to create test indicator"""
    return IndicatorResult(
        indicator_name=name,
        category=category,
        signal=signal,
        value=0.0,
        confidence=confidence,
        timestamp=datetime.utcnow()
    )


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
    analysis.calculate_overall_signal()
    
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


if __name__ == "__main__":
    print("=" * 60)
    print("Test Impact of Accuracy on Overall Scoring")
    print("=" * 60)
    
    # Scenario 1: All high confidence
    test_scenario(
        "All categories have high accuracy",
        trend_conf=0.9,
        momentum_conf=0.9,
        cycle_conf=0.9,
        volume_conf=0.9
    )
    
    # Scenario 2: Trend has low confidence
    test_scenario(
        "Trend low accuracy, rest high accuracy",
        trend_conf=0.3,
        momentum_conf=0.9,
        cycle_conf=0.9,
        volume_conf=0.9
    )
    
    # Scenario 3: Momentum has low confidence
    test_scenario(
        "Momentum low accuracy, rest high accuracy",
        trend_conf=0.9,
        momentum_conf=0.3,
        cycle_conf=0.9,
        volume_conf=0.9
    )
    
    # Scenario 4: Mixed confidences
    test_scenario(
        "Various accuracies",
        trend_conf=0.8,
        momentum_conf=0.6,
        cycle_conf=0.4,
        volume_conf=0.7
    )
    
    # Scenario 5: All low confidence
    test_scenario(
        "All categories have low accuracy",
        trend_conf=0.3,
        momentum_conf=0.3,
        cycle_conf=0.3,
        volume_conf=0.3
    )
    
    print(f"\n{'='*60}")
    print("Explanation:")
    print("  1. Each indicator's accuracy is used in calculating that category's score")
    print("  2. Average accuracy of each category is applied in that category's final weight")
    print("  3. Categories with higher accuracy have more impact on overall signal")
    print("  4. Overall confidence = 60% indicator agreement + 40% average accuracy")
    print("=" * 60)
