"""
Example usage of the Technical Analysis Service
"""

import asyncio
from datetime import datetime, timedelta
from models.schemas import Candle, AnalysisRequest
from services.analysis_service import TechnicalAnalysisService


def generate_sample_candles(num_candles: int = 100) -> list:
    """Generate sample candle data"""
    candles = []
    base_price = 40000
    base_time = datetime.now() - timedelta(hours=num_candles)
    
    for i in range(num_candles):
        # Simulate price movement
        open_price = base_price + (i * 50)
        close_price = open_price + ((i % 20) - 10) * 100
        high_price = max(open_price, close_price) + 200
        low_price = min(open_price, close_price) - 200
        
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=float(open_price),
            high=float(high_price),
            low=float(low_price),
            close=float(close_price),
            volume=float(1000 + (i * 50))
        ))
    
    return candles


async def main():
    """Main example function"""
    
    print("=== Technical Analysis Service Example ===\n")
    
    # Generate sample data
    print("Generating sample candle data...")
    candles = generate_sample_candles(100)
    print(f"Generated {len(candles)} candles\n")
    
    # Create analysis request
    request = AnalysisRequest(
        symbol="BTCUSDT",
        timeframe="1h",
        candles=candles
    )
    
    # Perform analysis
    print("Performing comprehensive technical analysis...")
    result = await TechnicalAnalysisService.analyze(request)
    
    # Display results
    print(f"\n=== Analysis Results for {result.symbol} ({result.timeframe}) ===\n")
    
    # Overall signals
    print("📊 Overall Signals:")
    print(f"  • Trend Signal: {result.overall_trend_signal.value if result.overall_trend_signal else 'N/A'}")
    print(f"  • Momentum Signal: {result.overall_momentum_signal.value if result.overall_momentum_signal else 'N/A'}")
    print(f"  • Cycle Signal: {result.overall_cycle_signal.value if result.overall_cycle_signal else 'N/A'}")
    print(f"  • Overall Signal: {result.overall_signal.value if result.overall_signal else 'N/A'}")
    print(f"  • Confidence: {result.overall_confidence:.2%}")
    print(f"\n  📐 Weighting: Trend 30%, Momentum 25%, Cycle 25%, Volume 20%\n")
    
    # Trend Indicators
    print("📈 Trend Indicators:")
    for indicator in result.trend_indicators[:5]:  # Show first 5
        print(f"  • {indicator.indicator_name}: {indicator.signal.value} "
              f"(confidence: {indicator.confidence:.2%})")
    
    # Momentum Indicators
    print("\n⚡ Momentum Indicators:")
    for indicator in result.momentum_indicators[:5]:
        print(f"  • {indicator.indicator_name}: {indicator.signal.value} "
              f"(value: {indicator.value:.2f}, confidence: {indicator.confidence:.2%})")
    
    # Cycle Indicators
    print("\n🔄 Cycle Indicators:")
    for indicator in result.cycle_indicators:
        print(f"  • {indicator.indicator_name}: {indicator.signal.value}")
        print(f"    {indicator.description}")
    
    # Volume Indicators
    print("\n📊 Volume Indicators:")
    for indicator in result.volume_indicators:
        print(f"  • {indicator.indicator_name}: {indicator.signal.value} "
              f"(confidence: {indicator.confidence:.2%})")
    
    # Volatility Indicators
    print("\n🌊 Volatility Indicators:")
    for indicator in result.volatility_indicators[:3]:
        print(f"  • {indicator.indicator_name}: {indicator.signal.value}")
    
    # Support/Resistance
    print("\n🎯 Support/Resistance:")
    for indicator in result.support_resistance_indicators:
        print(f"  • {indicator.indicator_name}: {indicator.signal.value}")
        if indicator.additional_values:
            for key, value in list(indicator.additional_values.items())[:3]:
                print(f"    - {key}: {value:.2f}")
    
    # Candlestick Patterns
    if result.candlestick_patterns:
        print("\n🕯️ Detected Candlestick Patterns:")
        for pattern in result.candlestick_patterns:
            print(f"  • {pattern.pattern_name}: {pattern.signal.value} "
                  f"(confidence: {pattern.confidence:.2%})")
            print(f"    {pattern.description}")
    
    # Elliott Wave Analysis
    if result.elliott_wave_analysis:
        print("\n🌊 Elliott Wave Analysis:")
        ew = result.elliott_wave_analysis
        print(f"  • Pattern: {ew.wave_pattern}")
        print(f"  • Current Wave: {ew.current_wave}")
        print(f"  • Signal: {ew.signal.value}")
        print(f"  • Confidence: {ew.confidence:.2%}")
        print(f"  • Projected Target: {ew.projected_target:.2f}")
        print(f"  • {ew.description}")
        print(f"  • Wave Points: {len(ew.waves)} pivots detected")
    
    # Market Phase Analysis (Dow Theory)
    if result.market_phase_analysis:
        print("\n📊 Market Phase Analysis (Dow Theory):")
        phase = result.market_phase_analysis
        print(f"  • Phase: {phase.market_phase}")
        print(f"  • Strength: {phase.phase_strength}")
        print(f"  • Overall Score: {phase.overall_score:.1f}/100")
        print(f"  • Trend Structure: {phase.trend_structure}")
        print(f"  • Volume Confirmation: {'Yes' if phase.volume_confirmation else 'No'}")
        print(f"  • Dow Theory Compliant: {'✅' if phase.dow_theory_compliance else '❌'}")
        print(f"\n  📝 Description:")
        print(f"     {phase.description}")
        print(f"\n  💡 Top Recommendations:")
        for i, rec in enumerate(phase.recommendations[:3], 1):
            print(f"     {i}. {rec}")
    
    print("\n" + "="*60)
    
    # Example: Analyze specific indicators only
    print("\n=== Specific Indicators Analysis ===\n")
    
    specific_results = await TechnicalAnalysisService.analyze_specific_indicators(
        candles=candles,
        indicator_names=["rsi", "macd", "bollinger", "pivot"]
    )
    
    for indicator in specific_results:
        print(f"{indicator.indicator_name}: {indicator.signal.value}")
        print(f"  Value: {indicator.value:.2f}")
        print(f"  Description: {indicator.description}\n")


if __name__ == "__main__":
    asyncio.run(main())
