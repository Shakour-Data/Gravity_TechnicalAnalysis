"""
Simple test for complete analysis with Market Phase
"""
import asyncio
from datetime import datetime, timedelta
from src.core.domain.entities import Candle
from gravity_tech.models.schemas import AnalysisRequest
from gravity_tech.services.analysis_service import TechnicalAnalysisService
import numpy as np

async def test_complete_analysis():
    # Generate sample data with uptrend
    candles = []
    base_time = datetime.now()
    base_price = 100.0
    
    for i in range(100):
        price = base_price + (i * 0.3) + np.random.normal(0, 1)
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=price - 0.5,
            high=price + 1.0,
            low=price - 1.0,
            close=price,
            volume=1000 + i * 10
        ))
    
    request = AnalysisRequest(
        symbol="BTCUSDT",
        timeframe="1h",
        candles=candles
    )
    
    print("Running complete analysis with Market Phase...")
    result = await TechnicalAnalysisService.analyze(request)
    
    print(f"\n✅ Analysis completed!")
    print(f"Symbol: {result.symbol}")
    print(f"Overall Signal: {result.overall_signal.value if result.overall_signal else 'N/A'}")
    
    if result.market_phase_analysis:
        print(f"\n📊 Market Phase Analysis:")
        phase = result.market_phase_analysis
        print(f"  Phase: {phase.market_phase}")
        print(f"  Strength: {phase.phase_strength}")
        print(f"  Score: {phase.overall_score:.1f}/100")
        print(f"  Dow Theory Compliant: {phase.dow_theory_compliance}")
        print(f"  Recommendations: {len(phase.recommendations)}")
    else:
        print("\n⚠️ Market Phase Analysis not available")
    
    print("\n✅ Complete analysis with Market Phase working!")

if __name__ == "__main__":
    asyncio.run(test_complete_analysis())
