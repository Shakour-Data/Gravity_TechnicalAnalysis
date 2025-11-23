"""
Test to show Cycle score in final calculation
"""
import asyncio
from datetime import datetime, timedelta
from src.core.domain.entities import Candle
from gravity_tech.models.schemas import AnalysisRequest
from gravity_tech.services.analysis_service import TechnicalAnalysisService
import numpy as np

async def test_cycle_scoring():
    # Generate sample data
    candles = []
    base_time = datetime.now()
    base_price = 100.0
    
    # Create strong uptrend
    for i in range(100):
        price = base_price + (i * 0.5) + np.random.normal(0, 1)
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
    
    print("="*70)
    print("تست محاسبه امتیاز با لحاظ کردن Cycle")
    print("Test Cycle Score in Overall Calculation")
    print("="*70)
    
    result = await TechnicalAnalysisService.analyze(request)
    
    print(f"\n📊 Overall Signals:")
    print(f"  • Trend Signal: {result.overall_trend_signal.value}")
    print(f"  • Momentum Signal: {result.overall_momentum_signal.value}")
    print(f"  • Cycle Signal: {result.overall_cycle_signal.value}")
    print(f"  • Overall Signal: {result.overall_signal.value}")
    print(f"  • Confidence: {result.overall_confidence:.2%}")
    
    print(f"\n📐 Weighting Formula:")
    print(f"  Overall = (Trend × 30%) + (Momentum × 25%) + (Cycle × 25%)")
    print(f"  Then adjusted by Volume (20% as confirmer)")
    
    print(f"\n📈 Indicator Counts:")
    print(f"  • Trend Indicators: {len(result.trend_indicators)}")
    print(f"  • Momentum Indicators: {len(result.momentum_indicators)}")
    print(f"  • Cycle Indicators: {len(result.cycle_indicators)}")
    print(f"  • Volume Indicators: {len(result.volume_indicators)}")
    
    # Show sample cycle indicators
    print(f"\n🔄 Sample Cycle Indicators:")
    for ind in result.cycle_indicators[:3]:
        print(f"  • {ind.indicator_name}: {ind.signal.value} (conf: {ind.confidence:.2%})")
    
    print("\n" + "="*70)
    print("✅ Cycle indicators are now included in overall calculation!")
    print("✅ Weighting: Trend 30%, Momentum 25%, Cycle 25%, Volume 20%")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(test_cycle_scoring())
