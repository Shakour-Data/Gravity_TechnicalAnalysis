"""Ensure cycle indicators feed into the combined signal calculation."""
from datetime import datetime, timedelta

import numpy as np
import pytest
from gravity_tech.core.contracts.analysis import AnalysisRequest
from gravity_tech.core.domain.entities import Candle, CoreSignalStrength as SignalStrength
from gravity_tech.services.analysis_service import TechnicalAnalysisService
from gravity_tech.services.signal_engine import compute_overall_signals


@pytest.mark.asyncio
async def test_cycle_scoring_strengthens_overall_signal():
    """Cycle indicators should be present and positively affect the final signal."""
    candles = []
    base_time = datetime.now()
    base_price = 100.0

    # Deterministic rising series with a gentle cyclic component to exercise cycle detectors
    for i in range(120):
        drift = i * 0.6
        cycle_component = 0.8 * np.sin(i / 5.0)
        price = base_price + drift + cycle_component
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=price - 0.3,
            high=price + 0.6,
            low=price - 0.6,
            close=price,
            volume=1500 + i * 12
        ))

    request = AnalysisRequest(symbol="BTCUSDT", timeframe="1h", candles=candles)
    result = await TechnicalAnalysisService.analyze(request)

    # Cycle signals must be present and non-trivial
    assert len(result.cycle_indicators) >= 6
    assert any(ind.signal != SignalStrength.NEUTRAL for ind in result.cycle_indicators)
    assert result.overall_cycle_signal in {
        SignalStrength.BULLISH_BROKEN,
        SignalStrength.BULLISH,
        SignalStrength.VERY_BULLISH,
    }

    # Overall signal should remain bullish on the strong uptrend
    assert result.overall_signal in {
        SignalStrength.BULLISH_BROKEN,
        SignalStrength.BULLISH,
        SignalStrength.VERY_BULLISH,
    }
    assert result.overall_confidence is not None and 0.0 <= result.overall_confidence <= 1.0

    for indicator in result.cycle_indicators:
        assert 0.0 <= indicator.confidence <= 1.0

    # Removing cycle inputs should not improve the overall score on this dataset
    stripped = result.model_copy(deep=True)
    stripped.cycle_indicators = []
    compute_overall_signals(stripped)
    assert result.overall_signal.get_score() > stripped.overall_signal.get_score()
