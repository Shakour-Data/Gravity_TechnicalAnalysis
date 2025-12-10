from datetime import datetime, timezone

from gravity_tech.core.contracts.analysis import TechnicalAnalysisResult
from gravity_tech.core.domain.entities import Candle, IndicatorCategory, IndicatorResult, SignalStrength
from gravity_tech.services.ingestion_payload import build_ingestion_payload


def test_build_ingestion_payload_basic_scores():
    candles = [
        Candle(
            timestamp=datetime.now(timezone.utc),
            open=100,
            high=110,
            low=95,
            close=105,
            volume=1_000_000,
            symbol="BTCUSDT",
            timeframe="1h",
        )
    ]

    trend_indicator = IndicatorResult(
        indicator_name="SMA",
        category=IndicatorCategory.TREND,
        signal=SignalStrength.VERY_BULLISH,
        value=55.0,
        confidence=0.8,
        additional_values=None,
    )

    momentum_indicator = IndicatorResult(
        indicator_name="RSI",
        category=IndicatorCategory.MOMENTUM,
        signal=SignalStrength.BULLISH,
        value=60.0,
        confidence=0.7,
        additional_values=None,
    )

    result = TechnicalAnalysisResult(
        symbol="BTCUSDT",
        timeframe="1h",
        trend_indicators=[trend_indicator],
        momentum_indicators=[momentum_indicator],
    )

    payload = build_ingestion_payload(result, candles)

    assert payload["symbol"] == "BTCUSDT"
    assert payload["price_at_analysis"] == 105
    assert payload["trend_score"] > 0
    assert payload["combined_score"] > 0
    assert payload["indicator_scores"], "indicator scores should be normalized for downstream storage"
