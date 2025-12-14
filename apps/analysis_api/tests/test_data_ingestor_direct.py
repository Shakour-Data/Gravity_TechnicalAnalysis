from datetime import datetime, timezone

from gravity_tech.core.contracts.analysis import TechnicalAnalysisResult
from gravity_tech.core.domain.entities import Candle, IndicatorCategory, IndicatorResult, SignalStrength
from gravity_tech.services.data_ingestor_service import DataIngestorService
from gravity_tech.services.ingestion_payload import build_ingestion_payload


def test_persist_direct_invokes_persist_entry(monkeypatch):
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
        signal=SignalStrength.BULLISH,
        value=55.0,
        confidence=0.8,
        additional_values=None,
    )

    result = TechnicalAnalysisResult(
        symbol="BTCUSDT",
        timeframe="1h",
        trend_indicators=[trend_indicator],
    )

    payload = build_ingestion_payload(result, candles)

    ingestor = DataIngestorService()
    ingestor.database_url = "postgresql://user:pass@localhost:5432/testdb"

    called = {"count": 0}

    def fake_persist_entry(entry, **kwargs):
        called["count"] += 1
        assert entry.symbol == "BTCUSDT"
        assert kwargs["indicator_scores"]

    monkeypatch.setattr(ingestor, "_persist_entry", fake_persist_entry)

    ingestor.persist_direct(payload)

    assert called["count"] == 1
