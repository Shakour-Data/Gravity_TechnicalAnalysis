from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from gravity_tech.main import app as api_app
from gravity_tech.config.settings import settings
from gravity_tech.core.contracts.analysis import TechnicalAnalysisResult
from gravity_tech.core.domain.entities import Candle
from gravity_tech.middleware.events import event_publisher
from gravity_tech.services import analysis_service
from gravity_tech.services.data_ingestor_service import data_ingestor


@pytest.mark.integration
def test_analyze_endpoint_persists_when_broker_disabled(monkeypatch):
    """
    End-to-end check: /api/v1/analyze -> ingestion path (direct persist when no broker).
    """

    # Configure ingestion flags (no broker)
    settings.enable_data_ingestion = True
    settings.kafka_enabled = False
    settings.rabbitmq_enabled = False

    # Stub analysis result
    result = TechnicalAnalysisResult(symbol="BTCUSDT", timeframe="1h")

    async def fake_analyze(request):  # noqa: ANN001
        return result

    # Track persistence and event publishing
    calls = {"persist": 0, "publish": 0}

    def fake_persist_direct(payload):
        calls["persist"] += 1
        assert payload["symbol"] == "BTCUSDT"
        assert payload["timeframe"] == "1h"

    async def fake_publish(*_args, **_kwargs):
        calls["publish"] += 1

    monkeypatch.setattr(analysis_service.TechnicalAnalysisService, "analyze", fake_analyze)
    monkeypatch.setattr(data_ingestor, "persist_direct", fake_persist_direct)
    monkeypatch.setattr(event_publisher, "publish", fake_publish)

    # Build 50 candles to satisfy request validation
    now = datetime.now(timezone.utc)
    candles = [
        {
            "timestamp": (now - timedelta(minutes=i)).isoformat(),
            "open": 100 + i * 0.01,
            "high": 101 + i * 0.01,
            "low": 99 + i * 0.01,
            "close": 100.5 + i * 0.01,
            "volume": 1_000_000 + i,
            "symbol": "BTCUSDT",
            "timeframe": "1h",
        }
        for i in range(50)
    ]

    client = TestClient(api_app)
    payload = {"symbol": "BTCUSDT", "timeframe": "1h", "candles": candles}
    response = client.post("/api/v1/analyze", json=payload)

    assert response.status_code == 200
    assert calls["persist"] == 1, "Direct persistence should be invoked when broker is disabled"
    assert calls["publish"] == 0, "No event should be published without an enabled broker"
