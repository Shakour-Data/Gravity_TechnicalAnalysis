from datetime import datetime, timedelta

import pytest
import requests

from gravity_tech.ml.data_connector import DataConnector


def test_fetch_candles_passes_interval(monkeypatch):
    connector = DataConnector(allow_mock_on_failure=False, max_retries=0)

    sample_ts = datetime.utcnow().isoformat()
    payload = {
        "candles": [
            {
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.5,
                "volume": 100,
                "timestamp": sample_ts,
            }
        ]
    }

    captured = {}

    def fake_request(path, params):
        captured["params"] = params
        return payload

    monkeypatch.setattr(connector, "_perform_request", fake_request)
    candles = connector.fetch_candles(symbol="ETHUSDT", interval="4h", limit=1)

    assert len(candles) == 1
    assert captured["params"]["interval"] == "4h"
    assert captured["params"]["symbol"] == "ETHUSDT"


def test_fetch_candles_mock_fallback(monkeypatch):
    connector = DataConnector(allow_mock_on_failure=True, max_retries=0)

    def fake_request(*args, **kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr(connector, "_perform_request", fake_request)
    now = datetime.utcnow()
    candles = connector.fetch_candles(
        symbol="BTCUSDT",
        interval="30m",
        limit=5,
        end_date=now,
        start_date=now - timedelta(hours=3),
    )

    assert len(candles) == 5
    assert connector.last_data_source == "mock"
