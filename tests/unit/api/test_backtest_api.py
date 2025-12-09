from fastapi.testclient import TestClient

from gravity_tech.main import app


def test_backtest_api_synthetic():
    client = TestClient(app)
    payload = {
        "symbol": "SYNTH",
        "min_confidence": 0.5,
        "persist": False,
    }
    resp = client.post("/api/v1/backtest", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data
    assert "trade_count" in data
