from fastapi.testclient import TestClient
from gravity_tech.main import app


def test_root_and_health(monkeypatch, mock_cache_manager):
    # Patch cache manager to avoid Redis dependency during startup
    monkeypatch.setattr("gravity_tech.services.cache_service.cache_manager", mock_cache_manager)

    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "service" in data or "status" in data

    resp2 = client.get("/health")
    assert resp2.status_code == 200
    assert resp2.json().get("status") in ("healthy", "ok", "ready")
