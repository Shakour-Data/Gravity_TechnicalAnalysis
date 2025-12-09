from fastapi.testclient import TestClient

from gravity_tech.main import app


def test_db_tables_whitelist():
    client = TestClient(app)
    resp = client.get("/api/v1/db/tables")
    assert resp.status_code == 200
    data = resp.json()
    assert "tables" in data
    assert isinstance(data["tables"], list)


def test_db_query_forbidden_table():
    client = TestClient(app)
    resp = client.get("/api/v1/db/query", params={"table": "unauthorized"})
    assert resp.status_code == 403
