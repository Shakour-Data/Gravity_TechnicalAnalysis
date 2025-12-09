import tempfile
from datetime import datetime

from gravity_tech.database.database_manager import DatabaseManager, DatabaseType


def test_save_backtest_run_sqlite_persists_row():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/backtest.db"
        manager = DatabaseManager(db_type=DatabaseType.SQLITE, sqlite_path=db_path, auto_setup=True)

        inserted_id = manager.save_backtest_run(
            symbol="TEST",
            source="db",
            interval="1d",
            params={"min_confidence": 0.6},
            metrics={"total_trades": 3, "win_rate": 0.66},
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 10),
            model_version="v1",
        )

        assert inserted_id is not None

        conn = manager.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, source, interval, model_version, params, metrics FROM backtest_runs WHERE id=?", (inserted_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row["symbol"] == "TEST"
        assert row["source"] == "db"
        assert row["interval"] == "1d"
        assert row["model_version"] == "v1"
        assert '"min_confidence": 0.6' in row["params"]
        assert '"total_trades": 3' in row["metrics"]


def test_save_backtest_run_json_fallback():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = f"{tmpdir}/backtests.json"
        manager = DatabaseManager(db_type=DatabaseType.JSON_FILE, json_path=json_path, auto_setup=True)

        inserted_id = manager.save_backtest_run(
            symbol="TEST2",
            source="synthetic",
            interval="1d",
            params=None,
            metrics={"win_rate": 0.5},
            period_start=None,
            period_end=None,
            model_version=None,
        )

        assert inserted_id == 1
        assert manager.json_data["backtest_runs"][0]["symbol"] == "TEST2"
