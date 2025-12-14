from datetime import datetime

import numpy as np
import pandas as pd
from gravity_tech.database.database_manager import DatabaseManager
from gravity_tech.ml import backtesting


def test_run_backtest_with_real_data_persists_summary(monkeypatch):
    saved_payload = {}

    class FakeDB(DatabaseManager):
        def __init__(self):
            pass

        def save_backtest_run(self, **kwargs):
            saved_payload.update(kwargs)
            return 99

    fake_db = FakeDB()

    def fake_load(symbol: str, limit: int):
        series = np.array([1.0, 1.1, 1.2], dtype=np.float32)
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        return series, series, series, series, dates

    def fake_run(self, highs, lows, closes, volume, dates):
        self.trades = [
            backtesting.TradeResult(
                entry_date=dates[0],
                entry_price=10.0,
                exit_date=dates[1],
                exit_price=12.0,
                pattern_type="gartley",
                direction="long",
                confidence=0.7,
                stop_loss=9.0,
                target_1=11.0,
                target_2=13.0,
                pnl=2.0,
                pnl_percent=20.0,
                outcome="win",
                hit_target="target2",
            ),
            backtesting.TradeResult(
                entry_date=dates[1],
                entry_price=12.0,
                exit_date=dates[2],
                exit_price=11.0,
                pattern_type="gartley",
                direction="long",
                confidence=0.65,
                stop_loss=10.5,
                target_1=12.5,
                target_2=13.5,
                pnl=-1.0,
                pnl_percent=-8.33,
                outcome="loss",
                hit_target="stop_loss",
            ),
        ]

    monkeypatch.setattr(backtesting, "_load_real_ohlcv", fake_load)
    monkeypatch.setattr(backtesting.PatternBacktester, "run_backtest", fake_run)
    monkeypatch.setattr(backtesting.PatternBacktester, "print_summary", lambda self: None)

    backtesting.run_backtest_with_real_data(
        symbol="XYZ",
        source="db",
        interval="1d",
        limit=3,
        min_confidence=0.6,
        persist=True,
        db_manager=fake_db,
        model_version="test-model",
    )

    assert saved_payload, "Expected save_backtest_run to be called"
    assert saved_payload["symbol"] == "XYZ"
    assert saved_payload["interval"] == "1d"
    assert saved_payload["params"]["min_confidence"] == 0.6
    assert saved_payload["metrics"]["win_rate"] > 0
    assert isinstance(saved_payload["period_start"], datetime)
    assert isinstance(saved_payload["period_end"], datetime)
    assert saved_payload["model_version"] == "test-model"
