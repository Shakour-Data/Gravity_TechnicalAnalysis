"""
Populate ml_weights_history with per-symbol, per-day category weights for the last 90 days.

Strategy:
- Use historical_scores (already computed for 1d) to derive category weights per symbol/day.
- For each (symbol, date) in the window, build a weights dict from the absolute category scores
  (trend, momentum, volume, volatility, cycle, support_resistance) normalized to sum to 1.
- Insert one row per symbol/day into ml_weights_history with lightweight metrics.

This does not retrain ML models; it backfills the table so every symbol/day in the 90-day
window has a weight snapshot. Safe to re-run; it wipes ml_weights_history unless SKIP_DELETE=1.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "TechAnalysis.db"
TIMEFRAME = "1d"
WINDOW_DAYS = 90

SKIP_DELETE = os.getenv("SKIP_DELETE", "").lower() in {"1", "true", "yes"}


def fast_pragmas(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-200000;")
    cur.execute("PRAGMA locking_mode=EXCLUSIVE;")
    con.commit()


def fetch_window(con: sqlite3.Connection) -> tuple[dt.date, dt.date]:
    cur = con.cursor()
    cur.execute("select max(date(timestamp)) from historical_scores where timeframe=?", (TIMEFRAME,))
    max_ts = cur.fetchone()[0]
    if not max_ts:
        raise RuntimeError("historical_scores is empty for timeframe 1d")
    end_date = dt.date.fromisoformat(max_ts)
    start_date = end_date - dt.timedelta(days=WINDOW_DAYS)
    return start_date, end_date


def fetch_scores(con: sqlite3.Connection, start: dt.date, end: dt.date) -> list[tuple]:
    cur = con.cursor()
    cur.execute(
        """
        select symbol, date(timestamp) as d,
               trend_score, momentum_score, volume_score,
               volatility_score, cycle_score, support_resistance_score,
               trend_confidence, momentum_confidence
        from historical_scores
        where timeframe=? and date(timestamp)>=? and date(timestamp)<=?
        order by symbol, d
        """,
        (TIMEFRAME, start.isoformat(), end.isoformat()),
    )
    return cur.fetchall()


def normalize_weights(values: dict[str, float]) -> dict[str, float]:
    abs_vals = {k: abs(v) for k, v in values.items()}
    total = sum(abs_vals.values()) or 1.0
    return {k: v / total for k, v in abs_vals.items()}


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")

    con = sqlite3.connect(DB_PATH)
    fast_pragmas(con)

    start_date, end_date = fetch_window(con)
    print(f"Backfilling ml_weights_history for {TIMEFRAME} from {start_date} to {end_date}")

    rows = fetch_scores(con, start_date, end_date)
    if not rows:
        print("No rows in historical_scores for this window; abort.")
        return

    if not SKIP_DELETE:
        con.execute("delete from ml_weights_history")
        con.commit()
    else:
        print("SKIP_DELETE enabled: existing ml_weights_history rows preserved.")

    buffer: list[tuple] = []
    for symbol, d, trend, momentum, volume, volatility, cycle, sr, tconf, mconf in rows:
        weights = normalize_weights(
            {
                "trend": float(trend or 0.0),
                "momentum": float(momentum or 0.0),
                "volume": float(volume or 0.0),
                "volatility": float(volatility or 0.0),
                "cycle": float(cycle or 0.0),
                "support_resistance": float(sr or 0.0),
            }
        )
        metadata = {
            "symbol": symbol,
            "date": d,
            "source": "historical_scores",
            "trend_score": trend,
            "momentum_score": momentum,
            "volume_score": volume,
            "volatility_score": volatility,
            "cycle_score": cycle,
            "support_resistance_score": sr,
        }
        training_date = dt.datetime.fromisoformat(f"{d}T00:00:00")
        training_acc = float((tconf or 0.0 + mconf or 0.0) / 2)
        buffer.append(
            (
                "daily_category_weights",
                "v0.1",
                "all",
                TIMEFRAME,
                json.dumps(weights),
                training_acc,
                training_acc,
                None,
                None,
                1,
                training_date.isoformat(),
                json.dumps(metadata, ensure_ascii=False),
            )
        )

    con.executemany(
        """
        insert into ml_weights_history (
            model_name, model_version, market_regime, timeframe, weights,
            training_accuracy, validation_accuracy, r2_score, mae,
            training_samples, training_date, metadata
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        buffer,
    )
    con.commit()
    print(f"Inserted {len(buffer)} rows into ml_weights_history.")
    con.close()


if __name__ == "__main__":
    main()
