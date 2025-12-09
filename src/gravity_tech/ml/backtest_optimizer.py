"""
Heuristic backtest parameter suggester based on stored `backtest_runs`.
Uses highest win_rate (and then profit_factor) to pick params per symbol/interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gravity_tech.database.database_manager import DatabaseManager


@dataclass
class BacktestParams:
    min_confidence: float
    limit: int | None = None
    interval: str | None = None
    source: str | None = None


def suggest_params(
    symbol: str,
    interval: str | None = None,
    *,
    db_manager: DatabaseManager | None = None,
    default_min_confidence: float = 0.6,
) -> BacktestParams:
    """
    Suggest backtest params for a symbol/interval using historical runs.
    Falls back to defaults when no history exists.
    """
    manager = db_manager or DatabaseManager(auto_setup=True)

    if manager.db_type == manager.db_type.JSON_FILE:
        runs = manager.json_data.get("backtest_runs", [])
    else:
        conn = manager.get_connection()
        cursor = conn.cursor()
        clause_interval = "AND interval = ?" if interval else ""
        params: list[Any] = [symbol]
        if interval:
            params.append(interval)
        cursor.execute(
            f"""
            SELECT params, metrics, interval
            FROM backtest_runs
            WHERE symbol = ?
            {clause_interval}
            ORDER BY created_at DESC
            """,
            tuple(params),
        )
        rows = cursor.fetchall()
        runs = [
            {
                "params": row["params"],
                "metrics": row["metrics"],
                "interval": row["interval"],
            }
            for row in rows
        ]

    best_min_conf = default_min_confidence
    best_limit = None
    best_interval = interval
    best_source = None
    best_score = -1.0

    for run in runs:
        try:
            params_obj = run["params"]
            metrics_obj = run["metrics"]
            if isinstance(params_obj, str):
                import json

                params_obj = json.loads(params_obj)
            if isinstance(metrics_obj, str):
                import json

                metrics_obj = json.loads(metrics_obj)

            win_rate = float(metrics_obj.get("win_rate", 0.0))
            profit_factor = float(metrics_obj.get("profit_factor", 0.0))
            score = win_rate * 0.7 + min(profit_factor, 5.0) * 0.3
            if score > best_score:
                best_score = score
                best_min_conf = float(params_obj.get("min_confidence", default_min_confidence))
                best_limit = params_obj.get("limit")
                best_interval = run.get("interval") or params_obj.get("interval") or interval
                best_source = run.get("source") or params_obj.get("source")
        except Exception:
            continue

    return BacktestParams(
        min_confidence=best_min_conf,
        limit=best_limit if isinstance(best_limit, int) else None,
        interval=best_interval,
        source=best_source,
    )
