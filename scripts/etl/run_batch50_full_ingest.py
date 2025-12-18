#!/usr/bin/env python3
"""
One-shot batch runner: fetch + analysis + fill all Postgres tables for the batch symbols.

Steps per run:
1) Select next N symbols from source SQLite (min_candles filter, oldest first), skipping symbols already present in target Postgres.
2) Call scripts/etl/run_batch50.py to fetch from TSETMC and run analysis (fills analysis_results).
3) Using the same symbol list, ingest baseline rows into:
   - historical_scores
   - historical_indicator_scores
   - tool_performance_history
   - backtest_runs
   - pattern_detection_results (placeholder)
   - ml_weights_history (one row)

Usage:
python scripts/etl/run_batch50_full_ingest.py \\
  --source-db "E:\\Shakour\\MyProjects\\GravityTseHisPrice\\data\\tse_data.db" \\
  --target-db "postgresql://postgres:Bedaan4D@127.0.0.1:5432/bedaan4d_db" \\
  --batch-size 50 --min-candles 120 --limit 300
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_BATCH50 = REPO_ROOT / "scripts" / "etl" / "run_batch50.py"


def _load_processed_symbols(target_dsn: str) -> Set[str]:
    """Return symbols that already exist in the target so we do not re-run the same batch."""
    processed: Set[str] = set()
    if not target_dsn.lower().startswith("postgres"):
        return processed

    conn = psycopg2.connect(target_dsn)
    cur = conn.cursor()
    # analysis_results and historical_scores both indicate a finished batch
    for table in ("analysis_results", "historical_scores"):
        try:
            cur.execute(f"SELECT DISTINCT symbol FROM {table}")
            processed.update(row[0] for row in cur.fetchall())
        except Exception:
            # If table is missing, skip silently so the script still runs.
            continue
    cur.close()
    conn.close()
    return processed


def pick_symbols(db_path: Path, batch_size: int, min_candles: int, already_done: Set[str]) -> List[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT lu.symbol
            FROM last_updates lu
            JOIN (
                SELECT ticker, COUNT(*) AS cnt
                FROM price_data
                GROUP BY ticker
            ) p ON lu.symbol = p.ticker
            WHERE p.cnt >= ?
            ORDER BY lu.last_date ASC, lu.symbol ASC
            """,
            (min_candles,),
        ).fetchall()
        filtered = [r[0] for r in rows if r[0] not in already_done]
        return filtered[:batch_size]
    finally:
        conn.close()


def run_batch50(source_db: Path, target_db: str, batch_size: int, min_candles: int, limit: int) -> None:
    cmd = [
        sys.executable,
        str(RUN_BATCH50),
        "--source-db",
        str(source_db),
        "--target-db",
        target_db,
        "--batch-size",
        str(batch_size),
        "--min-candles",
        str(min_candles),
        "--limit",
        str(limit),
        "--skip-verify",
    ]
    subprocess.run(cmd, check=True)


def safe_mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def ingest_baseline(
    symbols: Iterable[str],
    source_db: Path,
    target_dsn: str,
    candle_limit: int,
    timeframe: str = "1d",
) -> None:
    symbols_list = list(symbols)
    src = sqlite3.connect(source_db)
    src_cur = src.cursor()

    pg = psycopg2.connect(target_dsn)
    pg.autocommit = True
    cur = pg.cursor()

    insert_score_sql = """
    INSERT INTO historical_scores (
        symbol, ts, timeframe,
        trend_score, trend_confidence,
        momentum_score, momentum_confidence,
        combined_score, combined_confidence,
        trend_weight, momentum_weight,
        trend_signal, momentum_signal, combined_signal,
        recommendation, action, price_at_analysis,
        volume_score, volatility_score, cycle_score, support_resistance_score,
        raw_data, created_at
    ) VALUES (
        %(symbol)s, %(ts)s, %(timeframe)s,
        %(trend_score)s, %(trend_confidence)s,
        %(momentum_score)s, %(momentum_confidence)s,
        %(combined_score)s, %(combined_confidence)s,
        %(trend_weight)s, %(momentum_weight)s,
        %(trend_signal)s, %(momentum_signal)s, %(combined_signal)s,
        %(recommendation)s, %(action)s, %(price_at_analysis)s,
        %(volume_score)s, %(volatility_score)s, %(cycle_score)s, %(support_resistance_score)s,
        %(raw_data)s, %(created_at)s
    ) ON CONFLICT (symbol, ts, timeframe) DO NOTHING
    RETURNING id;
    """
    select_id_sql = "SELECT id FROM historical_scores WHERE symbol=%s AND ts=%s AND timeframe=%s"

    insert_ind_sql = """
    INSERT INTO historical_indicator_scores (
        score_id, symbol, ts, timeframe,
        indicator_name, indicator_category, indicator_params,
        value, confidence, signal, raw_value
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT DO NOTHING;
    """

    insert_tool_sql = """
    INSERT INTO tool_performance_history (
        tool_name, tool_category, symbol, timeframe, market_regime,
        volatility_level, trend_strength, volume_profile, prediction_type,
        prediction_value, confidence_score, actual_result, actual_price_change,
        success, accuracy, prediction_timestamp, result_timestamp,
        evaluation_period_hours, metadata, created_at, updated_at
    ) VALUES (
        %(tool_name)s, %(tool_category)s, %(symbol)s, %(timeframe)s, %(market_regime)s,
        %(volatility_level)s, %(trend_strength)s, %(volume_profile)s, %(prediction_type)s,
        %(prediction_value)s, %(confidence_score)s, %(actual_result)s, %(actual_price_change)s,
        %(success)s, %(accuracy)s, %(prediction_timestamp)s, %(result_timestamp)s,
        %(evaluation_period_hours)s, %(metadata)s, %(created_at)s, %(updated_at)s
    ) ON CONFLICT DO NOTHING;
    """

    insert_bt_sql = """
    INSERT INTO backtest_runs (
        symbol, source, interval, params, metrics,
        period_start, period_end, model_version, created_at
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT DO NOTHING;
    """

    insert_pattern_sql = """
    INSERT INTO pattern_detection_results (
        symbol, timeframe, timestamp, pattern_type, pattern_name,
        confidence, strength, start_time, end_time, start_price, end_price,
        prediction, target_price, stop_loss, metadata, created_at
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT DO NOTHING;
    """

    insert_weights_sql = """
    INSERT INTO ml_weights_history (
        model_name, model_version, market_regime, timeframe, weights,
        training_accuracy, validation_accuracy, r2_score, mae,
        training_samples, training_date, metadata, created_at,
        symbol, ts
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT DO NOTHING;
    """

    now = datetime.now(timezone.utc)
    processed = 0

    for sym in symbols_list:
        candles = src_cur.execute(
            """
            SELECT date, adj_close, adj_volume
            FROM price_data
            WHERE ticker=?
            ORDER BY date DESC
            LIMIT ?
            """,
            (sym, candle_limit),
        ).fetchall()
        if len(candles) < 2:
            continue

        candles = list(reversed(candles))
        closes = [float(c[1]) for c in candles]
        vols = [float(c[2] or 0.0) for c in candles]
        dates = [c[0] for c in candles]

        for idx in range(len(candles)):
            closes_slice = closes[: idx + 1]
            vols_slice = vols[: idx + 1]
            first_close = closes_slice[0]
            last_close = closes_slice[-1]
            if len(closes_slice) < 2:
                continue
            returns = [
                (closes_slice[i] - closes_slice[i - 1]) / closes_slice[i - 1]
                for i in range(1, len(closes_slice))
                if closes_slice[i - 1] != 0
            ]
            vol_level = (sum(r * r for r in returns) / len(returns)) ** 0.5 if returns else 0.0
            trend_change = (last_close - first_close) / (first_close or 1e-9)
            conf = min(0.99, max(0.05, abs(trend_change) * 2 + vol_level * 0.2))
            regime = "trending_bullish" if trend_change > 0.08 else "trending_bearish" if trend_change < -0.08 else "range"
            prediction = "bullish" if trend_change > 0.005 else "bearish" if trend_change < -0.005 else "neutral"
            vol_profile = "high" if vols_slice and (safe_mean(vols_slice) > sorted(vols_slice)[len(vols_slice) // 2]) else "normal"

            ts_val = datetime.fromisoformat(dates[idx]) if isinstance(dates[idx], str) else now
            if ts_val.tzinfo is None:
                ts_val = ts_val.replace(tzinfo=timezone.utc)

            payload = dict(
                symbol=sym,
                ts=ts_val,
                timeframe=timeframe,
                trend_score=trend_change,
                trend_confidence=conf,
                momentum_score=trend_change,
                momentum_confidence=conf,
                combined_score=trend_change,
                combined_confidence=conf,
                trend_weight=0.5,
                momentum_weight=0.5,
                trend_signal="BULLISH" if prediction == "bullish" else "BEARISH" if prediction == "bearish" else "NEUTRAL",
                momentum_signal="BULLISH" if prediction == "bullish" else "BEARISH" if prediction == "bearish" else "NEUTRAL",
                combined_signal="BULLISH" if prediction == "bullish" else "BEARISH" if prediction == "bearish" else "NEUTRAL",
                recommendation="BUY" if prediction == "bullish" else "SELL" if prediction == "bearish" else "HOLD",
                action="BUY" if prediction == "bullish" else "SELL" if prediction == "bearish" else "HOLD",
                price_at_analysis=last_close,
                volume_score=0.0,
                volatility_score=vol_level,
                cycle_score=0.0,
                support_resistance_score=0.0,
                raw_data=json.dumps({"regime": regime, "vol_profile": vol_profile}),
                created_at=now,
            )

            cur.execute(insert_score_sql, payload)
            res = cur.fetchone()
            if res:
                score_id = res[0]
            else:
                cur.execute(select_id_sql, (sym, ts_val, timeframe))
                fetched = cur.fetchone()
                score_id = fetched[0] if fetched else None

            if score_id:
                ind_rows = [
                    (score_id, sym, ts_val, timeframe, "volatility_std", "volatility", None, vol_level, conf, regime, None),
                    (score_id, sym, ts_val, timeframe, "trend_strength", "trend", None, trend_change, conf, prediction, None),
                ]
                for r in ind_rows:
                    cur.execute(insert_ind_sql, r)

            # per-symbol, per-day weights (time-series)
            weights = {
                "trend": 0.25,
                "momentum": 0.25,
                "volatility": 0.2,
                "volume": 0.15,
                "cycle": 0.1,
                "support_resistance": 0.05,
            }
            cur.execute(
                insert_weights_sql,
                (
                    "baseline_weights",
                    "v0.1",
                    "all",
                    timeframe,
                    json.dumps(weights),
                    0.5,
                    0.5,
                    0.1,
                    0.1,
                    len(closes_slice),
                    ts_val,
                    json.dumps({"source": "batch50_full_ingest"}),
                    now,
                    sym,
                    ts_val,
                ),
            )

        # one tool/backtest/pattern per symbol (latest ts)
        ts_val_last = datetime.fromisoformat(dates[-1]) if isinstance(dates[-1], str) else now
        if ts_val_last.tzinfo is None:
            ts_val_last = ts_val_last.replace(tzinfo=timezone.utc)
        last_close = closes[-1]
        first_close = closes[0]
        returns_all = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
            if closes[i - 1] != 0
        ]
        vol_level_all = (sum(r * r for r in returns_all) / len(returns_all)) ** 0.5 if returns_all else 0.0
        trend_change_all = (last_close - first_close) / (first_close or 1e-9)
        regime_all = "trending_bullish" if trend_change_all > 0.08 else "trending_bearish" if trend_change_all < -0.08 else "range"
        prediction_all = "bullish" if trend_change_all > 0.005 else "bearish" if trend_change_all < -0.005 else "neutral"
        conf_all = min(0.99, max(0.05, abs(trend_change_all) * 2 + vol_level_all * 0.2))
        vol_profile_all = "high" if vols and (safe_mean(vols) > sorted(vols)[len(vols) // 2]) else "normal"

        tool_row = dict(
            tool_name="baseline_trend",
            tool_category="trend",
            symbol=sym,
            timeframe=timeframe,
            market_regime=regime_all,
            volatility_level=vol_level_all,
            trend_strength=trend_change_all,
            volume_profile=vol_profile_all,
            prediction_type=prediction_all,
            prediction_value=trend_change_all,
            confidence_score=conf_all,
            actual_result=None,
            actual_price_change=None,
            success=None,
            accuracy=None,
            prediction_timestamp=ts_val_last,
            result_timestamp=None,
            evaluation_period_hours=24,
            metadata=json.dumps({"source": "batch50_full_ingest"}),
            created_at=now,
            updated_at=now,
        )
        cur.execute(insert_tool_sql, tool_row)

        params = {"strategy": "buy_hold", "window": len(closes)}
        max_close = max(closes)
        max_dd = min((c - max_close) / max_close for c in closes) if closes else 0.0
        metrics = {
            "buy_hold_return": (last_close / first_close - 1) if first_close else 0.0,
            "annualized_volatility": vol_level_all,
            "sharpe": (safe_mean(returns_all) / (vol_level_all + 1e-9)) * (252 ** 0.5) if returns_all else 0.0,
            "win_rate": sum(1 for r in returns_all if r > 0) / len(returns_all) if returns_all else 0.0,
            "max_drawdown": max_dd,
            "samples": len(returns_all),
        }
        cur.execute(
            insert_bt_sql,
            (
                sym,
                "batch50_full_ingest",
                timeframe,
                json.dumps(params),
                json.dumps(metrics),
                dates[0],
                dates[-1],
                "v0.1",
                now,
            ),
        )

        cur.execute(
            insert_pattern_sql,
            (
                sym,
                timeframe,
                ts_val_last,
                "baseline",
                "range_pattern",
                0.1,
                0.0,
                ts_val_last,
                ts_val_last,
                None,
                None,
                "NEUTRAL",
                None,
                None,
                json.dumps({"source": "batch50_full_ingest"}),
                now,
            ),
        )
        processed += 1

    cur.close()
    pg.close()
    src_cur.close()
    src.close()
    print(f"Ingest baseline rows for {processed} symbols.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch50 analysis and fill all Postgres tables in one shot.")
    parser.add_argument("--source-db", required=True, help="Path to source TSE SQLite DB.")
    parser.add_argument("--target-db", required=True, help="Postgres DSN for analysis/output.")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--min-candles", type=int, default=120)
    parser.add_argument("--limit", type=int, default=300, help="Max candles for analysis_results step.")
    parser.add_argument("--ingest-limit", type=int, default=300, help="Max candles for baseline ingest into historical tables.")
    args = parser.parse_args()

    source_db = Path(args.source_db).resolve()
    processed = _load_processed_symbols(args.target_db)
    symbols = pick_symbols(source_db, args.batch_size, args.min_candles, processed)
    if not symbols:
        print("No symbols selected; nothing to do (maybe all eligible symbols are already processed).")
        return

    try:
        print(f"Selected {len(symbols)} symbols for batch: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
    except UnicodeEncodeError:
        print(f"Selected {len(symbols)} symbols for batch.")
    run_batch50(source_db, args.target_db, args.batch_size, args.min_candles, args.limit)
    ingest_baseline(symbols, source_db, args.target_db, args.ingest_limit)
    print("Done.")


if __name__ == "__main__":
    main()
