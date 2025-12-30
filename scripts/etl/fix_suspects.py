#!/usr/bin/env python3
"""
Auto-fix suspicious data issues for a batch of symbols:
- Deduplicate tables with natural keys.
- Add unique indexes to prevent future dupes.
- Re-ingest historical series for symbols that lag behind source dates.

Usage:
python scripts/etl/fix_suspects.py \
  --target-db "postgresql://gravity:gravity_db_pass@127.0.0.1:5545/tech_analysis" \
  --source-db "E:\\Shakour\\MyProjects\\GravityTseHisPrice\\data\\tse_data.db" \
  --symbols-file batch1_symbols.txt \
  --ingest-limit 0
"""

from __future__ import annotations

import argparse
import sqlite3

# allow running as standalone
import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.etl.run_batch50_full_ingest import ingest_baseline  # type: ignore


def load_symbols(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_indexes(cur) -> None:
    # historical_indicator_scores: use COALESCE for NULL indicator_params so duplicates با NULL هم جلوگیری شود.
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_his_ind_scores ON historical_indicator_scores(symbol, ts, timeframe, indicator_name, coalesce(indicator_params::text,'__NULL__'));"
    )
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_ml_weights_symbol_ts_model ON ml_weights_history(symbol, ts, model_name, timeframe);"
    )
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_tool_perf_symbol_ts ON tool_performance_history(symbol, timeframe, prediction_timestamp, tool_name);"
    )
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_backtest_runs_symbol_period ON backtest_runs(symbol, interval, period_start, period_end, model_version);"
    )
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_pattern_detection_symbol_ts ON pattern_detection_results(symbol, timeframe, timestamp, pattern_type, pattern_name);"
    )


def dedup_table(cur, table: str, key: str) -> int:
    """Delete duplicate rows keeping the newest ctid."""
    cond = " AND ".join([f"a.{k.strip()} = b.{k.strip()}" for k in key.split(",")])
    sql = f"""
    DELETE FROM {table} a
    USING {table} b
    WHERE a.ctid < b.ctid
      AND {cond};
    """
    cur.execute(sql)
    return cur.rowcount


def dedup_all(cur) -> None:
    # targeted chunked dedup for indicator scores (largest source of dupes)
    total_removed = 0
    while True:
        cur.execute(
            """
            WITH d AS (
              SELECT a.ctid
              FROM historical_indicator_scores a
              JOIN historical_indicator_scores b
                ON a.symbol=b.symbol
               AND a.ts=b.ts
               AND a.timeframe=b.timeframe
               AND a.indicator_name=b.indicator_name
               AND a.indicator_params IS NOT DISTINCT FROM b.indicator_params
               AND a.ctid < b.ctid
              LIMIT 5000
            )
            DELETE FROM historical_indicator_scores h USING d WHERE h.ctid = d.ctid;
            """
        )
        removed = cur.rowcount
        total_removed += removed
        if removed == 0:
            break
    print(f"dedup historical_indicator_scores: removed {total_removed} rows")

    tasks = [
        ("ml_weights_history", "symbol,ts,model_name,timeframe"),
        ("tool_performance_history", "symbol,timeframe,prediction_timestamp,tool_name"),
        ("backtest_runs", "symbol,interval,period_start,period_end,model_version"),
        ("pattern_detection_results", "symbol,timeframe,timestamp,pattern_type,pattern_name"),
    ]
    for tbl, key in tasks:
        removed = dedup_table(cur, tbl, key)
        print(f"dedup {tbl}: removed {removed} rows")


def symbols_need_extension(pg_conn, src_db: Path, symbols: list[str]) -> list[str]:
    """Return symbols where historical_scores latest date < source latest date."""
    conn = sqlite3.connect(src_db)
    cur_src = conn.cursor()
    cur_src.execute(
        "SELECT ticker, MAX(date) FROM price_data WHERE ticker IN ({}) GROUP BY ticker".format(
            ",".join("?" for _ in symbols)
        ),
        symbols,
    )
    src_max = {row[0]: row[1] for row in cur_src.fetchall()}
    cur_src.close()
    conn.close()

    cur_pg = pg_conn.cursor()
    cur_pg.execute(
        "SELECT symbol, MAX(ts) FROM historical_scores WHERE symbol = ANY(%s) GROUP BY symbol",
        (symbols,),
    )
    pg_max = {row[0]: row[1].date().isoformat() if row[1] else None for row in cur_pg.fetchall()}
    cur_pg.close()

    need: list[str] = []
    for sym in symbols:
        smax = src_max.get(sym)
        pmax = pg_max.get(sym)
        if smax and (pmax is None or pmax < smax):
            need.append(sym)
    return need


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix duplicates and re-ingest missing history for symbols."
    )
    parser.add_argument("--target-db", required=True, help="Postgres DSN.")
    parser.add_argument("--source-db", required=True, help="SQLite DB with price_data.")
    parser.add_argument("--symbols-file", required=True, help="File with symbols, one per line.")
    parser.add_argument(
        "--ingest-limit", type=int, default=0, help="Limit candles for re-ingest (0 = all)."
    )
    args = parser.parse_args()

    symbols = load_symbols(Path(args.symbols_file))
    if not symbols:
        raise SystemExit("No symbols provided.")

    pg = psycopg2.connect(args.target_db)
    pg.autocommit = True
    cur = pg.cursor()

    dedup_all(cur)
    ensure_indexes(cur)
    cur.close()

    need = symbols_need_extension(pg, Path(args.source_db), symbols)
    pg.close()

    if need:
        ingest_baseline(
            need,
            Path(args.source_db),
            args.target_db,
            candle_limit=args.ingest_limit,
            timeframe="1d",
        )
    else:
        print("No symbols need re-ingest (historical_scores up-to-date with source).")

    print("Done.")


if __name__ == "__main__":
    main()
