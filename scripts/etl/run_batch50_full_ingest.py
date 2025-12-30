#!/usr/bin/env python3
"""
One-shot batch runner: fetch + analysis + fill all Postgres tables for the batch symbols.

Steps per run:
1) Select next N symbols from source SQLite (min_candles filter, oldest first), skipping symbols already present in target Postgres.
2) Call scripts/etl/run_batch50.py to fetch from TSETMC and run analysis (fills analysis_results).
3) Using the same symbol list, ingest time-series rows into:
   - historical_scores
   - historical_indicator_scores
   - tool_performance_history (per-day)
   - backtest_runs (per-day)
   - pattern_detection_results (per-day, baseline placeholder)
   - ml_weights_history (per-symbol, per-day)

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
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_BATCH50 = REPO_ROOT / "scripts" / "etl" / "run_batch50.py"


def _load_processed_symbols(target_dsn: str) -> set[str]:
    """Return symbols that already exist in analysis_results so we avoid redoing them."""
    processed: set[str] = set()
    if not target_dsn.lower().startswith("postgres"):
        return processed

    conn = psycopg2.connect(target_dsn)
    cur = conn.cursor()
    try:
        cur.execute("SELECT DISTINCT symbol FROM analysis_results")
        processed.update(row[0] for row in cur.fetchall())
    except Exception:
        # If table is missing, skip silently so the script still runs.
        pass
    cur.close()
    conn.close()
    return processed


def pick_symbols(
    db_path: Path, batch_size: int, min_candles: int, already_done: set[str]
) -> list[str]:
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


def run_batch50(
    source_db: Path, target_db: str, symbols: Sequence[str], limit: int, offline: bool
) -> None:
    """Delegate to run_batch50.py but force the exact symbols list."""
    sym_csv = ",".join(symbols)
    cmd = [
        sys.executable,
        str(RUN_BATCH50),
        "--source-db",
        str(source_db),
        "--target-db",
        target_db,
        "--symbols",
        sym_csv,
        "--limit",
        str(limit),
        "--skip-verify",
    ]
    if offline:
        cmd.append("--offline")
    subprocess.run(cmd, check=True)


def safe_mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def ingest_baseline(
    symbols: Iterable[str],
    source_db: Path,
    target_dsn: str,
    candle_limit: int,
    timeframe: str = "1d",
    trend_window: int = 30,
    disable_baseline_patterns: bool = True,
) -> None:
    symbols_list = list(symbols)
    src = sqlite3.connect(source_db)
    src_cur = src.cursor()
    limit_val = candle_limit if candle_limit and candle_limit > 0 else -1

    pg = psycopg2.connect(target_dsn)
    pg.autocommit = True
    cur = pg.cursor()

    # Ensure helpful indexes/uniques exist
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_historical_scores_symbol_ts ON historical_scores(symbol, ts, timeframe);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_historical_indicator_scores_symbol_ts ON historical_indicator_scores(symbol, ts, timeframe);"
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
    ) ON CONFLICT (symbol, ts, timeframe) DO UPDATE SET
        trend_score = EXCLUDED.trend_score,
        trend_confidence = EXCLUDED.trend_confidence,
        momentum_score = EXCLUDED.momentum_score,
        momentum_confidence = EXCLUDED.momentum_confidence,
        combined_score = EXCLUDED.combined_score,
        combined_confidence = EXCLUDED.combined_confidence,
        trend_weight = EXCLUDED.trend_weight,
        momentum_weight = EXCLUDED.momentum_weight,
        trend_signal = EXCLUDED.trend_signal,
        momentum_signal = EXCLUDED.momentum_signal,
        combined_signal = EXCLUDED.combined_signal,
        recommendation = EXCLUDED.recommendation,
        action = EXCLUDED.action,
        price_at_analysis = EXCLUDED.price_at_analysis,
        volume_score = EXCLUDED.volume_score,
        volatility_score = EXCLUDED.volatility_score,
        cycle_score = EXCLUDED.cycle_score,
        support_resistance_score = EXCLUDED.support_resistance_score,
        raw_data = EXCLUDED.raw_data,
        created_at = EXCLUDED.created_at
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

    now = datetime.now(UTC)
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
            (sym, limit_val),
        ).fetchall()
        if len(candles) < 2:
            continue

        candles = list(reversed(candles))
        closes = [float(c[1]) for c in candles]
        vols = [float(c[2] or 0.0) for c in candles]
        dates = [c[0] for c in candles]

        def _calc_cycle_score(prices: list[float]) -> float:
            """Simple oscillation proxy so cycle_score is not frozen at zero."""
            if len(prices) < 5:
                return 0.0
            returns = [
                (prices[i] - prices[i - 1]) / prices[i - 1]
                for i in range(1, len(prices))
                if prices[i - 1] != 0
            ]
            recent = returns[-7:] if len(returns) > 7 else returns
            drift = (prices[-1] - prices[0]) / (prices[0] or 1e-9)
            vol = (
                (sum((r - (sum(recent) / len(recent))) ** 2 for r in recent) / len(recent)) ** 0.5
                if recent
                else 0.0
            )
            osc = recent[-1] if recent else 0.0
            score = 0.5 * osc + 0.3 * drift + 0.2 * vol
            return float(max(-1.0, min(1.0, score)))

        def _calc_sr_score(prices: list[float]) -> float:
            """Heuristic S/R score: near support -> +1, near resistance -> -1."""
            if not prices:
                return 0.0
            lo = min(prices)
            hi = max(prices)
            if hi == lo:
                return 0.0
            pos = (prices[-1] - lo) / (hi - lo)  # 0 at support, 1 at resistance
            score = (0.5 - pos) * 2.0  # center=0, support=+1, resistance=-1
            return float(max(-1.0, min(1.0, score)))

        for idx in range(len(candles)):
            closes_slice = closes[: idx + 1]
            vols_slice = vols[: idx + 1]
            if trend_window > 1 and len(closes_slice) > trend_window:
                closes_slice = closes_slice[-trend_window:]
                vols_slice = vols_slice[-trend_window:]
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
            regime = (
                "trending_bullish"
                if trend_change > 0.08
                else "trending_bearish"
                if trend_change < -0.08
                else "range"
            )
            prediction = (
                "bullish"
                if trend_change > 0.005
                else "bearish"
                if trend_change < -0.005
                else "neutral"
            )
            vol_profile = (
                "high"
                if vols_slice and (safe_mean(vols_slice) > sorted(vols_slice)[len(vols_slice) // 2])
                else "normal"
            )

            ts_val = datetime.fromisoformat(dates[idx]) if isinstance(dates[idx], str) else now
            if ts_val.tzinfo is None:
                ts_val = ts_val.replace(tzinfo=UTC)

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
                trend_signal="BULLISH"
                if prediction == "bullish"
                else "BEARISH"
                if prediction == "bearish"
                else "NEUTRAL",
                momentum_signal="BULLISH"
                if prediction == "bullish"
                else "BEARISH"
                if prediction == "bearish"
                else "NEUTRAL",
                combined_signal="BULLISH"
                if prediction == "bullish"
                else "BEARISH"
                if prediction == "bearish"
                else "NEUTRAL",
                recommendation="BUY"
                if prediction == "bullish"
                else "SELL"
                if prediction == "bearish"
                else "HOLD",
                action="BUY"
                if prediction == "bullish"
                else "SELL"
                if prediction == "bearish"
                else "HOLD",
                price_at_analysis=last_close,
                volume_score=0.0,
                volatility_score=vol_level,
                cycle_score=_calc_cycle_score(closes_slice),
                support_resistance_score=_calc_sr_score(closes_slice),
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
                    (
                        score_id,
                        sym,
                        ts_val,
                        timeframe,
                        "volatility_std",
                        "volatility",
                        None,
                        vol_level,
                        conf,
                        regime,
                        None,
                    ),
                    (
                        score_id,
                        sym,
                        ts_val,
                        timeframe,
                        "trend_strength",
                        "trend",
                        None,
                        trend_change,
                        conf,
                        prediction,
                        None,
                    ),
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

            # per-day tool/backtest/pattern rows
            tool_row = dict(
                tool_name="baseline_trend",
                tool_category="trend",
                symbol=sym,
                timeframe=timeframe,
                market_regime=regime,
                volatility_level=vol_level,
                trend_strength=trend_change,
                volume_profile=vol_profile,
                prediction_type=prediction,
                prediction_value=trend_change,
                confidence_score=conf,
                actual_result=None,
                actual_price_change=None,
                success=None,
                accuracy=None,
                prediction_timestamp=ts_val,
                result_timestamp=None,
                evaluation_period_hours=24,
                metadata=json.dumps({"source": "batch50_full_ingest"}),
                created_at=now,
                updated_at=now,
            )
            cur.execute(insert_tool_sql, tool_row)

            returns_slice = [
                (closes_slice[i] - closes_slice[i - 1]) / closes_slice[i - 1]
                for i in range(1, len(closes_slice))
                if closes_slice[i - 1] != 0
            ]
            max_close_slice = max(closes_slice)
            max_dd_slice = (
                min((c - max_close_slice) / max_close_slice for c in closes_slice)
                if closes_slice
                else 0.0
            )
            params = {"strategy": "buy_hold", "window": len(closes_slice)}
            metrics = {
                "buy_hold_return": (closes_slice[-1] / closes_slice[0] - 1)
                if closes_slice and closes_slice[0]
                else 0.0,
                "annualized_volatility": vol_level,
                "sharpe": (safe_mean(returns_slice) / (vol_level + 1e-9)) * (252**0.5)
                if returns_slice
                else 0.0,
                "win_rate": sum(1 for r in returns_slice if r > 0) / len(returns_slice)
                if returns_slice
                else 0.0,
                "max_drawdown": max_dd_slice,
                "samples": len(returns_slice),
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
                    dates[idx],
                    "v0.1",
                    now,
                ),
            )

            if not disable_baseline_patterns:
                cur.execute(
                    insert_pattern_sql,
                    (
                        sym,
                        timeframe,
                        ts_val,
                        "baseline",
                        "range_pattern",
                        0.1,
                        0.0,
                        ts_val,
                        ts_val,
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
    parser = argparse.ArgumentParser(
        description="Run batch50 analysis and fill all Postgres tables in one shot."
    )
    parser.add_argument("--source-db", required=True, help="Path to source TSE SQLite DB.")
    parser.add_argument("--target-db", required=True, help="Postgres DSN for analysis/output.")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--min-candles", type=int, default=120)
    parser.add_argument(
        "--limit", type=int, default=300, help="Max candles for analysis_results step."
    )
    parser.add_argument(
        "--ingest-limit",
        type=int,
        default=0,
        help="Max candles for baseline ingest into historical tables (0 = all).",
    )
    parser.add_argument(
        "--trend-window",
        type=int,
        default=30,
        help="Rolling window for trend/volatility calculations (days).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not fetch from TSETMC; use existing source DB only.",
    )
    parser.add_argument(
        "--disable-baseline-patterns",
        action="store_true",
        default=True,
        help="Skip writing placeholder pattern_detection_results rows (prevents fixed confidence/strength values).",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated explicit symbols for this batch. When set, pick_symbols/processed lookup are skipped.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep running batches (size = batch-size) until no eligible symbols remain (based on analysis_results).",
    )
    args = parser.parse_args()

    source_db = Path(args.source_db).resolve()
    total_processed = 0
    iteration = 0
    explicit_symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    while True:
        if explicit_symbols:
            symbols = [s for s in explicit_symbols if s]
            # Ensure we only run once if explicit list is provided
            args.loop = False
        else:
            processed = _load_processed_symbols(args.target_db)
            symbols = pick_symbols(source_db, args.batch_size, args.min_candles, processed)
        if not symbols:
            if args.loop:
                print("No symbols remaining; loop complete.")
            else:
                print(
                    "No symbols selected; nothing to do (maybe all eligible symbols are already processed)."
                )
            break

        iteration += 1
        try:
            print(
                f"[batch {iteration}] Selected {len(symbols)} symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}"
            )
        except UnicodeEncodeError:
            print(f"[batch {iteration}] Selected {len(symbols)} symbols.")

        run_batch50(source_db, args.target_db, symbols, args.limit, offline=args.offline)
        ingest_baseline(
            symbols,
            source_db,
            args.target_db,
            args.ingest_limit,
            trend_window=args.trend_window,
            disable_baseline_patterns=args.disable_baseline_patterns,
        )
        total_processed += len(symbols)

        if not args.loop:
            break

    if total_processed:
        print(f"Done. Processed {total_processed} symbols in {iteration} batch(es).")


if __name__ == "__main__":
    main()
