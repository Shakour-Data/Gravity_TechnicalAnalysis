"""
Backfill the last 90 days of analysis data day-by-day for every symbol.

- Uses only past candles up to each day (no future leak).
- Computes full technical analysis via TechnicalAnalysisService.
- Persists summary scores, per-indicator rows, and pattern detections.
- Also writes lightweight backtest metrics and ML weight snapshots.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from _paths import ANALYSIS_SRC, REPO_ROOT, extend_sys_path

extend_sys_path()

from gravity_tech.core.contracts.analysis import AnalysisRequest  # noqa: E402
from gravity_tech.core.domain.entities import (  # noqa: E402
    Candle,
    IndicatorResult,
    SignalStrength,
)
from gravity_tech.services.analysis_service import TechnicalAnalysisService  # noqa: E402
from gravity_tech.services.ingestion_payload import build_ingestion_payload  # noqa: E402

DB_PATH = REPO_ROOT / "data" / "TechAnalysis.db"
TIMEFRAME = "1d"
WINDOW_DAYS = 90
LOOKBACK_BUFFER_DAYS = 240
PATTERN_LOOKBACK = 150
INSERT_CHUNK = 2_000

SYMBOL_OFFSET = int(os.getenv("SYMBOL_OFFSET", "0"))
SYMBOL_LIMIT = int(os.getenv("SYMBOL_LIMIT", "0"))
SKIP_DELETE = os.getenv("SKIP_DELETE", "").lower() in {"1", "true", "yes"}


def fast_pragmas(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-200000;")
    cur.execute("PRAGMA locking_mode=EXCLUSIVE;")
    con.commit()


def safe_iso(ts: dt.datetime | None) -> str:
    if ts is None:
        return dt.datetime.now(dt.UTC).isoformat()
    if isinstance(ts, dt.datetime):
        return ts.isoformat()
    return str(ts)


def fetch_window(con: sqlite3.Connection) -> tuple[dt.date, dt.date]:
    cur = con.cursor()
    cur.execute("select max(date(timestamp)) from market_data_cache where timeframe=?", (TIMEFRAME,))
    end = cur.fetchone()[0]
    if not end:
        raise RuntimeError("market_data_cache is empty for timeframe 1d")
    end_date = dt.date.fromisoformat(end)
    start_date = end_date - dt.timedelta(days=WINDOW_DAYS)
    return start_date, end_date


def fetch_symbols(con: sqlite3.Connection) -> list[str]:
    cur = con.cursor()
    cur.execute("select distinct symbol from market_data_cache where timeframe=?", (TIMEFRAME,))
    symbols = [r[0] for r in cur.fetchall()]
    if SYMBOL_OFFSET or SYMBOL_LIMIT:
        symbols = symbols[SYMBOL_OFFSET : (SYMBOL_OFFSET + SYMBOL_LIMIT) if SYMBOL_LIMIT else None]
    return symbols


def fetch_candles(
    con: sqlite3.Connection, symbol: str, start: dt.date, end: dt.date
) -> list[Candle]:
    buffer_start = start - dt.timedelta(days=LOOKBACK_BUFFER_DAYS)
    cur = con.cursor()
    cur.execute(
        """
        select timestamp, open, high, low, close, volume
        from market_data_cache
        where symbol=? and timeframe=? and date(timestamp)>=? and date(timestamp)<=?
        order by timestamp asc
        """,
        (symbol, TIMEFRAME, buffer_start.isoformat(), end.isoformat()),
    )
    candles: list[Candle] = []
    for ts, o, h, l, c, v in cur.fetchall():
        try:
            candles.append(
                Candle(
                    timestamp=dt.datetime.fromisoformat(ts),
                    open=float(o),
                    high=float(h),
                    low=float(l),
                    close=float(c),
                    volume=float(v),
                    symbol=symbol,
                    timeframe=TIMEFRAME,
                )
            )
        except Exception:
            continue
    return candles


def clean_window(con: sqlite3.Connection, start_date: dt.date) -> None:
    cur = con.cursor()
    cur.execute(
        "delete from historical_indicator_scores where timeframe=? and date(timestamp)>=?",
        (TIMEFRAME, start_date.isoformat()),
    )
    cur.execute(
        "delete from historical_scores where timeframe=? and date(timestamp)>=?",
        (TIMEFRAME, start_date.isoformat()),
    )
    cur.execute(
        "delete from pattern_detection_results where timeframe=? and date(timestamp)>=?",
        (TIMEFRAME, start_date.isoformat()),
    )
    cur.execute(
        "delete from backtest_runs where interval=? and date(period_start)>=?",
        (TIMEFRAME, start_date.isoformat()),
    )
    cur.execute("delete from ml_weights_history where timeframe=?", (TIMEFRAME,))
    con.commit()


def iter_indicators(result: object) -> Iterable[IndicatorResult]:
    for bucket in [
        getattr(result, "trend_indicators", []),
        getattr(result, "momentum_indicators", []),
        getattr(result, "cycle_indicators", []),
        getattr(result, "volume_indicators", []),
        getattr(result, "volatility_indicators", []),
        getattr(result, "support_resistance_indicators", []),
    ]:
        for ind in bucket:
            yield ind


def build_indicator_rows(
    result: object,
    score_id: int,
    analysis_ts: dt.datetime,
) -> list[tuple]:
    rows: list[tuple] = []
    result_symbol = getattr(result, "symbol", None)
    for ind in iter_indicators(result):
        value = getattr(ind, "value", None)
        if value is None:
            continue
        try:
            if not math.isfinite(float(value)):  # type: ignore[arg-type]
                continue
        except Exception:
            continue

        category = getattr(ind, "category", None)
        category_val = getattr(category, "value", category) or "UNKNOWN"
        params = getattr(ind, "additional_values", None)
        rows.append(
            (
                score_id,
                result_symbol,
                analysis_ts.isoformat(),
                TIMEFRAME,
                ind.indicator_name,
                category_val,
                json.dumps(params or {}),
                float(value),
                getattr(ind.signal, "name", getattr(ind.signal, "value", ind.signal)),
                float(getattr(ind, "confidence", 0.0) or 0.0),
            )
        )
    return rows


def signal_strength_value(sig: SignalStrength | str | None) -> float:
    if hasattr(sig, "get_score"):
        try:
            return float(sig.get_score())  # type: ignore[arg-type]
        except Exception:
            return 0.0
    try:
        return float(sig) if sig is not None else 0.0
    except Exception:
        return 0.0


def build_pattern_rows(
    symbol: str,
    result,
    price_lookup: dict[dt.datetime, Candle],
    analysis_ts: dt.datetime,
) -> list[tuple]:
    patterns: list = []
    if getattr(result, "classical_patterns", None):
        patterns.extend(result.classical_patterns)  # type: ignore[attr-defined]
    if getattr(result, "candlestick_patterns", None):
        patterns.extend(result.candlestick_patterns)  # type: ignore[attr-defined]

    if not patterns:
        return []

    rows: list[tuple] = []
    for p in patterns:
        name = getattr(p, "pattern_name", None) or getattr(p, "name", None)
        if not name:
            continue

        ptype = getattr(p, "pattern_type", None)
        start_time = getattr(p, "start_time", None)
        end_time = getattr(p, "end_time", None) or analysis_ts
        start_price = price_lookup.get(start_time).close if start_time in price_lookup else None
        end_price = price_lookup.get(end_time).close if end_time in price_lookup else None

        rows.append(
            (
                symbol,
                TIMEFRAME,
                safe_iso(end_time),
                getattr(ptype, "value", ptype) or "UNKNOWN",
                name,
                float(getattr(p, "confidence", 0.0) or 0.0),
                signal_strength_value(getattr(p, "signal", None)),
                safe_iso(start_time),
                safe_iso(end_time),
                start_price,
                end_price,
                getattr(getattr(p, "signal", None), "name", None) or str(getattr(p, "signal", "NEUTRAL")),
                getattr(p, "price_target", None),
                getattr(p, "stop_loss", None),
                json.dumps({"description": getattr(p, "description", None)}, ensure_ascii=False),
            )
        )
    return rows


def insert_summary(
    cur: sqlite3.Cursor,
    payload: dict,
    analysis_ts: dt.datetime,
) -> int:
    cur.execute(
        """
        insert or replace into historical_scores
        (symbol, timestamp, timeframe,
         trend_score, trend_confidence,
         momentum_score, momentum_confidence,
         combined_score, combined_confidence,
         trend_weight, momentum_weight,
         trend_signal, momentum_signal, combined_signal,
         volume_score, volatility_score, cycle_score, support_resistance_score,
         recommendation, action, price_at_analysis, raw_data,
         created_at, updated_at)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload["symbol"],
            analysis_ts.isoformat(),
            payload["timeframe"],
            float(payload.get("trend_score", 0.0) or 0.0),
            float(payload.get("trend_confidence", 0.0) or 0.0),
            float(payload.get("momentum_score", 0.0) or 0.0),
            float(payload.get("momentum_confidence", 0.0) or 0.0),
            float(payload.get("combined_score", 0.0) or 0.0),
            float(payload.get("combined_confidence", 0.0) or 0.0),
            float(payload.get("trend_weight", 0.5) or 0.5),
            float(payload.get("momentum_weight", 0.5) or 0.5),
            payload.get("trend_signal", "NEUTRAL"),
            payload.get("momentum_signal", "NEUTRAL"),
            payload.get("combined_signal", "NEUTRAL"),
            float(payload.get("volume_score", 0.0) or 0.0),
            float(payload.get("volatility_score", 0.0) or 0.0),
            float(payload.get("cycle_score", 0.0) or 0.0),
            float(payload.get("support_resistance_score", 0.0) or 0.0),
            payload.get("recommendation") or ("BUY" if payload.get("combined_score", 0) > 0 else "HOLD"),
            payload.get("action") or "HOLD",
            float(payload.get("price_at_analysis", 0.0) or 0.0),
            json.dumps(payload, default=str, ensure_ascii=False),
            analysis_ts.isoformat(),
            analysis_ts.isoformat(),
        ),
    )
    return int(cur.lastrowid)


def compute_backtest_row(
    candles: Sequence[Candle],
    start: dt.date,
    end: dt.date,
    symbol: str,
) -> tuple | None:
    filtered = [c for c in candles if start <= c.timestamp.date() <= end]
    if len(filtered) < 2:
        return None

    closes = np.array([float(c.close) for c in filtered], dtype=float)
    rets = np.diff(closes) / closes[:-1]
    buy_hold = float(closes[-1] / closes[0] - 1)
    avg_ret = float(np.mean(rets))
    ret_std = float(np.std(rets))
    sharpe = float((avg_ret / (ret_std + 1e-9)) * math.sqrt(252))
    win_rate = float(np.mean(rets > 0))
    max_close = np.maximum.accumulate(closes)
    drawdowns = (closes - max_close) / max_close
    max_dd = float(drawdowns.min()) if len(drawdowns) else 0.0

    metrics = {
        "buy_hold_return": buy_hold,
        "annualized_volatility": float(ret_std * math.sqrt(252)),
        "sharpe": sharpe,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "samples": int(len(rets)),
    }

    params = {"strategy": "buy_hold", "window_days": len(filtered)}

    return (
        symbol,
        "auto",
        TIMEFRAME,
        json.dumps(params),
        json.dumps(metrics),
        filtered[0].timestamp.isoformat(),
        filtered[-1].timestamp.isoformat(),
        "v0.1-auto",
        dt.datetime.now(dt.UTC).isoformat(),
    )


def load_future_returns(
    con: sqlite3.Connection, start: dt.date, end: dt.date
) -> dict[tuple[str, str], float]:
    cur = con.cursor()
    cur.execute(
        """
        select symbol, date(timestamp) as d, close
        from market_data_cache
        where timeframe=? and date(timestamp)>=? and date(timestamp)<=?
        order by symbol, timestamp
        """,
        (TIMEFRAME, start.isoformat(), (end + dt.timedelta(days=1)).isoformat()),
    )
    returns: dict[tuple[str, str], float] = {}
    last_by_symbol: dict[str, tuple[str, float]] = {}
    for symbol, d, close in cur.fetchall():
        if symbol in last_by_symbol:
            prev_date, prev_close = last_by_symbol[symbol]
            if prev_close:
                ret = (float(close) - prev_close) / prev_close
                returns[(symbol, prev_date)] = ret
        last_by_symbol[symbol] = (d, float(close))
    return returns


def compute_ml_weights(
    con: sqlite3.Connection, start: dt.date, end: dt.date
) -> tuple[dict[str, float], dict]:
    future_returns = load_future_returns(con, start, end)
    cur = con.cursor()
    cur.execute(
        """
        select symbol, date(timestamp) as d,
               trend_score, momentum_score, volume_score,
               volatility_score, cycle_score, support_resistance_score
        from historical_scores
        where timeframe=? and date(timestamp)>=? and date(timestamp)<=?
        """,
        (TIMEFRAME, start.isoformat(), end.isoformat()),
    )
    buckets: dict[str, list[tuple[float, float]]] = {
        "trend": [],
        "momentum": [],
        "volume": [],
        "volatility": [],
        "cycle": [],
        "support_resistance": [],
    }

    for row in cur.fetchall():
        symbol, d, trend, momentum, volume, volatility, cycle, sr = row
        fut = future_returns.get((symbol, d))
        if fut is None:
            continue
        buckets["trend"].append((float(trend or 0.0), fut))
        buckets["momentum"].append((float(momentum or 0.0), fut))
        buckets["volume"].append((float(volume or 0.0), fut))
        buckets["volatility"].append((float(volatility or 0.0), fut))
        buckets["cycle"].append((float(cycle or 0.0), fut))
        buckets["support_resistance"].append((float(sr or 0.0), fut))

    corrs: dict[str, float] = {}
    samples: dict[str, int] = {}
    for key, pairs in buckets.items():
        samples[key] = len(pairs)
        if len(pairs) < 5:
            corrs[key] = 0.0
            continue
        vals, futs = zip(*pairs)
        coef = float(np.corrcoef(vals, futs)[0, 1])
        if math.isnan(coef):
            coef = 0.0
        corrs[key] = coef

    abs_corrs = {k: abs(v) for k, v in corrs.items()}
    total = sum(abs_corrs.values()) or 1.0
    weights = {k: v / total for k, v in abs_corrs.items()}

    training_accuracy = float(np.mean(list(abs_corrs.values()))) if abs_corrs else 0.0
    r2_score = float(np.mean([c ** 2 for c in corrs.values()])) if corrs else 0.0
    mae = float(np.mean([abs(r) for r in future_returns.values()])) if future_returns else 0.0

    metadata = {
        "corr": corrs,
        "abs_corr": abs_corrs,
        "samples_per_category": samples,
        "window_days": WINDOW_DAYS,
        "timeframe": TIMEFRAME,
    }

    metrics = {
        "training_accuracy": training_accuracy,
        "validation_accuracy": training_accuracy,
        "r2_score": r2_score,
        "mae": mae,
        "training_samples": int(sum(samples.values())),
    }
    return weights, metadata | metrics


def insert_ml_weights(con: sqlite3.Connection, weights: dict[str, float], metadata: dict) -> None:
    cur = con.cursor()
    cur.execute(
        """
        insert into ml_weights_history (
            model_name, model_version, market_regime, timeframe, weights,
            training_accuracy, validation_accuracy, r2_score, mae,
            training_samples, training_date, metadata
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "category_weight_corr",
            "v0.1",
            "all",
            TIMEFRAME,
            json.dumps(weights),
            metadata.get("training_accuracy"),
            metadata.get("validation_accuracy"),
            metadata.get("r2_score"),
            metadata.get("mae"),
            metadata.get("training_samples"),
            dt.datetime.now(dt.UTC).isoformat(),
            json.dumps(metadata, ensure_ascii=False),
        ),
    )
    con.commit()


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB file not found: {DB_PATH}")

    con = sqlite3.connect(DB_PATH)
    fast_pragmas(con)
    start_date, end_date = fetch_window(con)

    print(f"Processing timeframe={TIMEFRAME}, start={start_date}, end={end_date}")
    symbols = fetch_symbols(con)
    print(f"Symbols to process: {len(symbols)}")

    if not SKIP_DELETE:
        clean_window(con, start_date)
        print("Cleaned existing rows in target window.")
    else:
        print("SKIP_DELETE enabled: existing rows will be preserved.")

    cur = con.cursor()
    indicator_buffer: list[tuple] = []
    pattern_buffer: list[tuple] = []
    backtest_buffer: list[tuple] = []
    total_summaries = 0

    for idx, symbol in enumerate(symbols, 1):
        candles = fetch_candles(con, symbol, start_date, end_date)
        if len(candles) < 60:
            continue
        price_lookup = {c.timestamp: c for c in candles}

        for cidx, candle in enumerate(candles):
            candle_date = candle.timestamp.date()
            if candle_date < start_date or candle_date > end_date:
                continue

            subset = candles[: cidx + 1]
            if len(subset) < 60:
                continue

            request = AnalysisRequest(symbol=symbol, timeframe=TIMEFRAME, candles=subset)
            result = TechnicalAnalysisService._analyze_sync(request)
            result.analysis_timestamp = candle.timestamp  # type: ignore[attr-defined]

            payload = build_ingestion_payload(result, subset)
            payload["analysis_timestamp"] = candle.timestamp

            summary_id = insert_summary(cur, payload, candle.timestamp)
            total_summaries += 1

            indicator_buffer.extend(build_indicator_rows(result, summary_id, candle.timestamp))
            pattern_buffer.extend(build_pattern_rows(symbol, result, price_lookup, candle.timestamp))

            if len(indicator_buffer) >= INSERT_CHUNK:
                cur.executemany(
                    """
                    insert or replace into historical_indicator_scores
                    (score_id, symbol, timestamp, timeframe, indicator_name, indicator_category,
                     indicator_params, value, signal, confidence)
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    indicator_buffer,
                )
                indicator_buffer.clear()
            if len(pattern_buffer) >= INSERT_CHUNK // 4:
                cur.executemany(
                    """
                    insert or replace into pattern_detection_results
                    (symbol, timeframe, timestamp, pattern_type, pattern_name,
                     confidence, strength, start_time, end_time, start_price,
                     end_price, prediction, target_price, stop_loss, metadata)
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    pattern_buffer,
                )
                pattern_buffer.clear()

        bt_row = compute_backtest_row(candles, start_date, end_date, symbol)
        if bt_row:
            backtest_buffer.append(bt_row)

        if idx % 20 == 0:
            con.commit()
            print(f"Processed {idx}/{len(symbols)} symbols...")

    if indicator_buffer:
        cur.executemany(
            """
            insert or replace into historical_indicator_scores
            (score_id, symbol, timestamp, timeframe, indicator_name, indicator_category,
             indicator_params, value, signal, confidence)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            indicator_buffer,
        )
    if pattern_buffer:
        cur.executemany(
            """
            insert or replace into pattern_detection_results
            (symbol, timeframe, timestamp, pattern_type, pattern_name,
             confidence, strength, start_time, end_time, start_price,
             end_price, prediction, target_price, stop_loss, metadata)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            pattern_buffer,
        )
    if backtest_buffer:
        cur.executemany(
            """
            insert or replace into backtest_runs
            (symbol, source, interval, params, metrics,
             period_start, period_end, model_version, created_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            backtest_buffer,
        )

    con.commit()

    weights, metadata = compute_ml_weights(con, start_date, end_date)
    insert_ml_weights(con, weights, metadata)

    print(
        f"Summaries: {total_summaries}, indicators: {cur.execute('select count(1) from historical_indicator_scores where timeframe=? and date(timestamp)>=?', (TIMEFRAME, start_date.isoformat())).fetchone()[0]}, "
        f"patterns: {cur.execute('select count(1) from pattern_detection_results where timeframe=? and date(timestamp)>=?', (TIMEFRAME, start_date.isoformat())).fetchone()[0]}, "
        f"backtests: {len(backtest_buffer)}"
    )
    con.close()


if __name__ == "__main__":
    main()
