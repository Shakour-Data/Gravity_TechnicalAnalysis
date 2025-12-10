"""
Populate pattern detections, lightweight backtest metrics, and ML weight snapshots
into SQLite for the last 120 days.

Tables covered:
- pattern_detection_results
- backtest_runs
- ml_weights_history (category-level weights derived from correlations)

The script is SQLite-friendly (fast PRAGMAs + chunked inserts) and reuses the
existing market_data_cache + historical_scores tables. It is safe to run
multiple times; set SKIP_DELETE=1 to avoid cleaning the current window.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

# Ensure stdout handles UTF-8 safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from gravity_tech.core.domain.entities import Candle, CoreSignalStrength, PatternType  # noqa: E402
from gravity_tech.core.patterns.candlestick import CandlestickPatterns  # noqa: E402
from gravity_tech.core.patterns.classical import ClassicalPatterns  # noqa: E402

DB_PATH = ROOT / "data" / "TechAnalysis.db"
TIMEFRAME = "1d"
WINDOW_DAYS = 120
PATTERN_LOOKBACK = 200
INSERT_CHUNK = 2_000

SYMBOL_OFFSET = int(os.getenv("SYMBOL_OFFSET", "0"))
SYMBOL_LIMIT = int(os.getenv("SYMBOL_LIMIT", "0"))
USE_SCORED_SYMBOLS = os.getenv("USE_SCORED_SYMBOLS", "").lower() in {"1", "true", "yes"}
SKIP_DELETE = os.getenv("SKIP_DELETE", "").lower() in {"1", "true", "yes"}


def safe_str(text: str | None) -> str:
    """Return console-safe string."""
    if text is None:
        return ""
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def fast_pragmas(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-200000;")  # ~200MB
    cur.execute("PRAGMA locking_mode=EXCLUSIVE;")
    con.commit()


def fetch_symbols_and_window(con: sqlite3.Connection) -> tuple[list[str], dt.date, dt.date]:
    cur = con.cursor()
    cur.execute("select max(date(timestamp)) from market_data_cache where timeframe=?", (TIMEFRAME,))
    max_ts = cur.fetchone()[0]
    if not max_ts:
        raise RuntimeError("market_data_cache is empty for timeframe 1d")
    end_date = dt.date.fromisoformat(max_ts)
    start_date = end_date - dt.timedelta(days=WINDOW_DAYS)
    if USE_SCORED_SYMBOLS:
        cur.execute(
            "select distinct symbol from historical_scores where timeframe=? and date(timestamp)>=?",
            (TIMEFRAME, start_date.isoformat()),
        )
        symbols = [r[0] for r in cur.fetchall()]
    else:
        cur.execute("select distinct symbol from market_data_cache where timeframe=?", (TIMEFRAME,))
        symbols = [r[0] for r in cur.fetchall()]
    return symbols, start_date, end_date


def fetch_candles(
    con: sqlite3.Connection, symbol: str, start: dt.date, end: dt.date
) -> list[Candle]:
    buffer_start = start - dt.timedelta(days=PATTERN_LOOKBACK)
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
    skipped = 0
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
            skipped += 1
    if skipped:
        print(f"[WARN] skipped {skipped} bad candles for {safe_str(symbol)}")
    return candles


def to_iso(ts: dt.datetime | None) -> str | None:
    if ts is None:
        return None
    return ts.isoformat()


def strength_score(sig: object) -> float:
    if isinstance(sig, CoreSignalStrength):
        return float(sig.get_score())
    try:
        return float(sig)  # type: ignore[arg-type]
    except Exception:
        return 0.0


def norm_signal(sig: object) -> str:
    if isinstance(sig, CoreSignalStrength):
        return sig.name
    if sig is None:
        return "UNKNOWN"
    return str(sig)


def detect_patterns_for_symbol(symbol: str, candles: list[Candle]) -> list[tuple]:
    """
    Run candlestick + classical pattern detectors on the recent window and
    return rows ready for insertion into pattern_detection_results.
    """
    if len(candles) < 5:
        return []

    window = candles[-PATTERN_LOOKBACK:]
    price_lookup = {c.timestamp: float(c.close) for c in window}
    pattern_results = []

    try:
        pattern_results.extend(CandlestickPatterns.detect_patterns(window))
    except Exception as exc:
        print(f"[WARN] candlestick detect failed for {safe_str(symbol)}: {exc}")

    try:
        pattern_results.extend(ClassicalPatterns.detect_all(window))
    except Exception as exc:
        print(f"[WARN] classical detect failed for {safe_str(symbol)}: {exc}")

    if not pattern_results:
        return []

    fallback_ts = window[-1].timestamp
    rows: list[tuple] = []
    for p in pattern_results:
        name = getattr(p, "pattern_name", None) or getattr(p, "name", None)
        ptype = getattr(p, "pattern_type", None)
        signal = getattr(p, "signal", None)
        confidence = getattr(p, "confidence", None)
        start_time = getattr(p, "start_time", None)
        end_time = getattr(p, "end_time", None) or fallback_ts
        target_price = getattr(p, "price_target", None) or getattr(p, "target_price", None)
        stop_loss = getattr(p, "stop_loss", None)
        description = getattr(p, "description", None)

        if not name:
            continue

        ts_val = end_time or fallback_ts
        norm_type = getattr(ptype, "value", None) or getattr(ptype, "name", None) or str(ptype)

        rows.append(
            (
                symbol,
                TIMEFRAME,
                to_iso(ts_val),
                norm_type,
                name,
                float(confidence) if confidence is not None else None,
                strength_score(signal),
                to_iso(start_time),
                to_iso(end_time),
                price_lookup.get(start_time) if start_time else None,
                price_lookup.get(end_time) if end_time else None,
                norm_signal(signal),
                target_price,
                stop_loss,
                json.dumps({"description": description}, ensure_ascii=False),
            )
        )
    return rows


def compute_backtest_row(symbol: str, candles: list[Candle], start: dt.date, end: dt.date) -> tuple | None:
    """Compute a simple buy-and-hold style backtest over the window."""
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
        dt.datetime.now(dt.timezone.utc).isoformat(),
    )


def load_future_returns(
    con: sqlite3.Connection, start: dt.date, end: dt.date
) -> dict[tuple[str, str], float]:
    """Build a mapping of (symbol, date) -> next-day return using market_data_cache."""
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
    rows = cur.fetchall()
    returns: dict[tuple[str, str], float] = {}
    last_by_symbol: dict[str, tuple[str, float]] = {}
    for symbol, d, close in rows:
        if symbol in last_by_symbol:
            prev_date, prev_close = last_by_symbol[symbol]
            if prev_close:
                ret = (float(close) - prev_close) / prev_close
                returns[(symbol, prev_date)] = ret
        last_by_symbol[symbol] = (d, float(close))
    return returns


def compute_ml_weights(con: sqlite3.Connection, start: dt.date, end: dt.date) -> tuple[dict[str, float], dict]:
    """
    Compute simple category-level weights using correlation between category
    scores and next-day returns.
    """
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
            dt.datetime.now(dt.timezone.utc).isoformat(),
            json.dumps(metadata, ensure_ascii=False),
        ),
    )
    con.commit()


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB file not found: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    fast_pragmas(con)
    symbols, start_date, end_date = fetch_symbols_and_window(con)
    if SYMBOL_OFFSET or SYMBOL_LIMIT:
        symbols = symbols[SYMBOL_OFFSET : (SYMBOL_OFFSET + SYMBOL_LIMIT) if SYMBOL_LIMIT else None]
        print(f"Batching symbols offset={SYMBOL_OFFSET} limit={SYMBOL_LIMIT} -> {len(symbols)} symbols")

    print(f"Processing timeframe={TIMEFRAME}, start={start_date}, end={end_date}, symbols={len(symbols)}")
    cur = con.cursor()

    if SKIP_DELETE:
        print("SKIP_DELETE enabled: existing rows will be preserved.")
    else:
        cur.execute(
            "delete from pattern_detection_results where timeframe=? and date(timestamp)>=?",
            (TIMEFRAME, start_date.isoformat()),
        )
        cur.execute(
            "delete from backtest_runs where interval=? and date(period_start)>=?",
            (TIMEFRAME, start_date.isoformat()),
        )
        cur.execute("delete from ml_weights_history")
        con.commit()

    pattern_buffer: list[tuple] = []
    backtest_buffer: list[tuple] = []
    total_patterns = 0
    total_backtests = 0

    for idx, symbol in enumerate(symbols, 1):
        candles = fetch_candles(con, symbol, start_date, end_date)
        if not candles:
            continue

        pattern_rows = detect_patterns_for_symbol(symbol, candles)
        pattern_buffer.extend(pattern_rows)

        bt_row = compute_backtest_row(symbol, candles, start_date, end_date)
        if bt_row:
            backtest_buffer.append(bt_row)

        if len(pattern_buffer) >= INSERT_CHUNK:
            cur.executemany(
                """
                insert into pattern_detection_results (
                    symbol, timeframe, timestamp, pattern_type, pattern_name,
                    confidence, strength, start_time, end_time, start_price,
                    end_price, prediction, target_price, stop_loss, metadata
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                pattern_buffer,
            )
            pattern_buffer.clear()
            con.commit()

        if len(backtest_buffer) >= INSERT_CHUNK // 4:
            cur.executemany(
                """
                insert into backtest_runs (
                    symbol, source, interval, params, metrics,
                    period_start, period_end, model_version, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                backtest_buffer,
            )
            backtest_buffer.clear()
            con.commit()

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(symbols)} symbols...")

    if pattern_buffer:
        cur.executemany(
            """
            insert into pattern_detection_results (
                symbol, timeframe, timestamp, pattern_type, pattern_name,
                confidence, strength, start_time, end_time, start_price,
                end_price, prediction, target_price, stop_loss, metadata
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            pattern_buffer,
        )
    total_patterns = cur.execute(
        "select count(1) from pattern_detection_results where timeframe=? and date(timestamp)>=?",
        (TIMEFRAME, start_date.isoformat()),
    ).fetchone()[0]

    if backtest_buffer:
        cur.executemany(
            """
            insert into backtest_runs (
                symbol, source, interval, params, metrics,
                period_start, period_end, model_version, created_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            backtest_buffer,
        )
    total_backtests = cur.execute(
        "select count(1) from backtest_runs where interval=? and date(period_start)>=?",
        (TIMEFRAME, start_date.isoformat()),
    ).fetchone()[0]
    con.commit()

    # Compute and store ML weights (category-level)
    weights, metadata = compute_ml_weights(con, start_date, end_date)
    insert_ml_weights(con, weights, metadata)

    print(
        f"patterns rows now: {total_patterns}, backtests rows now: {total_backtests}, "
        "ml_weights entries: 1"
    )
    con.close()


if __name__ == "__main__":
    main()
