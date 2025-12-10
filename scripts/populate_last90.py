"""
Batch populate last 90 days of indicator results into SQLite.

Strategy:
- Load candles per symbol for the last 90 days plus a 1-year lookback buffer
  (to satisfy long lookback indicators like ADX/Ichimoku/ATR).
- For each candle in the 90-day window, run all core indicator calculate_all
  methods and store per-indicator outputs into historical_indicator_scores.
- Derive a lightweight summary row into historical_scores by averaging
  normalized signals per category (simple -3..+3 scale).

This script is intentionally SQLite-friendly: chunked inserts inside a single
transaction with fast PRAGMA settings. It assumes the database file exists at
data/TechAnalysis.db and the market_data_cache table is already populated.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Ensure stdout can handle UTF-8 safely (Windows console)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

# Ensure local package imports work when run from repo root

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from gravity_tech.core.domain.entities import (  # noqa: E402
    Candle,  # noqa: E402
    IndicatorCategory,
    IndicatorResult,
    SignalStrength,
)
from gravity_tech.core.indicators.cycle import CycleIndicators  # noqa: E402
from gravity_tech.core.indicators.momentum import MomentumIndicators  # noqa: E402
from gravity_tech.core.indicators.support_resistance import (  # noqa: E402
    SupportResistanceIndicators,
)
from gravity_tech.core.indicators.trend import TrendIndicators  # noqa: E402
from gravity_tech.core.indicators.volatility import (  # noqa: E402
    VolatilityIndicators,
    convert_volatility_to_indicator_result,
)
from gravity_tech.core.indicators.volume import VolumeIndicators  # noqa: E402


def safe_str(text: str) -> str:
    """Return ASCII-safe string for console logging."""
    if text is None:
        return ""
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


DB_PATH = ROOT / "data" / "TechAnalysis.db"
TIMEFRAME = "1d"
WINDOW_DAYS = 90
LOOKBACK_BUFFER_DAYS = 200  # extra raw candles to satisfy long indicators
# For performance, the buffer can be reduced if you only care about
# indicators with shorter lookbacks; adjust as needed.
INSERT_CHUNK = 50_000

# Optional batching controls via environment variables
SYMBOL_OFFSET = int(os.getenv("SYMBOL_OFFSET", "0"))
SYMBOL_LIMIT = int(os.getenv("SYMBOL_LIMIT", "0"))
SKIP_DELETE = os.getenv("SKIP_DELETE", "").lower() in {"1", "true", "yes"}
SKIP_PROCESSED = os.getenv("SKIP_PROCESSED", "").lower() in {"1", "true", "yes"}
USE_ML_WEIGHTS = os.getenv("USE_ML_WEIGHTS", "").lower() in {"1", "true", "yes"}
WEIGHT_MODEL = os.getenv("WEIGHT_MODEL", "daily_category_weights")

# Simple mapping of SignalStrength to numeric score for aggregation
SIGNAL_SCORE = {
    "VERY_BEARISH": -3,
    "BEARISH": -2,
    "BEARISH_BROKEN": -1,
    "NEUTRAL": 0,
    "BULLISH_BROKEN": 1,
    "BULLISH": 2,
    "VERY_BULLISH": 3,
}


def fetch_symbols_and_date_range(con: sqlite3.Connection) -> tuple[list[str], dt.date, dt.date]:
    cur = con.cursor()
    cur.execute("select max(date(timestamp)) from market_data_cache where timeframe=?", (TIMEFRAME,))
    max_ts = cur.fetchone()[0]
    if not max_ts:
        raise RuntimeError("market_data_cache is empty for timeframe 1d")
    end_date = dt.date.fromisoformat(max_ts)
    start_date = end_date - dt.timedelta(days=WINDOW_DAYS)
    cur.execute("select distinct symbol from market_data_cache where timeframe=?", (TIMEFRAME,))
    symbols = [r[0] for r in cur.fetchall()]
    return symbols, start_date, end_date


def fetch_candles_for_symbol(
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
    rows = cur.fetchall()
    candles: list[Candle] = []
    skipped = 0
    for ts, o, h, low, c, v in rows:
        try:
            candles.append(
                Candle(
                    timestamp=dt.datetime.fromisoformat(ts),
                    open=float(o),
                    high=float(h),
                    low=float(low),
                    close=float(c),
                    volume=float(v),
                    symbol=symbol,
                    timeframe=TIMEFRAME,
                )
            )
        except Exception:
            skipped += 1
            continue
    if skipped:
        print(f"[WARN] skipped {skipped} bad candles for {safe_str(symbol)}")
    return candles


def collect_indicator_results(candles: list[Candle]) -> list[IndicatorResult]:
    """
    Run all indicator families' calculate_all on the provided candle history.
    Returns IndicatorResult list (flattens volatility dict).
    """
    results: list[IndicatorResult] = []
    results.extend(TrendIndicators.calculate_all(candles))
    results.extend(MomentumIndicators.calculate_all(candles))
    results.extend(VolumeIndicators.calculate_all(candles))
    results.extend(SupportResistanceIndicators.calculate_all(candles))
    results.extend(CycleIndicators.calculate_all(candles))

    # Volatility returns a dict of VolatilityResult; convert to IndicatorResult
    vol_dict = VolatilityIndicators.calculate_all(candles)
    for name, vol_result in vol_dict.items():
        results.append(convert_volatility_to_indicator_result(vol_result, name))
    return results


def prepare_indicator_rows(
    symbol: str,
    candles: list[Candle],
    start: dt.date,
    weight_lookup: dict[tuple[str, dt.date], dict[str, float]] | None = None,
) -> tuple[list[tuple], list[tuple]]:
    """
    Slide over candles; for those within start..end, compute indicators and return
    (indicator_rows, summary_rows).
    """
    indicator_rows: list[tuple] = []
    summary_rows: list[tuple] = []
    by_category_signal = defaultdict(list)
    by_category_conf = defaultdict(list)

    for idx, candle in enumerate(candles):
        candle_date = candle.timestamp.date()
        if candle_date < start:
            continue

        subset = candles[: idx + 1]
        try:
            results = collect_indicator_results(subset)
        except Exception as exc:
            print(f"[WARN] calc error symbol={safe_str(symbol)} ts={candle.timestamp.isoformat()} err={exc}")
            continue
        by_category_signal.clear()
        by_category_conf.clear()

        for res in results:
            # Skip invalid values
            if res.value is None:
                continue
            try:
                if float(res.value) != float(res.value):  # NaN check
                    continue
            except Exception:
                continue
            indicator_rows.append(
                (
                    None,  # score_id
                    symbol,
                    candle.timestamp.isoformat(),
                    TIMEFRAME,
                    res.indicator_name,
                    res.category.value if isinstance(res.category, IndicatorCategory) else str(res.category),
                    None,  # indicator_params
                    float(res.value),
                    res.signal.name if isinstance(res.signal, SignalStrength) else str(res.signal),
                    float(res.confidence),
                )
            )
            # Aggregate for summary scores
            sig_key = res.signal.name if isinstance(res.signal, SignalStrength) else str(res.signal)
            if sig_key in SIGNAL_SCORE:
                by_category_signal[res.category].append(SIGNAL_SCORE[sig_key])
            by_category_conf[res.category].append(res.confidence)

        def agg(cat: IndicatorCategory) -> tuple[float, float, str]:
            sigs = by_category_signal.get(cat, [])
            confs = by_category_conf.get(cat, [])
            score = float(np.mean(sigs)) if sigs else 0.0
            conf = float(np.mean(confs)) if confs else 0.0
            if score > 1:
                sig = "BULLISH"
            elif score > 0.3:
                sig = "BULLISH_BROKEN"
            elif score < -1:
                sig = "BEARISH"
            elif score < -0.3:
                sig = "BEARISH_BROKEN"
            else:
                sig = "NEUTRAL"
            return score, conf, sig

        trend_score, trend_conf, trend_sig = agg(IndicatorCategory.TREND)
        mom_score, mom_conf, mom_sig = agg(IndicatorCategory.MOMENTUM)
        vol_score, vol_conf, _ = agg(IndicatorCategory.VOLATILITY)
        volu_score, _, _ = agg(IndicatorCategory.VOLUME)
        cyc_score, _, _ = agg(IndicatorCategory.CYCLE)
        sr_score, _, _ = agg(IndicatorCategory.SUPPORT_RESISTANCE)

        weights = {}
        if weight_lookup is not None:
            weights = weight_lookup.get((symbol, candle_date), {}) or {}

        trend_w = float(weights.get("trend", 0.5))
        mom_w = float(weights.get("momentum", 0.5))
        w_sum = trend_w + mom_w
        if w_sum <= 0:
            trend_w = mom_w = 0.5
            w_sum = 1.0

        combined_score = (trend_score * trend_w + mom_score * mom_w) / w_sum
        combined_conf = (trend_conf * trend_w + mom_conf * mom_w) / w_sum
        if combined_score > 1:
            combined_sig = "BULLISH"
        elif combined_score < -1:
            combined_sig = "BEARISH"
        else:
            combined_sig = "NEUTRAL"

        summary_rows.append(
            (
                symbol,
                candle.timestamp.isoformat(),
                TIMEFRAME,
                trend_score,
                trend_conf,
                mom_score,
                mom_conf,
                combined_score,
                combined_conf,
                trend_w,
                mom_w,
                trend_sig,
                mom_sig,
                combined_sig,
                volu_score,
                vol_score,
                cyc_score,
                sr_score,
                None,  # recommendation
                None,  # action
                float(candle.close),
                None,  # raw_data
                dt.datetime.now(dt.UTC).isoformat(),
                dt.datetime.now(dt.UTC).isoformat(),
            )
        )

    return indicator_rows, summary_rows


def fast_pragmas(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-200000;")  # ~200MB
    cur.execute("PRAGMA locking_mode=EXCLUSIVE;")
    con.commit()


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB file not found: {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    fast_pragmas(con)
    symbols, start_date, end_date = fetch_symbols_and_date_range(con)
    if SYMBOL_OFFSET or SYMBOL_LIMIT:
        symbols = symbols[SYMBOL_OFFSET : (SYMBOL_OFFSET + SYMBOL_LIMIT) if SYMBOL_LIMIT else None]
        print(f"Batching symbols offset={SYMBOL_OFFSET} limit={SYMBOL_LIMIT} -> {len(symbols)} symbols")
    print(f"Processing timeframe={TIMEFRAME}, start={start_date}, end={end_date}, symbols={len(symbols)}")

    cur = con.cursor()
    if SKIP_DELETE:
        print("SKIP_DELETE enabled: existing rows for this window will not be deleted.")
    else:
        # Clean previous 90d to avoid duplicates
        cur.execute(
            "delete from historical_indicator_scores where timeframe=? and date(timestamp)>=?",
            (TIMEFRAME, start_date.isoformat()),
        )
        cur.execute(
            "delete from historical_scores where timeframe=? and date(timestamp)>=?",
            (TIMEFRAME, start_date.isoformat()),
        )
        con.commit()

    total_indicator_rows = 0
    total_summary_rows = 0

    weight_lookup: dict[tuple[str, dt.date], dict[str, float]] | None = None
    if USE_ML_WEIGHTS:
        cur.execute(
            """
            select json_extract(metadata, '$.symbol') as symbol,
                   date(training_date) as d,
                   weights
            from ml_weights_history
            where model_name=?
            """,
            (WEIGHT_MODEL,),
        )
        weight_lookup = {}
        for symbol, d, wjson in cur.fetchall():
            if not symbol or not d or not wjson:
                continue
            try:
                weights = json.loads(wjson)
            except Exception:
                continue
            weight_lookup[(symbol, dt.date.fromisoformat(d))] = weights
        print(f"Loaded weights for {len(weight_lookup)} symbol/day entries from ml_weights_history using model={WEIGHT_MODEL}")

    indicator_buffer: list[tuple] = []
    summary_buffer: list[tuple] = []

    for idx, symbol in enumerate(symbols, 1):
        if SKIP_PROCESSED:
            cur.execute(
                "select 1 from historical_scores where symbol=? and timeframe=? and date(timestamp)>=? limit 1",
                (symbol, TIMEFRAME, start_date.isoformat()),
            )
            if cur.fetchone():
                continue

        candles = fetch_candles_for_symbol(con, symbol, start_date, end_date)
        if not candles:
            continue
        try:
            ind_rows, sum_rows = prepare_indicator_rows(symbol, candles, start_date, weight_lookup)
        except Exception as exc:  # safeguard: continue other symbols
            print(f"[WARN] skip symbol={safe_str(symbol)} due to error: {exc}")
            continue
        indicator_buffer.extend(ind_rows)
        summary_buffer.extend(sum_rows)

        # Flush in chunks
        if len(indicator_buffer) >= INSERT_CHUNK:
            cur.executemany(
                """
                insert into historical_indicator_scores
                (score_id, symbol, timestamp, timeframe, indicator_name, indicator_category,
                 indicator_params, value, signal, confidence)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                indicator_buffer,
            )
            indicator_buffer.clear()
        if len(summary_buffer) >= INSERT_CHUNK // 10:
            cur.executemany(
                """
                insert into historical_scores
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
                summary_buffer,
            )
            summary_buffer.clear()

        total_indicator_rows += len(ind_rows)
        total_summary_rows += len(sum_rows)

        if idx % 50 == 0:
            con.commit()
            print(f"Processed {idx}/{len(symbols)} symbols...")

    # Final flush
    if indicator_buffer:
        cur.executemany(
            """
            insert into historical_indicator_scores
            (score_id, symbol, timestamp, timeframe, indicator_name, indicator_category,
             indicator_params, value, signal, confidence)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            indicator_buffer,
        )
    if summary_buffer:
        cur.executemany(
            """
            insert into historical_scores
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
            summary_buffer,
        )
    con.commit()
    print(f"Inserted indicators: {total_indicator_rows}, summaries: {total_summary_rows}")
    con.close()


if __name__ == "__main__":
    main()
