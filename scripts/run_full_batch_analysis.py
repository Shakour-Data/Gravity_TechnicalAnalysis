"""
Run full analysis batch for a set of symbols and store results in historical tables.

This script:
- Loads OHLCV for symbols (price_data) and indices (IDX_*, SEC_* from market_indices/sector_indices)
- Runs the complete analysis pipeline
- Normalizes all scores to [-100, 100]
- Persists into historical_scores + historical_indicator_scores

Usage:
  PYTHONPATH=src python scripts/run_full_batch_analysis.py \
    --pg-dsn postgresql://gravity:gravity@127.0.0.1:5544/tech_analysis \
    --limit 60 \
    --lookback-days 365

Notes:
- Requires at least 120 candles per symbol for the 5D pipeline.
"""

from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta, timezone
from typing import Any
import sys

import numpy as np
import psycopg
from pathlib import Path

# Ensure UTF-8 output even on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from gravity_tech.core.domain.entities import Candle
from gravity_tech.database.historical_manager import (
    HistoricalScoreEntry,
    HistoricalScoreManager,
    DailyWeightEntry,
)
from gravity_tech.ml.pipeline_factory import build_pipeline_from_weights
from gravity_tech.ml.multi_horizon_weights import HorizonWeights, MultiHorizonWeightLearner
from gravity_tech.ml.multi_horizon_analysis import MultiHorizonAnalyzer
from gravity_tech.ml.multi_horizon_momentum_analysis import MultiHorizonMomentumAnalyzer
from gravity_tech.ml.multi_horizon_volatility_analysis import MultiHorizonVolatilityAnalyzer
from gravity_tech.ml.complete_analysis_pipeline import CompleteAnalysisPipeline

# Ensure UTF-8 output even on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MIN_CANDLES = 120


def normalize_score(val: Any) -> float | None:
    """Normalize numeric score to [-100, 100]."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if -1.0 <= f <= 1.0:
        return float(f * 100.0)
    return float(np.clip(f, -100.0, 100.0))


def build_learner_from_daily(entries: list[DailyWeightEntry]) -> MultiHorizonWeightLearner | None:
    if not entries:
        return None
    horizons = [e.horizon for e in entries]
    learner = MultiHorizonWeightLearner(horizons=horizons)
    # assume same feature_names for all entries
    learner.feature_names = entries[0].feature_names
    horizon_weights: dict[str, HorizonWeights] = {}
    for e in entries:
        horizon_weights[e.horizon] = HorizonWeights(
            horizon=e.horizon,
            weights=e.feature_weights,
            metrics=e.metrics or {},
            confidence=e.confidence,
        )
    learner.horizon_weights = horizon_weights
    return learner


def fetch_candles_pg(conn: psycopg.Connection, symbol: str, lookback_days: int) -> list[Candle]:
    """Fetch candles for symbols, market indices (IDX_), and sector indices (SEC_)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    def to_candle(row, has_volume: bool) -> Candle | None:
        ts = datetime.combine(row[0], time.min, tzinfo=timezone.utc)
        o, h, l, c = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        v = float(row[5]) if has_volume and len(row) > 5 and row[5] is not None else 0.0
        # drop invalid OHLC
        if h < max(o, c) or l > min(o, c):
            return None
        return Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=v)

    if symbol.upper().startswith("IDX_"):
        code = symbol[4:]
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trading_date, open, high, low, close
                FROM tse_input.market_indices
                WHERE index_code = %s AND trading_date >= %s
                ORDER BY trading_date ASC
                """,
                (code, cutoff.date()),
            )
            rows = cur.fetchall()
        candles: list[Candle] = []
        for r in rows:
            if None in r[1:5]:
                continue
            cndl = to_candle(r, has_volume=False)
            if cndl:
                candles.append(cndl)
        return candles

    if symbol.upper().startswith("SEC_"):
        code = symbol[4:]
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trading_date, open, high, low, close
                FROM tse_input.sector_indices
                WHERE sector_code = %s AND trading_date >= %s
                ORDER BY trading_date ASC
                """,
                (code, cutoff.date()),
            )
            rows = cur.fetchall()
        candles: list[Candle] = []
        for r in rows:
            if None in r[1:5]:
                continue
            cndl = to_candle(r, has_volume=False)
            if cndl:
                candles.append(cndl)
        return candles

    # price_data symbols
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT trading_date, adj_open, adj_high, adj_low, adj_close, adj_volume
            FROM tse_input.price_data
            WHERE symbol = %s AND trading_date >= %s
            ORDER BY trading_date ASC
            """,
            (symbol, cutoff.date()),
        )
        rows = cur.fetchall()
    candles: list[Candle] = []
    for r in rows:
        if None in r[1:5]:
            continue
        cndl = to_candle(r, has_volume=True)
        if cndl:
            candles.append(cndl)
    return candles


def make_indicator_rows(result) -> list[dict]:
    """Build indicator score rows for historical_indicator_scores."""
    rows = []
    dims = result.decision.dimensions if result.decision else {}
    for name, dim in dims.items():
        rows.append(
            {
                "name": name,
                "category": "dimension",
                "params": {},
                "score": normalize_score(dim.volume_adjusted_score),
                "confidence": dim.confidence,
                "signal": getattr(dim.signal, "value", str(dim.signal)),
                "raw_value": dim.score,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run full analysis batch and persist historical scores.")
    parser.add_argument("--pg-dsn", required=True, help="Postgres DSN, e.g. postgresql://user:pass@host:port/db")
    parser.add_argument("--limit", type=int, default=60, help="Max symbols from price_data to process.")
    parser.add_argument("--lookback-days", type=int, default=365, help="Lookback window for candles.")
    parser.add_argument(
        "--weights-dir",
        default="ml_models/multi_horizon",
        help="Directory containing indicator_weights_btcusdt.json/pkl and dimension_weights_btcusdt.json/pkl",
    )
    parser.add_argument(
        "--vol-weights",
        default="models/volatility/volatility_weights.json",
        help="Path to volatility_weights.json (trained via train_multi_horizon_volatility.py)",
    )
    parser.add_argument(
        "--use-daily-weights",
        action="store_true",
        help="Load weights from daily_weights table (if available) based on last candle date.",
    )
    parser.add_argument(
        "--weights-symbol",
        default="GLOBAL",
        help="Symbol key used when storing daily_weights (default: GLOBAL).",
    )
    args = parser.parse_args()

    with psycopg.connect(args.pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT symbol FROM tse_input.price_data ORDER BY symbol LIMIT %s", (args.limit,))
            symbols = [r[0] for r in cur.fetchall()]
            cur.execute("SELECT DISTINCT index_code FROM tse_input.market_indices ORDER BY index_code")
            idx = ["IDX_" + r[0] for r in cur.fetchall() if r[0]]
            cur.execute("SELECT DISTINCT sector_code FROM tse_input.sector_indices ORDER BY sector_code")
            sec = ["SEC_" + str(r[0]) for r in cur.fetchall() if r[0] is not None]

        all_symbols = symbols + idx + sec
        print(f"Processing {len(all_symbols)} symbols (price={len(symbols)}, idx={len(idx)}, sec={len(sec)})")

        hist_manager = HistoricalScoreManager(args.pg_dsn)

        for sym in all_symbols:
            candles = fetch_candles_pg(conn, sym, args.lookback_days)
            if len(candles) < MIN_CANDLES:
                print(f"SKIP {sym}: not enough candles ({len(candles)})")
                continue

            try:
                pipeline = None
                if args.use_daily_weights:
                    as_of_date = candles[-1].timestamp.date()
                    # Try symbol-specific weights first; fallback به وزن GLOBAL یا وزن اعلام‌شده
                    trend_entries = hist_manager.load_daily_weights(as_of_date, "trend", sym)
                    momentum_entries = hist_manager.load_daily_weights(as_of_date, "momentum", sym)
                    volatility_entries = hist_manager.load_daily_weights(as_of_date, "volatility", sym)
                    if not trend_entries:
                        trend_entries = hist_manager.load_daily_weights(as_of_date, "trend", args.weights_symbol)
                    if not momentum_entries:
                        momentum_entries = hist_manager.load_daily_weights(as_of_date, "momentum", args.weights_symbol)
                    if not volatility_entries:
                        volatility_entries = hist_manager.load_daily_weights(as_of_date, "volatility", args.weights_symbol)

                    trend_learner = build_learner_from_daily(trend_entries)
                    momentum_learner = build_learner_from_daily(momentum_entries)
                    volatility_learner = build_learner_from_daily(volatility_entries)

                    if trend_learner and momentum_learner and volatility_learner:
                        pipeline = CompleteAnalysisPipeline(
                            candles=candles,
                            trend_analyzer=MultiHorizonAnalyzer(trend_learner),
                            momentum_analyzer=MultiHorizonMomentumAnalyzer(momentum_learner),
                            volatility_analyzer=MultiHorizonVolatilityAnalyzer(volatility_learner),
                            verbose=False,
                        )

                if pipeline is None:
                    trend_w = Path(args.weights_dir) / "indicator_weights_btcusdt.json"
                    momentum_w = Path(args.weights_dir) / "indicator_weights_btcusdt.json"
                    volatility_w = Path(args.vol_weights)
                    pipeline = build_pipeline_from_weights(
                        candles,
                        trend_weights_path=str(trend_w),
                        momentum_weights_path=str(momentum_w),
                        volatility_weights_path=str(volatility_w),
                        verbose=False,
                    )

                result = pipeline.analyze()
            except Exception as exc:
                print(f"FAIL {sym}: {exc}")
                continue

            # Build HistoricalScoreEntry
            trend = result.trend_score
            momentum = result.momentum_score
            vol = result.volatility_score
            cycle = result.cycle_score
            sr = result.sr_score
            decision = result.decision
            last_close = candles[-1].close

            entry = HistoricalScoreEntry(
                symbol=sym,
                timestamp=datetime.now(timezone.utc),
                timeframe="1d",
                trend_score=normalize_score(trend.score),
                trend_confidence=trend.accuracy,
                momentum_score=normalize_score(momentum.score),
                momentum_confidence=momentum.accuracy,
                combined_score=normalize_score(decision.final_score),
                combined_confidence=decision.final_confidence,
                trend_weight=0.5,
                momentum_weight=0.5,
                trend_signal=getattr(trend.signal, "value", str(trend.signal)),
                momentum_signal=getattr(momentum.signal, "value", str(momentum.signal)),
                combined_signal=decision.final_signal.value,
                recommendation=decision.recommendation,
                action=decision.final_signal.value,
                price_at_analysis=last_close,
                volume_score=0.0,
                volatility_score=normalize_score(vol.score),
                cycle_score=normalize_score(cycle.score),
                support_resistance_score=normalize_score(sr.score),
                raw_data=result.to_dict(),
            )

            indicator_rows = make_indicator_rows(result)

            try:
                hist_manager.save_score(
                    entry,
                    horizon_scores=None,
                    indicator_scores=indicator_rows,
                    patterns=None,
                    volume_analysis=None,
                    price_targets=None,
                )
                print(f"Saved {sym}")
            except Exception as exc:
                print(f"Failed save for {sym}: {exc}")


if __name__ == "__main__":
    main()
