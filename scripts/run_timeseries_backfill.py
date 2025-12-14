"""
Backfill full daily time-series of weights and analysis scores per symbol.

Steps per symbol:
1) For each day in [start_date, end_date], train weights (trend/momentum/volatility)
   on a rolling window and store in daily_weights.
2) Using that day's weights, run CompleteAnalysisPipeline on candles up to that day.
3) Store results in historical_scores (+ indicator scores).

This is intended for a pilot run on a limited set (e.g., 10 symbols + 5 indices).

Usage example:
  PYTHONPATH=src python scripts/run_timeseries_backfill.py \
    --pg-dsn postgresql://gravity:gravity@127.0.0.1:5544/tech_analysis \
    --symbol-limit 10 --index-limit 5 \
    --start-date 2024-06-01 --end-date 2024-12-01 \
    --lookback-days 240
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections import deque
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import psycopg
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from gravity_tech.core.domain.entities import Candle
from gravity_tech.database.historical_manager import (
    DailyWeightEntry,
    HistoricalScoreEntry,
    HistoricalScoreManager,
)
from gravity_tech.ml.complete_analysis_pipeline import CompleteAnalysisPipeline
from gravity_tech.ml.multi_horizon_analysis import MultiHorizonAnalyzer
from gravity_tech.ml.multi_horizon_momentum_analysis import MultiHorizonMomentumAnalyzer
from gravity_tech.ml.multi_horizon_volatility_analysis import MultiHorizonVolatilityAnalyzer
from gravity_tech.ml.multi_horizon_feature_extraction import MultiHorizonFeatureExtractor
from gravity_tech.ml.multi_horizon_volatility_features import MultiHorizonVolatilityFeatureExtractor
from gravity_tech.ml.multi_horizon_weights import HorizonWeights, MultiHorizonWeightLearner


def date_range(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def normalize_score(val: any) -> float | None:
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


def fetch_symbols(conn: psycopg.Connection, limit_symbols: int, limit_indices: int) -> list[str]:
    symbols: list[str] = []
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM tse_input.price_data ORDER BY symbol LIMIT %s", (limit_symbols,))
        symbols.extend([r[0] for r in cur.fetchall()])
        cur.execute("SELECT DISTINCT index_code FROM tse_input.market_indices ORDER BY index_code LIMIT %s", (limit_indices,))
        symbols.extend([f"IDX_{r[0]}" for r in cur.fetchall() if r[0]])
    return symbols


def fetch_candles(conn: psycopg.Connection, symbol: str, end_date: dt.date | None = None) -> list[Candle]:
    cutoff = end_date

    def to_candle(row, has_volume: bool) -> Candle | None:
        ts_date = row[0]
        if cutoff and ts_date > cutoff:
            return None
        ts = dt.datetime.combine(ts_date, dt.time.min, tzinfo=dt.timezone.utc)
        o, h, l, c = map(float, row[1:5])
        v = float(row[5]) if has_volume and len(row) > 5 and row[5] is not None else 0.0
        if h < max(o, c) or l > min(o, c):
            return None
        return Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=v)

    # Market indices (prefix IDX_)
    if symbol.upper().startswith("IDX_"):
        code = symbol[4:]
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trading_date, open, high, low, close
                FROM tse_input.market_indices
                WHERE index_code=%s
                ORDER BY trading_date
                """,
                (code,),
            )
            rows = cur.fetchall()
        candles: list[Candle] = []
        for r in rows:
            cndl = to_candle(r, has_volume=False)
            if cndl:
                candles.append(cndl)
        return candles

    # Sector indices (prefix SEC_)
    if symbol.upper().startswith("SEC_"):
        code = symbol[4:]
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trading_date, open, high, low, close
                FROM tse_input.sector_indices
                WHERE sector_code=%s
                ORDER BY trading_date
                """,
                (code,),
            )
            rows = cur.fetchall()
        candles: list[Candle] = []
        for r in rows:
            cndl = to_candle(r, has_volume=False)
            if cndl:
                candles.append(cndl)
        return candles

    with conn.cursor() as cur:
        cur.execute(
            "SELECT trading_date, adj_open, adj_high, adj_low, adj_close, adj_volume FROM tse_input.price_data WHERE symbol=%s ORDER BY trading_date",
            (symbol,),
        )
        rows = cur.fetchall()
    candles: list[Candle] = []
    for r in rows:
        cndl = to_candle(r, has_volume=True)
        if cndl:
            candles.append(cndl)
    return candles


def compute_targets(closes: np.ndarray, idx: int, horizons: list[int]) -> dict[str, float]:
    targets: dict[str, float] = {}
    for h in horizons:
        if idx + h >= len(closes):
            return {}
        future = closes[idx + h]
        current = closes[idx]
        pct_change = (future - current) / current * 100.0
        targets[f"return_{h}d"] = float(np.clip(pct_change / 50.0, -1.0, 1.0))
    return targets


def build_dataset_trend(candles: list[Candle], horizons: list[int], lookback: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    extractor = MultiHorizonFeatureExtractor(lookback_period=lookback, horizons=horizons)
    max_h = max(horizons)
    if len(candles) < lookback + max_h:
        return pd.DataFrame(), pd.DataFrame()

    closes = np.array([c.close for c in candles], dtype=float)
    window: deque[Candle] = deque(candles[:lookback], maxlen=lookback)
    X_rows: list[dict] = []
    y_rows: list[dict] = []

    for idx in range(lookback, len(candles) - max_h):
        try:
            features = {}
            features.update(extractor.extract_indicator_features(list(window)))
            features.update(extractor.extract_dimension_features(list(window)))
        except Exception:
            window.append(candles[idx])
            continue

        targets = compute_targets(closes, idx - 1, horizons)
        if not targets:
            window.append(candles[idx])
            continue

        X_rows.append(features)
        y_rows.append(targets)
        window.append(candles[idx])

    return pd.DataFrame(X_rows), pd.DataFrame(y_rows)


def build_dataset_volatility(candles: list[Candle], horizons: list[int], lookback: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    extractor = MultiHorizonVolatilityFeatureExtractor(lookback_period=lookback, horizons=horizons)
    return extractor.create_training_dataset(candles, horizons=horizons)


def train_weights(X: pd.DataFrame, Y: pd.DataFrame, horizons: list[str]) -> MultiHorizonWeightLearner:
    learner = MultiHorizonWeightLearner(horizons=horizons)
    learner.train(X, Y, verbose=False)
    return learner


def build_learner_from_daily(entries: list[DailyWeightEntry]) -> MultiHorizonWeightLearner | None:
    if not entries:
        return None
    horizons = [e.horizon for e in entries]
    learner = MultiHorizonWeightLearner(horizons=horizons)
    learner.feature_names = entries[0].feature_names
    learner.horizon_weights = {
        e.horizon: HorizonWeights(
            horizon=e.horizon,
            weights=e.feature_weights,
            metrics=e.metrics or {},
            confidence=e.confidence,
        )
        for e in entries
    }
    return learner


def ensure_weights_for_day(
    mgr: HistoricalScoreManager,
    symbol: str,
    current_date: dt.date,
    candles: list[Candle],
    horizons: list[str],
    horizon_ints: list[int],
    lookback: int,
):
    # Check existing per type
    need_types = []
    for t in ["trend", "momentum", "volatility"]:
        existing = mgr.load_daily_weights(current_date, t, symbol)
        if not existing:
            need_types.append(t)
    if not need_types:
        return

    for analysis_type in need_types:
        if len(candles) < lookback + max(horizon_ints):
            continue

        if analysis_type == "volatility":
            X, Y = build_dataset_volatility(candles, horizon_ints, lookback)
        else:
            X, Y = build_dataset_trend(candles, horizon_ints, lookback)
        if len(X) == 0 or len(Y) == 0:
            continue

        learner = train_weights(X, Y, horizons)
        for h in horizons:
            hw = learner.get_horizon_weights(h)
            if hw is None:
                continue
            mgr.save_daily_weights(
                DailyWeightEntry(
                    as_of_date=current_date,
                    analysis_type=analysis_type,
                    horizon=h,
                    feature_names=learner.feature_names or list(X.columns),
                    feature_weights=hw.weights,
                    metrics=hw.metrics,
                    confidence=hw.confidence,
                    symbol=symbol,
                )
            )


def main():
    parser = argparse.ArgumentParser(description="Backfill daily weights and scores per symbol.")
    parser.add_argument("--pg-dsn", required=True)
    parser.add_argument("--symbol-limit", type=int, default=10)
    parser.add_argument("--index-limit", type=int, default=5)
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma separated symbols to process (overrides limits if provided).",
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=240)
    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date)
    horizons = ["3d", "7d", "30d"]
    horizon_ints = [3, 7, 30]

    mgr = HistoricalScoreManager(args.pg_dsn)
    with psycopg.connect(args.pg_dsn, autocommit=True) as conn:
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        else:
            symbols = fetch_symbols(conn, args.symbol_limit, args.index_limit)
        print(f"Processing symbols: {symbols}")

        for sym in symbols:
            print(f"\n=== {sym} ===")
            candles_all = fetch_candles(conn, sym, end_date)
            if len(candles_all) < args.lookback_days + max(horizon_ints):
                print(f"Skip {sym}: not enough candles ({len(candles_all)})")
                continue

            for current_date in date_range(start_date, end_date):
                candles = [c for c in candles_all if c.timestamp.date() <= current_date]
                if len(candles) < args.lookback_days + max(horizon_ints):
                    continue

                # 1) ensure weights
                ensure_weights_for_day(mgr, sym, current_date, candles, horizons, horizon_ints, args.lookback_days)

                # 2) load weights
                trend_l = build_learner_from_daily(mgr.load_daily_weights(current_date, "trend", sym))
                mom_l = build_learner_from_daily(mgr.load_daily_weights(current_date, "momentum", sym))
                vol_l = build_learner_from_daily(mgr.load_daily_weights(current_date, "volatility", sym))
                if not (trend_l and mom_l and vol_l):
                    continue

                pipeline = CompleteAnalysisPipeline(
                    candles=candles,
                    trend_analyzer=MultiHorizonAnalyzer(trend_l),
                    momentum_analyzer=MultiHorizonMomentumAnalyzer(mom_l),
                    volatility_analyzer=MultiHorizonVolatilityAnalyzer(vol_l),
                    verbose=False,
                )
                try:
                    result = pipeline.analyze()
                except Exception as exc:
                    print(f"Fail analyze {sym} @ {current_date}: {exc}")
                    continue

                trend = result.trend_score
                momentum = result.momentum_score
                vol = result.volatility_score
                cycle = result.cycle_score
                sr = result.sr_score
                decision = result.decision
                last_close = candles[-1].close
                ts = dt.datetime.combine(current_date, dt.time.min, tzinfo=dt.timezone.utc)

                entry = HistoricalScoreEntry(
                    symbol=sym,
                    timestamp=ts,
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
                    mgr.save_score(entry, indicator_scores=indicator_rows)
                except Exception as exc:
                    print(f"Fail save {sym} @ {current_date}: {exc}")
                    continue

                print(f"Stored score {sym} @ {current_date}")


if __name__ == "__main__":
    main()
