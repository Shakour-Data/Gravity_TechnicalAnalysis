"""
Generate daily ML weights per horizon using a rolling lookback window and store
them in the database (daily_weights table).

Usage example:
  PYTHONPATH=src python scripts/build_daily_weights.py \
    --pg-dsn postgresql://gravity:gravity@127.0.0.1:5544/tech_analysis \
    --analysis-type trend \
    --symbol آبادا \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --lookback-days 240
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections import deque
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import psycopg

from gravity_tech.core.domain.entities import Candle
from gravity_tech.database.historical_manager import DailyWeightEntry, HistoricalScoreManager
from gravity_tech.ml.multi_horizon_feature_extraction import MultiHorizonFeatureExtractor
from gravity_tech.ml.multi_horizon_volatility_features import MultiHorizonVolatilityFeatureExtractor
from gravity_tech.ml.multi_horizon_weights import HorizonWeights, MultiHorizonWeightLearner


def fetch_candles(conn: psycopg.Connection, symbol: str, end_date: dt.date) -> list[Candle]:
    """Fetch all candles up to end_date (inclusive) from tse_input.price_data."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT trading_date, adj_open, adj_high, adj_low, adj_close, adj_volume
            FROM tse_input.price_data
            WHERE symbol = %s AND trading_date <= %s
            ORDER BY trading_date ASC
            """,
            (symbol, end_date),
        )
        rows = cur.fetchall()
    candles: list[Candle] = []
    for r in rows:
        if None in r[1:5]:
            continue
        cdate = r[0]
        ts = dt.datetime.combine(cdate, dt.time.min, tzinfo=dt.timezone.utc)
        o, h, l, c = map(float, r[1:5])
        v = float(r[5]) if r[5] is not None else 0.0
        if h < max(o, c) or l > min(o, c):
            continue
        candles.append(Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=v))
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
        features = {}
        try:
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


def date_range(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description="Build daily ML weights and store in DB.")
    parser.add_argument("--pg-dsn", required=True, help="Postgres DSN, e.g., postgresql://user:pass@host:port/db")
    parser.add_argument("--symbol", default=None, help="Symbol to use for training (default: first symbol in DB)")
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Generate weights for all symbols in price_data (one by one).",
    )
    parser.add_argument("--analysis-type", choices=["trend", "momentum", "volatility"], default="trend")
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=240, help="Rolling lookback window length")
    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date)
    horizons = ["3d", "7d", "30d"]
    horizon_ints = [int(h.replace("d", "")) for h in horizons]

    mgr = HistoricalScoreManager(args.pg_dsn)
    with psycopg.connect(args.pg_dsn, autocommit=True) as conn:
        # prepare symbol list
        symbols: list[str] = []
        if args.all_symbols:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT symbol FROM tse_input.price_data ORDER BY symbol")
                symbols = [r[0] for r in cur.fetchall()]
        else:
            symbol = args.symbol
            if symbol is None:
                with conn.cursor() as cur:
                    cur.execute("SELECT DISTINCT symbol FROM tse_input.price_data ORDER BY symbol LIMIT 1")
                    row = cur.fetchone()
                    if not row:
                        raise RuntimeError("No symbols found in price_data")
                    symbol = row[0]
            symbols = [symbol]

        for symbol in symbols:
            print(f"Using symbol for training: {symbol}")
            candles_all = fetch_candles(conn, symbol, end_date)
            if not candles_all:
                print(f"Skip {symbol}: no candles")
                continue

            for current_date in date_range(start_date, end_date):
                candles = [c for c in candles_all if c.timestamp.date() <= current_date]
                if len(candles) < args.lookback_days + max(horizon_ints):
                    continue

                if args.analysis_type == "volatility":
                    X, Y = build_dataset_volatility(candles, horizon_ints, args.lookback_days)
                else:
                    X, Y = build_dataset_trend(candles, horizon_ints, args.lookback_days)

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
                            analysis_type=args.analysis_type,
                            horizon=h,
                            feature_names=learner.feature_names or list(X.columns),
                            feature_weights=hw.weights,
                            metrics=hw.metrics,
                            confidence=hw.confidence,
                            symbol=symbol,
                        )
                    )
                print(f"Stored weights for {symbol} @ {current_date} ({args.analysis_type})")


if __name__ == "__main__":
    main()
