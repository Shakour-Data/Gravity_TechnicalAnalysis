"""
Compute daily indicators and dimension scores from tse_input.price_data and
populate tech_analysis.daily_indicator_values + daily_dimension_scores.

Assumptions:
- Postgres backend (DatabaseManager with DATABASE_URL).
- Input prices come from tse_input.price_data (trading_date, adj_*).
- Timeframe fixed to '1d'; ts is trading_date at 00:00 UTC.

Run (example):
  PG_DSN=postgresql://gravity:gravity@localhost:5432/tech_analysis \\
  python scripts/compute_daily_scores.py --symbols ALL --lookback-days 365
"""

from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta, timezone

import numpy as np
import pandas as pd
from gravity_tech.database.database_manager import DatabaseManager, DatabaseType
from database import TSEDatabaseConnector
from config import TSE_DB_FILE


def _ts_from_date(d: datetime | pd.Timestamp) -> datetime:
    if isinstance(d, pd.Timestamp):
        d = d.to_pydatetime()
    return datetime.combine(d.date(), time.min, tzinfo=timezone.utc)


def _compute_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a small set of indicators per row (daily)."""
    out = df.copy()
    out["return"] = out["close"].pct_change()

    out["sma_20"] = out["close"].rolling(20, min_periods=5).mean()
    out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["rsi_14"] = _rsi(out["close"], window=14)
    macd_line, macd_signal, macd_hist = _macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist
    out["atr_14"] = _atr(out, window=14)

    # Scores/signals
    out["trend_score"] = (out["ema_20"] - out["sma_20"]) / out["sma_20"]
    out["momentum_score"] = out["rsi_14"].apply(lambda x: (x - 50) / 50 if pd.notna(x) else 0)
    out["volatility_score"] = out["atr_14"] / out["close"]
    out["volume_score"] = (out["volume"] - out["volume"].rolling(20, min_periods=5).mean()) / (
        out["volume"].rolling(20, min_periods=5).std().replace(0, np.nan)
    )

    return out


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def compute_and_store(symbols: list[str], lookback_days: int, manager: DatabaseManager, tse_source: str | None = None):
    connector = TSEDatabaseConnector(tse_source or TSE_DB_FILE)
    conn = manager.get_connection()
    placeholder = manager.get_sql_placeholder()

    def clean_num(val):
        if val is None:
            return None
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        if pd.isna(f):
            return None
        return f

    # Prepare insert/upsert statements for daily tables
    dim_insert = f"""
        INSERT INTO tech_analysis.daily_dimension_scores
            (symbol, timeframe, ts, dimension, score, confidence, weight, signal, features)
        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder},
                {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
        ON CONFLICT (symbol, timeframe, ts, dimension) DO UPDATE SET
            score = EXCLUDED.score,
            confidence = EXCLUDED.confidence,
            weight = EXCLUDED.weight,
            signal = EXCLUDED.signal,
            features = EXCLUDED.features,
            updated_at = NOW()
    """

    ind_insert = f"""
        INSERT INTO tech_analysis.daily_indicator_values
            (symbol, timeframe, ts, dimension, indicator_name, indicator_params,
             value, score, signal, confidence, weight, source_window)
        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
                {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
                {placeholder}, {placeholder})
        ON CONFLICT (symbol, timeframe, ts, dimension, indicator_name) DO UPDATE SET
            value = EXCLUDED.value,
            score = EXCLUDED.score,
            signal = EXCLUDED.signal,
            confidence = EXCLUDED.confidence,
            weight = EXCLUDED.weight,
            indicator_params = EXCLUDED.indicator_params
    """

    for sym in symbols:
        raw = connector.fetch_price_data(sym)
        if not raw:
            continue

        df = pd.DataFrame(raw)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        df = df[df["timestamp"] >= cutoff]
        if df.empty:
            continue

        df["close"] = df["close"].astype(float)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)

        df = _compute_basic_indicators(df)

        dim_rows = []
        ind_rows = []
        for _, row in df.iterrows():
            ts_val = _ts_from_date(row["timestamp"])
            # Indicators per dimension (sample set)
            # Trend
            ind_rows.append((sym, "1d", ts_val, "trend", "sma_20", None, clean_num(row["sma_20"]), clean_num(row["trend_score"]), None, 0.5, None, 20))
            ind_rows.append((sym, "1d", ts_val, "trend", "ema_20", None, clean_num(row["ema_20"]), clean_num(row["trend_score"]), None, 0.5, None, 20))
            # Momentum
            ind_rows.append((sym, "1d", ts_val, "momentum", "rsi_14", None, clean_num(row["rsi_14"]), clean_num(row["momentum_score"]), None, 0.5, None, 14))
            ind_rows.append((sym, "1d", ts_val, "momentum", "macd", None, clean_num(row["macd"]), None, None, 0.5, None, None))
            ind_rows.append((sym, "1d", ts_val, "momentum", "macd_signal", None, clean_num(row["macd_signal"]), None, None, 0.5, None, None))
            ind_rows.append((sym, "1d", ts_val, "momentum", "macd_hist", None, clean_num(row["macd_hist"]), None, None, 0.5, None, None))
            # Volatility
            ind_rows.append((sym, "1d", ts_val, "volatility", "atr_14", None, clean_num(row["atr_14"]), clean_num(row["volatility_score"]), None, 0.5, None, 14))
            vol_std_val = df["return"].rolling(20, min_periods=5).std().loc[_]
            ind_rows.append((sym, "1d", ts_val, "volatility", "return_std_20", None,
                             clean_num(vol_std_val),
                             clean_num(row["volatility_score"]), None, 0.5, None, 20))
            # Volume
            ind_rows.append((sym, "1d", ts_val, "volume", "vol_zscore_20", None, clean_num(row["volume_score"]), clean_num(row["volume_score"]), None, 0.5, None, 20))

            # Dimension scores (simple aggregates for now)
            dim_rows.extend([
                (sym, "1d", ts_val, "trend", clean_num(row["trend_score"]), 0.5, None, "BULLISH" if clean_num(row["trend_score"]) and clean_num(row["trend_score"]) > 0 else "BEARISH" if clean_num(row["trend_score"]) and clean_num(row["trend_score"]) < 0 else "NEUTRAL", None),
                (sym, "1d", ts_val, "momentum", clean_num(row["momentum_score"]), 0.5, None, "BULLISH" if clean_num(row["momentum_score"]) and clean_num(row["momentum_score"]) > 0 else "BEARISH" if clean_num(row["momentum_score"]) and clean_num(row["momentum_score"]) < 0 else "NEUTRAL", None),
                (sym, "1d", ts_val, "volatility", clean_num(row["volatility_score"]), 0.5, None, "HIGH" if clean_num(row["volatility_score"]) and clean_num(row["volatility_score"]) > 0.05 else "NORMAL", None),
                (sym, "1d", ts_val, "volume", clean_num(row["volume_score"]), 0.5, None, "HIGH" if clean_num(row["volume_score"]) and clean_num(row["volume_score"]) > 0 else "LOW", None),
            ])

        with conn.cursor() as cur:
            if dim_rows:
                cur.executemany(dim_insert, dim_rows)
            if ind_rows:
                cur.executemany(ind_insert, ind_rows)
            conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Compute daily indicators and dimension scores into Postgres.")
    parser.add_argument("--symbols", default="AUTO", help="Comma-separated symbols or AUTO to read from price_data.")
    parser.add_argument("--lookback-days", type=int, default=365, help="Lookback window in days.")
    parser.add_argument("--pg-dsn", default=None, help="Override DATABASE_URL for this run.")
    parser.add_argument("--tse-dsn", default=None, help="Override TSE db path/DSN.")
    args = parser.parse_args()

    manager = DatabaseManager(connection_string=args.pg_dsn, auto_setup=False)
    if manager.db_type != DatabaseType.POSTGRESQL:
        raise SystemExit("This script requires PostgreSQL backend.")

    if args.symbols.upper() == "AUTO":
        # Pull symbol list from tse_input.price_data
        conn = manager.get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT symbol FROM tse_input.price_data")
            symbols = [r[0] for r in cur.fetchall()]
        manager.release_connection(conn)
    else:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    compute_and_store(symbols, lookback_days=args.lookback_days, manager=manager, tse_source=args.tse_dsn)
    print(f"✅ Computed daily indicators/dimensions for {len(symbols)} symbols.")


if __name__ == "__main__":
    main()
