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
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from config import TSE_DB_FILE
from database import TSEDatabaseConnector
from gravity_tech.database.database_manager import DatabaseManager, DatabaseType

DEFAULT_SQLITE_PATH = Path("data") / "TechAnalysis.db"


def _ts_from_date(d: datetime | pd.Timestamp) -> datetime:
    if isinstance(d, pd.Timestamp):
        d = d.to_pydatetime()
    return datetime.combine(d.date(), time.min, tzinfo=UTC)


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


def _schema_prefix(manager: DatabaseManager) -> str:
    return "tech_analysis." if manager.db_type == DatabaseType.POSTGRESQL else ""


def ensure_daily_tables(manager: DatabaseManager) -> None:
    """Create daily tables for both SQLite and PostgreSQL backends."""
    if manager.db_type == DatabaseType.POSTGRESQL:
        statements = [
            "CREATE SCHEMA IF NOT EXISTS tech_analysis;",
            """
            CREATE TABLE IF NOT EXISTS tech_analysis.daily_dimension_scores (
                id BIGSERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                ts TIMESTAMPTZ NOT NULL,
                dimension VARCHAR(50) NOT NULL,
                score NUMERIC(12,6),
                confidence NUMERIC(6,4),
                weight NUMERIC(6,4),
                signal VARCHAR(20),
                features JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                CONSTRAINT uq_daily_dim UNIQUE (symbol, timeframe, ts, dimension)
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_daily_dim_symbol_ts ON tech_analysis.daily_dimension_scores(symbol, ts);",
            "CREATE INDEX IF NOT EXISTS idx_daily_dim_dimension ON tech_analysis.daily_dimension_scores(dimension);",
            """
            CREATE TABLE IF NOT EXISTS tech_analysis.daily_indicator_values (
                id BIGSERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                ts TIMESTAMPTZ NOT NULL,
                dimension VARCHAR(50) NOT NULL,
                indicator_name VARCHAR(100) NOT NULL,
                indicator_params JSONB,
                value NUMERIC(20,10),
                score NUMERIC(12,6),
                signal VARCHAR(20),
                confidence NUMERIC(6,4),
                weight NUMERIC(6,4),
                source_window INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                CONSTRAINT uq_daily_indicator UNIQUE (symbol, timeframe, ts, dimension, indicator_name)
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_daily_ind_symbol_ts ON tech_analysis.daily_indicator_values(symbol, ts);",
            "CREATE INDEX IF NOT EXISTS idx_daily_ind_dimension ON tech_analysis.daily_indicator_values(dimension);",
            "CREATE INDEX IF NOT EXISTS idx_daily_ind_indicator ON tech_analysis.daily_indicator_values(indicator_name);",
        ]
    else:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS daily_dimension_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                ts TEXT NOT NULL,
                dimension TEXT NOT NULL,
                score REAL,
                confidence REAL,
                weight REAL,
                signal TEXT,
                features TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, ts, dimension)
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_daily_dim_symbol_ts ON daily_dimension_scores(symbol, ts);",
            "CREATE INDEX IF NOT EXISTS idx_daily_dim_dimension ON daily_dimension_scores(dimension);",
            """
            CREATE TABLE IF NOT EXISTS daily_indicator_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                ts TEXT NOT NULL,
                dimension TEXT NOT NULL,
                indicator_name TEXT NOT NULL,
                indicator_params TEXT,
                value REAL,
                score REAL,
                signal TEXT,
                confidence REAL,
                weight REAL,
                source_window INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, ts, dimension, indicator_name)
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_daily_ind_symbol_ts ON daily_indicator_values(symbol, ts);",
            "CREATE INDEX IF NOT EXISTS idx_daily_ind_dimension ON daily_indicator_values(dimension);",
            "CREATE INDEX IF NOT EXISTS idx_daily_ind_indicator ON daily_indicator_values(indicator_name);",
        ]

    for stmt in statements:
        manager.execute_query(stmt)


def compute_and_store(
    symbols: list[str],
    lookback_days: int,
    manager: DatabaseManager,
    tse_source: str | None = None,
    progress_cb=None,
):
    connector = TSEDatabaseConnector(tse_source or TSE_DB_FILE)
    conn = manager.get_connection()
    placeholder = manager.get_sql_placeholder()
    prefix = _schema_prefix(manager)
    now_func = "NOW()" if manager.db_type == DatabaseType.POSTGRESQL else "CURRENT_TIMESTAMP"

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
        INSERT INTO {prefix}daily_dimension_scores
            (symbol, timeframe, ts, dimension, score, confidence, weight, signal, features)
        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder},
                {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
        ON CONFLICT (symbol, timeframe, ts, dimension) DO UPDATE SET
            score = EXCLUDED.score,
            confidence = EXCLUDED.confidence,
            weight = EXCLUDED.weight,
            signal = EXCLUDED.signal,
            features = EXCLUDED.features,
            updated_at = {now_func}
    """

    ind_insert = f"""
        INSERT INTO {prefix}daily_indicator_values
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

    total = len(symbols)
    for idx, sym in enumerate(symbols, start=1):
        raw = connector.fetch_price_data(sym)
        wrote_rows = False
        if not raw:
            if progress_cb:
                progress_cb(idx, total, sym, wrote_rows)
            continue

        df = pd.DataFrame(raw)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
        df = df[df["timestamp"] >= cutoff]
        if df.empty:
            if progress_cb:
                progress_cb(idx, total, sym, wrote_rows)
            continue

        df["close"] = df["close"].astype(float)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)

        df = _compute_basic_indicators(df)

        dim_rows = []
        ind_rows = []
        rolling_std_20 = df["return"].rolling(20, min_periods=5).std()
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
            # Use row's index to access rolling_std_20
            vol_std_val = rolling_std_20.get(row.name, None)
            ind_rows.append((sym, "1d", ts_val, "volatility", "return_std_20", None,
                             clean_num(vol_std_val),
                             clean_num(row["volatility_score"]), None, 0.5, None, 20))
            # Volume
            ind_rows.append((sym, "1d", ts_val, "volume", "vol_zscore_20", None, clean_num(row["volume_score"]), clean_num(row["volume_score"]), None, 0.5, None, 20))

            # Dimension scores (simple aggregates for now)
            trend_score = clean_num(row["trend_score"])
            momentum_score = clean_num(row["momentum_score"])
            volatility_score = clean_num(row["volatility_score"])
            volume_score = clean_num(row["volume_score"])
            dim_rows.extend([
                (sym, "1d", ts_val, "trend", trend_score, 0.5, None,
                 "BULLISH" if trend_score is not None and trend_score > 0 else "BEARISH" if trend_score is not None and trend_score < 0 else "NEUTRAL", None),
                (sym, "1d", ts_val, "momentum", momentum_score, 0.5, None,
                 "BULLISH" if momentum_score is not None and momentum_score > 0 else "BEARISH" if momentum_score is not None and momentum_score < 0 else "NEUTRAL", None),
                (sym, "1d", ts_val, "volatility", volatility_score, 0.5, None,
                 "HIGH" if volatility_score is not None and volatility_score > 0.05 else "NORMAL", None),
                (sym, "1d", ts_val, "volume", volume_score, 0.5, None,
                 "HIGH" if volume_score is not None and volume_score > 0 else "LOW", None),
            ])

        cur = conn.cursor()
        try:
            if dim_rows:
                cur.executemany(dim_insert, dim_rows)
            if ind_rows:
                cur.executemany(ind_insert, ind_rows)
            if dim_rows or ind_rows:
                wrote_rows = True
            conn.commit()
        finally:
            cur.close()
        if progress_cb:
            progress_cb(idx, total, sym, wrote_rows)
    if manager.db_type == DatabaseType.POSTGRESQL:
        manager.release_connection(conn)


def main():
    parser = argparse.ArgumentParser(description="Compute daily indicators and dimension scores (PostgreSQL or SQLite).")
    parser.add_argument("--symbols", default="AUTO", help="Comma-separated symbols or AUTO to read from price_data.")
    parser.add_argument("--lookback-days", type=int, default=365, help="Lookback window in days.")
    parser.add_argument("--pg-dsn", default=None, help="Override DATABASE_URL for this run.")
    parser.add_argument("--db-type", choices=["auto", "postgresql", "sqlite"], default="auto", help="Select backend; default auto-detect.")
    parser.add_argument("--sqlite-path", default=None, help="Path to SQLite DB when db-type=sqlite (default: data/TechAnalysis.db).")
    parser.add_argument("--tse-dsn", default=None, help="Override TSE db path/DSN.")
    parser.add_argument("--max-symbols", type=int, default=0, help="Cap number of symbols to process (0 = all).")
    args = parser.parse_args()

    db_type = None
    if args.db_type == "postgresql":
        db_type = DatabaseType.POSTGRESQL
    elif args.db_type == "sqlite":
        db_type = DatabaseType.SQLITE

    sqlite_path = args.sqlite_path or DEFAULT_SQLITE_PATH
    manager = DatabaseManager(
        db_type=db_type,
        connection_string=args.pg_dsn,
        sqlite_path=str(sqlite_path),
        auto_setup=True,
        allow_fallback=True,
    )

    ensure_daily_tables(manager)
    tse_connector = TSEDatabaseConnector(args.tse_dsn or TSE_DB_FILE)

    if args.symbols.upper() == "AUTO":
        symbols = tse_connector.list_symbols(limit=100000, min_rows=120)
    else:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    if args.max_symbols and len(symbols) > args.max_symbols:
        symbols = symbols[: args.max_symbols]

    backend_str = manager.db_type.value if manager.db_type is not None else "unknown"
    total_symbols = len(symbols)
    print(f"[daily] Starting daily computation for {total_symbols} symbols (backend: {backend_str}).")

    def progress_cb(idx: int, total: int, symbol: str, wrote: bool) -> None:
        pct = (idx / total * 100) if total else 100.0
        state = "wrote" if wrote else "skipped"
        print(f"[daily] {pct:5.1f}% ({idx}/{total}) {state}: {symbol}")

    compute_and_store(
        symbols,
        lookback_days=args.lookback_days,
        manager=manager,
        tse_source=args.tse_dsn,
        progress_cb=progress_cb,
    )
    print(f"✅ Computed daily indicators/dimensions for {total_symbols} symbols (backend: {backend_str}).")


if __name__ == "__main__":
    main()
