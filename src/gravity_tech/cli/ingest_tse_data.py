"""
Ingest real TSE data into the project's operational database.

This CLI pulls OHLCV data from the external TSE SQLite database
(`TSE_DB_FILE` from `src.config`) and writes summarized analytics into the
project database managed by `DatabaseManager` (tool_performance_history +
historical_scores tables).

Usage examples:
  python -m gravity_tech.cli.ingest_tse_data --max-symbols 25 --limit 365 --reset
  python -m gravity_tech.cli.ingest_tse_data --symbols IRO1ABAD0002,IRO1FOLD0001
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from math import sqrt

import pandas as pd
from gravity_tech.database.database_manager import DatabaseManager
from gravity_tech.database.tse_data_source import tse_data_source


def _compute_indicators(df: pd.DataFrame) -> dict:
    """Compute basic technical indicators for storage."""
    # RSI (14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    last_rsi = float(rsi.iloc[-1]) if not rsi.empty else None

    # MACD (12,26,9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    last_macd = float(macd.iloc[-1]) if not macd.empty else None
    last_signal = float(signal.iloc[-1]) if not signal.empty else None
    last_hist = float(hist.iloc[-1]) if not hist.empty else None

    # 50/200 SMA crossover
    sma_short = df["close"].rolling(window=50, min_periods=50).mean()
    sma_long = df["close"].rolling(window=200, min_periods=200).mean()
    if not sma_short.empty and not sma_long.empty:
        last_cross = "golden" if sma_short.iloc[-1] > sma_long.iloc[-1] else "death"
    else:
        last_cross = None

    return {
        "rsi": last_rsi,
        "macd": last_macd,
        "macd_signal": last_signal,
        "macd_hist": last_hist,
        "sma_cross": last_cross,
    }


def summarize_candles(candles: list[dict]) -> dict:
    """Compute signals + indicators from OHLCV data."""
    df = pd.DataFrame(candles)
    if df.empty:
        raise ValueError("No candle data received")

    df = df.sort_values("timestamp")
    df["close"] = df["close"].astype(float)
    df["return"] = df["close"].pct_change()
    df["volume"] = df["volume"].astype(float)

    trend_change = (df["close"].iloc[-1] - df["close"].iloc[0]) / max(
        1e-9, df["close"].iloc[0]
    )
    vol = df["return"].std() * sqrt(252) if not df["return"].empty else 0.0
    vol_level = float(vol) if pd.notna(vol) else 0.0

    # Regime heuristics
    if trend_change > 0.08:
        regime = "trending_bullish"
        prediction = "bullish"
    elif trend_change < -0.08:
        regime = "trending_bearish"
        prediction = "bearish"
    else:
        regime = "range"
        prediction = "neutral"

    confidence = min(0.99, max(0.05, abs(trend_change) * 2 + vol_level * 0.2))

    # Simple volume profile
    vol_profile = "high" if df["volume"].mean() > df["volume"].median() else "normal"

    summary = {
        "regime": regime,
        "prediction": prediction,
        "confidence": float(confidence),
        "volatility": vol_level,
        "trend_strength": float(trend_change),
        "volume_profile": vol_profile,
        "price_at_analysis": float(df["close"].iloc[-1]),
        "last_timestamp": df["timestamp"].iloc[-1],
    }
    summary.update(_compute_indicators(df))
    return summary


def insert_historical_score(manager: DatabaseManager, symbol: str, timeframe: str, summary: dict) -> int:
    """Insert a row into historical_scores and return its ID."""
    placeholders = manager.get_sql_placeholder()
    phs = ", ".join([placeholders] * 24)
    query = f"""
        INSERT INTO historical_scores (
            symbol, timestamp, timeframe,
            trend_score, trend_confidence,
            momentum_score, momentum_confidence,
            combined_score, combined_confidence,
            trend_weight, momentum_weight,
            trend_signal, momentum_signal, combined_signal,
            volume_score, volatility_score, cycle_score, support_resistance_score,
            recommendation, action, price_at_analysis, raw_data,
            created_at, updated_at
        ) VALUES ({phs})
    """

    ts_iso = summary["last_timestamp"].isoformat() if isinstance(summary["last_timestamp"], datetime) else str(summary["last_timestamp"])
    now = datetime.now(UTC).isoformat()

    params = (
        symbol,
        ts_iso,
        timeframe,
        summary["trend_strength"],
        summary["confidence"],
        summary["trend_strength"],
        summary["confidence"],
        summary["trend_strength"],
        summary["confidence"],
        0.5,
        0.5,
        "BULLISH" if summary["prediction"] == "bullish" else "BEARISH" if summary["prediction"] == "bearish" else "NEUTRAL",
        "BULLISH" if summary["prediction"] == "bullish" else "BEARISH" if summary["prediction"] == "bearish" else "NEUTRAL",
        "BULLISH" if summary["prediction"] == "bullish" else "BEARISH" if summary["prediction"] == "bearish" else "NEUTRAL",
        0.0,
        summary["volatility"],
        0.0,
        0.0,
        None,
        None,
        summary["price_at_analysis"],
        json.dumps({"regime": summary["regime"], "volume_profile": summary["volume_profile"]}),
        now,
        now,
    )

    conn = manager.get_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)
    score_id = cursor.lastrowid
    conn.commit()
    cursor.close()
    if manager.db_type.name == "POSTGRESQL":  # pragma: no cover
        manager.release_connection(conn)
    return int(score_id)


def insert_indicator_scores(manager: DatabaseManager, score_id: int, symbol: str, timeframe: str, summary: dict):
    """Insert a few indicator-level rows tied to a historical_score."""
    placeholders = manager.get_sql_placeholder()
    query = f"""
        INSERT INTO historical_indicator_scores (
            score_id, symbol, timestamp, timeframe,
            indicator_name, indicator_category, indicator_params,
            value, signal, confidence, created_at
        ) VALUES ({placeholders}, {placeholders}, {placeholders}, {placeholders},
                  {placeholders}, {placeholders}, {placeholders},
                  {placeholders}, {placeholders}, {placeholders}, {placeholders})
    """
    now = datetime.now(UTC).isoformat()
    ts_iso = summary["last_timestamp"].isoformat() if isinstance(summary["last_timestamp"], datetime) else str(summary["last_timestamp"])
    rows = [
        (
            score_id,
            symbol,
            ts_iso,
            timeframe,
            "volatility_std",
            "volatility",
            None,
            summary["volatility"],
            "HIGH" if summary["volatility"] > 0.2 else "NORMAL",
            summary["confidence"],
            now,
        ),
        (
            score_id,
            symbol,
            ts_iso,
            timeframe,
            "trend_strength",
            "trend",
            None,
            summary["trend_strength"],
            summary["regime"],
            summary["confidence"],
            now,
        ),
        (
            score_id,
            symbol,
            ts_iso,
            timeframe,
            "rsi_14",
            "momentum",
            None,
            summary.get("rsi"),
            None,
            summary["confidence"],
            now,
        ),
        (
            score_id,
            symbol,
            ts_iso,
            timeframe,
            "macd_12_26_9",
            "momentum",
            json.dumps({"fast": 12, "slow": 26, "signal": 9}),
            summary.get("macd"),
            None,
            summary["confidence"],
            now,
        ),
        (
            score_id,
            symbol,
            ts_iso,
            timeframe,
            "macd_signal",
            "momentum",
            None,
            summary.get("macd_signal"),
            None,
            summary["confidence"],
            now,
        ),
        (
            score_id,
            symbol,
            ts_iso,
            timeframe,
            "macd_hist",
            "momentum",
            None,
            summary.get("macd_hist"),
            None,
            summary["confidence"],
            now,
        ),
        (
            score_id,
            symbol,
            ts_iso,
            timeframe,
            "sma_50_200_cross",
            "trend",
            json.dumps({"short": 50, "long": 200}),
            1
            if summary.get("sma_cross") == "golden"
            else -1 if summary.get("sma_cross") == "death"
            else 0,
            summary.get("sma_cross"),
            summary["confidence"],
            now,
        ),
    ]
    conn = manager.get_connection()
    cursor = conn.cursor()
    cursor.executemany(query, rows)
    conn.commit()
    cursor.close()
    if manager.db_type.name == "POSTGRESQL":  # pragma: no cover
        manager.release_connection(conn)


def ingest_symbol(manager: DatabaseManager, symbol: str, limit: int, timeframe: str):
    """Fetch symbol candles from TSE and persist summarized analytics."""
    candles = tse_data_source.fetch_price_data(ticker=symbol)
    if limit and len(candles) > limit:
        candles = candles[-limit:]
    if not candles:
        print(f"⚠️  No data for {symbol}, skipping")
        return

    summary = summarize_candles(candles)

    # Record high-level tool performance (using a simple baseline tool)
    manager.record_tool_performance(
        tool_name="baseline_trend",
        tool_category="trend",
        symbol=symbol,
        timeframe=timeframe,
        market_regime=summary["regime"],
        prediction_type=summary["prediction"],
        confidence_score=summary["confidence"],
        volatility_level=summary["volatility"],
        trend_strength=summary["trend_strength"],
        volume_profile=summary["volume_profile"],
        metadata={"source": "tse_ingest_cli"},
    )

    # Store historical score + indicator details
    score_id = insert_historical_score(manager, symbol, timeframe, summary)
    insert_indicator_scores(manager, score_id, symbol, timeframe, summary)

    print(f"✅ Ingested {symbol}: regime={summary['regime']}, prediction={summary['prediction']}, confidence={summary['confidence']:.2f}")


def get_last_score_ts(manager: DatabaseManager, symbol: str, timeframe: str) -> datetime | None:
    """Return last ingested timestamp for a symbol/timeframe."""
    rows = manager.execute_query(
        "SELECT MAX(timestamp) AS ts FROM historical_scores WHERE symbol = ? AND timeframe = ?",
        (symbol, timeframe),
        fetch=True,
    )
    if rows and rows[0].get("ts"):
        try:
            return datetime.fromisoformat(rows[0]["ts"])
        except Exception:
            return None
    return None


def ingest_full_history(manager: DatabaseManager, symbol: str, timeframe: str):
    """Ingest historical_scores for every candle (resume-aware)."""
    candles = tse_data_source.fetch_price_data(ticker=symbol)
    if not candles:
        print(f"⚠️  No data for {symbol}, skipping")
        return

    last_ts = get_last_score_ts(manager, symbol, timeframe)
    df = pd.DataFrame(candles)
    df = df.sort_values("timestamp")
    if last_ts:
        df = df[df["timestamp"] > last_ts]
    if df.empty:
        return

    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["return"] = df["close"].pct_change()
    vol_mean = df["volume"].mean()
    vol_std = df["volume"].std() or 1.0

    placeholders = manager.get_sql_placeholder()
    phs = ", ".join([placeholders] * 24)
    query = f"""
        INSERT INTO historical_scores (
            symbol, timestamp, timeframe,
            trend_score, trend_confidence,
            momentum_score, momentum_confidence,
            combined_score, combined_confidence,
            trend_weight, momentum_weight,
            trend_signal, momentum_signal, combined_signal,
            volume_score, volatility_score, cycle_score, support_resistance_score,
            recommendation, action, price_at_analysis, raw_data,
            created_at, updated_at
        ) VALUES ({phs})
    """

    rows: list[tuple] = []
    for _, row in df.iterrows():
        ts = row["timestamp"]
        ret = row["return"] if pd.notna(row["return"]) else 0.0
        conf = float(min(0.99, max(0.05, abs(ret) * 5)))
        sig = "BULLISH" if ret > 0.005 else "BEARISH" if ret < -0.005 else "NEUTRAL"
        vol_z = float((row["volume"] - vol_mean) / vol_std)
        vol_score = max(-3.0, min(3.0, vol_z))
        volat = float(df["return"].rolling(window=20, min_periods=5).std().iloc[_] or 0.0)
        now = datetime.now(UTC).isoformat()
        rows.append(
            (
                symbol,
                ts,
                timeframe,
                ret,
                conf,
                ret,
                conf,
                ret,
                conf,
                0.5,
                0.5,
                sig,
                sig,
                sig,
                vol_score,
                volat,
                0.0,
                0.0,
                None,
                None,
                row["close"],
                json.dumps({"vol_z": vol_score}),
                now,
                now,
            )
        )

    conn = manager.get_connection()
    cursor = conn.cursor()
    cursor.executemany(query, rows)
    conn.commit()
    cursor.close()
    if manager.db_type.name == "POSTGRESQL":  # pragma: no cover
        manager.release_connection(conn)
    print(f"✅ Ingested full history for {symbol}: rows={len(rows)}")


def reset_tables(manager: DatabaseManager):
    """Clear project DB tables to avoid mixing old test data with real ingest."""
    for table in [
        "tool_performance_history",
        "tool_performance_stats",
        "ml_weights_history",
        "tool_recommendations_log",
        "historical_scores",
        "historical_indicator_scores",
        "backtest_runs",
    ]:
        manager.execute_query(f"DELETE FROM {table}")
    conn = manager.get_connection()
    conn.commit()
    if manager.db_type.name == "POSTGRESQL":  # pragma: no cover
        manager.release_connection(conn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest TSE OHLCV data into the project database.")
    parser.add_argument("--symbols", help="Comma-separated tickers to ingest. If omitted, auto-selects from TSE DB.")
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=100000,
        help="Max symbols to auto-select when --symbols is not provided (default: all).",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Minimum candle rows per symbol when auto-selecting (default: 1 to include all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of most-recent candles per symbol. 0 means all history (default).",
    )
    parser.add_argument("--timeframe", default="1d", help="Logical timeframe label to store with records.")
    parser.add_argument("--reset", action="store_true", help="Clear target tables before ingest.")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip symbols already ingested (based on tool_performance_history).",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Also write per-candle historical_scores for all candles (resume-aware).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    manager = DatabaseManager(auto_setup=True)
    if args.reset:
        print("⏳ Resetting project database tables ...")
        reset_tables(manager)
        print("✅ Tables cleared.")

    symbols: Iterable[str]
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = tse_data_source.list_symbols(limit=args.max_symbols, min_rows=args.min_rows)

    symbols = list(symbols)
    if not symbols:
        raise SystemExit("No symbols found to ingest.")

    already: set[str] = set()
    if args.resume:
        try:
            rows = manager.execute_query(
                "SELECT DISTINCT symbol FROM tool_performance_history", fetch=True
            ) or []
            already = {row["symbol"] for row in rows if row.get("symbol")}
        except Exception:
            already = set()

    symbols_to_run = [s for s in symbols if s not in already]
    print(
        f"🚀 Ingesting {len(symbols_to_run)} symbols from TSE into project DB "
        f"(limit={'all' if args.limit == 0 else args.limit} candles each; resume skip {len(already)})"
    )

    for sym in symbols_to_run:
        try:
            if args.full_history:
                ingest_full_history(manager, sym, timeframe=args.timeframe)
            ingest_symbol(manager, sym, limit=args.limit, timeframe=args.timeframe)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            print(f"⚠️  Failed to ingest {sym}: {exc}")

    print("🎉 Ingest completed.")


if __name__ == "__main__":
    main()
