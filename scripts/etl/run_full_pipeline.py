#!/usr/bin/env python3
"""
End-to-end pipeline runner.

Loads OHLCV data from the TSE SQLite source database, runs the multi-horizon
analysis pipeline, and persists summarized results into data/TechAnalysis.db.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Dynamically set repo root and analysis models paths
REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_MODELS = REPO_ROOT / "ml_models"
sys.path.insert(0, str(REPO_ROOT / "apps" / "analysis_api" / "src"))

from config import TSE_DB_FILE  # type: ignore  # noqa: E402
from database import TSEDatabaseConnector  # type: ignore  # noqa: E402
from gravity_tech.core.domain.entities import Candle  # type: ignore  # noqa: E402
from gravity_tech.database.database_manager import (  # type: ignore  # noqa: E402
    DatabaseManager,
    DatabaseType,
)
from gravity_tech.ml.complete_analysis_pipeline import (
    CompleteAnalysisPipeline,  # type: ignore  # noqa: E402
)
from gravity_tech.ml.pipeline_factory import (  # type: ignore  # noqa: E402
    load_momentum_analyzer,
    load_trend_analyzer,
    load_volatility_analyzer,
)

DEFAULT_WEIGHTS = ANALYSIS_MODELS / "multi_horizon" / "indicator_weights_btcusdt.json"
DEFAULT_MODEL = ANALYSIS_MODELS / "multi_horizon" / "indicator_weights_btcusdt.pkl"
DEFAULT_TARGET_DB = REPO_ROOT / "data" / "TechAnalysis.db"

logger = logging.getLogger("full_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full TSE -> analysis -> TechAnalysis.db pipeline.")
    parser.add_argument("--source-db", default=TSE_DB_FILE, help="Path to the TSE source SQLite database.")
    parser.add_argument("--target-db", default=str(DEFAULT_TARGET_DB), help="Path to TechAnalysis.db (output).")
    parser.add_argument(
        "--symbols",
        help="Comma-separated symbols to process. If omitted, all available symbols are processed.",
    )
    parser.add_argument("--max-symbols", type=int, default=0, help="Optional cap on number of symbols to process.")
    parser.add_argument("--limit", type=int, default=500, help="Number of most recent candles per symbol.")
    parser.add_argument("--timeframe", default="1d", help="Logical timeframe label stored with results.")
    parser.add_argument(
        "--weights-json",
        default=str(DEFAULT_WEIGHTS),
        help="Path to multi-horizon indicator weights JSON (used for trend/momentum/volatility).",
    )
    parser.add_argument(
        "--weights-model",
        default=str(DEFAULT_MODEL),
        help="Path to the pickled model file matching --weights-json.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose pipeline logging.")
    parser.add_argument(
        "--disable-volume-matrix",
        action="store_true",
        help="Skip volume-dimension matrix stage when running the pipeline.",
    )
    return parser.parse_args()


def ensure_analysis_results_table(manager: DatabaseManager) -> None:
    """Create the analysis_results table if it does not exist."""
    if manager.db_type == DatabaseType.POSTGRESQL:
        create_sql = """
        CREATE TABLE IF NOT EXISTS analysis_results (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            analysis_date TIMESTAMP NOT NULL,
            final_signal VARCHAR(20) NOT NULL,
            confidence DOUBLE PRECISION DEFAULT 0.0,
            trend_score DOUBLE PRECISION DEFAULT 0.0,
            momentum_score DOUBLE PRECISION DEFAULT 0.0,
            volatility_score DOUBLE PRECISION DEFAULT 0.0,
            cycle_score DOUBLE PRECISION DEFAULT 0.0,
            sr_score DOUBLE PRECISION DEFAULT 0.0,
            volume_interaction_score DOUBLE PRECISION DEFAULT 0.0,
            decision_matrix_score DOUBLE PRECISION DEFAULT 0.0,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            UNIQUE(symbol, analysis_date)
        );
        """
    else:
        create_sql = """
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            analysis_date TEXT NOT NULL,
            final_signal TEXT NOT NULL,
            confidence REAL DEFAULT 0.0,
            trend_score REAL DEFAULT 0.0,
            momentum_score REAL DEFAULT 0.0,
            volatility_score REAL DEFAULT 0.0,
            cycle_score REAL DEFAULT 0.0,
            sr_score REAL DEFAULT 0.0,
            volume_interaction_score REAL DEFAULT 0.0,
            decision_matrix_score REAL DEFAULT 0.0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, analysis_date)
        );
        """
    manager.execute_query(create_sql)


def upsert_analysis_result(
    manager: DatabaseManager,
    symbol: str,
    timeframe: str,
    result,
    analysis_date: datetime | None = None,
) -> None:
    """Persist pipeline output into analysis_results."""
    ph = manager.get_sql_placeholder()
    created_at = datetime.now(UTC)
    volume_score = 0.0
    if result.volume_interactions:
        scores = [interaction.interaction_score for interaction in result.volume_interactions.values()]
        if scores:
            volume_score = sum(scores) / len(scores)

    columns = [
        "symbol",
        "analysis_date",
        "final_signal",
        "confidence",
        "trend_score",
        "momentum_score",
        "volatility_score",
        "cycle_score",
        "sr_score",
        "volume_interaction_score",
        "decision_matrix_score",
        "created_at",
    ]
    placeholders = ", ".join([ph] * len(columns))

    if manager.db_type == DatabaseType.POSTGRESQL:
        insert_sql = f"""
        INSERT INTO analysis_results ({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT (symbol, analysis_date) DO UPDATE SET
            final_signal = EXCLUDED.final_signal,
            confidence = EXCLUDED.confidence,
            trend_score = EXCLUDED.trend_score,
            momentum_score = EXCLUDED.momentum_score,
            volatility_score = EXCLUDED.volatility_score,
            cycle_score = EXCLUDED.cycle_score,
            sr_score = EXCLUDED.sr_score,
            volume_interaction_score = EXCLUDED.volume_interaction_score,
            decision_matrix_score = EXCLUDED.decision_matrix_score,
            created_at = EXCLUDED.created_at;
        """
    else:
        insert_sql = f"""
        INSERT OR REPLACE INTO analysis_results ({', '.join(columns)})
        VALUES ({placeholders});
        """

    analysis_date = analysis_date or result.timestamp
    params = (
        symbol,
        analysis_date if manager.db_type == DatabaseType.POSTGRESQL else analysis_date.isoformat(),
        result.decision.final_signal.value if result.decision else "NEUTRAL",
        float(result.decision.final_confidence if result.decision else 0.0),
        float(result.trend_score.score if result.trend_score else 0.0),
        float(result.momentum_score.score if result.momentum_score else 0.0),
        float(result.volatility_score.score if result.volatility_score else 0.0),
        float(result.cycle_score.score if result.cycle_score else 0.0),
        float(result.sr_score.score if result.sr_score else 0.0),
        float(volume_score),
        float(result.decision.final_score if result.decision else 0.0),
        created_at if manager.db_type == DatabaseType.POSTGRESQL else created_at.isoformat(),
    )
    manager.execute_query(insert_sql, params)
    logger.info("saved_result", extra={"symbol": symbol, "timeframe": timeframe})


def to_candles(raw_candles: list[dict], symbol: str, timeframe: str) -> list[Candle]:
    candles: list[Candle] = []
    for item in raw_candles:
        ts = item.get("timestamp")
        if ts is None:
            logger.warning("missing_timestamp", extra={"symbol": symbol})
            continue
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts).replace(tzinfo=UTC)
        elif isinstance(ts, datetime) and ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        try:
            candle = Candle(
                timestamp=ts,
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=float(item["volume"]),
                symbol=symbol,
                timeframe=timeframe,
            )
            candles.append(candle)
        except ValueError as e:
            logger.warning("invalid_candle_data", extra={"symbol": symbol, "error": str(e)})
            continue
    return candles


def load_analyzers(weights_json: str, weights_model: str):
    trend = load_trend_analyzer(weights_json, weights_model)
    momentum = load_momentum_analyzer(weights_json, weights_model)
    volatility = load_volatility_analyzer(weights_json, weights_model)
    return trend, momentum, volatility


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    source_path = Path(args.source_db)
    if not source_path.exists():
        raise SystemExit(f"Source database not found: {source_path}")
    target_db_str = args.target_db
    is_pg = str(target_db_str).lower().startswith("postgres")
    target_path = None
    if not is_pg:
        target_path = Path(target_db_str)
        target_path.parent.mkdir(parents=True, exist_ok=True)
    weights_json = Path(args.weights_json)
    weights_model = Path(args.weights_model)
    if not weights_json.exists():
        raise SystemExit(f"Weights JSON not found: {weights_json}")
    if not weights_model.exists():
        raise SystemExit(f"Model pickle not found: {weights_model}")

    source = TSEDatabaseConnector(str(source_path))
    if is_pg:
        manager = DatabaseManager(
            db_type=DatabaseType.POSTGRESQL,
            connection_string=target_db_str,
            auto_setup=True,
            allow_fallback=False,
        )
    else:
        manager = DatabaseManager(
            db_type=DatabaseType.SQLITE,
            sqlite_path=str(target_path),
            auto_setup=True,
            allow_fallback=True,
        )
    ensure_analysis_results_table(manager)

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else source.list_symbols(limit=args.max_symbols or 100000, min_rows=120)
    )
    if args.max_symbols and len(symbols) > args.max_symbols:
        symbols = symbols[: args.max_symbols]

    if not symbols:
        logger.warning("no_symbols_found", extra={"source_db": args.source_db})
        return

    trend_analyzer, momentum_analyzer, volatility_analyzer = load_analyzers(
        str(weights_json),
        str(weights_model),
    )

    processed = 0
    success = 0

    for symbol in symbols:
        try:
            raw = source.fetch_price_data(ticker=symbol)
        except Exception as exc:
            logger.error("fetch_failed", extra={"symbol": symbol, "error": str(exc)})
            continue

        if not raw:
            logger.info("no_data", extra={"symbol": symbol})
            continue

        if args.limit and len(raw) > args.limit:
            raw = raw[-args.limit :]

        candles = to_candles(raw, symbol=symbol, timeframe=args.timeframe)
        if len(candles) < 120:
            logger.info("insufficient_candles", extra={"symbol": symbol, "candles": len(candles)})
            continue

        # Rolling analysis for last 90 days
        start_date = datetime.now(UTC) - timedelta(days=90)
        daily_success = 0
        for i in range(91):  # 0 to 90 days
            end_date = start_date + timedelta(days=i)
            candles_filtered = [c for c in candles if c.timestamp <= end_date]
            if len(candles_filtered) < 120:
                continue

            try:
                pipeline = CompleteAnalysisPipeline(
                    candles_filtered,
                    verbose=args.verbose,
                    use_volume_matrix=not args.disable_volume_matrix,
                    trend_analyzer=trend_analyzer,
                    momentum_analyzer=momentum_analyzer,
                    volatility_analyzer=volatility_analyzer,
                )
                result = pipeline.analyze()
                upsert_analysis_result(manager, symbol, args.timeframe, result, analysis_date=end_date)
                daily_success += 1
            except Exception as exc:
                logger.error(
                    "pipeline_failed",
                    extra={"symbol": symbol, "date": end_date.isoformat()},
                    exc_info=True,
                )
        if daily_success > 0:
            success += 1
            logger.info("symbol_processed", extra={"symbol": symbol, "daily_analyses": daily_success})
        else:
            logger.warning("no_daily_analyses", extra={"symbol": symbol})
        processed += 1

    logger.info(
        "pipeline_complete",
        extra={"processed": processed, "success": success, "target_db": args.target_db},
    )
    manager.close()


if __name__ == "__main__":
    main()
