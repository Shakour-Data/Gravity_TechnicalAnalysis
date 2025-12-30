"""
Fix orphaned symbols in database

Finds symbols with candle data but no analysis results
and reprocesses them through the analysis pipeline.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "analysis_api" / "src"))

logger = structlog.get_logger()


async def find_symbols_without_analysis(db_engine) -> list[str]:
    """
    Find symbols that have candles but no analysis results

    Returns:
        List of symbol names without analysis
    """

    try:
        # Execute query to find orphaned symbols
        query = """
        SELECT DISTINCT c.symbol
        FROM candles c
        LEFT JOIN analysis_results ar ON c.symbol = ar.symbol
        WHERE ar.symbol IS NULL
        GROUP BY c.symbol
        HAVING COUNT(c.id) > 100
        ORDER BY c.symbol
        """

        # Note: Implementation depends on database connector
        # This is pseudocode; actual implementation uses SQLAlchemy or direct DB access

        logger.info("finding_orphaned_symbols", query_prepared=True)

        # For now, return empty list
        # In real implementation, execute actual query
        return []

    except Exception as e:
        logger.error("find_symbols_error", error=str(e))
        raise


async def get_symbol_candles(db_engine, symbol: str, limit: int = 500) -> list[dict[str, Any]]:
    """
    Get candles for specific symbol

    Args:
        db_engine: Database engine
        symbol: Asset symbol
        limit: Maximum candles to retrieve

    Returns:
        List of candle dictionaries
    """

    try:
        logger.info("fetching_symbol_candles", symbol=symbol, limit=limit)

        # Placeholder: actual implementation fetches from DB
        # Returns sorted by timestamp, newest first

        return []

    except Exception as e:
        logger.error("fetch_candles_error", symbol=symbol, error=str(e))
        return []


async def run_analysis(
    symbol: str, candles: list[dict[str, Any]], analysis_service
) -> dict[str, Any] | None:
    """
    Run analysis on symbol's candles

    Args:
        symbol: Asset symbol
        candles: List of candle data
        analysis_service: Analysis service (from Phase 2 DI)

    Returns:
        Analysis result dictionary or None if failed
    """

    try:
        if not candles:
            logger.warning("no_candles_for_analysis", symbol=symbol)
            return None

        logger.info("running_analysis", symbol=symbol, candles=len(candles))

        # Call analysis service
        # This uses DI from Phase 2
        result = await analysis_service.analyze(symbol=symbol, candles=candles)

        logger.info("analysis_complete", symbol=symbol, signal=result.get("signal"))
        return result

    except Exception as e:
        logger.error("analysis_error", symbol=symbol, error=str(e))
        return None


async def store_analysis_result(db_engine, symbol: str, result: dict[str, Any]) -> bool:
    """
    Store analysis result in database

    Args:
        db_engine: Database engine
        symbol: Asset symbol
        result: Analysis result

    Returns:
        True if successful
    """

    try:
        logger.info("storing_result", symbol=symbol)

        # Placeholder: actual implementation inserts into analysis_results table
        # INSERT OR REPLACE to handle duplicates

        return True

    except Exception as e:
        logger.error("store_result_error", symbol=symbol, error=str(e))
        return False


async def reprocess_missing_symbols(
    db_engine, analysis_service, batch_size: int = 10, limit: int | None = None
) -> dict[str, Any]:
    """
    Reprocess symbols without analysis results

    Args:
        db_engine: Database engine
        analysis_service: Analysis service (from DI)
        batch_size: Process symbols in batches
        limit: Max symbols to process (for testing)

    Returns:
        Statistics dictionary with results
    """

    logger.info("reprocessing_starting", batch_size=batch_size)

    try:
        # Find orphaned symbols
        missing_symbols = await find_symbols_without_analysis(db_engine)

        if limit:
            missing_symbols = missing_symbols[:limit]

        logger.info("found_symbols", count=len(missing_symbols), symbols=missing_symbols[:10])

        stats = {
            "total_symbols": len(missing_symbols),
            "processed": 0,
            "success": 0,
            "errors": 0,
            "start_time": datetime.now(),
            "symbols_processed": [],
        }

        # Process in batches
        for batch_idx in range(0, len(missing_symbols), batch_size):
            batch = missing_symbols[batch_idx : batch_idx + batch_size]
            logger.info("processing_batch", batch_num=batch_idx // batch_size + 1, count=len(batch))

            for symbol in batch:
                try:
                    # Get candles for symbol
                    candles = await get_symbol_candles(db_engine, symbol, limit=500)

                    if not candles:
                        logger.warning("no_candles", symbol=symbol)
                        stats["errors"] += 1
                        continue

                    # Run analysis
                    result = await run_analysis(symbol, candles, analysis_service)

                    if not result:
                        logger.error("analysis_failed", symbol=symbol)
                        stats["errors"] += 1
                        continue

                    # Store result
                    stored = await store_analysis_result(db_engine, symbol, result)

                    if stored:
                        stats["success"] += 1
                        stats["symbols_processed"].append(
                            {
                                "symbol": symbol,
                                "signal": result.get("signal"),
                                "confidence": result.get("confidence", 0),
                            }
                        )
                        logger.info("symbol_fixed", symbol=symbol)
                    else:
                        stats["errors"] += 1

                    stats["processed"] += 1

                except Exception as e:
                    stats["errors"] += 1
                    logger.error("symbol_processing_error", symbol=symbol, error=str(e))
                    continue

        stats["end_time"] = datetime.now()
        stats["duration_seconds"] = (stats["end_time"] - stats["start_time"]).total_seconds()
        stats["success_rate"] = (
            stats["success"] / stats["processed"] * 100 if stats["processed"] > 0 else 0
        )

        logger.info("reprocessing_complete", stats=stats)
        return stats

    except Exception as e:
        logger.error("reprocessing_failed", error=str(e))
        raise


async def main():
    """Main entry point"""

    import argparse

    from gravity_tech.config.unified_settings import get_settings
    from gravity_tech.infrastructure.container import get_container

    parser = argparse.ArgumentParser(description="Fix orphaned symbols in database")
    parser.add_argument(
        "--database-url", help="Target database URL (overrides config)", default=None
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Symbols to process per batch")
    parser.add_argument(
        "--limit", type=int, default=None, help="Max symbols to process (for testing)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't actually update database")

    args = parser.parse_args()

    # Setup logging
    structlog.configure(
        processors=[
            structlog.stdlib.ProcessorFormatter.wrap_for_logger_factory,
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger.info("fix_orphaned_symbols_starting", args=vars(args))

    try:
        # Get settings and DI container
        settings = get_settings()
        container = get_container()

        # Get services from DI
        db_engine = container.get("database")
        analysis_service = container.get("analysis_service")

        # Run reprocessing
        result = await reprocess_missing_symbols(
            db_engine=db_engine,
            analysis_service=analysis_service,
            batch_size=args.batch_size,
            limit=args.limit,
        )

        # Report results
        print(f"\n{'=' * 60}")
        print("Orphaned Symbol Fix Complete")
        print(f"{'=' * 60}")
        print(f"Total Symbols:      {result['total_symbols']}")
        print(f"Processed:          {result['processed']}")
        print(f"Success:            {result['success']}")
        print(f"Errors:             {result['errors']}")
        print(f"Success Rate:       {result['success_rate']:.1f}%")
        print(f"Duration:           {result['duration_seconds']:.1f}s")
        print(f"{'=' * 60}\n")

        return 0 if result["errors"] == 0 else 1

    except Exception as e:
        logger.error("main_error", error=str(e))
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
