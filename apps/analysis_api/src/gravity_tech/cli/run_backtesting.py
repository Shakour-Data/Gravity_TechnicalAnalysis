"""
Lightweight CLI wrapper for the pattern backtesting module.

Usage examples:
    python -m gravity_tech.cli.run_backtesting --symbol FOLD --source db --limit 1200
    python -m gravity_tech.cli.run_backtesting --symbol BTCUSDT --source connector --interval 1d
"""

from __future__ import annotations

import argparse

from gravity_tech.ml.backtest_optimizer import suggest_params
from gravity_tech.ml.backtesting import run_backtest_with_real_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run harmonic-pattern backtests using real data only.")
    parser.add_argument("--symbol", help="Ticker symbol to backtest (required for db/connector source).")
    parser.add_argument("--source", choices=["db", "connector"], default="db")
    parser.add_argument("--interval", default="1d", help="Interval for DataConnector requests (connector only).")
    parser.add_argument("--limit", type=int, default=1200, help="Number of bars to load.")
    parser.add_argument("--min-confidence", type=float, default=0.6, dest="min_confidence")
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Suggest params (min_confidence/limit) from stored backtest_runs history for this symbol/interval.",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist backtest summary to the application database (backtest_runs table).",
    )
    args = parser.parse_args()

    if args.source in {"db", "connector"} and not args.symbol:
        raise SystemExit("Please provide --symbol when using source=db or connector.")

    min_conf = args.min_confidence
    limit = args.limit
    if args.auto_tune:
        suggestion = suggest_params(symbol=args.symbol, interval=args.interval)
        if suggestion.min_confidence:
            min_conf = suggestion.min_confidence
        if suggestion.limit:
            limit = suggestion.limit
        if suggestion.interval:
            args.interval = suggestion.interval
        if suggestion.source and args.source == "db":
            args.source = suggestion.source

    run_backtest_with_real_data(
        symbol=args.symbol,
        source=args.source,
        interval=args.interval,
        limit=limit,
        min_confidence=min_conf,
        persist=args.persist,
    )


if __name__ == "__main__":
    main()
