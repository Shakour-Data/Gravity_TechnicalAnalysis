"""
CLI helper to execute the CompleteAnalysisPipeline using saved multi-horizon
weight/model artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from gravity_tech.core.domain.entities import Candle
from gravity_tech.ml.data_connector import DataConnector
from gravity_tech.ml.pipeline_factory import build_pipeline_from_weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Complete Analysis Pipeline using stored weights/models.",
    )
    parser.add_argument(
        "--candles",
        help="Path to a JSON file containing candle data "
        "(list of {'timestamp','open','high','low','close','volume'}).",
    )
    parser.add_argument(
        "--symbol",
        help="Fetch candles for this symbol via DataConnector instead of providing --candles.",
    )
    parser.add_argument(
        "--interval",
        default="1h",
        help="Candle interval to request when using --symbol (default: 1h).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of candles to fetch when using --symbol (default: 500).",
    )
    parser.add_argument("--trend-weights", required=True, help="Trend weights JSON path.")
    parser.add_argument("--momentum-weights", required=True, help="Momentum weights JSON path.")
    parser.add_argument("--volatility-weights", required=True, help="Volatility weights JSON path.")
    parser.add_argument("--trend-model", help="Optional pickled model for the trend learner.")
    parser.add_argument("--momentum-model", help="Optional pickled model for the momentum learner.")
    parser.add_argument("--volatility-model", help="Optional pickled model for the volatility learner.")
    parser.add_argument("--cycle-weights", help="Optional cycle weights JSON path.")
    parser.add_argument("--cycle-model", help="Optional pickled model for the cycle learner.")
    parser.add_argument("--sr-weights", help="Optional support/resistance weights JSON path.")
    parser.add_argument("--sr-model", help="Optional pickled model for the support/resistance learner.")
    parser.add_argument(
        "--disable-volume-matrix",
        action="store_true",
        help="Skip the volume-dimension matrix stage.",
    )
    parser.add_argument(
        "--output",
        default="pipeline_result.json",
        help="Path to write the pipeline result JSON (default: pipeline_result.json).",
    )
    args = parser.parse_args()

    candles = _resolve_candles(args)

    pipeline = build_pipeline_from_weights(
        candles=candles,
        trend_weights_path=args.trend_weights,
        momentum_weights_path=args.momentum_weights,
        volatility_weights_path=args.volatility_weights,
        trend_model_path=args.trend_model,
        momentum_model_path=args.momentum_model,
        volatility_model_path=args.volatility_model,
        cycle_weights_path=args.cycle_weights,
        cycle_model_path=args.cycle_model,
        sr_weights_path=args.sr_weights,
        sr_model_path=args.sr_model,
        verbose=True,
        use_volume_matrix=not args.disable_volume_matrix,
    )

    result = pipeline.analyze()
    payload = result.to_dict()

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Pipeline result saved to {output_path}")


def _load_candles(path: str) -> list[Candle]:
    with open(path, encoding="utf-8") as fh:
        payload: Sequence[dict] = json.load(fh)

    candles: list[Candle] = []
    for item in payload:
        candles.append(
            Candle(
                timestamp=_parse_timestamp(item["timestamp"]),
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=float(item["volume"]),
            )
        )
    return candles


def _resolve_candles(args) -> list[Candle]:
    if args.candles and args.symbol:
        raise SystemExit("Provide either --candles or --symbol, not both.")
    if args.candles:
        return _load_candles(args.candles)
    if args.symbol:
        connector = DataConnector()
        try:
            candles = connector.fetch_candles(
                symbol=args.symbol,
                interval=args.interval,
                limit=args.limit,
            )
        except Exception as e:
            raise SystemExit(f"Failed to fetch candles from data connector: {e}") from None
        if not candles:
            raise SystemExit("No candles retrieved from data connector.")
        return candles
    raise SystemExit("Either --candles or --symbol must be provided.")


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported timestamp format: {value}") from exc


if __name__ == "__main__":
    main()
