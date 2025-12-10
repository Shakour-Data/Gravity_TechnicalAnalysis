"""
Backtesting API endpoints (optional) with persistence to backtest_runs.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, status
import numpy as np
import pandas as pd
from gravity_tech.database.database_manager import DatabaseManager
from gravity_tech.ml.backtesting import PatternBacktester
from gravity_tech.patterns.harmonic import HarmonicPatternDetector
from pydantic import BaseModel, Field

try:
    from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover
    class _Noop:
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            return self
        def observe(self, *args, **kwargs):
            return self
    Counter = Histogram = lambda *args, **kwargs: _Noop()

logger = structlog.get_logger()

router = APIRouter(tags=["Backtesting"], prefix="/backtest")

BACKTEST_API_REQUESTS = Counter(
    "api_backtest_requests_total", "Total backtest API requests", ["status"]
)
BACKTEST_API_LATENCY = Histogram(
    "api_backtest_latency_seconds", "Backtest API latency in seconds"
)


class BacktestRequest(BaseModel):
    symbol: str | None = Field(None, description="Ticker symbol for real-data backtest")
    highs: list[float] | None = None
    lows: list[float] | None = None
    closes: list[float] | None = None
    volumes: list[float] | None = None
    dates: list[int] | None = None
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    window_size: int = Field(default=200, ge=100, le=500)
    step_size: int = Field(default=50, ge=10, le=100)
    persist: bool = Field(default=False, description="Persist summary to backtest_runs")


class BacktestMetrics(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    average_pnl: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    target1_hits: int | None = None
    target2_hits: int | None = None


class BacktestResponse(BaseModel):
    metrics: BacktestMetrics
    trade_count: int
    backtest_period: dict[str, str]
    analysis_time_ms: float
    model_version: str | None = None
    data_source: str | None = None
    warnings: list[str] | None = None


def _ensure_ohlcv_valid(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    dates: list[int] | None,
    window_size: int,
    step_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[datetime]]:
    """Validate OHLCV arrays and return sanitized numpy arrays + datetime list."""
    length = len(highs)
    arrays = [lows, closes, volumes]
    if any(len(arr) != length for arr in arrays):
        raise HTTPException(status_code=400, detail="highs/lows/closes/volumes must be the same length")

    min_required = max(window_size + step_size, 300)
    if length < min_required:
        raise HTTPException(status_code=400, detail=f"Provide at least {min_required} bars for backtest (got {length})")

    highs_arr = np.asarray(highs, dtype=float)
    lows_arr = np.asarray(lows, dtype=float)
    closes_arr = np.asarray(closes, dtype=float)
    volumes_arr = np.asarray(volumes, dtype=float)

    for name, arr in (("highs", highs_arr), ("lows", lows_arr), ("closes", closes_arr), ("volumes", volumes_arr)):
        if not np.all(np.isfinite(arr)):
            raise HTTPException(status_code=400, detail=f"{name} contains NaN/Inf values")

    if not np.all(highs_arr >= lows_arr):
        raise HTTPException(status_code=400, detail="All highs must be >= lows")

    if not np.all((closes_arr <= highs_arr) & (closes_arr >= lows_arr)):
        raise HTTPException(status_code=400, detail="All closes must be between highs and lows")

    if dates:
        if len(dates) != length:
            raise HTTPException(status_code=400, detail="dates length must match price arrays")
        try:
            dt_list = [datetime.fromtimestamp(d / 1000) for d in dates]
        except Exception as exc:  # pragma: no cover - input parse failure
            raise HTTPException(status_code=400, detail=f"Invalid date format: {exc}") from exc
        if any(dt_list[i] < dt_list[i - 1] for i in range(1, len(dt_list))):
            raise HTTPException(status_code=400, detail="dates must be non-decreasing")
    else:
        dt_list = [datetime.utcnow()] * length

    return highs_arr, lows_arr, closes_arr, volumes_arr, dt_list


def _synthetic_ohlcv(n_bars: int) -> tuple[list[float], list[float], list[float], list[float], list[datetime]]:
    """Generate lightweight synthetic OHLCV for fallback paths (tests/sandbox)."""
    rng = np.random.default_rng(42)
    base = 100.0
    drift = rng.normal(0.02, 0.01)
    prices = [base]
    for _ in range(n_bars - 1):
        prices.append(prices[-1] * (1 + drift + rng.normal(0, 0.002)))
    prices = np.array(prices, dtype=np.float32)
    highs = (prices + rng.normal(0.5, 0.2, size=n_bars)).tolist()
    lows = (prices - rng.normal(0.5, 0.2, size=n_bars)).tolist()
    closes = prices.tolist()
    volumes = (rng.normal(1_000_000, 50_000, size=n_bars)).tolist()
    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=n_bars, freq="H").to_pydatetime().tolist()
    return highs, lows, closes, volumes, dates


@router.post("", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """Run a backtest using provided OHLCV arrays or real data from the TSE database."""
    try:
        start_ts = time.perf_counter()

        detector = HarmonicPatternDetector(tolerance=0.15)
        backtester = PatternBacktester(detector=detector, classifier=None, min_confidence=request.min_confidence)

        warnings: list[str] = []
        data_source = "provided"

        if all(v is not None for v in (request.highs, request.lows, request.closes, request.volumes)):
            highs_arr, lows_arr, closes_arr, volumes_arr, dates = _ensure_ohlcv_valid(
                request.highs, request.lows, request.closes, request.volumes, request.dates,
                window_size=request.window_size, step_size=request.step_size
            )
        else:
            if not request.symbol:
                raise HTTPException(status_code=400, detail="Provide either OHLCV arrays or a symbol to load real data")
            symbol = request.symbol
            min_bars = max(request.window_size + request.step_size, 400)
            try:
                highs, lows, closes, volumes, dates_idx = backtester.generate_historical_data(
                    n_bars=min_bars,
                    symbol=symbol,
                )
                highs_arr, lows_arr, closes_arr, volumes_arr = (
                    np.asarray(highs, dtype=float),
                    np.asarray(lows, dtype=float),
                    np.asarray(closes, dtype=float),
                    np.asarray(volumes, dtype=float),
                )
                dates = list(pd.to_datetime(dates_idx))
                data_source = "tse_db"
            except Exception as exc:
                warnings.append(f"Real data unavailable for symbol={symbol}: {exc}. Using synthetic data.")
                highs, lows, closes, volumes, dates = _synthetic_ohlcv(min_bars)
                highs_arr, lows_arr, closes_arr, volumes_arr, dates = _ensure_ohlcv_valid(
                    highs, lows, closes, volumes, [int(d.timestamp() * 1000) for d in dates],
                    window_size=request.window_size, step_size=request.step_size
                )
                data_source = "synthetic"

        trades = backtester.run_backtest(
            highs=highs_arr,
            lows=lows_arr,
            closes=closes_arr,
            volume=volumes_arr,
            dates=dates,
            window_size=request.window_size,
            step_size=request.step_size,
        )
        metrics: dict[str, Any] = backtester.calculate_metrics()
        if "error" in metrics:
            warnings.append(str(metrics["error"]))
        duration_seconds = time.perf_counter() - start_ts

        response = BacktestResponse(
            metrics=BacktestMetrics(
                total_trades=metrics.get("total_trades", 0),
                winning_trades=metrics.get("winning_trades", 0),
                losing_trades=metrics.get("losing_trades", 0),
                win_rate=metrics.get("win_rate", 0.0),
                total_pnl=metrics.get("total_pnl", 0.0),
                average_pnl=metrics.get("avg_pnl", 0.0),
                profit_factor=metrics.get("profit_factor", 0.0),
                sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
                max_drawdown=metrics.get("max_drawdown", 0.0),
                target1_hits=metrics.get("target_hit_counts", {}).get("target1") if metrics.get("target_hit_counts") else None,
                target2_hits=metrics.get("target_hit_counts", {}).get("target2") if metrics.get("target_hit_counts") else None,
            ),
            trade_count=len(trades),
            backtest_period={
                "start": dates[0].isoformat() if dates else "",
                "end": dates[-1].isoformat() if dates else "",
            },
            analysis_time_ms=round(duration_seconds * 1000, 2),
            model_version=None,
            data_source=data_source,
            warnings=warnings or None,
        )

        if request.persist and data_source != "synthetic":
            try:
                dbm = DatabaseManager(auto_setup=True)
                dbm.save_backtest_run(
                    symbol=request.symbol or "unspecified",
                    source="api",
                    interval=None,
                    params=request.dict(exclude={"highs", "lows", "closes", "volumes", "dates"}),
                    metrics=metrics,
                    period_start=dates[0] if dates else None,
                    period_end=dates[-1] if dates else None,
                    model_version=None,
                )
            except Exception as exc:  # pragma: no cover - best-effort persistence
                logger.warning("backtest_persist_failed", exc_info=exc)

        BACKTEST_API_REQUESTS.labels("success").inc()
        BACKTEST_API_LATENCY.observe(duration_seconds)
        return response

    except Exception as exc:
        logger.error("backtest_api_error", error=str(exc))
        BACKTEST_API_REQUESTS.labels("error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backtest failed: {str(exc)}",
        ) from exc
