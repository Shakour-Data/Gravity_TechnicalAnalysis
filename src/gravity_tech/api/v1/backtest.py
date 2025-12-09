"""
Backtesting API endpoints (optional) with persistence to backtest_runs.
"""

from __future__ import annotations

import time
from datetime import datetime

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from gravity_tech.database.database_manager import DatabaseManager
from gravity_tech.ml.backtesting import PatternBacktester, run_backtest_with_synthetic_data
from gravity_tech.patterns.harmonic import HarmonicPatternDetector

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


@router.post("", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """
    Run a backtest using provided OHLCV arrays or synthetic data (if no data provided).
    """
    try:
        start_ts = time.perf_counter()

        detector = HarmonicPatternDetector(tolerance=0.15)
        backtester = PatternBacktester(detector=detector, classifier=None, min_confidence=request.min_confidence)

        if request.highs and request.lows and request.closes and request.volumes:
            highs = request.highs
            lows = request.lows
            closes = request.closes
            volumes = request.volumes
            dates = request.dates or list(range(len(closes)))
            dates = [datetime.fromtimestamp(d / 1000) for d in dates]
        else:
            highs, lows, closes, volumes, dates = backtester.generate_historical_data(n_bars=request.window_size * 2)

        trades = backtester.run_backtest(
            highs=highs,
            lows=lows,
            closes=closes,
            volume=volumes,
            dates=dates,
        )
        metrics = backtester.calculate_metrics()
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
        )

        if request.persist:
            try:
                dbm = DatabaseManager(auto_setup=True)
                dbm.save_backtest_run(
                    symbol=request.symbol or "synthetic",
                    source="api",
                    interval=None,
                    params=request.dict(),
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

    except Exception as e:
        logger.error("backtest_api_error", error=str(e))
        BACKTEST_API_REQUESTS.labels("error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backtest failed: {str(e)}",
        )
