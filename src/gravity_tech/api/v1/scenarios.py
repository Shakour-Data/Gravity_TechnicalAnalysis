"""
API Endpoints for Three-Scenario Analysis

Provides REST API for optimistic/neutral/pessimistic scenario analysis.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""


import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from gravity_tech.analysis.scenario_analysis import (
    ScenarioAnalyzer,
    ScenarioResult,
    ThreeScenarioAnalysis,
)
from gravity_tech.clients.data_service_client import DataServiceClient
from gravity_tech.config.settings import get_settings
from prometheus_client import Counter, Histogram

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/scenarios", tags=["Scenario Analysis"])

SCENARIO_API_REQUESTS = Counter(
    "api_scenario_requests_total", "Total scenario API requests", ["status"]
)
SCENARIO_API_LATENCY = Histogram(
    "api_scenario_latency_seconds", "Scenario API latency in seconds"
)


def get_data_client() -> DataServiceClient:
    """Dependency: Get Data Service client."""
    settings = get_settings()
    return DataServiceClient(
        base_url=settings.DATA_SERVICE_URL,
        timeout=30.0,
        max_retries=3,
        redis_url=settings.REDIS_URL,
        cache_ttl=21600  # 6 hours
    )


def get_scenario_analyzer(
    data_client: DataServiceClient = Depends(get_data_client)
) -> ScenarioAnalyzer:
    """Dependency: Get Scenario Analyzer with Data Service client."""
    return ScenarioAnalyzer(data_service_client=data_client)


@router.get("/{symbol}", response_model=ThreeScenarioAnalysis)
async def analyze_scenarios(
    symbol: str,
    timeframe: str = Query(default="1d", regex="^(1m|5m|15m|1h|4h|1d|1w)$"),
    lookback_days: int = Query(default=365, ge=30, le=1825),
    analyzer: ScenarioAnalyzer = Depends(get_scenario_analyzer)
):
    """
    تحلیل سه‌سناریویی (خوشبینانه، خنثی، بدبینانه)

    این endpoint سه سناریو برای نماد محاسبه می‌کند:
    - **Optimistic (خوشبینانه):** احتمال 65-75%، هدف 3×ATR، ریسک 0.5×ATR
    - **Neutral (خنثی):** احتمال 45-55%، هدف 1.5×ATR، ریسک 1×ATR
    - **Pessimistic (بدبینانه):** احتمال 25-35%، هدف 0.5×ATR، ریسک 1.5×ATR

    **بازدهی مورد انتظار:**
    ```
    E(Return) = P(opt) × R(opt) + P(neu) × R(neu) + P(pes) × R(pes)
    ```

    **مثال:**
    ```
    GET /api/v1/scenarios/AAPL?timeframe=1d&lookback_days=365
    ```

    **Response:**
    ```json
    {
      "symbol": "AAPL",
      "current_price": 180.5,
      "optimistic": {
        "score": 78.5,
        "probability": 70.0,
        "target_price": 195.0,
        "stop_loss": 178.0,
        "risk_reward_ratio": 3.0,
        "recommendation": "BUY"
      },
      "neutral": {
        "score": 65.0,
        "probability": 50.0,
        "target_price": 187.0,
        "stop_loss": 175.0,
        "risk_reward_ratio": 1.5
      },
      "pessimistic": {
        "score": 45.0,
        "probability": 30.0,
        "target_price": 182.0,
        "stop_loss": 172.0,
        "risk_reward_ratio": 0.5
      },
      "expected_return": 5.8,
      "expected_risk": 3.2,
      "sharpe_ratio": 1.81
    }
    ```

    Args:
        symbol: نماد سهم (مثال: AAPL، فولاد، BTC-USD)
        timeframe: بازه زمانی (1m, 5m, 15m, 1h, 4h, 1d, 1w)
        lookback_days: تعداد روزهای گذشته برای تحلیل (30-1825)

    Returns:
        ThreeScenarioAnalysis: تحلیل کامل سه سناریو با احتمالات و اهداف

    Raises:
        404: اگر نماد پیدا نشود
        400: اگر پارامترها نامعتبر باشند
        503: اگر Data Service در دسترس نباشد
    """
    logger.info(
        "scenario_analysis_request",
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days
    )

    try:
        import time
        start = time.perf_counter()
        # تحلیل سناریوها (داده از Data Service دریافت می‌شود)
        analysis = await analyzer.analyze_from_service(
            symbol=symbol,
            timeframe=timeframe,
            lookback_days=lookback_days
        )
        duration = time.perf_counter() - start
        SCENARIO_API_REQUESTS.labels("success").inc()
        SCENARIO_API_LATENCY.observe(duration)

        logger.info(
            "scenario_analysis_completed",
            symbol=symbol,
            expected_return=analysis.expected_return,
            sharpe_ratio=analysis.sharpe_ratio,
            recommended_scenario=analysis.recommended_scenario
        )

        return analysis

    except ValueError as e:
        SCENARIO_API_REQUESTS.labels("error").inc()
        logger.error("validation_error", symbol=symbol, error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        SCENARIO_API_REQUESTS.labels("error").inc()
        logger.error(
            "scenario_analysis_error",
            symbol=symbol,
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(
            status_code=503,
            detail=f"Failed to analyze scenarios: {str(e)}"
        ) from e


@router.get("/{symbol}/optimistic", response_model=ScenarioResult)
async def get_optimistic_scenario(
    symbol: str,
    timeframe: str = Query(default="1d", regex="^(1m|5m|15m|1h|4h|1d|1w)$"),
    lookback_days: int = Query(default=365, ge=30, le=1825),
    analyzer: ScenarioAnalyzer = Depends(get_scenario_analyzer)
):
    """
    فقط سناریو خوشبینانه

    Args:
        symbol: نماد
        timeframe: بازه زمانی
        lookback_days: روزهای گذشته

    Returns:
        ScenarioResult: فقط سناریو optimistic
    """
    analysis = await analyze_scenarios(symbol, timeframe, lookback_days, analyzer)
    return analysis.optimistic


@router.get("/{symbol}/neutral", response_model=ScenarioResult)
async def get_neutral_scenario(
    symbol: str,
    timeframe: str = Query(default="1d", regex="^(1m|5m|15m|1h|4h|1d|1w)$"),
    lookback_days: int = Query(default=365, ge=30, le=1825),
    analyzer: ScenarioAnalyzer = Depends(get_scenario_analyzer)
):
    """
    فقط سناریو خنثی

    Args:
        symbol: نماد
        timeframe: بازه زمانی
        lookback_days: روزهای گذشته

    Returns:
        ScenarioResult: فقط سناریو neutral
    """
    analysis = await analyze_scenarios(symbol, timeframe, lookback_days, analyzer)
    return analysis.neutral


@router.get("/{symbol}/pessimistic", response_model=ScenarioResult)
async def get_pessimistic_scenario(
    symbol: str,
    timeframe: str = Query(default="1d", regex="^(1m|5m|15m|1h|4h|1d|1w)$"),
    lookback_days: int = Query(default=365, ge=30, le=1825),
    analyzer: ScenarioAnalyzer = Depends(get_scenario_analyzer)
):
    """
    فقط سناریو بدبینانه

    Args:
        symbol: نماد
        timeframe: بازه زمانی
        lookback_days: روزهای گذشته

    Returns:
        ScenarioResult: فقط سناریو pessimistic
    """
    analysis = await analyze_scenarios(symbol, timeframe, lookback_days, analyzer)
    return analysis.pessimistic
