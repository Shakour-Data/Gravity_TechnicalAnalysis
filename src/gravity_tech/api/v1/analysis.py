"""
Technical Analysis endpoints (restored) for direct TA requests.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import structlog
from fastapi import APIRouter, HTTPException, status
from gravity_tech.core.contracts.analysis import AnalysisRequest, TechnicalAnalysisResult
from gravity_tech.core.domain.entities import Candle, IndicatorResult
from gravity_tech.services.analysis_service import TechnicalAnalysisService

from src.database import tse_data_source

logger = structlog.get_logger()

router = APIRouter(tags=["Technical Analysis"])


@router.get(
    "/analyze/historical/{symbol}",
    response_model=TechnicalAnalysisResult,
    summary="Analyze Historical Data",
    description="Fetch historical data from database and perform analysis",
)
async def analyze_historical(
    symbol: str,
    timeframe: str = "1d",
    days: int = 365,
) -> TechnicalAnalysisResult:
    """Analyze historical data from local database."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    raw_candles = tse_data_source.fetch_price_data(
        symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    if not raw_candles:
        raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")

    candles = [Candle(symbol=symbol, timeframe=timeframe, **c) for c in raw_candles]

    if len(candles) < 50:
        raise HTTPException(status_code=400, detail="Insufficient data for analysis (min 50 candles)")

    request = AnalysisRequest(symbol=symbol, timeframe=timeframe, candles=candles)
    return await TechnicalAnalysisService.analyze(request)


@router.post(
    "/analyze",
    response_model=TechnicalAnalysisResult,
    summary="Complete Technical Analysis",
    description="Perform comprehensive technical analysis with all indicators and patterns",
)
async def analyze_complete(request: AnalysisRequest) -> TechnicalAnalysisResult:
    """Perform complete technical analysis."""
    try:
        result = await TechnicalAnalysisService.analyze(request)
        return result
    except Exception as e:
        logger.error("analysis_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


@router.post(
    "/analyze/indicators",
    response_model=list[IndicatorResult],
    summary="Specific Indicators Analysis",
    description="Calculate specific indicators only",
)
async def analyze_specific_indicators(
    symbol: str,
    timeframe: str,
    candles: list[dict],
    indicator_names: list[str],
) -> list[IndicatorResult]:
    """Calculate specific indicators."""
    try:
        candle_objects = [Candle(**c) for c in candles]
        results = await TechnicalAnalysisService.analyze_specific_indicators(candle_objects, indicator_names)
        return results
    except Exception as e:
        logger.error("specific_indicators_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indicator calculation failed: {str(e)}",
        )


@router.get(
    "/indicators/list",
    summary="List Available Indicators",
    description="Get list of all available indicators and patterns",
)
async def list_indicators():
    """List all available indicators categorized by type."""
    return {
        "trend_indicators": [
            "SMA",
            "EMA",
            "WMA",
            "DEMA",
            "TEMA",
            "MACD",
            "ADX",
            "Parabolic SAR",
            "Supertrend",
            "Ichimoku",
        ],
        "momentum_indicators": [
            "RSI",
            "Stochastic",
            "CCI",
            "ROC",
            "Williams %R",
            "MFI",
            "Ultimate Oscillator",
            "TSI",
            "KST",
            "PMO",
        ],
        "cycle_indicators": [
            "Sine Wave",
            "Hilbert Transform - Dominant Cycle",
            "Detrended Price Oscillator (DPO)",
            "Schaff Trend Cycle (STC)",
            "Market Facilitation Index",
            "Cycle Period",
            "Phase Change Index",
            "Trend vs Cycle Identifier",
            "Autocorrelation Periodogram",
            "Cycle Phase Index",
        ],
        "volume_indicators": [
            "OBV",
            "CMF",
            "VWAP",
            "A/D Line",
            "Volume Profile",
            "PVT",
            "EMV",
            "VPT",
            "Volume Oscillator",
            "VWMA",
        ],
        "volatility_indicators": [
            "Bollinger Bands",
            "ATR",
            "Keltner Channel",
            "Donchian Channel",
            "Standard Deviation",
            "Historical Volatility",
            "Chandelier Exit",
            "Mass Index",
            "Ulcer Index",
            "RVI",
        ],
        "support_resistance_indicators": [
            "Pivot Points",
            "Fibonacci Retracement",
            "Fibonacci Extension",
            "Camarilla Pivots",
            "Woodie Pivots",
            "DeMark Pivots",
            "Support/Resistance Levels",
            "Floor Pivots",
            "Psychological Levels",
            "Previous High/Low",
        ],
        "candlestick_patterns": [
            "Doji",
            "Hammer",
            "Inverted Hammer",
            "Bullish Engulfing",
            "Bearish Engulfing",
            "Morning Star",
            "Evening Star",
            "Bullish Harami",
            "Bearish Harami",
            "Three White Soldiers",
            "Three Black Crows",
            "Piercing Pattern",
            "Dark Cloud Cover",
            "Tweezer Top",
            "Tweezer Bottom",
            "Marubozu",
        ],
        "elliott_wave_analysis": [
            "5-Wave Impulsive Pattern (1-2-3-4-5)",
            "3-Wave Corrective Pattern (A-B-C)",
            "Wave Rules Validation",
            "Fibonacci Projections",
            "Current Wave Identification",
        ],
        "market_phase_analysis": [
            "Accumulation Phase (Dow Theory)",
            "Markup Phase (Bullish Trend)",
            "Distribution Phase (Dow Theory)",
            "Markdown Phase (Bearish Trend)",
            "Transition Phase",
            "Volume Confirmation Analysis",
            "Trend Structure Analysis (Higher Highs/Lows)",
            "Trading Recommendations based on Phase",
        ],
    }


@router.get(
    "/health",
    summary="Service Health Check",
    description="Check if the analysis service is operational",
)
async def health_check():
    """Service health check."""
    return {
        "status": "healthy",
        "service": "technical-analysis",
        "version": "1.0.0",
    }
