"""
Historical Analysis API Endpoints

این API برای دسترسی به داده‌های historical تحلیل‌ها استفاده می‌شود.
قسمت دیتابیس از رویکرد هیبریدی.

Author: Gravity Tech Team
Date: November 20, 2025
Version: 1.0.0
License: MIT
"""

from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import structlog
import asyncio
from concurrent.futures import ThreadPoolExecutor

from gravity_tech.database.historical_manager import HistoricalScoreManager
from gravity_tech.services.cache_service import cache_manager

logger = structlog.get_logger()

router = APIRouter(tags=["Historical Analysis"], prefix="/historical")

# Thread pool for synchronous database operations
executor = ThreadPoolExecutor(max_workers=4)


# ============================================================================
# Request/Response Models
# ============================================================================

class HistoricalAnalysisRequest(BaseModel):
    """درخواست تحلیل historical"""
    symbol: str = Field(..., description="نماد معاملاتی")
    timeframe: str = Field(..., description="تایم‌فریم")
    start_date: Optional[datetime] = Field(None, description="تاریخ شروع")
    end_date: Optional[datetime] = Field(None, description="تاریخ پایان")
    limit: int = Field(100, description="حداکثر تعداد نتایج", ge=1, le=1000)


class HistoricalScoreSummary(BaseModel):
    """خلاصه امتیازات historical"""
    symbol: str
    timeframe: str
    date: datetime
    combined_score: float
    combined_confidence: float
    combined_signal: str
    trend_score: float
    momentum_score: float


# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "/analyze",
    response_model=List[HistoricalScoreSummary],
    summary="Historical Analysis Query",
    description="دریافت تحلیل‌های historical از دیتابیس"
)
async def get_historical_analysis(request: HistoricalAnalysisRequest):
    """
    دریافت تحلیل‌های historical برای یک نماد و تایم‌فریم
    
    - **symbol**: نماد معاملاتی (مثل BTCUSDT)
    - **timeframe**: تایم‌فریم (مثل 1h, 1d)
    - **start_date**: تاریخ شروع (اختیاری)
    - **end_date**: تاریخ پایان (اختیاری)
    - **limit**: حداکثر تعداد نتایج (1-1000)
    
    Returns: لیست تحلیل‌های historical با امتیازات
    """
    try:
        manager = HistoricalScoreManager()
        
        # تنظیم تاریخ‌ها اگر مشخص نشده
        end_date = request.end_date or datetime.utcnow()
        start_date = request.start_date or (end_date - timedelta(days=30))
        
        # اجرای synchronous database query در thread pool
        loop = asyncio.get_event_loop()
        entries = await loop.run_in_executor(
            executor,
            lambda: manager.get_scores_by_symbol_timeframe(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=start_date,
                end_date=end_date,
                limit=request.limit
            )
        )
        
        # تبدیل به response model
        results = []
        for entry in entries:
            results.append(HistoricalScoreSummary(
                symbol=entry.symbol,
                timeframe=entry.timeframe,
                date=entry.timestamp,
                combined_score=entry.combined_score,
                combined_confidence=entry.combined_confidence,
                combined_signal=entry.combined_signal,
                trend_score=entry.trend_score,
                momentum_score=entry.momentum_score
            ))
        
        logger.info(
            "historical_analysis_retrieved",
            symbol=request.symbol,
            timeframe=request.timeframe,
            count=len(results)
        )
        
        return results
    
    except Exception as e:
        logger.error("historical_analysis_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Historical analysis failed: {str(e)}"
        )


@router.get(
    "/symbols",
    summary="Available Symbols",
    description="دریافت لیست نمادهای موجود در historical data"
)
async def get_available_symbols():
    """دریافت لیست نمادهای موجود در دیتابیس historical"""
    try:
        loop = asyncio.get_event_loop()
        manager = HistoricalScoreManager()
        symbols = await loop.run_in_executor(executor, lambda: manager.get_available_symbols())
        return {"symbols": symbols}
    
    except Exception as e:
        logger.error("get_symbols_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get symbols: {str(e)}"
        )


@router.get(
    "/timeframes",
    summary="Available Timeframes",
    description="دریافت لیست تایم‌فریم‌های موجود در historical data"
)
async def get_available_timeframes(symbol: Optional[str] = None):
    """دریافت لیست تایم‌فریم‌های موجود"""
    try:
        loop = asyncio.get_event_loop()
        manager = HistoricalScoreManager()
        timeframes = await loop.run_in_executor(executor, lambda: manager.get_available_timeframes(symbol))
        return {"timeframes": timeframes}
    
    except Exception as e:
        logger.error("get_timeframes_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get timeframes: {str(e)}"
        )


@router.get(
    "/stats/{symbol}",
    summary="Symbol Statistics",
    description="آمار historical برای یک نماد"
)
async def get_symbol_stats(symbol: str, timeframe: Optional[str] = None):
    """دریافت آمار historical برای یک نماد"""
    try:
        loop = asyncio.get_event_loop()
        manager = HistoricalScoreManager()
        stats = await loop.run_in_executor(executor, lambda: manager.get_symbol_statistics(symbol, timeframe))
        return stats
    
    except Exception as e:
        logger.error("get_stats_error", symbol=symbol, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.delete(
    "/cleanup",
    summary="Cleanup Old Data",
    description="پاک کردن داده‌های قدیمی‌تر از تاریخ مشخص"
)
async def cleanup_old_data(days: int = Query(90, description="تعداد روز برای نگهداری داده‌ها")):
    """پاک کردن داده‌های قدیمی برای مدیریت فضای دیتابیس"""
    try:
        loop = asyncio.get_event_loop()
        manager = HistoricalScoreManager()
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        deleted_count = await loop.run_in_executor(executor, lambda: manager.cleanup_old_data(cutoff_date))
        
        logger.info("historical_data_cleaned", deleted_count=deleted_count, cutoff_date=cutoff_date)
        
        return {
            "message": f"Cleaned up {deleted_count} old records",
            "cutoff_date": cutoff_date.isoformat()
        }
    
    except Exception as e:
        logger.error("cleanup_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cleanup failed: {str(e)}"
        )