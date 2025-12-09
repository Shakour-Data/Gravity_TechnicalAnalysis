"""
API Endpoints for Dynamic Tool Recommendation

این ماژول endpoints برای:
- لیست تمام ابزارهای موجود (95+ tools)
- پیشنهاد پویای ابزارها بر اساس ML
- تحلیل با ابزارهای دلخواه کاربر

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, validator

from gravity_tech.services.tool_recommendation_service import ToolRecommendationService


router = APIRouter(prefix="/tools", tags=["tools"])
tool_service = ToolRecommendationService()


# ==================== Enums ====================

class TradingStyle(str, Enum):
    """سبک معامله‌گری"""
    SCALP = "scalp"
    DAY = "day"
    SWING = "swing"
    POSITION = "position"


class AnalysisGoal(str, Enum):
    """هدف تحلیل"""
    ENTRY_SIGNAL = "entry_signal"
    EXIT_SIGNAL = "exit_signal"
    RISK_MANAGEMENT = "risk_management"
    TREND_CONFIRMATION = "trend_confirmation"
    REVERSAL_DETECTION = "reversal_detection"


class ToolPriority(str, Enum):
    """اولویت ابزار"""
    MUST_USE = "must_use"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    AVOID = "avoid"


class ToolCategory(str, Enum):
    """دسته‌بندی ابزارها"""
    TREND_INDICATORS = "trend_indicators"
    MOMENTUM_INDICATORS = "momentum_indicators"
    VOLATILITY_INDICATORS = "volatility_indicators"
    VOLUME_INDICATORS = "volume_indicators"
    CYCLE_INDICATORS = "cycle_indicators"
    SUPPORT_RESISTANCE = "support_resistance"
    CANDLESTICK_PATTERNS = "candlestick_patterns"
    CLASSICAL_PATTERNS = "classical_patterns"
    ELLIOTT_WAVE = "elliott_wave"
    DIVERGENCE = "divergence"


# ==================== Request Models ====================

class ToolRecommendationRequest(BaseModel):
    """درخواست پیشنهاد ابزارها"""
    symbol: str = Field(..., description="نماد دارایی (مثلاً BTCUSDT)")
    timeframe: str = Field(default="1d", description="بازه زمانی (1m, 5m, 15m, 1h, 4h, 1d)")
    analysis_goal: AnalysisGoal = Field(
        default=AnalysisGoal.ENTRY_SIGNAL,
        description="هدف تحلیل"
    )
    trading_style: Optional[TradingStyle] = Field(
        default=TradingStyle.SWING,
        description="سبک معامله‌گری"
    )
    limit_candles: int = Field(
        default=200,
        ge=50,
        le=1000,
        description="تعداد کندل برای تحلیل"
    )
    top_n: int = Field(
        default=15,
        ge=5,
        le=50,
        description="تعداد ابزارهای پیشنهادی"
    )

    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]
        if v not in valid:
            raise ValueError(f"Timeframe must be one of {valid}")
        return v


class CustomAnalysisRequest(BaseModel):
    """درخواست تحلیل با ابزارهای دلخواه"""
    symbol: str = Field(..., description="نماد دارایی")
    timeframe: str = Field(default="1d", description="بازه زمانی")
    selected_tools: list[str] = Field(
        ...,
        min_items=1,
        max_items=30,
        description="ابزارهای انتخاب شده"
    )
    include_ml_scoring: bool = Field(
        default=True,
        description="آیا امتیازدهی ML انجام شود؟"
    )
    include_patterns: bool = Field(
        default=True,
        description="آیا الگوهای قیمتی شناسایی شوند؟"
    )
    limit_candles: int = Field(default=200, ge=50, le=1000)


class ToolFilterRequest(BaseModel):
    """فیلتر ابزارها"""
    categories: Optional[list[ToolCategory]] = Field(
        default=None,
        description="فیلتر بر اساس دسته"
    )
    min_accuracy: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="حداقل دقت تاریخی"
    )
    timeframe: Optional[str] = Field(
        default=None,
        description="مناسب برای timeframe خاص"
    )
    trading_style: Optional[TradingStyle] = Field(
        default=None,
        description="مناسب برای سبک معامله‌گری"
    )


# ==================== Response Models ====================

class ToolInfo(BaseModel):
    """اطلاعات یک ابزار"""
    name: str
    category: ToolCategory
    description: str
    parameters: dict[str, Any]
    best_for: list[str]
    timeframes: list[str]
    historical_accuracy: Optional[float] = None


class ToolRecommendation(BaseModel):
    """پیشنهاد یک ابزار"""
    name: str
    category: ToolCategory
    ml_weight: float = Field(..., description="وزن ML")
    confidence: float = Field(..., ge=0.0, le=1.0, description="اطمینان")
    historical_accuracy: str = Field(..., description="دقت تاریخی")
    reason: str = Field(..., description="دلیل پیشنهاد")
    priority: ToolPriority
    best_for: list[str]


class MarketContextInfo(BaseModel):
    """اطلاعات کانتکست بازار"""
    regime: str = Field(..., description="trending_bullish, trending_bearish, ranging, volatile")
    volatility: float = Field(..., ge=0.0, le=100.0)
    trend_strength: float = Field(..., ge=0.0, le=100.0)
    volume_profile: str = Field(..., description="high, medium, low")


class DynamicStrategy(BaseModel):
    """استراتژی پیشنهادی"""
    primary_tools: list[str]
    supporting_tools: list[str]
    confidence: float
    based_on: str
    regime: str
    expected_accuracy: str


class ToolRecommendationResponse(BaseModel):
    """پاسخ کامل پیشنهاد ابزارها"""
    symbol: str
    market_context: MarketContextInfo
    analysis_goal: str
    recommendations: dict[str, list[ToolRecommendation]] = Field(
        ...,
        description="دسته‌بندی شده: must_use, recommended, optional, avoid"
    )
    dynamic_strategy: DynamicStrategy
    ml_metadata: dict[str, Any]
    timestamp: datetime


class CustomAnalysisResponse(BaseModel):
    """پاسخ تحلیل با ابزارهای دلخواه"""
    symbol: str
    timeframe: str
    selected_tools: list[str]
    tool_results: dict[str, Any] = Field(
        ...,
        description="نتایج هر ابزار"
    )
    ml_scoring: Optional[dict[str, Any]] = Field(
        default=None,
        description="امتیازدهی ML (اگر فعال باشد)"
    )
    patterns_detected: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="الگوهای شناسایی شده"
    )
    summary: dict[str, Any]
    timestamp: datetime


class ToolListResponse(BaseModel):
    """لیست ابزارها"""
    total_tools: int
    categories: dict[ToolCategory, int]
    tools: list[ToolInfo]
    timestamp: datetime


def _tool_info_from_dict(entry: dict[str, Any]) -> ToolInfo:
    """Convert catalog dict entries to ToolInfo"""
    raw_cat = entry.get("category", ToolCategory.TREND_INDICATORS)
    category = raw_cat if isinstance(raw_cat, ToolCategory) else ToolCategory(raw_cat)
    return ToolInfo(
        name=entry.get("name", ""),
        category=category,
        description=entry.get("description", ""),
        parameters=entry.get("parameters", {}),
        best_for=entry.get("best_for", []),
        timeframes=entry.get("timeframes", []),
        historical_accuracy=entry.get("historical_accuracy"),
    )


# ==================== API Endpoints ====================

@router.get(
    "/",
    response_model=ToolListResponse,
    summary="لیست تمام ابزارهای موجود",
    description="دریافت لیست کامل 95+ ابزار تحلیل تکنیکال با فیلتر"
)
async def list_tools(
    category: Optional[ToolCategory] = Query(None, description="فیلتر بر اساس دسته"),
    timeframe: Optional[str] = Query(None, description="مناسب برای timeframe"),
    min_accuracy: Optional[float] = Query(None, ge=0.0, le=1.0, description="حداقل دقت"),
    limit: int = Query(100, ge=1, le=200, description="حداکثر تعداد")
):
    """
    GET /api/v1/tools/

    لیست تمام ابزارهای موجود با امکان فیلتر

    مثال:
    - GET /api/v1/tools/ → همه ابزارها
    - GET /api/v1/tools/?category=trend_indicators → فقط اندیکاتورهای ترند
    - GET /api/v1/tools/?min_accuracy=0.75 → ابزارهای با دقت بالا
    """

    tools = tool_service.list_tools(
        category=category.value if category else None,
        timeframe=timeframe,
        min_accuracy=min_accuracy,
        limit=limit,
    )

    tool_infos: list[ToolInfo] = [_tool_info_from_dict(t) for t in tools]
    category_counts: dict[ToolCategory, int] = {}
    for t in tool_infos:
        category_counts[t.category] = category_counts.get(t.category, 0) + 1

    return ToolListResponse(
        total_tools=len(tools),
        categories=category_counts,
        tools=tool_infos,
        timestamp=datetime.utcnow(),
    )


@router.post(
    "/recommend",
    response_model=ToolRecommendationResponse,
    summary="پیشنهاد پویای ابزارها",
    description="دریافت پیشنهادات هوشمند ML-based برای بهترین ابزارها"
)
async def recommend_tools(request: ToolRecommendationRequest):
    """
    POST /api/v1/tools/recommend

    پیشنهاد پویای ابزارها بر اساس:
    - وزن‌های ML
    - رژیم بازار
    - عملکرد تاریخی
    - سبک معامله‌گری

    مثال Request Body:
    {
        "symbol": "BTCUSDT",
        "timeframe": "1d",
        "analysis_goal": "entry_signal",
        "trading_style": "swing",
        "top_n": 15
    }
    """

    try:
        return await tool_service.build_recommendations(
            symbol=request.symbol,
            timeframe=request.timeframe,
            analysis_goal=request.analysis_goal.value,
            trading_style=request.trading_style.value if request.trading_style else "swing",
            limit_candles=request.limit_candles,
            top_n=request.top_n,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build recommendations: {exc}")


@router.post(
    "/analyze/custom",
    response_model=CustomAnalysisResponse,
    summary="تحلیل با ابزارهای دلخواه",
    description="اجرای تحلیل با ابزارهای انتخابی کاربر"
)
async def analyze_with_custom_tools(request: CustomAnalysisRequest):
    """
    POST /api/v1/tools/analyze/custom

    اجرای تحلیل تکنیکال با ابزارهای انتخاب شده توسط کاربر

    مثال Request Body:
    {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "selected_tools": ["MACD", "RSI", "ADX", "VWAP"],
        "include_ml_scoring": true,
        "include_patterns": true
    }
    """

    try:
        return await tool_service.analyze_custom_tools(
            symbol=request.symbol,
            timeframe=request.timeframe,
            selected_tools=request.selected_tools,
            include_ml_scoring=request.include_ml_scoring,
            include_patterns=request.include_patterns,
            limit_candles=request.limit_candles,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to analyze custom tools: {exc}")


@router.get(
    "/categories",
    summary="لیست دسته‌بندی ابزارها",
    description="دریافت لیست تمام دسته‌ها و تعداد ابزارها در هر دسته"
)
async def list_categories():
    """
    GET /api/v1/tools/categories

    لیست دسته‌بندی ابزارها
    """

    # TODO: Get from actual tool registry

    categories = {
        "trend_indicators": {
            "count": 10,
            "description": "اندیکاتورهای روند",
            "examples": ["SMA", "EMA", "MACD", "ADX"]
        },
        "momentum_indicators": {
            "count": 8,
            "description": "اندیکاتورهای مومنتوم",
            "examples": ["RSI", "Stochastic", "CCI", "Williams_R"]
        },
        "volatility_indicators": {
            "count": 10,
            "description": "اندیکاتورهای نوسان",
            "examples": ["Bollinger_Bands", "ATR", "Keltner_Channels"]
        },
        "volume_indicators": {
            "count": 10,
            "description": "اندیکاتورهای حجم",
            "examples": ["OBV", "VWAP", "Volume_Profile"]
        },
        "cycle_indicators": {
            "count": 10,
            "description": "اندیکاتورهای چرخه‌ای",
            "examples": ["Detrended_Price", "Schaff_Trend_Cycle"]
        },
        "support_resistance": {
            "count": 12,
            "description": "سطوح حمایت و مقاومت",
            "examples": ["Pivot_Points", "Fibonacci", "Gann_Levels"]
        },
        "candlestick_patterns": {
            "count": 40,
            "description": "الگوهای کندل استیک",
            "examples": ["Doji", "Hammer", "Engulfing", "Morning_Star"]
        },
        "classical_patterns": {
            "count": 15,
            "description": "الگوهای کلاسیک",
            "examples": ["Head_Shoulders", "Double_Top", "Triangle"]
        },
        "elliott_wave": {
            "count": 1,
            "description": "تحلیل امواج الیوت",
            "examples": ["Elliott_Wave_Analysis"]
        },
        "divergence": {
            "count": 3,
            "description": "تحلیل واگرایی",
            "examples": ["RSI_Divergence", "MACD_Divergence"]
        }
    }

    total = sum(cat["count"] for cat in categories.values())

    return {
        "total_tools": total,
        "total_categories": len(categories),
        "categories": categories,
        "timestamp": datetime.utcnow()
    }


@router.get(
    "/tool/{tool_name}",
    response_model=ToolInfo,
    summary="اطلاعات یک ابزار",
    description="دریافت اطلاعات کامل درباره یک ابزار خاص"
)
async def get_tool_info(tool_name: str):
    """
    GET /api/v1/tools/tool/{tool_name}

    اطلاعات کامل یک ابزار

    مثال:
    - GET /api/v1/tools/tool/MACD
    """

    tool = tool_service.get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    return _tool_info_from_dict(tool)


# ==================== Health Check ====================

@router.get("/health", include_in_schema=False)
async def health_check():
    """بررسی سلامت API"""
    return {
        "status": "healthy",
        "service": "tool_recommendation",
        "timestamp": datetime.utcnow()
    }
