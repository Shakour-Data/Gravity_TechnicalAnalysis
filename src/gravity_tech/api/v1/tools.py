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

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# TODO: Import from actual modules when integrated
# from ml.ml_tool_recommender import DynamicToolRecommender, MarketContext
# from services.tool_recommendation_service import ToolRecommendationService


router = APIRouter(prefix="/tools", tags=["tools"])


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
    selected_tools: List[str] = Field(
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
    categories: Optional[List[ToolCategory]] = Field(
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
    parameters: Dict[str, Any]
    best_for: List[str]
    timeframes: List[str]
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
    best_for: List[str]


class MarketContextInfo(BaseModel):
    """اطلاعات کانتکست بازار"""
    regime: str = Field(..., description="trending_bullish, trending_bearish, ranging, volatile")
    volatility: float = Field(..., ge=0.0, le=100.0)
    trend_strength: float = Field(..., ge=0.0, le=100.0)
    volume_profile: str = Field(..., description="high, medium, low")


class DynamicStrategy(BaseModel):
    """استراتژی پیشنهادی"""
    primary_tools: List[str]
    supporting_tools: List[str]
    confidence: float
    based_on: str
    regime: str
    expected_accuracy: str


class ToolRecommendationResponse(BaseModel):
    """پاسخ کامل پیشنهاد ابزارها"""
    symbol: str
    market_context: MarketContextInfo
    analysis_goal: str
    recommendations: Dict[str, List[ToolRecommendation]] = Field(
        ...,
        description="دسته‌بندی شده: must_use, recommended, optional, avoid"
    )
    dynamic_strategy: DynamicStrategy
    ml_metadata: Dict[str, Any]
    timestamp: datetime


class CustomAnalysisResponse(BaseModel):
    """پاسخ تحلیل با ابزارهای دلخواه"""
    symbol: str
    timeframe: str
    selected_tools: List[str]
    tool_results: Dict[str, Any] = Field(
        ...,
        description="نتایج هر ابزار"
    )
    ml_scoring: Optional[Dict[str, Any]] = Field(
        default=None,
        description="امتیازدهی ML (اگر فعال باشد)"
    )
    patterns_detected: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="الگوهای شناسایی شده"
    )
    summary: Dict[str, Any]
    timestamp: datetime


class ToolListResponse(BaseModel):
    """لیست ابزارها"""
    total_tools: int
    categories: Dict[ToolCategory, int]
    tools: List[ToolInfo]
    timestamp: datetime


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
    
    # TODO: Implement with actual tool registry
    
    # فعلاً داده شبیه‌سازی شده
    sample_tools = [
        ToolInfo(
            name="MACD",
            category=ToolCategory.TREND_INDICATORS,
            description="Moving Average Convergence Divergence",
            parameters={
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9
            },
            best_for=["تشخیص ترند", "واگرایی", "سیگنال خرید/فروش"],
            timeframes=["15m", "1h", "4h", "1d"],
            historical_accuracy=0.79
        ),
        ToolInfo(
            name="RSI",
            category=ToolCategory.MOMENTUM_INDICATORS,
            description="Relative Strength Index",
            parameters={
                "period": 14,
                "overbought": 70,
                "oversold": 30
            },
            best_for=["اشباع خرید/فروش", "واگرایی", "برگشت ترند"],
            timeframes=["5m", "15m", "1h", "4h", "1d"],
            historical_accuracy=0.76
        ),
        ToolInfo(
            name="ADX",
            category=ToolCategory.TREND_INDICATORS,
            description="Average Directional Index",
            parameters={
                "period": 14,
                "threshold": 25
            },
            best_for=["قدرت ترند", "تایید جهت"],
            timeframes=["1h", "4h", "1d"],
            historical_accuracy=0.82
        )
    ]
    
    # Apply filters
    filtered_tools = sample_tools
    
    if category:
        filtered_tools = [t for t in filtered_tools if t.category == category]
    
    if timeframe:
        filtered_tools = [t for t in filtered_tools if timeframe in t.timeframes]
    
    if min_accuracy is not None:
        filtered_tools = [
            t for t in filtered_tools 
            if t.historical_accuracy and t.historical_accuracy >= min_accuracy
        ]
    
    filtered_tools = filtered_tools[:limit]
    
    # Count by category
    category_counts = {}
    for tool in sample_tools:
        category_counts[tool.category] = category_counts.get(tool.category, 0) + 1
    
    return ToolListResponse(
        total_tools=len(sample_tools),
        categories=category_counts,
        tools=filtered_tools,
        timestamp=datetime.utcnow()
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
    
    # TODO: Integrate with actual DynamicToolRecommender
    
    # فعلاً پاسخ شبیه‌سازی شده
    return ToolRecommendationResponse(
        symbol=request.symbol,
        market_context=MarketContextInfo(
            regime="trending_bullish",
            volatility=45.5,
            trend_strength=72.3,
            volume_profile="high"
        ),
        analysis_goal=request.analysis_goal.value,
        recommendations={
            "must_use": [
                ToolRecommendation(
                    name="ADX",
                    category=ToolCategory.TREND_INDICATORS,
                    ml_weight=0.28,
                    confidence=0.87,
                    historical_accuracy="82.0%",
                    reason="در بازار روندی بسیار موثر است | وزن ML بالا (91.2%)",
                    priority=ToolPriority.MUST_USE,
                    best_for=["قدرت ترند", "تایید جهت حرکت"]
                ),
                ToolRecommendation(
                    name="MACD",
                    category=ToolCategory.TREND_INDICATORS,
                    ml_weight=0.24,
                    confidence=0.83,
                    historical_accuracy="79.0%",
                    reason="در بازار روندی بسیار موثر است | وزن ML بالا (85.7%)",
                    priority=ToolPriority.MUST_USE,
                    best_for=["تشخیص ترند", "سیگنال‌های خرید/فروش", "واگرایی"]
                )
            ],
            "recommended": [
                ToolRecommendation(
                    name="RSI",
                    category=ToolCategory.MOMENTUM_INDICATORS,
                    ml_weight=0.18,
                    confidence=0.76,
                    historical_accuracy="76.0%",
                    reason="برای تشخیص نقاط اصلاح در روند",
                    priority=ToolPriority.RECOMMENDED,
                    best_for=["شناسایی اشباع خرید/فروش", "واگرایی"]
                )
            ],
            "optional": [
                ToolRecommendation(
                    name="Stochastic",
                    category=ToolCategory.MOMENTUM_INDICATORS,
                    ml_weight=0.12,
                    confidence=0.68,
                    historical_accuracy="75.0%",
                    reason="ابزار استاندارد برای این شرایط",
                    priority=ToolPriority.OPTIONAL,
                    best_for=["اشباع خرید/فروش", "نقاط برگشت"]
                )
            ],
            "avoid": [
                ToolRecommendation(
                    name="Bollinger_Bands",
                    category=ToolCategory.VOLATILITY_INDICATORS,
                    ml_weight=0.05,
                    confidence=0.42,
                    historical_accuracy="74.0%",
                    reason="در بازار روندی قوی، باندها کم‌اثرتر هستند",
                    priority=ToolPriority.AVOID,
                    best_for=["محدوده قیمت", "شکست"]
                )
            ]
        },
        dynamic_strategy=DynamicStrategy(
            primary_tools=["ADX", "MACD", "RSI"],
            supporting_tools=["EMA", "VWAP"],
            confidence=0.84,
            based_on="تحلیل 3 ابزار برتر",
            regime="trending_bullish",
            expected_accuracy="84.0%"
        ),
        ml_metadata={
            "model_type": "lightgbm",
            "regime_weights": {
                "trend_indicators": 0.35,
                "momentum_indicators": 0.25,
                "volume_indicators": 0.15
            },
            "total_tools_analyzed": 95
        },
        timestamp=datetime.utcnow()
    )


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
    
    # TODO: Implement actual analysis with selected tools
    
    # فعلاً پاسخ شبیه‌سازی شده
    tool_results = {}
    
    for tool in request.selected_tools:
        if tool == "MACD":
            tool_results["MACD"] = {
                "macd": 125.3,
                "signal": 118.7,
                "histogram": 6.6,
                "signal_type": "bullish",
                "strength": 0.72
            }
        elif tool == "RSI":
            tool_results["RSI"] = {
                "value": 58.3,
                "signal": "neutral",
                "overbought": False,
                "oversold": False
            }
        elif tool == "ADX":
            tool_results["ADX"] = {
                "adx": 32.5,
                "plus_di": 28.3,
                "minus_di": 18.7,
                "trend_strength": "strong",
                "direction": "bullish"
            }
    
    ml_scoring = None
    if request.include_ml_scoring:
        ml_scoring = {
            "trend_score": 72.5,
            "momentum_score": 68.3,
            "volatility_score": 55.2,
            "combined_score": 67.8,
            "signal": "buy",
            "confidence": 0.78
        }
    
    patterns = None
    if request.include_patterns:
        patterns = [
            {
                "type": "Bullish_Engulfing",
                "confidence": 0.85,
                "location": "recent",
                "significance": "high"
            }
        ]
    
    return CustomAnalysisResponse(
        symbol=request.symbol,
        timeframe=request.timeframe,
        selected_tools=request.selected_tools,
        tool_results=tool_results,
        ml_scoring=ml_scoring,
        patterns_detected=patterns,
        summary={
            "overall_signal": "bullish",
            "tools_analyzed": len(request.selected_tools),
            "bullish_tools": 2,
            "bearish_tools": 0,
            "neutral_tools": 1,
            "consensus": "strong_buy"
        },
        timestamp=datetime.utcnow()
    )


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
    
    # TODO: Get from tool registry
    
    # فعلاً داده شبیه‌سازی شده
    if tool_name.upper() == "MACD":
        return ToolInfo(
            name="MACD",
            category=ToolCategory.TREND_INDICATORS,
            description="Moving Average Convergence Divergence - یکی از محبوب‌ترین اندیکاتورها برای تشخیص ترند و سیگنال‌های خرید/فروش",
            parameters={
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9,
                "price_type": "close"
            },
            best_for=[
                "تشخیص ترند و جهت بازار",
                "سیگنال‌های خرید و فروش",
                "شناسایی واگرایی",
                "تایید شکست‌ها"
            ],
            timeframes=["15m", "30m", "1h", "4h", "1d"],
            historical_accuracy=0.79
        )
    
    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")


# ==================== Health Check ====================

@router.get("/health", include_in_schema=False)
async def health_check():
    """بررسی سلامت API"""
    return {
        "status": "healthy",
        "service": "tool_recommendation",
        "timestamp": datetime.utcnow()
    }
