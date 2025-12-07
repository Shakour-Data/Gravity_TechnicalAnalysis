"""
Tool recommendation and catalog endpoints backed by the real catalog service.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, validator

from gravity_tech.services.tool_recommendation_service import ToolRecommendationService

router = APIRouter(prefix="/tools", tags=["tools"])
tool_service = ToolRecommendationService()


class TradingStyle(str, Enum):
    SCALP = "scalp"
    DAY = "day"
    SWING = "swing"
    POSITION = "position"


class AnalysisGoal(str, Enum):
    ENTRY_SIGNAL = "entry_signal"
    EXIT_SIGNAL = "exit_signal"
    RISK_MANAGEMENT = "risk_management"
    TREND_CONFIRMATION = "trend_confirmation"
    REVERSAL_DETECTION = "reversal_detection"


class ToolPriority(str, Enum):
    MUST_USE = "must_use"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    AVOID = "avoid"


class ToolCategory(str, Enum):
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


class ToolRecommendationRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field(default="1d", description="Requested timeframe")
    analysis_goal: AnalysisGoal = Field(default=AnalysisGoal.ENTRY_SIGNAL)
    trading_style: Optional[TradingStyle] = Field(default=TradingStyle.SWING)
    limit_candles: int = Field(default=200, ge=50, le=1000)
    top_n: int = Field(default=15, ge=5, le=50)

    @validator("timeframe")
    def validate_timeframe(cls, value: str) -> str:
        valid = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]
        if value not in valid:
            raise ValueError(f"Timeframe must be one of {valid}")
        return value


class CustomAnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = Field(default="1d")
    selected_tools: list[str] = Field(..., min_items=1, max_items=30)
    include_ml_scoring: bool = Field(default=True)
    include_patterns: bool = Field(default=True)
    limit_candles: int = Field(default=200, ge=50, le=1000)


class ToolFilterRequest(BaseModel):
    categories: Optional[list[ToolCategory]] = None
    min_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    timeframe: Optional[str] = None
    trading_style: Optional[TradingStyle] = None


class ToolInfo(BaseModel):
    name: str
    category: ToolCategory
    description: str
    parameters: dict[str, Any]
    best_for: list[str]
    timeframes: list[str]
    historical_accuracy: Optional[float] = None


class ToolRecommendation(BaseModel):
    name: str
    category: ToolCategory
    ml_weight: float
    confidence: float
    historical_accuracy: str
    reason: str
    priority: ToolPriority
    best_for: list[str]


class MarketContextInfo(BaseModel):
    regime: str
    volatility: float
    trend_strength: float
    volume_profile: str


class DynamicStrategy(BaseModel):
    primary_tools: list[str]
    supporting_tools: list[str]
    confidence: float
    based_on: str
    regime: str
    expected_accuracy: str


class ToolRecommendationResponse(BaseModel):
    symbol: str
    market_context: MarketContextInfo
    analysis_goal: str
    recommendations: dict[str, list[ToolRecommendation]]
    dynamic_strategy: DynamicStrategy
    ml_metadata: dict[str, Any]
    timestamp: datetime


class CustomAnalysisResponse(BaseModel):
    symbol: str
    timeframe: str
    selected_tools: list[str]
    tool_results: dict[str, Any]
    ml_scoring: Optional[dict[str, Any]] = None
    patterns_detected: Optional[list[dict[str, Any]]] = None
    summary: dict[str, Any]
    timestamp: datetime


class ToolListResponse(BaseModel):
    total_tools: int
    categories: dict[ToolCategory, int]
    tools: list[ToolInfo]
    timestamp: datetime


@router.get("/", response_model=ToolListResponse)
async def list_tools(
    category: Optional[ToolCategory] = Query(None),
    timeframe: Optional[str] = Query(None),
    min_accuracy: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=200),
):
    tools = tool_service.list_tools(
        category=category.value if category else None,
        timeframe=timeframe,
        min_accuracy=min_accuracy,
        limit=limit,
    )

    tool_infos = [
        ToolInfo(
            name=item["name"],
            category=ToolCategory(item["category"]),
            description=item["description"],
            parameters=item.get("parameters", {}),
            best_for=item.get("best_for", []),
            timeframes=item.get("timeframes", []),
            historical_accuracy=item.get("historical_accuracy"),
        )
        for item in tools
    ]

    category_counts = {
        ToolCategory(key): value for key, value in tool_service.category_summary().items()
    }

    return ToolListResponse(
        total_tools=len(tool_service.catalog),
        categories=category_counts,
        tools=tool_infos,
        timestamp=datetime.utcnow(),
    )


@router.post("/recommend", response_model=ToolRecommendationResponse)
async def recommend_tools(request: ToolRecommendationRequest):
    service_response = await tool_service.build_recommendations(
        symbol=request.symbol,
        timeframe=request.timeframe,
        analysis_goal=request.analysis_goal.value,
        trading_style=request.trading_style.value if request.trading_style else "swing",
        limit_candles=request.limit_candles,
        top_n=request.top_n,
    )

    market_context = service_response["market_context"]
    recs: dict[str, list[ToolRecommendation]] = {}

    for bucket, items in service_response["recommendations"].items():
        recs[bucket] = [
            ToolRecommendation(
                name=item["name"],
                category=ToolCategory(item["category"]),
                ml_weight=item["ml_weight"],
                confidence=item["confidence"],
                historical_accuracy=item["historical_accuracy"],
                reason=item["reason"],
                priority=ToolPriority(bucket.upper()),
                best_for=item.get("best_for", []),
            )
            for item in items
        ]

    strategy = DynamicStrategy(**service_response["dynamic_strategy"])

    return ToolRecommendationResponse(
        symbol=request.symbol,
        market_context=MarketContextInfo(
            regime=market_context["regime"],
            volatility=market_context["volatility"],
            trend_strength=market_context["trend_strength"],
            volume_profile=market_context["volume_profile"],
        ),
        analysis_goal=request.analysis_goal.value,
        recommendations=recs,
        dynamic_strategy=strategy,
        ml_metadata=service_response["ml_metadata"],
        timestamp=datetime.utcnow(),
    )


@router.post("/analyze/custom", response_model=CustomAnalysisResponse)
async def analyze_with_custom_tools(request: CustomAnalysisRequest):
    analysis = await tool_service.analyze_custom_tools(
        symbol=request.symbol,
        timeframe=request.timeframe,
        selected_tools=request.selected_tools,
        include_ml_scoring=request.include_ml_scoring,
        include_patterns=request.include_patterns,
        limit_candles=request.limit_candles,
    )

    return CustomAnalysisResponse(
        symbol=analysis["symbol"],
        timeframe=analysis["timeframe"],
        selected_tools=analysis["selected_tools"],
        tool_results=analysis["tool_results"],
        ml_scoring=analysis["ml_scoring"],
        patterns_detected=analysis["patterns_detected"],
        summary=analysis["summary"],
        timestamp=datetime.fromisoformat(analysis["timestamp"]),
    )


@router.get("/categories")
async def list_categories():
    return tool_service.list_categories()


@router.get("/tool/{tool_name}", response_model=ToolInfo)
async def get_tool_info(tool_name: str):
    tool = tool_service.get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    return ToolInfo(
        name=tool["name"],
        category=ToolCategory(tool["category"]),
        description=tool["description"],
        parameters=tool.get("parameters", {}),
        best_for=tool.get("best_for", []),
        timeframes=tool.get("timeframes", []),
        historical_accuracy=tool.get("historical_accuracy"),
    )


@router.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "healthy",
        "service": "tool_recommendation",
        "timestamp": datetime.utcnow(),
    }
