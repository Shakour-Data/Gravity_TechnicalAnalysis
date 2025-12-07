"""
High-level analysis contracts used by the API and service layer.

These models were historically defined in `gravity_tech.models.schemas`
but now live under the clean `core.contracts` namespace so that the rest
of the application no longer relies on the deprecated module.
"""

from __future__ import annotations

import warnings
from datetime import datetime

from gravity_tech.core.domain.entities import (
    Candle,
    ElliottWaveResult,
    IndicatorResult,
    PatternResult,
)
from gravity_tech.core.domain.entities import (
    CoreSignalStrength as SignalStrength,
)
from pydantic import BaseModel, Field

__all__ = [
    "AnalysisRequest",
    "ChartAnalysisResult",
    "MarketPhaseResult",
    "TechnicalAnalysisResult",
]


class ChartAnalysisResult(BaseModel):
    """Alternative chart analysis result (Renko, Three Line Break, Point & Figure)."""

    chart_type: str = Field(..., description="RENKO, THREE_LINE_BREAK, or POINT_FIGURE")
    signal: SignalStrength = Field(..., description="Chart signal")
    current_trend: str = Field(..., description="UP, DOWN, or CONSOLIDATION")
    consecutive_bars: int = Field(..., description="Number of consecutive bars in current direction")
    confidence: float = Field(ge=0.0, le=1.0, description="Analysis confidence")
    description: str = Field(..., description="Analysis description")

    @staticmethod
    def from_value(value: float) -> SignalStrength:
        """
        Convert numeric value (-1 to 1) to SignalStrength.

        Args:
            value: Normalized value between -1 and 1

        Returns:
            Corresponding SignalStrength
        """
        if value >= 0.8:
            return SignalStrength.VERY_BULLISH
        if value >= 0.4:
            return SignalStrength.BULLISH
        if value >= 0.1:
            return SignalStrength.BULLISH_BROKEN
        if value >= -0.1:
            return SignalStrength.NEUTRAL
        if value >= -0.4:
            return SignalStrength.BEARISH_BROKEN
        if value >= -0.8:
            return SignalStrength.BEARISH
        return SignalStrength.VERY_BEARISH


class MarketPhaseResult(BaseModel):
    """Market Phase Analysis Result based on Dow Theory."""

    market_phase: str = Field(..., description="Current market phase")
    phase_strength: str = Field(..., description="Strength of current phase")
    description: str = Field(..., description="Detailed phase description in Persian")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall phase score (0-100)")
    trend_structure: str = Field(..., description="Trend structure: uptrend, downtrend, contraction, etc.")
    volume_confirmation: bool = Field(..., description="Whether volume confirms the trend (Dow Theory)")
    recommendations: list[str] = Field(default_factory=list, description="Trading recommendations based on phase")
    detailed_scores: dict[str, float] = Field(
        default_factory=dict, description="Detailed scores for different aspects"
    )
    dow_theory_compliance: bool = Field(default=True, description="Confirms analysis follows Dow Theory principles")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")


class TechnicalAnalysisResult(BaseModel):
    """Comprehensive technical analysis result."""

    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe (e.g., 1h, 4h, 1d)")

    trend_indicators: list[IndicatorResult] = Field(default_factory=list, description="Trend indicators")
    momentum_indicators: list[IndicatorResult] = Field(default_factory=list, description="Momentum indicators")
    cycle_indicators: list[IndicatorResult] = Field(default_factory=list, description="Cycle indicators")
    volume_indicators: list[IndicatorResult] = Field(default_factory=list, description="Volume indicators")
    volatility_indicators: list[IndicatorResult] = Field(default_factory=list, description="Volatility indicators")
    support_resistance_indicators: list[IndicatorResult] = Field(
        default_factory=list, description="Support/Resistance indicators"
    )

    classical_patterns: list[PatternResult] = Field(default_factory=list, description="Classical chart patterns")
    candlestick_patterns: list[PatternResult] = Field(default_factory=list, description="Candlestick patterns")
    elliott_wave_analysis: ElliottWaveResult | None = Field(default=None, description="Elliott Wave analysis")

    renko_analysis: ChartAnalysisResult | None = Field(default=None, description="Renko chart analysis")
    three_line_break_analysis: ChartAnalysisResult | None = Field(
        default=None, description="Three Line Break analysis"
    )
    point_figure_analysis: ChartAnalysisResult | None = Field(default=None, description="Point & Figure analysis")

    market_phase_analysis: MarketPhaseResult | None = Field(
        default=None, description="Market phase analysis based on Dow Theory"
    )

    overall_trend_signal: SignalStrength | None = Field(default=None, description="Overall trend signal")
    overall_momentum_signal: SignalStrength | None = Field(default=None, description="Overall momentum signal")
    overall_cycle_signal: SignalStrength | None = Field(default=None, description="Overall cycle signal")
    overall_signal: SignalStrength | None = Field(default=None, description="Overall combined signal")
    overall_confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Overall confidence")

    ml_weights: dict[str, float] | None = Field(default=None, description="ML-predicted optimal weights")
    weights_source: str | None = Field(default="default", description="Source of weights: 'default', 'ml', 'adaptive'")

    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")

    def calculate_overall_signal(self):
        """
        Backward-compatible wrapper that delegates to the signal engine.

        The actual implementation lives in `gravity_tech.services.signal_engine`
        so that this Pydantic model remains a pure data contract.
        """

        warnings.warn(
            "TechnicalAnalysisResult.calculate_overall_signal() is deprecated; "
            "use gravity_tech.services.signal_engine.compute_overall_signals instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            from gravity_tech.services.signal_engine import compute_overall_signals
        except Exception as exc:  # pragma: no cover - guarded import
            raise RuntimeError("Signal engine is not available") from exc

        compute_overall_signals(self)


class AnalysisRequest(BaseModel):
    """Request for technical analysis."""

    symbol: str = Field(..., description="Trading symbol", json_schema_extra={"example": "BTCUSDT"})
    timeframe: str = Field(
        ...,
        description="Timeframe",
        json_schema_extra={"example": "1h"},
        pattern="^(1m|5m|15m|30m|1h|4h|1d|1w)$",
    )
    candles: list[Candle] = Field(..., description="Historical candle data", min_length=50)
    indicators: list[str] | None = Field(
        default=None, description="Specific indicators to calculate (if None, calculate all)"
    )
