"""
===============================================================================
DEPRECATION WARNING

This module is DEPRECATED as of Phase 2.1 (November 7, 2025).
All models have been migrated to: src.core.domain.entities

Please update your imports:
OLD: from models.schemas import Candle, SignalStrength, IndicatorResult
NEW: from src.core.domain.entities import Candle, CoreSignalStrength, IndicatorResult

This backward compatibility layer will be removed in Phase 2.2 (Day 3).
===============================================================================
"""

import warnings

warnings.warn(
    "Importing from models.schemas is deprecated. "
    "Use src.core.domain.entities instead. "
    "This module will be removed in Phase 2.2.",
    DeprecationWarning,
    stacklevel=2
)

# BACKWARD COMPATIBILITY LAYER (Phase 2.1)
# Import core entities from new location and re-export with aliases
from src.core.domain.entities import (
    Candle,
    CoreSignalStrength as SignalStrength,  # Alias for backward compatibility
    IndicatorCategory,
    IndicatorResult,
    PatternType,
    PatternResult,
    WavePoint,
    ElliottWaveResult,
)

# Note: ChartAnalysisResult, MarketPhaseResult, TechnicalAnalysisResult 
# are Pydantic models defined below (API layer models, not migrated)

__all__ = [
    "Candle",
    "SignalStrength",
    "IndicatorCategory",
    "IndicatorResult",
    "PatternType",
    "PatternResult",
    "WavePoint",
    "ElliottWaveResult",
    "ChartAnalysisResult",      # Pydantic model (defined below)
    "MarketPhaseResult",        # Pydantic model (defined below)
    "TechnicalAnalysisResult",  # Pydantic model (defined below)
    "AnalysisRequest",          # Pydantic model (defined below)
]

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum
from decimal import Decimal

# ============================================================================
# PYDANTIC MODELS (For API layer only - not migrated)
# ============================================================================

# Note: SignalStrength is imported from entities above (CoreSignalStrength alias)
# This is the new 7-level enum with Persian labels

class ChartAnalysisResult(BaseModel):
    """Alternative chart analysis result (Renko, Three Line Break, Point & Figure)"""

    chart_type: str = Field(
        ...,
        description="RENKO, THREE_LINE_BREAK, or POINT_FIGURE"
    )
    signal: SignalStrength = Field(..., description="Chart signal")
    current_trend: str = Field(
        ...,
        description="UP, DOWN, or CONSOLIDATION"
    )
    consecutive_bars: int = Field(
        ...,
        description="Number of consecutive bars in current direction"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Analysis confidence"
    )
    description: str = Field(..., description="Analysis description")

    @staticmethod
    def from_value(value: float) -> 'SignalStrength':
        """
        Convert numeric value (-1 to 1) to SignalStrength

        Args:
            value: Normalized value between -1 and 1

        Returns:
            Corresponding SignalStrength
        """
        if value >= 0.8:
            return SignalStrength.VERY_BULLISH
        elif value >= 0.4:
            return SignalStrength.BULLISH
        elif value >= 0.1:
            return SignalStrength.BULLISH_BROKEN
        elif value >= -0.1:
            return SignalStrength.NEUTRAL
        elif value >= -0.4:
            return SignalStrength.BEARISH_BROKEN
        elif value >= -0.8:
            return SignalStrength.BEARISH
        else:
            return SignalStrength.VERY_BEARISH

class MarketPhaseResult(BaseModel):
    """Market Phase Analysis Result based on Dow Theory"""
    market_phase: str = Field(
        ...,
        description="Current market phase: انباشت, صعود, توزیع, نزول, انتقال"
    )
    phase_strength: str = Field(
        ...,
        description="Strength of current phase"
    )
    description: str = Field(
        ...,
        description="Detailed phase description in Persian"
    )
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall phase score (0-100)"
    )
    trend_structure: str = Field(
        ...,
        description="Trend structure: uptrend, downtrend, contraction, etc."
    )
    volume_confirmation: bool = Field(
        ...,
        description="Whether volume confirms the trend (Dow Theory)"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Trading recommendations based on phase"
    )
    detailed_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Detailed scores for different aspects"
    )
    dow_theory_compliance: bool = Field(
        default=True,
        description="Confirms analysis follows Dow Theory principles"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Analysis timestamp"
    )

class TechnicalAnalysisResult(BaseModel):
    """Comprehensive technical analysis result (API response model)"""

    # (The rest of the class fields and methods should be properly indented here.)

    # Example of how to add a validator method inside the class:
    # @validator('volume')
    # def volume_must_be_non_negative(cls, v):
    #     """Validate volume is non-negative"""
    #     if v < 0:
    #         raise ValueError('Volume cannot be negative')
    #     return v


class TechnicalAnalysisResult(BaseModel):
    """Comprehensive technical analysis result"""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe (e.g., 1h, 4h, 1d)")
    
    # Indicators by category
    trend_indicators: List[IndicatorResult] = Field(
        default_factory=list,
        description="Trend indicators"
    )
    momentum_indicators: List[IndicatorResult] = Field(
        default_factory=list,
        description="Momentum indicators"
    )
    cycle_indicators: List[IndicatorResult] = Field(
        default_factory=list,
        description="Cycle indicators"
    )
    volume_indicators: List[IndicatorResult] = Field(
        default_factory=list,
        description="Volume indicators"
    )
    volatility_indicators: List[IndicatorResult] = Field(
        default_factory=list,
        description="Volatility indicators"
    )
    support_resistance_indicators: List[IndicatorResult] = Field(
        default_factory=list,
        description="Support/Resistance indicators"
    )
    
    # Pattern results
    classical_patterns: List[PatternResult] = Field(
        default_factory=list,
        description="Classical chart patterns"
    )
    candlestick_patterns: List[PatternResult] = Field(
        default_factory=list,
        description="Candlestick patterns"
    )
    elliott_wave_analysis: Optional[ElliottWaveResult] = Field(
        default=None,
        description="Elliott Wave analysis"
    )
    
    # Alternative chart analysis
    renko_analysis: Optional[ChartAnalysisResult] = Field(
        default=None,
        description="Renko chart analysis"
    )
    three_line_break_analysis: Optional[ChartAnalysisResult] = Field(
        default=None,
        description="Three Line Break analysis"
    )
    point_figure_analysis: Optional[ChartAnalysisResult] = Field(
        default=None,
        description="Point & Figure analysis"
    )
    
    # Market Phase Analysis (Dow Theory)
    market_phase_analysis: Optional[MarketPhaseResult] = Field(
        default=None,
        description="Market phase analysis based on Dow Theory"
    )
    
    # Overall summary
    overall_trend_signal: Optional[SignalStrength] = Field(
        default=None,
        description="Overall trend signal"
    )
    overall_momentum_signal: Optional[SignalStrength] = Field(
        default=None,
        description="Overall momentum signal"
    )
    overall_cycle_signal: Optional[SignalStrength] = Field(
        default=None,
        description="Overall cycle signal"
    )
    overall_signal: Optional[SignalStrength] = Field(
        default=None,
        description="Overall combined signal"
    )
    overall_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall confidence"
    )
    
    # ML-based weights (if available)
    ml_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="ML-predicted optimal weights"
    )
    weights_source: Optional[str] = Field(
        default="default",
        description="Source of weights: 'default', 'ml', 'adaptive'"
    )
    
    analysis_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Analysis timestamp"
    )
    
    def calculate_overall_signal(self):
        """Calculate overall signals based on all indicators with accuracy weighting"""
        
        def calc_category_score_and_accuracy(indicators: List[IndicatorResult]) -> tuple[float, float]:
            """
            Calculate category score and average accuracy
            Returns: (weighted_score, average_accuracy)
            """
            if not indicators:
                return 0.0, 0.0
            
            weighted_sum = sum(
                ind.signal.get_score() * ind.confidence 
                for ind in indicators
            )
            total_weight = sum(ind.confidence for ind in indicators)
            
            # Calculate average accuracy (confidence) for this category
            avg_accuracy = total_weight / len(indicators) if indicators else 0.0
            
            score = weighted_sum / total_weight if total_weight > 0 else 0.0
            return score, avg_accuracy
        
        # Calculate category scores and accuracies
        trend_score, trend_accuracy = calc_category_score_and_accuracy(self.trend_indicators)
        momentum_score, momentum_accuracy = calc_category_score_and_accuracy(self.momentum_indicators)
        cycle_score, cycle_accuracy = calc_category_score_and_accuracy(self.cycle_indicators)
        volume_score, volume_accuracy = calc_category_score_and_accuracy(self.volume_indicators)
        
        # Base weights: Trend 30%, Momentum 25%, Cycle 25%, Volume 20%
        base_weights = {
            'trend': 0.30,
            'momentum': 0.25,
            'cycle': 0.25,
            'volume': 0.20
        }
        
        # Apply accuracy to weights - categories with higher accuracy get more weight
        # Normalize accuracies so they sum to the original weight proportions
        accuracies = {
            'trend': trend_accuracy,
            'momentum': momentum_accuracy,
            'cycle': cycle_accuracy,
            'volume': volume_accuracy
        }
        
        # Calculate accuracy-adjusted weights
        total_weighted_accuracy = sum(
            base_weights[cat] * accuracies[cat] 
            for cat in base_weights.keys()
        )
        
        if total_weighted_accuracy > 0:
            adjusted_weights = {
                cat: (base_weights[cat] * accuracies[cat]) / total_weighted_accuracy
                for cat in base_weights.keys()
            }
        else:
            # Fallback to base weights if no accuracy info
            adjusted_weights = base_weights
        
        # Calculate overall score with accuracy-adjusted weights
        # Volume is used as confirmation, not direct addition
        overall_score = (
            (trend_score * adjusted_weights['trend']) + 
            (momentum_score * adjusted_weights['momentum']) + 
            (cycle_score * adjusted_weights['cycle'])
        )
        
        # Volume confirms or weakens the signal using adjusted weight
        volume_weight = adjusted_weights['volume']
        volume_confirmation = abs(volume_score) * volume_weight
        
        if overall_score * volume_score > 0:  # Same direction
            overall_score *= (1 + volume_confirmation)
        else:  # Different direction - volume disagrees
            overall_score *= (1 - volume_confirmation)
        
        # Clamp to [-2, 2] range
        overall_score = max(-2.0, min(2.0, overall_score))
        
        # Normalize to [-1, 1] for SignalStrength conversion
        normalized_score = overall_score / 2.0
        
        self.overall_trend_signal = SignalStrength.from_value(trend_score / 2.0)
        self.overall_momentum_signal = SignalStrength.from_value(momentum_score / 2.0)
        self.overall_cycle_signal = SignalStrength.from_value(cycle_score / 2.0)
        self.overall_signal = SignalStrength.from_value(normalized_score)
        
        # Calculate overall confidence based on:
        # 1. Agreement between indicators (lower std dev = higher confidence)
        # 2. Average accuracy of all categories
        all_scores = []
        all_confidences = []
        
        for indicators in [
            self.trend_indicators,
            self.momentum_indicators,
            self.cycle_indicators,
            self.volume_indicators
        ]:
            all_scores.extend([ind.signal.get_score() for ind in indicators])
            all_confidences.extend([ind.confidence for ind in indicators])
        
        if all_scores and all_confidences:
            import numpy as np
            
            # Agreement factor: Lower standard deviation = higher confidence
            std_dev = np.std(all_scores)
            agreement_confidence = max(0.0, min(1.0, 1.0 - (std_dev / 4.0)))
            
            # Accuracy factor: Average accuracy of all indicators
            accuracy_confidence = np.mean(all_confidences)
            
            # Combined confidence: 60% agreement + 40% accuracy
            self.overall_confidence = (agreement_confidence * 0.6) + (accuracy_confidence * 0.4)
        else:
            self.overall_confidence = 0.5


class AnalysisRequest(BaseModel):
    """Request for technical analysis"""
    symbol: str = Field(..., description="Trading symbol", example="BTCUSDT")
    timeframe: str = Field(
        ...,
        description="Timeframe",
        example="1h",
        pattern="^(1m|5m|15m|30m|1h|4h|1d|1w)$"
    )
    candles: List[Candle] = Field(
        ...,
        description="Historical candle data",
        min_items=50
    )
    indicators: Optional[List[str]] = Field(
        default=None,
        description="Specific indicators to calculate (if None, calculate all)"
    )
