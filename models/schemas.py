""""""

Core data models for technical analysisCore data models for technical analysis



⚠️ DEPRECATION WARNING ⚠️⚠️ DEPRECATION WARNING ⚠️

================================================================================This module is DEPRECATED as of Phase 2.1 (November 7, 2025).

This module is DEPRECATED as of Phase 2.1 (November 7, 2025).All models have been migrated to: src.core.domain.entities

All core entities have been migrated to: src.core.domain.entities

Please update your imports:

Please update your imports:OLD: from models.schemas import Candle, SignalStrength, IndicatorResult

  OLD: from models.schemas import Candle, SignalStrength, IndicatorResultNEW: from src.core.domain.entities import Candle, CoreSignalStrength, IndicatorResult

  NEW: from src.core.domain.entities import Candle, CoreSignalStrength, IndicatorResult

This backward compatibility layer will be removed in Phase 2.2 (Day 3).

This backward compatibility layer will be removed in Phase 2.2 (Day 3)."""

================================================================================import warnings

"""

import warnings# Issue deprecation warning

warnings.warn(

# Issue deprecation warning on import    "Importing from models.schemas is deprecated. "

warnings.warn(    "Use src.core.domain.entities instead. "

    "Importing from models.schemas is deprecated. "    "This module will be removed in Phase 2.2.",

    "Use src.core.domain.entities instead. "    DeprecationWarning,

    "This module will be removed in Phase 2.2.",    stacklevel=2

    DeprecationWarning,)

    stacklevel=2

)# ============================================================================

# BACKWARD COMPATIBILITY LAYER (Phase 2.1)

# ============================================================================# ============================================================================

# BACKWARD COMPATIBILITY LAYER (Phase 2.1 - Temporary)# Import all entities from new location and re-export

# ============================================================================from src.core.domain.entities import (

# Import core entities from new location and re-export with aliases    Candle,

    CoreSignalStrength as SignalStrength,  # Alias for compatibility

from src.core.domain.entities import (    IndicatorCategory,

    # Core entities    IndicatorResult,

    Candle,    PatternType,

    CoreSignalStrength as SignalStrength,  # Alias for backward compatibility    PatternResult,

    IndicatorCategory,    WavePoint,

    IndicatorResult,    ElliottWaveResult,

    PatternType,)

    PatternResult,

    WavePoint,# Re-export for backward compatibility

    ElliottWaveResult,__all__ = [

)    "Candle",

    "SignalStrength",

# Re-export for backward compatibility    "IndicatorCategory", 

__all__ = [    "IndicatorResult",

    "Candle",    "PatternType",

    "SignalStrength",    "PatternResult",

    "IndicatorCategory",     "WavePoint",

    "IndicatorResult",    "ElliottWaveResult",

    "PatternType",]

    "PatternResult",

    "WavePoint",# ============================================================================

    "ElliottWaveResult",# LEGACY PYDANTIC MODELS (Will be deleted in Phase 2.2)

    # Pydantic models still needed by API layer# ============================================================================

    "ChartAnalysisResult",from pydantic import BaseModel, Field, validator

    "MarketPhaseResult",from typing import List, Optional, Dict

    "TechnicalAnalysisResult",from datetime import datetime

]from enum import Enum

from decimal import Decimal

# ============================================================================

# PYDANTIC MODELS (For API layer only - not migrated yet)

# ============================================================================class SignalStrength(str, Enum):

# These models are still needed by the FastAPI layer for request/response    """Signal strength enum with Persian names"""

# They will be migrated to api/models/ in a future phase    VERY_BULLISH = "بسیار صعودی"

    BULLISH = "صعودی"

from pydantic import BaseModel, Field    BULLISH_BROKEN = "صعودی شکسته شده"

from typing import List, Optional, Dict    NEUTRAL = "خنثی"

from datetime import datetime    BEARISH_BROKEN = "نزولی شکسته شده"

    BEARISH = "نزولی"

    VERY_BEARISH = "بسیار نزولی"

class ChartAnalysisResult(BaseModel):    

    """Alternative chart analysis result (Renko, Three Line Break, Point & Figure)"""    @staticmethod

    chart_type: str = Field(    def from_value(value: float) -> 'SignalStrength':

        ...,        """

        description="RENKO, THREE_LINE_BREAK, or POINT_FIGURE"        Convert numeric value (-1 to 1) to SignalStrength

    )        

    signal: SignalStrength = Field(..., description="Chart signal")        Args:

    current_trend: str = Field(            value: Normalized value between -1 and 1

        ...,            

        description="UP, DOWN, or CONSOLIDATION"        Returns:

    )            Corresponding SignalStrength

    consecutive_bars: int = Field(        """

        ...,        if value >= 0.8:

        description="Number of consecutive bars in current direction"            return SignalStrength.VERY_BULLISH

    )        elif value >= 0.4:

    confidence: float = Field(            return SignalStrength.BULLISH

        ge=0.0,        elif value >= 0.1:

        le=1.0,            return SignalStrength.BULLISH_BROKEN

        description="Analysis confidence"        elif value >= -0.1:

    )            return SignalStrength.NEUTRAL

    description: str = Field(..., description="Analysis description")        elif value >= -0.4:

            return SignalStrength.BEARISH_BROKEN

        elif value >= -0.8:

class MarketPhaseResult(BaseModel):            return SignalStrength.BEARISH

    """Market Phase Analysis Result based on Dow Theory"""        else:

    market_phase: str = Field(            return SignalStrength.VERY_BEARISH

        ...,    

        description="Current market phase: انباشت, صعود, توزیع, نزول, انتقال"    def get_score(self) -> float:

    )        """Get numeric score for this signal"""

    phase_strength: str = Field(        scores = {

        ...,            SignalStrength.VERY_BULLISH: 2.0,

        description="Strength of current phase"            SignalStrength.BULLISH: 1.0,

    )            SignalStrength.BULLISH_BROKEN: 0.5,

    description: str = Field(            SignalStrength.NEUTRAL: 0.0,

        ...,            SignalStrength.BEARISH_BROKEN: -0.5,

        description="Detailed phase description in Persian"            SignalStrength.BEARISH: -1.0,

    )            SignalStrength.VERY_BEARISH: -2.0,

    overall_score: float = Field(        }

        ...,        return scores.get(self, 0.0)

        ge=0.0,

        le=100.0,

        description="Overall phase score (0-100)"class Candle(BaseModel):

    )    """

    trend_structure: str = Field(    Represents a single OHLCV candlestick

        ...,    

        description="Trend structure: uptrend, downtrend, contraction, etc."    ⚠️ CRITICAL: All input prices and volume MUST be adjusted for:

    )    - Stock splits (تقسیم سهام)

    volume_confirmation: bool = Field(    - Dividends (سود سهام)

        ...,    - Rights issues (افزایش سرمایه)

        description="Whether volume confirms the trend (Dow Theory)"    - All corporate actions

    )    

    recommendations: List[str] = Field(    Using unadjusted data will produce incorrect technical analysis results!

        default_factory=list,    Adjustment must be done BEFORE sending data to this microservice.

        description="Trading recommendations based on phase"    """

    )    timestamp: datetime

    detailed_scores: Dict[str, float] = Field(    open: float = Field(..., description="Opening price (must be adjusted)")

        default_factory=dict,    high: float = Field(..., description="High price (must be adjusted)")

        description="Detailed scores for different aspects"    low: float = Field(..., description="Low price (must be adjusted)")

    )    close: float = Field(..., description="Closing price (must be adjusted)")

    dow_theory_compliance: bool = Field(    volume: float = Field(default=0.0, description="Volume (must be adjusted)")

        default=True,    

        description="Confirms analysis follows Dow Theory principles"    @validator('open', 'high', 'low', 'close')

    )    def prices_must_be_positive(cls, v):

    timestamp: datetime = Field(        """Validate prices are positive"""

        default_factory=datetime.utcnow,        if v <= 0:

        description="Analysis timestamp"            raise ValueError('Prices must be positive')

    )        return v

    

    @validator('volume')

class TechnicalAnalysisResult(BaseModel):    def volume_must_be_non_negative(cls, v):

    """Comprehensive technical analysis result (API response model)"""        """Validate volume is non-negative"""

    symbol: str = Field(..., description="Trading symbol")        if v < 0:

    timeframe: str = Field(..., description="Timeframe (e.g., 1h, 4h, 1d)")            raise ValueError('Volume cannot be negative')

            return v

    # Indicators by category    

    trend_indicators: List[IndicatorResult] = Field(    @validator('high')

        default_factory=list,    def high_must_be_highest(cls, v, values):

        description="Trend indicators"        """Validate high is the highest price"""

    )        if 'low' in values and 'open' in values and 'close' in values:

    momentum_indicators: List[IndicatorResult] = Field(            if v < max(values['low'], values['open'], values['close']):

        default_factory=list,                raise ValueError('high must be >= all other prices')

        description="Momentum indicators"        return v

    )    

    cycle_indicators: List[IndicatorResult] = Field(    @validator('low')

        default_factory=list,    def low_must_be_lowest(cls, v, values):

        description="Cycle indicators"        """Validate low is the lowest price"""

    )        if 'high' in values and 'open' in values and 'close' in values:

    volume_indicators: List[IndicatorResult] = Field(            if v > min(values['high'], values['open'], values['close']):

        default_factory=list,                raise ValueError('low must be <= all other prices')

        description="Volume indicators"        return v

    )    

    volatility_indicators: List[IndicatorResult] = Field(    @property

        default_factory=list,    def typical_price(self) -> float:

        description="Volatility indicators"        """Calculate typical price: (H + L + C) / 3"""

    )        return (self.high + self.low + self.close) / 3

    support_resistance_indicators: List[IndicatorResult] = Field(    

        default_factory=list,    @property

        description="Support/Resistance indicators"    def is_bullish(self) -> bool:

    )        """Check if candle is bullish"""

            return self.close > self.open

    # Pattern results    

    classical_patterns: List[PatternResult] = Field(    @property

        default_factory=list,    def is_bearish(self) -> bool:

        description="Classical chart patterns"        """Check if candle is bearish"""

    )        return self.close < self.open

    candlestick_patterns: List[PatternResult] = Field(    

        default_factory=list,    @property

        description="Candlestick patterns"    def body_size(self) -> float:

    )        """Get candle body size"""

    elliott_wave_analysis: Optional[ElliottWaveResult] = Field(        return abs(self.close - self.open)

        default=None,    

        description="Elliott Wave analysis"    @property

    )    def upper_shadow(self) -> float:

            """Get upper shadow size"""

    # Alternative chart analysis        return self.high - max(self.open, self.close)

    renko_analysis: Optional[ChartAnalysisResult] = Field(    

        default=None,    @property

        description="Renko chart analysis"    def lower_shadow(self) -> float:

    )        """Get lower shadow size"""

    three_line_break_analysis: Optional[ChartAnalysisResult] = Field(        return min(self.open, self.close) - self.low

        default=None,    

        description="Three Line Break analysis"    @property

    )    def total_range(self) -> float:

    point_figure_analysis: Optional[ChartAnalysisResult] = Field(        """Get total price range"""

        default=None,        return self.high - self.low

        description="Point & Figure analysis"    

    )    def true_range(self, previous_candle: Optional['Candle'] = None) -> float:

            """

    # Market Phase Analysis (Dow Theory)        Calculate True Range for ATR

    market_phase_analysis: Optional[MarketPhaseResult] = Field(        

        default=None,        Args:

        description="Market phase analysis based on Dow Theory"            previous_candle: Previous candle for TR calculation

    )            

            Returns:

    # Overall summary            True Range value

    overall_trend_signal: Optional[SignalStrength] = Field(        """

        default=None,        if previous_candle is None:

        description="Overall trend signal"            return self.high - self.low

    )        

    overall_momentum_signal: Optional[SignalStrength] = Field(        return max(

        default=None,            self.high - self.low,

        description="Overall momentum signal"            abs(self.high - previous_candle.close),

    )            abs(self.low - previous_candle.close)

    overall_cycle_signal: Optional[SignalStrength] = Field(        )

        default=None,

        description="Overall cycle signal"

    )class IndicatorCategory(str, Enum):

    overall_signal: Optional[SignalStrength] = Field(    """Indicator category types"""

        default=None,    TREND = "روند"

        description="Overall combined signal"    MOMENTUM = "مومنتوم"

    )    CYCLE = "سیکل"

    overall_confidence: Optional[float] = Field(    VOLUME = "حجم"

        default=None,    VOLATILITY = "نوسان"

        ge=0.0,    SUPPORT_RESISTANCE = "حمایت و مقاومت"

        le=1.0,

        description="Overall confidence"

    )class IndicatorResult(BaseModel):

        """Result from a single indicator calculation"""

    # ML-based weights (if available)    indicator_name: str = Field(..., description="Name of the indicator")

    ml_weights: Optional[Dict[str, float]] = Field(    category: IndicatorCategory = Field(..., description="Indicator category")

        default=None,    signal: SignalStrength = Field(..., description="Signal strength")

        description="ML-predicted optimal weights"    value: float = Field(..., description="Primary indicator value")

    )    additional_values: Optional[Dict[str, float]] = Field(

    weights_source: Optional[str] = Field(        default=None,

        default="default",        description="Additional values for multi-line indicators"

        description="Source of weights: 'default', 'ml', 'adaptive'"    )

    )    confidence: float = Field(

            default=0.75,

    analysis_timestamp: datetime = Field(        ge=0.0,

        default_factory=datetime.utcnow,        le=1.0,

        description="Analysis timestamp"        description="Confidence level (0-1)"

    )    )

        description: Optional[str] = Field(

    def calculate_overall_signal(self):        default=None,

        """Calculate overall signals based on all indicators with accuracy weighting"""        description="Human-readable description"

            )

        def calc_category_score_and_accuracy(indicators: List[IndicatorResult]) -> tuple[float, float]:    timestamp: datetime = Field(

            """        default_factory=datetime.utcnow,

            Calculate category score and average accuracy        description="Analysis timestamp"

            Returns: (weighted_score, average_accuracy)    )

            """

            if not indicators:

                return 0.0, 0.0class PatternType(str, Enum):

                """Chart pattern types"""

            weighted_sum = sum(    CLASSICAL = "کلاسیک"

                ind.signal.get_score() * ind.confidence     CANDLESTICK = "کندل استیک"

                for ind in indicators

            )

            total_weight = sum(ind.confidence for ind in indicators)class PatternResult(BaseModel):

                """Result from pattern recognition"""

            # Calculate average accuracy (confidence) for this category    pattern_name: str = Field(..., description="Pattern name")

            avg_accuracy = total_weight / len(indicators) if indicators else 0.0    pattern_type: PatternType = Field(..., description="Pattern type")

                signal: SignalStrength = Field(..., description="Pattern signal")

            score = weighted_sum / total_weight if total_weight > 0 else 0.0    confidence: float = Field(

            return score, avg_accuracy        ge=0.0,

                le=1.0,

        # Calculate category scores and accuracies        description="Pattern confidence"

        trend_score, trend_accuracy = calc_category_score_and_accuracy(self.trend_indicators)    )

        momentum_score, momentum_accuracy = calc_category_score_and_accuracy(self.momentum_indicators)    start_time: datetime = Field(..., description="Pattern start time")

        cycle_score, cycle_accuracy = calc_category_score_and_accuracy(self.cycle_indicators)    end_time: datetime = Field(..., description="Pattern end time")

        volume_score, volume_accuracy = calc_category_score_and_accuracy(self.volume_indicators)    price_target: Optional[float] = Field(

                default=None,

        # Base weights: Trend 30%, Momentum 25%, Cycle 25%, Volume 20%        description="Price target"

        base_weights = {    )

            'trend': 0.30,    stop_loss: Optional[float] = Field(

            'momentum': 0.25,        default=None,

            'cycle': 0.25,        description="Stop loss level"

            'volume': 0.20    )

        }    description: str = Field(..., description="Pattern description")

        

        # Apply accuracy to weights - categories with higher accuracy get more weight

        accuracies = {class WavePoint(BaseModel):

            'trend': trend_accuracy,    """Elliott Wave point"""

            'momentum': momentum_accuracy,    wave_number: int = Field(..., description="Wave number")

            'cycle': cycle_accuracy,    price: float = Field(..., description="Wave price level")

            'volume': volume_accuracy    timestamp: datetime = Field(..., description="Wave timestamp")

        }    wave_type: str = Field(..., description="PEAK or TROUGH")

        

        # Calculate accuracy-adjusted weights

        total_weighted_accuracy = sum(class ElliottWaveResult(BaseModel):

            base_weights[cat] * accuracies[cat]     """Elliott Wave analysis result"""

            for cat in base_weights.keys()    wave_pattern: str = Field(

        )        ...,

                description="IMPULSIVE or CORRECTIVE"

        if total_weighted_accuracy > 0:    )

            adjusted_weights = {    current_wave: int = Field(

                cat: (base_weights[cat] * accuracies[cat]) / total_weighted_accuracy        ...,

                for cat in base_weights.keys()        description="Current wave number"

            }    )

        else:    waves: List[WavePoint] = Field(

            # Fallback to base weights if no accuracy info        ...,

            adjusted_weights = base_weights        description="List of wave points"

            )

        # Calculate overall score with accuracy-adjusted weights    signal: SignalStrength = Field(..., description="Overall wave signal")

        overall_score = (    confidence: float = Field(

            (trend_score * adjusted_weights['trend']) +         ge=0.0,

            (momentum_score * adjusted_weights['momentum']) +         le=1.0,

            (cycle_score * adjusted_weights['cycle'])        description="Wave pattern confidence"

        )    )

            projected_target: Optional[float] = Field(

        # Volume confirms or weakens the signal        default=None,

        volume_weight = adjusted_weights['volume']        description="Projected price target"

        volume_confirmation = abs(volume_score) * volume_weight    )

            description: str = Field(..., description="Wave analysis description")

        if overall_score * volume_score > 0:  # Same direction

            overall_score *= (1 + volume_confirmation)

        else:  # Different direction - volume disagreesclass ChartAnalysisResult(BaseModel):

            overall_score *= (1 - volume_confirmation)    """Alternative chart analysis result"""

            chart_type: str = Field(

        # Clamp to [-2, 2] range        ...,

        overall_score = max(-2.0, min(2.0, overall_score))        description="RENKO, THREE_LINE_BREAK, or POINT_FIGURE"

            )

        # Normalize to [-1, 1] for SignalStrength conversion    signal: SignalStrength = Field(..., description="Chart signal")

        normalized_score = overall_score / 2.0    current_trend: str = Field(

                ...,

        self.overall_trend_signal = SignalStrength.from_value(trend_score / 2.0)        description="UP, DOWN, or CONSOLIDATION"

        self.overall_momentum_signal = SignalStrength.from_value(momentum_score / 2.0)    )

        self.overall_cycle_signal = SignalStrength.from_value(cycle_score / 2.0)    consecutive_bars: int = Field(

        self.overall_signal = SignalStrength.from_value(normalized_score)        ...,

                description="Number of consecutive bars in current direction"

        # Calculate overall confidence as weighted average of category confidences    )

        category_confidences = [    confidence: float = Field(

            (trend_accuracy, adjusted_weights['trend']),        ge=0.0,

            (momentum_accuracy, adjusted_weights['momentum']),        le=1.0,

            (cycle_accuracy, adjusted_weights['cycle']),        description="Analysis confidence"

            (volume_accuracy, adjusted_weights['volume'])    )

        ]    description: str = Field(..., description="Analysis description")

        

        total_weight = sum(weight for _, weight in category_confidences)

        if total_weight > 0:class MarketPhaseResult(BaseModel):

            self.overall_confidence = sum(    """Market Phase Analysis Result based on Dow Theory"""

                conf * weight for conf, weight in category_confidences    market_phase: str = Field(

            ) / total_weight        ...,

        else:        description="Current market phase: انباشت, صعود, توزیع, نزول, انتقال"

            self.overall_confidence = 0.0    )

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
