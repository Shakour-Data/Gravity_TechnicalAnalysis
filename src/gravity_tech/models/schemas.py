"""
===============================================================================
DEPRECATION WARNING

`gravity_tech.models.schemas` has been replaced by
`gravity_tech.core.contracts.analysis`. Import contracts from the new module
instead of this compatibility shim. This module will be removed in Phase 2.2.
===============================================================================
"""

import warnings

from gravity_tech.core.contracts.analysis import (  # noqa: F401
    AnalysisRequest,
    ChartAnalysisResult,
    MarketPhaseResult,
    TechnicalAnalysisResult,
)
from gravity_tech.core.domain.entities import (  # noqa: F401
    Candle,
    ElliottWaveResult,
    FibonacciLevel,
    FibonacciResult,
    IndicatorCategory,
    IndicatorResult,
    MarketData,
    PatternResult,
    PatternType,
    PredictionResult,
    PredictionSignal,
    SSEMessage,
    SubscriptionType,
    TransformerResult,
    WavePoint,
    WebSocketMessage,
)
from gravity_tech.core.domain.entities import (  # noqa: F401
    CoreSignalStrength as SignalStrength,
)

warnings.warn(
    "Importing from gravity_tech.models.schemas is deprecated. "
    "Switch to gravity_tech.core.contracts.analysis and gravity_tech.core.domain.entities.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    # Contracts
    "AnalysisRequest",
    "ChartAnalysisResult",
    "MarketPhaseResult",
    "TechnicalAnalysisResult",
    # Domain entities
    "Candle",
    "SignalStrength",
    "IndicatorCategory",
    "IndicatorResult",
    "PatternType",
    "PatternResult",
    "WavePoint",
    "ElliottWaveResult",
    "FibonacciLevel",
    "FibonacciResult",
    "SubscriptionType",
    "WebSocketMessage",
    "SSEMessage",
    "MarketData",
    "TransformerResult",
    "PredictionResult",
    "PredictionSignal",
]
