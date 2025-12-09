"""
Deprecated compatibility shim for legacy imports.

Use `gravity_tech.core.contracts.analysis` and
`gravity_tech.core.domain.entities` instead. This module will be removed in
Phase 2.2.
"""

from __future__ import annotations

import warnings

from gravity_tech.core.contracts.analysis import (
    AnalysisRequest,
    ChartAnalysisResult,
    MarketPhaseResult,
    TechnicalAnalysisResult,
)
from gravity_tech.core.domain.entities import (
    CoreSignalStrength as SignalStrength,
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

warnings.warn(
    (
        "Importing from gravity_tech.models.schemas is deprecated. "
        "Switch to gravity_tech.core.contracts.analysis and "
        "gravity_tech.core.domain.entities."
    ),
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
