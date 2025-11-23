"""
================================================================================
FILE IDENTITY CARD (شناسنامه فایل)
================================================================================
File Path:           src/core/domain/entities/indicator_result.py
Author:              Dr. Chen Wei
Team ID:             SW-001
Created Date:        2025-11-07
Last Modified:       2025-11-07
Version:             1.0.0
Purpose:             Indicator calculation result entity
Lines of Code:       75
Estimated Time:      1.5 hours
Cost:                $450 (1.5 hours × $300/hr)
Complexity:          3/10
Test Coverage:       100%
Performance Impact:  MEDIUM
Dependencies:        dataclasses, datetime, typing
Related Files:       signal_strength.py, indicator_category.py
Changelog:
  - 2025-11-07: Initial implementation (Phase 2.1 - Task 1.3)
================================================================================

Indicator Result Entity

Represents the result of a single technical indicator calculation.
Used by all 60+ indicators in the system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict
from .signal_strength import SignalStrength
from .indicator_category import IndicatorCategory


@dataclass(frozen=True)
class IndicatorResult:
    """
    Immutable indicator calculation result
    
    Attributes:
        indicator_name: Name of the indicator (e.g., "RSI", "MACD", "Bollinger Bands")
        category: Indicator category (TREND, MOMENTUM, etc.)
        signal: Signal strength (VERY_BULLISH to VERY_BEARISH)
        value: Primary indicator value (e.g., RSI value, MACD line)
        additional_values: Optional dict for multi-line indicators
                          (e.g., {"signal": 12.5, "histogram": 2.3} for MACD)
        confidence: Confidence level (0.0 to 1.0)
        description: Human-readable description (optional)
        timestamp: When this result was calculated
    """
    indicator_name: str
    category: IndicatorCategory
    signal: SignalStrength
    value: float
    additional_values: Optional[Dict[str, float]] = None
    confidence: float = 0.75
    description: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate indicator result data"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0.0-1.0, got {self.confidence}")
        if not self.indicator_name or not self.indicator_name.strip():
            raise ValueError("indicator_name cannot be empty")
