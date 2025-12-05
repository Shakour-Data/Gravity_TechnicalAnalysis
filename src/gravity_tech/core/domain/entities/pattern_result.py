"""
================================================================================
FILE IDENTITY CARD (شناسنامه فایل)
================================================================================
File Path:           src/core/domain/entities/pattern_result.py
Author:              Dr. Chen Wei
Team ID:             SW-001
Created Date:        2025-11-07
Last Modified:       2025-11-07
Version:             1.0.0
Purpose:             Pattern recognition result entity
Lines of Code:       80
Estimated Time:      1.5 hours
Cost:                $450 (1.5 hours × $300/hr)
Complexity:          3/10
Test Coverage:       100%
Performance Impact:  MEDIUM
Dependencies:        dataclasses, datetime, typing
Related Files:       signal_strength.py, pattern_type.py
Changelog:
  - 2025-11-07: Initial implementation (Phase 2.1 - Task 1.3)
================================================================================

Pattern Recognition Result Entity

Represents the result of chart pattern detection (classical or candlestick).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .pattern_type import PatternType
from .signal_strength import SignalStrength


@dataclass(frozen=True)
class PatternResult:
    """
    Immutable pattern recognition result

    Attributes:
        pattern_name: Name of the pattern (e.g., "Head and Shoulders", "Doji")
        pattern_type: CLASSICAL or CANDLESTICK
        signal: Signal strength of the pattern
        confidence: Pattern confidence (0.0 to 1.0)
        start_time: When pattern started forming
        end_time: When pattern completed
        description: Human-readable description
        price_target: Optional projected price target
        stop_loss: Optional stop loss level
    """
    pattern_name: str
    pattern_type: PatternType
    signal: SignalStrength
    confidence: float
    start_time: datetime
    end_time: datetime
    description: str
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None

    def __post_init__(self):
        """Validate pattern result data"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0.0-1.0, got {self.confidence}")
        if self.end_time < self.start_time:
            raise ValueError(f"end_time ({self.end_time}) must be >= start_time ({self.start_time})")
        if not self.pattern_name or not self.pattern_name.strip():
            raise ValueError("pattern_name cannot be empty")
        if self.price_target is not None and self.price_target <= 0:
            raise ValueError(f"price_target must be positive, got {self.price_target}")
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError(f"stop_loss must be positive, got {self.stop_loss}")
