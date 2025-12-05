"""
================================================================================
FILE IDENTITY CARD (شناسنامه فایل)
================================================================================
File Path:           src/core/domain/entities/pattern_type.py
Author:              Dr. Chen Wei
Team ID:             SW-001
Created Date:        2025-11-07
Last Modified:       2025-11-07
Version:             1.0.0
Purpose:             Pattern type enumeration
Lines of Code:       25
Estimated Time:      0.5 hours
Cost:                $150 (0.5 hours × $300/hr)
Complexity:          1/10
Test Coverage:       100%
Performance Impact:  LOW
Dependencies:        enum (stdlib)
Related Files:       pattern_result.py
Changelog:
  - 2025-11-07: Initial implementation (Phase 2.1 - Task 1.3)
================================================================================

Pattern Type Enumeration

Defines 2 main pattern types: Classical and Candlestick.
"""

from enum import Enum


class PatternType(str, Enum):
    """Chart pattern types"""
    CLASSICAL = "CLASSICAL"
    CANDLESTICK = "CANDLESTICK"

    def __str__(self) -> str:
        return self.value
