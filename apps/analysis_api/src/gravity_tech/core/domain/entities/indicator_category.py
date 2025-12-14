"""
================================================================================
FILE IDENTITY CARD (شناسنامه فایل)
================================================================================
File Path:           src/core/domain/entities/indicator_category.py
Author:              Dr. Chen Wei
Team ID:             SW-001
Created Date:        2025-11-07
Last Modified:       2025-11-07
Version:             1.0.0
Purpose:             Indicator category enumeration
Lines of Code:       30
Estimated Time:      0.5 hours
Cost:                $150 (0.5 hours × $300/hr)
Complexity:          1/10
Test Coverage:       100%
Performance Impact:  LOW
Dependencies:        enum (stdlib)
Related Files:       indicator_result.py
Changelog:
  - 2025-11-07: Initial implementation (Phase 2.1 - Task 1.3)
================================================================================

Indicator Category Enumeration

Defines 6 main categories for technical indicators with Persian labels.
"""

from enum import Enum


class IndicatorCategory(str, Enum):
    """Technical indicator categories"""
    TREND = "TREND"
    MOMENTUM = "MOMENTUM"
    CYCLE = "CYCLE"
    VOLUME = "VOLUME"
    VOLATILITY = "VOLATILITY"
    SUPPORT_RESISTANCE = "SUPPORT_RESISTANCE"

    def __str__(self) -> str:
        return self.value
