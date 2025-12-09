"""
Harmonic Pattern Detection Module

This module implements detection algorithms for harmonic chart patterns including:
- Gartley Pattern (222 pattern)
- Butterfly Pattern
- Bat Pattern
- Crab Pattern

All patterns are validated using Fibonacci ratios with configurable tolerance.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class PatternType(Enum):
    """Harmonic pattern types."""
    GARTLEY = "gartley"
    BUTTERFLY = "butterfly"
    BAT = "bat"
    CRAB = "crab"
    UNKNOWN = "unknown"


class PatternDirection(Enum):
    """Pattern direction (bullish/bearish)."""
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class FibonacciRatio:
    """Fibonacci ratio with tolerance."""
    target: float
    tolerance: float = 0.05

    def matches(self, value: float) -> bool:
        """Check if value matches ratio within tolerance."""
        return abs(value - self.target) <= self.tolerance

    def deviation(self, value: float) -> float:
        """Calculate deviation from target ratio."""
        return abs(value - self.target) / self.target


@dataclass
class PatternPoint:
    """A point in the harmonic pattern."""
    index: int
    price: float
    label: str  # X, A, B, C, D


@dataclass
class HarmonicPattern:
    """Detected harmonic pattern."""
    pattern_type: PatternType
    direction: PatternDirection
    points: dict[str, PatternPoint]  # X, A, B, C, D
    ratios: dict[str, float]  # XA, AB, BC, CD ratios
    confidence: float  # 0-100
    fibonacci_accuracy: float  # How well ratios match Fibonacci levels
    completion_point: float  # Expected D point price
    stop_loss: float
    target_1: float
    target_2: float


class FibonacciLevels:
    """Standard Fibonacci retracement and extension levels."""
    # Retracement levels
    R_382 = 0.382
    R_500 = 0.500
    R_618 = 0.618
    R_786 = 0.786
    R_886 = 0.886

    # Extension levels
    E_1272 = 1.272
    E_1414 = 1.414
    E_1618 = 1.618
    E_2000 = 2.000
    E_2618 = 2.618
    E_3618 = 3.618


class HarmonicPatternDetector:
    """
    Detects harmonic patterns in price data using Fibonacci ratios.

    Harmonic patterns are geometric price patterns that use Fibonacci numbers
    to identify precise turning points. They consist of 5 points (X, A, B, C, D)
    forming specific ratio relationships.
    """

    def __init__(self, tolerance: float = 0.05, min_pattern_bars: int = 20):
        """
        Initialize harmonic pattern detector.

        Args:
            tolerance: Allowed deviation from Fibonacci ratios (default 5%)
            min_pattern_bars: Minimum bars required for a valid pattern
        """
        self.tolerance = tolerance
        self.min_pattern_bars = min_pattern_bars

        # Pattern definitions with Fibonacci ratios
        self.pattern_definitions = {
            PatternType.GARTLEY: {
                'XA_BC': FibonacciRatio(0.618, tolerance),  # B retraces 61.8% of XA
                'AB_CD': FibonacciRatio(0.786, tolerance),  # D retraces 78.6% of AB
                'XA_AD': FibonacciRatio(0.786, tolerance),  # D is 78.6% extension of XA
            },
            PatternType.BUTTERFLY: {
                'XA_BC': FibonacciRatio(0.786, tolerance),  # B retraces 78.6% of XA
                'AB_CD': FibonacciRatio(1.618, tolerance),  # D extends 161.8% of AB (CD = 1.618 * AB)
                'XA_AD': FibonacciRatio(1.272, tolerance),  # D is 127.2% extension of XA
            },
            PatternType.BAT: {
                'XA_BC': FibonacciRatio(0.500, tolerance),  # B retraces 50% of XA
                'AB_CD': FibonacciRatio(0.886, tolerance),  # D retraces 88.6% of AB
                'XA_AD': FibonacciRatio(0.886, tolerance),  # D is 88.6% extension of XA
            },
            PatternType.CRAB: {
                'XA_BC': FibonacciRatio(0.618, tolerance),  # B retraces 61.8% of XA
                'AB_CD': FibonacciRatio(2.618, tolerance),  # D extends 261.8% of AB (CD = 2.618 * AB)
                'XA_AD': FibonacciRatio(1.618, tolerance),  # D is 161.8% extension of XA
            }
        }

    def detect_patterns(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> list[HarmonicPattern]:
        """
        Detect all harmonic patterns in price data.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices

        Returns:
            List of detected harmonic patterns
        """
        if len(highs) < self.min_pattern_bars:
            return []

        patterns = []

        # Find pivot points (local highs and lows)
        pivot_highs = self._find_pivot_highs(highs)
        pivot_lows = self._find_pivot_lows(lows)

        # Try to form patterns from pivot points
        patterns.extend(self._detect_bullish_patterns(pivot_lows, pivot_highs, closes))
        patterns.extend(self._detect_bearish_patterns(pivot_highs, pivot_lows, closes))

        return patterns

    def _find_pivot_highs(self, highs: np.ndarray, window: int = 5) -> list[tuple[int, float]]:
        """Find local high pivot points."""
        pivots = []
        for i in range(window, len(highs) - window):
            if all(highs[i] >= highs[i-window:i]) and all(highs[i] >= highs[i+1:i+window+1]):
                pivots.append((i, highs[i]))
        return pivots

    def _find_pivot_lows(self, lows: np.ndarray, window: int = 5) -> list[tuple[int, float]]:
        """Find local low pivot points."""
        pivots = []
        for i in range(window, len(lows) - window):
            if all(lows[i] <= lows[i-window:i]) and all(lows[i] <= lows[i+1:i+window+1]):
                pivots.append((i, lows[i]))
        return pivots

    def _detect_bullish_patterns(
        self,
        pivot_lows: list[tuple[int, float]],
        pivot_highs: list[tuple[int, float]],
        closes: np.ndarray
    ) -> list[HarmonicPattern]:
        """Detect bullish harmonic patterns (reversal upward)."""
        patterns = []

        # Need at least 5 pivots to form X-A-B-C-D
        if len(pivot_lows) < 3 or len(pivot_highs) < 2:
            return patterns

        # Try different combinations of pivots
        for i in range(len(pivot_lows) - 2):
            # Check if pivots are in correct sequence
            x_idx, x_price = pivot_lows[i]

            # Find A (high after X)
            a_candidates = [(idx, price) for idx, price in pivot_highs if idx > x_idx]
            if not a_candidates:
                continue
            a_idx, a_price = a_candidates[0]

            # Find B (low after A)
            b_candidates = [(idx, price) for idx, price in pivot_lows if idx > a_idx]
            if not b_candidates:
                continue
            b_idx, b_price = b_candidates[0]

            # Find C (high after B)
            c_candidates = [(idx, price) for idx, price in pivot_highs if idx > b_idx]
            if not c_candidates:
                continue
            c_idx, c_price = c_candidates[0]

            # Find D (low after C) - completion point
            d_candidates = [(idx, price) for idx, price in pivot_lows if idx > c_idx]
            if not d_candidates:
                continue
            d_idx, d_price = d_candidates[0]

            # Create pattern points
            points = {
                'X': PatternPoint(x_idx, x_price, 'X'),
                'A': PatternPoint(a_idx, a_price, 'A'),
                'B': PatternPoint(b_idx, b_price, 'B'),
                'C': PatternPoint(c_idx, c_price, 'C'),
                'D': PatternPoint(d_idx, d_price, 'D')
            }

            # Calculate ratios
            xa_move = a_price - x_price
            ab_move = a_price - b_price
            cd_move = c_price - d_price
            ad_move = d_price - x_price

            if xa_move == 0 or ab_move == 0:
                continue

            ratios = {
                'XA_BC': ab_move / xa_move if xa_move != 0 else 0,
                'AB_CD': cd_move / ab_move if ab_move != 0 else 0,
                'XA_AD': ad_move / xa_move if xa_move != 0 else 0
            }

            # Identify pattern type
            pattern_type, confidence = self._identify_pattern_type(ratios)

            if pattern_type != PatternType.UNKNOWN and confidence > 0.5:
                # Calculate targets and stop loss
                stop_loss = d_price - (a_price - x_price) * 0.1
                target_1 = d_price + (a_price - d_price) * 0.382
                target_2 = d_price + (a_price - d_price) * 0.618

                pattern = HarmonicPattern(
                    pattern_type=pattern_type,
                    direction=PatternDirection.BULLISH,
                    points=points,
                    ratios=ratios,
                    confidence=confidence * 100,
                    fibonacci_accuracy=confidence,
                    completion_point=d_price,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2
                )
                patterns.append(pattern)

        return patterns

    def _detect_bearish_patterns(
        self,
        pivot_highs: list[tuple[int, float]],
        pivot_lows: list[tuple[int, float]],
        closes: np.ndarray
    ) -> list[HarmonicPattern]:
        """Detect bearish harmonic patterns (reversal downward)."""
        patterns = []

        # Need at least 5 pivots to form X-A-B-C-D
        if len(pivot_highs) < 3 or len(pivot_lows) < 2:
            return patterns

        # Try different combinations of pivots
        for i in range(len(pivot_highs) - 2):
            # X point (high)
            x_idx, x_price = pivot_highs[i]

            # Find A (low after X)
            a_candidates = [(idx, price) for idx, price in pivot_lows if idx > x_idx]
            if not a_candidates:
                continue
            a_idx, a_price = a_candidates[0]

            # Find B (high after A)
            b_candidates = [(idx, price) for idx, price in pivot_highs if idx > a_idx]
            if not b_candidates:
                continue
            b_idx, b_price = b_candidates[0]

            # Find C (low after B)
            c_candidates = [(idx, price) for idx, price in pivot_lows if idx > b_idx]
            if not c_candidates:
                continue
            c_idx, c_price = c_candidates[0]

            # Find D (high after C) - completion point
            d_candidates = [(idx, price) for idx, price in pivot_highs if idx > c_idx]
            if not d_candidates:
                continue
            d_idx, d_price = d_candidates[0]

            # Create pattern points
            points = {
                'X': PatternPoint(x_idx, x_price, 'X'),
                'A': PatternPoint(a_idx, a_price, 'A'),
                'B': PatternPoint(b_idx, b_price, 'B'),
                'C': PatternPoint(c_idx, c_price, 'C'),
                'D': PatternPoint(d_idx, d_price, 'D')
            }

            # Calculate ratios (inverted for bearish)
            xa_move = x_price - a_price
            ab_move = b_price - a_price
            cd_move = d_price - c_price
            ad_move = x_price - d_price

            if xa_move == 0 or ab_move == 0:
                continue

            ratios = {
                'XA_BC': ab_move / xa_move if xa_move != 0 else 0,
                'AB_CD': cd_move / ab_move if ab_move != 0 else 0,
                'XA_AD': ad_move / xa_move if xa_move != 0 else 0
            }

            # Identify pattern type
            pattern_type, confidence = self._identify_pattern_type(ratios)

            if pattern_type != PatternType.UNKNOWN and confidence > 0.5:
                # Calculate targets and stop loss
                stop_loss = d_price + (x_price - a_price) * 0.1
                target_1 = d_price - (d_price - a_price) * 0.382
                target_2 = d_price - (d_price - a_price) * 0.618

                pattern = HarmonicPattern(
                    pattern_type=pattern_type,
                    direction=PatternDirection.BEARISH,
                    points=points,
                    ratios=ratios,
                    confidence=confidence * 100,
                    fibonacci_accuracy=confidence,
                    completion_point=d_price,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2
                )
                patterns.append(pattern)

        return patterns

    def _identify_pattern_type(self, ratios: dict[str, float]) -> tuple[PatternType, float]:
        """
        Identify pattern type from ratios and calculate confidence.

        Returns:
            (pattern_type, confidence) where confidence is 0-1
        """
        best_pattern = PatternType.UNKNOWN
        best_confidence = 0.0

        for pattern_type, pattern_ratios in self.pattern_definitions.items():
            # Calculate how well ratios match this pattern
            deviations = []

            for ratio_name, fib_ratio in pattern_ratios.items():
                if ratio_name in ratios:
                    actual_value = ratios[ratio_name]
                    deviation = fib_ratio.deviation(actual_value)
                    deviations.append(deviation)

            if deviations:
                # Confidence based on average deviation (lower is better)
                avg_deviation = np.mean(deviations)
                confidence = max(0, 1 - avg_deviation * 2)  # Scale to 0-1

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_pattern = pattern_type

        return best_pattern, best_confidence


def detect_gartley(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    tolerance: float = 0.05
) -> dict:
    """
    Detect Gartley 222 pattern (most common harmonic pattern).

    Pattern characteristics:
    - B: 61.8% retracement of XA
    - D: 78.6% retracement of XA
    - Fibonacci relationships between all points

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        tolerance: Allowed deviation from Fibonacci ratios

    Returns:
        Dictionary with pattern detection results
    """
    detector = HarmonicPatternDetector(tolerance=tolerance)
    patterns = detector.detect_patterns(highs, lows, closes)

    # Filter for Gartley patterns only
    gartley_patterns = [p for p in patterns if p.pattern_type == PatternType.GARTLEY]

    if gartley_patterns:
        best_pattern = max(gartley_patterns, key=lambda p: p.confidence)
        return {
            'detected': True,
            'direction': best_pattern.direction.value,
            'confidence': best_pattern.confidence,
            'completion_point': best_pattern.completion_point,
            'stop_loss': best_pattern.stop_loss,
            'target_1': best_pattern.target_1,
            'target_2': best_pattern.target_2,
            'ratios': best_pattern.ratios
        }

    return {
        'detected': False,
        'direction': None,
        'confidence': 0.0,
        'completion_point': None,
        'stop_loss': None,
        'target_1': None,
        'target_2': None,
        'ratios': {}
    }


def detect_butterfly(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    tolerance: float = 0.05
) -> dict:
    """
    Detect Butterfly pattern.

    Pattern characteristics:
    - B: 78.6% retracement of XA
    - D: 127.2% extension of XA
    - Large CD leg (161.8% of AB)

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        tolerance: Allowed deviation from Fibonacci ratios

    Returns:
        Dictionary with pattern detection results
    """
    detector = HarmonicPatternDetector(tolerance=tolerance)
    patterns = detector.detect_patterns(highs, lows, closes)

    butterfly_patterns = [p for p in patterns if p.pattern_type == PatternType.BUTTERFLY]

    if butterfly_patterns:
        best_pattern = max(butterfly_patterns, key=lambda p: p.confidence)
        return {
            'detected': True,
            'direction': best_pattern.direction.value,
            'confidence': best_pattern.confidence,
            'completion_point': best_pattern.completion_point,
            'stop_loss': best_pattern.stop_loss,
            'target_1': best_pattern.target_1,
            'target_2': best_pattern.target_2,
            'ratios': best_pattern.ratios
        }

    return {
        'detected': False,
        'direction': None,
        'confidence': 0.0,
        'completion_point': None,
        'stop_loss': None,
        'target_1': None,
        'target_2': None,
        'ratios': {}
    }


def detect_bat(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    tolerance: float = 0.05
) -> dict:
    """
    Detect Bat pattern.

    Pattern characteristics:
    - B: 38.2-50% retracement of XA
    - D: 88.6% retracement of XA
    - Precise Fibonacci relationships

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        tolerance: Allowed deviation from Fibonacci ratios

    Returns:
        Dictionary with pattern detection results
    """
    detector = HarmonicPatternDetector(tolerance=tolerance)
    patterns = detector.detect_patterns(highs, lows, closes)

    bat_patterns = [p for p in patterns if p.pattern_type == PatternType.BAT]

    if bat_patterns:
        best_pattern = max(bat_patterns, key=lambda p: p.confidence)
        return {
            'detected': True,
            'direction': best_pattern.direction.value,
            'confidence': best_pattern.confidence,
            'completion_point': best_pattern.completion_point,
            'stop_loss': best_pattern.stop_loss,
            'target_1': best_pattern.target_1,
            'target_2': best_pattern.target_2,
            'ratios': best_pattern.ratios
        }

    return {
        'detected': False,
        'direction': None,
        'confidence': 0.0,
        'completion_point': None,
        'stop_loss': None,
        'target_1': None,
        'target_2': None,
        'ratios': {}
    }


def detect_crab(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    tolerance: float = 0.05
) -> dict:
    """
    Detect Crab pattern.

    Pattern characteristics:
    - B: 61.8% retracement of XA (or 38.2%)
    - D: 161.8% extension of XA
    - Extreme CD leg (261.8% of AB)

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        tolerance: Allowed deviation from Fibonacci ratios

    Returns:
        Dictionary with pattern detection results
    """
    detector = HarmonicPatternDetector(tolerance=tolerance)
    patterns = detector.detect_patterns(highs, lows, closes)

    crab_patterns = [p for p in patterns if p.pattern_type == PatternType.CRAB]

    if crab_patterns:
        best_pattern = max(crab_patterns, key=lambda p: p.confidence)
        return {
            'detected': True,
            'direction': best_pattern.direction.value,
            'confidence': best_pattern.confidence,
            'completion_point': best_pattern.completion_point,
            'stop_loss': best_pattern.stop_loss,
            'target_1': best_pattern.target_1,
            'target_2': best_pattern.target_2,
            'ratios': best_pattern.ratios
        }

    return {
        'detected': False,
        'direction': None,
        'confidence': 0.0,
        'completion_point': None,
        'stop_loss': None,
        'target_1': None,
        'target_2': None,
        'ratios': {}
    }


def detect_all_harmonic_patterns(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    tolerance: float = 0.05
) -> dict:
    """
    Detect all harmonic patterns in one call.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        tolerance: Allowed deviation from Fibonacci ratios

    Returns:
        Dictionary with all detected patterns
    """
    detector = HarmonicPatternDetector(tolerance=tolerance)
    all_patterns = detector.detect_patterns(highs, lows, closes)

    result = {
        'total_patterns': len(all_patterns),
        'gartley': [],
        'butterfly': [],
        'bat': [],
        'crab': []
    }

    for pattern in all_patterns:
        pattern_data = {
            'direction': pattern.direction.value,
            'confidence': pattern.confidence,
            'completion_point': pattern.completion_point,
            'stop_loss': pattern.stop_loss,
            'target_1': pattern.target_1,
            'target_2': pattern.target_2,
            'ratios': pattern.ratios
        }

        if pattern.pattern_type == PatternType.GARTLEY:
            result['gartley'].append(pattern_data)
        elif pattern.pattern_type == PatternType.BUTTERFLY:
            result['butterfly'].append(pattern_data)
        elif pattern.pattern_type == PatternType.BAT:
            result['bat'].append(pattern_data)
        elif pattern.pattern_type == PatternType.CRAB:
            result['crab'].append(pattern_data)

    return result
