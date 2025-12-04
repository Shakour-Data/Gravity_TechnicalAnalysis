"""
================================================================================
Fibonacci Tools Implementation

Comprehensive Fibonacci analysis tools including:
- Retracements
- Extensions
- Arcs
- Fans
- Projections
- Confluence analysis

Last Updated: 2025-11-07 (Phase 2.1 - Task 1.3)
================================================================================
"""

import math
from typing import List, Dict, Optional, Tuple, Any
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
from datetime import datetime

from gravity_tech.core.domain.entities import FibonacciLevel, FibonacciResult
from gravity_tech.core.domain.entities.signal_strength import SignalStrength


@dataclass
class FibonacciTools:
    """
    Comprehensive Fibonacci analysis tools.

    Provides methods for calculating various Fibonacci levels,
    projections, and confluence analysis for technical analysis.
    """

    # Standard Fibonacci ratios
    FIBONACCI_RATIOS = {
        'retracements': [0.236, 0.382, 0.5, 0.618, 0.786],
        'extensions': [0.618, 1.0, 1.236, 1.382, 1.618, 2.0, 2.618, 4.236],
        'arcs': [0.382, 0.5, 0.618, 0.786],
        'fans': [0.382, 0.5, 0.618, 0.786],
        'projections': [0.618, 1.0, 1.236, 1.618, 2.618]
    }

    # Customizable precision
    DEFAULT_PRECISION = 8

    def calculate_retracements(
        self,
        high: Decimal,
        low: Decimal,
        ratios: Optional[List[float]] = None
    ) -> List[FibonacciLevel]:
        """
        Calculate Fibonacci retracement levels.

        Args:
            high: Peak price
            low: Trough price
            ratios: Custom ratios (default: standard retracements)

        Returns:
            List of FibonacciLevel objects
        """
        if ratios is None:
            ratios = self.FIBONACCI_RATIOS['retracements']

        range_size = high - low
        levels = []

        for ratio in ratios:
            price = low + (range_size * Decimal(str(ratio)))
            price = price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

            level = FibonacciLevel(
                ratio=float(ratio),
                price=float(price),
                level_type="MEDIUM",
                strength=0.0,  # Default value, adjust as needed
                touches=0,    # Default value, adjust as needed
                description=f"{ratio:.3f} retracement"
            )
            levels.append(level)

        return levels

    def calculate_extensions(
        self,
        high: Decimal,
        low: Decimal,
        ratios: Optional[List[float]] = None
    ) -> List[FibonacciLevel]:
        """
        Calculate Fibonacci extension levels.

        Args:
            high: Peak price
            low: Trough price
            ratios: Custom ratios (default: standard extensions)

        Returns:
            List of FibonacciLevel objects
        """
        if ratios is None:
            ratios = self.FIBONACCI_RATIOS['extensions']

        range_size = high - low
        levels = []

        for ratio in ratios:
            price = high + (range_size * Decimal(str(ratio)))
            price = price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

            level = FibonacciLevel(
                ratio=float(ratio),
                price=float(price),
                level_type="STRONG",
                strength=0.0,  # Default value, adjust as needed
                touches=0,    # Default value, adjust as needed
                description=f"{ratio:.3f} extension"
            )
            levels.append(level)

        return levels

    def calculate_arcs(
        self,
        center_point: Tuple[Decimal, int],
        radius_point: Tuple[Decimal, int],
        time_point: int,
        ratios: Optional[List[float]] = None
    ) -> List[FibonacciLevel]:
        """
        Calculate Fibonacci arc levels.

        Args:
            center_point: (price, time_index) for center
            radius_point: (price, time_index) for radius
            time_point: Time index to calculate arcs for
            ratios: Custom ratios (default: standard arcs)

        Returns:
            List of FibonacciLevel objects
        """
        if ratios is None:
            ratios = self.FIBONACCI_RATIOS['arcs']

        center_price, center_time = center_point
        radius_price, radius_time = radius_point

        # Calculate radius
        radius = abs(radius_price - center_price)

        levels = []

        for ratio in ratios:
            # Calculate arc price at given time
            time_diff = time_point - center_time
            arc_price = center_price + (radius * Decimal(str(ratio)) * Decimal(str(math.sin(time_diff))))

            arc_price = arc_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

            level = FibonacciLevel(
                ratio=float(ratio),
                price=float(arc_price),
                level_type="WEAK",
                strength=0.0,  # Default value, adjust as needed
                touches=0,    # Default value, adjust as needed
                description=f"{ratio:.3f} arc"
            )
            levels.append(level)

        return levels

    def calculate_fans(
        self,
        origin_point: Tuple[Decimal, int],
        high_point: Tuple[Decimal, int],
        time_point: int,
        ratios: Optional[List[float]] = None
    ) -> List[FibonacciLevel]:
        """
        Calculate Fibonacci fan lines.

        Args:
            origin_point: (price, time_index) for origin
            high_point: (price, time_index) for high point
            time_point: Time index to calculate fan for
            ratios: Custom ratios (default: standard fans)

        Returns:
            List of FibonacciLevel objects
        """
        if ratios is None:
            ratios = self.FIBONACCI_RATIOS['fans']

        origin_price, origin_time = origin_point
        high_price, high_time = high_point

        levels = []

        for ratio in ratios:
            # Calculate fan line price at given time
            time_ratio = (time_point - origin_time) / (high_time - origin_time)
            fan_price = origin_price + ((high_price - origin_price) * Decimal(str(ratio)) * Decimal(str(time_ratio)))

            fan_price = fan_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

            level = FibonacciLevel(
                ratio=float(ratio),
                price=float(fan_price),
                level_type="WEAK",
                strength=0.0,  # Default value, adjust as needed
                touches=0,    # Default value, adjust as needed
                description=f"{ratio:.3f} fan"
            )
            levels.append(level)

        return levels

    def calculate_projections(
        self,
        wave1_start: Decimal,
        wave1_end: Decimal,
        wave2_end: Decimal,
        ratios: Optional[List[float]] = None
    ) -> List[FibonacciLevel]:
        """
        Calculate Fibonacci projections (Wave 3 projections).

        Args:
            wave1_start: Start of wave 1
            wave1_end: End of wave 1
            wave2_end: End of wave 2
            ratios: Custom ratios (default: standard projections)

        Returns:
            List of FibonacciLevel objects
        """
        if ratios is None:
            ratios = self.FIBONACCI_RATIOS['projections']

        wave1_length = wave1_end - wave1_start
        levels = []

        for ratio in ratios:
            projection = wave2_end + (wave1_length * Decimal(str(ratio)))
            projection = projection.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

            level = FibonacciLevel(
                ratio=float(ratio),
                price=float(projection),
                level_type="VERY_STRONG",
                strength=0.0,  # Default value, adjust as needed
                touches=0,    # Default value, adjust as needed
                description=f"{ratio:.3f} projection"
            )
            levels.append(level)

        return levels

    def find_fibonacci_levels(
        self,
        candles: List[Any],
        lookback_periods: int = 100,
        min_swing_size: Decimal = Decimal('0.01')
    ) -> FibonacciResult:
        """
        Find significant Fibonacci levels from price action.

        Args:
            candles: List of candle data
            lookback_periods: Number of periods to analyze
            min_swing_size: Minimum swing size to consider

        Returns:
            FibonacciResult with all calculated levels
        """
        if len(candles) < lookback_periods:
            lookback_periods = len(candles)

        recent_candles = candles[-lookback_periods:]

        # Find swing highs and lows
        swing_highs, swing_lows = self._find_swings(recent_candles, min_swing_size)

        all_levels = []

        # Calculate retracements from recent swings
        for i in range(len(swing_highs) - 1):
            high1 = swing_highs[i]
            high2 = swing_highs[i + 1]

            if high2 > high1:  # Uptrend
                retracements = self.calculate_retracements(high2, high1)
                all_levels.extend(retracements)

        for i in range(len(swing_lows) - 1):
            low1 = swing_lows[i]
            low2 = swing_lows[i + 1]

            if low2 < low1:  # Downtrend
                retracements = self.calculate_retracements(low1, low2)
                all_levels.extend(retracements)

        # Calculate extensions
        if swing_highs and swing_lows:
            recent_high = max(swing_highs)
            recent_low = min(swing_lows)

            extensions = self.calculate_extensions(recent_high, recent_low)
            all_levels.extend(extensions)

        # Remove duplicates and sort by price
        unique_levels = self._remove_duplicate_levels(all_levels)

        # Separate levels into retracements and extensions
        retracement_levels = {}
        extension_levels = {}
        for level in unique_levels:
            if 'retracement' in level.description:
                retracement_levels[f"{level.ratio:.3f}"] = level.price
            elif 'extension' in level.description:
                extension_levels[f"{level.ratio:.3f}"] = level.price

        # Create confluence zones
        confluence_zones = self.analyze_fibonacci_confluence(unique_levels)

        # Determine signal based on current price
        current_price = recent_candles[-1].close if recent_candles else 0.0
        signal = self._determine_signal(unique_levels, current_price)

        confidence = self._calculate_confidence(unique_levels, recent_candles)
        description = f"Fibonacci analysis completed with {len(unique_levels)} levels found"

        return FibonacciResult(
            retracement_levels=retracement_levels,
            extension_levels=extension_levels,
            confluence_zones=[{"price": float(k), "levels": [l.description for l in v]} for k, v in confluence_zones.items()],
            signal=signal,
            confidence=confidence,
            description=description
        )

    def analyze_fibonacci_confluence(
        self,
        levels: List[FibonacciLevel],
        tolerance: Decimal = Decimal('0.001')
    ) -> Dict[Decimal, List[FibonacciLevel]]:
        """
        Analyze confluence between Fibonacci levels.

        Args:
            levels: List of Fibonacci levels
            tolerance: Price tolerance for confluence

        Returns:
            Dictionary mapping price levels to confluent Fibonacci levels
        """
        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x.price)

        confluence_zones = {}

        for level in sorted_levels:
            # Find levels within tolerance
            nearby_levels = [
                l for l in sorted_levels
                if abs(l.price - level.price) <= tolerance
            ]

            if len(nearby_levels) > 1:
                # Calculate average price for confluence zone
                avg_price = sum(l.price for l in nearby_levels) / len(nearby_levels)
                avg_price = Decimal(str(avg_price)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

                if avg_price not in confluence_zones:
                    confluence_zones[avg_price] = nearby_levels

        return confluence_zones

    def _find_swings(
        self,
        candles: List[Any],
        min_swing_size: Decimal
    ) -> Tuple[List[Decimal], List[Decimal]]:
        """
        Find swing highs and lows in candle data.

        Args:
            candles: List of candle data
            min_swing_size: Minimum swing size

        Returns:
            Tuple of (swing_highs, swing_lows)
        """
        swing_highs = []
        swing_lows = []

        for i in range(2, len(candles) - 2):
            current = candles[i]

            # Check for swing high
            if (current.high > candles[i-1].high and
                current.high > candles[i-2].high and
                current.high > candles[i+1].high and
                current.high > candles[i+2].high):

                swing_size = current.high - min(candles[i-1].low, candles[i-2].low,
                                               candles[i+1].low, candles[i+2].low)
                if swing_size >= min_swing_size:
                    swing_highs.append(current.high)

            # Check for swing low
            if (current.low < candles[i-1].low and
                current.low < candles[i-2].low and
                current.low < candles[i+1].low and
                current.low < candles[i+2].low):

                swing_size = max(candles[i-1].high, candles[i-2].high,
                                candles[i+1].high, candles[i+2].high) - current.low
                if swing_size >= min_swing_size:
                    swing_lows.append(current.low)

        return swing_highs, swing_lows

    def _remove_duplicate_levels(
        self,
        levels: List[FibonacciLevel],
        tolerance: Decimal = Decimal('0.01')
    ) -> List[FibonacciLevel]:
        """
        Remove duplicate levels within tolerance.

        Args:
            levels: List of Fibonacci levels
            tolerance: Price tolerance for duplicates

        Returns:
            List of unique Fibonacci levels
        """
        if not levels:
            return []

        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x.price)
        unique_levels = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            if abs(level.price - unique_levels[-1].price) > tolerance:
                unique_levels.append(level)

        return unique_levels

    def _calculate_confidence(
        self,
        levels: List[FibonacciLevel],
        candles: List[Any]
    ) -> float:
        """
        Calculate confidence score for Fibonacci analysis.

        Args:
            levels: Fibonacci levels found
            candles: Candle data used for analysis

        Returns:
            Confidence score between 0 and 1
        """
        if not levels or not candles:
            return 0.0

        # Base confidence on number of levels and price action
        level_count_factor = min(len(levels) / 20, 1.0)  # Max at 20 levels

        # Check if current price is near any level
        current_price = candles[-1].close
        near_level_factor = 0.0

        for level in levels:
            if abs(current_price - level.price) / current_price < 0.01:  # Within 1%
                near_level_factor = 1.0
                break

        # Combine factors
        confidence = (level_count_factor * 0.6) + (near_level_factor * 0.4)

        return min(confidence, 1.0)

    def _determine_signal(
        self,
        levels: List[FibonacciLevel],
        current_price: float
    ) -> SignalStrength:
        """
        Determine trading signal based on Fibonacci levels and current price.

        Args:
            levels: List of Fibonacci levels
            current_price: Current market price

        Returns:
            SignalStrength enum value
        """
        if not levels:
            return SignalStrength.NEUTRAL

        # Find nearest support and resistance levels
        support_levels = [l for l in levels if l.price <= current_price]
        resistance_levels = [l for l in levels if l.price >= current_price]

        if not support_levels or not resistance_levels:
            return SignalStrength.NEUTRAL

        nearest_support = max(support_levels, key=lambda x: x.price)
        nearest_resistance = min(resistance_levels, key=lambda x: x.price)

        support_distance = abs(current_price - nearest_support.price) / current_price
        resistance_distance = abs(current_price - nearest_resistance.price) / current_price

        # Determine signal based on proximity to levels
        if support_distance < 0.005 and resistance_distance > 0.01:  # Close to support
            return SignalStrength.BULLISH
        elif resistance_distance < 0.005 and support_distance > 0.01:  # Close to resistance
            return SignalStrength.BEARISH
        elif support_distance < 0.01 and resistance_distance < 0.01:  # Between levels
            return SignalStrength.NEUTRAL
        else:
            return SignalStrength.NEUTRAL
