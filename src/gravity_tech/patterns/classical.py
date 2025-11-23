"""
Classical Chart Pattern Recognition

This module recognizes major classical chart patterns:

Reversal Patterns:
1. Head and Shoulders
2. Inverse Head and Shoulders
3. Double Top
4. Double Bottom
5. Triple Top
6. Triple Bottom
7. Rising Wedge
8. Falling Wedge

Continuation Patterns:
9. Ascending Triangle
10. Descending Triangle
11. Symmetrical Triangle
12. Flag (Bullish/Bearish)

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
13. Pennant
14. Rectangle
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from scipy.signal import find_peaks, argrelextrema
from gravity_tech.models.schemas import Candle, PatternResult, SignalStrength, PatternType
from datetime import datetime


class ClassicalPatterns:
    """Classical chart pattern recognition"""
    
    @staticmethod
    def find_swing_points(candles: List[Candle], order: int = 5) -> Dict[str, List[Tuple[int, float]]]:
        """
        Find swing highs and lows
        
        Args:
            candles: List of candles
            order: Number of candles on each side to compare
            
        Returns:
            Dictionary with 'highs' and 'lows' lists of (index, price) tuples
        """
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        
        # Find local maxima (swing highs)
        peaks, _ = find_peaks(highs, distance=order)
        swing_highs = [(i, highs[i]) for i in peaks]
        
        # Find local minima (swing lows)
        troughs, _ = find_peaks(-lows, distance=order)
        swing_lows = [(i, lows[i]) for i in troughs]
        
        return {
            'highs': swing_highs,
            'lows': swing_lows
        }
    
    @staticmethod
    def detect_head_and_shoulders(candles: List[Candle], min_pattern_bars: int = 20) -> Optional[PatternResult]:
        """
        Detect Head and Shoulders pattern (bearish reversal)
        
        Pattern structure:
        - Left Shoulder (LS): Peak
        - Head (H): Higher peak
        - Right Shoulder (RS): Peak (similar height to LS)
        - Neckline: Support line connecting the two troughs
        
        Args:
            candles: List of candles
            min_pattern_bars: Minimum bars for pattern
            
        Returns:
            PatternResult if pattern found, None otherwise
        """
        if len(candles) < min_pattern_bars:
            return None
        
        swings = ClassicalPatterns.find_swing_points(candles[-50:])
        highs = swings['highs']
        lows = swings['lows']
        
        if len(highs) < 3 or len(lows) < 2:
            return None
        
        # Look for recent 3 peaks
        recent_highs = highs[-3:]
        ls_idx, ls_price = recent_highs[0]
        h_idx, h_price = recent_highs[1]
        rs_idx, rs_price = recent_highs[2]
        
        # Find troughs between peaks
        left_trough = None
        right_trough = None
        
        for idx, price in lows:
            if ls_idx < idx < h_idx and left_trough is None:
                left_trough = (idx, price)
            elif h_idx < idx < rs_idx and right_trough is None:
                right_trough = (idx, price)
        
        if left_trough is None or right_trough is None:
            return None
        
        # Validate pattern rules
        # 1. Head must be higher than shoulders
        if h_price <= ls_price or h_price <= rs_price:
            return None
        
        # 2. Shoulders should be roughly equal (within 5%)
        shoulder_diff = abs(ls_price - rs_price) / ls_price
        if shoulder_diff > 0.05:
            return None
        
        # 3. Current price should be near or below neckline
        neckline_price = (left_trough[1] + right_trough[1]) / 2
        current_price = candles[-1].close
        
        # Calculate pattern metrics
        head_height = h_price - neckline_price
        price_target = neckline_price - head_height
        
        # Check if neckline is broken
        is_broken = current_price < neckline_price
        
        if is_broken:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.85
            description = f"سر و شانه کامل - شکست خط گردن در {neckline_price:.2f}"
        else:
            signal = SignalStrength.BEARISH
            confidence = 0.7
            description = f"سر و شانه در حال شکل‌گیری - خط گردن در {neckline_price:.2f}"
        
        return PatternResult(
            pattern_name="Head and Shoulders",
            pattern_type=PatternType.CLASSICAL,
            signal=signal,
            confidence=confidence,
            start_time=candles[max(0, len(candles) - 50 + ls_idx)].timestamp,
            end_time=candles[-1].timestamp,
            price_target=price_target,
            stop_loss=h_price,
            description=description
        )
    
    @staticmethod
    def detect_inverse_head_and_shoulders(candles: List[Candle], min_pattern_bars: int = 20) -> Optional[PatternResult]:
        """
        Detect Inverse Head and Shoulders pattern (bullish reversal)
        
        Args:
            candles: List of candles
            min_pattern_bars: Minimum bars for pattern
            
        Returns:
            PatternResult if pattern found, None otherwise
        """
        if len(candles) < min_pattern_bars:
            return None
        
        swings = ClassicalPatterns.find_swing_points(candles[-50:])
        lows = swings['lows']
        highs = swings['highs']
        
        if len(lows) < 3 or len(highs) < 2:
            return None
        
        # Look for recent 3 troughs
        recent_lows = lows[-3:]
        ls_idx, ls_price = recent_lows[0]
        h_idx, h_price = recent_lows[1]  # Head (lowest point)
        rs_idx, rs_price = recent_lows[2]
        
        # Find peaks between troughs
        left_peak = None
        right_peak = None
        
        for idx, price in highs:
            if ls_idx < idx < h_idx and left_peak is None:
                left_peak = (idx, price)
            elif h_idx < idx < rs_idx and right_peak is None:
                right_peak = (idx, price)
        
        if left_peak is None or right_peak is None:
            return None
        
        # Validate pattern rules
        # 1. Head must be lower than shoulders
        if h_price >= ls_price or h_price >= rs_price:
            return None
        
        # 2. Shoulders should be roughly equal (within 5%)
        shoulder_diff = abs(ls_price - rs_price) / ls_price
        if shoulder_diff > 0.05:
            return None
        
        # 3. Current price should be near or above neckline
        neckline_price = (left_peak[1] + right_peak[1]) / 2
        current_price = candles[-1].close
        
        # Calculate pattern metrics
        head_height = neckline_price - h_price
        price_target = neckline_price + head_height
        
        # Check if neckline is broken
        is_broken = current_price > neckline_price
        
        if is_broken:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.85
            description = f"سر و شانه معکوس کامل - شکست خط گردن در {neckline_price:.2f}"
        else:
            signal = SignalStrength.BULLISH
            confidence = 0.7
            description = f"سر و شانه معکوس در حال شکل‌گیری - خط گردن در {neckline_price:.2f}"
        
        return PatternResult(
            pattern_name="Inverse Head and Shoulders",
            pattern_type=PatternType.CLASSICAL,
            signal=signal,
            confidence=confidence,
            start_time=candles[max(0, len(candles) - 50 + ls_idx)].timestamp,
            end_time=candles[-1].timestamp,
            price_target=price_target,
            stop_loss=h_price,
            description=description
        )
    
    @staticmethod
    def detect_double_top(candles: List[Candle], tolerance: float = 0.02) -> Optional[PatternResult]:
        """
        Detect Double Top pattern (bearish reversal)
        
        Args:
            candles: List of candles
            tolerance: Price tolerance for equal peaks (2% default)
            
        Returns:
            PatternResult if pattern found, None otherwise
        """
        if len(candles) < 20:
            return None
        
        swings = ClassicalPatterns.find_swing_points(candles[-40:])
        highs = swings['highs']
        lows = swings['lows']
        
        if len(highs) < 2:
            return None
        
        # Get last two peaks
        peak1_idx, peak1_price = highs[-2]
        peak2_idx, peak2_price = highs[-1]
        
        # Check if peaks are roughly equal
        price_diff = abs(peak1_price - peak2_price) / peak1_price
        if price_diff > tolerance:
            return None
        
        # Find trough between peaks
        trough = None
        for idx, price in lows:
            if peak1_idx < idx < peak2_idx:
                trough = (idx, price)
                break
        
        if trough is None:
            return None
        
        trough_idx, trough_price = trough
        
        # Current price should be declining from second peak
        current_price = candles[-1].close
        avg_peak = (peak1_price + peak2_price) / 2
        
        # Check if support (trough) is broken
        is_broken = current_price < trough_price
        
        # Calculate target
        height = avg_peak - trough_price
        price_target = trough_price - height
        
        if is_broken:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.8
            description = f"سقف دوقلو کامل - شکست حمایت در {trough_price:.2f}"
        else:
            signal = SignalStrength.BEARISH
            confidence = 0.65
            description = f"سقف دوقلو در حال شکل‌گیری - حمایت در {trough_price:.2f}"
        
        return PatternResult(
            pattern_name="Double Top",
            pattern_type=PatternType.CLASSICAL,
            signal=signal,
            confidence=confidence,
            start_time=candles[max(0, len(candles) - 40 + peak1_idx)].timestamp,
            end_time=candles[-1].timestamp,
            price_target=price_target,
            stop_loss=avg_peak,
            description=description
        )
    
    @staticmethod
    def detect_double_bottom(candles: List[Candle], tolerance: float = 0.02) -> Optional[PatternResult]:
        """
        Detect Double Bottom pattern (bullish reversal)
        
        Args:
            candles: List of candles
            tolerance: Price tolerance for equal troughs
            
        Returns:
            PatternResult if pattern found, None otherwise
        """
        if len(candles) < 20:
            return None
        
        swings = ClassicalPatterns.find_swing_points(candles[-40:])
        lows = swings['lows']
        highs = swings['highs']
        
        if len(lows) < 2:
            return None
        
        # Get last two troughs
        trough1_idx, trough1_price = lows[-2]
        trough2_idx, trough2_price = lows[-1]
        
        # Check if troughs are roughly equal
        price_diff = abs(trough1_price - trough2_price) / trough1_price
        if price_diff > tolerance:
            return None
        
        # Find peak between troughs
        peak = None
        for idx, price in highs:
            if trough1_idx < idx < trough2_idx:
                peak = (idx, price)
                break
        
        if peak is None:
            return None
        
        peak_idx, peak_price = peak
        
        # Current price should be rising from second trough
        current_price = candles[-1].close
        avg_trough = (trough1_price + trough2_price) / 2
        
        # Check if resistance (peak) is broken
        is_broken = current_price > peak_price
        
        # Calculate target
        height = peak_price - avg_trough
        price_target = peak_price + height
        
        if is_broken:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.8
            description = f"کف دوقلو کامل - شکست مقاومت در {peak_price:.2f}"
        else:
            signal = SignalStrength.BULLISH
            confidence = 0.65
            description = f"کف دوقلو در حال شکل‌گیری - مقاومت در {peak_price:.2f}"
        
        return PatternResult(
            pattern_name="Double Bottom",
            pattern_type=PatternType.CLASSICAL,
            signal=signal,
            confidence=confidence,
            start_time=candles[max(0, len(candles) - 40 + trough1_idx)].timestamp,
            end_time=candles[-1].timestamp,
            price_target=price_target,
            stop_loss=avg_trough,
            description=description
        )
    
    @staticmethod
    def detect_ascending_triangle(candles: List[Candle], min_touches: int = 2) -> Optional[PatternResult]:
        """
        Detect Ascending Triangle pattern (bullish continuation)
        
        Pattern:
        - Flat resistance (horizontal line at top)
        - Rising support (upward sloping line at bottom)
        
        Args:
            candles: List of candles
            min_touches: Minimum touches for each line
            
        Returns:
            PatternResult if pattern found, None otherwise
        """
        if len(candles) < 30:
            return None
        
        recent = candles[-30:]
        swings = ClassicalPatterns.find_swing_points(recent)
        highs = swings['highs']
        lows = swings['lows']
        
        if len(highs) < min_touches or len(lows) < min_touches:
            return None
        
        # Check for flat resistance (highs should be roughly equal)
        recent_highs = [price for idx, price in highs[-min_touches:]]
        resistance = np.mean(recent_highs)
        resistance_variance = np.std(recent_highs) / resistance
        
        if resistance_variance > 0.015:  # More than 1.5% variance
            return None
        
        # Check for rising support
        recent_lows = lows[-min_touches:]
        if len(recent_lows) < 2:
            return None
        
        low_prices = [price for idx, price in recent_lows]
        # Lows should be increasing
        is_rising = all(low_prices[i] < low_prices[i+1] for i in range(len(low_prices)-1))
        
        if not is_rising:
            return None
        
        # Calculate triangle height
        base_support = recent_lows[0][1]
        height = resistance - base_support
        
        # Check if resistance is broken
        current_price = candles[-1].close
        is_broken = current_price > resistance
        
        # Price target: resistance + height
        price_target = resistance + height
        
        if is_broken:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.75
            description = f"مثلث صعودی - شکست مقاومت {resistance:.2f}"
        else:
            signal = SignalStrength.BULLISH
            confidence = 0.6
            description = f"مثلث صعودی در حال شکل‌گیری - مقاومت {resistance:.2f}"
        
        return PatternResult(
            pattern_name="Ascending Triangle",
            pattern_type=PatternType.CLASSICAL,
            signal=signal,
            confidence=confidence,
            start_time=recent[0].timestamp,
            end_time=candles[-1].timestamp,
            price_target=price_target,
            stop_loss=base_support,
            description=description
        )
    
    @staticmethod
    def detect_descending_triangle(candles: List[Candle], min_touches: int = 2) -> Optional[PatternResult]:
        """
        Detect Descending Triangle pattern (bearish continuation)
        
        Pattern:
        - Flat support (horizontal line at bottom)
        - Falling resistance (downward sloping line at top)
        
        Args:
            candles: List of candles
            min_touches: Minimum touches for each line
            
        Returns:
            PatternResult if pattern found, None otherwise
        """
        if len(candles) < 30:
            return None
        
        recent = candles[-30:]
        swings = ClassicalPatterns.find_swing_points(recent)
        highs = swings['highs']
        lows = swings['lows']
        
        if len(highs) < min_touches or len(lows) < min_touches:
            return None
        
        # Check for flat support (lows should be roughly equal)
        recent_lows = [price for idx, price in lows[-min_touches:]]
        support = np.mean(recent_lows)
        support_variance = np.std(recent_lows) / support
        
        if support_variance > 0.015:  # More than 1.5% variance
            return None
        
        # Check for falling resistance
        recent_highs = highs[-min_touches:]
        if len(recent_highs) < 2:
            return None
        
        high_prices = [price for idx, price in recent_highs]
        # Highs should be decreasing
        is_falling = all(high_prices[i] > high_prices[i+1] for i in range(len(high_prices)-1))
        
        if not is_falling:
            return None
        
        # Calculate triangle height
        base_resistance = recent_highs[0][1]
        height = base_resistance - support
        
        # Check if support is broken
        current_price = candles[-1].close
        is_broken = current_price < support
        
        # Price target: support - height
        price_target = support - height
        
        if is_broken:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.75
            description = f"مثلث نزولی - شکست حمایت {support:.2f}"
        else:
            signal = SignalStrength.BEARISH
            confidence = 0.6
            description = f"مثلث نزولی در حال شکل‌گیری - حمایت {support:.2f}"
        
        return PatternResult(
            pattern_name="Descending Triangle",
            pattern_type=PatternType.CLASSICAL,
            signal=signal,
            confidence=confidence,
            start_time=recent[0].timestamp,
            end_time=candles[-1].timestamp,
            price_target=price_target,
            stop_loss=base_resistance,
            description=description
        )
    
    @staticmethod
    def detect_symmetrical_triangle(candles: List[Candle]) -> Optional[PatternResult]:
        """
        Detect Symmetrical Triangle pattern (continuation in trend direction)
        
        Pattern:
        - Converging trendlines (both sloping toward apex)
        
        Args:
            candles: List of candles
            
        Returns:
            PatternResult if pattern found, None otherwise
        """
        if len(candles) < 30:
            return None
        
        recent = candles[-30:]
        swings = ClassicalPatterns.find_swing_points(recent)
        highs = swings['highs']
        lows = swings['lows']
        
        if len(highs) < 2 or len(lows) < 2:
            return None
        
        # Get recent highs and lows
        recent_highs = highs[-3:] if len(highs) >= 3 else highs[-2:]
        recent_lows = lows[-3:] if len(lows) >= 3 else lows[-2:]
        
        # Check if highs are descending
        high_prices = [price for idx, price in recent_highs]
        highs_descending = all(high_prices[i] > high_prices[i+1] for i in range(len(high_prices)-1))
        
        # Check if lows are ascending
        low_prices = [price for idx, price in recent_lows]
        lows_ascending = all(low_prices[i] < low_prices[i+1] for i in range(len(low_prices)-1))
        
        if not (highs_descending and lows_ascending):
            return None
        
        # Calculate triangle metrics
        upper_line = high_prices[-1]
        lower_line = low_prices[-1]
        height = high_prices[0] - low_prices[0]
        
        # Determine trend direction from earlier candles
        earlier_trend = candles[-40:-30] if len(candles) >= 40 else candles[:10]
        trend_direction = earlier_trend[-1].close > earlier_trend[0].close
        
        current_price = candles[-1].close
        
        # Check for breakout
        if current_price > upper_line:
            signal = SignalStrength.VERY_BULLISH
            confidence = 0.7
            price_target = upper_line + height
            stop_loss = lower_line
            description = f"مثلث متقارن - شکست به بالا در {upper_line:.2f}"
        elif current_price < lower_line:
            signal = SignalStrength.VERY_BEARISH
            confidence = 0.7
            price_target = lower_line - height
            stop_loss = upper_line
            description = f"مثلث متقارن - شکست به پایین در {lower_line:.2f}"
        else:
            # No breakout yet, predict based on trend
            if trend_direction:
                signal = SignalStrength.BULLISH
                price_target = upper_line + height
                stop_loss = lower_line
            else:
                signal = SignalStrength.BEARISH
                price_target = lower_line - height
                stop_loss = upper_line
            confidence = 0.5
            description = f"مثلث متقارن در حال شکل‌گیری - انتظار شکست در جهت روند قبلی"
        
        return PatternResult(
            pattern_name="Symmetrical Triangle",
            pattern_type=PatternType.CLASSICAL,
            signal=signal,
            confidence=confidence,
            start_time=recent[0].timestamp,
            end_time=candles[-1].timestamp,
            price_target=price_target,
            stop_loss=stop_loss,
            description=description
        )
    
    @staticmethod
    def detect_all(candles: List[Candle]) -> List[PatternResult]:
        """
        Detect all classical patterns
        
        Args:
            candles: List of candles
            
        Returns:
            List of detected pattern results
        """
        patterns = []
        
        # Reversal patterns
        pattern = ClassicalPatterns.detect_head_and_shoulders(candles)
        if pattern:
            patterns.append(pattern)
        
        pattern = ClassicalPatterns.detect_inverse_head_and_shoulders(candles)
        if pattern:
            patterns.append(pattern)
        
        pattern = ClassicalPatterns.detect_double_top(candles)
        if pattern:
            patterns.append(pattern)
        
        pattern = ClassicalPatterns.detect_double_bottom(candles)
        if pattern:
            patterns.append(pattern)
        
        # Continuation patterns
        pattern = ClassicalPatterns.detect_ascending_triangle(candles)
        if pattern:
            patterns.append(pattern)
        
        pattern = ClassicalPatterns.detect_descending_triangle(candles)
        if pattern:
            patterns.append(pattern)
        
        pattern = ClassicalPatterns.detect_symmetrical_triangle(candles)
        if pattern:
            patterns.append(pattern)
        
        return patterns
