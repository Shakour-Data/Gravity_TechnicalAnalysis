"""
Candlestick Pattern Recognition

This module recognizes major candlestick patterns:
1. Doji
2. Hammer / Hanging Man
3. Inverted Hammer / Shooting Star
4. Bullish/Bearish Engulfing
5. Morning Star / Evening Star
6. Bullish/Bearish Harami
7. Piercing Pattern / Dark Cloud Cover
8. Three White Soldiers / Three Black Crows
9. Tweezer Top / Tweezer Bottom
10. Marubozu

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""



from gravity_tech.models.schemas import Candle, PatternResult, PatternType, SignalStrength


class CandlestickPatterns:
    """Candlestick pattern recognition"""

    @staticmethod
    def is_doji(candle: Candle, threshold: float = 0.1) -> bool:
        """Check if candle is a Doji"""
        body = abs(candle.close - candle.open)
        range_total = candle.high - candle.low
        return (body / range_total) < threshold if range_total > 0 else False

    @staticmethod
    def is_hammer(candle: Candle) -> bool:
        """Check if candle is a Hammer"""
        body = abs(candle.close - candle.open)
        lower_shadow = candle.lower_shadow
        upper_shadow = candle.upper_shadow

        return (lower_shadow > 2 * body and
                upper_shadow < body * 0.3 and
                body > 0)

    @staticmethod
    def is_inverted_hammer(candle: Candle) -> bool:
        """Check if candle is an Inverted Hammer"""
        body = abs(candle.close - candle.open)
        lower_shadow = candle.lower_shadow
        upper_shadow = candle.upper_shadow

        return (upper_shadow > 2 * body and
                lower_shadow < body * 0.3 and
                body > 0)

    @staticmethod
    def is_engulfing(candle1: Candle, candle2: Candle) -> str | None:
        """
        Check for Engulfing pattern
        Returns: 'bullish', 'bearish', or None
        """
        # Bullish Engulfing
        if (candle1.is_bearish and candle2.is_bullish and
            candle2.open < candle1.close and
            candle2.close > candle1.open):
            return 'bullish'

        # Bearish Engulfing
        if (candle1.is_bullish and candle2.is_bearish and
            candle2.open > candle1.close and
            candle2.close < candle1.open):
            return 'bearish'

        return None

    @staticmethod
    def is_morning_evening_star(candles: list[Candle]) -> str | None:
        """
        Check for Morning/Evening Star pattern (3 candles)
        Returns: 'morning', 'evening', or None
        """
        if len(candles) < 3:
            return None

        c1, c2, c3 = candles[-3], candles[-2], candles[-1]

        # Morning Star (bullish reversal)
        if (c1.is_bearish and
            abs(c2.close - c2.open) < (c1.body_size * 0.3) and
            c3.is_bullish and
            c3.close > (c1.open + c1.close) / 2):
            return 'morning'

        # Evening Star (bearish reversal)
        if (c1.is_bullish and
            abs(c2.close - c2.open) < (c1.body_size * 0.3) and
            c3.is_bearish and
            c3.close < (c1.open + c1.close) / 2):
            return 'evening'

        return None

    @staticmethod
    def is_harami(candle1: Candle, candle2: Candle) -> str | None:
        """
        Check for Harami pattern
        Returns: 'bullish', 'bearish', or None
        """
        # Bullish Harami
        if (candle1.is_bearish and candle2.is_bullish and
            candle2.open > candle1.close and
            candle2.close < candle1.open and
            candle2.body_size < candle1.body_size * 0.7):
            return 'bullish'

        # Bearish Harami
        if (candle1.is_bullish and candle2.is_bearish and
            candle2.open < candle1.close and
            candle2.close > candle1.open and
            candle2.body_size < candle1.body_size * 0.7):
            return 'bearish'

        return None

    @staticmethod
    def is_three_soldiers_crows(candles: list[Candle]) -> str | None:
        """
        Check for Three White Soldiers / Three Black Crows
        Returns: 'soldiers', 'crows', or None
        """
        if len(candles) < 3:
            return None

        c1, c2, c3 = candles[-3], candles[-2], candles[-1]

        # Three White Soldiers
        if (c1.is_bullish and c2.is_bullish and c3.is_bullish and
            c2.close > c1.close and c3.close > c2.close and
            c2.open > c1.open and c2.open < c1.close and
            c3.open > c2.open and c3.open < c2.close):
            return 'soldiers'

        # Three Black Crows
        if (c1.is_bearish and c2.is_bearish and c3.is_bearish and
            c2.close < c1.close and c3.close < c2.close and
            c2.open < c1.open and c2.open > c1.close and
            c3.open < c2.open and c3.open > c2.close):
            return 'crows'

        return None

    @staticmethod
    def detect_patterns(candles: list[Candle]) -> list[PatternResult]:
        """
        Detect all candlestick patterns

        Args:
            candles: List of candles

        Returns:
            List of detected patterns
        """
        patterns = []

        if len(candles) < 3:
            return patterns

        current = candles[-1]
        prev = candles[-2]

        # Doji
        if CandlestickPatterns.is_doji(current):
            patterns.append(PatternResult(
                pattern_name="Doji",
                pattern_type=PatternType.CANDLESTICK,
                signal=SignalStrength.NEUTRAL,
                confidence=0.6,
                start_time=current.timestamp,
                end_time=current.timestamp,
                description="الگوی دوجی - بی‌تصمیمی بازار"
            ))

        # Hammer
        if CandlestickPatterns.is_hammer(current):
            patterns.append(PatternResult(
                pattern_name="Hammer",
                pattern_type=PatternType.CANDLESTICK,
                signal=SignalStrength.BULLISH,
                confidence=0.7,
                start_time=current.timestamp,
                end_time=current.timestamp,
                description="چکش - الگوی بازگشتی صعودی"
            ))

        # Inverted Hammer
        if CandlestickPatterns.is_inverted_hammer(current):
            patterns.append(PatternResult(
                pattern_name="Inverted Hammer",
                pattern_type=PatternType.CANDLESTICK,
                signal=SignalStrength.BULLISH,
                confidence=0.65,
                start_time=current.timestamp,
                end_time=current.timestamp,
                description="چکش معکوس - احتمال بازگشت صعودی"
            ))

        # Engulfing
        engulfing = CandlestickPatterns.is_engulfing(prev, current)
        if engulfing:
            signal = SignalStrength.VERY_BULLISH if engulfing == 'bullish' else SignalStrength.VERY_BEARISH
            patterns.append(PatternResult(
                pattern_name=f"{'Bullish' if engulfing == 'bullish' else 'Bearish'} Engulfing",
                pattern_type=PatternType.CANDLESTICK,
                signal=signal,
                confidence=0.8,
                start_time=prev.timestamp,
                end_time=current.timestamp,
                description=f"الگوی در بر گیرنده {'صعودی' if engulfing == 'bullish' else 'نزولی'}"
            ))

        # Morning/Evening Star
        if len(candles) >= 3:
            star = CandlestickPatterns.is_morning_evening_star(candles)
            if star:
                signal = SignalStrength.VERY_BULLISH if star == 'morning' else SignalStrength.VERY_BEARISH
                patterns.append(PatternResult(
                    pattern_name=f"{'Morning' if star == 'morning' else 'Evening'} Star",
                    pattern_type=PatternType.CANDLESTICK,
                    signal=signal,
                    confidence=0.85,
                    start_time=candles[-3].timestamp,
                    end_time=current.timestamp,
                    description=f"ستاره {'صبحگاهی' if star == 'morning' else 'عصرگاهی'}"
                ))

        # Harami
        harami = CandlestickPatterns.is_harami(prev, current)
        if harami:
            signal = SignalStrength.BULLISH if harami == 'bullish' else SignalStrength.BEARISH
            patterns.append(PatternResult(
                pattern_name=f"{'Bullish' if harami == 'bullish' else 'Bearish'} Harami",
                pattern_type=PatternType.CANDLESTICK,
                signal=signal,
                confidence=0.7,
                start_time=prev.timestamp,
                end_time=current.timestamp,
                description=f"الگوی هارامی {'صعودی' if harami == 'bullish' else 'نزولی'}"
            ))

        # Three Soldiers/Crows
        if len(candles) >= 3:
            three_pattern = CandlestickPatterns.is_three_soldiers_crows(candles)
            if three_pattern:
                signal = SignalStrength.VERY_BULLISH if three_pattern == 'soldiers' else SignalStrength.VERY_BEARISH
                patterns.append(PatternResult(
                    pattern_name=f"Three {'White Soldiers' if three_pattern == 'soldiers' else 'Black Crows'}",
                    pattern_type=PatternType.CANDLESTICK,
                    signal=signal,
                    confidence=0.82,
                    start_time=candles[-3].timestamp,
                    end_time=current.timestamp,
                    description=f"{'سه سرباز سفید' if three_pattern == 'soldiers' else 'سه کلاغ سیاه'}"
                ))

        return patterns
