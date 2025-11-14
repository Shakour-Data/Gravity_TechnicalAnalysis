"""
Divergence Detection System

تشخیص واگرایی در اندیکاتورهای مومنتوم:
- Regular Divergence (واگرایی معمولی)
- Hidden Divergence (واگرایی پنهان)

واگرایی = قدرتمندترین سیگنال مومنتوم

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from gravity_tech.models.schemas import Candle


class DivergenceType(Enum):
    """نوع واگرایی"""
    REGULAR_BULLISH = "REGULAR_BULLISH"          # واگرایی معمولی صعودی
    REGULAR_BEARISH = "REGULAR_BEARISH"          # واگرایی معمولی نزولی
    HIDDEN_BULLISH = "HIDDEN_BULLISH"            # واگرایی پنهان صعودی
    HIDDEN_BEARISH = "HIDDEN_BEARISH"            # واگرایی پنهان نزولی
    NONE = "NONE"                                 # بدون واگرایی


@dataclass
class SwingPoint:
    """نقطه سوئینگ (قله یا دره)"""
    index: int
    value: float
    is_high: bool  # True = High/Peak, False = Low/Trough


@dataclass
class DivergenceResult:
    """
    نتیجه تشخیص واگرایی
    """
    divergence_type: DivergenceType
    strength: float  # [0, 1] قدرت واگرایی
    confidence: float  # [0, 1] اعتماد
    description: str
    
    # جزئیات
    price_swing1: Optional[SwingPoint] = None
    price_swing2: Optional[SwingPoint] = None
    indicator_swing1: Optional[SwingPoint] = None
    indicator_swing2: Optional[SwingPoint] = None
    
    def get_signal_score(self) -> float:
        """
        تبدیل واگرایی به امتیاز سیگنال [-2, 2]
        """
        if self.divergence_type == DivergenceType.REGULAR_BULLISH:
            return 2.0 * self.strength  # [0, 2]
        elif self.divergence_type == DivergenceType.REGULAR_BEARISH:
            return -2.0 * self.strength  # [-2, 0]
        elif self.divergence_type == DivergenceType.HIDDEN_BULLISH:
            return 1.5 * self.strength  # [0, 1.5]
        elif self.divergence_type == DivergenceType.HIDDEN_BEARISH:
            return -1.5 * self.strength  # [-1.5, 0]
        else:
            return 0.0


class DivergenceDetector:
    """
    تشخیص‌دهنده واگرایی
    """
    
    def __init__(
        self,
        lookback: int = 20,
        min_swing_distance: int = 5,
        swing_threshold: float = 0.02  # 2%
    ):
        """
        Initialize divergence detector
        
        Args:
            lookback: تعداد کندل‌های بررسی شده
            min_swing_distance: حداقل فاصله بین دو سوئینگ
            swing_threshold: حداقل اختلاف برای تشخیص سوئینگ (درصد)
        """
        self.lookback = lookback
        self.min_swing_distance = min_swing_distance
        self.swing_threshold = swing_threshold
    
    def detect(
        self,
        candles: List[Candle],
        indicator_values: List[float],
        indicator_name: str = "Indicator"
    ) -> DivergenceResult:
        """
        تشخیص واگرایی بین قیمت و اندیکاتور
        
        Args:
            candles: لیست کندل‌ها
            indicator_values: مقادیر اندیکاتور (هم‌اندازه با candles)
            indicator_name: نام اندیکاتور برای توضیحات
            
        Returns:
            DivergenceResult
        """
        if len(candles) < self.lookback or len(indicator_values) < self.lookback:
            return DivergenceResult(
                divergence_type=DivergenceType.NONE,
                strength=0.0,
                confidence=0.5,
                description="داده کافی برای تشخیص واگرایی نیست"
            )
        
        # استخراج بخش مورد نظر
        recent_candles = candles[-self.lookback:]
        recent_indicators = indicator_values[-self.lookback:]
        
        # پیدا کردن swing points در قیمت
        price_highs = [c.high for c in recent_candles]
        price_lows = [c.low for c in recent_candles]
        
        price_swing_highs = self._find_swing_points(price_highs, is_high=True)
        price_swing_lows = self._find_swing_points(price_lows, is_high=False)
        
        # پیدا کردن swing points در اندیکاتور
        indicator_swing_highs = self._find_swing_points(recent_indicators, is_high=True)
        indicator_swing_lows = self._find_swing_points(recent_indicators, is_high=False)
        
        # بررسی واگرایی معمولی صعودی (Regular Bullish)
        regular_bullish = self._check_regular_bullish_divergence(
            price_swing_lows, indicator_swing_lows
        )
        
        # بررسی واگرایی معمولی نزولی (Regular Bearish)
        regular_bearish = self._check_regular_bearish_divergence(
            price_swing_highs, indicator_swing_highs
        )
        
        # بررسی واگرایی پنهان صعودی (Hidden Bullish)
        hidden_bullish = self._check_hidden_bullish_divergence(
            price_swing_lows, indicator_swing_lows
        )
        
        # بررسی واگرایی پنهان نزولی (Hidden Bearish)
        hidden_bearish = self._check_hidden_bearish_divergence(
            price_swing_highs, indicator_swing_highs
        )
        
        # انتخاب قوی‌ترین واگرایی
        divergences = [
            (DivergenceType.REGULAR_BULLISH, regular_bullish),
            (DivergenceType.REGULAR_BEARISH, regular_bearish),
            (DivergenceType.HIDDEN_BULLISH, hidden_bullish),
            (DivergenceType.HIDDEN_BEARISH, hidden_bearish)
        ]
        
        # فیلتر واگرایی‌های یافت شده
        found_divergences = [(dtype, data) for dtype, data in divergences if data is not None]
        
        if not found_divergences:
            return DivergenceResult(
                divergence_type=DivergenceType.NONE,
                strength=0.0,
                confidence=0.5,
                description="واگرایی تشخیص داده نشد"
            )
        
        # انتخاب قوی‌ترین (Regular > Hidden)
        for dtype, data in found_divergences:
            if dtype in [DivergenceType.REGULAR_BULLISH, DivergenceType.REGULAR_BEARISH]:
                return self._create_result(dtype, data, indicator_name)
        
        # اگر فقط Hidden وجود داشت
        dtype, data = found_divergences[0]
        return self._create_result(dtype, data, indicator_name)
    
    def _find_swing_points(
        self,
        values: List[float],
        is_high: bool,
        window: int = 3
    ) -> List[SwingPoint]:
        """
        پیدا کردن swing points (قله‌ها یا دره‌ها)
        
        Args:
            values: لیست مقادیر
            is_high: True برای High (قله)، False برای Low (دره)
            window: تعداد نقاط قبل و بعد برای مقایسه
        """
        swings = []
        
        for i in range(window, len(values) - window):
            current = values[i]
            
            if is_high:
                # بررسی قله: باید از همه نقاط قبل و بعد بیشتر باشد
                is_swing = all(current >= values[j] for j in range(i - window, i + window + 1) if j != i)
            else:
                # بررسی دره: باید از همه نقاط قبل و بعد کمتر باشد
                is_swing = all(current <= values[j] for j in range(i - window, i + window + 1) if j != i)
            
            if is_swing:
                swings.append(SwingPoint(
                    index=i,
                    value=current,
                    is_high=is_high
                ))
        
        return swings
    
    def _check_regular_bullish_divergence(
        self,
        price_lows: List[SwingPoint],
        indicator_lows: List[SwingPoint]
    ) -> Optional[Tuple[SwingPoint, SwingPoint, SwingPoint, SwingPoint]]:
        """
        واگرایی معمولی صعودی:
        - قیمت: Lower Low (LL)
        - اندیکاتور: Higher Low (HL)
        → احتمال برگشت صعودی
        """
        if len(price_lows) < 2 or len(indicator_lows) < 2:
            return None
        
        # آخرین دو Low
        p_low2 = price_lows[-1]
        p_low1 = price_lows[-2]
        
        # پیدا کردن نزدیک‌ترین indicator lows
        i_low2 = self._find_nearest_swing(indicator_lows, p_low2.index)
        i_low1 = self._find_nearest_swing(indicator_lows, p_low1.index)
        
        if i_low2 is None or i_low1 is None:
            return None
        
        # شرط واگرایی: قیمت پایین‌تر اما اندیکاتور بالاتر
        price_makes_lower_low = p_low2.value < p_low1.value
        indicator_makes_higher_low = i_low2.value > i_low1.value
        
        if price_makes_lower_low and indicator_makes_higher_low:
            return (p_low1, p_low2, i_low1, i_low2)
        
        return None
    
    def _check_regular_bearish_divergence(
        self,
        price_highs: List[SwingPoint],
        indicator_highs: List[SwingPoint]
    ) -> Optional[Tuple[SwingPoint, SwingPoint, SwingPoint, SwingPoint]]:
        """
        واگرایی معمولی نزولی:
        - قیمت: Higher High (HH)
        - اندیکاتور: Lower High (LH)
        → احتمال برگشت نزولی
        """
        if len(price_highs) < 2 or len(indicator_highs) < 2:
            return None
        
        p_high2 = price_highs[-1]
        p_high1 = price_highs[-2]
        
        i_high2 = self._find_nearest_swing(indicator_highs, p_high2.index)
        i_high1 = self._find_nearest_swing(indicator_highs, p_high1.index)
        
        if i_high2 is None or i_high1 is None:
            return None
        
        # شرط: قیمت بالاتر اما اندیکاتور پایین‌تر
        price_makes_higher_high = p_high2.value > p_high1.value
        indicator_makes_lower_high = i_high2.value < i_high1.value
        
        if price_makes_higher_high and indicator_makes_lower_high:
            return (p_high1, p_high2, i_high1, i_high2)
        
        return None
    
    def _check_hidden_bullish_divergence(
        self,
        price_lows: List[SwingPoint],
        indicator_lows: List[SwingPoint]
    ) -> Optional[Tuple[SwingPoint, SwingPoint, SwingPoint, SwingPoint]]:
        """
        واگرایی پنهان صعودی:
        - قیمت: Higher Low (HL)
        - اندیکاتور: Lower Low (LL)
        → ادامه روند صعودی
        """
        if len(price_lows) < 2 or len(indicator_lows) < 2:
            return None
        
        p_low2 = price_lows[-1]
        p_low1 = price_lows[-2]
        
        i_low2 = self._find_nearest_swing(indicator_lows, p_low2.index)
        i_low1 = self._find_nearest_swing(indicator_lows, p_low1.index)
        
        if i_low2 is None or i_low1 is None:
            return None
        
        # شرط: قیمت بالاتر اما اندیکاتور پایین‌تر
        price_makes_higher_low = p_low2.value > p_low1.value
        indicator_makes_lower_low = i_low2.value < i_low1.value
        
        if price_makes_higher_low and indicator_makes_lower_low:
            return (p_low1, p_low2, i_low1, i_low2)
        
        return None
    
    def _check_hidden_bearish_divergence(
        self,
        price_highs: List[SwingPoint],
        indicator_highs: List[SwingPoint]
    ) -> Optional[Tuple[SwingPoint, SwingPoint, SwingPoint, SwingPoint]]:
        """
        واگرایی پنهان نزولی:
        - قیمت: Lower High (LH)
        - اندیکاتور: Higher High (HH)
        → ادامه روند نزولی
        """
        if len(price_highs) < 2 or len(indicator_highs) < 2:
            return None
        
        p_high2 = price_highs[-1]
        p_high1 = price_highs[-2]
        
        i_high2 = self._find_nearest_swing(indicator_highs, p_high2.index)
        i_high1 = self._find_nearest_swing(indicator_highs, p_high1.index)
        
        if i_high2 is None or i_high1 is None:
            return None
        
        # شرط: قیمت پایین‌تر اما اندیکاتور بالاتر
        price_makes_lower_high = p_high2.value < p_high1.value
        indicator_makes_higher_high = i_high2.value > i_high1.value
        
        if price_makes_lower_high and indicator_makes_higher_high:
            return (p_high1, p_high2, i_high1, i_high2)
        
        return None
    
    def _find_nearest_swing(
        self,
        swings: List[SwingPoint],
        target_index: int,
        max_distance: int = 5
    ) -> Optional[SwingPoint]:
        """پیدا کردن نزدیک‌ترین swing به index مورد نظر"""
        nearest = None
        min_dist = float('inf')
        
        for swing in swings:
            dist = abs(swing.index - target_index)
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                nearest = swing
        
        return nearest
    
    def _create_result(
        self,
        divergence_type: DivergenceType,
        data: Tuple[SwingPoint, SwingPoint, SwingPoint, SwingPoint],
        indicator_name: str
    ) -> DivergenceResult:
        """ایجاد نتیجه واگرایی"""
        p1, p2, i1, i2 = data
        
        # محاسبه قدرت واگرایی
        price_change = abs(p2.value - p1.value) / p1.value
        indicator_change = abs(i2.value - i1.value) / abs(i1.value) if i1.value != 0 else 0
        
        # هرچه تفاوت بیشتر، قدرت بیشتر
        strength = min((price_change + indicator_change) / 0.2, 1.0)  # نرمال به [0, 1]
        
        # Confidence بر اساس فاصله زمانی
        time_distance = abs(p2.index - p1.index)
        confidence = 0.7 + min(time_distance / 50, 0.2)  # 0.7-0.9
        
        # توضیحات
        descriptions = {
            DivergenceType.REGULAR_BULLISH: f"واگرایی صعودی: قیمت Lower Low اما {indicator_name} Higher Low - احتمال برگشت صعودی",
            DivergenceType.REGULAR_BEARISH: f"واگرایی نزولی: قیمت Higher High اما {indicator_name} Lower High - احتمال برگشت نزولی",
            DivergenceType.HIDDEN_BULLISH: f"واگرایی پنهان صعودی: قیمت Higher Low اما {indicator_name} Lower Low - ادامه روند صعودی",
            DivergenceType.HIDDEN_BEARISH: f"واگرایی پنهان نزولی: قیمت Lower High اما {indicator_name} Higher High - ادامه روند نزولی"
        }
        
        return DivergenceResult(
            divergence_type=divergence_type,
            strength=strength,
            confidence=confidence,
            description=descriptions.get(divergence_type, "واگرایی تشخیص داده شد"),
            price_swing1=p1,
            price_swing2=p2,
            indicator_swing1=i1,
            indicator_swing2=i2
        )


# ═══════════════════════════════════════════════════════════
# تست
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from datetime import datetime, timedelta
    import random
    
    print("\n" + "="*70)
    print("🧪 Testing Divergence Detection")
    print("="*70)
    
    # ساخت داده تست با واگرایی معمولی صعودی
    base_time = datetime.now() - timedelta(days=30)
    candles = []
    indicator_values = []
    
    for i in range(30):
        # قیمت: Lower Low در انتها
        if i < 10:
            price = 50000 - i * 200
        elif i < 20:
            price = 48000 + (i - 10) * 150
        else:
            price = 49500 - (i - 20) * 250  # Lower Low
        
        # اندیکاتور: Higher Low در انتها
        if i < 10:
            indicator = 40 - i * 2
        elif i < 20:
            indicator = 20 + (i - 10) * 1.5
        else:
            indicator = 35 - (i - 20) * 1  # Higher Low (بالاتر از 20)
        
        candle = Candle(
            timestamp=(base_time + timedelta(days=i)).isoformat(),
            open=price,
            high=price + 200,
            low=price - 200,
            close=price + random.uniform(-100, 100),
            volume=1000000
        )
        
        candles.append(candle)
        indicator_values.append(indicator)
    
    # تشخیص واگرایی
    detector = DivergenceDetector(lookback=25)
    result = detector.detect(candles, indicator_values, "RSI")
    
    print(f"\n🔍 Divergence Detected:")
    print(f"   Type: {result.divergence_type.value}")
    print(f"   Strength: {result.strength:.2f}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Signal Score: {result.get_signal_score():.2f}")
    print(f"   📋 {result.description}")
    
    if result.price_swing1:
        print(f"\n   Price Swings:")
        print(f"     Swing 1: ${result.price_swing1.value:.2f} at index {result.price_swing1.index}")
        print(f"     Swing 2: ${result.price_swing2.value:.2f} at index {result.price_swing2.index}")
        print(f"   Indicator Swings:")
        print(f"     Swing 1: {result.indicator_swing1.value:.2f} at index {result.indicator_swing1.index}")
        print(f"     Swing 2: {result.indicator_swing2.value:.2f} at index {result.indicator_swing2.index}")
    
    print("\n" + "="*70)
    print("✅ Divergence detection tested!")
    print("="*70)
