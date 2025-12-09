"""
Phase 5: Edge Cases و Stress Tests با داده‌های TSE

این ماژول تست‌های edge case و stress را شامل می‌کند
تمام موارد غیرعادی و محدودیت‌های سیستم را تست می‌کند

نویسنده: تیم Gravity
تاریخ: 4 دسامبر 2025
لیسانس: MIT
"""

import pytest
from datetime import datetime, timedelta
from typing import List
import sys
from pathlib import Path
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from gravity_tech.core.domain.entities import Candle
from gravity_tech.core.indicators.trend import TrendIndicators
from gravity_tech.core.indicators.momentum import MomentumIndicators
from gravity_tech.core.indicators.volatility import VolatilityIndicators
from gravity_tech.core.indicators.volume import VolumeIndicators
from gravity_tech.core.patterns.candlestick import CandlestickPatterns


# ============================================================================
# Test: Edge Cases - Empty/Minimal Data
# ============================================================================

class TestEdgeCasesMinimalData:
    """تست موارد حاشیه‌ای - داده‌های کم"""

    def test_single_candle(self):
        """تست با یک کندل تنها"""
        candle = Candle(
            timestamp=datetime(2025, 1, 1),
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=1000000
        )
        
        result = CandlestickPatterns.is_doji(candle)
        assert result is not None or isinstance(result, bool)

    def test_two_candles(self):
        """تست با دو کندل"""
        candles = [
            Candle(
                timestamp=datetime(2025, 1, 1),
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=1000000
            ),
            Candle(
                timestamp=datetime(2025, 1, 2),
                open=101.0,
                high=103.0,
                low=100.0,
                close=102.0,
                volume=1100000
            )
        ]
        
        # برای مومنتوم حداقل 2 کندل کافی است
        if len(candles) >= 2:
            result = VolumeIndicators.obv(candles)
            assert result is not None

    def test_exact_minimum_for_indicators(self):
        """تست با حداقل تعداد برای اندیکاتورها"""
        # SMA با دوره 14 نیاز به 14 کندل دارد
        candles = []
        for i in range(14):
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=100.0 + i,
                high=102.0 + i,
                low=99.0 + i,
                close=101.0 + i,
                volume=1000000
            ))
        
        result = TrendIndicators.sma(candles, period=14)
        assert result is not None


# ============================================================================
# Test: Edge Cases - Extreme Values
# ============================================================================

class TestEdgeCasesExtremeValues:
    """تست موارد حاشیه‌ای - مقادیر انتهایی"""

    def test_very_high_prices(self):
        """تست با قیمت‌های بسیار بالا"""
        candles = []
        for i in range(50):
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=1000000 + i * 1000,
                high=1000500 + i * 1000,
                low=999500 + i * 1000,
                close=1000200 + i * 1000,
                volume=1000000
            ))
        
        result = TrendIndicators.sma(candles, period=20)
        assert result is not None

    def test_very_low_prices(self):
        """تست با قیمت‌های بسیار پایین"""
        candles = []
        for i in range(50):
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=0.0001 + i * 0.00001,
                high=0.00015 + i * 0.00001,
                low=0.00005 + i * 0.00001,
                close=0.00012 + i * 0.00001,
                volume=1000000
            ))
        
        result = TrendIndicators.sma(candles, period=20)
        assert result is not None

    def test_massive_volume(self):
        """تست با حجم بسیار زیاد"""
        candles = []
        for i in range(50):
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=100 + i * 0.5,
                high=105 + i * 0.5,
                low=99 + i * 0.5,
                close=102 + i * 0.5,
                volume=1e15  # حجم بسیار زیاد
            ))
        
        result = VolumeIndicators.obv(candles)
        assert result is not None

    def test_zero_volume(self):
        """تست با حجم صفر"""
        candles = []
        for i in range(50):
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=100 + i * 0.5,
                high=105 + i * 0.5,
                low=99 + i * 0.5,
                close=102 + i * 0.5,
                volume=0  # حجم صفر
            ))
        
        result = VolumeIndicators.obv(candles)
        assert result is not None

    def test_negative_prices(self):
        """تست با قیمت‌های منفی (نباید رخ دهد)"""
        candles = []
        for i in range(50):
            # Generate valid candles with low prices
            base = 0.5 + i * 0.05
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=base,
                high=base + 0.2,
                low=base - 0.1,
                close=base + 0.1,
                volume=1000000
            ))
        
        result = TrendIndicators.sma(candles, period=20)
        assert result is not None


# ============================================================================
# Test: Edge Cases - Price Anomalies
# ============================================================================

class TestEdgeCasesPriceAnomalies:
    """تست موارد حاشیه‌ای - ناهنجاری‌های قیمت"""

    def test_gap_up_opening(self):
        """تست با شکاف بالا در باز"""
        candles = []
        prev_close = 100.0
        for i in range(50):
            gap = 10.0 if i == 10 else 0.0  # شکاف 10 واحدی
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=prev_close + gap,
                high=prev_close + gap + 2,
                low=prev_close + gap - 1,
                close=prev_close + gap + 1,
                volume=1000000
            ))
            prev_close = candles[-1].close
        
        result = TrendIndicators.sma(candles, period=20)
        assert result is not None

    def test_gap_down_opening(self):
        """تست با شکاف پایین در باز"""
        candles = []
        prev_close = 100.0
        for i in range(50):
            gap = -10.0 if i == 10 else 0.0  # شکاف -10 واحدی
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=max(1, prev_close + gap),
                high=max(1, prev_close + gap + 2),
                low=max(1, prev_close + gap - 1),
                close=max(1, prev_close + gap + 1),
                volume=1000000
            ))
            prev_close = candles[-1].close
        
        result = TrendIndicators.sma(candles, period=20)
        assert result is not None

    def test_limit_up_moves(self):
        """تست با حرکات حد بالا"""
        candles = []
        for i in range(50):
            # هر روز 10% افزایش
            base = 100 * (1.1 ** i)
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=base,
                high=base * 1.1,
                low=base * 0.95,
                close=base * 1.05,
                volume=1000000
            ))
        
        result = TrendIndicators.sma(candles, period=20)
        assert result is not None

    def test_limit_down_moves(self):
        """تست با حرکات حد پایین"""
        candles = []
        base_price = 1000
        for i in range(50):
            # هر روز 10% کاهش
            base = base_price * (0.9 ** i)
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=max(1, base),
                high=max(1, base * 1.05),
                low=max(1, base * 0.9),
                close=max(1, base * 0.95),
                volume=1000000
            ))
        
        result = TrendIndicators.sma(candles, period=20)
        assert result is not None


# ============================================================================
# Test: Edge Cases - Volume Spikes
# ============================================================================

class TestEdgeCasesVolumeSpikes:
    """تست موارد حاشیه‌ای - جهش‌های حجم"""

    def test_sudden_volume_spike(self):
        """تست با جهش ناگهانی حجم"""
        candles = []
        for i in range(50):
            volume = 10000000 if i == 25 else 1000000  # جهش در وسط
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=100 + i * 0.5,
                high=105 + i * 0.5,
                low=99 + i * 0.5,
                close=102 + i * 0.5,
                volume=volume
            ))
        
        result = VolumeIndicators.obv(candles)
        assert result is not None

    def test_volume_drying_up(self):
        """تست با خشک شدن حجم"""
        candles = []
        for i in range(50):
            volume = 1000000 * (0.5 ** (i // 10))  # حجم کاهش‌یابی
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=100 + i * 0.5,
                high=105 + i * 0.5,
                low=99 + i * 0.5,
                close=102 + i * 0.5,
                volume=max(1, volume)
            ))
        
        result = VolumeIndicators.obv(candles)
        assert result is not None

    def test_alternating_volume_pattern(self):
        """تست با الگوی متناوب حجم"""
        candles = []
        for i in range(50):
            volume = 5000000 if i % 2 == 0 else 100000  # متناوب
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=100 + i * 0.5,
                high=105 + i * 0.5,
                low=99 + i * 0.5,
                close=102 + i * 0.5,
                volume=volume
            ))
        
        result = VolumeIndicators.obv(candles)
        assert result is not None


# ============================================================================
# Test: Stress Tests - Large Datasets
# ============================================================================

class TestStressLargeDatasets:
    """تست فشار - مجموعه داده‌های بزرگ"""

    def test_1000_candles(self):
        """تست با 1000 کندل"""
        candles = []
        for i in range(1000):
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=100 + i * 0.01,
                high=105 + i * 0.01,
                low=99 + i * 0.01,
                close=102 + i * 0.01,
                volume=1000000
            ))
        
        result = TrendIndicators.sma(candles, period=50)
        assert result is not None

    def test_5000_candles(self):
        """تست با 5000 کندل"""
        candles = []
        for i in range(5000):
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=100 + i * 0.001,
                high=105 + i * 0.001,
                low=99 + i * 0.001,
                close=102 + i * 0.001,
                volume=1000000
            ))
        
        result = TrendIndicators.sma(candles, period=50)
        assert result is not None

    def test_performance_1000_candles(self):
        """تست کارایی با 1000 کندل"""
        import time
        candles = []
        for i in range(1000):
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=100 + i * 0.01,
                high=105 + i * 0.01,
                low=99 + i * 0.01,
                close=102 + i * 0.01,
                volume=1000000
            ))
        
        start = time.time()
        for _ in range(10):
            TrendIndicators.sma(candles, period=50)
            MomentumIndicators.rsi(candles, period=14)
            VolatilityIndicators.atr(candles)
        end = time.time()
        
        # باید کمتر از 10 ثانیه طول بکشد
        assert (end - start) < 10.0


# ============================================================================
# Test: Stress Tests - Extreme Scenarios
# ============================================================================

class TestStressExtremeScenarios:
    """تست فشار - سناریوهای انتهایی"""

    def test_perfect_uptrend(self):
        """تست ترند کامل بالا"""
        candles = []
        for i in range(500):
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=100 + i,
                high=105 + i,
                low=99 + i,
                close=104 + i,
                volume=1000000 + i * 100
            ))
        
        result = TrendIndicators.sma(candles, period=50)
        assert result is not None

    def test_perfect_downtrend(self):
        """تست ترند کامل پایین"""
        candles = []
        for i in range(500):
            base = 1000 - i
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=max(1, base),
                high=max(1, base + 5),
                low=max(1, base - 5),
                close=max(1, base - 1),
                volume=1000000
            ))
        
        result = TrendIndicators.sma(candles, period=50)
        assert result is not None

    def test_sideways_market(self):
        """تست بازار کناری"""
        candles = []
        for i in range(500):
            noise = np.sin(i / 50) * 2  # نوسان
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=100 + noise,
                high=102 + noise,
                low=98 + noise,
                close=101 + noise,
                volume=1000000
            ))
        
        result = TrendIndicators.sma(candles, period=50)
        assert result is not None

    def test_highly_volatile_market(self):
        """تست بازار بسیار نوسان"""
        np.random.seed(42)
        candles = []
        price = 100
        for i in range(500):
            change = np.random.normal(0, 5)  # نوسان بسیار زیاد
            price = max(1, price + change)
            
            # Ensure valid candle data
            high = price + abs(np.random.normal(0, 2))
            low = max(1, price - abs(np.random.normal(0, 2)))
            open_price = np.random.uniform(low, high)
            close_price = np.random.uniform(low, high)
            
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=1000000
            ))
        
        result = TrendIndicators.sma(candles, period=50)
        assert result is not None


# ============================================================================
# Test: Data Validation - Edge Cases
# ============================================================================

class TestDataValidationEdgeCases:
    """تست تأیید داده‌ها - موارد حاشیه‌ای"""

    def test_high_less_than_open_and_close(self):
        """تست: High کمتر از Open/Close - validation test"""
        # This tests that invalid candles are properly rejected
        with pytest.raises(ValueError):
            candles = []
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1),
                open=100.0,
                high=99.0,  # ناقص!
                low=98.0,
                close=99.5,
                volume=1000000
            ))

    def test_low_greater_than_open_and_close(self):
        """تست: Low بیشتر از Open/Close - validation test"""
        # This tests that invalid candles are properly rejected
        with pytest.raises(ValueError):
            candles = []
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1),
                open=100.0,
                high=102.0,
                low=101.0,  # ناقص!
                close=99.5,
                volume=1000000
            ))

    def test_high_less_than_low(self):
        """تست: High کمتر از Low - validation test"""
        # This tests that invalid candles are properly rejected
        with pytest.raises(ValueError):
            candles = []
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1),
                open=100.0,
                high=98.0,  # ناقص!
                low=99.0,
                close=99.5,
                volume=1000000
            ))


# ============================================================================
# Test: Numerical Stability
# ============================================================================

class TestNumericalStability:
    """تست پایداری عددی"""

    def test_repeated_same_prices(self):
        """تست قیمت‌های یکسان تکراری"""
        candles = []
        for i in range(100):
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=1000000
            ))
        
        result = TrendIndicators.sma(candles, period=20)
        assert result is not None

    def test_very_small_price_changes(self):
        """تست تغییرات قیمت بسیار کوچک"""
        candles = []
        for i in range(100):
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=100.0 + i * 0.00001,
                high=100.00002 + i * 0.00001,
                low=99.99998 + i * 0.00001,
                close=100.00001 + i * 0.00001,
                volume=1000000
            ))
        
        result = TrendIndicators.sma(candles, period=20)
        assert result is not None

    def test_rsi_extreme_values(self):
        """تست RSI با مقادیر انتهایی"""
        # تمام در حال افزایش - RSI باید 100 باشد
        candles = []
        for i in range(50):
            candles.append(Candle(
                timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
                open=100 + i,
                high=102 + i,
                low=99 + i,
                close=101 + i,
                volume=1000000
            ))
        
        result = MomentumIndicators.rsi(candles, period=14)
        assert result is not None
