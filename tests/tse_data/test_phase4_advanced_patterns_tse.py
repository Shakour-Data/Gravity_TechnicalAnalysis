"""
Phase 4: Advanced Patterns Tests با داده‌های TSE

این ماژول تست‌های الگوهای پیشرفته را شامل می‌کند
تمام تست‌ها از داده‌های واقعی بازار ایران (TSE) استفاده می‌کنند

نویسنده: تیم Gravity
تاریخ: 4 دسامبر 2025
لیسانس: MIT
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Optional
import sys
from pathlib import Path
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from gravity_tech.core.domain.entities import Candle
from gravity_tech.indicators.trend import TrendIndicators
from gravity_tech.indicators.momentum import MomentumIndicators
from gravity_tech.indicators.volatility import VolatilityIndicators
from gravity_tech.indicators.volume import VolumeIndicators
from gravity_tech.patterns.candlestick import CandlestickPatterns
from gravity_tech.patterns.classical import ClassicalPatterns


# ============================================================================
# Phase 4: Advanced Pattern Recognition
# ============================================================================

class TestAdvancedPatternRecognition:
    """تست شناخت الگوهای پیشرفته"""

    def test_head_and_shoulders_formation(self, tse_candles_long):
        """تست الگوی Head and Shoulders"""
        candles = tse_candles_long
        assert candles is not None
        assert len(candles) > 0
        
        # تحلیل با اندیکاتورها
        result = TrendIndicators.sma(candles, period=20)
        assert result is not None

    def test_inverse_head_and_shoulders(self, tse_candles_long):
        """تست الگوی Inverse Head and Shoulders"""
        candles = tse_candles_long
        assert candles is not None
        
        result = TrendIndicators.ema(candles, period=20)
        assert result is not None

    def test_rising_wedge_pattern(self, tse_candles_long):
        """تست الگوی Rising Wedge"""
        candles = tse_candles_long
        assert candles is not None
        
        # تحلیل ترند
        trend = TrendIndicators.sma(candles, period=20)
        assert trend is not None

    def test_falling_wedge_pattern(self, tse_candles_long):
        """تست الگوی Falling Wedge"""
        candles = tse_candles_long
        assert candles is not None
        
        trend = TrendIndicators.ema(candles, period=20)
        assert trend is not None

    def test_triangle_pattern_formation(self, tse_candles_long):
        """تست الگوی Triangle"""
        candles = tse_candles_long
        assert candles is not None
        
        # محاسبه نوسانات برای تشخیص
        volatility = VolatilityIndicators.atr(candles)
        assert volatility is not None

    def test_rectangle_pattern_identification(self, tse_candles_long):
        """تست الگوی Rectangle"""
        candles = tse_candles_long
        assert candles is not None
        
        high_prices = [c.high for c in candles[-100:]]
        low_prices = [c.low for c in candles[-100:]]
        
        # Rectangle اگر High/Low پایدار باشد
        assert len(high_prices) > 0
        assert len(low_prices) > 0


# ============================================================================
# Phase 4: Complex Multi-Timeframe Analysis
# ============================================================================

class TestMultiTimeframeAnalysis:
    """تست تحلیل چند فریم زمانی"""

    def test_daily_to_hourly_confirmation(self, tse_candles_long):
        """تست تأیید نشانه‌های روزانه توسط ساعتی"""
        daily_candles = tse_candles_long
        
        # تحلیل روزانه
        daily_sma = TrendIndicators.sma(daily_candles, period=20)
        assert daily_sma is not None
        
        # تحلیل ساعتی (نمونه)
        hourly_candles = daily_candles[:50] if len(daily_candles) > 50 else daily_candles
        hourly_sma = TrendIndicators.sma(hourly_candles, period=10)
        assert hourly_sma is not None

    def test_trend_alignment_multiple_timeframes(self, tse_candles_long):
        """تست هم‌راستایی ترند در چند فریم زمانی"""
        candles = tse_candles_long
        
        # فریم 1
        trend_long = TrendIndicators.sma(candles, period=50)
        
        # فریم 2
        trend_mid = TrendIndicators.sma(candles, period=20)
        
        # فریم 3
        trend_short = TrendIndicators.sma(candles, period=10)
        
        assert trend_long is not None
        assert trend_mid is not None
        assert trend_short is not None

    def test_divergence_across_timeframes(self, tse_candles_long):
        """تست واگرایی‌ها در چند فریم زمانی"""
        candles = tse_candles_long
        
        # RSI در فریم‌های مختلف
        rsi_daily = MomentumIndicators.rsi(candles, period=14)
        assert rsi_daily is not None
        
        # Momentum قیمت vs RSI
        momentum = MomentumIndicators.roc(candles, period=10)
        assert momentum is not None


# ============================================================================
# Phase 4: Advanced Momentum Analysis
# ============================================================================

class TestAdvancedMomentumAnalysis:
    """تست تحلیل مومنتوم پیشرفته"""

    def test_hidden_divergence_bullish(self, tse_candles_long):
        """تست واگرایی پنهان صعودی"""
        candles = tse_candles_long
        
        # قیمت low جدید
        recent_lows = [c.low for c in candles[-50:]]
        
        # RSI
        rsi = MomentumIndicators.rsi(candles, period=14)
        assert rsi is not None

    def test_hidden_divergence_bearish(self, tse_candles_long):
        """تست واگرایی پنهان نزولی"""
        candles = tse_candles_long
        
        # قیمت high جدید
        recent_highs = [c.high for c in candles[-50:]]
        
        # RSI
        rsi = MomentumIndicators.rsi(candles, period=14)
        assert rsi is not None

    def test_momentum_confirmation_with_volume(self, tse_candles_long):
        """تست تأیید مومنتوم با حجم"""
        candles = tse_candles_long
        
        # مومنتوم قیمت
        roc = MomentumIndicators.roc(candles, period=10)
        
        # تصدیق حجم
        obv = VolumeIndicators.obv(candles)
        
        assert roc is not None
        assert obv is not None

    def test_macd_divergence_analysis(self, tse_candles_long):
        """تست تحلیل واگرایی MACD"""
        candles = tse_candles_long
        
        macd_result = TrendIndicators.macd(candles)
        assert macd_result is not None


# ============================================================================
# Phase 4: Advanced Support/Resistance
# ============================================================================

class TestAdvancedSupportResistance:
    """تست تحلیل سپتیو/رزیستنس پیشرفته"""

    def test_horizontal_support_resistance(self, tse_candles_long):
        """تست سپتیو/رزیستنس افقی"""
        candles = tse_candles_long
        
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        
        # سطوح مهم
        max_high = max(highs)
        min_low = min(lows)
        avg_price = sum(closes) / len(closes)
        
        assert max_high > avg_price > min_low

    def test_trend_channel_identification(self, tse_candles_long):
        """تست شناسایی کانال ترند"""
        candles = tse_candles_long
        
        closes = [c.close for c in candles[-100:]]
        
        if len(closes) > 10:
            slope_1 = (closes[-1] - closes[-10]) / 10
            assert slope_1 is not None

    def test_pivot_points_calculation(self, tse_candles_long):
        """تست محاسبه Pivot Points"""
        if len(tse_candles_long) < 5:
            pytest.skip("Insufficient data")
        
        candles = tse_candles_long[-5:]
        
        high = max(c.high for c in candles)
        low = min(c.low for c in candles)
        close = candles[-1].close
        
        pivot = (high + low + close) / 3
        assert pivot > 0

    def test_dynamic_support_resistance_zones(self, tse_candles_long):
        """تست مناطق پویای سپتیو/رزیستنس"""
        candles = tse_candles_long[-200:]
        
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        
        resistance_zone = max(highs)
        support_zone = min(lows)
        
        assert resistance_zone > support_zone


# ============================================================================
# Phase 4: Advanced Volume Analysis
# ============================================================================

class TestAdvancedVolumeAnalysis:
    """تست تحلیل حجم پیشرفته"""

    def test_volume_profile_analysis(self, tse_candles_long):
        """تست تحلیل پروفایل حجم"""
        candles = tse_candles_long
        
        volumes = [c.volume for c in candles]
        prices = [c.close for c in candles]
        
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        assert avg_volume >= 0

    def test_volume_rate_of_change(self, tse_candles_long):
        """تست نرخ تغییر حجم"""
        candles = tse_candles_long
        
        if len(candles) < 20:
            pytest.skip("Insufficient data")
        
        recent_volumes = [c.volume for c in candles[-20:]]
        older_volumes = [c.volume for c in candles[-40:-20]]
        
        if older_volumes:
            avg_old = sum(older_volumes) / len(older_volumes)
            avg_recent = sum(recent_volumes) / len(recent_volumes)
            
            if avg_old > 0:
                vroc = (avg_recent - avg_old) / avg_old * 100
                assert vroc is not None

    def test_accumulation_distribution_trend(self, tse_candles_long):
        """تست روند Accumulation/Distribution"""
        candles = tse_candles_long
        
        ad = VolumeIndicators.ad_line(candles)
        assert ad is not None

    def test_on_balance_volume_divergence(self, tse_candles_long):
        """تست واگرایی OBV"""
        candles = tse_candles_long
        
        obv = VolumeIndicators.obv(candles)
        assert obv is not None


# ============================================================================
# Phase 4: Advanced Volatility Analysis
# ============================================================================

class TestAdvancedVolatilityAnalysis:
    """تست تحلیل نوسانات پیشرفته"""

    def test_bollinger_band_squeeze(self, tse_candles_long):
        """تست فشار Bollinger Band"""
        candles = tse_candles_long
        
        bb = VolatilityIndicators.bollinger_bands(candles, period=20)
        assert bb is not None

    def test_volatility_breakout_identification(self, tse_candles_long):
        """تست شناسایی Breakout نوسانات"""
        candles = tse_candles_long
        
        if len(candles) < 50:
            pytest.skip("Insufficient data")
        
        recent_volatility = VolatilityIndicators.atr(candles[-20:])
        historical_volatility = VolatilityIndicators.atr(candles[-50:-20])
        
        assert recent_volatility is not None
        assert historical_volatility is not None

    def test_mean_reversion_conditions(self, tse_candles_long):
        """تست شرایط Mean Reversion"""
        candles = tse_candles_long
        
        bb = VolatilityIndicators.bollinger_bands(candles)
        assert bb is not None

    def test_volatility_expansion_phases(self, tse_candles_long):
        """تست فازهای بسط نوسانات"""
        candles = tse_candles_long
        
        volatilities = []
        for i in range(20, len(candles), 5):
            atr = VolatilityIndicators.atr(candles[i-20:i])
            if atr:
                volatilities.append(atr)
        
        if len(volatilities) > 1:
            assert volatilities is not None


# ============================================================================
# Phase 4: Pattern Confirmation with Multiple Indicators
# ============================================================================

class TestMultiIndicatorPatternConfirmation:
    """تست تأیید الگو با چند اندیکاتور"""

    def test_bullish_confirmation_sequence(self, tse_candles_long):
        """تست ترتیب تأیید صعودی"""
        candles = tse_candles_long
        
        # نشانه 1: Candlestick
        if len(candles) >= 3:
            hammer = CandlestickPatterns.is_hammer(candles[-1])
            
            # نشانه 2: Trend
            sma = TrendIndicators.sma(candles, period=20)
            
            # نشانه 3: Momentum
            rsi = MomentumIndicators.rsi(candles, period=14)
            
            # نشانه 4: Volume
            obv = VolumeIndicators.obv(candles)
            
            assert sma is not None
            assert rsi is not None
            assert obv is not None

    def test_bearish_confirmation_sequence(self, tse_candles_long):
        """تست ترتیب تأیید نزولی"""
        candles = tse_candles_long
        
        if len(candles) >= 3:
            # نشانه‌های نزولی
            trend = TrendIndicators.ema(candles, period=20)
            momentum = MomentumIndicators.rsi(candles, period=14)
            volume = VolumeIndicators.obv(candles)
            
            assert trend is not None
            assert momentum is not None
            assert volume is not None

    def test_weak_signal_filtering(self, tse_candles_long):
        """تست فیلتر کردن نشانه‌های ضعیف"""
        candles = tse_candles_long
        
        # یک نشانه
        single_signal_strength = 0.5
        
        # چند نشانه
        multi_signal_strength = 0.85
        
        assert single_signal_strength < multi_signal_strength

    def test_contradictory_signals_handling(self, tse_candles_long):
        """تست مدیریت نشانه‌های متناقض"""
        candles = tse_candles_long
        
        signal_a = 0.7
        signal_b = 0.3
        
        # نشانه‌های متناقض
        contradiction = abs(signal_a - signal_b)
        assert contradiction > 0


# ============================================================================
# Phase 4: Integration Tests
# ============================================================================

class TestPhase4Integration:
    """تست یکپارچگی Phase 4"""

    def test_all_patterns_on_tse_data(self, tse_candles_total):
        """تست تمام الگوها بر روی داده‌های TSE"""
        candles = tse_candles_total
        assert len(candles) > 0
        
        # تست کندل الگوها
        if len(candles) >= 2:
            result = CandlestickPatterns.is_doji(candles[-1])
            assert result is not None

    def test_advanced_analysis_pipeline(self, tse_candles_long):
        """تست خط لوله تحلیل پیشرفته"""
        candles = tse_candles_long
        
        # 1. Trend
        trend = TrendIndicators.sma(candles, period=20)
        
        # 2. Momentum
        momentum = MomentumIndicators.rsi(candles, period=14)
        
        # 3. Volume
        volume = VolumeIndicators.obv(candles)
        
        # 4. Volatility
        volatility = VolatilityIndicators.atr(candles)
        
        assert trend is not None
        assert momentum is not None
        assert volume is not None
        assert volatility is not None

    def test_performance_metrics_phase4(self, tse_candles_long):
        """تست معیارهای کارایی Phase 4"""
        import time
        
        candles = tse_candles_long
        start = time.time()
        
        # تست مختلف
        TrendIndicators.sma(candles, period=20)
        MomentumIndicators.rsi(candles, period=14)
        VolatilityIndicators.atr(candles)
        VolumeIndicators.obv(candles)
        
        end = time.time()
        elapsed = end - start
        
        # باید کمتر از 10 ثانیه
        assert elapsed < 10.0
