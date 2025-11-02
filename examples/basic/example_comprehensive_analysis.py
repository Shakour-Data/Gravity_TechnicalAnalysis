"""
مثال کامل: تحلیل جامع با همه 4 بُعد

این فایل نشان می‌دهد چگونه:
1. اندیکاتورهای تکنیکال
2. الگوهای شمعی
3. امواج الیوت
4. الگوهای کلاسیک

را با هم ترکیب کنیم و تصمیم نهایی بگیریم.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List

from models.schemas import Candle, SignalStrength
from indicators.trend import TrendIndicators
from patterns.candlestick import CandlestickPatterns
from patterns.elliott_wave import ElliottWaveAnalyzer
from patterns.classical import ClassicalPatterns


def create_realistic_market_data(trend: str = "bullish", candles_count: int = 100) -> List[Candle]:
    """
    ساخت داده واقع‌گرایانه بازار
    
    Args:
        trend: 'bullish', 'bearish', or 'sideways'
        candles_count: تعداد کندل‌ها
    """
    candles = []
    base_time = datetime.now() - timedelta(hours=candles_count)
    base_price = 50000
    
    for i in range(candles_count):
        # شبیه‌سازی حرکت واقعی قیمت
        if trend == "bullish":
            # روند صعودی با نوسانات
            trend_component = i * 50
            noise = np.random.normal(0, 200)
        elif trend == "bearish":
            # روند نزولی با نوسانات
            trend_component = -i * 50
            noise = np.random.normal(0, 200)
        else:
            # رنج (sideways)
            trend_component = np.sin(i / 10) * 500
            noise = np.random.normal(0, 100)
        
        price = base_price + trend_component + noise
        
        # ساخت کندل با high/low/open/close واقعی
        daily_range = abs(np.random.normal(300, 100))
        open_price = price + np.random.uniform(-daily_range/2, daily_range/2)
        close_price = price + np.random.uniform(-daily_range/2, daily_range/2)
        high_price = max(open_price, close_price) + abs(np.random.normal(50, 20))
        low_price = min(open_price, close_price) - abs(np.random.normal(50, 20))
        
        volume = abs(np.random.normal(1000000, 200000))
        
        candle = Candle(
            timestamp=base_time + timedelta(hours=i),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        )
        candles.append(candle)
    
    return candles


def comprehensive_analysis(candles: List[Candle], symbol: str = "BTCUSDT"):
    """
    تحلیل جامع با همه 4 بُعد
    
    Args:
        candles: لیست کندل‌ها
        symbol: نماد ارز یا سهم
    """
    print("\n" + "═" * 80)
    print(f"🔍 تحلیل جامع برای {symbol}")
    print("═" * 80)
    
    current_price = candles[-1].close
    print(f"\n💰 قیمت فعلی: ${current_price:,.2f}")
    
    # ════════════════════════════════════════════════════════
    # 1️⃣ اندیکاتورهای تکنیکال
    # ════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("1️⃣  اندیکاتورهای تکنیکال")
    print("─" * 80)
    
    # SMA
    sma_result = TrendIndicators.sma(candles, period=20)
    print(f"   SMA(20): {sma_result.signal.value} (دقت: {sma_result.confidence:.2%})")
    
    # EMA
    ema_result = TrendIndicators.ema(candles, period=20)
    print(f"   EMA(20): {ema_result.signal.value} (دقت: {ema_result.confidence:.2%})")
    
    # MACD
    macd_result = TrendIndicators.macd(candles)
    print(f"   MACD: {macd_result.signal.value} (دقت: {macd_result.confidence:.2%})")
    
    # ADX
    adx_result = TrendIndicators.adx(candles)
    print(f"   ADX: {adx_result.signal.value} (دقت: {adx_result.confidence:.2%})")
    
    # محاسبه امتیاز کلی روند
    trend_signals = [sma_result, ema_result, macd_result, adx_result]
    trend_score = sum(s.value * s.confidence for s in trend_signals) / sum(s.confidence for s in trend_signals)
    trend_accuracy = sum(s.confidence for s in trend_signals) / len(trend_signals)
    
    print(f"\n   📊 امتیاز کلی روند: {trend_score:+.2f}")
    print(f"   📊 دقت روند: {trend_accuracy:.2%}")
    
    if trend_score > 0.6:
        trend_direction = "صعودی قوی 📈"
    elif trend_score > 0.3:
        trend_direction = "صعودی 📈"
    elif trend_score > -0.3:
        trend_direction = "خنثی ⚪"
    elif trend_score > -0.6:
        trend_direction = "نزولی 📉"
    else:
        trend_direction = "نزولی قوی 📉"
    
    print(f"   ➡️  نتیجه: {trend_direction}")
    
    # ════════════════════════════════════════════════════════
    # 2️⃣ الگوهای شمعی
    # ════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("2️⃣  الگوهای شمعی")
    print("─" * 80)
    
    candlestick_patterns = CandlestickPatterns.detect_patterns(candles)
    
    if candlestick_patterns:
        for pattern in candlestick_patterns[-3:]:  # آخرین 3 الگو
            signal_emoji = "🟢" if "صعودی" in pattern.signal.value else "🔴" if "نزولی" in pattern.signal.value else "🟡"
            print(f"   {signal_emoji} {pattern.pattern_name}")
            print(f"      سیگنال: {pattern.signal.value}")
            print(f"      دقت: {pattern.confidence:.2%}")
            print(f"      توضیح: {pattern.description}")
    else:
        print("   ⚪ الگوی خاصی شناسایی نشد")
    
    # ════════════════════════════════════════════════════════
    # 3️⃣ امواج الیوت
    # ════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("3️⃣  امواج الیوت")
    print("─" * 80)
    
    analyzer = ElliottWaveAnalyzer()
    elliott_result = analyzer.analyze(candles)
    
    if elliott_result:
        wave_emoji = "🌊"
        print(f"   {wave_emoji} الگو: {elliott_result.wave_pattern}")
        print(f"   {wave_emoji} موج فعلی: {elliott_result.current_wave}")
        print(f"   {wave_emoji} سیگنال: {elliott_result.signal.value}")
        print(f"   {wave_emoji} دقت: {elliott_result.confidence:.2%}")
        
        if elliott_result.projected_target:
            print(f"   🎯 هدف پیش‌بینی: ${elliott_result.projected_target:,.2f}")
        
        print(f"   📝 توضیح: {elliott_result.description}")
    else:
        print("   ⚪ امواج الیوت شناسایی نشد (نیاز به داده بیشتر)")
    
    # ════════════════════════════════════════════════════════
    # 4️⃣ الگوهای کلاسیک
    # ════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("4️⃣  الگوهای کلاسیک")
    print("─" * 80)
    
    classical_patterns = ClassicalPatterns.detect_all(candles)
    
    if classical_patterns:
        for pattern in classical_patterns:
            signal_emoji = "🟢" if "صعودی" in pattern.signal.value else "🔴" if "نزولی" in pattern.signal.value else "🟡"
            print(f"   {signal_emoji} {pattern.pattern_name}")
            print(f"      سیگنال: {pattern.signal.value}")
            print(f"      دقت: {pattern.confidence:.2%}")
            print(f"      هدف: ${pattern.price_target:,.2f}")
            print(f"      استاپ: ${pattern.stop_loss:,.2f}")
            print(f"      توضیح: {pattern.description}")
    else:
        print("   ⚪ الگوی کلاسیک شناسایی نشد")
    
    # ════════════════════════════════════════════════════════
    # 🎯 تصمیم نهایی
    # ════════════════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("🎯 تصمیم نهایی (ترکیب همه ابعاد)")
    print("═" * 80)
    
    # شمارش سیگنال‌ها
    bullish_signals = 0
    bearish_signals = 0
    total_confidence = 0
    
    # اندیکاتورها
    if trend_score > 0.3:
        bullish_signals += 1
        total_confidence += trend_accuracy
        print(f"   ✅ اندیکاتورها: صعودی (دقت {trend_accuracy:.0%})")
    elif trend_score < -0.3:
        bearish_signals += 1
        total_confidence += trend_accuracy
        print(f"   ✅ اندیکاتورها: نزولی (دقت {trend_accuracy:.0%})")
    else:
        print(f"   ⚪ اندیکاتورها: خنثی")
    
    # الگوهای شمعی
    if candlestick_patterns:
        latest_pattern = candlestick_patterns[-1]
        if latest_pattern.signal in [SignalStrength.VERY_BULLISH, SignalStrength.BULLISH]:
            bullish_signals += 1
            total_confidence += latest_pattern.confidence
            print(f"   ✅ الگوی شمعی: صعودی ({latest_pattern.pattern_name}, دقت {latest_pattern.confidence:.0%})")
        elif latest_pattern.signal in [SignalStrength.VERY_BEARISH, SignalStrength.BEARISH]:
            bearish_signals += 1
            total_confidence += latest_pattern.confidence
            print(f"   ✅ الگوی شمعی: نزولی ({latest_pattern.pattern_name}, دقت {latest_pattern.confidence:.0%})")
    
    # امواج الیوت
    if elliott_result:
        if elliott_result.signal in [SignalStrength.VERY_BULLISH, SignalStrength.BULLISH]:
            bullish_signals += 1
            total_confidence += elliott_result.confidence
            print(f"   ✅ امواج الیوت: صعودی (موج {elliott_result.current_wave}, دقت {elliott_result.confidence:.0%})")
        elif elliott_result.signal in [SignalStrength.VERY_BEARISH, SignalStrength.BEARISH]:
            bearish_signals += 1
            total_confidence += elliott_result.confidence
            print(f"   ✅ امواج الیوت: نزولی (موج {elliott_result.current_wave}, دقت {elliott_result.confidence:.0%})")
    
    # الگوهای کلاسیک
    if classical_patterns:
        for pattern in classical_patterns:
            if pattern.signal in [SignalStrength.VERY_BULLISH, SignalStrength.BULLISH]:
                bullish_signals += 1
                total_confidence += pattern.confidence
                print(f"   ✅ الگوی کلاسیک: صعودی ({pattern.pattern_name}, دقت {pattern.confidence:.0%})")
            elif pattern.signal in [SignalStrength.VERY_BEARISH, SignalStrength.BEARISH]:
                bearish_signals += 1
                total_confidence += pattern.confidence
                print(f"   ✅ الگوی کلاسیک: نزولی ({pattern.pattern_name}, دقت {pattern.confidence:.0%})")
    
    # محاسبه اعتماد کلی
    total_signals = bullish_signals + bearish_signals
    if total_signals > 0:
        overall_confidence = total_confidence / total_signals
    else:
        overall_confidence = 0
    
    print("\n" + "─" * 80)
    print(f"   📊 سیگنال‌های صعودی: {bullish_signals}")
    print(f"   📊 سیگنال‌های نزولی: {bearish_signals}")
    print(f"   📊 اعتماد کلی: {overall_confidence:.2%}")
    print("─" * 80)
    
    # تصمیم نهایی
    if bullish_signals > bearish_signals and bullish_signals >= 2:
        if overall_confidence >= 0.8:
            decision = "🟢 خرید قوی"
            action = "ورود با حجم کامل"
        else:
            decision = "🟢 خرید"
            action = "ورود با حجم متوسط"
    elif bearish_signals > bullish_signals and bearish_signals >= 2:
        if overall_confidence >= 0.8:
            decision = "🔴 فروش قوی"
            action = "خروج کامل یا شورت"
        else:
            decision = "🔴 فروش"
            action = "خروج جزئی"
    else:
        decision = "🟡 صبر و انتظار"
        action = "بدون اقدام - نیاز به سیگنال بیشتر"
    
    print(f"\n   🎯 تصمیم: {decision}")
    print(f"   🎯 اقدام: {action}")
    
    # اهداف قیمتی
    print("\n" + "─" * 80)
    if bullish_signals > bearish_signals:
        targets = []
        if elliott_result and elliott_result.projected_target:
            targets.append(("امواج الیوت", elliott_result.projected_target))
        for pattern in classical_patterns:
            if pattern.signal in [SignalStrength.VERY_BULLISH, SignalStrength.BULLISH]:
                targets.append((pattern.pattern_name, pattern.price_target))
        
        if targets:
            print("   🎯 اهداف قیمتی:")
            for name, target in targets:
                profit_percent = ((target - current_price) / current_price) * 100
                print(f"      • {name}: ${target:,.2f} (+{profit_percent:.1f}%)")
    
    elif bearish_signals > bullish_signals:
        targets = []
        if elliott_result and elliott_result.projected_target:
            targets.append(("امواج الیوت", elliott_result.projected_target))
        for pattern in classical_patterns:
            if pattern.signal in [SignalStrength.VERY_BEARISH, SignalStrength.BEARISH]:
                targets.append((pattern.pattern_name, pattern.price_target))
        
        if targets:
            print("   🎯 اهداف قیمتی:")
            for name, target in targets:
                loss_percent = ((target - current_price) / current_price) * 100
                print(f"      • {name}: ${target:,.2f} ({loss_percent:.1f}%)")
    
    print("═" * 80 + "\n")


if __name__ == "__main__":
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "مثال جامع: تحلیل 4 بُعدی" + " " * 34 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # سناریو 1: روند صعودی
    print("\n📈 سناریو 1: بازار صعودی")
    bullish_candles = create_realistic_market_data(trend="bullish", candles_count=100)
    comprehensive_analysis(bullish_candles, symbol="BTCUSDT")
    
    # سناریو 2: روند نزولی
    print("\n📉 سناریو 2: بازار نزولی")
    bearish_candles = create_realistic_market_data(trend="bearish", candles_count=100)
    comprehensive_analysis(bearish_candles, symbol="ETHUSD")
    
    # سناریو 3: رنج
    print("\n⚪ سناریو 3: بازار رنج")
    sideways_candles = create_realistic_market_data(trend="sideways", candles_count=100)
    comprehensive_analysis(sideways_candles, symbol="BNBUSDT")
    
    print("\n✅ همه سناریوها تکمیل شد!\n")
