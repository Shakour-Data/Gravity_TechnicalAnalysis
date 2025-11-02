"""
مثال استفاده از 5-Dimensional Decision Matrix
==============================================

این اسکریپت نحوه استفاده از سیستم تصمیم‌گیری 5 بُعدی را نشان می‌دهد.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from datetime import datetime, timedelta
from ml.five_dimensional_decision_matrix import (
    FiveDimensionalDecisionMatrix,
    DecisionSignal,
    RiskLevel
)
from models.schemas import (
    Candle,
    TrendScore,
    MomentumScore,
    VolatilityScore,
    CycleScore,
    SupportResistanceScore,
    SignalStrength
)


def create_sample_candles(count: int = 100) -> list:
    """ایجاد کندل‌های نمونه برای تست"""
    candles = []
    base_price = 50000
    base_time = datetime.now() - timedelta(hours=count)
    
    for i in range(count):
        # شبیه‌سازی روند صعودی با نوسانات
        price_change = (i * 50) + (i % 10) * 100
        open_price = base_price + price_change
        close_price = open_price + (100 if i % 3 == 0 else -50)
        high_price = max(open_price, close_price) + 50
        low_price = min(open_price, close_price) - 50
        volume = 1000 + (i * 10)
        
        candles.append(Candle(
            timestamp=base_time + timedelta(hours=i),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        ))
    
    return candles


def create_sample_scores():
    """ایجاد نمونه scores برای هر dimension"""
    
    # سناریو 1: همه dimensions صعودی (Very Strong Buy)
    trend_score = TrendScore(
        score=0.85,
        signal=SignalStrength.VERY_BULLISH,
        accuracy=0.88,
        indicators_count=10,
        indicators=[],
        description="روند صعودی قوی با تایید 9 از 10 اندیکاتور"
    )
    
    momentum_score = MomentumScore(
        score=0.75,
        signal=SignalStrength.BULLISH,
        accuracy=0.82,
        indicators_count=8,
        indicators=[],
        description="مومنتوم مثبت، RSI: 65, MACD صعودی"
    )
    
    volatility_score = VolatilityScore(
        score=0.45,
        signal=SignalStrength.BULLISH,
        accuracy=0.75,
        indicators_count=8,
        indicators=[],
        description="نوسان متوسط، BB در حال گسترش"
    )
    
    cycle_score = CycleScore(
        score=0.70,
        signal=SignalStrength.BULLISH,
        accuracy=0.80,
        indicators_count=7,
        phase="MARKUP",
        phase_strength=0.75,
        indicators=[],
        description="فاز صعود (Markup) قوی"
    )
    
    sr_score = SupportResistanceScore(
        score=0.65,
        signal=SignalStrength.BULLISH,
        accuracy=0.77,
        indicators_count=6,
        nearest_level_type="SUPPORT",
        nearest_level_distance=0.02,
        indicators=[],
        description="بالای حمایت قوی $49,000"
    )
    
    return trend_score, momentum_score, volatility_score, cycle_score, sr_score


def print_decision_report(decision):
    """چاپ گزارش کامل تصمیم"""
    
    print("=" * 80)
    print("📊 گزارش تصمیم‌گیری 5 بُعدی (5D Decision Matrix)")
    print("=" * 80)
    print(f"\n⏰ زمان تحلیل: {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # سیگنال نهایی
    print(f"\n🎯 سیگنال نهایی: {decision.final_signal.value}")
    print(f"📈 امتیاز نهایی: {decision.final_score:+.3f} (از -1 تا +1)")
    print(f"✅ اطمینان: {decision.final_confidence * 100:.1f}%")
    print(f"💪 قدرت سیگنال: {decision.signal_strength * 100:.1f}%")
    
    # ریسک
    print(f"\n⚠️ سطح ریسک: {decision.risk_level.value}")
    if decision.risk_factors:
        print("   عوامل ریسک:")
        for factor in decision.risk_factors:
            print(f"   - {factor}")
    
    # وضعیت هر dimension
    print("\n" + "─" * 80)
    print("📊 وضعیت هر بُعد (Dimension):")
    print("─" * 80)
    
    for name, dim in decision.dimensions.items():
        print(f"\n{dim.name}:")
        print(f"  امتیاز پایه: {dim.score:+.3f}")
        print(f"  تعدیل حجم: {dim.volume_adjustment:+.3f}")
        print(f"  امتیاز نهایی: {dim.volume_adjusted_score:+.3f}")
        print(f"  اطمینان: {dim.confidence * 100:.1f}%")
        print(f"  وزن در تصمیم نهایی: {dim.weight * 100:.1f}%")
        print(f"  وضعیت: {dim.description}")
    
    # تحلیل توافق
    print("\n" + "─" * 80)
    print("🤝 تحلیل توافق بین ابعاد:")
    print("─" * 80)
    
    agreement = decision.agreement
    print(f"\nتوافق کلی: {agreement.overall_agreement * 100:.1f}%")
    
    print(f"\n🟢 ابعاد صعودی ({len(agreement.bullish_dimensions)}):")
    for dim in agreement.bullish_dimensions:
        print(f"   - {dim}")
    
    print(f"\n🔴 ابعاد نزولی ({len(agreement.bearish_dimensions)}):")
    for dim in agreement.bearish_dimensions:
        print(f"   - {dim}")
    
    print(f"\n⚪ ابعاد خنثی ({len(agreement.neutral_dimensions)}):")
    for dim in agreement.neutral_dimensions:
        print(f"   - {dim}")
    
    print(f"\n💪 قوی‌ترین بُعد: {agreement.strongest_dimension}")
    print(f"🔻 ضعیف‌ترین بُعد: {agreement.weakest_dimension}")
    
    if agreement.conflicting:
        print("\n⚠️ هشدار: تناقض بین ابعاد وجود دارد!")
    
    # توصیه‌ها
    print("\n" + "═" * 80)
    print("💡 توصیه‌های معاملاتی:")
    print("═" * 80)
    
    print(f"\n{decision.recommendation}")
    
    print(f"\n📍 استراتژی ورود:")
    print(f"   {decision.entry_strategy}")
    
    print(f"\n📍 استراتژی خروج:")
    print(f"   {decision.exit_strategy}")
    
    print(f"\n🛑 پیشنهاد استاپ لاس:")
    print(f"   {decision.stop_loss_suggestion}")
    
    print(f"\n🎯 پیشنهاد حد سود:")
    print(f"   {decision.take_profit_suggestion}")
    
    # شرایط بازار
    print(f"\n🌍 شرایط کلی بازار:")
    print(f"   {decision.market_condition}")
    
    # نکات کلیدی
    if decision.key_insights:
        print(f"\n💎 نکات کلیدی:")
        for insight in decision.key_insights:
            print(f"   {insight}")
    
    print("\n" + "=" * 80)


def demo_scenario_1_very_strong_buy():
    """سناریو 1: سیگنال خرید بسیار قوی"""
    print("\n\n" + "🟢" * 40)
    print("سناریو 1: Very Strong Buy - همه ابعاد صعودی")
    print("🟢" * 40)
    
    candles = create_sample_candles(100)
    trend, momentum, volatility, cycle, sr = create_sample_scores()
    
    # ایجاد matrix
    matrix = FiveDimensionalDecisionMatrix(
        candles=candles,
        use_volume_matrix=False  # برای سادگی، بدون volume matrix
    )
    
    # تحلیل
    decision = matrix.analyze(trend, momentum, volatility, cycle, sr)
    
    # چاپ گزارش
    print_decision_report(decision)
    
    return decision


def demo_scenario_2_conflicting():
    """سناریو 2: تناقض بین ابعاد"""
    print("\n\n" + "🟡" * 40)
    print("سناریو 2: Conflicting Signals - تناقض بین ابعاد")
    print("🟡" * 40)
    
    candles = create_sample_candles(100)
    
    # Trend صعودی
    trend = TrendScore(
        score=0.70,
        signal=SignalStrength.BULLISH,
        accuracy=0.80,
        indicators_count=10,
        indicators=[],
        description="روند صعودی"
    )
    
    # اما Momentum نزولی (واگرایی!)
    momentum = MomentumScore(
        score=-0.40,
        signal=SignalStrength.BEARISH,
        accuracy=0.75,
        indicators_count=8,
        indicators=[],
        description="واگرایی نزولی، RSI: 78 (اشباع خرید)"
    )
    
    # Cycle در فاز توزیع
    cycle = CycleScore(
        score=-0.30,
        signal=SignalStrength.BEARISH,
        accuracy=0.70,
        indicators_count=7,
        phase="DISTRIBUTION",
        phase_strength=0.65,
        indicators=[],
        description="فاز توزیع - احتمال ریزش"
    )
    
    # Volatility بالا
    volatility = VolatilityScore(
        score=0.60,
        signal=SignalStrength.BULLISH,
        accuracy=0.65,
        indicators_count=8,
        indicators=[],
        description="نوسان بالا - ریسک افزایش یافته"
    )
    
    # S/R در مقاومت
    sr = SupportResistanceScore(
        score=-0.20,
        signal=SignalStrength.BEARISH,
        accuracy=0.70,
        indicators_count=6,
        nearest_level_type="RESISTANCE",
        nearest_level_distance=0.01,
        indicators=[],
        description="نزدیک مقاومت قوی $52,000"
    )
    
    matrix = FiveDimensionalDecisionMatrix(candles=candles, use_volume_matrix=False)
    decision = matrix.analyze(trend, momentum, volatility, cycle, sr)
    
    print_decision_report(decision)
    
    return decision


def demo_scenario_3_neutral():
    """سناریو 3: بازار خنثی"""
    print("\n\n" + "⚪" * 40)
    print("سناریو 3: Neutral Market - بازار خنثی")
    print("⚪" * 40)
    
    candles = create_sample_candles(100)
    
    # همه dimensions خنثی
    trend = TrendScore(
        score=0.05,
        signal=SignalStrength.NEUTRAL,
        accuracy=0.65,
        indicators_count=10,
        indicators=[],
        description="روند خنثی، رنج"
    )
    
    momentum = MomentumScore(
        score=-0.10,
        signal=SignalStrength.NEUTRAL,
        accuracy=0.60,
        indicators_count=8,
        indicators=[],
        description="مومنتوم خنثی، RSI: 48"
    )
    
    volatility = VolatilityScore(
        score=0.15,
        signal=SignalStrength.NEUTRAL,
        accuracy=0.70,
        indicators_count=8,
        indicators=[],
        description="نوسان پایین، consolidation"
    )
    
    cycle = CycleScore(
        score=0.10,
        signal=SignalStrength.NEUTRAL,
        accuracy=0.68,
        indicators_count=7,
        phase="ACCUMULATION",
        phase_strength=0.50,
        indicators=[],
        description="فاز انباشت اولیه"
    )
    
    sr = SupportResistanceScore(
        score=0.08,
        signal=SignalStrength.NEUTRAL,
        accuracy=0.65,
        indicators_count=6,
        nearest_level_type="NONE",
        nearest_level_distance=0.10,
        indicators=[],
        description="دور از سطوح کلیدی"
    )
    
    matrix = FiveDimensionalDecisionMatrix(candles=candles, use_volume_matrix=False)
    decision = matrix.analyze(trend, momentum, volatility, cycle, sr)
    
    print_decision_report(decision)
    
    return decision


def main():
    """اجرای همه سناریوها"""
    
    print("\n" + "=" * 80)
    print("🚀 نمایش سیستم تصمیم‌گیری 5 بُعدی (5D Decision Matrix)")
    print("=" * 80)
    
    # سناریو 1: خرید قوی
    decision1 = demo_scenario_1_very_strong_buy()
    
    # سناریو 2: تناقض
    decision2 = demo_scenario_2_conflicting()
    
    # سناریو 3: خنثی
    decision3 = demo_scenario_3_neutral()
    
    # خلاصه مقایسه
    print("\n\n" + "=" * 80)
    print("📊 مقایسه سناریوها:")
    print("=" * 80)
    
    scenarios = [
        ("سناریو 1: Very Strong Buy", decision1),
        ("سناریو 2: Conflicting", decision2),
        ("سناریو 3: Neutral", decision3)
    ]
    
    for name, decision in scenarios:
        print(f"\n{name}:")
        print(f"  سیگنال: {decision.final_signal.value}")
        print(f"  امتیاز: {decision.final_score:+.3f}")
        print(f"  اطمینان: {decision.final_confidence * 100:.1f}%")
        print(f"  قدرت: {decision.signal_strength * 100:.1f}%")
        print(f"  ریسک: {decision.risk_level.value}")
        print(f"  توافق: {decision.agreement.overall_agreement * 100:.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ پایان نمایش")
    print("=" * 80)


if __name__ == "__main__":
    main()
