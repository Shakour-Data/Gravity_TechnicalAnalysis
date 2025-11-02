"""
مثال کامل سیستم تاریخی امتیازدهی
====================================

این مثال نشان می‌دهد چگونه:
1. امتیازها به صورت خودکار ذخیره شوند
2. امتیازهای تاریخی بازیابی شوند
3. نمودارهای سری زمانی ترسیم شوند
4. عملکرد اندیکاتورها تحلیل شود
"""

from datetime import datetime, timedelta
from database.historical_manager import HistoricalScoreManager, HistoricalScoreEntry
from ml.multi_horizon_analysis import MultiHorizonTrendAnalyzer
from ml.multi_horizon_momentum_analysis import MultiHorizonMomentumAnalyzer
import matplotlib.pyplot as plt
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
# 1️⃣ تنظیمات
# ═══════════════════════════════════════════════════════════════════

DATABASE_URL = "postgresql://trading_user:password@localhost:5432/trading_db"
SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"


# ═══════════════════════════════════════════════════════════════════
# 2️⃣ تابع اصلی تحلیل با ذخیره خودکار
# ═══════════════════════════════════════════════════════════════════

def analyze_and_save(
    symbol: str,
    candles: list,
    price: float,
    timestamp: datetime,
    db_manager: HistoricalScoreManager
):
    """
    تحلیل کامل + ذخیره در دیتابیس
    
    این تابع باید هر بار که تحلیل می‌کنید فراخوانی شود
    """
    
    print(f"\n{'='*70}")
    print(f"🔍 Analyzing {symbol} at {timestamp}")
    print(f"{'='*70}")
    
    # 1. استخراج ویژگی‌ها
    trend_extractor = TrendFeatureExtractor()
    momentum_extractor = MomentumFeatureExtractor()
    
    trend_features = trend_extractor.extract(candles)
    momentum_features = momentum_extractor.extract(candles)
    
    # 2. تحلیل روند
    trend_analyzer = MultiHorizonTrendAnalyzer.load("models/trend")
    trend_result = trend_analyzer.analyze(trend_features)
    
    # 3. تحلیل مومنتوم
    momentum_analyzer = MultiHorizonMomentumAnalyzer.load("models/momentum")
    momentum_result = momentum_analyzer.analyze(momentum_features)
    
    # 4. محاسبه Combined
    trend_weight = 0.6
    momentum_weight = 0.4
    
    # میانگین وزن‌دار
    trend_overall = sum(h.score * h.confidence for h in trend_result) / sum(h.confidence for h in trend_result)
    momentum_overall = sum(h.score * h.confidence for h in momentum_result) / sum(h.confidence for h in momentum_result)
    
    combined_score = (trend_overall * trend_weight) + (momentum_overall * momentum_weight)
    combined_confidence = (
        sum(h.confidence for h in trend_result) / len(trend_result) * trend_weight +
        sum(h.confidence for h in momentum_result) / len(momentum_result) * momentum_weight
    ) * 2
    
    # 5. تعیین سیگنال‌ها و توصیه
    def get_signal(score):
        if score >= 0.8: return "VERY_BULLISH"
        elif score >= 0.4: return "BULLISH"
        elif score >= -0.1: return "NEUTRAL"
        elif score >= -0.4: return "BEARISH"
        else: return "VERY_BEARISH"
    
    def get_recommendation(score):
        if score >= 0.7: return "STRONG_BUY"
        elif score >= 0.3: return "BUY"
        elif score >= -0.1: return "HOLD"
        elif score >= -0.3: return "SELL"
        else: return "STRONG_SELL"
    
    # 6. ساخت Entry برای دیتابیس
    score_entry = HistoricalScoreEntry(
        symbol=symbol,
        timestamp=timestamp,
        timeframe=TIMEFRAME,
        trend_score=trend_overall,
        trend_confidence=sum(h.confidence for h in trend_result) / len(trend_result),
        momentum_score=momentum_overall,
        momentum_confidence=sum(h.confidence for h in momentum_result) / len(momentum_result),
        combined_score=combined_score,
        combined_confidence=combined_confidence,
        trend_weight=trend_weight,
        momentum_weight=momentum_weight,
        trend_signal=get_signal(trend_overall),
        momentum_signal=get_signal(momentum_overall),
        combined_signal=get_signal(combined_score),
        recommendation=get_recommendation(combined_score),
        action=get_recommendation(combined_score),  # simplified
        price_at_analysis=price
    )
    
    # 7. جمع‌آوری horizon scores
    horizon_scores = []
    for h in trend_result:
        horizon_scores.append({
            'horizon': f'{h.horizon}d',
            'analysis_type': 'TREND',
            'score': h.score,
            'confidence': h.confidence,
            'signal': h.signal.value
        })
    
    for h in momentum_result:
        horizon_scores.append({
            'horizon': f'{h.horizon}d',
            'analysis_type': 'MOMENTUM',
            'score': h.score,
            'confidence': h.confidence,
            'signal': h.signal.value
        })
    
    # 8. جمع‌آوری indicator scores (اگر موجود باشد)
    indicator_scores = []
    if hasattr(trend_features, 'get_indicator_details'):
        for ind in trend_features.get_indicator_details():
            indicator_scores.append({
                'name': ind['name'],
                'category': 'TREND',
                'params': ind.get('params', {}),
                'score': ind['score'],
                'confidence': ind['confidence'],
                'signal': ind['signal'],
                'raw_value': ind.get('value')
            })
    
    # 9. جمع‌آوری patterns (اگر موجود باشد)
    patterns = []
    if hasattr(candles, 'detected_patterns'):
        for p in candles.detected_patterns:
            patterns.append({
                'type': p['type'],
                'name': p['name'],
                'score': p['score'],
                'confidence': p['confidence'],
                'signal': p['signal'],
                'description': p.get('description'),
                'candle_indices': p.get('indices', []),
                'price_levels': p.get('levels', {}),
                'projected_target': p.get('target')
            })
    
    # 10. ذخیره در دیتابیس
    try:
        score_id = db_manager.save_score(
            score_entry,
            horizon_scores=horizon_scores,
            indicator_scores=indicator_scores if indicator_scores else None,
            patterns=patterns if patterns else None
        )
        
        print(f"✅ Saved to database with ID: {score_id}")
        print(f"📊 Trend: {trend_overall:.3f}, Momentum: {momentum_overall:.3f}, Combined: {combined_score:.3f}")
        print(f"🎯 Recommendation: {score_entry.recommendation}")
        
        return score_id
        
    except Exception as e:
        print(f"❌ Error saving to database: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# 3️⃣ بازیابی و نمایش امتیازهای تاریخی
# ═══════════════════════════════════════════════════════════════════

def show_historical_scores(symbol: str, days: int = 30):
    """
    نمایش و رسم نمودار امتیازهای تاریخی
    """
    with HistoricalScoreManager(DATABASE_URL) as manager:
        # دریافت آخرین امتیاز
        latest = manager.get_latest_score(symbol, TIMEFRAME)
        if latest:
            print(f"\n{'='*70}")
            print(f"📊 Latest Score for {symbol}")
            print(f"{'='*70}")
            print(f"Timestamp: {latest['timestamp']}")
            print(f"Price: ${latest['price_at_analysis']:,.2f}")
            print(f"Trend: {latest['trend_score']:.3f} (confidence: {latest['trend_confidence']:.2f})")
            print(f"Momentum: {latest['momentum_score']:.3f} (confidence: {latest['momentum_confidence']:.2f})")
            print(f"Combined: {latest['combined_score']:.3f} (confidence: {latest['combined_confidence']:.2f})")
            print(f"Signal: {latest['combined_signal']}")
            print(f"Recommendation: {latest['recommendation']}")
        
        # دریافت سری زمانی
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        timeseries = manager.get_score_timeseries(symbol, from_date, to_date, TIMEFRAME)
        
        if not timeseries:
            print(f"\n⚠️  No historical data found for {symbol}")
            return
        
        # تبدیل به DataFrame
        df = pd.DataFrame(timeseries)
        
        print(f"\n{'='*70}")
        print(f"📈 Historical Data: {len(df)} records over {days} days")
        print(f"{'='*70}")
        print(df.head())
        
        # رسم نمودار
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # نمودار 1: امتیازها
        axes[0].plot(df['timestamp'], df['trend_score'], label='Trend', linewidth=2, color='blue')
        axes[0].plot(df['timestamp'], df['momentum_score'], label='Momentum', linewidth=2, color='orange')
        axes[0].plot(df['timestamp'], df['combined_score'], label='Combined', linewidth=2, color='green')
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].axhline(y=0.7, color='green', linestyle=':', alpha=0.3, label='Strong Buy')
        axes[0].axhline(y=-0.7, color='red', linestyle=':', alpha=0.3, label='Strong Sell')
        axes[0].set_title(f'{symbol} - Historical Scores', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Score [-1, +1]')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # نمودار 2: قیمت
        axes[1].plot(df['timestamp'], df['price'], linewidth=2, color='black')
        axes[1].set_title(f'{symbol} - Price', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Price ($)')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'historical_scores_{symbol}_{days}d.png', dpi=150)
        print(f"\n💾 Chart saved: historical_scores_{symbol}_{days}d.png")
        plt.show()


# ═══════════════════════════════════════════════════════════════════
# 4️⃣ تحلیل عملکرد اندیکاتورها
# ═══════════════════════════════════════════════════════════════════

def analyze_indicator_performance(symbol: str, days: int = 30):
    """
    تحلیل اینکه کدام اندیکاتورها دقیق‌تر بودند
    """
    with HistoricalScoreManager(DATABASE_URL) as manager:
        performance = manager.get_indicator_performance(symbol, days)
        
        if not performance:
            print(f"\n⚠️  No indicator data found")
            return
        
        print(f"\n{'='*70}")
        print(f"🎯 Indicator Performance for {symbol} (Last {days} days)")
        print(f"{'='*70}")
        
        df = pd.DataFrame(performance)
        df = df.sort_values('avg_confidence', ascending=False)
        
        print("\nTop 10 Most Reliable Indicators:")
        print(df[['indicator_name', 'indicator_category', 'avg_confidence', 'avg_score', 'usage_count']].head(10).to_string(index=False))
        
        # نمودار
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_20 = df.head(20)
        colors = ['blue' if cat == 'TREND' else 'orange' for cat in top_20['indicator_category']]
        
        ax.barh(top_20['indicator_name'], top_20['avg_confidence'], color=colors, alpha=0.7)
        ax.set_xlabel('Average Confidence')
        ax.set_title(f'Top 20 Indicator Performance - {symbol}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.7, label='Trend'),
            Patch(facecolor='orange', alpha=0.7, label='Momentum')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(f'indicator_performance_{symbol}_{days}d.png', dpi=150)
        print(f"\n💾 Chart saved: indicator_performance_{symbol}_{days}d.png")
        plt.show()


# ═══════════════════════════════════════════════════════════════════
# 5️⃣ دریافت امتیاز در یک تاریخ خاص
# ═══════════════════════════════════════════════════════════════════

def get_score_at_specific_date(symbol: str, date_str: str):
    """
    دریافت امتیاز در یک تاریخ خاص
    
    مثال: get_score_at_specific_date("BTCUSDT", "2024-01-15 10:00:00")
    """
    target_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    
    with HistoricalScoreManager(DATABASE_URL) as manager:
        score = manager.get_score_at_date(symbol, target_date, TIMEFRAME)
        
        if score:
            print(f"\n{'='*70}")
            print(f"📅 Score at {date_str}")
            print(f"{'='*70}")
            print(f"Symbol: {symbol}")
            print(f"Actual timestamp: {score['timestamp']}")
            print(f"Price: ${score['price']:,.2f}")
            print(f"Trend Score: {score['trend_score']:.3f}")
            print(f"Momentum Score: {score['momentum_score']:.3f}")
            print(f"Combined Score: {score['combined_score']:.3f}")
            print(f"Recommendation: {score['recommendation']}")
        else:
            print(f"\n❌ No score found for {symbol} at {date_str}")


# ═══════════════════════════════════════════════════════════════════
# 6️⃣ مثال اجرای کامل
# ═══════════════════════════════════════════════════════════════════

def main():
    """
    مثال کامل استفاده از سیستم تاریخی
    """
    print("\n" + "="*70)
    print("🚀 Historical Scoring System - Complete Example")
    print("="*70)
    
    # 1. نمایش آخرین امتیازها و نمودارها
    print("\n▶️  Step 1: Showing historical scores...")
    try:
        show_historical_scores(SYMBOL, days=30)
    except Exception as e:
        print(f"⚠️  Error: {e}")
    
    # 2. تحلیل عملکرد اندیکاتورها
    print("\n▶️  Step 2: Analyzing indicator performance...")
    try:
        analyze_indicator_performance(SYMBOL, days=30)
    except Exception as e:
        print(f"⚠️  Error: {e}")
    
    # 3. دریافت امتیاز در تاریخ خاص
    print("\n▶️  Step 3: Getting score at specific date...")
    try:
        one_week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:00:00")
        get_score_at_specific_date(SYMBOL, one_week_ago)
    except Exception as e:
        print(f"⚠️  Error: {e}")
    
    print("\n" + "="*70)
    print("✅ Example completed!")
    print("="*70)
    
    print("""
💡 نکات مهم:

1. هر بار که analyze() فراخوانی می‌شود، نتایج خودکار ذخیره می‌شوند
2. تمام امتیازها، اندیکاتورها، و الگوها تاریخی هستند
3. می‌توانید امتیاز هر تاریخی را بازیابی کنید
4. نمودارهای سری زمانی برای تحلیل روند
5. آمار عملکرد برای بهبود سیستم
6. Backtest برای ارزیابی استراتژی

📊 API Endpoints که باید ایجاد شوند:

GET /api/v1/history/BTCUSDT?from=2024-01-01&to=2024-01-31
GET /api/v1/history/BTCUSDT/latest
GET /api/v1/history/BTCUSDT/at/2024-01-15T10:00:00
GET /api/v1/indicators/performance?symbol=BTCUSDT&days=30
GET /api/v1/patterns/success-rate?days=90
    """)


if __name__ == "__main__":
    # تست ساده بدون دیتابیس واقعی
    print("""
⚠️  برای اجرای این مثال، ابتدا:

1. PostgreSQL را نصب کنید
2. دیتابیس ایجاد کنید: CREATE DATABASE trading_db;
3. Schema را اجرا کنید: psql -d trading_db -f database/schemas.sql
4. CONNECTION_STRING را در کد تنظیم کنید

سپس این فایل را اجرا کنید.
    """)
    
    # main()  # Uncomment وقتی دیتابیس آماده شد
