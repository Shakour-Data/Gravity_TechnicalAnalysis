"""
Populate Historical Indicator Scores
اسکریپت ذخیره مقادیر اندیکاتورها برای هر نماد و هر روز

Author: Gravity Tech Team
Date: December 6, 2025
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path


def populate_indicator_values():
    """
    اندیکاتورها را برای تمام نمادها و تاریخ‌ها ذخیره کن
    """
    db_path = Path("data/gravity_tech.db")
    if not db_path.exists():
        print("❌ Database not found")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 1. دریافت تمام historical_scores موجود
        cursor.execute("""
            SELECT id, ticker, analysis_date, timeframe,
                   trend_score, momentum_score, combined_score,
                   volume_score, volatility_score, cycle_score
            FROM historical_scores
            ORDER BY ticker, analysis_date
        """)
        
        scores = cursor.fetchall()
        print(f"📊 Found {len(scores)} historical scores")
        
        # 2. برای هر score، indicator values را درج کن
        indicators_data = []
        
        for score_id, ticker, analysis_date, timeframe, trend_score, momentum_score, \
            combined_score, volume_score, volatility_score, cycle_score in scores:

            # تعریف اندیکاتورها و مقادیرشان
            indicators = [
                {
                    'name': 'TREND_SCORE',
                    'category': 'TREND',
                    'value': trend_score,
                    'params': {'method': 'moving_average'},
                    'signal': 'BULLISH' if trend_score > 0.55 else 'BEARISH' if trend_score < 0.45 else 'NEUTRAL'
                },
                {
                    'name': 'MOMENTUM_SCORE',
                    'category': 'MOMENTUM',
                    'value': momentum_score,
                    'params': {'method': 'rsi_macd'},
                    'signal': 'BULLISH' if momentum_score > 0.55 else 'BEARISH' if momentum_score < 0.45 else 'NEUTRAL'
                },
                {
                    'name': 'VOLUME_SCORE',
                    'category': 'VOLUME',
                    'value': volume_score,
                    'params': {'method': 'obv_ad'},
                    'signal': 'BULLISH' if volume_score > 0.55 else 'BEARISH' if volume_score < 0.45 else 'NEUTRAL'
                },
                {
                    'name': 'VOLATILITY_SCORE',
                    'category': 'VOLATILITY',
                    'value': volatility_score,
                    'params': {'method': 'atr_bb'},
                    'signal': 'HIGH' if volatility_score > 0.6 else 'LOW'
                },
                {
                    'name': 'CYCLE_SCORE',
                    'category': 'CYCLE',
                    'value': cycle_score,
                    'params': {'method': 'fibonacci'},
                    'signal': 'CYCLE_UP' if cycle_score > 0.5 else 'CYCLE_DOWN'
                },
                {
                    'name': 'COMBINED_SCORE',
                    'category': 'COMPOSITE',
                    'value': combined_score,
                    'params': {'weights': {'trend': 0.6, 'momentum': 0.4}},
                    'signal': 'STRONG_BUY' if combined_score > 0.65 else 'BUY' if combined_score > 0.55 else 'STRONG_SELL' if combined_score < 0.35 else 'SELL' if combined_score < 0.45 else 'HOLD'
                }
            ]
            
            # درج اندیکاتورها
            for ind in indicators:
                indicators_data.append((
                    score_id,
                    ticker,
                    analysis_date,
                    timeframe,
                    ind['name'],
                    ind['category'],
                    json.dumps(ind['params']),
                    ind['value'],
                    ind['signal'],
                    0.85
                ))

        print(f"💾 Inserting {len(indicators_data)} indicator values...")

        # Batch insert
        cursor.executemany("""
            INSERT OR IGNORE INTO historical_indicator_scores
            (score_id, ticker, analysis_date, timeframe, indicator_name,
             indicator_category, indicator_params, value, signal, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, indicators_data)

        conn.commit()

        # 3. تعداد کل رکورد‌های درج شده
        cursor.execute("SELECT COUNT(*) FROM historical_indicator_scores")
        total_count = cursor.fetchone()[0]

        print(f"✅ Successfully inserted indicator values!")
        print(f"   • Total indicator records: {total_count:,}")
        print(f"   • Average indicators per score: {total_count / len(scores):.1f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("📈 Populating Indicator Values")
    print("=" * 60)
    populate_indicator_values()
    print("=" * 60)

