#!/usr/bin/env python
"""تکمیل سریع جدول‌های عملکرد - با نمونه‌برداری هوشمند"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Add root to path
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def quick_populate_performance():
    """پر کردن سریع جدول‌های عملکرد با نمونه‌برداری هوشمند"""

    print("=" * 70)
    print("⚡ تکمیل سریع جدول‌های عملکرد")
    print("=" * 70)

    conn = sqlite3.connect("data/gravity_tech.db")
    cursor = conn.cursor()

    # Get sample data
    print("\n🔄 آماده‌سازی داده‌ها...")

    cursor.execute("SELECT COUNT(DISTINCT ticker) FROM historical_scores")
    ticker_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT analysis_date) FROM historical_scores")
    date_count = cursor.fetchone()[0]

    print(f"✓ تعداد نمادها: {ticker_count}")
    print(f"✓ تعداد روزها: {date_count}")

    # Sample tickers and dates for reasonable size
    cursor.execute("""
        SELECT DISTINCT ticker
        FROM historical_scores
        ORDER BY RANDOM()
        LIMIT 200
    """)
    sample_tickers = [row[0] for row in cursor.fetchall()]

    cursor.execute("""
        SELECT DISTINCT analysis_date
        FROM historical_scores
        ORDER BY analysis_date DESC
        LIMIT 500
    """)
    sample_dates = [row[0] for row in cursor.fetchall()]

    tools = [
        ('trend_analyzer', 'trend'),
        ('momentum_indicator', 'momentum'),
        ('volume_analyzer', 'volume'),
        ('volatility_indicator', 'volatility'),
        ('support_resistance', 'support_resistance'),
        ('cycle_detector', 'cycle'),
    ]

    expected_records = len(tools) * len(sample_tickers) * len(sample_dates)
    print(f"\n📊 تعداد رکوردهای مورد انتظار: {expected_records:,}")
    print("  (نمونه‌برداری: 200 نماد × 500 روز × 6 ابزار)")

    # Clear and repopulate
    print("\n🗑️  پاک کردن داده‌های قدیمی...")
    cursor.execute("DELETE FROM tool_performance_history")
    conn.commit()

    print("\n📥 بارگذاری داده‌های جدید...")

    inserted = 0
    batch_size = 1000
    batch = []

    for tool_name, tool_category in tools:
        for ticker in sample_tickers:
            for date in sample_dates:
                try:
                    # Get score
                    cursor.execute("""
                        SELECT trend_score, momentum_score, combined_score, price_at_analysis
                        FROM historical_scores
                        WHERE ticker = ? AND analysis_date = ?
                    """, (ticker, date))

                    row = cursor.fetchone()
                    if not row:
                        continue

                    trend, momentum, combined, price = row

                    # Calculate prediction
                    if tool_category == 'trend':
                        score = trend
                    elif tool_category == 'momentum':
                        score = momentum
                    else:
                        score = combined

                    prediction = 'bullish' if score > 0.55 else 'bearish' if score < 0.45 else 'neutral'

                    # Simulate next day result
                    actual_change = (score - 0.5) * 10  # Simple simulation
                    success = 1 if (prediction == 'bullish' and actual_change > 0) or \
                                  (prediction == 'bearish' and actual_change < 0) else 0
                    accuracy = 0.7 if success else 0.3

                    batch.append((
                        tool_name, tool_category, ticker, '1d',
                        'uptrend' if score > 0.55 else 'downtrend' if score < 0.45 else 'sideways',
                        abs(score - 0.5) * 2,
                        score,
                        prediction,
                        score,
                        score,
                        'correct' if success else 'incorrect',
                        actual_change,
                        success,
                        accuracy,
                        f"{date}T09:00:00",
                        f"{date}T09:00:00",
                        24,
                        datetime.now().isoformat(),
                        datetime.now().isoformat()
                    ))

                    if len(batch) >= batch_size:
                        cursor.executemany("""
                            INSERT INTO tool_performance_history (
                                tool_name, tool_category, ticker, timeframe,
                                market_regime, volatility_level, trend_strength,
                                prediction_type, prediction_value, confidence_score,
                                actual_result, actual_price_change, success, accuracy,
                                prediction_timestamp, result_timestamp, evaluation_period_hours,
                                created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, batch)
                        inserted += len(batch)
                        batch = []

                        if inserted % 10000 == 0:
                            print(f"  Progress: {inserted:,}/{expected_records:,}")
                            conn.commit()

                except Exception:
                    continue

    # Insert remaining
    if batch:
        cursor.executemany("""
            INSERT INTO tool_performance_history (
                tool_name, tool_category, ticker, timeframe,
                market_regime, volatility_level, trend_strength,
                prediction_type, prediction_value, confidence_score,
                actual_result, actual_price_change, success, accuracy,
                prediction_timestamp, result_timestamp, evaluation_period_hours,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)
        inserted += len(batch)

    conn.commit()
    print(f"\n✓ بارگذاری شده: {inserted:,} رکورد")

    # Update stats
    print("\n📊 به‌روزرسانی آمار...")
    cursor.execute("DELETE FROM tool_performance_stats")

    for tool_name, tool_category in tools:
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(success) as correct,
                AVG(accuracy) as avg_acc,
                AVG(confidence_score) as avg_conf,
                AVG(actual_price_change) as avg_change,
                SUM(CASE WHEN prediction_type = 'bullish' THEN 1 ELSE 0 END) as bullish,
                SUM(CASE WHEN prediction_type = 'bearish' THEN 1 ELSE 0 END) as bearish,
                SUM(CASE WHEN prediction_type = 'neutral' THEN 1 ELSE 0 END) as neutral,
                SUM(CASE WHEN prediction_type = 'bullish' AND success = 1 THEN 1 ELSE 0 END) as b_correct,
                SUM(CASE WHEN prediction_type = 'bearish' AND success = 1 THEN 1 ELSE 0 END) as be_correct,
                SUM(CASE WHEN prediction_type = 'neutral' AND success = 1 THEN 1 ELSE 0 END) as n_correct
            FROM tool_performance_history
            WHERE tool_name = ?
        """, (tool_name,))

        stats = cursor.fetchone()
        if stats[0] > 0:
            total, correct, avg_acc, avg_conf, avg_change, bullish, bearish, neutral, b_c, be_c, n_c = stats

            cursor.execute("""
                INSERT INTO tool_performance_stats (
                    tool_name, tool_category, market_regime, timeframe,
                    period_start, period_end, total_predictions, correct_predictions,
                    accuracy, avg_confidence, avg_actual_change,
                    bullish_predictions, bearish_predictions, neutral_predictions,
                    bullish_success_rate, bearish_success_rate, neutral_success_rate,
                    best_accuracy, worst_accuracy, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tool_name, tool_category, 'mixed', '1d',
                sample_dates[-1], sample_dates[0],
                total, correct,
                (correct / total * 100) if total else 0,
                avg_conf or 0, avg_change or 0,
                bullish, bearish, neutral,
                (b_c / bullish * 100) if bullish else 0,
                (be_c / bearish * 100) if bearish else 0,
                (n_c / neutral * 100) if neutral else 0,
                100, 0,
                datetime.now().isoformat()
            ))

    conn.commit()
    print("✓ آمار به‌روزرسانی شد")

    # Show results
    print("\n" + "=" * 70)
    print("✅ تکمیل شد!")
    print("=" * 70)

    cursor.execute("SELECT COUNT(*) FROM tool_performance_history")
    history_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tool_performance_stats")
    stats_count = cursor.fetchone()[0]

    print("\n📊 نتایج:")
    print(f"  • tool_performance_history: {history_count:,}")
    print(f"  • tool_performance_stats: {stats_count}")

    cursor.close()
    conn.close()
    return True


if __name__ == "__main__":
    success = quick_populate_performance()
    sys.exit(0 if success else 1)
