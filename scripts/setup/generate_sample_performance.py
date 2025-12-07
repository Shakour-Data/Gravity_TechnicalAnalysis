#!/usr/bin/env python
"""Generate Sample Tool Performance Data - تولید نمونه داده‌های عملکرد"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def generate_sample_performance():
    """تولید نمونه داده‌های عملکرد برای 100 نماد × 100 روز آخر × 6 ابزار"""

    print("=" * 70)
    print("🔧 تولید نمونه داده‌های عملکرد")
    print("=" * 70)

    conn = sqlite3.connect("data/gravity_tech.db")
    cursor = conn.cursor()

    tools = [
        ('trend_analyzer', 'trend'),
        ('momentum_indicator', 'momentum'),
        ('volume_analyzer', 'volume'),
        ('volatility_indicator', 'volatility'),
        ('support_resistance', 'support_resistance'),
        ('cycle_detector', 'cycle'),
    ]

    # Get top 100 tickers by data volume
    cursor.execute("""
        SELECT ticker, COUNT(*) as cnt
        FROM historical_scores
        GROUP BY ticker
        ORDER BY cnt DESC
        LIMIT 100
    """)
    tickers = [row[0] for row in cursor.fetchall()]

    # Get last 100 dates
    cursor.execute("""
        SELECT DISTINCT analysis_date
        FROM historical_scores
        ORDER BY analysis_date DESC
        LIMIT 100
    """)
    dates = [row[0] for row in cursor.fetchall()]

    print("\n📊 پارامترها:")
    print(f"  • تعداد ابزارها: {len(tools)}")
    print(f"  • تعداد نمادها: {len(tickers)}")
    print(f"  • تعداد روزها: {len(dates)}")
    print(f"  • کل رکوردها: {len(tools) * len(tickers) * len(dates):,}")

    cursor.execute("DELETE FROM tool_performance_history")
    conn.commit()

    total = 0
    dates_list = list(reversed(dates))  # از قدیم به جدید

    for tool_name, tool_category in tools:
        for ticker in tickers:
            for idx, date in enumerate(dates_list):
                try:
                    cursor.execute("""
                        SELECT trend_score, momentum_score, combined_score, price_at_analysis
                        FROM historical_scores
                        WHERE ticker = ? AND analysis_date = ?
                    """, (ticker, date))

                    row = cursor.fetchone()
                    if not row:
                        continue

                    trend_score, momentum_score, combined_score, price = row

                    # Tool-specific score
                    if tool_category == 'trend':
                        score = trend_score
                    elif tool_category == 'momentum':
                        score = momentum_score
                    else:
                        score = combined_score

                    prediction = 'bullish' if score > 0.55 else 'bearish' if score < 0.45 else 'neutral'

                    # Get next day result
                    actual_change = 0
                    success = 0
                    if idx + 1 < len(dates_list):
                        next_date = dates_list[idx + 1]
                        cursor.execute("""
                            SELECT price_at_analysis
                            FROM historical_scores
                            WHERE ticker = ? AND analysis_date = ?
                        """, (ticker, next_date))

                        next_row = cursor.fetchone()
                        if next_row and price > 0:
                            next_price = next_row[0]
                            actual_change = ((next_price - price) / price * 100)
                            success = 1 if (
                                (prediction == 'bullish' and actual_change > 0) or
                                (prediction == 'bearish' and actual_change < 0) or
                                (prediction == 'neutral' and abs(actual_change) < 1)
                            ) else 0

                    cursor.execute("""
                        INSERT INTO tool_performance_history (
                            tool_name, tool_category, ticker, timeframe,
                            market_regime, volatility_level, trend_strength,
                            prediction_type, prediction_value, confidence_score,
                            actual_result, actual_price_change, success, accuracy,
                            prediction_timestamp, result_timestamp, evaluation_period_hours,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
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
                        0.7 if success else 0.3,
                        f"{date}T09:00:00",
                        f"{dates_list[idx+1] if idx+1 < len(dates_list) else date}T09:00:00",
                        24,
                        datetime.now().isoformat(),
                        datetime.now().isoformat()
                    ))

                    total += 1

                    if total % 10000 == 0:
                        conn.commit()
                        print(f"  Progress: {total:,}")

                except Exception:
                    pass

    conn.commit()

    # Update stats
    print("\n📊 محاسبه آمار...")
    cursor.execute("DELETE FROM tool_performance_stats")

    for tool_name, tool_category in tools:
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(success) as correct,
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
            total_pred, correct, avg_conf, avg_change, bullish, bearish, neutral, b_correct, be_correct, n_correct = stats

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
                dates_list[0], dates_list[-1],
                int(total_pred), int(correct),
                (correct / total_pred * 100),
                avg_conf or 0, avg_change or 0,
                int(bullish), int(bearish), int(neutral),
                (b_correct / max(bullish, 1) * 100),
                (be_correct / max(bearish, 1) * 100),
                (n_correct / max(neutral, 1) * 100),
                90.0, 40.0,
                datetime.now().isoformat()
            ))

    conn.commit()

    print(f"\n✓ کل رکوردها: {total:,}")
    print("\n" + "=" * 70)
    print("✅ تکمیل شد!")
    print("=" * 70)

    cursor.close()
    conn.close()


if __name__ == "__main__":
    generate_sample_performance()

