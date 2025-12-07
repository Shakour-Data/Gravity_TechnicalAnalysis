#!/usr/bin/env python
"""Generate Complete Tool Performance Data - تولید داده‌های کامل عملکرد ابزارها"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Add root to path
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def generate_tool_performance_data():
    """تولید داده‌های عملکرد ابزار برای تمام روزها و نمادها"""

    print("=" * 70)
    print("🔧 تولید داده‌های عملکرد ابزارها")
    print("=" * 70)

    project_db_path = "data/gravity_tech.db"
    conn = sqlite3.connect(project_db_path)
    cursor = conn.cursor()

    tools = [
        ('trend_analyzer', 'trend'),
        ('momentum_indicator', 'momentum'),
        ('volume_analyzer', 'volume'),
        ('volatility_indicator', 'volatility'),
        ('support_resistance', 'support_resistance'),
        ('cycle_detector', 'cycle'),
    ]

    print("\n🔄 دریافت تمام تاریخ‌های موجود...")

    # Get all unique dates from historical_scores
    cursor.execute("""
        SELECT DISTINCT analysis_date
        FROM historical_scores
        ORDER BY analysis_date
    """)
    dates = [row[0] for row in cursor.fetchall()]
    print(f"✓ تعداد روزهای منحصر: {len(dates)}")

    # Get all tickers
    cursor.execute("SELECT DISTINCT ticker FROM historical_scores")
    tickers = [row[0] for row in cursor.fetchall()]
    print(f"✓ تعداد نمادهای منحصر: {len(tickers)}")

    print("\n📊 تولید داده‌های عملکرد...")
    print(f"  • تعداد ابزارها: {len(tools)}")
    print(f"  • تعداد روزها: {len(dates)}")
    print(f"  • تعداد نمادها: {len(tickers)}")
    print(f"  • کل رکوردهای مورد انتظار: {len(tools) * len(dates) * len(tickers):,}")

    total_inserted = 0
    progress_count = 0
    total_expected = len(tools) * len(dates) * len(tickers)

    # Clear existing tool_performance_history
    cursor.execute("DELETE FROM tool_performance_history")
    conn.commit()

    for tool_name, tool_category in tools:
        for date in dates:
            for ticker in tickers:
                progress_count += 1

                if progress_count % 100000 == 0:
                    print(f"  Progress: {progress_count:,}/{total_expected:,}")

                try:
                    # Get historical score for this date and ticker
                    cursor.execute("""
                        SELECT trend_score, momentum_score, combined_score, price_at_analysis
                        FROM historical_scores
                        WHERE ticker = ? AND analysis_date = ?
                    """, (ticker, date))

                    score_row = cursor.fetchone()
                    if not score_row:
                        continue

                    trend_score, momentum_score, combined_score, price = score_row

                    # Determine prediction based on score
                    if combined_score > 0.55:
                        prediction = 'bullish'
                        pred_value = 0.7 + combined_score * 0.3
                    elif combined_score < 0.45:
                        prediction = 'bearish'
                        pred_value = 0.3 - (1 - combined_score) * 0.3
                    else:
                        prediction = 'neutral'
                        pred_value = 0.5

                    # Tool-specific adjustments
                    if tool_category == 'trend':
                        confidence = trend_score
                        pred_value = trend_score
                    elif tool_category == 'momentum':
                        confidence = momentum_score
                        pred_value = momentum_score
                    elif tool_category == 'volume':
                        confidence = 0.5 + (momentum_score - 0.5) * 0.5
                        pred_value = confidence
                    elif tool_category == 'volatility':
                        confidence = 0.5 + abs(combined_score - 0.5) * 0.5
                        pred_value = confidence
                    else:
                        confidence = combined_score
                        pred_value = combined_score

                    # Get next day's data if available
                    next_date_idx = dates.index(date) + 1 if date in dates else -1
                    if next_date_idx >= 0 and next_date_idx < len(dates):
                        next_date = dates[next_date_idx]
                        cursor.execute("""
                            SELECT price_at_analysis
                            FROM historical_scores
                            WHERE ticker = ? AND analysis_date = ?
                        """, (ticker, next_date))

                        next_row = cursor.fetchone()
                        if next_row:
                            next_price = next_row[0]
                            actual_change = ((next_price - price) / price * 100) if price > 0 else 0
                            success = 1 if (prediction == 'bullish' and actual_change > 0) or \
                                          (prediction == 'bearish' and actual_change < 0) or \
                                          (prediction == 'neutral' and abs(actual_change) < 1) else 0
                            accuracy = 0.7 + abs(actual_change) / 100 if success else 0.3
                        else:
                            actual_change = 0
                            success = 0
                            accuracy = 0.5
                    else:
                        actual_change = 0
                        success = 0
                        accuracy = 0.5

                    prediction_ts = f"{date}T09:00:00"
                    result_ts = f"{dates[next_date_idx] if next_date_idx >= 0 and next_date_idx < len(dates) else date}T09:00:00"

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
                        'uptrend' if combined_score > 0.55 else 'downtrend' if combined_score < 0.45 else 'sideways',
                        abs(combined_score - 0.5) * 2,  # volatility_level
                        combined_score,  # trend_strength
                        prediction,
                        pred_value,
                        confidence,
                        'correct' if success else 'incorrect',
                        actual_change,
                        success,
                        accuracy,
                        prediction_ts,
                        result_ts,
                        24,
                        datetime.now().isoformat(),
                        datetime.now().isoformat()
                    ))

                    total_inserted += 1

                except Exception:
                    pass

    conn.commit()
    print(f"\n✓ تعداد رکوردهای inserted: {total_inserted:,}")

    # Now generate tool_performance_stats
    print("\n📊 تولید آمار عملکرد ابزارها...")

    cursor.execute("DELETE FROM tool_performance_stats")

    for tool_name, tool_category in tools:
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as correct,
                AVG(accuracy) as avg_accuracy,
                AVG(confidence_score) as avg_confidence,
                AVG(actual_price_change) as avg_change,
                SUM(CASE WHEN prediction_type = 'bullish' THEN 1 ELSE 0 END) as bullish_preds,
                SUM(CASE WHEN prediction_type = 'bearish' THEN 1 ELSE 0 END) as bearish_preds,
                SUM(CASE WHEN prediction_type = 'neutral' THEN 1 ELSE 0 END) as neutral_preds,
                SUM(CASE WHEN prediction_type = 'bullish' AND success = 1 THEN 1 ELSE 0 END) as bullish_correct,
                SUM(CASE WHEN prediction_type = 'bearish' AND success = 1 THEN 1 ELSE 0 END) as bearish_correct,
                SUM(CASE WHEN prediction_type = 'neutral' AND success = 1 THEN 1 ELSE 0 END) as neutral_correct,
                MAX(accuracy) as best_acc,
                MIN(accuracy) as worst_acc
            FROM tool_performance_history
            WHERE tool_name = ?
        """, (tool_name,))

        stats = cursor.fetchone()
        if stats[0] == 0:
            continue

        total, correct, avg_acc, avg_conf, avg_change, bullish, bearish, neutral, \
            bullish_correct, bearish_correct, neutral_correct, best_acc, worst_acc = stats

        accuracy = (correct / total * 100) if total > 0 else 0
        bullish_sr = (bullish_correct / max(bullish, 1) * 100) if bullish else 0
        bearish_sr = (bearish_correct / max(bearish, 1) * 100) if bearish else 0
        neutral_sr = (neutral_correct / max(neutral, 1) * 100) if neutral else 0

        try:
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
                dates[0] if dates else datetime.now().isoformat(),
                dates[-1] if dates else datetime.now().isoformat(),
                int(total) if total else 0,
                int(correct) if correct else 0,
                accuracy,
                avg_conf or 0,
                avg_change or 0,
                int(bullish) if bullish else 0,
                int(bearish) if bearish else 0,
                int(neutral) if neutral else 0,
                bullish_sr,
                bearish_sr,
                neutral_sr,
                best_acc or 0,
                worst_acc or 0,
                datetime.now().isoformat()
            ))
        except Exception:
            pass

    conn.commit()
    print("✓ آمار عملکرد ابزارها ایجاد شد")

    # Show final stats
    print("\n" + "=" * 70)
    print("✅ تولید داده‌های عملکرد کامل شد!")
    print("=" * 70)

    cursor.execute("SELECT COUNT(*) FROM tool_performance_history")
    history_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tool_performance_stats")
    stats_count = cursor.fetchone()[0]

    print("\n📊 آمارهای نهایی:")
    print(f"  • tool_performance_history: {history_count:,}")
    print(f"  • tool_performance_stats: {stats_count}")

    cursor.close()
    conn.close()

    return True


if __name__ == "__main__":
    success = generate_tool_performance_data()
    sys.exit(0 if success else 1)
