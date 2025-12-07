#!/usr/bin/env python
"""Complete Data Population Script - تکمیل دیتابیس"""

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add root to path
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def populate_historical_scores():
    """historical_scores را با قیمت‌های واقعی پر کن"""
    print("\n📊 Populating historical_scores...")

    tse_db_path = r"E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db"
    project_db_path = "data/gravity_tech.db"

    try:
        tse_conn = sqlite3.connect(tse_db_path)
        tse_cursor = tse_conn.cursor()

        project_conn = sqlite3.connect(project_db_path)
        project_cursor = project_conn.cursor()

        # Get unique tickers
        project_cursor.execute("SELECT DISTINCT ticker FROM tse_reference LIMIT 100")
        tickers = [row[0] for row in project_cursor.fetchall()]

        loaded_count = 0

        for ticker in tickers:
            tse_cursor.execute("""
                SELECT date, adj_close, adj_volume
                FROM price_data
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 100
            """, (ticker,))

            records = tse_cursor.fetchall()

            for date, close, volume in records:
                try:
                    change_pct = (volume / max(close, 1) * 100) if volume else 0
                    trend_score = 0.5 + (change_pct / 100) * 0.4
                    momentum_score = 0.5 + (change_pct / 50) * 0.3
                    combined_score = (trend_score + momentum_score) / 2

                    signal = 'BULLISH' if combined_score > 0.55 else 'BEARISH' if combined_score < 0.45 else 'NEUTRAL'

                    project_cursor.execute("""
                        INSERT OR IGNORE INTO historical_scores (
                            ticker, analysis_date, timeframe,
                            trend_score, momentum_score, combined_score,
                            trend_signal, momentum_signal, combined_signal,
                            price_at_analysis, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ticker, date, '1d',
                        trend_score, momentum_score, combined_score,
                        signal, signal, signal,
                        float(close or 0), datetime.now().isoformat()
                    ))
                    loaded_count += 1
                except Exception:
                    pass

        project_conn.commit()
        print(f"✓ Loaded {loaded_count} historical scores records")

        tse_cursor.close()
        tse_conn.close()
        project_cursor.close()
        project_conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")


def populate_market_data_cache():
    """market_data_cache را از TSE قیمت‌های واقعی با پر کن"""
    print("\n📦 Populating market_data_cache...")

    tse_db_path = r"E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db"
    project_db_path = "data/gravity_tech.db"

    try:
        tse_conn = sqlite3.connect(tse_db_path)
        tse_cursor = tse_conn.cursor()

        project_conn = sqlite3.connect(project_db_path)
        project_cursor = project_conn.cursor()

        # Get unique tickers from tse_reference
        project_cursor.execute("SELECT DISTINCT ticker FROM tse_reference LIMIT 100")
        tickers = [row[0] for row in project_cursor.fetchall()]

        loaded_count = 0

        for ticker in tickers:
            # Get price data for this ticker
            tse_cursor.execute("""
                SELECT date, adj_open, adj_high, adj_low, adj_close, adj_volume
                FROM price_data
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 500
            """, (ticker,))

            records = tse_cursor.fetchall()

            for date, open_p, high, low, close, volume in records:
                try:
                    project_cursor.execute("""
                        INSERT OR IGNORE INTO market_data_cache (
                            ticker, timeframe, cache_date,
                            open, high, low, close, volume, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ticker, '1d', date,
                        float(open_p or 0), float(high or 0), float(low or 0),
                        float(close or 0), float(volume or 0),
                        datetime.now().isoformat()
                    ))
                    loaded_count += 1
                except Exception:
                    pass

        project_conn.commit()
        print(f"✓ Loaded {loaded_count} market data records")

        tse_cursor.close()
        tse_conn.close()
        project_cursor.close()
        project_conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")


def populate_tool_performance_history():
    """tool_performance_history را با نمونه داده پر کن"""
    print("\n🔧 Populating tool_performance_history...")

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

    # Get tickers from tse_reference instead
    cursor.execute("SELECT DISTINCT ticker FROM tse_reference LIMIT 50")
    tickers = [row[0] for row in cursor.fetchall()]

    loaded_count = 0

    for tool_name, tool_category in tools:
        for ticker in tickers:
            for i in range(20):  # 20 predictions per tool per ticker
                try:
                    prediction_ts = (datetime.now() - timedelta(days=i)).isoformat()
                    result_ts = (datetime.now() - timedelta(days=i-1)).isoformat()

                    pred_value = 0.5 + (i % 10) * 0.05
                    actual_change = (i % 3) * 0.1 - 0.1
                    success = 1 if abs(actual_change) > 0.05 else 0
                    accuracy = 0.6 + (i % 5) * 0.08

                    cursor.execute("""
                        INSERT OR IGNORE INTO tool_performance_history (
                            tool_name, tool_category, ticker, timeframe,
                            market_regime, volatility_level, trend_strength,
                            prediction_type, prediction_value, confidence_score,
                            actual_result, actual_price_change, success, accuracy,
                            prediction_timestamp, result_timestamp, evaluation_period_hours,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        tool_name, tool_category, ticker, '1d',
                        'uptrend' if i % 2 == 0 else 'downtrend',
                        0.5, 0.6,
                        'bullish' if pred_value > 0.5 else 'bearish',
                        pred_value, 0.7 + (i % 3) * 0.1,
                        'correct' if success else 'incorrect',
                        actual_change, success, accuracy,
                        prediction_ts, result_ts, 24,
                        datetime.now().isoformat(), datetime.now().isoformat()
                    ))
                    loaded_count += 1
                except Exception:
                    pass

    conn.commit()
    print(f"✓ Loaded {loaded_count} tool performance records")

    cursor.close()
    conn.close()


def populate_tool_performance_stats():
    """tool_performance_stats را محاسبه و پر کن"""
    print("\n📊 Populating tool_performance_stats...")

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

    loaded_count = 0

    for tool_name, tool_category in tools:
        # Get stats from history
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as correct,
                AVG(accuracy) as avg_accuracy,
                AVG(confidence_score) as avg_confidence,
                AVG(actual_price_change) as avg_change,
                SUM(CASE WHEN prediction_value > 0.5 THEN 1 ELSE 0 END) as bullish_preds,
                SUM(CASE WHEN prediction_value < 0.5 THEN 1 ELSE 0 END) as bearish_preds,
                SUM(CASE WHEN prediction_value >= 0.45 AND prediction_value <= 0.55 THEN 1 ELSE 0 END) as neutral_preds
            FROM tool_performance_history
            WHERE tool_name = ?
        """, (tool_name,))

        stats = cursor.fetchone()
        if stats[0] == 0:
            continue

        total, correct, avg_acc, avg_conf, avg_change, bullish, bearish, neutral = stats

        accuracy = (correct / total * 100) if total > 0 else 0
        bullish_sr = (sum([1 for _ in range(int(bullish))] if bullish else []) / max(bullish, 1) * 100) if bullish else 0
        bearish_sr = (sum([1 for _ in range(int(bearish))] if bearish else []) / max(bearish, 1) * 100) if bearish else 0

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO tool_performance_stats (
                    tool_name, tool_category, market_regime, timeframe,
                    period_start, period_end, total_predictions, correct_predictions,
                    accuracy, avg_confidence, avg_actual_change,
                    bullish_predictions, bearish_predictions, neutral_predictions,
                    bullish_success_rate, bearish_success_rate, neutral_success_rate,
                    best_accuracy, worst_accuracy, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tool_name, tool_category, 'mixed', '1d',
                (datetime.now() - timedelta(days=30)).isoformat(),
                datetime.now().isoformat(),
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
                65.0,
                95.0,
                50.0,
                datetime.now().isoformat()
            ))
            loaded_count += 1
        except Exception:
            pass

    conn.commit()
    print(f"✓ Loaded {loaded_count} tool stats records")

    cursor.close()
    conn.close()


def populate_ml_weights_history():
    """ml_weights_history را پر کن"""
    print("\n🧠 Populating ml_weights_history...")

    project_db_path = "data/gravity_tech.db"
    conn = sqlite3.connect(project_db_path)
    cursor = conn.cursor()

    models = [
        'trend_model_v1',
        'momentum_model_v1',
        'combined_model_v1',
    ]

    loaded_count = 0

    for model_name in models:
        for version_idx in range(3):
            try:
                weights = {
                    'trend_weight': 0.3 + version_idx * 0.05,
                    'momentum_weight': 0.4 + version_idx * 0.05,
                    'volume_weight': 0.2 + version_idx * 0.05,
                    'volatility_weight': 0.1,
                }

                cursor.execute("""
                    INSERT INTO ml_weights_history (
                        model_name, model_version, market_regime, timeframe,
                        weights, training_accuracy, validation_accuracy, r2_score, mae,
                        training_samples, training_date, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name,
                    f"v{version_idx + 1}",
                    'mixed',
                    '1d',
                    json.dumps(weights),
                    0.75 + version_idx * 0.05,
                    0.72 + version_idx * 0.05,
                    0.68 + version_idx * 0.05,
                    0.05 - version_idx * 0.01,
                    5000 + version_idx * 1000,
                    (datetime.now() - timedelta(days=7 - version_idx)).isoformat(),
                    datetime.now().isoformat()
                ))
                loaded_count += 1
            except Exception:
                pass

    conn.commit()
    print(f"✓ Loaded {loaded_count} ML weight records")

    cursor.close()
    conn.close()


def populate_tool_recommendations_log():
    """tool_recommendations_log را پر کن"""
    print("\n💡 Populating tool_recommendations_log...")

    project_db_path = "data/gravity_tech.db"
    conn = sqlite3.connect(project_db_path)
    cursor = conn.cursor()

    # Get tickers
    cursor.execute("SELECT DISTINCT ticker FROM tse_reference LIMIT 30")
    tickers = [row[0] for row in cursor.fetchall()]

    loaded_count = 0

    for ticker_idx, ticker in enumerate(tickers):
        for rec_idx in range(5):  # 5 recommendations per ticker
            try:
                request_id = f"REQ_{ticker}_{ticker_idx}_{rec_idx}"
                recommended_tools = json.dumps([
                    'trend_analyzer',
                    'momentum_indicator',
                    'support_resistance'
                ])

                cursor.execute("""
                    INSERT INTO tool_recommendations_log (
                        request_id, user_id, ticker, timeframe, analysis_goal,
                        trading_style, market_regime, volatility_level, trend_strength,
                        recommended_tools, ml_weights, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    request_id,
                    f"user_{ticker_idx}",
                    ticker,
                    '1d',
                    'trend_following',
                    'swing_trading',
                    'uptrend' if rec_idx % 2 == 0 else 'downtrend',
                    0.4 + rec_idx * 0.1,
                    0.6 + rec_idx * 0.05,
                    recommended_tools,
                    json.dumps({'trend': 0.5, 'momentum': 0.5}),
                    (datetime.now() - timedelta(days=rec_idx)).isoformat()
                ))
                loaded_count += 1
            except Exception:
                pass

    conn.commit()
    print(f"✓ Loaded {loaded_count} recommendation records")

    cursor.close()
    conn.close()


def populate_pattern_detection_results():
    """pattern_detection_results را پر کن"""
    print("\n🎯 Populating pattern_detection_results...")

    project_db_path = "data/gravity_tech.db"
    conn = sqlite3.connect(project_db_path)
    cursor = conn.cursor()

    # Get tickers from tse_reference
    cursor.execute("SELECT DISTINCT ticker FROM tse_reference LIMIT 40")
    tickers = [row[0] for row in cursor.fetchall()]

    patterns = [
        ('support', 'Support Level'),
        ('resistance', 'Resistance Level'),
        ('head_shoulders', 'Head and Shoulders'),
        ('triangle', 'Triangle'),
        ('double_bottom', 'Double Bottom'),
        ('double_top', 'Double Top'),
    ]

    loaded_count = 0

    for ticker in tickers:
        for pattern_type, pattern_name in patterns:
            for pat_idx in range(3):  # 3 patterns per type per ticker
                try:
                    detection_date = (datetime.now() - timedelta(days=pat_idx * 10)).isoformat()
                    start_date = (datetime.now() - timedelta(days=pat_idx * 10 + 20)).isoformat()
                    end_date = detection_date

                    cursor.execute("""
                        INSERT INTO pattern_detection_results (
                            ticker, timeframe, detection_date, pattern_type, pattern_name,
                            confidence, strength, start_date, end_date, start_price, end_price,
                            prediction, target_price, stop_loss, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ticker, '1d', detection_date, pattern_type, pattern_name,
                        0.65 + pat_idx * 0.1,
                        0.7 + pat_idx * 0.08,
                        start_date, end_date,
                        1000 + pat_idx * 50,
                        1000 + pat_idx * 100 + 50,
                        'bullish' if pat_idx % 2 == 0 else 'bearish',
                        1050 + pat_idx * 100,
                        950 + pat_idx * 50,
                        datetime.now().isoformat()
                    ))
                    loaded_count += 1
                except Exception:
                    pass

    conn.commit()
    print(f"✓ Loaded {loaded_count} pattern detection records")

    cursor.close()
    conn.close()


def main():
    """Main function"""
    print("=" * 70)
    print("🚀 Complete Database Population")
    print("=" * 70)

    try:
        populate_historical_scores()
        populate_market_data_cache()
        populate_tool_performance_history()
        populate_tool_performance_stats()
        populate_ml_weights_history()
        populate_tool_recommendations_log()
        populate_pattern_detection_results()

        # Show final stats
        print("\n" + "=" * 70)
        print("✅ Database population completed!")
        print("=" * 70)

        conn = sqlite3.connect("data/gravity_tech.db")
        cursor = conn.cursor()

        tables = [
            'tse_reference',
            'historical_scores',
            'tool_performance_history',
            'tool_performance_stats',
            'ml_weights_history',
            'tool_recommendations_log',
            'market_data_cache',
            'pattern_detection_results'
        ]

        print("\n📊 Final Database Statistics:")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  • {table}: {count} records")

        cursor.close()
        conn.close()

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
