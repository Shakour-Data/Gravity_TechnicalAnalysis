#!/usr/bin/env python
"""Complete ML Weights Population - تکمیل کامل جدول وزن‌های یادگیری ماشین"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def populate_complete_ml_weights():
    """تکمیل کامل جدول ml_weights_history"""

    print("=" * 70)
    print("🧠 تکمیل کامل جدول ML Weights History")
    # Clear existing data
    print("\n🗑️  پاک کردن داده‌های قدیمی...")
    conn = sqlite3.connect("data/gravity_tech.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ml_weights_history")
    conn.commit()

    # Get all tickers from tse_reference
    cursor.execute("SELECT DISTINCT ticker FROM tse_reference")
    tickers = [row[0] for row in cursor.fetchall()]

    # Prepare: Get last 50 trading dates (to limit data volume while covering all tickers)
    cursor.execute("SELECT DISTINCT analysis_date FROM historical_scores WHERE timeframe = '1d' ORDER BY analysis_date DESC LIMIT 50")
    recent_dates = sorted([row[0] for row in cursor.fetchall()])  # Sort ascending for chronological order

    # Prepare: For each ticker, get only the recent 100 dates
    ticker_dates = {}
    for ticker in tickers:
        cursor.execute("SELECT analysis_date FROM historical_scores WHERE ticker = ? AND timeframe = '1d' AND analysis_date IN ({}) ORDER BY analysis_date".format(','.join(['?']*len(recent_dates))), [ticker] + recent_dates)
        dates = [row[0] for row in cursor.fetchall()]
        if dates:
            ticker_dates[ticker] = dates

    models = [
        'trend_analyzer',
        'momentum_indicator',
        'volume_analyzer',
        'volatility_indicator',
        'support_resistance_detector',
        'cycle_detector',
        'combined_technical_model',
        'ml_ensemble_model',
        'adaptive_model',
    ]

    market_regimes = [
        'bullish',
        'bearish',
        'sideways',
        'high_volatility',
        'low_volatility',
        'mixed',
    ]

    # حذف متغیر timeframes چون فقط 1d استفاده می‌شود

    versions_per_model = 5  # 5 versions per model

    print("\n📊 تولید داده برای:")
    print(f"  • {len(models)} مدل")
    print(f"  • {len(market_regimes)} رژیم بازار")
    print("  • فقط بازه زمانی 1d")
    print(f"  • {versions_per_model} نسخه برای هر مدل")
    print(f"  • {len(ticker_dates)} نماد با تاریخ واقعی")
    print("  • فقط 50 روز تجاری قبل")
    total_dates = sum(len(dates) for dates in ticker_dates.values())
    expected_records = len(models) * len(market_regimes) * versions_per_model * total_dates
    print(f"\n📈 تعداد رکوردهای مورد انتظار: {expected_records:,}")

    loaded_count = 0
    batch_size = 1000
    batch_data = []

    print("\n📥 در حال تولید داده...")

    for ticker, dates in ticker_dates.items():
        for analysis_date in dates:
            for _model_idx, model_name in enumerate(models, 1):
                for regime in market_regimes:
                    for version_idx in range(1, versions_per_model + 1):
                        # فقط تایم‌فریم 1d
                        timeframe = '1d'
                        # Generate realistic weights based on model type
                        if 'trend' in model_name.lower():
                            weights = {
                                'trend_weight': 0.6 + version_idx * 0.02,
                                'momentum_weight': 0.2,
                                'volume_weight': 0.1,
                                'volatility_weight': 0.1,
                            }
                        elif 'momentum' in model_name.lower():
                            weights = {
                                'trend_weight': 0.2,
                                'momentum_weight': 0.6 + version_idx * 0.02,
                                'volume_weight': 0.1,
                                'volatility_weight': 0.1,
                            }
                        elif 'volume' in model_name.lower():
                            weights = {
                                'trend_weight': 0.2,
                                'momentum_weight': 0.2,
                                'volume_weight': 0.5 + version_idx * 0.02,
                                'volatility_weight': 0.1,
                            }
                        elif 'volatility' in model_name.lower():
                            weights = {
                                'trend_weight': 0.2,
                                'momentum_weight': 0.2,
                                'volume_weight': 0.1,
                                'volatility_weight': 0.5 + version_idx * 0.02,
                            }
                        elif 'support_resistance' in model_name.lower():
                            weights = {
                                'support_weight': 0.4 + version_idx * 0.02,
                                'resistance_weight': 0.4,
                                'volume_weight': 0.1,
                                'trend_weight': 0.1,
                            }
                        elif 'cycle' in model_name.lower():
                            weights = {
                                'cycle_strength': 0.5 + version_idx * 0.02,
                                'phase_accuracy': 0.3,
                                'amplitude': 0.1,
                                'frequency': 0.1,
                            }
                        else:  # combined/ensemble/adaptive models
                            weights = {
                                'trend_weight': 0.25 + version_idx * 0.01,
                                'momentum_weight': 0.25 + version_idx * 0.01,
                                'volume_weight': 0.25,
                                'volatility_weight': 0.25,
                            }

                        # Calculate performance metrics based on regime and version
                        base_accuracy = 0.65
                        if regime == 'bullish':
                            base_accuracy += 0.05
                        elif regime == 'bearish':
                            base_accuracy += 0.03
                        elif regime == 'high_volatility':
                            base_accuracy -= 0.05

                        training_accuracy = min(0.95, base_accuracy + version_idx * 0.015)
                        validation_accuracy = training_accuracy - 0.03
                        r2_score = validation_accuracy - 0.05
                        mae = max(0.01, 0.08 - version_idx * 0.005)

                        training_samples = 5000 + version_idx * 500
                        training_date = analysis_date  # تاریخ واقعی هر روز معاملاتی

                        batch_data.append((
                            ticker,
                            model_name,
                            f"v{version_idx}",
                            regime,
                            timeframe,
                            json.dumps(weights),
                            training_accuracy,
                            validation_accuracy,
                            r2_score,
                            mae,
                            training_samples,
                            training_date,
                            datetime.now().isoformat()
                        ))

                        loaded_count += 1

                        # Batch insert
                        if len(batch_data) >= batch_size:
                            cursor.executemany("""
                                INSERT INTO ml_weights_history (
                                    ticker, model_name, model_version, market_regime, timeframe,
                                    weights, training_accuracy, validation_accuracy, r2_score, mae,
                                    training_samples, training_date, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, batch_data)
                            conn.commit()
                            batch_data = []
                            if loaded_count % 100000 == 0:
                                print(f"  Progress: {loaded_count:,}/{expected_records:,}")

    # Insert remaining records
    if batch_data:
        cursor.executemany("""
            INSERT INTO ml_weights_history (
                ticker, model_name, model_version, market_regime, timeframe,
                weights, training_accuracy, validation_accuracy, r2_score, mae,
                training_samples, training_date, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch_data)
        conn.commit()

    print(f"\n✓ بارگذاری شده: {loaded_count:,} رکورد")

    # Verify final count
    cursor.execute("SELECT COUNT(*) FROM ml_weights_history")
    final_count = cursor.fetchone()[0]

    print("\n" + "=" * 70)
    print("✅ تکمیل شد!")
    print("=" * 70)
    print(f"\n📊 نتیجه نهایی: {final_count:,} رکورد در ml_weights_history")

    cursor.close()
    conn.close()
    print("=" * 70)
    print(f"\n📊 نتیجه نهایی: {final_count:,} رکورد در ml_weights_history")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    try:
        populate_complete_ml_weights()
    except Exception as e:
        print(f"\n❌ خطا: {e}")
        sys.exit(1)
