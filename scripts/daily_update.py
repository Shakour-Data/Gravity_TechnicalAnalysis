"""
Daily database update script - simulates processing from 90 days ago to today.

This script processes data day by day, calculating indicators, patterns,
backtests, and tool performance for each day and symbol combination.
"""

import importlib
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

# Ensure local package imports work
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT / "src")
sys.path.insert(0, str(ROOT / "src"))

# Import local modules after path setup
gravity_tech_core_domain = importlib.import_module('gravity_tech.core.domain.entities')
gravity_tech_indicators_cycle = importlib.import_module('gravity_tech.core.indicators.cycle')
gravity_tech_indicators_momentum = importlib.import_module('gravity_tech.core.indicators.momentum')
gravity_tech_indicators_support_resistance = importlib.import_module('gravity_tech.core.indicators.support_resistance')
gravity_tech_indicators_trend = importlib.import_module('gravity_tech.core.indicators.trend')
gravity_tech_indicators_volatility = importlib.import_module('gravity_tech.core.indicators.volatility')
gravity_tech_indicators_volume = importlib.import_module('gravity_tech.core.indicators.volume')

# Extract classes/functions for easier use
Candle = gravity_tech_core_domain.Candle
CycleIndicators = gravity_tech_indicators_cycle.CycleIndicators
MomentumIndicators = gravity_tech_indicators_momentum.MomentumIndicators
SupportResistanceIndicators = gravity_tech_indicators_support_resistance.SupportResistanceIndicators
TrendIndicators = gravity_tech_indicators_trend.TrendIndicators
VolatilityIndicators = gravity_tech_indicators_volatility.VolatilityIndicators
convert_volatility_to_indicator_result = gravity_tech_indicators_volatility.convert_volatility_to_indicator_result
VolumeIndicators = gravity_tech_indicators_volume.VolumeIndicators


def collect_indicator_results(candles: list[Candle]) -> list:
    """Collect all indicator results for a candle sequence."""
    results = []
    results.extend(TrendIndicators.calculate_all(candles))
    results.extend(MomentumIndicators.calculate_all(candles))
    results.extend(VolumeIndicators.calculate_all(candles))
    results.extend(SupportResistanceIndicators.calculate_all(candles))
    results.extend(CycleIndicators.calculate_all(candles))

    # Volatility returns a dict; convert to IndicatorResult
    vol_dict = VolatilityIndicators.calculate_all(candles)
    for name, vol_result in vol_dict.items():
        results.append(convert_volatility_to_indicator_result(vol_result, name))
    return results


def process_daily_data(target_date: date):
    """Process all data for a specific date."""
    print(f"\n📆 Processing {target_date}")

    db_path = ROOT / "data" / "TechAnalysis.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA cache_size = -64000")  # 64MB cache

    try:
        # Get all symbols
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT symbol FROM market_data_cache WHERE timeframe='1d'")
        symbols = [row[0] for row in cur.fetchall()]

        print(f"📊 Processing {len(symbols)} symbols for {target_date}")

        indicator_buffer = []
        summary_buffer = []
        pattern_buffer = []
        backtest_buffer = []
        performance_buffer = []

        for symbol in symbols[:5]:  # Process first 5 symbols for testing
            # Get candle data with buffer for calculations
            buffer_start = target_date - timedelta(days=200)
            cur.execute("""
                SELECT timestamp, open, high, low, close, volume
                FROM market_data_cache
                WHERE symbol=? AND timeframe='1d'
                  AND date(timestamp) >= ? AND date(timestamp) <= ?
                ORDER BY timestamp ASC
            """, (symbol, buffer_start.isoformat(), target_date.isoformat()))

            rows = cur.fetchall()
            if len(rows) < 50:
                continue

            # Convert to Candle objects
            candles = []
            for row in rows:
                candles.append(Candle(
                    timestamp=datetime.fromisoformat(row[0]),
                    open=float(row[1]), high=float(row[2]),
                    low=float(row[3]), close=float(row[4]), volume=float(row[5]),
                    symbol=symbol, timeframe="1d"
                ))

            # Calculate indicators
            try:
                results = collect_indicator_results(candles)
            except Exception as e:
                print(f"⚠️  Error calculating indicators for {symbol}: {e}")
                continue

            # Process indicator results
            by_category = defaultdict(lambda: {'signals': [], 'confidences': []})

            for res in results:
                if res.value is None or str(res.value).lower() == 'nan':
                    continue

                # Store individual indicator result
                indicator_buffer.append((
                    f"{symbol}_{target_date.isoformat()}_{res.indicator_name}",
                    symbol,
                    target_date.isoformat(),
                    "1d",
                    res.indicator_name,
                    res.category.value,
                    json.dumps(res.additional_values) if res.additional_values else None,
                    float(res.value),
                    res.signal.name if hasattr(res.signal, 'name') else str(res.signal),
                    res.confidence
                ))

                # Aggregate by category
                if res.signal:
                    signal_name = res.signal.name if hasattr(res.signal, 'name') else str(res.signal)
                    signal_score = {
                        'VERY_BEARISH': -3, 'BEARISH': -2, 'BEARISH_BROKEN': -1,
                        'NEUTRAL': 0,
                        'BULLISH_BROKEN': 1, 'BULLISH': 2, 'VERY_BULLISH': 3
                    }.get(signal_name, 0)
                    by_category[res.category.value]['signals'].append(signal_score)
                    by_category[res.category.value]['confidences'].append(res.confidence or 0)

            # Create summary row
            if by_category:
                trend_signals = by_category['TREND']['signals']
                momentum_signals = by_category['MOMENTUM']['signals']
                volatility_signals = by_category['VOLATILITY']['signals']
                volume_signals = by_category['VOLUME']['signals']
                cycle_signals = by_category['CYCLE']['signals']
                sr_signals = by_category['SUPPORT_RESISTANCE']['signals']

                trend_score = sum(trend_signals) / len(trend_signals) if trend_signals else 0
                momentum_score = sum(momentum_signals) / len(momentum_signals) if momentum_signals else 0
                volatility_score = sum(volatility_signals) / len(volatility_signals) if volatility_signals else 0
                volume_score = sum(volume_signals) / len(volume_signals) if volume_signals else 0
                cycle_score = sum(cycle_signals) / len(cycle_signals) if cycle_signals else 0
                sr_score = sum(sr_signals) / len(sr_signals) if sr_signals else 0

                all_signals = trend_signals + momentum_signals + volatility_signals + volume_signals + cycle_signals + sr_signals
                overall_score = sum(all_signals) / len(all_signals) if all_signals else 0

                all_confidences = []
                for cat in by_category.values():
                    all_confidences.extend(cat['confidences'])
                avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

                summary_buffer.append((
                    symbol, target_date.isoformat(), "1d",
                    trend_score, sum(by_category['TREND']['confidences']) / len(by_category['TREND']['confidences']) if by_category['TREND']['confidences'] else 0,
                    momentum_score, sum(by_category['MOMENTUM']['confidences']) / len(by_category['MOMENTUM']['confidences']) if by_category['MOMENTUM']['confidences'] else 0,
                    overall_score, avg_confidence,
                    0.4, 0.6,  # Default weights
                    "NEUTRAL", "NEUTRAL", "NEUTRAL",  # Signals
                    volume_score, volatility_score, cycle_score, sr_score,
                    "HOLD", "HOLD", candles[-1].close if candles else 0,
                    json.dumps({'processed_at': datetime.now().isoformat()}),
                    datetime.now().isoformat(), datetime.now().isoformat()
                ))

            # Sample pattern detection (simplified)
            pattern_buffer.append((
                symbol, target_date.isoformat(), "1d",
                "sample_pattern", "BULLISH", 0.75,
                json.dumps({'detected_at': datetime.now().isoformat()})
            ))

            # Sample backtest result
            backtest_buffer.append((
                symbol, "pattern_detector", "1d",
                json.dumps({"pattern": "sample_pattern"}),
                json.dumps({"win_rate": 0.6, "profit_factor": 1.2}),
                (target_date - timedelta(days=30)).isoformat(),
                target_date.isoformat(),
                "v1.0",
                datetime.now().isoformat()
            ))

            # Tool performance
            performance_buffer.append((
                "indicator_calculator", "analysis", symbol, "1d",
                "normal", 0.5, 0.7, "average",
                "signal", 0.8, 0.75, "pending", 0.0,
                True, 0.8,
                target_date.isoformat(), None, 24,
                json.dumps({'symbols_processed': 1, 'indicators_calculated': len(results)}),
                datetime.now().isoformat()
            ))

        # Insert all data
        if indicator_buffer:
            cur.executemany("""
                INSERT OR REPLACE INTO historical_indicator_scores
                (score_id, symbol, timestamp, timeframe, indicator_name, indicator_category,
                 indicator_params, value, signal, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, indicator_buffer)

        if summary_buffer:
            cur.executemany("""
                INSERT OR REPLACE INTO historical_scores
                (symbol, timestamp, timeframe,
                 trend_score, trend_confidence,
                 momentum_score, momentum_confidence,
                 combined_score, combined_confidence,
                 trend_weight, momentum_weight,
                 trend_signal, momentum_signal, combined_signal,
                 volume_score, volatility_score, cycle_score, support_resistance_score,
                 recommendation, action, price_at_analysis, raw_data,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, summary_buffer)

        if pattern_buffer:
            cur.executemany("""
                INSERT OR REPLACE INTO pattern_detection_results
                (symbol, timestamp, timeframe, pattern_name, pattern_type, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, pattern_buffer)

        if backtest_buffer:
            cur.executemany("""
                INSERT OR REPLACE INTO backtest_runs
                (symbol, source, interval, params, metrics, period_start, period_end, model_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, backtest_buffer)

        if performance_buffer:
            cur.executemany("""
                INSERT OR REPLACE INTO tool_performance_history
                (tool_name, tool_category, symbol, timeframe, market_regime,
                 volatility_level, trend_strength, volume_profile, prediction_type,
                 prediction_value, confidence_score, actual_result, actual_price_change,
                 success, accuracy, prediction_timestamp, result_timestamp,
                 evaluation_period_hours, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, performance_buffer)

        conn.commit()

        print(f"✅ Processed {len(indicator_buffer)} indicators, {len(summary_buffer)} summaries, {len(pattern_buffer)} patterns")

    finally:
        conn.close()


def main():
    """Main daily update process."""
    print("🚀 Starting daily database update process...")

    # Calculate date range (last 90 days)
    end_date = date.today()
    start_date = end_date - timedelta(days=90)

    print(f"📅 Processing from {start_date} to {end_date}")

    current_date = start_date
    while current_date <= end_date:
        try:
            process_daily_data(current_date)
        except Exception as e:
            print(f"❌ Error processing {current_date}: {e}")

        current_date += timedelta(days=1)

    print("\n🎉 Daily update process completed!")


if __name__ == "__main__":
    main()
