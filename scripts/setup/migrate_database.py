"""
Database Migration: Fix DateTime columns and Naming Inconsistencies
مهاجرت دیتابیس: اصلاح ستون‌های تاریخ/زمان و نام‌گذاری

Steps:
1. Enable FOREIGN KEY constraints
2. Convert TEXT dates to DATETIME
3. Standardize column naming (ticker/analysis_date everywhere)
4. Recreate tables with proper schema
"""

import sqlite3
from datetime import datetime
from pathlib import Path


def migrate_database():
    db_path = Path("data/gravity_tech.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n" + "="*80)
    print("🔄 DATABASE MIGRATION IN PROGRESS")
    print("="*80)

    try:
        # Step 1: Enable FOREIGN KEY constraints
        print("\n1️⃣ Enabling FOREIGN KEY constraints...")
        cursor.execute("PRAGMA foreign_keys = ON")
        print("   ✅ FOREIGN KEY constraints enabled")

        # Step 2: Fix historical_indicator_scores - rename columns and convert types
        print("\n2️⃣ Fixing historical_indicator_scores table...")
        print("   • Renaming 'symbol' → 'ticker'")
        print("   • Renaming 'timestamp' → 'analysis_date'")
        print("   • Converting date columns to DATETIME")

        # Create new table with correct schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_indicator_scores_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                score_id INTEGER,
                ticker TEXT NOT NULL,
                analysis_date DATETIME NOT NULL,
                timeframe TEXT NOT NULL,
                indicator_name TEXT NOT NULL,
                indicator_category TEXT,
                indicator_params TEXT,
                value REAL NOT NULL,
                signal TEXT,
                confidence REAL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (score_id) REFERENCES historical_scores(id) ON DELETE CASCADE
            )
        """)

        # Copy data from old table, converting dates
        cursor.execute("""
            INSERT INTO historical_indicator_scores_new
            (id, score_id, ticker, analysis_date, timeframe, indicator_name,
             indicator_category, indicator_params, value, signal, confidence, created_at)
            SELECT
                id, score_id, symbol, datetime(timestamp), timeframe, indicator_name,
                indicator_category, indicator_params, value, signal, confidence, created_at
            FROM historical_indicator_scores
        """)

        # Get count
        cursor.execute("SELECT COUNT(*) FROM historical_indicator_scores_new")
        count = cursor.fetchone()[0]
        print(f"   ✅ Migrated {count:,} records")

        # Drop old table and rename
        cursor.execute("DROP TABLE historical_indicator_scores")
        cursor.execute("ALTER TABLE historical_indicator_scores_new RENAME TO historical_indicator_scores")
        print("   ✅ Old table dropped, new table activated")

        # Recreate indexes
        print("   • Recreating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicator_ticker_date ON historical_indicator_scores(ticker, analysis_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicator_name ON historical_indicator_scores(indicator_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicator_category ON historical_indicator_scores(indicator_category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicator_score_id ON historical_indicator_scores(score_id)")
        print("   ✅ Indexes recreated")

        # Step 3: Fix historical_scores - convert dates to DATETIME
        print("\n3️⃣ Fixing historical_scores table...")
        print("   • Converting 'analysis_date' to DATETIME")
        print("   • Converting 'updated_at' to DATETIME")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_scores_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                analysis_date DATETIME NOT NULL,
                timeframe TEXT NOT NULL,
                trend_score REAL DEFAULT 0.0,
                trend_confidence REAL DEFAULT 0.0,
                trend_signal TEXT DEFAULT 'NEUTRAL',
                momentum_score REAL DEFAULT 0.0,
                momentum_confidence REAL DEFAULT 0.0,
                momentum_signal TEXT DEFAULT 'NEUTRAL',
                combined_score REAL DEFAULT 0.0,
                combined_confidence REAL DEFAULT 0.0,
                combined_signal TEXT DEFAULT 'NEUTRAL',
                volume_score REAL DEFAULT 0.0,
                volatility_score REAL DEFAULT 0.0,
                cycle_score REAL DEFAULT 0.0,
                support_resistance_score REAL DEFAULT 0.0,
                trend_weight REAL DEFAULT 0.5,
                momentum_weight REAL DEFAULT 0.5,
                recommendation TEXT,
                action TEXT,
                price_at_analysis REAL,
                raw_data TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            INSERT INTO historical_scores_new
            SELECT
                id, ticker, datetime(analysis_date), timeframe,
                trend_score, trend_confidence, trend_signal,
                momentum_score, momentum_confidence, momentum_signal,
                combined_score, combined_confidence, combined_signal,
                volume_score, volatility_score, cycle_score, support_resistance_score,
                trend_weight, momentum_weight,
                recommendation, action, price_at_analysis, raw_data,
                created_at, datetime(updated_at)
            FROM historical_scores
        """)

        cursor.execute("SELECT COUNT(*) FROM historical_scores_new")
        count = cursor.fetchone()[0]
        print(f"   ✅ Migrated {count:,} records")

        cursor.execute("DROP TABLE historical_scores")
        cursor.execute("ALTER TABLE historical_scores_new RENAME TO historical_scores")
        print("   ✅ Old table dropped, new table activated")

        # Recreate indexes
        print("   • Recreating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_scores_ticker ON historical_scores(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_scores_date ON historical_scores(analysis_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_scores_combined ON historical_scores(combined_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_historical_scores_ticker_date ON historical_scores(ticker, analysis_date)")
        print("   ✅ Indexes recreated")

        # Step 4: Fix tool_performance_history
        print("\n4️⃣ Fixing tool_performance_history table...")
        print("   • Converting date/time columns to DATETIME")

        # Check if table exists and has data
        cursor.execute("SELECT COUNT(*) FROM tool_performance_history")
        count = cursor.fetchone()[0]

        if count > 0:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_performance_history_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    tool_category TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    market_regime TEXT NOT NULL,
                    volatility_level REAL,
                    trend_strength REAL,
                    volume_profile TEXT,
                    prediction_type TEXT NOT NULL,
                    prediction_value REAL,
                    confidence_score REAL,
                    actual_result TEXT,
                    actual_price_change REAL,
                    success INTEGER,
                    accuracy REAL,
                    prediction_timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    result_timestamp DATETIME,
                    evaluation_period_hours INTEGER,
                    metadata TEXT,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                INSERT INTO tool_performance_history_new
                SELECT
                    id, tool_name, tool_category, symbol, timeframe, market_regime,
                    volatility_level, trend_strength, volume_profile,
                    prediction_type, prediction_value, confidence_score,
                    actual_result, actual_price_change, success, accuracy,
                    datetime(prediction_timestamp), datetime(result_timestamp),
                    evaluation_period_hours, metadata, created_at, datetime(updated_at)
                FROM tool_performance_history
            """)

            cursor.execute("SELECT COUNT(*) FROM tool_performance_history_new")
            new_count = cursor.fetchone()[0]
            print(f"   ✅ Migrated {new_count:,} records")

            cursor.execute("DROP TABLE tool_performance_history")
            cursor.execute("ALTER TABLE tool_performance_history_new RENAME TO tool_performance_history")
            print("   ✅ Old table dropped, new table activated")
        else:
            print("   ⓘ Table is empty, skipping migration")

        # Commit all changes
        conn.commit()

        print("\n" + "="*80)
        print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nChanges made:")
        print("  1. Enabled FOREIGN KEY constraints")
        print("  2. Converted TEXT dates to DATETIME in all tables")
        print("  3. Standardized column naming:")
        print("     • 'symbol' → 'ticker' in historical_indicator_scores")
        print("     • 'timestamp' → 'analysis_date' in historical_indicator_scores")
        print("     • 'symbol' → 'ticker' in tool_performance_history")
        print("  4. Recreated all indexes")
        print("\n" + "="*80 + "\n")

    except Exception as e:
        conn.rollback()
        print(f"\n❌ MIGRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        cursor.close()
        conn.close()

    return True

if __name__ == "__main__":
    success = migrate_database()
    exit(0 if success else 1)
