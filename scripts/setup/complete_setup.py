#!/usr/bin/env python
"""Complete Database Setup - تنظیم کامل دیتابیس یکپارچه"""

import sqlite3
import sys
from pathlib import Path

# Add root to path
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def cleanup_old_databases():
    """تمام دیتابیس‌های قدیمی را حذف کنید"""
    print("\n🗑️  تمیز کردن دیتابیس‌های قدیمی...")

    old_dbs = [
        "data/gravity_project.db",
        "data/gravity_tech.db",
        "data/tool_performance.db",
        "data/tse_data.db"
    ]

    for db_path in old_dbs:
        path = Path(db_path)
        if path.exists():
            path.unlink()
            print(f"  ✓ Deleted: {db_path}")


def create_unified_schema():
    """طراحی یکپارچه دیتابیس"""

    statements = [
        """CREATE TABLE IF NOT EXISTS tse_reference (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT UNIQUE NOT NULL,
            company_name TEXT,
            sector_id INTEGER,
            sector_name TEXT,
            market_type TEXT,
            last_synced_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        );""",

        """CREATE INDEX IF NOT EXISTS idx_tse_reference_ticker ON tse_reference(ticker);""",

        """CREATE TABLE IF NOT EXISTS historical_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            analysis_date DATETIME NOT NULL,
            timeframe TEXT NOT NULL DEFAULT '1d',
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
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ticker) REFERENCES tse_reference(ticker),
            UNIQUE(ticker, analysis_date, timeframe)
        );""",

        """CREATE INDEX IF NOT EXISTS idx_historical_scores_ticker ON historical_scores(ticker);""",
        """CREATE INDEX IF NOT EXISTS idx_historical_scores_date ON historical_scores(analysis_date);""",
        """CREATE INDEX IF NOT EXISTS idx_historical_scores_combined ON historical_scores(combined_score);""",
        """CREATE INDEX IF NOT EXISTS idx_historical_scores_ticker_date ON historical_scores(ticker, analysis_date);""",

        """CREATE TABLE IF NOT EXISTS tool_performance_history (
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
            result_timestamp TEXT,
            evaluation_period_hours INTEGER,
            metadata TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ticker) REFERENCES tse_reference(ticker)
        );""",

        """CREATE INDEX IF NOT EXISTS idx_tool_perf_tool ON tool_performance_history(tool_name);""",
        """CREATE INDEX IF NOT EXISTS idx_tool_perf_ticker ON tool_performance_history(ticker);""",
        """CREATE INDEX IF NOT EXISTS idx_tool_perf_timestamp ON tool_performance_history(prediction_timestamp);""",

        """CREATE TABLE IF NOT EXISTS tool_performance_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL,
            tool_category TEXT NOT NULL,
            market_regime TEXT,
            timeframe TEXT,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            total_predictions INTEGER NOT NULL DEFAULT 0,
            correct_predictions INTEGER NOT NULL DEFAULT 0,
            accuracy REAL,
            avg_confidence REAL,
            avg_actual_change REAL,
            bullish_predictions INTEGER DEFAULT 0,
            bearish_predictions INTEGER DEFAULT 0,
            neutral_predictions INTEGER DEFAULT 0,
            bullish_success_rate REAL,
            bearish_success_rate REAL,
            neutral_success_rate REAL,
            best_accuracy REAL,
            worst_accuracy REAL,
            best_ticker TEXT,
            worst_ticker TEXT,
            last_updated DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(tool_name, market_regime, timeframe, period_start, period_end)
        );""",

        """CREATE INDEX IF NOT EXISTS idx_tool_stats_tool ON tool_performance_stats(tool_name);""",
        """CREATE INDEX IF NOT EXISTS idx_tool_stats_accuracy ON tool_performance_stats(accuracy);""",

        """CREATE TABLE IF NOT EXISTS ml_weights_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            market_regime TEXT,
            timeframe TEXT,
            weights TEXT NOT NULL,
            training_accuracy REAL,
            validation_accuracy REAL,
            r2_score REAL,
            mae REAL,
            training_samples INTEGER,
            training_date TEXT NOT NULL,
            metadata TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        );""",

        """CREATE INDEX IF NOT EXISTS idx_ml_weights_ticker ON ml_weights_history(ticker);""",
        """CREATE INDEX IF NOT EXISTS idx_ml_weights_model ON ml_weights_history(model_name, model_version);""",
        """CREATE INDEX IF NOT EXISTS idx_ml_weights_date ON ml_weights_history(training_date);""",

        """CREATE TABLE IF NOT EXISTS tool_recommendations_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            user_id TEXT,
            ticker TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            analysis_goal TEXT,
            trading_style TEXT,
            market_regime TEXT NOT NULL,
            volatility_level REAL,
            trend_strength REAL,
            recommended_tools TEXT NOT NULL,
            ml_weights TEXT,
            user_feedback TEXT,
            tools_actually_used TEXT,
            trade_result TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            feedback_at TEXT,
            FOREIGN KEY (ticker) REFERENCES tse_reference(ticker)
        );""",

        """CREATE INDEX IF NOT EXISTS idx_recommendations_request ON tool_recommendations_log(request_id);""",
        """CREATE INDEX IF NOT EXISTS idx_recommendations_ticker ON tool_recommendations_log(ticker);""",
        """CREATE INDEX IF NOT EXISTS idx_recommendations_created ON tool_recommendations_log(created_at);""",

        """CREATE TABLE IF NOT EXISTS market_data_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            cache_date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ticker) REFERENCES tse_reference(ticker),
            UNIQUE(ticker, timeframe, cache_date)
        );""",

        """CREATE INDEX IF NOT EXISTS idx_market_data_ticker ON market_data_cache(ticker);""",
        """CREATE INDEX IF NOT EXISTS idx_market_data_date ON market_data_cache(cache_date);""",

        """CREATE TABLE IF NOT EXISTS pattern_detection_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            detection_date TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            pattern_name TEXT NOT NULL,
            confidence REAL,
            strength REAL,
            start_date TEXT,
            end_date TEXT,
            start_price REAL,
            end_price REAL,
            prediction TEXT,
            target_price REAL,
            stop_loss REAL,
            metadata TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ticker) REFERENCES tse_reference(ticker)
        );""",

        """CREATE INDEX IF NOT EXISTS idx_pattern_ticker ON pattern_detection_results(ticker);""",
        """CREATE INDEX IF NOT EXISTS idx_pattern_type ON pattern_detection_results(pattern_type);""",
        """CREATE INDEX IF NOT EXISTS idx_pattern_date ON pattern_detection_results(detection_date);""",
    ]

    return statements


def complete_setup():
    """تنظیم کامل دیتابیس"""

    print("=" * 70)
    print("🔧 تنظیم کامل دیتابیس یکپارچه Gravity Tech")
    print("=" * 70)

    try:
        # Step 1: Cleanup
        cleanup_old_databases()

        # Step 2: Create database
        print("\n📦 ایجاد دیتابیس جدید...")
        db_path = "data/gravity_tech.db"
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print(f"✓ Database created: {db_path}")

        # Step 3: Create schema
        print("\n📋 ایجاد جداول...")
        statements = create_unified_schema()
        for stmt in statements:
            cursor.execute(stmt)
        conn.commit()

        print("✓ Tables created successfully")

        # Step 4: Verify
        print("\n📊 تایید ساختار...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()

        print(f"✓ Total tables: {len(tables)}")
        for table in tables:
            print(f"  • {table[0]}")

        # Step 5: Load TSE reference data
        print("\n🔗 بارگذاری داده‌های مرجع TSE...")
        tse_db_path = r"E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db"

        if Path(tse_db_path).exists():
            tse_conn = sqlite3.connect(tse_db_path)
            tse_cursor = tse_conn.cursor()

            tse_cursor.execute("""
                SELECT ticker, name, sector_id
                FROM companies
                LIMIT 1000
            """)

            companies = tse_cursor.fetchall()

            for ticker, name, sector_id in companies:
                cursor.execute("""
                    INSERT OR IGNORE INTO tse_reference (ticker, company_name, sector_id)
                    VALUES (?, ?, ?)
                """, (ticker, name, sector_id))

            conn.commit()
            print(f"✓ Loaded {len(companies)} ticker references from TSE")
            tse_conn.close()
        else:
            print(f"⚠️ TSE database not found: {tse_db_path}")

        # Step 6: Summary
        print("\n" + "=" * 70)
        print("✅ دیتابیس با موفقیت تنظیم شد!")
        print("=" * 70)
        print(f"\n📍 Database location: {db_path}")
        print(f"📍 Tables created: {len(tables)}")
        print("📍 Ready for data loading!")

        cursor.close()
        conn.close()

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = complete_setup()
    sys.exit(0 if success else 1)
