"""
Database Optimization Script

این اسکریپت برای بهینه‌سازی دیتابیس historical scores استفاده می‌شود.
شامل ایجاد ایندکس‌ها و بهینه‌سازی queryها.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """بهینه‌ساز دیتابیس برای historical scores"""

    def __init__(self, db_path: str = "data/gravity_tech.db"):
        """
        Args:
            db_path: مسیر فایل دیتابیس
        """
        self.db_path = db_path
        self._ensure_db_directory()

    def _ensure_db_directory(self):
        """ایجاد دایرکتوری دیتابیس اگر وجود ندارد"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def create_indexes(self):
        """ایجاد ایندکس‌های بهینه برای queryهای historical"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            logger.info("🔧 Creating database indexes...")

            # ایندکس برای queryهای اصلی بر اساس symbol و timeframe
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_historical_scores_symbol_timeframe_timestamp
                    ON historical_scores (symbol, timeframe, timestamp DESC)
                """)
            except sqlite3.OperationalError as e:
                logger.warning(f"⚠️ Could not create index idx_historical_scores_symbol_timeframe_timestamp: {e}")

            # ایندکس برای فیلتر بر اساس امتیاز
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_historical_scores_combined_score
                    ON historical_scores (combined_score)
                """)
            except sqlite3.OperationalError as e:
                logger.warning(f"⚠️ Could not create index idx_historical_scores_combined_score: {e}")

            # ایندکس برای فیلتر تاریخ
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_historical_scores_timestamp
                    ON historical_scores (timestamp DESC)
                """)
            except sqlite3.OperationalError as e:
                logger.warning(f"⚠️ Could not create index idx_historical_scores_timestamp: {e}")

            # ایندکس برای confidence
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_historical_scores_confidence
                    ON historical_scores (combined_confidence)
                """)
            except sqlite3.OperationalError as e:
                logger.warning(f"⚠️ Could not create index idx_historical_scores_confidence: {e}")

            # ایندکس‌های ترکیبی برای queryهای پیچیده
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_historical_scores_symbol_timeframe_score
                    ON historical_scores (symbol, timeframe, combined_score)
                """)
            except sqlite3.OperationalError as e:
                logger.warning(f"⚠️ Could not create index idx_historical_scores_symbol_timeframe_score: {e}")

            # ایندکس برای horizon scores (only if table exists)
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_historical_horizon_scores_score_id
                    ON historical_horizon_scores (score_id)
                """)
            except sqlite3.OperationalError as e:
                logger.warning(f"⚠️ Could not create index idx_historical_horizon_scores_score_id (table may not exist): {e}")

            # ایندکس برای indicator scores (only if table exists)
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_historical_indicator_scores_score_id
                    ON historical_indicator_scores (score_id)
                """)
            except sqlite3.OperationalError as e:
                logger.warning(f"⚠️ Could not create index idx_historical_indicator_scores_score_id (table may not exist): {e}")

            # ایندکس برای patterns (only if table exists)
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_historical_patterns_score_id
                    ON historical_patterns (score_id)
                """)
            except sqlite3.OperationalError as e:
                logger.warning(f"⚠️ Could not create index idx_historical_patterns_score_id (table may not exist): {e}")

            # ایندکس برای volume analysis (only if table exists)
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_historical_volume_analysis_score_id
                    ON historical_volume_analysis (score_id)
                """)
            except sqlite3.OperationalError as e:
                logger.warning(f"⚠️ Could not create index idx_historical_volume_analysis_score_id (table may not exist): {e}")

            # ایندکس برای price targets (only if table exists)
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_historical_price_targets_score_id
                    ON historical_price_targets (score_id)
                """)
            except sqlite3.OperationalError as e:
                logger.warning(f"⚠️ Could not create index idx_historical_price_targets_score_id (table may not exist): {e}")

            conn.commit()
            logger.info("✅ Database indexes created (with warnings for missing tables)")

        except Exception as e:
            logger.error(f"❌ Error creating indexes: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    def analyze_query_performance(self):
        """تحلیل عملکرد queryها"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            logger.info("📊 Analyzing query performance...")

            # فعال کردن SQLite statistics
            cursor.execute("PRAGMA analysis_limit = 1000;")
            cursor.execute("PRAGMA optimize;")

            # بررسی اندازه جدول
            cursor.execute("SELECT COUNT(*) FROM historical_scores")
            total_records = cursor.fetchone()[0]
            logger.info(f"📈 Total historical records: {total_records:,}")

            # بررسی اندازه ایندکس‌ها
            cursor.execute("""
                SELECT name, sql FROM sqlite_master
                WHERE type = 'index' AND name LIKE 'idx_historical%'
            """)

            indexes = cursor.fetchall()
            logger.info(f"📋 Found {len(indexes)} historical indexes:")
            for name, sql in indexes:
                logger.info(f"  - {name}")

            # پیشنهاد بهینه‌سازی‌ها
            if total_records > 10000:
                logger.info("💡 Recommendations for large dataset:")
                logger.info("  - Consider partitioning by date ranges")
                logger.info("  - Implement data archiving for old records")
                logger.info("  - Use summary tables for aggregations")

        finally:
            cursor.close()
            conn.close()

    def create_summary_tables(self):
        """ایجاد جداول خلاصه برای queryهای سریع‌تر"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            logger.info("📊 Creating summary tables...")

            # جدول خلاصه روزانه
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS historical_daily_summary (
                        symbol TEXT,
                        timeframe TEXT,
                        date DATE,
                        avg_score REAL,
                        min_score REAL,
                        max_score REAL,
                        avg_confidence REAL,
                        total_records INTEGER,
                        bullish_count INTEGER,
                        bearish_count INTEGER,
                        neutral_count INTEGER,
                        PRIMARY KEY (symbol, timeframe, date)
                    )
                """)
            except sqlite3.OperationalError as e:
                logger.warning(f"⚠️ Could not create historical_daily_summary table: {e}")

            # جدول خلاصه هفتگی
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS historical_weekly_summary (
                        symbol TEXT,
                        timeframe TEXT,
                        week_start DATE,
                        avg_score REAL,
                        trend_strength REAL,
                        consistency_score REAL,
                        total_records INTEGER,
                        PRIMARY KEY (symbol, timeframe, week_start)
                    )
                """)
            except sqlite3.OperationalError as e:
                logger.warning(f"⚠️ Could not create historical_weekly_summary table: {e}")

            # ایجاد ایندکس برای جداول خلاصه
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_daily_summary_symbol_timeframe
                    ON historical_daily_summary (symbol, timeframe, date DESC)
                """)
            except sqlite3.OperationalError as e:
                logger.warning(f"⚠️ Could not create index idx_daily_summary_symbol_timeframe: {e}")

            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_weekly_summary_symbol_timeframe
                    ON historical_weekly_summary (symbol, timeframe, week_start DESC)
                """)
            except sqlite3.OperationalError as e:
                logger.warning(f"⚠️ Could not create index idx_weekly_summary_symbol_timeframe: {e}")

            conn.commit()
            logger.info("✅ Summary tables created (with warnings for issues)")

        except Exception as e:
            logger.error(f"❌ Error creating summary tables: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    def populate_summary_tables(self):
        """پرکردن جداول خلاصه با داده‌های موجود"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            logger.info("🔄 Populating summary tables...")

            # پاک کردن داده‌های قدیمی
            cursor.execute("DELETE FROM historical_daily_summary")
            cursor.execute("DELETE FROM historical_weekly_summary")

            # پرکردن جدول روزانه
            cursor.execute("""
                INSERT INTO historical_daily_summary
                SELECT
                    symbol,
                    timeframe,
                    DATE(timestamp) as date,
                    AVG(combined_score) as avg_score,
                    MIN(combined_score) as min_score,
                    MAX(combined_score) as max_score,
                    AVG(combined_confidence) as avg_confidence,
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN combined_score > 0 THEN 1 END) as bullish_count,
                    COUNT(CASE WHEN combined_score < 0 THEN 1 END) as bearish_count,
                    COUNT(CASE WHEN combined_score = 0 THEN 1 END) as neutral_count
                FROM historical_scores
                GROUP BY symbol, timeframe, DATE(timestamp)
                ORDER BY symbol, timeframe, date
            """)

            # پرکردن جدول هفتگی
            cursor.execute("""
                INSERT INTO historical_weekly_summary
                SELECT
                    symbol,
                    timeframe,
                    DATE(timestamp, 'weekday 0', '-6 days') as week_start,
                    AVG(combined_score) as avg_score,
                    ABS(AVG(combined_score)) as trend_strength,
                    1.0 - (MAX(combined_score) - MIN(combined_score)) / 2.0 as consistency_score,
                    COUNT(*) as total_records
                FROM historical_scores
                GROUP BY symbol, timeframe, DATE(timestamp, 'weekday 0', '-6 days')
                ORDER BY symbol, timeframe, week_start
            """)

            conn.commit()
            logger.info("✅ Summary tables populated")

        except Exception as e:
            logger.error(f"❌ Error populating summary tables: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    def optimize_database(self):
        """اجرای کامل بهینه‌سازی دیتابیس"""
        logger.info("🚀 Starting database optimization...")

        try:
            self.create_indexes()
            self.analyze_query_performance()
            self.create_summary_tables()
            self.populate_summary_tables()

            logger.info("✅ Database optimization completed successfully")

        except Exception as e:
            logger.error(f"❌ Database optimization failed: {e}")
            raise


def main():
    """تابع اصلی"""
    optimizer = DatabaseOptimizer()

    try:
        optimizer.optimize_database()
        logger.info("🎉 All optimizations completed!")

    except Exception as e:
        logger.error(f"💥 Optimization failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())