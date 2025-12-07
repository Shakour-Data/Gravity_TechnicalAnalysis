"""
Create All Database Tables - ایجاد تمام جداول دیتابیس پروژه

این اسکریپت تمام 7 جدول اصلی را برای پروژه ایجاد می‌کند.

استفاده:
    python scripts/setup/create_all_tables.py

Author: Gravity Tech Team
Date: December 5, 2025
Version: 1.0.0
License: MIT
"""

import importlib
import sqlite3
import sys
import traceback
from pathlib import Path

# Add root to path
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_project_schema = importlib.import_module("database.project_schema")
ALL_TABLES_SQL = _project_schema.ALL_TABLES_SQL


def create_all_tables():
    """ایجاد تمام جداول دیتابیس پروژه"""

    print("=" * 70)
    print("🗄️  ایجاد تمام جداول دیتابیس Gravity Tech")
    print("=" * 70)

    db_path = "data/gravity_project.db"
    print(f"\n📍 مسیر دیتابیس: {db_path}")

    try:
        # Create database directory if needed
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect to SQLite
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("\n📋 در حال ایجاد جداول...")

        # Execute all tables SQL
        cursor.executescript(ALL_TABLES_SQL)
        conn.commit()

        print("✅ جداول با موفقیت ایجاد شدند")

        # Get table info
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = cursor.fetchall()

        print("\n" + "=" * 70)
        print("📊 جداول ایجاد شده")
        print("=" * 70)

        table_list = [
            ("1", "historical_scores", "نتایج تحلیل تکنیکال"),
            ("2", "tool_performance_history", "سابقه عملکرد ابزارها"),
            ("3", "tool_performance_stats", "آمار عملکرد ابزارها"),
            ("4", "ml_weights_history", "سابقه وزن‌های ML"),
            ("5", "tool_recommendations_log", "لاگ توصیه‌های ابزار"),
            ("6", "market_data_cache", "کش داده‌های بازار"),
            ("7", "pattern_detection_results", "نتایج تشخیص الگوها"),
        ]

        for num, table_name, description in table_list:
            status = "✓" if any(t[0] == table_name for t in tables) else "✗"
            print(f"{status} {num}. {table_name:<35} - {description}")

        print("\n" + "=" * 70)
        print(f"✅ دیتابیس پروژه با موفقیت ایجاد شد! ({len(tables)} جدول)")
        print("=" * 70)

        cursor.close()
        conn.close()

        return True

    except Exception as e:
        print(f"\n❌ خطا در ایجاد جداول: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = create_all_tables()
    sys.exit(0 if success else 1)

