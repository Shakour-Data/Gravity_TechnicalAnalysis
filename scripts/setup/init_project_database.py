"""
Initialize Project Database - دیتابیس اختصاصی پروژه

این اسکریپت دیتابیس جدیدی برای پروژه Gravity Tech ایجاد می‌کند
که کاملاً از دیتابیس ورودی (TSE data) جدا است.

استفاده:
    python scripts/setup/init_project_database.py

Author: Gravity Tech Team
Date: December 5, 2025
Version: 1.0.0
License: MIT
"""

import sys
from pathlib import Path

try:
    from database import DatabaseManager, DatabaseType
except ModuleNotFoundError:
    _root = Path(__file__).parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from database import DatabaseManager, DatabaseType


def create_project_database():
    """
    دیتابیس اختصاصی پروژه را ایجاد کنید

    این دیتابیس برای ذخیره‌سازی موارد زیر استفاده می‌شود:
    - نتایج تحلیل تکنیکال (Historical Scores)
    - عملکرد ابزارهای تحلیل (Tool Performance)
    - وزن‌های ML (ML Weights History)
    - توصیه‌های ابزارها (Tool Recommendations)
    """

    print("=" * 70)
    print("🗄️  دیتابیس اختصاصی پروژه Gravity Tech")
    print("=" * 70)

    # Use SQLite for project database
    db_path = "data/gravity_project.db"

    print(f"\n📍 مسیر دیتابیس: {db_path}")
    print("📦 نوع دیتابیس: SQLite")

    try:
        # Initialize database manager
        db_manager = DatabaseManager(
            db_type=DatabaseType.SQLITE,
            sqlite_path=db_path,
            auto_setup=False
        )

        print("\n✅ اتصال به دیتابیس برقرار شد")

        # Setup database schema
        print("\n📋 در حال ایجاد جداول...")
        db_manager.setup_database()

        print("✅ جداول با موفقیت ایجاد شدند")

        # Get database info
        info = db_manager.get_database_info()

        print("\n" + "=" * 70)
        print("📊 اطلاعات دیتابیس")
        print("=" * 70)
        print(f"✓ نوع: {info.get('type', 'Unknown')}")
        print(f"✓ مسیر: {info.get('path', 'N/A')}")
        print(f"✓ تعداد جداول: {info.get('table_count', 0)}")

        if 'tables' in info:
            print("\n📋 جداول:")
            for table in info['tables']:
                print(f"   • {table}")

        print("\n" + "=" * 70)
        print("✅ دیتابیس پروژه با موفقیت ایجاد شد!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ خطا در ایجاد دیتابیس: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = create_project_database()
    sys.exit(0 if success else 1)
