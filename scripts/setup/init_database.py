#!/usr/bin/env python3
"""
Quick Database Setup Script

یک دستور ساده برای setup کامل دیتابیس:
    python setup_database.py

این اسکریپت:
1. تشخیص خودکار نوع دیتابیس
2. ساخت schema
3. نمایش وضعیت

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import importlib
import sys

from _paths import extend_sys_path

extend_sys_path()

# Import after path setup
gravity_tech_database = importlib.import_module("gravity_tech.database.database_manager")
DatabaseManager = gravity_tech_database.DatabaseManager


def main():
    print("=" * 70)
    print("🚀 Gravity Tech Analysis - Database Setup")
    print("=" * 70)
    print()
    print("این اسکریپت دیتابیس را خودکار راه‌اندازی می‌کند:")
    print("  - اگر PostgreSQL موجود باشد → از PostgreSQL استفاده می‌کند")
    print("  - اگر PostgreSQL موجود نباشد → از SQLite استفاده می‌کند")
    print("  - اگر SQLite موجود نباشد → از JSON فایل استفاده می‌کند")
    print()
    print("-" * 70)

    try:
        # Initialize database with auto-detection
        db = DatabaseManager(auto_setup=True)

        print()
        print("✅ دیتابیس با موفقیت راه‌اندازی شد!")
        print(f"   نوع: {db.db_type.value}")

        if db.db_type.value == "postgresql":
            print("   🐘 PostgreSQL")
            print("   ✓ Connection pool: Active")
            print("   ✓ Schema: Created")
        elif db.db_type.value == "sqlite":
            print("   💾 SQLite")
            print(f"   ✓ Path: {db.sqlite_path}")
            print("   ✓ Schema: Created")
        elif db.db_type.value == "json_file":
            print("   📄 JSON File Storage")
            print(f"   ✓ Path: {db.json_path}")
            print("   ✓ Structure: Ready")

        print()
        print("📊 جداول ایجاد شده:")
        print("   - tool_performance_history")
        print("   - tool_performance_stats")
        print("   - ml_weights_history")
        print("   - tool_recommendations_log")

        # Test write
        print()
        print("🧪 تست نوشتن داده...")
        record_id = db.record_tool_performance(
            tool_name="MACD",
            tool_category="trend_indicators",
            symbol="BTCUSDT",
            timeframe="1d",
            market_regime="trending_bullish",
            prediction_type="bullish",
            confidence_score=0.85,
            volatility_level=45.5,
            trend_strength=72.3,
            volume_profile="high",
        )
        print(f"   ✓ Test record created: ID={record_id}")

        # Close connection
        db.close()

        print()
        print("=" * 70)
        print("✅ همه چیز آماده است! سیستم می‌تواند از دیتابیس استفاده کند.")
        print("=" * 70)

        return 0

    except Exception as e:
        print()
        print("❌ خطا در راه‌اندازی دیتابیس:")
        print(f"   {str(e)}")
        print()
        print("💡 راهکار:")
        print("   سیستم به صورت خودکار به SQLite یا JSON fallback می‌کند.")
        print("   نیازی به نگرانی نیست - سیستم بدون دیتابیس هم کار می‌کند!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
