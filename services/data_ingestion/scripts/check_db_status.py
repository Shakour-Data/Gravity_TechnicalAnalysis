"""
Complete database status check script
"""

import os
import sqlite3
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DB_FILE


def check_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    print("=" * 80)
    print("DATABASE STATUS REPORT")
    print("=" * 80)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    print("📊 TABLES IN DATABASE:")
    print("-" * 80)
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {table:30s} {count:>10,} records")
    print()

    # Check indices_info
    print("📈 MARKET INDICES INFO:")
    print("-" * 80)
    cursor.execute(
        "SELECT index_code, index_name_fa, index_type FROM indices_info ORDER BY index_code"
    )
    indices = cursor.fetchall()
    for code, name, type_ in indices:
        cursor.execute("SELECT COUNT(*) FROM market_indices WHERE index_code = ?", (code,))
        count = cursor.fetchone()[0]
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {code:10s} {name:30s} {count:>10,} records")
    print()

    # Check sector indices
    print("🏢 SECTOR INDICES:")
    print("-" * 80)
    cursor.execute("""
        SELECT s.sector_id, s.sector_name, COUNT(si.id) as record_count
        FROM sectors s
        LEFT JOIN sector_indices si ON s.sector_id = si.sector_code
        GROUP BY s.sector_id, s.sector_name
        ORDER BY s.sector_name
    """)
    sectors = cursor.fetchall()
    total_sectors = 0
    loaded_sectors = 0
    for sector_id, name, count in sectors:
        if count > 0:
            status = "✓"
            loaded_sectors += 1
        else:
            status = "✗"
        total_sectors += 1
        print(f"  {status} [{sector_id:3d}] {name:50s} {count:>10,} records")
    print(f"\n  Summary: {loaded_sectors}/{total_sectors} sectors loaded")
    print()

    # Check date ranges
    print("📅 DATE RANGES:")
    print("-" * 80)

    # Market indices date range
    cursor.execute("SELECT MIN(date), MAX(date) FROM market_indices")
    result = cursor.fetchone()
    if result[0]:
        print(f"  Market Indices: {result[0]} to {result[1]}")

    # Sector indices date range
    cursor.execute("SELECT MIN(date), MAX(date) FROM sector_indices")
    result = cursor.fetchone()
    if result[0]:
        print(f"  Sector Indices: {result[0]} to {result[1]}")

    # Price data date range
    cursor.execute("SELECT MIN(date), MAX(date) FROM price_data")
    result = cursor.fetchone()
    if result[0]:
        print(f"  Price Data:     {result[0]} to {result[1]}")
    print()

    # Check for issues
    print("⚠️  POTENTIAL ISSUES:")
    print("-" * 80)
    issues = []

    # Check for empty critical tables
    critical_tables = ["sectors", "markets", "panels", "companies"]
    for table in critical_tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        if cursor.fetchone()[0] == 0:
            issues.append(f"Table '{table}' is empty")

    # Check for companies without sector
    cursor.execute("SELECT COUNT(*) FROM companies WHERE sector_id IS NULL")
    null_sectors = cursor.fetchone()[0]
    if null_sectors > 0:
        issues.append(f"{null_sectors} companies without sector_id")

    if issues:
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("  ✓ No critical issues found")
    print()

    print("=" * 80)
    print("DATABASE CHECK COMPLETE")
    print("=" * 80)

    conn.close()


if __name__ == "__main__":
    check_database()
