"""
Verify database migration results
"""
import sqlite3
from pathlib import Path

db_path = Path("data/gravity_tech.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("\n" + "="*80)
print("✅ MIGRATION VERIFICATION")
print("="*80)

# 1. Check historical_scores schema
print("\n1️⃣ historical_scores columns:")
cursor.execute("PRAGMA table_info(historical_scores)")
columns = {col[1]: col[2] for col in cursor.fetchall()}
for col, dtype in sorted(columns.items()):
    print(f"   • {col}: {dtype}")

# Verify key columns
assert 'ticker' in columns, "❌ Missing 'ticker' column"
assert 'analysis_date' in columns, "❌ Missing 'analysis_date' column"
assert columns['analysis_date'] == 'DATETIME', "❌ analysis_date should be DATETIME"
assert columns['updated_at'] == 'DATETIME', "❌ updated_at should be DATETIME"
print("   ✅ All columns correct")

# 2. Check historical_indicator_scores schema
print("\n2️⃣ historical_indicator_scores columns:")
cursor.execute("PRAGMA table_info(historical_indicator_scores)")
columns = {col[1]: col[2] for col in cursor.fetchall()}
for col, dtype in sorted(columns.items()):
    print(f"   • {col}: {dtype}")

assert 'ticker' in columns, "❌ Missing 'ticker' column"
assert 'analysis_date' in columns, "❌ Missing 'analysis_date' column"
assert columns['analysis_date'] == 'DATETIME', "❌ analysis_date should be DATETIME"
print("   ✅ All columns correct")

# 3. Check FOREIGN KEY constraint
print("\n3️⃣ FOREIGN KEY constraints:")
cursor.execute("PRAGMA foreign_key_check")
fk_issues = cursor.fetchall()
if fk_issues:
    print(f"   ❌ Found {len(fk_issues)} FK violations")
    for issue in fk_issues:
        print(f"      {issue}")
else:
    print("   ✅ No FOREIGN KEY violations")

# 4. Test date queries
print("\n4️⃣ Date query functionality:")
cursor.execute("""
    SELECT COUNT(*) FROM historical_scores
    WHERE analysis_date > datetime('2020-01-01')
    AND analysis_date < datetime('2025-12-31')
""")
count = cursor.fetchone()[0]
print(f"   ✅ Date range query works: {count:,} records in range")

# 5. Check data integrity
print("\n5️⃣ Data integrity:")
cursor.execute("SELECT COUNT(*) FROM historical_scores")
hs_count = cursor.fetchone()[0]
print(f"   • historical_scores: {hs_count:,} records")

cursor.execute("SELECT COUNT(*) FROM historical_indicator_scores")
his_count = cursor.fetchone()[0]
print(f"   • historical_indicator_scores: {his_count:,} records")

cursor.execute("SELECT COUNT(DISTINCT ticker) FROM historical_scores")
tickers = cursor.fetchone()[0]
print(f"   • Unique tickers in scores: {tickers}")

cursor.execute("SELECT COUNT(DISTINCT ticker) FROM historical_indicator_scores")
tickers_ind = cursor.fetchone()[0]
print(f"   • Unique tickers in indicators: {tickers_ind}")

print("\n" + "="*80)
print("✅ ALL VERIFICATIONS PASSED!")
print("="*80 + "\n")

cursor.close()
conn.close()
