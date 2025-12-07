"""
Comprehensive Database Issues Analysis
بررسی جامع مشکلات دیتابیسی پروژه
"""

import sqlite3
from pathlib import Path


def analyze_database_issues():
    db_path = Path("data/gravity_tech.db")

    if not db_path.exists():
        print("❌ Database file not found!")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n" + "="*80)
    print("🔍 DATABASE ISSUES ANALYSIS")
    print("="*80)

    issues = []

    # 1. Check tables
    print("\n1️⃣ SCHEMA CONSISTENCY ISSUES:")
    print("-" * 80)

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"   Total tables: {len(tables)}")

    # Check for naming inconsistency
    naming_issues = []
    for table in tables:
        if 'historical' in table:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in cursor.fetchall()]

            if 'symbol' in columns and 'ticker' in columns:
                naming_issues.append(f"   ⚠️ {table}: Contains BOTH 'symbol' AND 'ticker'")
            elif 'timestamp' in columns and 'analysis_date' in columns:
                naming_issues.append(f"   ⚠️ {table}: Contains BOTH 'timestamp' AND 'analysis_date'")

    if naming_issues:
        issues.extend(naming_issues)
        for issue in naming_issues:
            print(issue)
    else:
        print("   ✅ No column naming conflicts")

    # 2. Check historical_scores inconsistency
    print("\n2️⃣ HISTORICAL_SCORES TABLE STRUCTURE ISSUES:")
    print("-" * 80)

    cursor.execute("PRAGMA table_info(historical_scores)")
    hs_columns = {col[1]: col[2] for col in cursor.fetchall()}
    print(f"   Columns in historical_scores: {len(hs_columns)}")

    # Check for redundant columns
    if 'symbol' in hs_columns and 'ticker' in hs_columns:
        issues.append("   ⚠️ historical_scores: Contains BOTH 'symbol' AND 'ticker' (redundant)")
        print("   ⚠️ historical_scores: BOTH 'symbol' AND 'ticker' present (REDUNDANT)")
    elif 'ticker' in hs_columns:
        print("   ✅ Uses 'ticker' column")
    elif 'symbol' in hs_columns:
        print("   ✅ Uses 'symbol' column")

    if 'timestamp' in hs_columns and 'analysis_date' in hs_columns:
        issues.append("   ⚠️ historical_scores: Contains BOTH 'timestamp' AND 'analysis_date' (redundant)")
        print("   ⚠️ historical_scores: BOTH 'timestamp' AND 'analysis_date' present (REDUNDANT)")
    elif 'analysis_date' in hs_columns:
        print("   ✅ Uses 'analysis_date' column")
    elif 'timestamp' in hs_columns:
        print("   ✅ Uses 'timestamp' column")

    # 3. Check data type inconsistencies
    print("\n3️⃣ DATA TYPE INCONSISTENCIES:")
    print("-" * 80)

    for table in ['historical_scores', 'historical_indicator_scores', 'tool_performance_history']:
        if table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = {col[1]: col[2] for col in cursor.fetchall()}

            # Check for TEXT instead of DATETIME/TIMESTAMP
            text_dates = [f"{col}" for col in columns if ('date' in col.lower() or 'time' in col.lower()) and columns[col] == 'TEXT']
            if text_dates:
                issues.append(f"   ⚠️ {table}: Date/Time columns stored as TEXT instead of proper types: {text_dates}")
                print(f"   ⚠️ {table}: Date/Time as TEXT: {text_dates}")

    # 4. Check for NULL values and constraints
    print("\n4️⃣ DATA INTEGRITY ISSUES:")
    print("-" * 80)

    cursor.execute("SELECT COUNT(*) FROM historical_scores WHERE ticker IS NULL OR analysis_date IS NULL")
    null_count = cursor.fetchone()[0]
    if null_count > 0:
        issues.append(f"   ⚠️ historical_scores: {null_count:,} rows with NULL ticker or analysis_date")
        print(f"   ⚠️ historical_scores: {null_count:,} rows with NULL values")
    else:
        print("   ✅ No NULL values in key columns")

    # 5. Check for duplicate records
    print("\n5️⃣ DUPLICATE RECORDS:")
    print("-" * 80)

    cursor.execute("""
        SELECT ticker, analysis_date, timeframe, COUNT(*) as cnt
        FROM historical_scores
        GROUP BY ticker, analysis_date, timeframe
        HAVING cnt > 1
        LIMIT 10
    """)

    duplicates = cursor.fetchall()
    if duplicates:
        issues.append(f"   ⚠️ historical_scores: Found {len(duplicates)} duplicate key combinations")
        print(f"   ⚠️ Found {len(duplicates)} duplicate key combinations:")
        for dup in duplicates:
            print(f"      {dup[0]} | {dup[1]} | {dup[2]} (count: {dup[3]})")
    else:
        print("   ✅ No duplicate records")

    # 6. Check foreign key references
    print("\n6️⃣ FOREIGN KEY ISSUES:")
    print("-" * 80)

    cursor.execute("""
        SELECT COUNT(*) FROM historical_indicator_scores
        WHERE score_id NOT IN (SELECT id FROM historical_scores) OR score_id IS NULL
    """)

    orphaned = cursor.fetchone()[0]
    if orphaned > 0:
        issues.append(f"   ⚠️ historical_indicator_scores: {orphaned:,} orphaned records (invalid score_id)")
        print(f"   ⚠️ Orphaned records in indicator_scores: {orphaned:,}")
    else:
        print("   ✅ All foreign key references valid")

    # 7. Check index efficiency
    print("\n7️⃣ INDEX COVERAGE:")
    print("-" * 80)

    # List all indexes
    cursor.execute("""
        SELECT name, tbl_name FROM sqlite_master
        WHERE type='index' AND tbl_name IN ('historical_scores', 'historical_indicator_scores', 'tool_performance_history')
        ORDER BY tbl_name
    """)

    indexes = cursor.fetchall()
    if indexes:
        print(f"   Total indexes: {len(indexes)}")
        for idx in indexes:
            print(f"      • {idx[1]}: {idx[0]}")
    else:
        issues.append("   ⚠️ Missing indexes on frequently queried columns")
        print("   ⚠️ Missing indexes")

    # 8. Check data volume and storage
    print("\n8️⃣ DATA VOLUME STATISTICS:")
    print("-" * 80)

    cursor.execute("SELECT COUNT(*) FROM historical_scores")
    hs_count = cursor.fetchone()[0]
    print(f"   historical_scores: {hs_count:,} records")

    cursor.execute("SELECT COUNT(*) FROM historical_indicator_scores")
    his_count = cursor.fetchone()[0]
    print(f"   historical_indicator_scores: {his_count:,} records")

    cursor.execute("SELECT COUNT(*) FROM tool_performance_history")
    tph_count = cursor.fetchone()[0]
    print(f"   tool_performance_history: {tph_count:,} records")

    if his_count > hs_count * 10:
        issues.append(f"   ⚠️ historical_indicator_scores has {his_count / hs_count:.1f}x more records than historical_scores (potential bloat)")
        print(f"   ⚠️ Indicator table is {his_count / hs_count:.1f}x larger than scores table")

    # 9. Check for schema mismatch between code and database
    print("\n9️⃣ SCHEMA VS CODE MISMATCH:")
    print("-" * 80)

    cursor.execute("PRAGMA table_info(historical_indicator_scores)")
    ind_columns = [col[1] for col in cursor.fetchall()]

    expected_ind_cols = ['id', 'score_id', 'symbol', 'timestamp', 'timeframe', 'indicator_name', 'indicator_category', 'indicator_params', 'value', 'signal', 'confidence', 'created_at']

    # Check in project_schema.py what's defined
    missing_cols = set(expected_ind_cols) - set(ind_columns)
    extra_cols = set(ind_columns) - set(expected_ind_cols)

    if missing_cols or extra_cols:
        if missing_cols:
            issues.append(f"   ⚠️ Missing columns in DB: {missing_cols}")
            print(f"   ⚠️ Missing in DB: {missing_cols}")
        if extra_cols:
            print(f"   ⚠️ Extra in DB: {extra_cols}")
    else:
        print("   ✅ Schema matches definitions")

    # 10. Check consistency of historical_scores vs historical_indicator_scores
    print("\n🔟 CONSISTENCY: historical_scores vs indicator_scores")
    print("-" * 80)

    # Get column names from both tables
    cursor.execute("PRAGMA table_info(historical_scores)")
    hs_cols = [col[1] for col in cursor.fetchall()]

    cursor.execute("PRAGMA table_info(historical_indicator_scores)")
    his_cols = [col[1] for col in cursor.fetchall()]

    # Check for naming consistency
    hs_has_ticker = 'ticker' in hs_cols
    his_has_symbol = 'symbol' in his_cols

    hs_has_analysis_date = 'analysis_date' in hs_cols
    his_has_timestamp = 'timestamp' in his_cols

    if hs_has_ticker != his_has_symbol:
        issues.append("   ⚠️ Naming mismatch: historical_scores uses 'ticker', indicator_scores uses 'symbol'")
        print("   ⚠️ NAMING MISMATCH: 'ticker' vs 'symbol'")

    if hs_has_analysis_date != his_has_timestamp:
        issues.append("   ⚠️ Naming mismatch: historical_scores uses 'analysis_date', indicator_scores uses 'timestamp'")
        print("   ⚠️ NAMING MISMATCH: 'analysis_date' vs 'timestamp'")

    # Summary
    print("\n" + "="*80)
    print("📋 SUMMARY OF ISSUES")
    print("="*80)

    if issues:
        print(f"\n🔴 TOTAL ISSUES FOUND: {len(issues)}\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("\n✅ NO CRITICAL ISSUES FOUND")

    print("\n" + "="*80)

    cursor.close()
    conn.close()

if __name__ == "__main__":
    analyze_database_issues()
