import sqlite3
from datetime import datetime, timedelta

conn = sqlite3.connect('data/TechAnalysis.db')
cursor = conn.cursor()

# Get unique symbols in analysis_results
cursor.execute("SELECT DISTINCT symbol FROM analysis_results")
symbols_in_results = [row[0] for row in cursor.fetchall()]
print(f"Unique symbols in analysis_results: {len(symbols_in_results)}")
print("Sample symbols:", symbols_in_results[:10])

# Get date range in analysis_results
cursor.execute("SELECT MIN(analysis_date), MAX(analysis_date) FROM analysis_results")
min_date, max_date = cursor.fetchone()
print(f"Date range in analysis_results: {min_date} to {max_date}")

# Check if all dates are within last 90 days
current_date = datetime.now()
ninety_days_ago = current_date - timedelta(days=90)
print(f"90 days ago: {ninety_days_ago.isoformat()}")

# Count records within last 90 days
cursor.execute("SELECT COUNT(*) FROM analysis_results WHERE analysis_date >= ?", (ninety_days_ago.isoformat(),))
count_recent = cursor.fetchone()[0]
total_count = cursor.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0]
print(f"Records in last 90 days: {count_recent} out of {total_count}")

# Check if all symbols have data in last 90 days
cursor.execute("""
SELECT symbol, COUNT(*) as count
FROM analysis_results
WHERE analysis_date >= ?
GROUP BY symbol
""", (ninety_days_ago.isoformat(),))
symbols_with_recent_data = cursor.fetchall()
print(f"Symbols with data in last 90 days: {len(symbols_with_recent_data)}")

conn.close()
