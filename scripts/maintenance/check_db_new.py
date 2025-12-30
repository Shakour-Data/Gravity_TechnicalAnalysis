import sqlite3

conn = sqlite3.connect("data/TechAnalysis.db")
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print("Tables in TechAnalysis.db:")
for table in tables:
    table_name = table[0]
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"- {table_name}: {count} rows")

    # Also show a sample row if exists
    if count > 0:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
        sample = cursor.fetchone()
        print(f"  Sample row: {sample}")

conn.close()
