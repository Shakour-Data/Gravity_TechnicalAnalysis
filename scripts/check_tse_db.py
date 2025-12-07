import sqlite3

db_path = r'E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in cursor.fetchall()]

print("📊 Tables in GravityTseHisPrice database:")
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"  • {table}: {count} records")

    # Get column info
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()
    print(f"    Columns: {', '.join([c[1] for c in columns])}")

conn.close()
