import sqlite3

# Check the backup DB in workspace
print("Checking backup DB in workspace:")
conn = sqlite3.connect('data/TechAnalysis.db.bak')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables in backup database:")
for table in tables:
    print(f"- {table[0]}")

if tables:
    for table_name in [t[0] for t in tables]:
        print(f"\nSchema for table '{table_name}':")
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  {col[1]} ({col[2]})")

conn.close()

print("\n" + "="*50 + "\n")

# Check multiple source DBs
dbs_to_check = [
    r'E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data_backup.db',
    r'E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db',
    r'E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data_codex_backup.db'
]

for db_path in dbs_to_check:
    print(f"Checking {db_path}:")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        print("Tables:")
        for table in tables:
            print(f"- {table[0]}")

        # Look for tables that might contain price data
        price_tables = [t[0] for t in tables if 'price' in t[0].lower() or 'data' in t[0].lower() or 'candle' in t[0].lower()]
        if price_tables:
            print(f"Potential price data tables: {price_tables}")
            for table_name in price_tables[:3]:  # Check first 3
                print(f"\nSchema for table '{table_name}':")
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                for col in columns:
                    print(f"  {col[1]} ({col[2]})")
                # Sample data
                try:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                    rows = cursor.fetchall()
                    print(f"Sample data ({len(rows)} rows):")
                    for row in rows:
                        print(f"  {row}")
                except Exception:
                    pass

        conn.close()
    except Exception as e:
        print(f"Error checking {db_path}: {e}")

    print("\n" + "-"*30 + "\n")
