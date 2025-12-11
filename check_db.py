import sqlite3

conn = sqlite3.connect('data/TechAnalysis.db')
cursor = conn.cursor()

# Get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print('Tables:', [t[0] for t in tables])
print()

# Get schema for each table
for table in tables:
    table_name = table[0]
    print(f'Table: {table_name}')
    cursor.execute(f'PRAGMA table_info({table_name})')
    columns = cursor.fetchall()
    print(f'  Columns: {[col[1] for col in columns]}')
    cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
    count = cursor.fetchone()[0]
    print(f'  Rows: {count}')
    print()

conn.close()