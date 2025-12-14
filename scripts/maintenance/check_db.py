import sqlite3

conn = sqlite3.connect('data/TechAnalysis.db.bak')
cursor = conn.cursor()
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = [row[0] for row in cursor.fetchall()]
print('Tables:', tables)

# Check market_data_cache
if 'market_data_cache' in tables:
    cursor.execute('SELECT COUNT(*) FROM market_data_cache')
    count = cursor.fetchone()[0]
    print(f'market_data_cache has {count} rows')

conn.close()