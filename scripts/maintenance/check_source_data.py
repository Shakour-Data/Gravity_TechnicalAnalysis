import sqlite3

conn = sqlite3.connect('E:\\Shakour\\MyProjects\\GravityTseHisPrice\\data\\tse_data.db')
cursor = conn.cursor()

cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
print('Tables:', tables)

cursor.execute('SELECT COUNT(*) FROM price_data')
total = cursor.fetchone()[0]
print('Total price records:', total)

cursor.execute('SELECT ticker, COUNT(*) FROM price_data GROUP BY ticker ORDER BY COUNT(*) DESC LIMIT 10')
results = cursor.fetchall()
print('Top 10 symbols by record count:')
for ticker, count in results:
    print(f'  {ticker}: {count}')

cursor.execute('SELECT ticker, MIN(date), MAX(date), COUNT(*) FROM price_data WHERE ticker = "وسپه" GROUP BY ticker')
result = cursor.fetchone()
print('Sample symbol (وسپه) date range:', result)

cursor.execute('PRAGMA table_info(price_data)')
columns = cursor.fetchall()
print('Columns in price_data:', columns)

cursor.execute('SELECT date, timestamp FROM price_data WHERE ticker = "وسپه" ORDER BY date DESC LIMIT 5')
results = cursor.fetchall()
print('Sample records:', results)

conn.close()