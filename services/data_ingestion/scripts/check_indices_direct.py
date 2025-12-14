import sqlite3
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DB_FILE

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Count records
cursor.execute('SELECT COUNT(*) FROM market_indices')
mi = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM sector_indices')
si = cursor.fetchone()[0]

print(f"market_indices: {mi} records")
print(f"sector_indices: {si} records")

# Check schema
print("\nmarket_indices schema:")
cursor.execute('PRAGMA table_info(market_indices)')
for row in cursor.fetchall():
    print(f"  {row[1]} ({row[2]})")

print("\nsector_indices schema:")
cursor.execute('PRAGMA table_info(sector_indices)')
for row in cursor.fetchall():
    print(f"  {row[1]} ({row[2]})")

# Sample records
print("\nmarket_indices sample (first 3):")
cursor.execute('SELECT * FROM market_indices LIMIT 3')
for row in cursor.fetchall():
    print(f"  {row}")

print("\nsector_indices sample (first 3):")
cursor.execute('SELECT * FROM sector_indices LIMIT 3')
for row in cursor.fetchall():
    print(f"  {row}")

conn.close()
