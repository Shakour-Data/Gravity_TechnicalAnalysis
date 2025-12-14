import sqlite3
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DB_FILE

conn = sqlite3.connect(DB_FILE)
conn.execute('DELETE FROM market_indices')
conn.commit()
conn.close()
print('Market indices table cleared')
