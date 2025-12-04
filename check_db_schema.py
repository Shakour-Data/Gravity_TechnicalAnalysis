import sqlite3
from pathlib import Path

db_path = Path(r"E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db")

if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    print("Available tables:")
    for table in tables:
        print(f"  - {table}")
        
        # Get column info for each table
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"    Columns: {', '.join(columns)}")
    
    conn.close()
else:
    print(f"Database not found at {db_path}")
