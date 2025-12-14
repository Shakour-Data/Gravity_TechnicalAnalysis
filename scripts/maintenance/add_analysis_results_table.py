#!/usr/bin/env python3
"""
Add analysis_results table to the database
"""
import sqlite3
from pathlib import Path

# Path to the database
db_path = Path("data/TechAnalysis.db")

# SQL to create the table
create_table_sql = """
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    analysis_date DATETIME NOT NULL,
    final_signal VARCHAR(20) NOT NULL,
    confidence DECIMAL(5, 4) DEFAULT 0.0,
    trend_score DECIMAL(5, 4) DEFAULT 0.0,
    momentum_score DECIMAL(5, 4) DEFAULT 0.0,
    volatility_score DECIMAL(5, 4) DEFAULT 0.0,
    cycle_score DECIMAL(5, 4) DEFAULT 0.0,
    sr_score DECIMAL(5, 4) DEFAULT 0.0,
    volume_interaction_score DECIMAL(5, 4) DEFAULT 0.0,
    decision_matrix_score DECIMAL(5, 4) DEFAULT 0.0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

def main():
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(create_table_sql)
        conn.commit()
        print("✅ analysis_results table created successfully")
    except Exception as e:
        print(f"❌ Error creating table: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
