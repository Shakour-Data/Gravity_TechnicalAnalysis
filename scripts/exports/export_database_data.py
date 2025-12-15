import csv
import os
import sqlite3
from pathlib import Path

# مسیر دیتابیس اصلی
DB_PATH = 'data/TechAnalysis.db.bak'

# پوشه خروجی برای فایل‌های CSV
OUTPUT_DIR = 'data/exports'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def export_table_to_csv(conn, table_name, output_dir):
    """Export a table to CSV file"""
    cursor = conn.cursor()

    # Get column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [col[1] for col in cursor.fetchall()]

    # Get all data
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    # Write to CSV
    csv_path = Path(output_dir) / f"{table_name}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(columns)
        # Write data
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows from {table_name} to {csv_path}")

def main():
    if not Path(DB_PATH).exists():
        print(f"Database file {DB_PATH} not found!")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    print(f"Found {len(tables)} tables in database:")
    for table in tables:
        print(f"- {table}")

    print("\nExporting data...")

    # Export each table
    for table in tables:
        try:
            export_table_to_csv(conn, table, OUTPUT_DIR)
        except Exception as e:
            print(f"Error exporting {table}: {e}")

    conn.close()
    print(f"\nAll data exported to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

