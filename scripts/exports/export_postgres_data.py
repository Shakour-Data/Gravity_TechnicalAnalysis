import csv
import os
from pathlib import Path

import psycopg2

# تنظیمات اتصال به PostgreSQL
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "gravity",
    "password": "gravity",
    "database": "gravity",
}

# پوشه خروجی برای فایل‌های CSV
OUTPUT_DIR = "data/postgres_exports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_all_tables(conn):
    """دریافت لیست تمام جداول در دیتابیس"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    tables = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return tables


def export_table_to_csv(conn, table_name, output_dir):
    """Export یک جدول به فایل CSV"""
    cursor = conn.cursor()

    # دریافت نام ستون‌ها
    cursor.execute(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position;
    """)
    columns = [col[0] for col in cursor.fetchall()]

    # دریافت تمام داده‌ها
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    # نوشتن به CSV
    csv_path = Path(output_dir) / f"{table_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # نوشتن هدر
        writer.writerow(columns)
        # نوشتن داده‌ها
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows from {table_name} to {csv_path}")
    cursor.close()


def main():
    try:
        # اتصال به دیتابیس
        conn = psycopg2.connect(**DB_CONFIG)
        print("Connected to PostgreSQL database successfully!")

        # دریافت لیست جداول
        tables = get_all_tables(conn)
        print(f"Found {len(tables)} tables in database:")
        for table in tables:
            print(f"- {table}")

        print("\nExporting data...")

        # Export هر جدول
        for table in tables:
            try:
                export_table_to_csv(conn, table, OUTPUT_DIR)
            except Exception as e:
                print(f"Error exporting {table}: {e}")

        conn.close()
        print(f"\nAll data exported to {OUTPUT_DIR}")

    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
