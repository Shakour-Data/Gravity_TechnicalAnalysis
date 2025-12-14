from pathlib import Path

# مسیر دیتابیس
DB_PATH = 'data/TechAnalysis.db.bak'

# فایل خروجی SQL dump
OUTPUT_SQL = 'data/backup_full.sql'

def dump_database(db_path, output_path):
    """Dump entire database to SQL file"""
    if not Path(db_path).exists():
        print(f"Database file {db_path} not found!")
        return

    print(f"Dumping database {db_path} to {output_path}...")

    # Use sqlite3 command line tool for dump
    import subprocess
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            subprocess.run(['sqlite3', db_path, '.dump'], stdout=f, check=True, text=True)
        print(f"Database successfully dumped to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during dump: {e}")
    except FileNotFoundError:
        print("sqlite3 command not found. Please ensure SQLite is installed.")

def main():
    dump_database(DB_PATH, OUTPUT_SQL)

if __name__ == "__main__":
    main()