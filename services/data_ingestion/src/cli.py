import argparse
import json
import os
import shutil
import sqlite3
from datetime import datetime

import jdatetime

from src.config import COMPANIES_FILE, DB_FILE, MARKETS_FILE, PANELS_FILE, SECTORS_FILE
from src.database import init_price_data
from src.encoding_utils import ensure_utf8_console

ensure_utf8_console()


def _backup_database_for_connection(conn):
    """Create a timestamped backup of the current database file if possible."""
    try:
        row = conn.execute("PRAGMA database_list;").fetchone()
        db_path = row[2] if row and len(row) > 2 else DB_FILE
    except Exception:
        db_path = DB_FILE

    if not db_path or db_path in (":memory:", "") or not os.path.exists(db_path):
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.dirname(db_path) or "."
    backup_path = os.path.join(backup_dir, f"tse_data_backup_pre_drop_{ts}.db")
    try:
        shutil.copy2(db_path, backup_path)
        return backup_path
    except Exception:
        return None


def get_connection():
    return sqlite3.connect(DB_FILE)


def create_db(args):
    init_price_data.create_tables()
    print("Database and tables created.")


def load_initial(args):
    with open(SECTORS_FILE) as f:
        sectors = json.load(f)
    init_price_data.insert_sectors(sectors)
    with open(MARKETS_FILE) as f:
        markets = json.load(f)
    init_price_data.insert_markets(markets)
    with open(PANELS_FILE) as f:
        panels = json.load(f)
    init_price_data.insert_panels(panels)

    with open(COMPANIES_FILE) as f:
        companies = json.load(f)

    init_price_data.insert_companies(companies)
    print("Initial data loaded.")


def reload_table(args):
    table = args.table
    file = args.file
    with open(file) as f:
        data = json.load(f)
    func = getattr(init_price_data, f"insert_{table}")
    func(data)
    print(f"Table '{table}' reloaded from {file}.")


def drop_table(args):
    table = args.table
    conn = get_connection()
    backup_path = _backup_database_for_connection(conn)
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    conn.close()
    if backup_path:
        print(f"Table '{table}' dropped. Backup saved to {backup_path}.")
    else:
        print(f"Table '{table}' dropped. (Backup skipped)")


def update_db(args):
    # For full update, just reload all initial data
    load_initial(args)
    print("Database updated.")


def update_table(args):
    reload_table(args)


def list_sectors(args):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT sector_id, sector_name FROM sectors ORDER BY sector_name")
    sectors = cursor.fetchall()
    conn.close()
    for sector in sectors:
        print(f"{sector[0]}: {sector[1]}")


def list_companies(args):
    sector_id = args.sector_id
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT c.ticker, c.name, s.sector_name
        FROM companies c
        LEFT JOIN sectors s ON c.sector_id = s.sector_id
        WHERE c.sector_id = ?
        ORDER BY c.name
    """,
        (sector_id,),
    )
    companies = cursor.fetchall()
    conn.close()
    for company in companies:
        print(f"{company[0]}: {company[1]}")


def get_price_data(args):
    ticker = args.ticker
    limit = args.limit
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT date, adj_close, adj_volume
        FROM price_data
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT ?
    """,
        (ticker, limit),
    )
    prices = cursor.fetchall()
    conn.close()
    for price in prices:
        print(f"Date: {price[0]}, Close: {price[1]}, Volume: {price[2]}")


def main():
    def create_indices_tables(args):
        from src.database import init_price_data

        init_price_data.create_indices_tables()

    def load_market_indices(args):
        import os
        import sys

        from src.database import init_price_data

        # Add scripts directory to path
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        if not os.path.exists(os.path.join(scripts_dir, "gravity_tse.py")):
            raise ImportError(f"'gravity_tse.py' not found in {scripts_dir}")
        import gravity_tse as gpy  # type: ignore

        today_jalali = jdatetime.date.today().strftime("%Y-%m-%d")
        indices = [
            ("CWI", "شاخص کل", gpy.Get_CWI_History),
            ("EWI", "شاخص هم وزن", gpy.Get_EWI_History),
            ("CWPI", "شاخص کل قیمت", gpy.Get_CWPI_History),
            ("EWPI", "شاخص هم وزن قیمت", gpy.Get_EWPI_History),
            ("FFI", "شاخص مالی", gpy.Get_FFI_History),
            ("MKT1I", "شاخص بازار اول", gpy.Get_MKT1I_History),
            ("INDI", "شاخص صنعت", gpy.Get_INDI_History),
            ("ACT50", "شاخص 50 شرکت فعال", gpy.Get_ACT50_History),
            ("LCI30", "شاخص 30 شرکت بزرگ", gpy.Get_LCI30_History),
        ]
        for code, name_fa, func in indices:
            print(f"Loading {name_fa} ({code})...")
            try:
                df = func(start_date="1395-01-01", end_date=today_jalali, double_date=True)
                if df is not None and not df.empty:
                    init_price_data.insert_market_indices(code, name_fa, df)
                    print(f"  ✓ Loaded {len(df)} records")
                else:
                    print("  ⚠ No data returned")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        print("Market indices loading complete.")

    def load_sector_indices(args):
        import json
        import os
        import sys

        from src.config import SECTORS_FILE
        from src.database import init_price_data

        # Add scripts directory to path
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
        sys.path.insert(0, scripts_dir)
        import gravity_tse as gpy  # type: ignore

        # Load sector mapping from JSON
        with open(SECTORS_FILE) as f:
            json.load(f)

        # Map sector names to codes
        sector_mapping = {
            "فلزات اساسی": 27,
            "خودرو و ساخت قطعات": 34,
            "محصولات شیمیایی": 44,
            "سرمایه گذاریها": 56,
            "سیمان، آهک و گچ": 53,
            "مواد و محصولات دارویی": 43,
            "زراعت و خدمات وابسته": 1,
            "استخراج کانه های فلزی": 13,
            "لاستیک و پلاستیک": 25,
            "ساخت محصولات فلزی": 28,
            "ماشین آلات و دستگاه های برقی": 31,
            "رایانه و فعالیت های وابسته به آن": 72,
            "اطلاعات و ارتباطات": 73,
            "خرده فروشی،باستثنای وسایل نقلیه موتوری": 47,
        }

        # Use mapped sector names from fetcher
        from src.fetcher import DataFetcher

        today_jalali = jdatetime.date.today().strftime("%Y-%m-%d")

        for sector_name_json, sector_code in sector_mapping.items():
            mapped_name = DataFetcher.SECTOR_NAME_MAPPING.get(sector_name_json, sector_name_json)
            print(f"Loading {sector_name_json} ({sector_code})...")
            try:
                df = gpy.Get_SectorIndex_History(
                    sector=mapped_name,
                    start_date="1395-01-01",
                    end_date=today_jalali,
                    double_date=True,
                )
                if df is not None and not df.empty:
                    init_price_data.insert_sector_indices(sector_code, sector_name_json, df)
                    print(f"  ✓ Loaded {len(df)} records")
                else:
                    print("  ⚠ No data returned")
            except Exception as e:
                print(f"  ✗ Error: {e}")

        print("Sector indices loaded.")

    parser = argparse.ArgumentParser(description="TSE Database CLI")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "create-indices-tables", help="Create market and sector indices tables"
    ).set_defaults(func=create_indices_tables)
    subparsers.add_parser("load-market-indices", help="Load market indices data").set_defaults(
        func=load_market_indices
    )
    subparsers.add_parser("load-sector-indices", help="Load sector indices data").set_defaults(
        func=load_sector_indices
    )
    subparsers.add_parser("create-db", help="Create database and tables").set_defaults(
        func=create_db
    )
    subparsers.add_parser("load-initial", help="Load initial data").set_defaults(func=load_initial)

    def init_all(args):
        create_db(args)
        create_indices_tables(args)
        load_initial(args)

        from src.fetcher import DataFetcher

        print("Checking and loading price and indices data...")
        DataFetcher.run()

        print("Database, tables, and all initial data loaded.")

    subparsers.add_parser(
        "init-all", help="Create DB, tables, and load all initial data"
    ).set_defaults(func=init_all)

    def load_all_prices(args):
        from src.fetcher import DataFetcher

        DataFetcher.run()

    subparsers.add_parser(
        "load-all-prices", help="Fetch and load all price and indices data (cached)"
    ).set_defaults(func=load_all_prices)

    reload_parser = subparsers.add_parser("reload-table", help="Reload a table from JSON")
    reload_parser.add_argument("table", choices=["companies", "sectors", "markets", "panels"])
    reload_parser.add_argument("file", help="Path to JSON file")
    reload_parser.set_defaults(func=reload_table)
    drop_parser = subparsers.add_parser("drop-table", help="Drop a table")
    drop_parser.add_argument(
        "table", choices=["companies", "sectors", "markets", "panels", "price_data", "last_updates"]
    )
    drop_parser.set_defaults(func=drop_table)
    subparsers.add_parser("update-db", help="Update all tables").set_defaults(func=update_db)

    update_parser = subparsers.add_parser("update-table", help="Update a table from JSON")
    update_parser.add_argument("table", choices=["companies", "sectors", "markets", "panels"])
    update_parser.add_argument("file", help="Path to JSON file")
    update_parser.set_defaults(func=update_table)
    subparsers.add_parser("list-sectors", help="List all sectors").set_defaults(func=list_sectors)

    companies_parser = subparsers.add_parser("list-companies", help="List companies in a sector")
    companies_parser.add_argument("sector_id", type=int)
    companies_parser.set_defaults(func=list_companies)

    price_parser = subparsers.add_parser("get-price-data", help="Get price data for a ticker")
    price_parser.add_argument("ticker")
    price_parser.add_argument("--limit", type=int, default=10)
    price_parser.set_defaults(func=get_price_data)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
