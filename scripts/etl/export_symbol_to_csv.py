

import logging
import os
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine

# تنظیمات اتصال به دیتابیس PostgreSQL (مطابق docker-compose.stack.yml)
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = int(os.environ.get("PG_PORT", 5545))
PG_USER = os.environ.get("PG_USER", "gravity")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "gravity_db_pass")
PG_DB = os.environ.get("PG_DB", "tech_analysis")
EXPORT_BASE = Path("data/CSV_Output_dam")

# تنظیم logging حرفه‌ای
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)



def export_symbol_data(symbol_or_index: str, export_base=EXPORT_BASE, fuzzy: bool = False):
    """
    Export all data for a given symbol/index/company name from all PostgreSQL tables to CSV files in a dedicated folder.
    Also generates a summary CSV with metadata for each table.
    If fuzzy=True, uses LIKE/ILIKE for partial matches.
    """
    export_dir = export_base / symbol_or_index
    export_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            dbname=PG_DB
        )
        cur = conn.cursor()
        # SQLAlchemy engine for pandas
        db_url = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
        engine = create_engine(db_url)
        logging.info(f"Connected to PostgreSQL at {PG_HOST}:{PG_PORT} (DB: {PG_DB})")

        # دریافت لیست جداول دیتابیس
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        """)
        tables = [row[0] for row in cur.fetchall()]
        logging.info(f"Found {len(tables)} tables in the database.")

        for table in tables:
            try:
                cur.execute(sql.SQL("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s;"), [table])
                columns_info = cur.fetchall()
                # انتخاب همه ستون‌های متنی (character varying, text, char, ...)
                text_types = {'character varying', 'text', 'varchar', 'char', 'character'}
                text_cols = [(col, dtype) for col, dtype in columns_info if dtype in text_types]
                if not text_cols:
                    continue
                found = False
                for col, dtype in text_cols:
                    try:
                        if fuzzy:
                            query = sql.SQL("SELECT * FROM {} WHERE {} ILIKE %(val)s").format(
                                sql.Identifier(table), sql.Identifier(col)
                            )
                            param = {"val": f"%{symbol_or_index}%"}
                        else:
                            query = sql.SQL("SELECT * FROM {} WHERE {} = %(val)s").format(
                                sql.Identifier(table), sql.Identifier(col)
                            )
                            param = {"val": symbol_or_index}
                        df = pd.read_sql_query(query.as_string(conn), engine, params=param)
                    except Exception as col_err:
                        logging.error(f"Skipping column '{col}' in table '{table}' (type={dtype}) due to error: {col_err}")
                        continue
                    if not df.empty:
                        # پیدا کردن آخرین تاریخ اگر ستون تاریخ وجود دارد
                        date_col = None
                        for dcol in ['date', 'datetime', 'created_at', 'updated_at']:
                            for c in df.columns:
                                if c.lower() == dcol:
                                    date_col = c
                                    break
                            if date_col:
                                break
                        latest_date = None
                        if date_col:
                            try:
                                latest_date = pd.to_datetime(df[date_col]).max()
                                if pd.notnull(latest_date):
                                    latest_date = latest_date.strftime('%Y%m%d')
                            except Exception:
                                latest_date = None
                        fname = f"{table}__{col}"
                        if latest_date:
                            fname += f"_{latest_date}"
                        fname += ".csv"
                        out_path = export_dir / fname
                        df.to_csv(out_path, index=False)
                        found = True
                        logging.info(f"Exported {len(df)} rows from table '{table}' (column '{col}') to '{fname}'")
                        summary.append({
                            'table': table,
                            'column': col,
                            'csv_file': fname,
                            'row_count': len(df),
                            'columns': ','.join(df.columns),
                            'latest_date': latest_date if latest_date else '',
                        })
                if not found:
                    logging.info(f"No data found for '{symbol_or_index}' in table '{table}' (any text column)")
            except Exception as e:
                logging.error(f"Error processing table '{table}': {e}")

        # ذخیره summary
        if summary:
            summary_df = pd.DataFrame(summary)
            summary_df.to_csv(export_dir / "_summary.csv", index=False)
            logging.info("Summary metadata saved to _summary.csv")
        else:
            logging.warning(f"No data found for '{symbol_or_index}' in any table.")

        cur.close()
        conn.close()
        logging.info(f"All available data for '{symbol_or_index}' exported to {export_dir}")
    except Exception as e:
        logging.critical(f"Failed to export data: {e}")



def list_unique_symbols():
    """
    لیست تمام مقادیر منحصربه‌فرد symbol/index/name/ticker در همه جداول دیتابیس را استخراج می‌کند.
    """
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            dbname=PG_DB
        )
        cur = conn.cursor()
        logging.info(f"Connected to PostgreSQL at {PG_HOST}:{PG_PORT} (DB: {PG_DB})")
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        """)
        tables = [row[0] for row in cur.fetchall()]
        all_values = set()
        for table in tables:
            cur.execute(sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_name = %s;"), [table])
            columns = [row[0] for row in cur.fetchall()]
            for col in columns:
                if col.lower() in ['symbol', 'index', 'name', 'ticker']:
                    try:
                        cur.execute(sql.SQL("SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL").format(
                            sql.Identifier(col), sql.Identifier(table), sql.Identifier(col)
                        ))
                        values = [str(row[0]) for row in cur.fetchall() if row[0] is not None]
                        for v in values:
                            all_values.add(v)
                    except Exception as e:
                        logging.error(f"Error reading {col} from {table}: {e}")
        if all_values:
            print("\nنمادها/شاخص‌ها/نام‌های موجود در دیتابیس:")
            for v in sorted(all_values):
                print(v)
        else:
            print("هیچ مقدار منحصربه‌فردی یافت نشد.")
        cur.close()
        conn.close()
        return sorted(all_values)
    except Exception as e:
        logging.critical(f"Failed to list unique symbols: {e}")
        return []
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="""
        حرفه‌ای: استخراج تمام داده‌های مربوط به یک نماد/شاخص/نام شرکت از تمام جداول دیتابیس PostgreSQL و ذخیره به صورت CSV در فولدر اختصاصی.
        - هر جدول در یک فایل CSV جداگانه ذخیره می‌شود.
        - اگر جدول ستونی با نام تاریخ داشته باشد، تاریخ آخرین داده در نام فایل می‌آید.
        - یک فایل _summary.csv شامل متادیتا (تعداد رکورد، نام ستون‌ها و ...) تولید می‌شود.
        - لاگ‌گذاری و مدیریت خطا حرفه‌ای است.
        - با دستور --list می‌توانید لیست تمام نمادها/شاخص‌ها/نام‌های موجود را ببینید.
        - با --fuzzy جستجوی تقریبی (مانند بخشی از نام شرکت یا نماد) انجام می‌شود.
        - با --auto همه مقادیر را به صورت خودکار استخراج و خروجی می‌گیرد.
        """
    )
    parser.add_argument("symbol", nargs="?", help="نماد، شاخص یا نام شرکت مورد نظر (برای لیست مقادیر موجود، --list را بزنید)")
    parser.add_argument("--out", default=str(EXPORT_BASE), help="مسیر فولدر خروجی (پیش‌فرض: data/CSV_Output_dam)")
    parser.add_argument("--list", action="store_true", help="لیست تمام نمادها/شاخص‌ها/نام‌های موجود در دیتابیس")
    parser.add_argument("--fuzzy", action="store_true", help="جستجوی تقریبی (contains/LIKE) برای نماد یا نام شرکت")
    parser.add_argument("--auto", action="store_true", help="استخراج خودکار همه مقادیر موجود (batch export)")
    args = parser.parse_args()
    if args.list:
        list_unique_symbols()
    elif args.auto:
        all_syms = list_unique_symbols()
        for sym in all_syms:
            print(f"\n--- Exporting: {sym} ---")
            export_symbol_data(sym, Path(args.out), fuzzy=False)
    elif args.symbol:
        export_symbol_data(args.symbol, Path(args.out), fuzzy=args.fuzzy)
    else:
        print("لطفاً نماد/شاخص/نام شرکت را وارد کنید یا --list یا --auto را برای استخراج گروهی بزنید.")
