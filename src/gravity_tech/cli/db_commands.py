"""
Database CLI Commands

دستورات CLI برای مدیریت دیتابیس:
- ایجاد دیتابیس و جداول
- ذخیره‌سازی داده‌ها
- به‌روزرسانی schema
- بازنشانی جداول
- بررسی وضعیت
- بکاپ و بازیابی

Author: Gravity Tech Team
Date: December 5, 2025
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

import gzip
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import click
from gravity_tech.config.settings import settings
from gravity_tech.database.database_manager import DatabaseManager, DatabaseType

try:
    from psycopg2 import Error as PsycopgError  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PsycopgError = ()  # type: ignore


DEFAULT_SQLITE_PATH = settings.sqlite_path
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _validate_identifier(value: str, label: str) -> str:
    if not value or not IDENTIFIER_RE.fullmatch(value):
        raise click.BadParameter(
            f"{label} باید فقط شامل حروف، اعداد یا '_' باشد.",
            param_hint=label
        )
    return value


def _format_db_error(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        return "فایل دیتابیس پیدا نشد. مسیر را بررسی کنید."
    if isinstance(error, PermissionError):
        return "دسترسی نوشتن/خواندن به فایل وجود ندارد."
    if isinstance(error, sqlite3.Error):
        return f"خطای SQLite: {error}"
    if PsycopgError and isinstance(error, PsycopgError):
        return f"خطای PostgreSQL: {error}"
    return str(error)


def _parse_filters(filters: tuple[str, ...]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for raw_filter in filters:
        if "=" not in raw_filter:
            raise click.BadParameter(
                "فیلتر باید به شکل column=value باشد.",
                param_hint="filter"
            )
        column, value = raw_filter.split("=", 1)
        column = column.strip()
        value = value.strip()
        _validate_identifier(column, "نام ستون")
        parsed.append((column, value))
    return parsed


def _build_filter_clause(
    parsed_filters: list[tuple[str, str]],
    db_manager: DatabaseManager
) -> tuple[str, tuple[Any, ...]]:
    if not parsed_filters:
        return "", ()

    placeholder = db_manager.get_sql_placeholder()
    clauses: list[str] = []
    params: list[Any] = []

    for column, value in parsed_filters:
        clauses.append(f"{column} = {placeholder}")
        params.append(value)

    return " AND ".join(clauses), tuple(params)


def _parse_query_params(
    params: tuple[str, ...],
    params_json: str | None
) -> tuple[Any, ...] | None:
    if params_json:
        try:
            parsed = json.loads(params_json)
        except json.JSONDecodeError as exc:
            raise click.BadParameter(
                f"JSON پارامترها معتبر نیست: {exc}"
            ) from exc
        if not isinstance(parsed, list):
            raise click.BadParameter("پارامترها باید لیست JSON باشند.")
        return tuple(parsed)

    if params:
        return tuple(params)

    return None


def _open_backup_writer(path: Path, compress: bool):
    if compress:
        return gzip.open(path, 'wt', encoding='utf-8')
    return open(path, 'w', encoding='utf-8')


def _open_backup_reader(path: str):
    if path.lower().endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, encoding='utf-8')


def _record_matches_filters(record: dict[str, Any], filters: list[tuple[str, str]]) -> bool:
    for column, value in filters:
        if str(record.get(column)) != value:
            return False
    return True


def _stream_backup_to_file(
    db_manager: DatabaseManager,
    tables: list[str],
    output_file: Path,
    compress: bool,
    chunk_size: int
) -> dict[str, int]:
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "database": db_manager.get_database_info()
    }
    table_counts: dict[str, int] = {}

    with _open_backup_writer(output_file, compress) as handle:
        handle.write('{\n')
        handle.write('  "metadata": ')
        metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
        handle.write(metadata_json.replace('\n', '\n  '))
        handle.write(',\n')
        handle.write('  "data": {\n')

        for idx, table in enumerate(tables):
            handle.write(f'    "{table}": [\n')
            record_count = 0
            first_record = True
            for record in db_manager.stream_table_records(table, chunk_size=chunk_size):
                record_count += 1
                serialized = json.dumps(record, ensure_ascii=False, default=str)
                prefix = "" if first_record else ","
                handle.write(f'      {prefix}{serialized}\n')
                first_record = False
            handle.write('    ]')
            if idx < len(tables) - 1:
                handle.write(',\n')
            else:
                handle.write('\n')
            table_counts[table] = record_count

        handle.write('  }\n')
        handle.write('}\n')

    return table_counts


@click.group()
def db_cli():
    """دستورات مدیریت دیتابیس Gravity Tech"""
    pass


@db_cli.command()
@click.option('--type', 'db_type',
              type=click.Choice(['postgresql', 'sqlite', 'auto'], case_sensitive=False),
              default='auto',
              help='نوع دیتابیس (auto برای تشخیص خودکار)')
@click.option('--connection', 'connection_string',
              default=None,
              help='رشته اتصال PostgreSQL (مثال: postgresql://user:pass@localhost/dbname)')
@click.option('--sqlite-path', 'sqlite_path',
              default=DEFAULT_SQLITE_PATH,
              show_default=True,
              help='مسیر فایل SQLite')
@click.option('--force', is_flag=True,
              help='بازنویسی دیتابیس موجود')
def init(db_type: str, connection_string: str | None, sqlite_path: str, force: bool):
    """
    ایجاد دیتابیس و جداول

    مثال:
        python -m gravity_tech.cli.db_commands init
        python -m gravity_tech.cli.db_commands init --type postgresql --connection "postgresql://user:pass@localhost/gravity"
        python -m gravity_tech.cli.db_commands init --type sqlite --sqlite-path data/mydb.db
    """
    click.echo("🚀 در حال راه‌اندازی دیتابیس...")

    try:
        # Determine database type
        if db_type == 'auto':
            db_type_enum = None  # Let DatabaseManager auto-detect
            click.echo("🔍 تشخیص خودکار نوع دیتابیس...")
        elif db_type == 'postgresql':
            db_type_enum = DatabaseType.POSTGRESQL
            click.echo("🐘 استفاده از PostgreSQL")
        else:
            db_type_enum = DatabaseType.SQLITE
            click.echo("💾 استفاده از SQLite")

        # Initialize database manager
        db_manager = DatabaseManager(
            db_type=db_type_enum,
            connection_string=connection_string,
            sqlite_path=sqlite_path,
            auto_setup=False  # We'll setup manually
        )

        # Check if database exists (use public method)
        db_info = db_manager.get_database_info()
        if not force and db_info.get('connected', False):
            if not click.confirm('⚠️ دیتابیس قبلاً ایجاد شده است. آیا می‌خواهید بازنویسی کنید؟'):
                click.echo("❌ عملیات لغو شد.")
                return

        # Setup database
        click.echo("📦 در حال ایجاد جداول...")
        db_manager.setup_database()

        # Show database info
        info = db_manager.get_database_info()
        click.echo("\n✅ دیتابیس با موفقیت ایجاد شد!")
        click.echo(f"   نوع: {info['type']}")
        click.echo(f"   مسیر: {info.get('path', info.get('connection', 'N/A'))}")
        click.echo(f"   تعداد جداول: {info.get('table_count', 0)}")

    except Exception as e:
        click.echo(f"❌ خطا در ایجاد دیتابیس: {_format_db_error(e)}", err=True)
        sys.exit(1)


@db_cli.command()
@click.option('--sqlite-path', 'sqlite_path',
              default=DEFAULT_SQLITE_PATH,
              show_default=True,
              help='مسیر فایل SQLite')
def status(sqlite_path: str):
    """
    نمایش وضعیت دیتابیس

    مثال:
        python -m gravity_tech.cli.db_commands status
    """
    click.echo("🔍 بررسی وضعیت دیتابیس...")

    try:

        db_manager = DatabaseManager(sqlite_path=sqlite_path, auto_setup=False)
        info = db_manager.get_database_info()
        if not info.get('connected', False):
            click.echo("❌ دیتابیس یافت نشد. ابتدا با دستور 'init' دیتابیس را ایجاد کنید.")
            sys.exit(1)

        stats = cast(dict[str, int], db_manager.get_statistics())

        click.echo("\n📊 Database Status:")
        click.echo(f"   Type: {info['type']}")
        click.echo(f"   Status: {'✅ Active' if info.get('connected', False) else '❌ Inactive'}")
        click.echo(f"   Path: {info.get('path', info.get('connection', 'N/A'))}")
        click.echo(f"   Table count: {info.get('table_count', 0)}")

        if stats:
            click.echo("\n📈 Table Stats:")
            for table, count in stats.items():
                click.echo(f"   {table}: {count:,} records")

    except Exception as e:
        click.echo(f"❌ خطا در بررسی وضعیت: {_format_db_error(e)}", err=True)
        sys.exit(1)


@db_cli.command()
@click.argument('table_name')
@click.option('--sqlite-path', 'sqlite_path',
              default=DEFAULT_SQLITE_PATH,
              show_default=True,
              help='مسیر فایل SQLite')
@click.option('--force', is_flag=True,
              help='بدون تأیید بازنشانی کن')
def reset_table(table_name: str, sqlite_path: str, force: bool):
    """
    بازنشانی یک جدول (حذف تمام داده‌ها)

    مثال:
        python -m gravity_tech.cli.db_commands reset-table historical_scores
        python -m gravity_tech.cli.db_commands reset-table historical_scores --force
    """
    original_table_name = table_name
    click.echo(f"⚠️ در حال بازنشانی جدول '{original_table_name}'...")

    try:
        table_name = _validate_identifier(table_name, "نام جدول")
    except click.BadParameter as err:
        click.echo(f"❌ {err.message}", err=True)
        return

    if not force:
        if not click.confirm(f'آیا مطمئن هستید که می‌خواهید تمام داده‌های جدول "{table_name}" را حذف کنید?'):
            click.echo("❌ عملیات لغو شد.")
            return

    try:
        db_manager = DatabaseManager(sqlite_path=sqlite_path, auto_setup=False)

        # Get count before
        stats_before = cast(dict[str, int], db_manager.get_statistics())
        count_before = stats_before.get(table_name, 0)

        # Reset table
        db_manager.reset_table(table_name)

        click.echo(f"✅ جدول '{table_name}' با موفقیت بازنشانی شد.")
        click.echo(f"   تعداد رکوردهای حذف شده: {count_before:,}")

    except Exception as e:
        click.echo(f"❌ خطا در بازنشانی جدول: {_format_db_error(e)}", err=True)
        sys.exit(1)


@db_cli.command()
@click.option('--sqlite-path', 'sqlite_path',
              default=DEFAULT_SQLITE_PATH,
              show_default=True,
              help='مسیر فایل SQLite')
@click.option('--force', is_flag=True,
              help='بدون تأیید بازنشانی کن')
def reset_all(sqlite_path: str, force: bool):
    """
    بازنشانی تمام جداول (حذف تمام داده‌ها)

    مثال:
        python -m gravity_tech.cli.db_commands reset-all
        python -m gravity_tech.cli.db_commands reset-all --force
    """
    click.echo("⚠️⚠️⚠️ در حال بازنشانی تمام جداول...")

    if not force:
        if not click.confirm('آیا مطمئن هستید که می‌خواهید تمام داده‌های تمام جداول را حذف کنید?'):
            click.echo("❌ عملیات لغو شد.")
            return
        if not click.confirm('این عملیات قابل بازگشت نیست! آیا کاملاً مطمئن هستید?'):
            click.echo("❌ عملیات لغو شد.")
            return

    try:
        db_manager = DatabaseManager(sqlite_path=sqlite_path, auto_setup=False)

        # Get stats before
        stats_before = cast(dict[str, int], db_manager.get_statistics())
        total_before = sum(stats_before.values())

        # Reset all tables
        for table_name in stats_before.keys():
            db_manager.reset_table(table_name)
            click.echo(f"   ✓ {table_name}: {stats_before[table_name]:,} رکورد حذف شد")

        click.echo("\n✅ تمام جداول با موفقیت بازنشانی شدند.")
        click.echo(f"   مجموع رکوردهای حذف شده: {total_before:,}")

    except Exception as e:
        click.echo(f"❌ خطا در بازنشانی جداول: {_format_db_error(e)}", err=True)
        sys.exit(1)


@db_cli.command()
@click.option('--sqlite-path', 'sqlite_path',
              default=DEFAULT_SQLITE_PATH,
              show_default=True,
              help='مسیر فایل SQLite')
def migrate(sqlite_path: str):
    """
    به‌روزرسانی schema دیتابیس

    این دستور schema دیتابیس را با آخرین تغییرات همگام می‌کند.

    مثال:
        python -m gravity_tech.cli.db_commands migrate
    """
    click.echo("🔄 در حال به‌روزرسانی schema دیتابیس...")

    try:
        db_manager = DatabaseManager(sqlite_path=sqlite_path, auto_setup=False)

        # Run migrations
        click.echo("📝 بررسی تغییرات...")
        migrations_applied = db_manager.run_migrations()

        if migrations_applied:
            click.echo(f"✅ {len(migrations_applied)} migration اعمال شد:")
            for migration in migrations_applied:
                click.echo(f"   ✓ {migration}")
        else:
            click.echo("✅ دیتابیس به‌روز است. نیازی به migration نیست.")

    except Exception as e:
        click.echo(f"❌ خطا در migration: {_format_db_error(e)}", err=True)
        sys.exit(1)


@db_cli.command()
@click.option('--output', 'output_path',
              default=None,
              help='مسیر فایل خروجی (پیش‌فرض: backup_YYYYMMDD_HHMMSS.json[.gz])')
@click.option('--sqlite-path', 'sqlite_path',
              default=DEFAULT_SQLITE_PATH,
              show_default=True,
              help='مسیر فایل SQLite')
@click.option('--tables', 'table_names',
              default=None,
              help='لیست جداول برای backup (با کاما جدا شوند، پیش‌فرض: همه)')
@click.option('--compress/--no-compress',
              default=True,
              help='ذخیره فایل به شکل gzip برای کاهش حجم')
@click.option('--chunk-size', 'chunk_size',
              default=1000,
              type=int,
              show_default=True,
              help='تعداد رکورد خوانده شده در هر مرحله (برای کاهش مصرف حافظه)')
def backup(output_path: str | None, sqlite_path: str, table_names: str | None,
           compress: bool, chunk_size: int):
    """
    ایجاد backup از دیتابیس

    مثال:
        python -m gravity_tech.cli.db_commands backup
        python -m gravity_tech.cli.db_commands backup --output backup.json
        python -m gravity_tech.cli.db_commands backup --tables historical_scores,tool_performance
    """
    click.echo("💾 در حال ایجاد backup...")

    if chunk_size <= 0:
        raise click.BadParameter("chunk-size باید بزرگ‌تر از صفر باشد.", param_hint="chunk-size")

    try:
        db_manager = DatabaseManager(sqlite_path=sqlite_path, auto_setup=False)

        # Parse table names
        if table_names:
            tables = [
                _validate_identifier(name.strip(), "نام جدول")
                for name in table_names.split(',')
                if name.strip()
            ]
        else:
            tables = db_manager.get_tables()

        if not tables:
            click.echo("ℹ️ دیتابیسی برای backup موجود نیست.")
            return

        # Generate filename if not provided
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            extension = ".json.gz" if compress else ".json"
            output_path = f"backup_{timestamp}{extension}"

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        table_counts = _stream_backup_to_file(
            db_manager=db_manager,
            tables=tables,
            output_file=output_file,
            compress=compress,
            chunk_size=chunk_size
        )

        total_records = sum(table_counts.values())
        file_size = output_file.stat().st_size / 1024  # KB

        click.echo("✅ Backup با موفقیت ایجاد شد:")
        click.echo(f"   فایل: {output_file}")
        click.echo(f"   حجم: {file_size:.2f} KB")
        click.echo(f"   تعداد رکوردها: {total_records:,}")
        click.echo(f"   جداول: {', '.join(tables)}")
        click.echo(f"   فشرده‌سازی: {'فعال' if compress else 'غیرفعال'}")

    except Exception as e:
        click.echo(f"❌ خطا در ایجاد backup: {_format_db_error(e)}", err=True)
        sys.exit(1)


@db_cli.command()
@click.argument('backup_file', type=click.Path(exists=True))
@click.option('--sqlite-path', 'sqlite_path',
              default=DEFAULT_SQLITE_PATH,
              show_default=True,
              help='مسیر فایل SQLite')
@click.option('--force', is_flag=True,
              help='بازنویسی داده‌های موجود')
def restore(backup_file: str, sqlite_path: str, force: bool):
    """
    بازیابی دیتابیس از backup

    مثال:
        python -m gravity_tech.cli.db_commands restore backup.json
        python -m gravity_tech.cli.db_commands restore backup_20251205_120000.json --force
    """
    click.echo(f"♻️ در حال بازیابی از backup: {backup_file}")

    if not force:
        if not click.confirm('⚠️ این عملیات داده‌های موجود را بازنویسی می‌کند. ادامه می‌دهید?'):
            click.echo("❌ عملیات لغو شد.")
            return

    try:
        db_manager = DatabaseManager(sqlite_path=sqlite_path, auto_setup=False)

        # Load backup file
        with _open_backup_reader(backup_file) as f:
            backup_data = json.load(f)

        # Restore
        click.echo("📥 در حال بازیابی داده‌ها...")
        result = db_manager.restore_backup(backup_data)

        click.echo("✅ Backup با موفقیت بازیابی شد:")
        for table, count in result.items():
            click.echo(f"   ✓ {table}: {count:,} رکورد")

    except Exception as e:
        click.echo(f"❌ خطا در بازیابی backup: {_format_db_error(e)}", err=True)
        sys.exit(1)


@db_cli.command()
@click.argument('json_file', type=click.Path(exists=True))
@click.option('--table', 'table_name',
              required=True,
              help='نام جدول برای insert')
@click.option('--sqlite-path', 'sqlite_path',
              default=DEFAULT_SQLITE_PATH,
              show_default=True,
              help='مسیر فایل SQLite')
@click.option('--batch-size', 'batch_size',
              default=1000,
              help='تعداد رکوردها در هر batch')
def import_data(json_file: str, table_name: str, sqlite_path: str, batch_size: int):
    """
    import داده‌ها از فایل JSON

    مثال:
        python -m gravity_tech.cli.db_commands import-data data.json --table historical_scores
    """
    click.echo(f"📥 در حال import داده‌ها از: {json_file}")

    try:
        db_manager = DatabaseManager(sqlite_path=sqlite_path, auto_setup=False)

        # Load data
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)

        # Import
        if isinstance(data, list):
            records = cast(list[dict[str, Any]], data)
        elif isinstance(data, dict) and table_name in data:
            records = cast(list[dict[str, Any]], data[table_name])
        else:
            raise ValueError("فرمت فایل JSON نامعتبر است")

        click.echo(f"📊 {len(records):,} رکورد یافت شد...")

        # Import in batches
        imported = 0
        with click.progressbar(length=len(records), label='در حال import') as bar:
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                db_manager.bulk_insert(table_name, batch)
                imported += len(batch)
                bar.update(len(batch))

        click.echo(f"✅ {imported:,} رکورد با موفقیت import شدند.")

    except Exception as e:
        click.echo(f"❌ خطا در import داده‌ها: {_format_db_error(e)}", err=True)
        sys.exit(1)


@db_cli.command()
@click.argument('table_name')
@click.option('--output', 'output_file',
              default=None,
              help='مسیر فایل خروجی (پیش‌فرض: TABLE_NAME_export.json)')
@click.option('--sqlite-path', 'sqlite_path',
              default=DEFAULT_SQLITE_PATH,
              show_default=True,
              help='مسیر فایل SQLite')
@click.option('--limit', 'limit',
              default=None,
              type=int,
              help='محدودیت تعداد رکوردها')
@click.option('--filter', 'filters',
              multiple=True,
              help='فیلترهای ایمن column=value (قابل تکرار)')
def export_table(table_name: str, output_file: str | None, sqlite_path: str,
                 limit: int | None, filters: tuple[str, ...]):
    """
    export یک جدول به فایل JSON

    مثال:
        python -m gravity_tech.cli.db_commands export-table historical_scores
        python -m gravity_tech.cli.db_commands export-table historical_scores --limit 100
        python -m gravity_tech.cli.db_commands export-table historical_scores --filter symbol=BTCUSDT --filter timeframe=1h
    """
    original_table_name = table_name
    click.echo(f"📤 در حال export جدول '{original_table_name}'...")
    parsed_filters = _parse_filters(filters)

    try:
        db_manager = DatabaseManager(sqlite_path=sqlite_path, auto_setup=False)
        if db_manager.db_type == DatabaseType.JSON_FILE:
            filter_clause = ""
            params: tuple[Any, ...] = ()
        else:
            filter_clause, params = _build_filter_clause(parsed_filters, db_manager)

        table_name = _validate_identifier(table_name, "نام جدول")

        query = f"SELECT * FROM {table_name}"
        if filter_clause:
            query += f" WHERE {filter_clause}"
        if limit:
            query += f" LIMIT {limit}"

        if db_manager.db_type == DatabaseType.JSON_FILE:
            records: list[dict[str, Any]] = []
            for record in db_manager.stream_table_records(table_name):
                if parsed_filters and not _record_matches_filters(record, parsed_filters):
                    continue
                records.append(record)
                if limit and len(records) >= limit:
                    break
        else:
            records = cast(
                list[dict[str, Any]],
                db_manager.execute_query(
                    query,
                    params=params if params else None,
                    fetch=True
                )
            ) or []

        # Generate filename if not provided
        if not output_file:
            output_file = f"{table_name}_export.json"

        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2, default=str)

        file_size = output_path.stat().st_size / 1024  # KB

        click.echo("✅ جدول با موفقیت export شد:")
        click.echo(f"   فایل: {output_path}")
        click.echo(f"   حجم: {file_size:.2f} KB")
        click.echo(f"   تعداد رکوردها: {len(records):,}")

    except Exception as e:
        click.echo(f"❌ خطا در export جدول: {_format_db_error(e)}", err=True)
        sys.exit(1)


@db_cli.command()
@click.argument('query')
@click.option('--sqlite-path', 'sqlite_path',
              default=DEFAULT_SQLITE_PATH,
              show_default=True,
              help='مسیر فایل SQLite')
@click.option('--output', 'output_file',
              default=None,
              help='ذخیره نتایج در فایل JSON')
@click.option('--param', 'params',
              multiple=True,
              help='پارامتر برای placeholder (می‌توانید چند بار تکرار کنید)')
@click.option('--params-json', 'params_json',
              default=None,
              help='تعریف پارامترها به صورت JSON list (مثال: "[\\"BTCUSDT\\", 10]")')
def query(query: str, sqlite_path: str, output_file: str | None,
          params: tuple[str, ...], params_json: str | None):
    """
    اجرای یک query دلخواه

    مثال:
        python -m gravity_tech.cli.db_commands query "SELECT COUNT(*) FROM historical_scores"
        python -m gravity_tech.cli.db_commands query "SELECT * FROM historical_scores WHERE symbol = %s" --param BTCUSDT
    """
    click.echo("🔍 در حال اجرای query...")

    try:
        db_manager = DatabaseManager(sqlite_path=sqlite_path, auto_setup=False)
        if db_manager.db_type == DatabaseType.JSON_FILE:
            click.echo("❌ اجرای query روی ذخیره JSON پشتیبانی نمی‌شود.", err=True)
            sys.exit(1)

        query_params = _parse_query_params(params, params_json)

        results = cast(
            list[dict[str, Any]],
            db_manager.execute_query(query, params=query_params, fetch=True)
        ) or []

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            click.echo(f"✅ نتایج در '{output_file}' ذخیره شدند.")
        else:
            click.echo(f"\n📊 نتایج ({len(results)} رکورد):")
            for i, row in enumerate(results[:10], 1):
                click.echo(f"\n   رکورد {i}:")
                for key, value in row.items():
                    click.echo(f"      {key}: {value}")

            if len(results) > 10:
                click.echo(f"\n   ... و {len(results) - 10} رکورد دیگر")

        click.echo(f"\n✅ Query با موفقیت اجرا شد. ({len(results)} رکورد)")

    except Exception as e:
        click.echo(f"❌ خطا در اجرای query: {_format_db_error(e)}", err=True)
        sys.exit(1)


@db_cli.command()
@click.option('--sqlite-path', 'sqlite_path',
              default=DEFAULT_SQLITE_PATH,
              show_default=True,
              help='مسیر فایل SQLite')
def tables(sqlite_path: str):
    """
    نمایش لیست جداول

    مثال:
        python -m gravity_tech.cli.db_commands tables
    """
    click.echo("📋 جداول موجود:")

    try:
        db_manager = DatabaseManager(sqlite_path=sqlite_path, auto_setup=False)

        table_list = cast(list[str], db_manager.get_tables())
        stats = cast(dict[str, int], db_manager.get_statistics())

        if not table_list:
            click.echo("   هیچ جدولی یافت نشد.")
            return

        click.echo(f"\n   تعداد کل: {len(table_list)} جدول\n")

        for table in table_list:
            count = stats.get(table, 0)
            click.echo(f"   • {table}: {count:,} رکورد")

    except Exception as e:
        click.echo(f"❌ خطا در دریافت لیست جداول: {_format_db_error(e)}", err=True)
        sys.exit(1)


@db_cli.command()
@click.argument('table_name')
@click.option('--sqlite-path', 'sqlite_path',
              default=DEFAULT_SQLITE_PATH,
              show_default=True,
              help='مسیر فایل SQLite')
def schema(table_name: str, sqlite_path: str):
    """
    نمایش schema یک جدول

    مثال:
        python -m gravity_tech.cli.db_commands schema historical_scores
    """
    click.echo(f"📐 Schema جدول '{table_name}':")

    try:
        db_manager = DatabaseManager(sqlite_path=sqlite_path, auto_setup=False)

        schema_info = cast(list[dict[str, Any]], db_manager.get_table_schema(table_name))

        click.echo("\n   ستون‌ها:")
        for col in schema_info:
            nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
            default = f" DEFAULT {col.get('default')}" if col.get('default') else ""
            click.echo(f"      • {col['name']}: {col['type']} {nullable}{default}")

    except Exception as e:
        click.echo(f"❌ خطا در دریافت schema: {_format_db_error(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    db_cli()
