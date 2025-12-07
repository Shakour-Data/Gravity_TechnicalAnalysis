"""
Gravity Tech Database CLI

اسکریپت مستقل برای مدیریت دیتابیس

استفاده:
    python scripts/db_cli.py init
    python scripts/db_cli.py status
    python scripts/db_cli.py --help

Author: Gravity Tech Team
Date: December 5, 2025
Version: 1.0.0
License: MIT
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import click

# Add root to path FIRST before any relative imports
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Import from the database package (not src/database.py)  # noqa: E402
from database.database_manager import DatabaseManager, DatabaseType  # noqa: E402


@click.group()
def cli():
    """🗄️ دستورات مدیریت دیتابیس Gravity Tech"""
    pass


@cli.command()
@click.option('--type', 'db_type',
              type=click.Choice(['postgresql', 'sqlite', 'auto'], case_sensitive=False),
              default='auto',
              help='نوع دیتابیس (auto برای تشخیص خودکار)')
@click.option('--connection', 'connection_string',
              default=None,
              help='رشته اتصال PostgreSQL')
@click.option('--sqlite-path', 'sqlite_path',
              default='data/gravity_tech.db',
              help='مسیر فایل SQLite')
@click.option('--force', is_flag=True,
              help='بازنویسی دیتابیس موجود')
def init(db_type: str, connection_string: str | None, sqlite_path: str, force: bool):
    """ایجاد دیتابیس و جداول

    مثال:
        python scripts/db_cli.py init
        python scripts/db_cli.py init --type sqlite --sqlite-path data/mydb.db
    """
    click.echo("🚀 در حال راه‌اندازی دیتابیس...")

    try:
        # Determine database type
        if db_type == 'auto':
            db_type_enum = None
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
            auto_setup=False
        )

        # Check if database exists
        if not force and db_manager._check_database_exists():
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
        click.echo(f"❌ خطا در ایجاد دیتابیس: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--sqlite-path', 'sqlite_path',
              default='data/gravity_tech.db',
              help='مسیر فایل SQLite')
def status(sqlite_path: str):
    """
    نمایش وضعیت دیتابیس

    مثال:
        python scripts/db_cli.py status
    """
    click.echo("🔍 بررسی وضعیت دیتابیس...")

    try:
        db_manager = DatabaseManager(sqlite_path=sqlite_path, auto_setup=False)

        if not db_manager._check_database_exists():
            click.echo("❌ دیتابیس یافت نشد. ابتدا با دستور 'init' دیتابیس را ایجاد کنید.")
            sys.exit(1)

        info = db_manager.get_database_info()
        stats = db_manager.get_statistics()

        click.echo("\n📊 وضعیت دیتابیس:")
        click.echo(f"   نوع: {info['type']}")
        click.echo(f"   وضعیت: {'✅ فعال' if info.get('connected', False) else '❌ غیرفعال'}")
        click.echo(f"   مسیر: {info.get('path', info.get('connection', 'N/A'))}")
        click.echo(f"   تعداد جداول: {info.get('table_count', 0)}")

        if stats:
            click.echo("\n📈 آمار:")
            for table, count in stats.items():
                click.echo(f"   {table}: {count:,} رکورد")

    except Exception as e:
        click.echo(f"❌ خطا در بررسی وضعیت: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--sqlite-path', 'sqlite_path',
              default='data/gravity_tech.db',
              help='مسیر فایل SQLite')
def tables(sqlite_path: str):
    """
    نمایش لیست جداول

    مثال:
        python scripts/db_cli.py tables
    """
    click.echo("📋 جداول موجود:")

    try:
        db_manager = DatabaseManager(sqlite_path=sqlite_path, auto_setup=False)

        table_list = db_manager.get_tables()
        stats = db_manager.get_statistics()

        if not table_list:
            click.echo("   هیچ جدولی یافت نشد.")
            return

        click.echo(f"\n   تعداد کل: {len(table_list)} جدول\n")

        for table in table_list:
            count = stats.get(table, 0)
            click.echo(f"   • {table}: {count:,} رکورد")

    except Exception as e:
        click.echo(f"❌ خطا در دریافت لیست جداول: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('table_name')
@click.option('--sqlite-path', 'sqlite_path',
              default='data/gravity_tech.db',
              help='مسیر فایل SQLite')
@click.option('--force', is_flag=True,
              help='بدون تأیید بازنشانی کن')
def reset_table(table_name: str, sqlite_path: str, force: bool):
    """
    بازنشانی یک جدول (حذف تمام داده‌ها)

    مثال:
        python scripts/db_cli.py reset-table historical_scores
        python scripts/db_cli.py reset-table historical_scores --force
    """
    click.echo(f"⚠️ در حال بازنشانی جدول '{table_name}'...")

    if not force:
        if not click.confirm(f'آیا مطمئن هستید که می‌خواهید تمام داده‌های جدول "{table_name}" را حذف کنید?'):
            click.echo("❌ عملیات لغو شد.")
            return

    try:
        db_manager = DatabaseManager(sqlite_path=sqlite_path, auto_setup=False)

        # Get count before
        stats_before = db_manager.get_statistics()
        count_before = stats_before.get(table_name, 0)

        # Reset table
        db_manager.reset_table(table_name)

        click.echo(f"✅ جدول '{table_name}' با موفقیت بازنشانی شد.")
        click.echo(f"   تعداد رکوردهای حذف شده: {count_before:,}")

    except Exception as e:
        click.echo(f"❌ خطا در بازنشانی جدول: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', 'output_path',
              default=None,
              help='مسیر فایل خروجی')
@click.option('--sqlite-path', 'sqlite_path',
              default='data/gravity_tech.db',
              help='مسیر فایل SQLite')
def backup(output_path: str | None, sqlite_path: str):
    """
    ایجاد backup از دیتابیس

    مثال:
        python scripts/db_cli.py backup
        python scripts/db_cli.py backup --output my_backup.json
    """
    click.echo("💾 در حال ایجاد backup...")

    try:
        db_manager = DatabaseManager(sqlite_path=sqlite_path, auto_setup=False)

        # Create backup
        backup_data = db_manager.create_backup()

        # Generate filename if not provided
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"backup_{timestamp}.json"

        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2, default=str)

        # Show stats
        total_records = sum(len(records) for records in backup_data.get('data', {}).values())
        file_size = output_file.stat().st_size / 1024  # KB

        click.echo("✅ Backup با موفقیت ایجاد شد:")
        click.echo(f"   فایل: {output_file}")
        click.echo(f"   حجم: {file_size:.2f} KB")
        click.echo(f"   تعداد رکوردها: {total_records:,}")

    except Exception as e:
        click.echo(f"❌ خطا در ایجاد backup: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('backup_file', type=click.Path(exists=True))
@click.option('--sqlite-path', 'sqlite_path',
              default='data/gravity_tech.db',
              help='مسیر فایل SQLite')
@click.option('--force', is_flag=True,
              help='بازنویسی داده‌های موجود')
def restore(backup_file: str, sqlite_path: str, force: bool):
    """
    بازیابی دیتابیس از backup

    مثال:
        python scripts/db_cli.py restore backup.json
        python scripts/db_cli.py restore backup_20251205_120000.json --force
    """
    click.echo(f"♻️ در حال بازیابی از backup: {backup_file}")

    if not force:
        if not click.confirm('⚠️ این عملیات داده‌های موجود را بازنویسی می‌کند. ادامه می‌دهید?'):
            click.echo("❌ عملیات لغو شد.")
            return

    try:
        db_manager = DatabaseManager(sqlite_path=sqlite_path, auto_setup=False)

        # Load backup file
        with open(backup_file, encoding='utf-8') as f:
            backup_data = json.load(f)

        # Restore
        click.echo("📥 در حال بازیابی داده‌ها...")
        result = db_manager.restore_backup(backup_data)

        click.echo("✅ Backup با موفقیت بازیابی شد:")
        for table, count in result.items():
            click.echo(f"   ✓ {table}: {count:,} رکورد")

    except Exception as e:
        click.echo(f"❌ خطا در بازیابی backup: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
