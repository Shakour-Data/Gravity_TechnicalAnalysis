#!/usr/bin/env python3
"""
Database Backup and Recovery Script

Creates automated backups of the technical analysis database
and provides disaster recovery capabilities.

Author: Gravity Tech Team
Date: November 20, 2025
Version: 1.0.0
License: MIT
"""

import os
import sys
import json
import shutil
import sqlite3
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import structlog

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.gravity_tech.database.database_manager import DatabaseManager
from src.gravity_tech.config.settings import settings

logger = structlog.get_logger()


class DatabaseBackupManager:
    """
    Comprehensive database backup and recovery manager
    """

    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = Path(project_root) / backup_dir
        self.backup_dir.mkdir(exist_ok=True)

        # Backup retention settings
        self.retention_days = 30
        self.max_backups = 50

    async def create_backup(self, backup_type: str = "full") -> str:
        """
        Create a database backup

        Args:
            backup_type: Type of backup ('full', 'incremental', 'schema_only')

        Returns:
            Path to the backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{backup_type}_{timestamp}"

        if backup_type == "full":
            backup_path = await self._create_full_backup(backup_name)
        elif backup_type == "schema_only":
            backup_path = await self._create_schema_backup(backup_name)
        else:
            raise ValueError(f"Unsupported backup type: {backup_type}")

        # Clean up old backups
        await self._cleanup_old_backups()

        logger.info("backup_created", path=str(backup_path), type=backup_type)
        return str(backup_path)

    async def _create_full_backup(self, backup_name: str) -> Path:
        """Create a full database backup"""
        backup_path = self.backup_dir / f"{backup_name}.db"

        # For SQLite, we can simply copy the database file
        db_path = Path(project_root) / "data" / "gravity_tech.db"

        if db_path.exists():
            shutil.copy2(db_path, backup_path)
        else:
            # Create empty database if it doesn't exist
            conn = sqlite3.connect(backup_path)
            conn.close()

        # Create backup metadata
        metadata = {
            "backup_type": "full",
            "timestamp": datetime.now().isoformat(),
            "database_path": str(db_path),
            "backup_path": str(backup_path),
            "version": settings.app_version
        }

        metadata_path = self.backup_dir / f"{backup_name}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return backup_path

    async def _create_schema_backup(self, backup_name: str) -> Path:
        """Create a schema-only backup"""
        schema_path = self.backup_dir / f"{backup_name}_schema.sql"

        # Read schema from schemas.sql
        schema_file = project_root / "src" / "gravity_tech" / "database" / "schemas.sql"
        if schema_file.exists():
            shutil.copy2(schema_file, schema_path)
        else:
            # Generate schema from database
            await self._export_schema_to_file(schema_path)

        return schema_path

    async def _export_schema_to_file(self, schema_path: Path):
        """Export database schema to SQL file"""
        db_manager = DatabaseManager()
        await db_manager.initialize()

        try:
            conn = db_manager.get_connection()
            cursor = conn.cursor()

            # Get all table schemas
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            with open(schema_path, 'w') as f:
                f.write("-- Database Schema Export\n")
                f.write(f"-- Generated: {datetime.now().isoformat()}\n\n")

                for table in tables:
                    table_name = table[0]
                    if not table_name.startswith('sqlite_'):
                        # Get CREATE statement
                        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
                        create_stmt = cursor.fetchone()
                        if create_stmt and create_stmt[0]:
                            f.write(f"{create_stmt[0]};\n\n")

                            # Get indexes
                            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='{table_name}' AND sql IS NOT NULL;")
                            indexes = cursor.fetchall()
                            for index in indexes:
                                f.write(f"{index[0]};\n")
                            if indexes:
                                f.write("\n")

        finally:
            await db_manager.close()

    async def restore_backup(self, backup_path: str, target_db: Optional[str] = None) -> bool:
        """
        Restore database from backup

        Args:
            backup_path: Path to backup file
            target_db: Target database path (optional)

        Returns:
            True if successful
        """
        backup_file = Path(backup_path)

        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        if target_db is None:
            target_db = str(project_root / "data" / "gravity_tech.db")

        target_path = Path(target_db)
        target_path.parent.mkdir(exist_ok=True)

        # Create backup of current database before restore
        if target_path.exists():
            backup_current = target_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
            shutil.copy2(target_path, backup_current)
            logger.info("current_db_backed_up", path=str(backup_current))

        try:
            if backup_file.suffix == '.db':
                # Full database restore
                shutil.copy2(backup_file, target_path)
            elif backup_file.suffix == '.sql':
                # Schema restore
                await self._restore_from_sql(backup_file, target_path)
            else:
                raise ValueError(f"Unsupported backup format: {backup_file.suffix}")

            logger.info("backup_restored", from_path=str(backup_file), to_path=str(target_path))
            return True

        except Exception as e:
            logger.error("backup_restore_failed", error=str(e), backup_path=str(backup_file))
            # Restore the backup of current database
            if 'backup_current' in locals() and backup_current.exists():
                shutil.copy2(backup_current, target_path)
                logger.info("rolled_back_to_previous_state")
            raise

    async def _restore_from_sql(self, sql_file: Path, target_db: Path):
        """Restore database from SQL schema file"""
        conn = sqlite3.connect(target_db)

        try:
            with open(sql_file, 'r') as f:
                sql_content = f.read()

            # Split SQL commands and execute them
            commands = [cmd.strip() for cmd in sql_content.split(';') if cmd.strip() and not cmd.strip().startswith('--')]

            for command in commands:
                if command:
                    conn.execute(command)

            conn.commit()

        finally:
            conn.close()

    async def _cleanup_old_backups(self):
        """Clean up old backup files based on retention policy"""
        try:
            # Get all backup files
            backup_files = list(self.backup_dir.glob("backup_*"))

            if len(backup_files) <= self.max_backups:
                return

            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Remove old backups
            for old_backup in backup_files[self.max_backups:]:
                old_backup.unlink()
                logger.info("old_backup_removed", path=str(old_backup))

            # Also remove backups older than retention period
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            for backup_file in backup_files:
                if datetime.fromtimestamp(backup_file.stat().st_mtime) < cutoff_date:
                    backup_file.unlink()
                    logger.info("expired_backup_removed", path=str(backup_file))

        except Exception as e:
            logger.warning("backup_cleanup_failed", error=str(e))

    async def list_backups(self) -> list:
        """List all available backups"""
        backups = []

        for backup_file in self.backup_dir.glob("backup_*"):
            if backup_file.suffix in ['.db', '.sql']:
                # Try to read metadata
                metadata_file = backup_file.with_suffix('.json')
                metadata = {}

                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass

                backups.append({
                    "filename": backup_file.name,
                    "path": str(backup_file),
                    "size": backup_file.stat().st_size,
                    "created": datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                    "type": metadata.get('backup_type', 'unknown'),
                    **metadata
                })

        return sorted(backups, key=lambda x: x['created'], reverse=True)

    async def get_backup_info(self, backup_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific backup"""
        metadata_file = self.backup_dir / f"{backup_name}.json"

        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        # Fallback: basic file info
        backup_file = self.backup_dir / backup_name
        if backup_file.exists():
            return {
                "filename": backup_name,
                "path": str(backup_file),
                "size": backup_file.stat().st_size,
                "created": datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                "type": "unknown"
            }

        return None


async def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Database Backup and Recovery")
    parser.add_argument("action", choices=["backup", "restore", "list", "info"])
    parser.add_argument("--type", choices=["full", "schema_only"], default="full")
    parser.add_argument("--backup-path", help="Path to backup file for restore")
    parser.add_argument("--backup-name", help="Name of backup for info command")

    args = parser.parse_args()

    backup_manager = DatabaseBackupManager()

    if args.action == "backup":
        backup_path = await backup_manager.create_backup(args.type)
        print(f"Backup created: {backup_path}")

    elif args.action == "restore":
        if not args.backup_path:
            print("Error: --backup-path required for restore")
            return
        success = await backup_manager.restore_backup(args.backup_path)
        print(f"Restore {'successful' if success else 'failed'}")

    elif args.action == "list":
        backups = await backup_manager.list_backups()
        print("Available backups:")
        for backup in backups:
            print(f"  {backup['filename']} ({backup['type']}) - {backup['created']}")

    elif args.action == "info":
        if not args.backup_name:
            print("Error: --backup-name required for info")
            return
        info = await backup_manager.get_backup_info(args.backup_name)
        if info:
            print(json.dumps(info, indent=2))
        else:
            print("Backup not found")


if __name__ == "__main__":
    asyncio.run(main())