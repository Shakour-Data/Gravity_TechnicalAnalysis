"""
Database schema migration utilities

Integrates Alembic for automatic schema version control.
Enables auto-generation of migrations and validation.
"""

import os
from typing import Any

import structlog
from sqlalchemy import MetaData, inspect
from sqlalchemy.engine import Engine

logger = structlog.get_logger()


class MigrationConfig:
    """Configuration for migrations"""

    def __init__(
        self,
        database_url: str,
        migrations_dir: str = "./migrations",
        alembic_ini: str = "./alembic.ini",
    ):
        self.database_url = database_url
        self.migrations_dir = migrations_dir
        self.alembic_ini = alembic_ini

    def validate(self) -> bool:
        """Validate migration configuration"""
        if not os.path.exists(self.alembic_ini):
            logger.error("migration_config_invalid", missing_alembic_ini=self.alembic_ini)
            return False

        if not os.path.exists(self.migrations_dir):
            logger.warning("migrations_dir_not_found", path=self.migrations_dir)
            # Create it
            os.makedirs(self.migrations_dir, exist_ok=True)

        return True


class SchemaManager:
    """Manage database schema and migrations"""

    def __init__(self, engine: Engine):
        """Initialize schema manager"""
        self.engine = engine
        self.inspector = inspect(engine)
        self.metadata = MetaData()

    def get_tables(self) -> list[str]:
        """Get list of all tables in database"""
        tables = self.inspector.get_table_names()
        logger.info("tables_found", count=len(tables), tables=tables)
        return tables

    def get_table_columns(self, table_name: str) -> list[dict[str, Any]]:
        """Get columns for specific table"""
        try:
            columns = self.inspector.get_columns(table_name)
            logger.info("columns_found", table=table_name, count=len(columns))
            return columns
        except Exception as e:
            logger.error("get_columns_error", table=table_name, error=str(e))
            return []

    def get_table_indexes(self, table_name: str) -> list[dict[str, Any]]:
        """Get indexes for specific table"""
        try:
            indexes = self.inspector.get_indexes(table_name)
            logger.info("indexes_found", table=table_name, count=len(indexes))
            return indexes
        except Exception as e:
            logger.error("get_indexes_error", table=table_name, error=str(e))
            return []

    def get_primary_key(self, table_name: str) -> list[str]:
        """Get primary key columns for table"""
        try:
            pk = self.inspector.get_pk_constraint(table_name)
            logger.info("primary_key_found", table=table_name, pk=pk)
            return pk.get("constrained_columns", [])
        except Exception as e:
            logger.error("get_pk_error", table=table_name, error=str(e))
            return []

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        exists = table_name in self.inspector.get_table_names()
        logger.debug("table_exists", table=table_name, exists=exists)
        return exists

    def column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if column exists in table"""
        if not self.table_exists(table_name):
            return False

        columns = self.inspector.get_columns(table_name)
        column_names = [col["name"] for col in columns]

        exists = column_name in column_names
        logger.debug("column_exists", table=table_name, column=column_name, exists=exists)
        return exists

    def get_schema_info(self) -> dict[str, Any]:
        """Get complete schema information"""
        tables = self.get_tables()
        schema = {}

        for table_name in tables:
            schema[table_name] = {
                "columns": self.get_table_columns(table_name),
                "indexes": self.get_table_indexes(table_name),
                "primary_key": self.get_primary_key(table_name),
            }

        logger.info("schema_info_retrieved", tables=len(tables))
        return schema


class SchemaValidator:
    """Validate database schema consistency"""

    # Expected schema definition
    EXPECTED_SCHEMA = {
        "candles": {
            "columns": {
                "id": "INTEGER",
                "symbol": "TEXT",
                "timestamp": "TEXT",
                "open": "REAL/DECIMAL",
                "high": "REAL/DECIMAL",
                "low": "REAL/DECIMAL",
                "close": "REAL/DECIMAL",
                "volume": "REAL/DECIMAL",
            },
            "indexes": ["symbol", "timestamp"],
            "primary_key": ["id"],
        },
        "analysis_results": {
            "columns": {
                "id": "INTEGER",
                "symbol": "TEXT",
                "signal": "TEXT",
                "confidence": "REAL/DECIMAL",
                "timestamp": "TEXT",
            },
            "indexes": ["symbol", "timestamp"],
            "primary_key": ["id"],
        },
    }

    def __init__(self, engine: Engine):
        """Initialize validator"""
        self.engine = engine
        self.manager = SchemaManager(engine)

    def validate_tables(self, expected_tables: list[str]) -> dict[str, bool]:
        """Validate all required tables exist"""
        logger.info("validating_tables", expected=len(expected_tables))

        results = {}
        for table_name in expected_tables:
            exists = self.manager.table_exists(table_name)
            results[table_name] = exists

            if not exists:
                logger.error("missing_table", table=table_name)
            else:
                logger.info("table_valid", table=table_name)

        return results

    def validate_columns(self, table_name: str, expected_columns: list[str]) -> dict[str, bool]:
        """Validate table has all required columns"""
        logger.info("validating_columns", table=table_name, expected=len(expected_columns))

        results = {}
        for column_name in expected_columns:
            exists = self.manager.column_exists(table_name, column_name)
            results[column_name] = exists

            if not exists:
                logger.error("missing_column", table=table_name, column=column_name)

        return results

    def validate_indexes(self, table_name: str, expected_indexes: list[str]) -> dict[str, bool]:
        """Validate table has required indexes"""
        logger.info("validating_indexes", table=table_name, expected=len(expected_indexes))

        actual_indexes = self.manager.get_table_indexes(table_name)

        results = {}
        for index_name in expected_indexes:
            # Check if index contains this column
            has_index = any(index_name in idx.get("column_names", []) for idx in actual_indexes)
            results[index_name] = has_index

            if not has_index:
                logger.warning("missing_index", table=table_name, index=index_name)

        return results

    def validate_all(self) -> dict[str, Any]:
        """Validate entire schema"""
        logger.info("schema_validation_starting")

        validation_results = {}

        for table_name, spec in self.EXPECTED_SCHEMA.items():
            logger.info("validating_table", table=table_name)

            validation_results[table_name] = {
                "tables": self.validate_tables([table_name]),
                "columns": self.validate_columns(table_name, list(spec["columns"].keys())),
                "indexes": self.validate_indexes(table_name, spec.get("indexes", [])),
            }

        # Determine overall validation status
        all_valid = all(
            all(v.values())
            for table_results in validation_results.values()
            for v in table_results.values()
        )

        logger.info("schema_validation_complete", valid=all_valid)

        return {
            "valid": all_valid,
            "details": validation_results,
        }

    def validate_before_load(self) -> bool:
        """Validate schema before loading data"""
        result = self.validate_all()

        if result["valid"]:
            logger.info("schema_valid_for_load")
            return True
        else:
            logger.error("schema_invalid_for_load", details=result["details"])
            return False


class MigrationGenerator:
    """Generate Alembic migrations from SQLAlchemy models"""

    def __init__(self, alembic_dir: str = "./migrations"):
        """Initialize migration generator"""
        self.alembic_dir = alembic_dir

    def generate_migration(self, message: str, autogenerate: bool = True) -> bool:
        """
        Generate migration file

        Args:
            message: Migration message
            autogenerate: Use autogenerate (requires alembic installed)

        Returns:
            True if successful
        """
        try:
            import subprocess

            # Run alembic revision command
            cmd = ["alembic", "revision", "--autogenerate", "-m", message]

            result = subprocess.run(cmd, cwd=self.alembic_dir, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("migration_generated", message=message)
                return True
            else:
                logger.error("migration_generation_failed", error=result.stderr)
                return False

        except Exception as e:
            logger.error("migration_error", error=str(e))
            return False

    def upgrade_database(self, revision: str = "head") -> bool:
        """
        Apply migrations to database

        Args:
            revision: Target revision (default: head)

        Returns:
            True if successful
        """
        try:
            import subprocess

            cmd = ["alembic", "upgrade", revision]
            result = subprocess.run(cmd, cwd=self.alembic_dir, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("database_upgraded", revision=revision)
                return True
            else:
                logger.error("database_upgrade_failed", error=result.stderr)
                return False

        except Exception as e:
            logger.error("upgrade_error", error=str(e))
            return False

    def downgrade_database(self, revision: str) -> bool:
        """
        Downgrade database to specific revision

        Args:
            revision: Target revision

        Returns:
            True if successful
        """
        try:
            import subprocess

            cmd = ["alembic", "downgrade", revision]
            result = subprocess.run(cmd, cwd=self.alembic_dir, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("database_downgraded", revision=revision)
                return True
            else:
                logger.error("database_downgrade_failed", error=result.stderr)
                return False

        except Exception as e:
            logger.error("downgrade_error", error=str(e))
            return False
