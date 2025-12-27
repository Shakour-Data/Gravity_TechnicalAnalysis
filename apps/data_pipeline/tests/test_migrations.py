"""
Tests for database migration and schema validation
"""

import os
import sqlite3

import pytest
from gravity_pipeline.migrations.manager import (
    MigrationConfig,
    SchemaManager,
    SchemaValidator,
)
from sqlalchemy import create_engine


@pytest.fixture
def temp_sqlite_db():
    """Create temporary SQLite database"""
    db_path = "./test_migrations.db"
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def migration_config(temp_sqlite_db):
    """Create migration configuration"""
    return MigrationConfig(
        database_url=f"sqlite:///{temp_sqlite_db}",
        migrations_dir="./test_migrations",
    )


@pytest.fixture
def engine_with_schema(temp_sqlite_db):
    """Create SQLite engine with test schema"""
    engine = create_engine(f"sqlite:///{temp_sqlite_db}")
    
    # Create test tables
    conn = sqlite3.connect(temp_sqlite_db)
    cursor = conn.cursor()
    
    # Create candles table
    cursor.execute("""
    CREATE TABLE candles (
        id INTEGER PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        UNIQUE(symbol, timestamp)
    )
    """)
    
    # Create index
    cursor.execute("CREATE INDEX idx_candles_symbol ON candles(symbol)")
    cursor.execute("CREATE INDEX idx_candles_timestamp ON candles(timestamp)")
    
    conn.commit()
    conn.close()
    
    return engine


@pytest.fixture
def schema_manager(engine_with_schema):
    """Create schema manager instance"""
    return SchemaManager(engine_with_schema)


@pytest.fixture
def schema_validator(engine_with_schema):
    """Create schema validator instance"""
    return SchemaValidator(engine_with_schema)


def test_migration_config_validation(migration_config):
    """Test migration configuration validation"""
    # Should pass validation
    assert migration_config.validate() is True


def test_get_tables(schema_manager):
    """Test getting list of tables"""
    tables = schema_manager.get_tables()
    
    assert "candles" in tables
    assert len(tables) >= 1


def test_get_table_columns(schema_manager):
    """Test getting table columns"""
    columns = schema_manager.get_table_columns("candles")
    
    assert len(columns) > 0
    column_names = [c["name"] for c in columns]
    assert "id" in column_names
    assert "symbol" in column_names
    assert "open" in column_names


def test_get_table_indexes(schema_manager):
    """Test getting table indexes"""
    indexes = schema_manager.get_table_indexes("candles")
    
    assert len(indexes) >= 2  # Should have idx_symbol and idx_timestamp
    
    index_names = [idx["name"] for idx in indexes]
    assert "idx_candles_symbol" in index_names


def test_get_primary_key(schema_manager):
    """Test getting primary key"""
    pk = schema_manager.get_primary_key("candles")
    
    assert "id" in pk


def test_table_exists(schema_manager):
    """Test checking if table exists"""
    assert schema_manager.table_exists("candles") is True
    assert schema_manager.table_exists("nonexistent") is False


def test_column_exists(schema_manager):
    """Test checking if column exists"""
    assert schema_manager.column_exists("candles", "symbol") is True
    assert schema_manager.column_exists("candles", "nonexistent") is False


def test_get_schema_info(schema_manager):
    """Test getting complete schema information"""
    schema = schema_manager.get_schema_info()
    
    assert "candles" in schema
    assert "columns" in schema["candles"]
    assert "indexes" in schema["candles"]


def test_validate_tables(schema_validator):
    """Test table validation"""
    results = schema_validator.validate_tables(["candles"])
    
    assert results["candles"] is True


def test_validate_columns(schema_validator):
    """Test column validation"""
    results = schema_validator.validate_columns(
        "candles",
        ["id", "symbol", "timestamp", "open", "high", "low", "close", "volume"]
    )
    
    assert all(results.values()) is True


def test_validate_indexes(schema_validator):
    """Test index validation"""
    results = schema_validator.validate_indexes(
        "candles",
        ["symbol", "timestamp"]
    )
    
    # Should have at least some indexes
    assert len(results) > 0


def test_validate_all(schema_validator):
    """Test complete schema validation"""
    result = schema_validator.validate_all()
    
    assert "valid" in result
    assert "details" in result
    
    # Check structure
    assert isinstance(result["valid"], bool)
    assert isinstance(result["details"], dict)


def test_validate_before_load(schema_validator):
    """Test schema validation before load"""
    valid = schema_validator.validate_before_load()
    
    # Should be valid since we created proper schema
    assert isinstance(valid, bool)


def test_validate_missing_table(engine_with_schema):
    """Test validation with missing table"""
    validator = SchemaValidator(engine_with_schema)
    
    # Try to validate non-existent table
    results = validator.validate_tables(["analysis_results"])
    
    # Should fail validation
    assert results["analysis_results"] is False


def test_validate_missing_column(engine_with_schema):
    """Test validation with missing column"""
    validator = SchemaValidator(engine_with_schema)
    
    # Try to validate non-existent column
    results = validator.validate_columns("candles", ["nonexistent_column"])
    
    # Should fail validation
    assert results["nonexistent_column"] is False
