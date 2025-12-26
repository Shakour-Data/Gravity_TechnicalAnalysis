"""
SQLite data loader

Loads OHLCV candles into SQLite database.
"""

import sqlite3
from typing import Any, Dict, List, Optional

import structlog

from gravity_pipeline.loaders.base import Loader

logger = structlog.get_logger()


class SQLiteLoader(Loader):
    """Load OHLCV data into SQLite"""
    
    def __init__(
        self,
        db_path: str = "./data/gravity.db",
        batch_size: int = 500,
        table_name: str = "candles",
        auto_create_table: bool = True,
    ):
        """
        Initialize SQLite loader
        
        Args:
            db_path: Path to SQLite database
            batch_size: Number of records to batch in single insert
            table_name: Target table name
            auto_create_table: Create table if not exists
        """
        super().__init__()
        self.db_path = db_path
        self.batch_size = batch_size
        self.table_name = table_name
        self.auto_create_table = auto_create_table
        self.connection: Optional[sqlite3.Connection] = None
    
    async def validate_connection(self) -> bool:
        """Test connection to SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            logger.info("sqlite_connection_valid", db_path=self.db_path)
            return True
        except Exception as e:
            logger.error("sqlite_connection_error", error=str(e))
            return False
    
    async def load(self, candles: List[Dict[str, Any]]) -> int:
        """
        Load candles into SQLite
        
        Args:
            candles: List of candle dicts with keys:
                - timestamp, open, high, low, close, volume, symbol (optional)
                
        Returns:
            Number of successfully loaded records
        """
        
        logger.info("sqlite_load_starting", count=len(candles), batch_size=self.batch_size)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if needed
            if self.auto_create_table:
                await self._create_table_if_not_exists(cursor)
            
            # Insert in batches
            loaded = 0
            for i in range(0, len(candles), self.batch_size):
                batch = candles[i:i + self.batch_size]
                
                try:
                    await self._insert_batch(cursor, batch)
                    loaded += len(batch)
                    
                    logger.info(
                        "batch_loaded",
                        batch_num=i // self.batch_size + 1,
                        count=len(batch)
                    )
                
                except Exception as e:
                    self.error_count += 1
                    logger.error("batch_load_error", error=str(e), batch_num=i // self.batch_size + 1)
                    continue
            
            # Commit changes
            conn.commit()
            conn.close()
            
            self.loaded_count += loaded
            
            logger.info("sqlite_load_complete", total=loaded, errors=self.error_count)
            return loaded
        
        except Exception as e:
            logger.error("sqlite_load_error", error=str(e))
            raise
    
    async def _create_table_if_not_exists(self, cursor: sqlite3.Cursor):
        """Create candles table if not exists"""
        
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        )
        """
        
        try:
            cursor.execute(create_sql)
            logger.info("table_created", table=self.table_name)
        except Exception as e:
            logger.warning("table_creation_warning", error=str(e))
    
    async def _insert_batch(self, cursor: sqlite3.Cursor, candles: List[Dict]):
        """Insert batch of candles"""
        
        insert_sql = f"""
        INSERT OR REPLACE INTO {self.table_name}
        (symbol, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        data = [
            (
                candle.get("symbol", "UNKNOWN"),
                candle.get("timestamp"),
                float(candle.get("open", 0)),
                float(candle.get("high", 0)),
                float(candle.get("low", 0)),
                float(candle.get("close", 0)),
                float(candle.get("volume", 0)),
            )
            for candle in candles
        ]
        
        cursor.executemany(insert_sql, data)
    
    async def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
        logger.info("sqlite_loader_closed")
