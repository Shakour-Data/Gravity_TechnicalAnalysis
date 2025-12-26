"""
PostgreSQL data loader

Loads OHLCV candles into PostgreSQL database.
"""

from typing import List, Dict, Any, Optional
import asyncpg
import structlog
from gravity_pipeline.loaders.base import Loader

logger = structlog.get_logger()


class PostgreSQLLoader(Loader):
    """Load OHLCV data into PostgreSQL"""
    
    def __init__(
        self,
        connection_url: str,
        batch_size: int = 1000,
        table_name: str = "candles",
        max_pool_size: int = 10,
    ):
        """
        Initialize PostgreSQL loader
        
        Args:
            connection_url: PostgreSQL connection URL
                Format: postgresql://user:pass@host:port/database
            batch_size: Number of records to batch in single insert
            table_name: Target table name
            max_pool_size: Maximum connection pool size
        """
        super().__init__()
        self.connection_url = connection_url
        self.batch_size = batch_size
        self.table_name = table_name
        self.max_pool_size = max_pool_size
        self.pool: Optional[asyncpg.Pool] = None
    
    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool"""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.connection_url,
                max_size=self.max_pool_size,
                min_size=1,
            )
        return self.pool
    
    async def validate_connection(self) -> bool:
        """Test connection to PostgreSQL"""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            logger.info("postgres_connection_valid")
            return True
        except Exception as e:
            logger.error("postgres_connection_error", error=str(e))
            return False
    
    async def load(self, candles: List[Dict[str, Any]]) -> int:
        """
        Load candles into PostgreSQL
        
        Args:
            candles: List of candle dicts
                
        Returns:
            Number of successfully loaded records
        """
        
        logger.info("postgres_load_starting", count=len(candles), batch_size=self.batch_size)
        
        try:
            pool = await self._get_pool()
            
            # Create table if needed
            async with pool.acquire() as conn:
                await self._create_table_if_not_exists(conn)
            
            # Insert in batches
            loaded = 0
            for i in range(0, len(candles), self.batch_size):
                batch = candles[i:i + self.batch_size]
                
                try:
                    async with pool.acquire() as conn:
                        inserted = await self._insert_batch(conn, batch)
                        loaded += inserted
                    
                    logger.info(
                        "batch_loaded",
                        batch_num=i // self.batch_size + 1,
                        count=inserted
                    )
                
                except Exception as e:
                    self.error_count += 1
                    logger.error(
                        "batch_load_error",
                        error=str(e),
                        batch_num=i // self.batch_size + 1
                    )
                    continue
            
            self.loaded_count += loaded
            
            logger.info("postgres_load_complete", total=loaded, errors=self.error_count)
            return loaded
        
        except Exception as e:
            logger.error("postgres_load_error", error=str(e))
            raise
    
    async def _create_table_if_not_exists(self, conn: asyncpg.Connection):
        """Create candles table if not exists"""
        
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open DECIMAL NOT NULL,
            high DECIMAL NOT NULL,
            low DECIMAL NOT NULL,
            close DECIMAL NOT NULL,
            volume DECIMAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        );
        
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_symbol 
            ON {self.table_name}(symbol);
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp 
            ON {self.table_name}(timestamp);
        """
        
        try:
            await conn.execute(create_sql)
            logger.info("table_created", table=self.table_name)
        except Exception as e:
            logger.warning("table_creation_warning", error=str(e))
    
    async def _insert_batch(self, conn: asyncpg.Connection, candles: List[Dict]) -> int:
        """Insert batch of candles"""
        
        insert_sql = f"""
        INSERT INTO {self.table_name}
        (symbol, timestamp, open, high, low, close, volume)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (symbol, timestamp)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
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
        
        result = await conn.executemany(insert_sql, data)
        return len(candles)
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
        logger.info("postgres_loader_closed")
