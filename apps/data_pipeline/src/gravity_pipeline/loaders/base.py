"""
Base loader interface for pipeline

All loaders (SQLite, PostgreSQL, etc) implement this contract.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import structlog

logger = structlog.get_logger()


class Loader(ABC):
    """Abstract base class for data loaders"""
    
    def __init__(self):
        """Initialize loader"""
        self.loaded_count = 0
        self.error_count = 0
        self.batch_size = 500
    
    @abstractmethod
    async def load(self, candles: List[Dict[str, Any]]) -> int:
        """
        Load candles into database
        
        Args:
            candles: List of candle dictionaries
            
        Returns:
            Number of successfully loaded candles
        """
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """
        Test connection to database
        
        Returns:
            True if connection valid, False otherwise
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics"""
        return {
            "loaded": self.loaded_count,
            "errors": self.error_count,
        }
    
    async def close(self):
        """Cleanup resources (override if needed)"""
        logger.info("loader_closed", type=self.__class__.__name__)
