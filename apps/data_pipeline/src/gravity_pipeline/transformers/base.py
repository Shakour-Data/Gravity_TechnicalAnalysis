"""
Base transformer interface for pipeline

All transformers implement this contract.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import structlog

logger = structlog.get_logger()


class Transformer(ABC):
    """Abstract base class for data transformers"""
    
    def __init__(self):
        """Initialize transformer"""
        self.processed_count = 0
        self.error_count = 0
    
    @abstractmethod
    async def transform(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform data
        
        Args:
            data: List of raw data items
            
        Returns:
            List of transformed data items
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transformation statistics"""
        return {
            "processed": self.processed_count,
            "errors": self.error_count,
        }
