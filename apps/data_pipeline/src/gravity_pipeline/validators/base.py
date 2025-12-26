"""
Base validator interface for pipeline

All validators implement this contract.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import structlog

logger = structlog.get_logger()


class ValidationResult:
    """Result of validation"""
    
    def __init__(self, valid: bool, valid_count: int, invalid_count: int, errors: list = None):
        self.valid = valid
        self.valid_count = valid_count
        self.invalid_count = invalid_count
        self.errors = errors or []


class Validator(ABC):
    """Abstract base class for data validators"""
    
    def __init__(self):
        """Initialize validator"""
        self.checked_count = 0
        self.invalid_count = 0
    
    @abstractmethod
    async def validate(self, data: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Validate data quality
        
        Args:
            data: List of data items to validate
            
        Returns:
            Tuple of (valid_count, invalid_count)
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            "checked": self.checked_count,
            "invalid": self.invalid_count,
        }
