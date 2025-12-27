"""
Base validator interface for pipeline

All validators implement this contract.
"""

from abc import ABC, abstractmethod
from typing import Any

import structlog

logger = structlog.get_logger()


class ValidationResult:
    """Result of validation"""

    def __init__(
        self,
        valid: bool,
        valid_count: int,
        invalid_count: int,
        errors: list[Any] | None = None,
    ) -> None:
        self.valid = valid
        self.valid_count = valid_count
        self.invalid_count = invalid_count
        self.errors = errors if errors is not None else []


class Validator(ABC):
    """Abstract base class for data validators"""

    def __init__(self) -> None:
        """Initialize validator"""
        self.checked_count = 0
        self.invalid_count = 0

    @abstractmethod
    async def validate(self, data: list[dict[str, Any]]) -> tuple[int, int]:
        """
        Validate data quality

        Args:
            data: List of data items to validate

        Returns:
            Tuple of (valid_count, invalid_count)
        """
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get validation statistics"""
        return {
            "checked": self.checked_count,
            "invalid": self.invalid_count,
        }
