"""
================================================================================
Infrastructure Contracts (Abstract Interfaces)
================================================================================

Abstract base classes for external service adapters.
These define the contract that implementations must follow.

Purpose: Decouple core logic from external service implementations.
Enables easy mocking for testing and swapping implementations.

Usage:
    # Define interface
    class CacheBackend(ABC):
        @abstractmethod
        async def get(self, key: str) -> Optional[Any]:
            pass

    # Implement for Redis
    class RedisCacheBackend(CacheBackend):
        async def get(self, key: str) -> Optional[Any]:
            return await self.redis.get(key)

    # Use in code
    def get_from_cache(cache: CacheBackend) -> Any:
        return await cache.get("key")
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class CacheBackend(ABC):
    """Abstract cache interface"""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize cache connection"""
        pass

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """
        Get value by key

        Args:
            key: Cache key

        Returns:
            Value if exists, None otherwise
        """
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """
        Set value with TTL

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete key from cache"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close cache connection"""
        pass


class DatabaseBackend(ABC):
    """Abstract database interface"""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize database connection"""
        pass

    @abstractmethod
    async def execute(self, query: str, params: dict | None = None) -> Any:
        """
        Execute query (INSERT, UPDATE, DELETE)

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Number of affected rows
        """
        pass

    @abstractmethod
    async def fetch_one(self, query: str, params: dict | None = None) -> dict | None:
        """
        Fetch single row

        Args:
            query: SQL SELECT query
            params: Query parameters

        Returns:
            Single row as dict or None
        """
        pass

    @abstractmethod
    async def fetch_all(self, query: str, params: dict | None = None) -> list[dict]:
        """
        Fetch all matching rows

        Args:
            query: SQL SELECT query
            params: Query parameters

        Returns:
            List of rows as dicts
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close database connection"""
        pass


class ExternalDataService(ABC):
    """Abstract external data service"""

    @abstractmethod
    async def get_candles(
        self, symbol: str, timeframe: str, start: datetime, end: datetime, adjusted: bool = True
    ) -> list[dict[str, Any]]:
        """
        Get OHLCV candles

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1m, 5m, 1h, 1d, 1w)
            start: Start datetime
            end: End datetime
            adjusted: Return adjusted prices

        Returns:
            List of candles as dicts with OHLCV
        """
        pass

    @abstractmethod
    async def get_metadata(self, symbol: str) -> dict[str, Any]:
        """Get symbol metadata"""
        pass


class EventPublisher(ABC):
    """Abstract event publishing interface"""

    @abstractmethod
    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        """
        Publish event

        Args:
            event_type: Type of event
            payload: Event data
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close publisher connection"""
        pass


class Logger(ABC):
    """Abstract logging interface"""

    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Log info level"""
        pass

    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """Log warning level"""
        pass

    @abstractmethod
    def error(self, message: str, **kwargs) -> None:
        """Log error level"""
        pass

    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """Log debug level"""
        pass


class MetricsCollector(ABC):
    """Abstract metrics interface"""

    @abstractmethod
    def increment_counter(self, name: str, value: int = 1, labels: dict | None = None) -> None:
        """Increment counter metric"""
        pass

    @abstractmethod
    def record_histogram(self, name: str, value: float, labels: dict | None = None) -> None:
        """Record histogram metric"""
        pass

    @abstractmethod
    def set_gauge(self, name: str, value: float, labels: dict | None = None) -> None:
        """Set gauge metric"""
        pass
