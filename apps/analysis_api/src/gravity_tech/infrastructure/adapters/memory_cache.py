"""
================================================================================
In-Memory Cache Adapter (for testing and development)
================================================================================

Simple in-memory cache implementation.
Useful for:
- Unit tests (no dependencies)
- Development without Redis
- Integration tests with isolated state

No external dependencies, pure Python.
"""

from datetime import datetime, timedelta
from typing import Any

import structlog

logger = structlog.get_logger()


class MemoryCacheAdapter:
    """In-memory cache implementation"""

    def __init__(self, default_ttl: int = 300) -> None:
        """
        Initialize memory cache

        Args:
            default_ttl: Default TTL in seconds
        """
        self.data: dict[str, Any] = {}
        self.ttls: dict[str, datetime] = {}
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
        logger.info("memory_cache_created", default_ttl=default_ttl)

    async def initialize(self) -> None:
        """No-op for memory cache"""
        logger.info("memory_cache_initialized")

    async def get(self, key: str) -> Any | None:
        """Get value from cache"""
        # Check if key exists
        if key not in self.data:
            self.misses += 1
            logger.debug("cache_miss", key=key)
            return None

        # Check TTL
        if key in self.ttls:
            if datetime.now() > self.ttls[key]:
                # Expired
                del self.data[key]
                del self.ttls[key]
                self.misses += 1
                logger.debug("cache_expired", key=key)
                return None

        # Hit
        self.hits += 1
        logger.debug("cache_hit", key=key)
        return self.data[key]

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache"""
        ttl = self.default_ttl if ttl is None else ttl
        self.data[key] = value
        if ttl <= 0:
            self.ttls[key] = datetime.now()
        else:
            self.ttls[key] = datetime.now() + timedelta(seconds=ttl)
        logger.debug("cache_set", key=key, ttl=ttl)

    async def delete(self, key: str) -> None:
        """Delete key from cache"""
        if key in self.data:
            del self.data[key]
            if key in self.ttls:
                del self.ttls[key]
            logger.debug("cache_deleted", key=key)

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        if key not in self.data:
            return False

        # Check TTL
        if key in self.ttls:
            if datetime.now() > self.ttls[key]:
                await self.delete(key)
                return False

        return True

    async def clear(self) -> None:
        """Clear all cache"""
        self.data.clear()
        self.ttls.clear()
        logger.info("cache_cleared")

    async def close(self) -> None:
        """Close cache"""
        await self.clear()
        logger.info("memory_cache_closed")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {
            "type": "memory",
            "items": len(self.data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "total_requests": total,
        }
