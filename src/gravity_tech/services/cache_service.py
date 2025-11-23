"""
Redis Cache Manager with Connection Pooling and Error Handling

Complete caching management for performance improvement.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import json
import asyncio
from typing import Optional, Any, Callable
from functools import wraps
import hashlib
import structlog
from redis import asyncio as aioredis
from redis.asyncio.connection import ConnectionPool

from gravity_tech.config.settings import settings

logger = structlog.get_logger()


class CacheManager:
    """
    Redis Cache Manager with advanced features.
    
    Features:
    - Connection pooling
    - Auto retry
    - Error handling
    - Serialization/Deserialization
    - TTL management
    - Cache invalidation
    
    Example:
        >>> cache = CacheManager()
        >>> await cache.initialize()
        >>> await cache.set("key", {"data": "value"}, ttl=300)
        >>> data = await cache.get("key")
    """
    
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.connection_pool: Optional[ConnectionPool] = None
        self._is_available = False
    
    async def initialize(self):
        """Initialize Redis connection."""
        if not settings.cache_enabled:
            logger.info("cache_disabled")
            return
        
        try:
            # Create connection pool
            self.connection_pool = ConnectionPool(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                max_connections=50,
                decode_responses=False,  # For storing binary data
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            
            # Create Redis client
            self.redis = aioredis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            await self.redis.ping()
            self._is_available = True
            
            logger.info(
                "redis_cache_initialized",
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db
            )
        
        except Exception as e:
            logger.error("redis_initialization_failed", error=str(e))
            self._is_available = False
            # In case of error, continue without cache
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Deserialized value or None
        """
        if not self._is_available or not self.redis:
            return None
        
        try:
            value = await self.redis.get(key)
            
            if value is None:
                logger.debug("cache_miss", key=key)
                return None
            
            logger.debug("cache_hit", key=key)
            return json.loads(value.decode('utf-8'))
        
        except Exception as e:
            logger.warning("cache_get_error", key=key, error=str(e))
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value (must be JSON-serializable)
            ttl: Expiration time (seconds), uses settings if None
        
        Returns:
            True on success
        """
        if not self._is_available or not self.redis:
            return False
        
        try:
            serialized = json.dumps(value).encode('utf-8')
            ttl = ttl or settings.cache_ttl
            
            await self.redis.setex(key, ttl, serialized)
            logger.debug("cache_set", key=key, ttl=ttl)
            return True
        
        except Exception as e:
            logger.warning("cache_set_error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True on success
        """
        if not self._is_available or not self.redis:
            return False
        
        try:
            await self.redis.delete(key)
            logger.debug("cache_deleted", key=key)
            return True
        
        except Exception as e:
            logger.warning("cache_delete_error", key=key, error=str(e))
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching the pattern.
        
        Args:
            pattern: Redis pattern (e.g., "user:*")
        
        Returns:
            Number of deleted keys
        """
        if not self._is_available or not self.redis:
            return 0
        
        try:
            cursor = 0
            deleted_count = 0
            
            while True:
                cursor, keys = await self.redis.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                
                if keys:
                    await self.redis.delete(*keys)
                    deleted_count += len(keys)
                
                if cursor == 0:
                    break
            
            logger.info("cache_pattern_deleted", pattern=pattern, count=deleted_count)
            return deleted_count
        
        except Exception as e:
            logger.error("cache_pattern_delete_error", pattern=pattern, error=str(e))
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self._is_available or not self.redis:
            return False
        
        try:
            return bool(await self.redis.exists(key))
        except Exception as e:
            logger.warning("cache_exists_error", key=key, error=str(e))
            return False
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL of key (seconds)."""
        if not self._is_available or not self.redis:
            return None
        
        try:
            ttl = await self.redis.ttl(key)
            return ttl if ttl > 0 else None
        except Exception as e:
            logger.warning("cache_ttl_error", key=key, error=str(e))
            return None
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter."""
        if not self._is_available or not self.redis:
            return None
        
        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logger.warning("cache_increment_error", key=key, error=str(e))
            return None
    
    async def health_check(self) -> bool:
        """Check Redis connection health."""
        if not self._is_available or not self.redis:
            return False
        
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            logger.error("redis_health_check_failed", error=str(e))
            self._is_available = False
            return False
    
    async def close(self):
        """Close connections."""
        if self.redis:
            await self.redis.close()
        
        if self.connection_pool:
            await self.connection_pool.disconnect()
        
        logger.info("redis_connection_closed")


# Global instance
cache_manager = CacheManager()


def cache_key_generator(*args, **kwargs) -> str:
    """
    Generate unique key for cache.
    
    Args:
        *args: Function arguments
        **kwargs: Keyword arguments
    
    Returns:
        Hashed key
    """
    # Combine all arguments
    key_parts = [str(arg) for arg in args]
    key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
    key_string = ":".join(key_parts)
    
    # Hash to shorten
    return hashlib.md5(key_string.encode()).hexdigest()


def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "cache",
    key_generator: Optional[Callable] = None
):
    """
    Decorator for caching function output.
    
    Args:
        ttl: Expiration time (seconds)
        key_prefix: Cache key prefix
        key_generator: Function for generating key (optional)
    
    Example:
        >>> @cached(ttl=300, key_prefix="analysis")
        ... async def analyze_symbol(symbol: str, timeframe: str):
        ...     # Heavy computations
        ...     return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = cache_key_generator(*args, **kwargs)
            
            full_key = f"{key_prefix}:{func.__name__}:{cache_key}"
            
            # Try to get from cache with graceful degradation
            try:
                cached_result = await cache_manager.get(full_key)
                if cached_result is not None:
                    logger.debug(
                        "function_cache_hit",
                        function=func.__name__,
                        key=full_key
                    )
                    return cached_result
            except Exception as e:
                logger.warning("cache_get_failed", error=str(e), key=full_key)
                # Continue to execute function even if cache fails
            
            # Execute function
            logger.debug(
                "function_cache_miss",
                function=func.__name__,
                key=full_key
            )
            result = await func(*args, **kwargs)
            
            # Save in cache with graceful degradation
            try:
                await cache_manager.set(full_key, result, ttl=ttl)
            except Exception as e:
                logger.warning("cache_set_failed", error=str(e), key=full_key)
                # Continue even if cache set fails
            
            return result
        
        return wrapper
    return decorator


class CacheWarmer:
    """
    Cache warming strategies for improved performance
    
    Pre-loads frequently accessed data into cache
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.warming_tasks = []
    
    async def warm_common_symbols(self, symbols: list = None):
        """
        Warm cache with common trading symbols
        
        Args:
            symbols: List of symbols to warm (default: major crypto pairs)
        """
        if symbols is None:
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
        
        logger.info("starting_cache_warming", symbols=symbols)
        
        for symbol in symbols:
            # Create warming task for each symbol
            task = asyncio.create_task(self._warm_symbol_cache(symbol))
            self.warming_tasks.append(task)
        
        # Wait for all warming tasks to complete
        await asyncio.gather(*self.warming_tasks, return_exceptions=True)
        logger.info("cache_warming_completed")
    
    async def _warm_symbol_cache(self, symbol: str):
        """
        Warm cache for a specific symbol with common timeframes
        """
        timeframes = ["1h", "4h", "1d", "1w"]
        
        for timeframe in timeframes:
            try:
                # Create cache key for historical data
                cache_key = f"historical:{symbol}:{timeframe}:recent"
                
                # Check if already in cache
                existing = await self.cache.get(cache_key)
                if existing is None:
                    # Placeholder for actual data loading logic
                    # In real implementation, this would load from database
                    placeholder_data = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "status": "warming_placeholder",
                        "last_updated": str(asyncio.get_event_loop().time())
                    }
                    
                    await self.cache.set(cache_key, placeholder_data, ttl=3600)  # 1 hour TTL
                    logger.debug("warmed_cache_key", key=cache_key)
                
            except Exception as e:
                logger.warning("cache_warming_failed", symbol=symbol, timeframe=timeframe, error=str(e))
    
    async def warm_ml_models(self):
        """
        Warm cache with ML model metadata
        """
        try:
            model_keys = [
                "ml:model:pattern_classifier:info",
                "ml:model:multi_horizon:info",
                "ml:model:weights:latest"
            ]
            
            for key in model_keys:
                existing = await self.cache.get(key)
                if existing is None:
                    # Placeholder metadata
                    metadata = {
                        "model_type": key.split(":")[2],
                        "status": "available",
                        "last_loaded": str(asyncio.get_event_loop().time())
                    }
                    await self.cache.set(key, metadata, ttl=7200)  # 2 hours TTL
            
            logger.info("ml_model_cache_warmed")
            
        except Exception as e:
            logger.warning("ml_model_cache_warming_failed", error=str(e))


# Global instances
cache_manager = CacheManager()
cache_warmer = CacheWarmer(cache_manager)
