"""
Redis Cache Manager با Connection Pooling و Error Handling

مدیریت کامل caching برای بهبود performance
"""

import json
import asyncio
from typing import Optional, Any, Callable
from functools import wraps
import hashlib
import structlog
from redis import asyncio as aioredis
from redis.asyncio.connection import ConnectionPool

from config.settings import settings

logger = structlog.get_logger()


class CacheManager:
    """
    مدیریت Redis Cache با قابلیت‌های پیشرفته
    
    ویژگی‌ها:
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
        """راه‌اندازی اولیه Redis connection"""
        if not settings.cache_enabled:
            logger.info("cache_disabled")
            return
        
        try:
            # ایجاد connection pool
            self.connection_pool = ConnectionPool(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                max_connections=50,
                decode_responses=False,  # برای ذخیره binary data
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            
            # ایجاد Redis client
            self.redis = aioredis.Redis(connection_pool=self.connection_pool)
            
            # تست اتصال
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
            # در صورت خطا، ادامه می‌دهیم بدون cache
    
    async def get(self, key: str) -> Optional[Any]:
        """
        دریافت مقدار از cache
        
        Args:
            key: کلید cache
        
        Returns:
            مقدار deserialize شده یا None
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
        ذخیره مقدار در cache
        
        Args:
            key: کلید cache
            value: مقدار (باید JSON-serializable باشد)
            ttl: زمان انقضا (ثانیه)، اگر None باشد از settings استفاده می‌شود
        
        Returns:
            True در صورت موفقیت
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
        حذف کلید از cache
        
        Args:
            key: کلید cache
        
        Returns:
            True در صورت موفقیت
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
        حذف تمام کلیدهایی که با pattern مچ می‌شوند
        
        Args:
            pattern: الگوی Redis (مثلاً "user:*")
        
        Returns:
            تعداد کلیدهای حذف شده
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
        """بررسی وجود کلید در cache"""
        if not self._is_available or not self.redis:
            return False
        
        try:
            return bool(await self.redis.exists(key))
        except Exception as e:
            logger.warning("cache_exists_error", key=key, error=str(e))
            return False
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """دریافت TTL باقی‌مانده کلید (ثانیه)"""
        if not self._is_available or not self.redis:
            return None
        
        try:
            ttl = await self.redis.ttl(key)
            return ttl if ttl > 0 else None
        except Exception as e:
            logger.warning("cache_ttl_error", key=key, error=str(e))
            return None
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """افزایش counter"""
        if not self._is_available or not self.redis:
            return None
        
        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logger.warning("cache_increment_error", key=key, error=str(e))
            return None
    
    async def health_check(self) -> bool:
        """بررسی سلامت Redis connection"""
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
        """بستن اتصالات"""
        if self.redis:
            await self.redis.close()
        
        if self.connection_pool:
            await self.connection_pool.disconnect()
        
        logger.info("redis_connection_closed")


# Global instance
cache_manager = CacheManager()


def cache_key_generator(*args, **kwargs) -> str:
    """
    تولید کلید منحصر به فرد برای cache
    
    Args:
        *args: آرگومان‌های تابع
        **kwargs: کیلید ورد آرگومان‌ها
    
    Returns:
        کلید hash شده
    """
    # ترکیب تمام آرگومان‌ها
    key_parts = [str(arg) for arg in args]
    key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
    key_string = ":".join(key_parts)
    
    # Hash برای کوتاه‌تر کردن
    return hashlib.md5(key_string.encode()).hexdigest()


def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "cache",
    key_generator: Optional[Callable] = None
):
    """
    Decorator برای cache کردن خروجی توابع
    
    Args:
        ttl: زمان انقضا (ثانیه)
        key_prefix: پیشوند کلید cache
        key_generator: تابع برای تولید کلید (اختیاری)
    
    Example:
        >>> @cached(ttl=300, key_prefix="analysis")
        ... async def analyze_symbol(symbol: str, timeframe: str):
        ...     # محاسبات سنگین
        ...     return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # تولید کلید cache
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = cache_key_generator(*args, **kwargs)
            
            full_key = f"{key_prefix}:{func.__name__}:{cache_key}"
            
            # تلاش برای دریافت از cache با graceful degradation
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
            
            # اجرای تابع
            logger.debug(
                "function_cache_miss",
                function=func.__name__,
                key=full_key
            )
            result = await func(*args, **kwargs)
            
            # ذخیره در cache با graceful degradation
            try:
                await cache_manager.set(full_key, result, ttl=ttl)
            except Exception as e:
                logger.warning("cache_set_failed", error=str(e), key=full_key)
                # Continue even if cache set fails
            
            return result
        
        return wrapper
    return decorator
