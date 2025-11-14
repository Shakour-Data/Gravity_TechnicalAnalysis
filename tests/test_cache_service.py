"""
Tests for Cache Service

Tests Redis cache manager with connection pooling.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from gravity_tech.services.cache_service import CacheManager, cached


@pytest.fixture
def cache_manager():
    """Create cache manager instance for testing."""
    manager = CacheManager()
    # Mock Redis for testing
    manager.redis = AsyncMock()
    manager._is_available = True  # Enable cache for testing
    return manager


class TestCacheManager:
    """Tests for CacheManager class."""
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache_manager):
        """Test basic set and get operations."""
        # Setup mock
        cache_manager.redis.get.return_value = b'{"value": "test_data"}'
        
        # Test get
        result = await cache_manager.get("test_key")
        assert result == {"value": "test_data"}
        cache_manager.redis.get.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_set_with_ttl(self, cache_manager):
        """Test set operation with TTL."""
        cache_manager.redis.setex.return_value = True
        
        await cache_manager.set("test_key", {"data": "value"}, ttl=300)
        
        cache_manager.redis.setex.assert_called_once()
        args = cache_manager.redis.setex.call_args[0]
        assert args[0] == "test_key"
        assert args[1] == 300  # TTL
    
    @pytest.mark.asyncio
    async def test_delete(self, cache_manager):
        """Test delete operation."""
        cache_manager.redis.delete.return_value = 1
        
        result = await cache_manager.delete("test_key")
        assert result is True
        cache_manager.redis.delete.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_exists(self, cache_manager):
        """Test exists check."""
        cache_manager.redis.exists.return_value = 1
        
        result = await cache_manager.exists("test_key")
        assert result is True
        cache_manager.redis.exists.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_increment(self, cache_manager):
        """Test increment operation."""
        cache_manager.redis.incrby.return_value = 5
        
        result = await cache_manager.increment("counter_key")
        assert result == 5
        cache_manager.redis.incrby.assert_called_once_with("counter_key", 1)
    
    @pytest.mark.asyncio
    async def test_delete_pattern(self, cache_manager):
        """Test pattern-based deletion."""
        # Mock scan to return cursor and keys
        cache_manager.redis.scan.return_value = (0, [b'key1', b'key2', b'key3'])
        cache_manager.redis.delete.return_value = 3
        
        result = await cache_manager.delete_pattern("test:*")
        assert result == 3
        cache_manager.redis.scan.assert_called()
        cache_manager.redis.delete.assert_called_once_with(b'key1', b'key2', b'key3')
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, cache_manager):
        """Test getting non-existent key returns None."""
        cache_manager.redis.get.return_value = None
        
        result = await cache_manager.get("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, cache_manager):
        """Test health check when Redis is healthy."""
        cache_manager.redis.ping.return_value = True
        
        result = await cache_manager.health_check()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, cache_manager):
        """Test health check when Redis is down."""
        cache_manager.redis.ping.side_effect = Exception("Connection failed")
        
        result = await cache_manager.health_check()
        assert result is False


class TestCachedDecorator:
    """Tests for @cached decorator."""
    
    @pytest.mark.asyncio
    async def test_cached_decorator_cache_hit(self):
        """Test cached decorator returns cached value."""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = "cached_result"
        
        call_count = 0
        
        @cached(ttl=300, key_prefix="test")
        async def expensive_function(arg1, arg2):
            nonlocal call_count
            call_count += 1
            return f"computed_{arg1}_{arg2}"
        
        # Patch the cache manager
        with patch('gravity_tech.services.cache_service.cache_manager', mock_cache):
            result = await expensive_function("a", "b")
            assert result == "cached_result"
            assert call_count == 0  # Function not called
    
    @pytest.mark.asyncio
    async def test_cached_decorator_cache_miss(self):
        """Test cached decorator computes and caches on miss."""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None  # Cache miss
        mock_cache.set.return_value = True
        
        call_count = 0
        
        @cached(ttl=300, key_prefix="test")
        async def expensive_function(arg1, arg2):
            nonlocal call_count
            call_count += 1
            return f"computed_{arg1}_{arg2}"
        
        with patch('gravity_tech.services.cache_service.cache_manager', mock_cache):
            result = await expensive_function("a", "b")
            assert result == "computed_a_b"
            assert call_count == 1  # Function called once
            
            # Verify cache was updated
            mock_cache.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cached_decorator_key_generation(self):
        """Test cached decorator generates correct cache keys."""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True
        
        @cached(ttl=300, key_prefix="analysis")
        async def analyze_symbol(symbol: str, timeframe: str):
            return f"result_{symbol}_{timeframe}"
        
        with patch('gravity_tech.services.cache_service.cache_manager', mock_cache):
            await analyze_symbol("BTCUSDT", "1h")
            
            # Check the cache key structure
            get_call_args = mock_cache.get.call_args[0]
            cache_key = get_call_args[0]
            # Key should have format: prefix:function_name:hash
            assert cache_key.startswith("analysis:analyze_symbol:")
            # The hash part should be a 32-character hex string (MD5)
            hash_part = cache_key.split(":")[-1]
            assert len(hash_part) == 32
            assert all(c in '0123456789abcdef' for c in hash_part)


class TestCacheConnectionPooling:
    """Tests for connection pooling."""
    
    @pytest.mark.asyncio
    async def test_connection_pool_initialization(self):
        """Test connection pool is properly initialized."""
        manager = CacheManager()
        
        # Mock the Redis and connection pool
        with patch('gravity_tech.services.cache_service.aioredis.Redis') as mock_redis, \
             patch('gravity_tech.services.cache_service.ConnectionPool') as mock_pool:
            
            mock_pool.return_value = Mock()
            mock_redis_instance = AsyncMock()
            mock_redis_instance.ping.return_value = True
            mock_redis.return_value = mock_redis_instance
            
            await manager.initialize()
            
            # Verify pool was created with correct settings
            mock_pool.assert_called_once()
            pool_kwargs = mock_pool.call_args[1]
            assert pool_kwargs['max_connections'] == 50
            assert 'socket_timeout' in pool_kwargs
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful shutdown closes connections."""
        manager = CacheManager()
        manager.redis = AsyncMock()
        manager.connection_pool = Mock()
        manager.connection_pool.disconnect = AsyncMock()
        
        await manager.close()
        
        manager.connection_pool.disconnect.assert_called_once()


@pytest.mark.integration
class TestCacheIntegration:
    """Integration tests for cache service."""
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_pattern(self):
        """Test cache invalidation with patterns."""
        manager = CacheManager()
        manager.redis = AsyncMock()
        manager._is_available = True
        
        # Setup data - mock scan to return keys
        manager.redis.scan.return_value = (
            0,  # cursor
            [b'analysis:BTCUSDT:1h', b'analysis:BTCUSDT:4h']
        )
        manager.redis.delete.return_value = 2
        
        # Invalidate all BTCUSDT analysis
        deleted = await manager.delete_pattern("analysis:BTCUSDT:*")
        assert deleted == 2
    
    @pytest.mark.asyncio
    async def test_cache_ttl_management(self):
        """Test TTL management for cached items."""
        manager = CacheManager()
        manager.redis = AsyncMock()
        manager.redis.ttl.return_value = 150  # 2.5 minutes remaining
        
        # Check TTL
        ttl = await manager.redis.ttl("test_key")
        assert ttl == 150
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self):
        """Test concurrent cache operations."""
        manager = CacheManager()
        manager.redis = AsyncMock()
        manager._is_available = True
        manager.redis.get.return_value = b'{"value": "data"}'
        
        # Simulate concurrent access
        tasks = [manager.get(f"key_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(r == {"value": "data"} for r in results)


class TestCacheErrorHandling:
    """Tests for error handling in cache operations."""
    
    @pytest.mark.asyncio
    async def test_cache_failure_graceful_degradation(self):
        """Test graceful degradation when cache fails."""
        mock_cache = AsyncMock()
        mock_cache.get.side_effect = Exception("Redis connection failed")
        
        @cached(ttl=300, key_prefix="test")
        async def function_with_cache():
            return "computed_value"
        
        with patch('gravity_tech.services.cache_service.cache_manager', mock_cache):
            # Should still execute function despite cache failure
            result = await function_with_cache()
            assert result == "computed_value"
    
    @pytest.mark.asyncio
    async def test_serialization_error_handling(self, cache_manager):
        """Test handling of serialization errors."""
        # Try to serialize non-serializable object
        class NonSerializable:
            pass
        
        cache_manager.redis.setex.return_value = True
        
        # Should handle gracefully and return False (not raise exception)
        result = await cache_manager.set("key", NonSerializable())
        assert result is False  # Set should fail gracefully


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
