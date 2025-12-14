"""
Unit tests for src/gravity_tech/services/cache_service.py

Tests cache management functionality.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.gravity_tech.services.cache_service import CacheManager, CacheWarmer, cache_key_generator


class TestCacheManager:
    """Test CacheManager functionality."""

    @pytest.fixture
    async def cache_manager(self):
        """Create a cache manager instance."""
        manager = CacheManager()
        yield manager
        # Cleanup if needed

    def test_init(self):
        """Test CacheManager initialization."""
        manager = CacheManager()

        assert manager.redis is None
        assert manager.connection_pool is None
        assert manager._is_available is False

    @patch('redis.asyncio.from_url')
    async def test_initialize_success(self, mock_from_url):
        """Test successful cache initialization."""
        mock_redis = AsyncMock()
        mock_from_url.return_value = mock_redis

        manager = CacheManager()
        result = await manager.initialize()

        assert result is True
        assert manager.redis == mock_redis
        assert manager._is_available is True
        mock_from_url.assert_called_once()

    @patch('redis.asyncio.from_url')
    async def test_initialize_failure(self, mock_from_url):
        """Test cache initialization failure."""
        mock_from_url.side_effect = Exception("Connection failed")

        manager = CacheManager()
        result = await manager.initialize()

        assert result is False
        assert manager.redis is None
        assert manager._is_available is False

    async def test_get_without_initialization(self):
        """Test get operation without initialization."""
        manager = CacheManager()

        result = await manager.get("test_key")
        assert result is None

    async def test_set_without_initialization(self):
        """Test set operation without initialization."""
        manager = CacheManager()

        result = await manager.set("test_key", "value")
        assert result is False

    @patch('redis.asyncio.from_url')
    async def test_get_success(self, mock_from_url):
        """Test successful get operation."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = b'{"data": "value"}'
        mock_from_url.return_value = mock_redis

        manager = CacheManager()
        await manager.initialize()

        result = await manager.get("test_key")

        assert result == {"data": "value"}
        mock_redis.get.assert_called_once_with("test_key")

    @patch('redis.asyncio.from_url')
    async def test_get_none(self, mock_from_url):
        """Test get operation returning None."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_from_url.return_value = mock_redis

        manager = CacheManager()
        await manager.initialize()

        result = await manager.get("test_key")

        assert result is None

    @patch('redis.asyncio.from_url')
    async def test_set_success(self, mock_from_url):
        """Test successful set operation."""
        mock_redis = AsyncMock()
        mock_redis.setex.return_value = True
        mock_from_url.return_value = mock_redis

        manager = CacheManager()
        await manager.initialize()

        result = await manager.set("test_key", {"data": "value"}, ttl=300)

        assert result is True
        mock_redis.setex.assert_called_once_with("test_key", 300, '{"data": "value"}')

    @patch('redis.asyncio.from_url')
    async def test_delete_success(self, mock_from_url):
        """Test successful delete operation."""
        mock_redis = AsyncMock()
        mock_redis.delete.return_value = 1
        mock_from_url.return_value = mock_redis

        manager = CacheManager()
        await manager.initialize()

        result = await manager.delete("test_key")

        assert result is True
        mock_redis.delete.assert_called_once_with("test_key")

    @patch('redis.asyncio.from_url')
    async def test_exists_success(self, mock_from_url):
        """Test successful exists operation."""
        mock_redis = AsyncMock()
        mock_redis.exists.return_value = 1
        mock_from_url.return_value = mock_redis

        manager = CacheManager()
        await manager.initialize()

        result = await manager.exists("test_key")

        assert result is True
        mock_redis.exists.assert_called_once_with("test_key")

    @patch('redis.asyncio.from_url')
    async def test_health_check_success(self, mock_from_url):
        """Test successful health check."""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_from_url.return_value = mock_redis

        manager = CacheManager()
        await manager.initialize()

        result = await manager.health_check()

        assert result is True
        mock_redis.ping.assert_called_once()

    @patch('redis.asyncio.from_url')
    async def test_health_check_failure(self, mock_from_url):
        """Test health check failure."""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Connection error")
        mock_from_url.return_value = mock_redis

        manager = CacheManager()
        await manager.initialize()

        result = await manager.health_check()

        assert result is False

    def test_cache_key_generator(self):
        """Test cache key generation."""
        key = cache_key_generator("prefix", "symbol", "timeframe", param1="value1", param2=42)

        assert isinstance(key, str)
        assert "prefix" in key
        assert "symbol" in key
        assert "timeframe" in key

    @patch('redis.asyncio.from_url')
    async def test_close(self, mock_from_url):
        """Test cache manager close operation."""
        mock_redis = AsyncMock()
        mock_from_url.return_value = mock_redis

        manager = CacheManager()
        await manager.initialize()

        await manager.close()

        mock_redis.close.assert_called_once()
        assert manager.redis is None
        assert manager._is_available is False


class TestCacheWarmer:
    """Test CacheWarmer functionality."""

    @pytest.fixture
    async def cache_warmer(self):
        """Create a cache warmer instance."""
        mock_cache = AsyncMock()
        warmer = CacheWarmer(mock_cache)
        return warmer

    async def test_init(self):
        """Test CacheWarmer initialization."""
        mock_cache = AsyncMock()
        warmer = CacheWarmer(mock_cache)

        assert warmer.cache == mock_cache

    async def test_warm_common_symbols_default(self):
        """Test warming cache with default symbols."""
        mock_cache = AsyncMock()
        warmer = CacheWarmer(mock_cache)

        await warmer.warm_common_symbols()

        # Should call _warm_symbol_cache for each default symbol
        assert mock_cache.set.call_count > 0
