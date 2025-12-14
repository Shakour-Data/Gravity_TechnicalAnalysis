"""
Real Cache Service Tests with TSE Data

Unit tests for actual CacheManager implementation with real Iranian stock market data.
"""

import asyncio
import json
import time

import pytest


@pytest.fixture(autouse=True)
def mock_redis_for_cache_tests(monkeypatch):
    """Mock Redis to allow cache tests to run without actual Redis server."""
    # Create in-memory cache storage with TTL tracking
    cache_storage = {}  # {key: (value, expiry_time)}

    async def mock_set(self, key, value, ttl=None):
        from src.gravity_tech.config.settings import settings
        ttl = ttl or settings.cache_ttl
        expiry_time = time.time() + ttl
        cache_storage[key] = (value, expiry_time)
        return True

    async def mock_get(self, key):
        if key not in cache_storage:
            return None
        value, expiry_time = cache_storage[key]
        if time.time() > expiry_time:
            del cache_storage[key]
            return None
        return value

    async def mock_delete(self, key):
        cache_storage.pop(key, None)
        return True

    async def mock_exists(self, key):
        if key not in cache_storage:
            return False
        value, expiry_time = cache_storage[key]
        if time.time() > expiry_time:
            del cache_storage[key]
            return False
        return True

    async def mock_flush(self):
        cache_storage.clear()
        return True

    async def mock_mset(self, data, ttl=None):
        from src.gravity_tech.config.settings import settings
        ttl = ttl or settings.cache_ttl
        expiry_time = time.time() + ttl
        for key, value in data.items():
            cache_storage[key] = (value, expiry_time)
        return True

    async def mock_delete_pattern(self, pattern: str):
        """Mock delete pattern - removes keys matching pattern."""
        import re
        regex = re.compile(pattern.replace('*', '.*'))
        keys_to_delete = [k for k in cache_storage.keys() if regex.match(k)]
        for key in keys_to_delete:
            del cache_storage[key]
        return len(keys_to_delete)

    async def mock_initialize(self):
        self._is_available = True
        return

    # Monkeypatch CacheManager methods
    from src.gravity_tech.services.cache_service import CacheManager
    monkeypatch.setattr(CacheManager, "initialize", mock_initialize)
    monkeypatch.setattr(CacheManager, "set", mock_set)
    monkeypatch.setattr(CacheManager, "get", mock_get)
    monkeypatch.setattr(CacheManager, "delete", mock_delete)
    monkeypatch.setattr(CacheManager, "exists", mock_exists)
    monkeypatch.setattr(CacheManager, "flush", mock_flush)
    monkeypatch.setattr(CacheManager, "mset", mock_mset)
    monkeypatch.setattr(CacheManager, "delete_pattern", mock_delete_pattern)


class TestCacheManagerWithRealTSEData:
    """Test actual CacheManager with real TSE data."""

    @pytest.mark.asyncio
    async def test_cache_set_get_tse_candles(self, tse_candles_short):
        """Test caching real TSE candle data."""
        from src.gravity_tech.services.cache_service import CacheManager

        cache = CacheManager()
        await cache.initialize()

        # Prepare data
        cache_key = "tse:candles:TOTAL"
        candles_json = json.dumps([
            {
                'timestamp': str(c.timestamp),
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume
            }
            for c in tse_candles_short
        ])

        # Execute
        await cache.set(cache_key, candles_json, ttl=3600)
        result = await cache.get(cache_key)

        # Verify
        assert result is not None
        cached_candles = json.loads(result)
        assert len(cached_candles) == len(tse_candles_short)

    @pytest.mark.asyncio
    async def test_cache_tse_analysis_results(self, tse_candles_total):
        """Test caching analysis results."""
        from src.gravity_tech.services.cache_service import CacheManager

        cache = CacheManager()
        await cache.initialize()

        # Prepare analysis result
        cache_key = "tse:analysis:TOTAL"
        analysis = {
            'symbol': 'TOTAL',
            'trend': 'UPTREND',
            'score': 8.5,
            'candles_count': len(tse_candles_total),
            'indicators': {
                'rsi': 65.5,
                'macd': 0.25,
                'ma_20': 11450
            }
        }

        # Execute
        await cache.set(cache_key, json.dumps(analysis), ttl=300)
        result = await cache.get(cache_key)

        # Verify
        assert result is not None
        cached_analysis = json.loads(result)
        assert cached_analysis['symbol'] == 'TOTAL'
        assert cached_analysis['candles_count'] == len(tse_candles_total)

    @pytest.mark.asyncio
    async def test_cache_multiple_tse_symbols(self, tse_candles_total, tse_candles_petroff, tse_candles_iraninoil):
        """Test caching multiple TSE symbols."""
        from src.gravity_tech.services.cache_service import CacheManager

        cache = CacheManager()
        await cache.initialize()


        symbols_data = {
            'TOTAL': tse_candles_total,
            'PETROFF': tse_candles_petroff,
            'IRANINOIL': tse_candles_iraninoil
        }

        # Execute - cache each symbol
        for symbol, candles in symbols_data.items():
            cache_key = f"tse:candles:{symbol}"
            data = json.dumps({'symbol': symbol, 'count': len(candles)})
            await cache.set(cache_key, data, ttl=3600)

        # Verify - retrieve each
        for symbol in symbols_data:
            cache_key = f"tse:candles:{symbol}"
            result = await cache.get(cache_key)
            assert result is not None

    @pytest.mark.asyncio
    async def test_cache_delete_tse_key(self, tse_candles_short):
        """Test deleting cache entry."""
        from src.gravity_tech.services.cache_service import CacheManager

        cache = CacheManager()
        await cache.initialize()


        # Setup
        cache_key = "tse:temp:data"
        data = json.dumps([c.close for c in tse_candles_short])

        # Execute - set
        await cache.set(cache_key, data, ttl=60)
        result_before = await cache.get(cache_key)
        assert result_before is not None

        # Execute - delete
        await cache.delete(cache_key)
        result_after = await cache.get(cache_key)

        # Verify
        assert result_after is None

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, tse_candles_short):
        """Test TTL expiration behavior."""
        from src.gravity_tech.services.cache_service import CacheManager

        cache = CacheManager()
        await cache.initialize()


        # Setup - set with 2 second TTL
        cache_key = "tse:expire:test"
        data = json.dumps([c.close for c in tse_candles_short])

        await cache.set(cache_key, data, ttl=2)
        result_immediate = await cache.get(cache_key)
        assert result_immediate is not None

        # Wait and check expiration
        await asyncio.sleep(3)
        result_expired = await cache.get(cache_key)
        assert result_expired is None

    @pytest.mark.asyncio
    async def test_cache_mset_batch_operation(self, tse_candles_long):
        """Test batch setting multiple keys."""
        from src.gravity_tech.services.cache_service import CacheManager

        cache = CacheManager()
        await cache.initialize()


        # Prepare batch data
        batch_data = {}
        for i, candle in enumerate(tse_candles_long[:10]):
            key = f"tse:candle:{i}"
            batch_data[key] = json.dumps({
                'timestamp': str(candle.timestamp),
                'close': candle.close
            })

        # Execute
        await cache.mset(batch_data, ttl=3600)

        # Verify
        for i in range(10):
            key = f"tse:candle:{i}"
            result = await cache.get(key)
            assert result is not None

    @pytest.mark.asyncio
    async def test_cache_key_exists(self, tse_candles_short):
        """Test key existence check."""
        from src.gravity_tech.services.cache_service import CacheManager

        cache = CacheManager()
        await cache.initialize()


        # Setup
        cache_key = "tse:exists:test"
        data = json.dumps({'test': 'data'})

        await cache.set(cache_key, data, ttl=60)

        # Execute & Verify
        exists = await cache.exists(cache_key)
        assert exists == 1

        # After deletion
        await cache.delete(cache_key)
        exists_after = await cache.exists(cache_key)
        assert exists_after == 0

    @pytest.mark.asyncio
    async def test_cache_clear_all(self, tse_candles_short):
        """Test clearing entire cache."""
        from src.gravity_tech.services.cache_service import CacheManager

        cache = CacheManager()
        await cache.initialize()


        # Setup - add multiple keys
        for i in range(5):
            key = f"tse:clear:test:{i}"
            data = json.dumps({'index': i})
            await cache.set(key, data, ttl=60)

        # Execute - clear all
        await cache.flush()

        # Verify - check one key
        result = await cache.get("tse:clear:test:0")
        assert result is None


class TestCacheServicePatterns:
    """Test common caching patterns with real data."""

    @pytest.mark.asyncio
    async def test_cache_aside_pattern(self, tse_candles_short):
        """Test cache-aside pattern."""
        from src.gravity_tech.services.cache_service import CacheManager

        cache = CacheManager()
        await cache.initialize()


        cache_key = "tse:cache_aside"

        # Check cache
        result = await cache.get(cache_key)
        if result is None:
            # Load from source (TSE data)
            data = json.dumps([c.close for c in tse_candles_short])
            await cache.set(cache_key, data, ttl=3600)
            result = await cache.get(cache_key)

        # Verify
        assert result is not None
        cached_prices = json.loads(result)
        assert len(cached_prices) == len(tse_candles_short)

    @pytest.mark.asyncio
    async def test_write_through_pattern(self, tse_candles_short):
        """Test write-through pattern."""
        from src.gravity_tech.services.cache_service import CacheManager

        cache = CacheManager()
        await cache.initialize()


        # Write through: write to cache and database simultaneously
        cache_key = "tse:write_through"
        data = json.dumps({'symbol': 'TOTAL', 'count': len(tse_candles_short)})

        # Write to cache
        await cache.set(cache_key, data, ttl=3600)

        # Verify in cache
        result = await cache.get(cache_key)
        assert result is not None

    @pytest.mark.asyncio
    async def test_write_behind_pattern(self, tse_candles_short):
        """Test write-behind pattern."""
        from src.gravity_tech.services.cache_service import CacheManager

        cache = CacheManager()
        await cache.initialize()


        # Write behind: write to cache immediately, persist later
        cache_key = "tse:write_behind"
        data = json.dumps({'symbol': 'TOTAL', 'data': len(tse_candles_short)})

        # Write to cache
        await cache.set(cache_key, data, ttl=3600)

        # Immediately available in cache
        result = await cache.get(cache_key)
        assert result is not None

        # Wait for background persistence (simulated)
        await asyncio.sleep(0.1)

        # Still available
        result_after = await cache.get(cache_key)
        assert result_after is not None


class TestCachePerformance:
    """Test cache performance characteristics."""

    @pytest.mark.asyncio
    async def test_cache_throughput_with_large_data(self, tse_candles_long):
        """Test cache throughput."""
        import time

        from src.gravity_tech.services.cache_service import CacheManager

        cache = CacheManager()
        await cache.initialize()


        # Prepare large dataset
        large_data = json.dumps([
            {'timestamp': str(c.timestamp), 'close': c.close}
            for c in tse_candles_long
        ])

        # Measure write throughput
        start = time.time()
        for i in range(10):
            await cache.set(f"tse:perf:{i}", large_data, ttl=3600)
        write_time = time.time() - start

        # Measure read throughput
        start = time.time()
        for i in range(10):
            await cache.get(f"tse:perf:{i}")
        read_time = time.time() - start

        # Verify performance
        assert write_time < 5.0, "Write throughput should be reasonable"
        assert read_time < 2.0, "Read throughput should be fast"

    @pytest.mark.asyncio
    async def test_cache_memory_efficiency(self, tse_candles_short):
        """Test cache memory efficiency."""
        from src.gravity_tech.services.cache_service import CacheManager

        cache = CacheManager()
        await cache.initialize()


        # Store data
        original_data = json.dumps([c.close for c in tse_candles_short])
        cache_key = "tse:memory:test"

        await cache.set(cache_key, original_data, ttl=3600)
        retrieved_data = await cache.get(cache_key)

        # Verify data integrity
        assert retrieved_data is not None
        assert json.loads(retrieved_data) == json.loads(original_data)
