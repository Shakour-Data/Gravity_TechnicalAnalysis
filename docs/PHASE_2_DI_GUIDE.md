# Phase 2: Dependency Injection & Testability - Implementation Guide

**Status:** ✅ COMPLETED  
**Date:** December 26, 2025  
**Owner:** Architecture Team

---

## 📋 What Was Implemented

### 1. **Service Container** 
**File:** `infrastructure/container.py`

- Centralized dependency management
- Singleton and transient service lifecycle
- Factory pattern for service creation
- Proper init/close lifecycle

**Key Functions:**
- `get_container()` - Get global container
- `container.register()` - Register service
- `container.get()` - Resolve service
- `container.close()` - Cleanup (async)

**Example:**
```python
container = get_container()
service = container.get("analysis_service")
result = await service.analyze(candles)
```

---

### 2. **Abstract Contracts (Interfaces)**
**File:** `infrastructure/contracts.py`

Define interfaces that implementations must follow:
- `CacheBackend` - Cache abstraction
- `DatabaseBackend` - Database abstraction  
- `ExternalDataService` - External data source
- `EventPublisher` - Event publishing
- `Logger` - Logging abstraction
- `MetricsCollector` - Metrics abstraction

**Benefits:**
- Enables easy mocking for tests
- Allows swapping implementations
- Documents expected behavior

---

### 3. **In-Memory Cache Adapter**
**File:** `infrastructure/adapters/memory_cache.py`

Pure Python cache implementation:
- No external dependencies
- TTL support
- Hit/miss tracking
- Statistics reporting

**Use Cases:**
- Unit tests
- Development without Redis
- Integration tests

---

### 4. **Container Factories**
**File:** `infrastructure/container_factories.py`

Factory functions for service creation:
- `create_cache_service()` - Redis or Memory fallback
- `create_database_service()` - PostgreSQL or SQLite
- `create_analysis_service()`
- `create_tool_recommendation_service()`
- `create_data_ingestor_service()`
- `create_event_publisher()` - Kafka/RabbitMQ with fallback

**Smart Fallbacks:**
```python
# Try Redis first, fall back to Memory
if settings.cache.backend == "redis":
    try:
        return RedisCacheAdapter(...)
    except:
        return MemoryCacheAdapter()  # Fallback
```

---

### 5. **Test Fixtures & Mocking**
**File:** `tests/conftest.py` (Enhanced)

**New Fixtures:**
- `mock_cache()` - AsyncMock for unit tests
- `memory_cache()` - Real in-memory cache
- `mock_database()` - AsyncMock for isolation
- `test_container()` - Pre-configured with mocks
- `isolated_container()` - Fresh container per test
- `reset_global_state()` - Autouse cleanup
- `sample_candles_new()` - Test data

**Example - Unit Test:**
```python
@pytest.mark.asyncio
async def test_analysis_service_no_cache(test_container, mock_cache):
    # Service without cache
    mock_cache.get.return_value = None
    
    service = test_container.get("analysis_service")
    result = await service.analyze(candles)
    
    assert result.signal == "BUY"
    mock_cache.set.assert_called_once()  # Cache was set
```

---

### 6. **Example DI Endpoint**
**File:** `api/v1/example_di.py`

Demonstrates proper DI pattern with FastAPI:

```python
@router.post("/analyze")
async def analyze(
    request: AnalysisRequest,
    service = Depends(get_analysis_service),
    cache = Depends(get_cache_service),
) -> TechnicalAnalysisResult:
    # Service is injected, not global
    result = await service.analyze(request.candles)
    await cache.set(key, result)
    return result
```

**Advantages:**
✅ Testable - inject mocks easily  
✅ Flexible - swap implementations  
✅ Clean - no hidden dependencies  
✅ Typed - full type hints  

---

## 🚀 How to Use

### For Developers

**1. Register a New Service:**
```python
# In container_factories.py
def create_my_service(container):
    cache = container.get("cache")
    return MyService(cache=cache)

# In container.py _setup_container()
container.register("my_service", create_my_service, singleton=True)
```

**2. Use Service in Endpoint:**
```python
@router.post("/my-endpoint")
async def my_endpoint(
    request: MyRequest,
    service = Depends(get_my_service),  # Injected!
):
    return await service.do_something(request)
```

**3. Test Service with Mock:**
```python
@pytest.mark.asyncio
async def test_my_service(test_container):
    service = test_container.get("my_service")
    result = await service.do_something(data)
    assert result is not None
```

### Configuration

Services respect `unified_settings.py`:

```python
# .env
CACHE_BACKEND=redis          # or memory
CACHE_HOST=localhost
DATABASE_ENGINE=postgresql   # or sqlite
DATABASE_URL=postgresql://...
```

---

## 🧪 Testing Examples

### Unit Test (No Dependencies)
```python
@pytest.mark.unit
async def test_indicator_calculation():
    from gravity_tech.core.indicators.trend import calculate_sma
    
    result = calculate_sma([1, 2, 3, 4, 5], period=2)
    assert result[-1] == 4.5  # No mocks needed
```

### Service Test (With Mocks)
```python
@pytest.mark.unit
async def test_analysis_service_with_mock_cache(test_container):
    service = test_container.get("analysis_service")
    result = await service.analyze(candles)
    
    assert result.signal in ["BUY", "SELL", "NEUTRAL"]
```

### Integration Test (Real Services)
```python
@pytest.mark.integration
async def test_analyze_endpoint_full_stack(client, memory_cache):
    response = client.post("/api/v1/analyze", json={
        "symbol": "BTCUSDT",
        "candles": sample_candles
    })
    
    assert response.status_code == 200
    assert "signal" in response.json()
```

### API Test (End-to-End)
```python
@pytest.mark.api
async def test_analyze_endpoint_e2e(client):
    response = client.post("/api/v1/analyze", json={
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "candles": sample_candles
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["signal"] in ["BUY", "SELL", "NEUTRAL"]
    assert 0 <= data["confidence"] <= 100
```

---

## ✅ Checklist: Migrating Existing Code

- [ ] Create contract interface for your service
- [ ] Implement adapter
- [ ] Create factory function
- [ ] Register in `_setup_container()`
- [ ] Update endpoint to use `Depends()`
- [ ] Remove global singleton imports
- [ ] Add tests with mock
- [ ] Test with real service
- [ ] Delete old global code

**Example Migration:**

**Before:**
```python
# ❌ Anti-pattern: Global singleton
from services import analysis_service

@router.post("/analyze")
async def analyze(request):
    return analysis_service.analyze(request.candles)
```

**After:**
```python
# ✅ Pattern: Dependency Injection
@router.post("/analyze")
async def analyze(
    request,
    service = Depends(get_analysis_service)
):
    return await service.analyze(request.candles)
```

---

## 🎯 Next Steps (Phase 3)

- [ ] Consolidate ETL pipeline
- [ ] Fix database schema versioning
- [ ] Analyze missing symbols
- [ ] Migrate all endpoints to DI

---

## 📚 Resources

- **Container:** `infrastructure/container.py`
- **Contracts:** `infrastructure/contracts.py`
- **Factories:** `infrastructure/container_factories.py`
- **Cache Adapter:** `infrastructure/adapters/memory_cache.py`
- **Example:** `api/v1/example_di.py`
- **Tests:** `tests/conftest.py`

---

## 🤝 Questions?

- **DI Pattern:** See `example_di.py` for reference
- **Testing:** See `conftest.py` for available fixtures
- **Contracts:** See `contracts.py` for service interfaces
- **Mocking:** See test examples above

---

**Document Version:** 1.0  
**Last Updated:** December 26, 2025  
**Phase Status:** ✅ Complete
