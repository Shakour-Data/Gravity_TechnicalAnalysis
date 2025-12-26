# Gravity Technical Analysis - Architecture Improvement Roadmap

**تاریخ**: دسامبر 2025  
**وضعیت**: نسخه 1.0 (اصلاح و استقرار)  
**مالک**: Gravity Tech Team  
**مدت کل**: 12-16 هفته

---

## 📋 خلاصه اجرایی

این Roadmap بر رفع **12 دسته مشکل معماری** متمرکز است. با اجرای این برنامه:
- ✅ Test coverage: 57.85% → 80%+
- ✅ Deployment time: کاهش 40%
- ✅ Development velocity: افزایش 30%
- ✅ Production incidents: کاهش 60%

---

## 🎯 اهداف کلیدی

| هدف | هدف‌گذاری | موعد |
|-----|---------|------|
| حذف security leaks (credentials) | 100% | ۲ روز |
| Dependency injection implementation | 95%+ | 3 هفته |
| Test coverage >= 70% | 100% | 5 هفته |
| CI/CD Pipeline اساسی | 100% | 2 هفته |
| Monolithic → Structured Modules | 100% | 8 هفته |
| Documentation sync | 100% | 4 هفته |

---

## 🚀 فاز‌ها

## **فاز 0: Response Team (1-2 روز) - CRITICAL**

### 0.1: حذف فوری Security Leaks

**مسائل:**
- `database_connection_info.txt` - حاوی credentials
- `temp_*` فایل‌ها - حساس‌ها
- Git history contaminated

**اقدامات:**
```bash
# 1. حذف از working directory
rm -f database_connection_info.txt
rm -rf temp_gravity_*
rm -f batch1_*.txt status.txt gravity_full_prompt.txt

# 2. تمیزکردن git history (BFG Repo-Cleaner)
brew install bfg  # یا choco install bfg اگر Windows
bfg --delete-files database_connection_info.txt
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 3. اضافه کردن به .gitignore
echo "database_connection_info.txt" >> .gitignore
echo "temp_*/" >> .gitignore
echo "batch*.txt" >> .gitignore
echo "status.txt" >> .gitignore
```

**Deliverables:**
- ✅ `.gitignore` updated
- ✅ Git history cleaned
- ✅ CI/CD secret scanning enabled

**Owner:** DevOps  
**Deadline:** روز ۱

---

### 0.2: فوری Rotate Credentials

- تغییر تمام DB passwords
- تجدید API keys
- Scan Docker Hub برای leaked images

**Owner:** Security  
**Deadline:** روز ۲

---

## **فاز 1: Foundation & Infrastructure (2-3 هفته)**

### 1.1: CI/CD Pipeline Setup

**مهم:** بدون CI/CD، هیچ تغییری safe نیست.

**مراحل:**
```yaml
GitHub Actions Workflow:
  Triggers:
    - on: [push, pull_request]
  
  Jobs:
    - lint:
        - runs: python -m flake8 apps/ --max-line-length=120
        - runs: python -m black --check apps/
        - time: 2 min
    
    - test-unit:
        - runs: pytest tests/unit/ -v --cov=apps/
        - cov-fail-under: 70%
        - time: 5 min
    
    - test-integration:
        - needs: [lint, test-unit]
        - runs: pytest tests/api/ --tb=short
        - time: 10 min
        - requires: Docker Compose
    
    - security-scan:
        - runs: bandit -r apps/
        - runs: safety check -r requirements.txt
        - time: 3 min
    
    - docker-build:
        - needs: [test-unit, test-integration]
        - builds: image for main branch only
        - push: ghcr.io/shakour-data/gravity:latest
        - time: 8 min
```

**Files to Create:**
- `.github/workflows/ci.yml`
- `.github/workflows/security.yml`
- `.github/workflows/deploy-staging.yml`

**Owner:** DevOps  
**Timeline:** 1 week  
**Deliverables:**
- ✅ CI passes consistently
- ✅ Coverage reports in PR
- ✅ Automated testing gate

---

### 1.2: Config Management Consolidation

**مشکل فعلی:**
```
Configs in:
├── .env.example
├── .env
├── configs/
│   ├── VERSION
│   ├── tools/
│   │   └── catalog.json
│   └── editorconfig
├── settings.py
└── docker-compose.*.yml files
```

**راهکار:**
```python
# apps/analysis_api/src/gravity_tech/config/unified_settings.py

from dataclasses import dataclass, field
from typing import Optional
import os
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    engine: str  # sqlite, postgresql
    url: str
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20

@dataclass
class CacheConfig:
    """Cache configuration"""
    enabled: bool = True
    backend: str = "redis"  # redis, memory, none
    host: str = "localhost"
    port: int = 6379
    ttl_seconds: int = 300
    db: int = 0

@dataclass
class MLConfig:
    """ML Model configuration"""
    model_path: str = "ml_models/"
    pattern_classifier_enabled: bool = True
    weight_optimizer_enabled: bool = False
    gpu_enabled: bool = False

@dataclass
class FeatureFlags:
    """Feature flags - centralized"""
    enable_scenarios: bool = False
    expose_db_explorer: bool = False
    enable_data_ingestion: bool = True
    enable_ml_inference: bool = True
    enable_harmonic_patterns: bool = True
    eureka_enabled: bool = False
    kafka_enabled: bool = False
    rabbitmq_enabled: bool = False
    metrics_enabled: bool = True

@dataclass
class Settings:
    """Master settings object"""
    environment: Environment = Environment.DEVELOPMENT
    app_name: str = "Gravity Technical Analysis"
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    debug: bool = False
    
    # Components
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Security
    jwt_secret: str = os.getenv("JWT_SECRET", "dev-secret-key")
    cors_origins: list = field(default_factory=lambda: ["*"])
    
    # Observability
    tracing_enabled: bool = False
    metrics_port: int = 9090
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment"""
        env = Environment(os.getenv("ENVIRONMENT", "development"))
        
        database = DatabaseConfig(
            engine=os.getenv("DB_ENGINE", "sqlite"),
            url=os.getenv("DATABASE_URL", "sqlite:///./gravity.db"),
        )
        
        cache = CacheConfig(
            enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            backend=os.getenv("CACHE_BACKEND", "redis"),
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
        )
        
        features = FeatureFlags(
            enable_scenarios=os.getenv("ENABLE_SCENARIOS", "false").lower() == "true",
            expose_db_explorer=os.getenv("EXPOSE_DB_EXPLORER", "false").lower() == "true",
        )
        
        return cls(
            environment=env,
            database=database,
            cache=cache,
            features=features,
            debug=env == Environment.DEVELOPMENT,
        )

# Global instance (lazy loaded)
_settings_instance: Optional[Settings] = None

def get_settings() -> Settings:
    """Get singleton settings instance"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings.from_env()
    return _settings_instance

def reset_settings():
    """Reset settings (for testing)"""
    global _settings_instance
    _settings_instance = None
```

**Environment Files:**
```env
# .env.development (git tracked, no secrets)
ENVIRONMENT=development
DATABASE_URL=sqlite:///./gravity.db
CACHE_BACKEND=memory
ENABLE_SCENARIOS=false
EXPOSE_DB_EXPLORER=true
LOG_LEVEL=DEBUG

# .env.production (git ignored, secrets)
ENVIRONMENT=production
DATABASE_URL=postgresql://user:pass@prod-db:5432/gravity
CACHE_BACKEND=redis
JWT_SECRET=<vault-secret>
ENABLE_SCENARIOS=true
LOG_LEVEL=INFO
```

**Owner:** Backend  
**Timeline:** 1 week  
**Deliverables:**
- ✅ Single source of config truth
- ✅ Env-specific files
- ✅ No hardcoded values

---

### 1.3: Create Repository Structure Documentation

**File:** `docs/REPOSITORY_STRUCTURE.md`

```markdown
# Repository Structure Guide

## Directory Layout

```
gravity-ta/
├── apps/
│   └── analysis_api/                 # FastAPI service
│       ├── src/gravity_tech/
│       │   ├── main.py              # Entry point
│       │   ├── api/v1/              # REST endpoints
│       │   ├── core/                # Business logic (NO external deps)
│       │   │   ├── domain/          # Entities, value objects
│       │   │   ├── indicators/      # Technical indicators
│       │   │   ├── patterns/        # Pattern detection
│       │   │   └── analysis/        # Analysis engines
│       │   ├── services/            # Use cases
│       │   ├── infrastructure/      # Adapters for external deps
│       │   ├── middleware/          # HTTP middleware, logging, etc
│       │   └── ml/                  # ML models
│       └── tests/                   # Unit & integration tests
│
├── services/
│   └── data_ingestion/              # Data ETL service
│       ├── src/
│       ├── scripts/
│       └── tests/
│
├── scripts/                         # Shared utility scripts
│   ├── run_full_pipeline.py        # DEPRECATED - use services/
│   ├── database/
│   └── utils/
│
├── ml_models/                       # ML model artifacts
│
├── data/                            # Local data (git ignored)
│
└── docs/
    ├── architecture/
    ├── guides/
    └── operations/

## Layer Definitions

### Production Code Layers

| Layer | Location | Purpose | Dependencies |
|-------|----------|---------|--------------|
| Domain | `core/domain/` | Entities, value objects, enums | None (pure Python) |
| Indicators | `core/indicators/` | Technical analysis calculations | numpy, pandas |
| Patterns | `core/patterns/` | Pattern detection algorithms | Domain, Indicators |
| Services | `services/` | Use cases, orchestration | Domain, Infrastructure |
| API | `api/v1/` | REST endpoints | FastAPI, Services |
| Infrastructure | `infrastructure/` | External service adapters | requests, psycopg2, redis |
| Middleware | `middleware/` | Cross-cutting concerns | Monitoring, Auth, Logging |

### Experimental/Non-Production

| Location | Status | Migration Plan |
|----------|--------|---|
| `experiments/` | Draft code | Move to production when ready |
| `temp_*` | Temporary | Delete after use |
| `.archived` | Legacy | Keep but don't import |

## Guidelines

### ✅ Do's
- Import only from same layer or inner layers
- `api/v1/` can import from `services/`, `core/`, `infrastructure/`
- `services/` can import from `core/`, `infrastructure/`
- Keep `core/` pure (only stdlib + numpy/pandas)

### ❌ Don'ts
- `core/` must NOT import from `api/`, `services/`, `infrastructure/`
- Avoid circular imports
- Don't hardcode paths; use config
```

**Owner:** Technical Lead  
**Timeline:** 2 days  
**Deliverables:**
- ✅ Clear structure documented
- ✅ Import guidelines clear
- ✅ Layer boundaries defined

---

## **فاز 2: Dependency Injection & Testability (3-4 هفته)**

### 2.1: Remove Global Singletons

**مشکل:**
```python
# services/cache_service.py
cache_manager = CacheManager(...)  # Global!

# services/data_ingestor_service.py
ingestor = DataIngestor(...)  # Global!

# usage in api/v1/analysis.py
from services import cache_manager  # Direct dependency
```

**راهکار: Dependency Injection Container**

```python
# apps/analysis_api/src/gravity_tech/infrastructure/container.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional
import structlog

logger = structlog.get_logger()

class ServiceContainer:
    """Dependency injection container"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}
        self._singletons: Dict[str, Any] = {}
    
    def register(self, name: str, factory: callable, singleton: bool = False):
        """Register a service"""
        self._factories[name] = factory
        logger.info("service_registered", name=name, singleton=singleton)
    
    def get(self, name: str) -> Any:
        """Get service instance (singleton or new)"""
        if name not in self._factories:
            raise ValueError(f"Service not registered: {name}")
        
        factory = self._factories[name]
        
        # Check if singleton already created
        if name in self._singletons:
            return self._singletons[name]
        
        # Create instance
        instance = factory(self)
        
        # Cache if singleton
        if self._factories[name].__self__.__dict__.get('_is_singleton'):
            self._singletons[name] = instance
        
        return instance
    
    def reset(self):
        """Reset all services (for testing)"""
        for service in self._singletons.values():
            if hasattr(service, 'close'):
                service.close()
        self._singletons.clear()
        logger.info("container_reset")

# Global container instance
_container: Optional[ServiceContainer] = None

def get_container() -> ServiceContainer:
    """Get global container"""
    global _container
    if _container is None:
        _container = ServiceContainer()
        _setup_container(_container)
    return _container

def _setup_container(container: ServiceContainer):
    """Register all services"""
    from gravity_tech.config.unified_settings import get_settings
    from gravity_tech.services.cache_service import CacheManager
    from gravity_tech.services.analysis_service import TechnicalAnalysisService
    
    settings = get_settings()
    
    # Register cache
    container.register(
        "cache",
        lambda c: CacheManager(
            backend=settings.cache.backend,
            host=settings.cache.host,
            port=settings.cache.port,
        ),
        singleton=True
    )
    
    # Register analysis service
    container.register(
        "analysis_service",
        lambda c: TechnicalAnalysisService(
            cache=c.get("cache")
        ),
        singleton=True
    )
    
    logger.info("container_setup_complete")
```

**Usage in API:**
```python
# apps/analysis_api/src/gravity_tech/api/v1/analysis.py

from fastapi import APIRouter, Depends
from gravity_tech.infrastructure.container import get_container

router = APIRouter()

async def get_analysis_service(container = Depends(lambda: get_container())):
    """Dependency for analysis service"""
    return container.get("analysis_service")

@router.post("/analyze")
async def analyze(
    request: AnalysisRequest,
    service = Depends(get_analysis_service)
):
    return await service.analyze(request)
```

**Owner:** Backend  
**Timeline:** 2 weeks  
**Deliverables:**
- ✅ No global imports
- ✅ All services injectable
- ✅ Tests can use mock container

---

### 2.2: Create Infrastructure/Adapter Layer

**مهم:** External services (DB, Cache, etc) باید abstract interfaces داشته باشند.

```python
# apps/analysis_api/src/gravity_tech/infrastructure/contracts.py

from abc import ABC, abstractmethod
from typing import Any, Optional, List
from datetime import datetime

class CacheBackend(ABC):
    """Abstract cache interface"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value with TTL"""
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete key"""
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""

class DatabaseBackend(ABC):
    """Abstract database interface"""
    
    @abstractmethod
    async def execute(self, query: str, params: dict) -> Any:
        """Execute query"""
    
    @abstractmethod
    async def fetch_one(self, query: str, params: dict) -> Optional[dict]:
        """Fetch single row"""
    
    @abstractmethod
    async def fetch_all(self, query: str, params: dict) -> List[dict]:
        """Fetch all rows"""

class ExternalDataService(ABC):
    """Abstract external data service"""
    
    @abstractmethod
    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> List[dict]:
        """Get OHLCV candles"""
```

**Implementations:**

```python
# apps/analysis_api/src/gravity_tech/infrastructure/adapters/redis_cache.py

from gravity_tech.infrastructure.contracts import CacheBackend
import redis.asyncio as redis

class RedisCacheAdapter(CacheBackend):
    """Redis implementation of cache"""
    
    def __init__(self, host: str, port: int, db: int = 0):
        self.redis = None
        self.host = host
        self.port = port
        self.db = db
    
    async def initialize(self):
        self.redis = await redis.from_url(
            f"redis://{self.host}:{self.port}/{self.db}"
        )
    
    async def get(self, key: str):
        value = await self.redis.get(key)
        return value.decode() if value else None
    
    # ... implement others

# apps/analysis_api/src/gravity_tech/infrastructure/adapters/memory_cache.py

from gravity_tech.infrastructure.contracts import CacheBackend
from datetime import datetime, timedelta

class MemoryCacheAdapter(CacheBackend):
    """In-memory implementation (for testing)"""
    
    def __init__(self):
        self.data = {}
        self.ttls = {}
    
    async def get(self, key: str):
        if key not in self.data:
            return None
        
        # Check TTL
        if key in self.ttls:
            if datetime.now() > self.ttls[key]:
                del self.data[key]
                del self.ttls[key]
                return None
        
        return self.data[key]
    
    # ... implement others
```

**Owner:** Backend  
**Timeline:** 1.5 weeks  
**Deliverables:**
- ✅ All external deps abstracted
- ✅ Multiple implementations possible
- ✅ Easy to mock in tests

---

### 2.3: Add Comprehensive Mocking Framework

```python
# apps/analysis_api/tests/conftest.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from gravity_tech.infrastructure.container import ServiceContainer
from gravity_tech.infrastructure.contracts import CacheBackend, DatabaseBackend
from gravity_tech.infrastructure.adapters.memory_cache import MemoryCacheAdapter

@pytest.fixture
def mock_cache() -> CacheBackend:
    """Provide mock cache"""
    return AsyncMock(spec=CacheBackend)

@pytest.fixture
def memory_cache() -> MemoryCacheAdapter:
    """Provide in-memory cache for integration tests"""
    return MemoryCacheAdapter()

@pytest.fixture
def mock_database() -> DatabaseBackend:
    """Provide mock database"""
    return AsyncMock(spec=DatabaseBackend)

@pytest.fixture
def test_container(memory_cache, mock_database) -> ServiceContainer:
    """Provide test container with mocks"""
    container = ServiceContainer()
    container.register("cache", lambda c: memory_cache, singleton=True)
    container.register("database", lambda c: mock_database, singleton=True)
    return container

# Usage in tests
@pytest.mark.asyncio
async def test_analysis_with_mocks(test_container):
    service = test_container.get("analysis_service")
    result = await service.analyze(request)
    assert result.signal == "BUY"
```

**Owner:** QA/Backend  
**Timeline:** 1 week  
**Deliverables:**
- ✅ All services mockable
- ✅ Test fixtures standardized
- ✅ CI can run offline

---

## **فاز 3: Data & Database (2-3 هفته)**

### 3.1: Consolidate ETL Pipeline

**مشکل فعلی:**
```
services/data_ingestion/       (Main)
├── scripts/
├── web/
└── main.py

scripts/                        (Duplicates)
├── run_full_pipeline.py       ❌ DEPRECATED
├── migrate_*.py               ❌ SCATTERED
└── etl/
```

**راهکار:**
```
apps/data_pipeline/             (New - centralized)
├── src/
│   └── gravity_pipeline/
│       ├── __init__.py
│       ├── config.py           # Pipeline-specific config
│       ├── models/             # Domain entities for pipeline
│       ├── extractors/         # Data source adapters
│       │   ├── tse_extractor.py
│       │   └── tse_api.py
│       ├── transformers/       # Data transformation
│       │   ├── cleaner.py
│       │   └── validator.py
│       ├── loaders/            # Data sink adapters
│       │   ├── sqlite_loader.py
│       │   └── postgres_loader.py
│       ├── orchestrator.py     # Pipeline coordination
│       └── migrations/         # Alembic migrations
├── tests/
└── main.py                     # Entry point
```

**Implementation:**
```python
# apps/data_pipeline/src/gravity_pipeline/orchestrator.py

from enum import Enum
from typing import List, Optional
import structlog

logger = structlog.get_logger()

class PipelineStage(Enum):
    EXTRACT = "extract"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    LOAD = "load"
    DEDUPLICATE = "deduplicate"

class DataPipeline:
    """Unified data pipeline orchestrator"""
    
    def __init__(self, config):
        self.config = config
        self.stages_completed = []
    
    async def run_full(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        skip_stages: Optional[List[PipelineStage]] = None
    ):
        """Run complete pipeline"""
        
        skip_stages = skip_stages or []
        
        try:
            # Stage 1: Extract
            if PipelineStage.EXTRACT not in skip_stages:
                logger.info("stage_start", stage="extract")
                candles = await self._extract(symbols, start_date)
                logger.info("stage_complete", stage="extract", count=len(candles))
            
            # Stage 2: Transform
            if PipelineStage.TRANSFORM not in skip_stages:
                logger.info("stage_start", stage="transform")
                candles = await self._transform(candles)
                logger.info("stage_complete", stage="transform", count=len(candles))
            
            # Stage 3: Validate
            if PipelineStage.VALIDATE not in skip_stages:
                logger.info("stage_start", stage="validate")
                candles = await self._validate(candles)
                logger.info("stage_complete", stage="validate", count=len(candles))
            
            # Stage 4: Deduplicate
            if PipelineStage.DEDUPLICATE not in skip_stages:
                logger.info("stage_start", stage="deduplicate")
                candles = await self._deduplicate(candles)
                logger.info("stage_complete", stage="deduplicate", count=len(candles))
            
            # Stage 5: Load
            if PipelineStage.LOAD not in skip_stages:
                logger.info("stage_start", stage="load")
                loaded_count = await self._load(candles)
                logger.info("stage_complete", stage="load", count=loaded_count)
            
            logger.info("pipeline_complete", status="success")
            return {"status": "success", "processed": len(candles)}
        
        except Exception as e:
            logger.error("pipeline_failed", error=str(e))
            raise
    
    async def _extract(self, symbols, start_date):
        """Extract raw data from TSE"""
        # Implementation
        pass
    
    async def _transform(self, candles):
        """Transform and normalize data"""
        # Implementation
        pass
    
    async def _validate(self, candles):
        """Validate data quality"""
        # Implementation
        pass
    
    async def _deduplicate(self, candles):
        """Remove duplicates"""
        # Implementation
        pass
    
    async def _load(self, candles):
        """Load into target database"""
        # Implementation
        pass
```

**Migration Path:**
1. Create new `apps/data_pipeline/`
2. Move logic from `services/data_ingestion/` 
3. Move utility from `scripts/` 
4. Keep old locations with deprecation warnings for 1 release
5. Remove old code in next major version

**Owner:** Data Team  
**Timeline:** 2 weeks  
**Deliverables:**
- ✅ Single source of truth
- ✅ Stages independently runnable
- ✅ Better error tracking

---

### 3.2: Database Schema & Versioning

**Fix Alembic & Migrations:**

```bash
# apps/data_pipeline/migrations/env.py

# Enable auto-generation
target_metadata = db.metadata

def run_migrations_online():
    """Run migrations in online mode"""
    
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix='sqlalchemy.',
        poolclass=pool.NullPool
    )
    
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # Important for SQLite
        )
        
        with context.begin_transaction():
            context.run_migrations()
```

**Schema Validation:**

```python
# apps/data_pipeline/src/gravity_pipeline/validators/schema.py

from sqlalchemy import inspect

class SchemaValidator:
    """Validate database schema consistency"""
    
    def __init__(self, db_engine):
        self.engine = db_engine
    
    def validate_tables_exist(self, expected_tables: List[str]) -> bool:
        """Check if all required tables exist"""
        inspector = inspect(self.engine)
        existing = inspector.get_table_names()
        missing = set(expected_tables) - set(existing)
        
        if missing:
            logger.error("missing_tables", tables=missing)
            return False
        
        return True
    
    def validate_column_types(self, table: str, expected_cols: dict) -> bool:
        """Validate column types"""
        inspector = inspect(self.engine)
        existing_cols = {
            col['name']: col['type']
            for col in inspector.get_columns(table)
        }
        
        for col_name, col_type in expected_cols.items():
            if col_name not in existing_cols:
                logger.error("missing_column", table=table, column=col_name)
                return False
        
        return True
```

**Owner:** Database  
**Timeline:** 1 week  
**Deliverables:**
- ✅ Auto migrations working
- ✅ Schema validation before load
- ✅ Version tracking in DB

---

### 3.3: Fix Missing Analysis Symbols

**مشکل:** 50+ symbols بدون analysis results

**راهکار:**
```python
# scripts/data_pipeline/fix_orphaned_symbols.py

async def reprocess_missing_symbols():
    """Reprocess symbols without analysis results"""
    
    # Find orphaned symbols
    missing = await find_symbols_without_analysis()
    logger.info("found_missing_symbols", count=len(missing))
    
    # Reprocess each
    for symbol in missing:
        try:
            # Get last 500 candles
            candles = await db.get_candles(symbol, limit=500)
            
            # Run analysis pipeline
            result = await analysis_service.analyze(candles)
            
            # Store result
            await db.store_analysis_result(symbol, result)
            
            logger.info("symbol_reprocessed", symbol=symbol)
        
        except Exception as e:
            logger.error("reprocess_failed", symbol=symbol, error=str(e))
    
    logger.info("reprocessing_complete", symbols_processed=len(missing))
```

**Run via:**
```bash
python -m gravity_pipeline.scripts.fix_orphaned_symbols \
  --database-url postgresql://user:pass@localhost/gravity
```

**Owner:** Data Team  
**Timeline:** 3 days  
**Deliverables:**
- ✅ All symbols analyzed
- ✅ DB consistency restored

---

## **فاز 4: Testing & Quality (2-3 هفته)**

### 4.1: Improve Test Coverage to 70%+

**Current:** 57.85%  
**Target:** 80%+

**Strategy:**

```
Unit Tests (50% → 65%)
├── core/indicators/         (all functions)
├── core/patterns/           (all pathways)
└── core/analysis/           (multi-horizon logic)

Integration Tests (5% → 12%)
├── api/v1/*                 (endpoint tests)
├── data_pipeline/           (end-to-end)
└── services/                (service integration)

E2E Tests (2% → 3%)
├── Full analysis flow
├── Data pipeline flow
└── Error scenarios
```

**Tool: pytest with coverage gates**

```yaml
# pytest.ini
[pytest]
addopts = 
    --cov=apps/
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=70
    -v
    --tb=short
```

**Owner:** QA  
**Timeline:** 2 weeks  
**Deliverables:**
- ✅ 70%+ coverage
- ✅ All critical paths tested
- ✅ Coverage report in CI

---

### 4.2: Add Security Testing

```python
# apps/analysis_api/tests/test_security.py

import pytest
from bandit.main import main as bandit_main

class TestSecurityHeaders:
    """Test security headers in responses"""
    
    @pytest.mark.asyncio
    async def test_hsts_header(self, client):
        response = client.get("/api/health")
        assert "Strict-Transport-Security" in response.headers
    
    @pytest.mark.asyncio
    async def test_no_server_header(self, client):
        response = client.get("/api/health")
        assert "Server" not in response.headers
    
    @pytest.mark.asyncio
    async def test_csp_header(self, client):
        response = client.get("/api/health")
        assert "Content-Security-Policy" in response.headers

class TestInputValidation:
    """Test input validation"""
    
    @pytest.mark.asyncio
    async def test_sql_injection_attempt(self, client):
        malicious = "'; DROP TABLE symbols; --"
        response = client.post(
            "/api/v1/analyze",
            json={"symbol": malicious}
        )
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_xss_attempt(self, client):
        malicious = "<script>alert('xss')</script>"
        response = client.post(
            "/api/v1/analyze",
            json={"symbol": malicious}
        )
        assert response.status_code == 400
```

**Owner:** Security  
**Timeline:** 1 week  
**Deliverables:**
- ✅ No security tests failing
- ✅ OWASP Top 10 covered
- ✅ Input validation tested

---

## **فاز 5: Code Organization & Cleanup (2-3 هفته)**

### 5.1: Remove Legacy Code & Deprecations

**Deprecated Items to Remove:**

```python
# Delete:
❌ apps/analysis_api/src/gravity_tech/models/schemas_backup.py
❌ scripts/run_full_pipeline.py (use data_pipeline instead)
❌ scripts/migrate_sqlite_to_pg.py (use alembic)
❌ All temp_*.py files
❌ Archived test files in tests/archived/

# Add deprecation warnings to:
⚠️ services/data_ingestion/ (migrate to apps/data_pipeline)
⚠️ Any hardcoded paths
```

**Deprecation Pattern:**
```python
import warnings

def old_function():
    warnings.warn(
        "old_function is deprecated, use new_function instead",
        DeprecationWarning,
        stacklevel=2
    )
    # ... implementation
```

**Owner:** Backend  
**Timeline:** 1 week  
**Deliverables:**
- ✅ No dead code
- ✅ Clear deprecation path
- ✅ Migration guide

---

### 5.2: Refactor Services for Clarity

**Pattern:** Services should follow Single Responsibility

```python
# BAD: Everything in one service
class AnalysisService:
    def analyze(self): ...
    def cache_result(self): ...
    def publish_event(self): ...
    def store_in_db(self): ...

# GOOD: Separated concerns
class TechnicalAnalysisService:
    """Pure analysis logic"""
    def analyze(self, candles): ...

class AnalysisResultPersister:
    """Handle storage"""
    def save(self, result): ...

class AnalysisEventPublisher:
    """Handle notifications"""
    def publish(self, result): ...
```

**Apply to:**
- `TechnicalAnalysisService` → keep pure
- `ToolRecommendationService` → keep pure
- Create new `AnalysisResultPersister`
- Create new `AnalysisEventPublisher`

**Owner:** Backend  
**Timeline:** 1.5 weeks  
**Deliverables:**
- ✅ Single responsibility
- ✅ Easier to test
- ✅ Easier to reuse

---

## **فاز 6: Documentation & Communication (1-2 هفته)**

### 6.1: Update Architecture Documentation

**Files to Update:**
- `docs/architecture/SYSTEM_ARCHITECTURE_DIAGRAMS.md` - sync with reality
- `docs/guides/QUICK_START.md` - add new patterns
- `docs/operations/DEPLOYMENT_GUIDE.md` - add CI/CD
- `README.md` - update structure section

**New Files to Create:**
- `docs/ARCHITECTURE_DECISIONS.md` - ADRs
- `docs/TESTING_STRATEGY.md` - testing approach
- `docs/CONTRIBUTING.md` - development guide
- `docs/SECURITY.md` - security best practices

**Owner:** Tech Lead  
**Timeline:** 1 week  
**Deliverables:**
- ✅ All docs current
- ✅ New developers can onboard
- ✅ Architecture clear

---

### 6.2: API Reference Update

```python
# Auto-generate OpenAPI with better descriptions

@router.post(
    "/api/v1/analyze",
    response_model=AnalysisResponse,
    tags=["Technical Analysis"],
    summary="Perform complete technical analysis",
    description="""
    Comprehensive technical analysis including:
    - 60+ technical indicators (6 dimensions)
    - Multi-horizon analysis (3D, 7D, 30D)
    - Elliott wave patterns
    - Harmonic patterns with ML scoring
    - Real-time signals and confidence
    
    **Minimum Requirements:**
    - At least 60 valid OHLCV candles
    - Timeframe: 1m to 1w
    
    **Response includes:**
    - Signal (BUY/SELL/NEUTRAL)
    - Confidence (0-100%)
    - Dimension scores
    - Pattern detections
    - Risk assessment
    """,
    responses={
        200: {"description": "Successful analysis"},
        400: {"description": "Invalid input"},
        503: {"description": "Service unavailable"},
    }
)
async def analyze(request: AnalysisRequest):
    pass
```

**Owner:** Tech Lead  
**Timeline:** 1 week  
**Deliverables:**
- ✅ Auto-generated docs
- ✅ Examples in docs
- ✅ Clear error codes

---

## **فاز 7: Deployment & Operations (1-2 هفته)**

### 7.1: Kubernetes Deployment

**Create standardized manifests:**

```yaml
# infra/k8s/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gravity-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gravity-api
  template:
    metadata:
      labels:
        app: gravity-api
    spec:
      containers:
      - name: api
        image: ghcr.io/shakour-data/gravity:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: gravity-secrets
              key: database-url
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

**Owner:** DevOps  
**Timeline:** 1 week  
**Deliverables:**
- ✅ K8s manifests tested
- ✅ Helm charts optional
- ✅ Auto-scaling configured

---

### 7.2: Monitoring & Alerting

```python
# apps/analysis_api/src/gravity_tech/middleware/metrics.py

from prometheus_client import Counter, Histogram, Gauge

# API Metrics
api_request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

# Analysis Metrics
analysis_duration = Histogram(
    'analysis_duration_seconds',
    'Time to perform analysis'
)

analysis_errors = Counter(
    'analysis_errors_total',
    'Total analysis errors',
    ['error_type']
)

# Cache Metrics
cache_hits = Counter(
    'cache_hits_total',
    'Cache hits'
)

cache_misses = Counter(
    'cache_misses_total',
    'Cache misses'
)

# Alert Rules
alerts = """
groups:
- name: gravity
  rules:
  - alert: HighErrorRate
    expr: rate(analysis_errors_total[5m]) > 0.05
    for: 5m
    annotations:
      summary: "High error rate detected"
  
  - alert: SlowAnalysis
    expr: histogram_quantile(0.95, api_request_duration) > 5
    annotations:
      summary: "Slow analysis detected"
  
  - alert: CacheMissRate
    expr: |
      rate(cache_misses_total[5m]) / 
      (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m])) > 0.8
    annotations:
      summary: "High cache miss rate"
"""
```

**Owner:** DevOps  
**Timeline:** 1 week  
**Deliverables:**
- ✅ Prometheus scrape configured
- ✅ Grafana dashboards
- ✅ Alert rules defined

---

## 📊 Timeline Overview

```
Week 1-2:   Faze 0-1 (Security, CI/CD, Config)
Week 3-5:   Phase 2 (DI, Adapters, Mocking)
Week 6-8:   Phase 3 (ETL, Database)
Week 9-10:  Phase 4 (Testing)
Week 11-12: Phase 5-6 (Cleanup, Docs)
Week 13-14: Phase 7 (Deployment)
```

---

## ✅ Success Metrics

| Metric | Current | Target | Owner |
|--------|---------|--------|-------|
| Test Coverage | 57.85% | 80%+ | QA |
| CI/CD Pass Rate | 0% | 100% | DevOps |
| Security Issues | 3 | 0 | Security |
| Avg Deploy Time | 30 min | 5 min | DevOps |
| Tech Debt | High | Low | Tech Lead |
| Documentation Sync | 30% | 95% | Tech Lead |

---

## 🎯 Next Steps

### Immediate (Today)
- [ ] Approve roadmap
- [ ] Assign owners for each phase
- [ ] Create GitHub Project board
- [ ] Schedule kickoff meeting

### This Week
- [ ] Start Phase 0 (security)
- [ ] Create CI/CD pipeline
- [ ] Setup dependency injection

### Next Week
- [ ] Phase 1 complete
- [ ] Phase 2 started
- [ ] First PRs with CI passing

---

## 📞 Questions & Support

**Contact:** Tech Lead  
**Review Schedule:** Weekly sync-ups  
**Escalation:** Architecture review board

---

**Document Version:** 1.0  
**Last Updated:** December 26, 2025  
**Next Review:** January 2, 2026
