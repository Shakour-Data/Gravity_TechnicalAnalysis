# Phase 3: Data & Database Consolidation

**Status:** 🔄 IN PROGRESS  
**Timeline:** 2-3 weeks  
**Owner:** Data Team  
**Completion Target:** Week 8-9

---

## 📋 Overview

Phase 3 consolidates all ETL operations into a unified `apps/data_pipeline/` module, replacing scattered logic in `services/data_ingestion/` and `scripts/`. This improves maintainability, testability, and provides a single source of truth for data operations.

### Key Objectives

1. **3.1 Consolidate ETL Pipeline** - Merge duplicate code
2. **3.2 Database Schema & Versioning** - Fix Alembic migrations
3. **3.3 Fix Missing Analysis Symbols** - Restore DB consistency

---

## 🎯 3.1: Consolidate ETL Pipeline

### Current State

```
services/data_ingestion/          ❌ Main but scattered
├── scripts/
├── web/
└── main.py

scripts/                          ❌ Duplicates logic
├── run_full_pipeline.py         (Deprecated)
├── migrate_sqlite_to_pg.py      (Scattered)
└── analysis/
    └── compute_daily_scores.py
```

### Target State

```
apps/data_pipeline/              ✅ Unified, centralized
├── src/gravity_pipeline/
│   ├── config.py               ✅ DONE
│   ├── orchestrator.py         🔄 IN PROGRESS
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── tse_extractor.py    (Extract from TSE)
│   │   └── tse_api.py          (TSE API client)
│   ├── transformers/
│   │   ├── __init__.py
│   │   ├── cleaner.py          (Data cleaning)
│   │   ├── normalizer.py       (Standardize format)
│   │   └── enricher.py         (Add computed fields)
│   ├── validators/
│   │   ├── __init__.py
│   │   ├── quality.py          (Data quality checks)
│   │   └── schema.py           (Schema validation)
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── sqlite_loader.py
│   │   └── postgres_loader.py
│   ├── models/                 (Domain entities)
│   ├── migrations/             (Alembic)
│   ├── __init__.py
│   └── main.py                 (Entry point)
├── tests/
│   ├── test_orchestrator.py
│   ├── test_extractors.py
│   ├── test_transformers.py
│   ├── test_validators.py
│   └── test_loaders.py
├── requirements.txt
├── README.md
└── Dockerfile
```

### Implementation Details

#### Stage 1: EXTRACT

**Purpose:** Retrieve raw data from source systems

**Implementations:**
- `TSEExtractor` - From TSE database or API
- `BinanceExtractor` - From Binance API (future)
- `CSVExtractor` - From CSV files (future)

**Example Usage:**
```python
from gravity_pipeline.extractors import TSEExtractor

extractor = TSEExtractor(
    api_key="...",
    base_url="https://api.tse.ir"
)

candles = await extractor.extract(
    symbols=["BTCUSDT", "ETHUSDT"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    limit=1000
)
```

#### Stage 2: TRANSFORM

**Purpose:** Normalize and clean raw data

**Operations:**
- Rename fields to standard names (open, high, low, close, volume, timestamp)
- Convert data types (string → float, etc.)
- Calculate derived fields (SMA, RSI, etc.)
- Handle missing values

**Example Usage:**
```python
from gravity_pipeline.transformers import DataCleaner

cleaner = DataCleaner(
    remove_outliers=True,
    fill_missing_with="previous"
)

cleaned = await cleaner.transform(raw_candles)
```

#### Stage 3: VALIDATE

**Purpose:** Ensure data quality before loading

**Checks:**
- No NaN/Inf values
- high >= low (OHLC validity)
- volume >= 0
- Timestamp ordering
- Duplicate detection

**Example Usage:**
```python
from gravity_pipeline.validators import DataQualityValidator

validator = DataQualityValidator(
    check_ohlc=True,
    check_volume=True,
    check_timestamps=True
)

valid, invalid = await validator.validate(candles)
```

#### Stage 4: DEDUPLICATE

**Purpose:** Remove duplicate records

**Strategy:**
- Dedup key: (symbol, timestamp)
- Compare hash of OHLC values
- Keep first occurrence, discard later ones

#### Stage 5: LOAD

**Purpose:** Persist cleaned data to target database

**Implementations:**
- `SQLiteLoader` - For development
- `PostgreSQLLoader` - For production
- `BulkLoader` - Batch inserts (1000+ records)

**Example Usage:**
```python
from gravity_pipeline.loaders import PostgreSQLLoader

loader = PostgreSQLLoader(
    connection_url="postgresql://user:pass@localhost/gravity",
    batch_size=1000
)

loaded_count = await loader.load(cleaned_candles)
```

### Full Pipeline Example

```python
import asyncio
from gravity_pipeline.orchestrator import DataPipeline, PipelineConfig

async def main():
    config = PipelineConfig(
        source_db_url="sqlite:///./data/source.db",
        target_db_url="postgresql://user:pass@localhost/gravity",
        batch_size=500,
        max_workers=4
    )
    
    pipeline = DataPipeline(config)
    
    result = await pipeline.run_full(
        symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        start_date="2024-01-01",
        skip_stages=[],  # Run all stages
        limit=None
    )
    
    print(f"Pipeline Result: {result}")
    # Output:
    # {
    #   "status": "success",
    #   "stats": {
    #     "extracted": 10000,
    #     "transformed": 10000,
    #     "validated": 9950,
    #     "deduplicated": 50,
    #     "loaded": 9950,
    #     "errors": 0
    #   },
    #   "duration_seconds": 45.2
    # }

asyncio.run(main())
```

### Testing Strategy

```python
# tests/test_orchestrator.py
@pytest.mark.asyncio
async def test_full_pipeline_flow(test_container, mock_extractor):
    """Test complete pipeline execution"""
    mock_extractor.return_value = [
        {"timestamp": "2024-01-01", "open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000},
        {"timestamp": "2024-01-02", "open": 105, "high": 115, "low": 95, "close": 110, "volume": 1200},
    ]
    
    config = PipelineConfig(source_db_url="sqlite:///:memory:")
    pipeline = DataPipeline(config)
    
    result = await pipeline.run_full(symbols=["TEST"])
    
    assert result["status"] == "success"
    assert result["stats"]["extracted"] == 2
    assert result["stats"]["loaded"] >= 2

@pytest.mark.asyncio
async def test_skip_stages(config):
    """Test skipping specific pipeline stages"""
    pipeline = DataPipeline(config)
    
    result = await pipeline.run_full(
        symbols=["TEST"],
        skip_stages=[PipelineStage.VALIDATE]
    )
    
    assert PipelineStage.VALIDATE not in result["stages_completed"]

@pytest.mark.asyncio
async def test_error_handling(config):
    """Test error handling and recovery"""
    pipeline = DataPipeline(config)
    
    with pytest.raises(Exception):
        await pipeline.run_full(symbols=[])
```

### Migration Path

**Week 1:**
- [ ] Create new `apps/data_pipeline/` with core structure
- [ ] Implement all extractors with interfaces
- [ ] Implement all transformers with interfaces
- [ ] Implement all validators with interfaces
- [ ] Implement all loaders with interfaces
- [ ] Create comprehensive tests

**Week 2:**
- [ ] Move logic from `services/data_ingestion/` → new pipeline
- [ ] Move logic from `scripts/` → new pipeline
- [ ] Add deprecation warnings to old locations
- [ ] Update documentation with new import paths
- [ ] Test with real TSE data

**Week 3:**
- [ ] Remove old code locations
- [ ] Performance tuning (batching, parallelization)
- [ ] Add monitoring/metrics
- [ ] Deploy to staging

---

## 🎯 3.2: Database Schema & Versioning

### Current Issues

- ❌ Alembic migrations not auto-generating
- ❌ Schema validation missing
- ❌ No version tracking in DB

### Solution

#### 1. Fix Alembic Configuration

```python
# apps/data_pipeline/migrations/env.py

def run_migrations_online():
    """Run migrations in online mode"""
    
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = os.environ.get("DATABASE_URL")
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    
    with connectable.begin() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # Important for SQLite compatibility
            compare_type=True,      # Compare column types
            compare_server_default=True,  # Compare default values
        )
        
        with context.begin_transaction():
            context.run_migrations()
```

#### 2. Auto-Generate Migrations

```bash
# Generate migration from current models
cd apps/data_pipeline
alembic revision --autogenerate -m "Add new columns to candles table"

# Review generated migration in migrations/versions/

# Apply migration
alembic upgrade head
```

#### 3. Schema Validation

```python
# apps/data_pipeline/src/gravity_pipeline/validators/schema.py

from sqlalchemy import inspect
import structlog

logger = structlog.get_logger()

class SchemaValidator:
    """Validate database schema consistency"""
    
    def __init__(self, engine):
        self.engine = engine
        self.inspector = inspect(engine)
    
    def validate_tables(self, expected_tables: list[str]) -> bool:
        """Check if all required tables exist"""
        existing = set(self.inspector.get_table_names())
        expected = set(expected_tables)
        missing = expected - existing
        
        if missing:
            logger.error("schema_invalid", missing_tables=list(missing))
            return False
        
        logger.info("schema_valid_tables")
        return True
    
    def validate_columns(self, table: str, expected_cols: dict) -> bool:
        """Validate column types"""
        try:
            actual = {
                col["name"]: col["type"]
                for col in self.inspector.get_columns(table)
            }
            
            for col_name, col_type in expected_cols.items():
                if col_name not in actual:
                    logger.error("schema_invalid", missing_column=col_name, table=table)
                    return False
            
            logger.info("schema_valid_columns", table=table)
            return True
        
        except Exception as e:
            logger.error("schema_validation_error", error=str(e))
            return False
    
    def validate_indexes(self, table: str, expected_indexes: list[str]) -> bool:
        """Validate indexes exist"""
        existing = {
            idx["name"]
            for idx in self.inspector.get_indexes(table)
        }
        missing = set(expected_indexes) - existing
        
        if missing:
            logger.warning("missing_indexes", table=table, indexes=list(missing))
            return False
        
        return True
    
    def validate_all(self) -> dict:
        """Validate entire schema"""
        from gravity_pipeline.models import EXPECTED_SCHEMA
        
        results = {}
        
        for table, spec in EXPECTED_SCHEMA.items():
            results[table] = {
                "columns": self.validate_columns(table, spec["columns"]),
                "indexes": self.validate_indexes(table, spec.get("indexes", [])),
            }
        
        all_valid = all(
            all(v.values()) for v in results.values()
        )
        
        if all_valid:
            logger.info("schema_fully_valid")
        
        return {"valid": all_valid, "details": results}
```

#### 4. Pre-Load Schema Check

```python
# apps/data_pipeline/src/gravity_pipeline/loaders/base.py

class BaseLoader:
    """Base loader with schema validation"""
    
    async def load(self, candles: list) -> int:
        """Load with schema validation"""
        
        # Validate schema before load
        validator = SchemaValidator(self.engine)
        if not validator.validate_all()["valid"]:
            raise SchemaValidationError("Schema invalid, cannot load data")
        
        # Perform actual load
        return await self._do_load(candles)
    
    async def _do_load(self, candles: list) -> int:
        """Override in subclasses"""
        raise NotImplementedError
```

---

## 🎯 3.3: Fix Missing Analysis Symbols

### Problem Analysis

**Current State:**
- 50+ symbols have no analysis results
- These are "orphaned" - data exists but analysis missing
- DB consistency degraded

**Root Causes:**
- Pipeline crashes mid-run
- Analysis service not triggered
- Incomplete data migration

### Solution Script

```python
# scripts/data_pipeline/fix_orphaned_symbols.py

import asyncio
from typing import List
import structlog
from gravity_pipeline.orchestrator import DataPipeline, PipelineConfig
from gravity_pipeline.validators import SchemaValidator
from gravity_tech.services.analysis_service import AnalysisService
from gravity_tech.infrastructure.container import get_container

logger = structlog.get_logger()

async def find_symbols_without_analysis(db_engine) -> List[str]:
    """Find symbols that have candles but no analysis results"""
    
    query = """
    SELECT DISTINCT c.symbol
    FROM candles c
    LEFT JOIN analysis_results ar ON c.symbol = ar.symbol
    WHERE ar.symbol IS NULL
    GROUP BY c.symbol
    HAVING COUNT(c.id) > 100
    """
    
    result = await db.execute(query)
    missing = [row[0] for row in result]
    
    logger.info("found_orphaned_symbols", count=len(missing), symbols=missing[:10])
    return missing

async def reprocess_missing_symbols():
    """Reprocess symbols without analysis results"""
    
    # Get container from Phase 2
    container = get_container()
    analysis_service = container.get("analysis_service")
    db = container.get("database")
    
    # Find missing symbols
    missing_symbols = await find_symbols_without_analysis(db)
    logger.info("reprocessing_symbols", count=len(missing_symbols))
    
    success_count = 0
    error_count = 0
    
    for symbol in missing_symbols:
        try:
            # Get last N candles
            candles = await db.get_candles(symbol, limit=500)
            
            if not candles:
                logger.warning("no_candles_for_symbol", symbol=symbol)
                continue
            
            # Run analysis
            logger.info("analyzing_symbol", symbol=symbol, candles=len(candles))
            result = await analysis_service.analyze(candles)
            
            # Store result
            await db.store_analysis_result(
                symbol=symbol,
                result=result,
                timestamp=datetime.now()
            )
            
            success_count += 1
            logger.info("symbol_processed", symbol=symbol, signal=result.signal)
        
        except Exception as e:
            error_count += 1
            logger.error("symbol_failed", symbol=symbol, error=str(e))
            continue
    
    logger.info(
        "reprocessing_complete",
        total=len(missing_symbols),
        success=success_count,
        errors=error_count
    )
    
    return {
        "total_symbols": len(missing_symbols),
        "processed": success_count,
        "errors": error_count,
        "success_rate": success_count / len(missing_symbols) if missing_symbols else 0
    }

async def main():
    """Main entry point"""
    result = await reprocess_missing_symbols()
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Running the Fix

```bash
# From project root
cd apps/analysis_api

# Run the fix script
python ../../scripts/data_pipeline/fix_orphaned_symbols.py \
  --database-url postgresql://user:pass@localhost/gravity \
  --limit 100  # Process max 100 symbols for testing

# Monitor progress
tail -f logs/fix_orphaned_symbols.log
```

### Validation

After running the fix:

```python
# Verify all symbols now have analysis
SELECT COUNT(DISTINCT c.symbol) as total_symbols,
       COUNT(DISTINCT ar.symbol) as analyzed_symbols
FROM candles c
LEFT JOIN analysis_results ar ON c.symbol = ar.symbol;

# Should show: total_symbols == analyzed_symbols
```

---

## 📊 Success Metrics

### 3.1: ETL Consolidation
- ✅ Single source of truth for all ETL logic
- ✅ All stages independently runnable
- ✅ Code duplication reduced by 80%
- ✅ Test coverage >= 80% for pipeline module
- ✅ Processing time documented

### 3.2: Database Schema
- ✅ Auto-migrations working
- ✅ Schema validation before load
- ✅ DB version tracking enabled
- ✅ No schema errors in CI

### 3.3: Missing Symbols
- ✅ All symbols have analysis results
- ✅ DB consistency restored
- ✅ Zero orphaned symbols
- ✅ Fix script automated

---

## 🚀 Integration with Phase 2 (DI)

The pipeline leverages Phase 2's Dependency Injection:

```python
# Use container from Phase 2
from gravity_tech.infrastructure.container import get_container

container = get_container()

# Get services
analysis_service = container.get("analysis_service")
database = container.get("database")
cache = container.get("cache")

# Run pipeline with DI services
pipeline = DataPipeline(config)
result = await pipeline.run_full(symbols=["BTCUSDT"])
```

---

## 📅 Timeline

| Week | Task | Owner | Status |
|------|------|-------|--------|
| 6 | 3.1a Create base orchestrator | Data | 🔄 |
| 6 | 3.1b Implement extractors | Data | 🟡 |
| 6-7 | 3.1c Implement transformers | Data | 🟡 |
| 7 | 3.1d Implement validators | Data | 🟡 |
| 7 | 3.1e Implement loaders | Data | 🟡 |
| 7 | 3.2 Fix Alembic & schema | DevOps | 🟡 |
| 8 | 3.3 Fix orphaned symbols | Data | 🟡 |
| 8 | Testing & integration | QA | 🟡 |

**Legend:** ✅ Complete | 🔄 In Progress | 🟡 Not Started | ❌ Blocked

---

## 📝 Next Steps

1. **Immediate (This Week):**
   - [ ] Complete orchestrator.py implementation
   - [ ] Create extractor interfaces and base class
   - [ ] Implement TSEExtractor

2. **Short-term (Next Week):**
   - [ ] Implement all transformers
   - [ ] Implement all validators
   - [ ] Implement all loaders
   - [ ] Create comprehensive tests

3. **Medium-term (Week 3):**
   - [ ] Fix Alembic migrations
   - [ ] Run schema validation
   - [ ] Fix orphaned symbols
   - [ ] Performance tuning

---

## 🔗 Related Documents

- [Phase 2 DI Guide](./PHASE_2_DI_GUIDE.md) - DI usage patterns
- [Architecture Fix Roadmap](../ARCHITECTURE_FIX_ROADMAP.md) - Full 16-week plan
- [API Reference](./guides/API_REFERENCE.md) - API endpoints
- [Data Storage](./data-storage.md) - Database schema

