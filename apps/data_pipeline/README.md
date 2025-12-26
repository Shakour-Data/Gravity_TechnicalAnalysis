# Gravity Data Pipeline

**Unified ETL pipeline for Gravity Technical Analysis**

This module consolidates all data extraction, transformation, and loading operations from:
- `services/data_ingestion/` (legacy)
- `scripts/run_full_pipeline.py` (legacy)
- `scripts/migrate_*.py` (legacy)

## Architecture

```
DataPipeline (Orchestrator)
├── Extract    → Get data from TSE API
├── Transform  → Normalize and clean
├── Validate   → Check data quality
├── Deduplicate → Remove duplicates
└── Load      → Persist to database
```

## Quick Start

### Installation

```bash
pip install -r requirements/base.txt
```

### Configuration

Set environment variables:
```bash
export SOURCE_DB_URL=sqlite:///./data/gravity_source.db
export TARGET_DB_URL=sqlite:///./data/gravity.db
export PIPELINE_LOG_LEVEL=INFO
export PIPELINE_BATCH_SIZE=500
```

Or create `.env.pipeline`:
```env
SOURCE_DB_URL=sqlite:///./data/gravity_source.db
TARGET_DB_URL=sqlite:///./data/gravity.db
TSE_API_BASE_URL=https://api.tse.ir/api
PIPELINE_LOG_LEVEL=INFO
PIPELINE_BATCH_SIZE=500
PIPELINE_MAX_WORKERS=4
```

### Run Full Pipeline

```python
from gravity_pipeline.orchestrator import DataPipeline, PipelineConfig
from gravity_pipeline.config import PipelineEnvironmentConfig
import asyncio

config = PipelineConfig(**PipelineEnvironmentConfig().as_dict)
pipeline = DataPipeline(config)

result = await pipeline.run_full(
    symbols=['SYMBOL1', 'SYMBOL2'],
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31)
)

print(pipeline.get_pipeline_stats())
```

### Run Specific Stage

```python
# Skip validation, run others
result = await pipeline.run_full(
    skip_stages=[PipelineStage.VALIDATE]
)

# Run only extraction
candles = await pipeline._extract(
    symbols=['SYMBOL1'],
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31)
)
```

## Schema Validation

Before loading data, validate target database schema:

```python
from gravity_pipeline.validators import SchemaValidator

validator = SchemaValidator(engine)
is_valid = validator.validate_tables_exist(['candles', 'signals'])
```

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

## Stage Details

### Extract
- **Input:** Symbols, date range
- **Output:** List of OHLCV records
- **Source:** TSE API

### Transform
- **Input:** Raw OHLCV records
- **Processing:**
  - Normalize column names
  - Convert data types
  - Calculate derived fields
  - Handle missing values
- **Output:** Clean records

### Validate
- **Input:** Clean records
- **Checks:**
  - Required fields present
  - Data type compliance
  - Value range validation
  - Anomaly detection
- **Output:** Valid records list

### Deduplicate
- **Input:** Valid records
- **Processing:**
  - Identify duplicates by symbol+timestamp
  - Keep most recent
  - Log removed
- **Output:** Unique records

### Load
- **Input:** Unique records
- **Processing:**
  - Batch insert/update
  - Handle conflicts
  - Maintain referential integrity
- **Output:** Records in database

## Migration from Legacy Code

### From `scripts/run_full_pipeline.py`

Before:
```python
python scripts/run_full_pipeline.py \
  --source-db sqlite:///gravity_source.db \
  --target-db sqlite:///gravity.db
```

After:
```python
from gravity_pipeline.orchestrator import DataPipeline, PipelineConfig

config = PipelineConfig(
    source_db_url="sqlite:///gravity_source.db",
    target_db_url="sqlite:///gravity.db"
)
pipeline = DataPipeline(config)
await pipeline.run_full()
```

### From `services/data_ingestion/`

Legacy service still works but imports are redirected. Gradual migration recommended:
1. Use new `DataPipeline` for new scripts
2. Keep legacy imports working (with deprecation warnings)
3. Migrate existing code over 1-2 releases
4. Remove legacy code in v2.0

## Observability

All stages log structured data:
```json
{
  "event": "pipeline_completed",
  "stages_completed": ["extract", "transform", "validate", "load"],
  "total_duration": 45.23,
  "overall_success_rate": 99.5
}
```

View pipeline statistics:
```python
stats = pipeline.get_pipeline_stats()
print(f"Processed: {stats['total_records_processed']}")
print(f"Success Rate: {stats['overall_success_rate']:.1f}%")
```

## Troubleshooting

**Connection Error:**
```
pipeline_failed: Could not connect to source database
```
- Check `SOURCE_DB_URL` environment variable
- Verify database file exists or PostgreSQL is running

**Validation Failed:**
```
validate_stage_failed: Data validation failed
```
- Check data quality in source
- Review error log for specific field issues
- May need to adjust transformation logic

**Load Failed:**
```
load_stage_failed: Could not insert records
```
- Check target database schema
- Use `SchemaValidator.get_table_info()` to debug
- May need to run migrations first

## Performance Tuning

Adjust for your environment:
```python
config = PipelineConfig(
    source_db_url="...",
    target_db_url="...",
    batch_size=1000,      # Larger = faster but more memory
    max_workers=8,        # More workers = parallel extraction
    retry_count=5         # More retries = resilience
)
```

## See Also

- `docs/processes/INGESTION_METRICS.md` - Monitoring
- `docs/REPOSITORY_STRUCTURE.md` - Architecture
- `ARCHITECTURE_FIX_ROADMAP.md` - Phased implementation

---

**Status:** Phase 3 Implementation  
**Last Updated:** December 26, 2025  
**Maintainer:** Data Team
