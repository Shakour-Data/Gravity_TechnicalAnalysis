# Gravity Stack Overview

This repository now contains both services that previously lived in
separate projects:

| Layer | Location | Responsibility |
|-------|----------|----------------|
| Data Ingestion | `services/data_ingestion` | Fetch raw OHLCV, indices, company metadata from TSE, persist into SQLite (`tse_data.db`). |
| Technical Analysis | `src/` (FastAPI service) | Consume the raw data (via PostgreSQL), run ML pipelines, expose REST/WebSocket APIs. |

## Data Flow

```
gravity_tse client ──► data_ingestion (SQLite) ──► migrate_sqlite_to_pg.py ──► PostgreSQL (tse_input)
                                                                               │
                                                                               └──► analysis batch ─► tech_analysis.*
```

1. `services/data_ingestion` uses `gravity_tse.py` plus JSON metadata to maintain
   `services/data_ingestion/data/tse_data.db`.
2. `scripts/run_stack_pipeline.py` orchestrates:
   - ingestion CLI (`init-all` or `load-all-prices`);
   - schema creation in PostgreSQL via `scripts/postgres_schema.sql`;
   - migration into `tse_input.*` tables;
   - historical analysis batch (`scripts/run_full_batch_analysis.py`).
3. The FastAPI app reads from `tech_analysis.*` tables for serving APIs.

## Docker Services

`docker-compose.stack.yml` provisions:

1. `postgres`: shared database (`tech_analysis` DB, schemas `tse_input` + `tech_analysis`).
2. `analysis-api`: FastAPI service (exposes port `8000`).
3. `ingestion-runner`: runs `scripts/run_stack_pipeline.py --mode daily` in a loop
   (default interval = 24h) and shares `services/data_ingestion/data` as a volume.

> ⚠️ Drop `gravity_tse.py` and metadata JSON files under `services/data_ingestion/scripts` and
> `services/data_ingestion/data/BasicTseInformation/` before building the ingestion image.

## Manual Runbook

1. `docker compose -f docker-compose.stack.yml up -d postgres`.
2. `python scripts/run_stack_pipeline.py --mode init --pg-dsn postgresql://gravity:gravity_db_pass@localhost:5544/tech_analysis`.
3. `docker compose -f docker-compose.stack.yml up -d analysis-api` to expose APIs.
4. Schedule `python scripts/run_stack_pipeline.py --mode daily ...` (or rely on the compose service).

Use `--skip-*` flags on `run_stack_pipeline.py` when you only want parts of the flow.
