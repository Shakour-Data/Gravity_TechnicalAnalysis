# Copilot Instructions for Gravity Technical Analysis

## Project Overview
- **Purpose:** Unified platform for technical analysis, pattern detection, and tool recommendation using FastAPI, classic indicators, harmonic patterns, and ML models.
- **Major Components:**
  - `apps/analysis_api/`: Main FastAPI/ML service (see `src/` for core logic)
  - `services/data_ingestion/`: ETL for TSE market data (SQLite → PostgreSQL)
  - `scripts/`: Pipeline, ETL, and utility scripts
  - `data/`: Local DBs, exports, and processed data

## Architecture & Data Flow
- **API Layer:** FastAPI endpoints in `apps/analysis_api/src/gravity_tech/main.py` route to service logic.
- **Domain Layer:** All core business logic/entities in `src/core/domain/entities/` (immutable, validated dataclasses, no external deps)
- **ETL:** `services/data_ingestion/` ingests TSE data, stores in SQLite, then syncs to PostgreSQL for analysis.
- **ML Models:** Place model files in `ml_models/` (see README for required filenames)
- **Deployment:** Use `docker-compose.stack.yml` for local stack (API, ingestion worker, PostgreSQL)

## Developer Workflows
- **Install:** `pip install -r requirements.txt` (Python 3.12)
- **Run API:**
  - Set `PYTHONPATH=apps/analysis_api/src`
  - `uvicorn gravity_tech.main:app --host 0.0.0.0 --port 8000 --reload`
- **Test:**
  - `pytest` (all tests)
  - Use VSCode tasks for targeted test runs (see `.vscode/tasks.json`)
- **Full Pipeline:**
  - `python scripts/run_full_pipeline.py --source-db <src> --target-db <dst> [--limit N]`
- **Data Ingestion:**
  - `python main.py create-db` / `load-initial` / `load-all-prices` in `services/data_ingestion/web/`

## Project Conventions
- **Domain Entities:** Only use imports from `src/core/domain/entities/` (not `models/schemas.py`)
- **Immutability:** All core entities are frozen dataclasses, validated in `__post_init__`
- **Type Safety:** Full type hints required
- **No Frameworks in Domain:** No FastAPI/Pydantic in domain layer
- **ML Model Loading:** Endpoints requiring ML expect model files in `ml_models/` with specific names (see README)
- **API Docs:** Interactive docs at `/api/docs` when running API

## Key References
- Main API: `apps/analysis_api/src/gravity_tech/main.py`
- Domain Entities: `apps/analysis_api/src/gravity_tech/core/domain/entities/`
- ETL: `services/data_ingestion/`
- Deployment: `docker-compose.stack.yml`, `docs/operations/DEPLOYMENT_GUIDE.md`
- API Reference: `docs/guides/API_REFERENCE.md`
- Architecture: `docs/architecture/SYSTEM_ARCHITECTURE_DIAGRAMS.md`

## Examples
- See `README.md` and `docs/guides/QUICK_START.md` for end-to-end usage and sample requests.
- For new indicators/patterns, follow the dataclass/enum patterns in `domain/entities/` and update imports accordingly.

---
_Last updated: 2025-12-15_
