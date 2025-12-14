# Gravity Data Ingestion Service

_Formerly the standalone **GravityTseHisPrice** project_

This service is responsible for downloading reference data, OHLCV candles, market/sector
indices, and metadata from the Tehran Stock Exchange (TSE). The code is kept almost intact
from the original repository but now lives under `services/data_ingestion` so it can ship
side-by-side with the technical analysis API.

The service writes into a local SQLite database (`data/tse_data.db`) and the unified pipeline
(`/scripts/run_stack_pipeline.py`) migrates the results into PostgreSQL (`tse_input.*` schema).

## Layout

- `src/`: CLI (`cli.py`), fetcher logic, and database helpers.
- `scripts/`: Utility scripts (db checks, fixes, helpers for `gravity_tse.py`).
- `data/`: Runtime artifacts (SQLite DB, JSON exports). Empty by default.
- `tests/` & `docs/`: Original references from GravityTseHisPrice.
- `Dockerfile`: Minimal container for scheduled/one-off ingestion runs.

## Prerequisites

1. Python 3.11+ (matches the root project requirement of Python 3.12).
2. `gravity_tse.py` (not distributed here). Drop it inside `services/data_ingestion/scripts/`.
3. Valid TSE credentials/API access for the upstream gravity_tse client.
4. Initial metadata JSON files under `data/BasicTseInformation/` (same as the original project).

## CLI Usage (inside this folder)

```bash
cd services/data_ingestion
python -m venv .venv && .\.venv\Scripts\activate  # optional
pip install -r requirements.txt
python main.py --help
```

Common commands:

```bash
python main.py init-all         # create tables + load metadata + fetch full history
python main.py load-all-prices  # incremental refresh based on last_updates
python main.py load-initial     # only bootstrap metadata JSON into SQLite
```

All commands write to `data/tse_data.db`. The unified pipeline copies the content into the
shared PostgreSQL instance that powers the analytics service.

## Container Usage

```bash
docker build -t gravity-data-ingestion services/data_ingestion
docker run --rm \
  -e GRAVITY_TSE_SCRIPT=/data/gravity_tse.py \
  -v /path/to/BasicTseInformation:/app/data/BasicTseInformation \
  -v /path/to/gravity_tse.py:/app/scripts/gravity_tse.py:ro \
  gravity-data-ingestion python main.py load-all-prices
```

Persist `data/` as a volume if you want to reuse the SQLite cache between runs.

## Integration With The Analysis Stack

1. Run the CLI (or container) to refresh `data/tse_data.db`.
2. Execute `python scripts/migrate_sqlite_to_pg.py --pg-dsn <postgres-url>` from the repo root
   to push raw data into `tse_input.*` tables.
3. Run `python scripts/run_full_batch_analysis.py --pg-dsn <postgres-url>` to compute
   historical scores.

The helper `scripts/run_stack_pipeline.py` automates all three steps (initial load + daily refresh).
