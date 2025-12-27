# Project Structure

**Version:** 1.0.0  
**Last Updated:** 2025-12-26

## Directory Map
```
Gravity_TechnicalAnalysis/
├─ apps/
│  ├─ analysis_api/
│  │  ├─ src/gravity_tech/       # FastAPI routers, services, core domain, ML, database helpers
│  │  └─ tests/                  # Unit/integration/slow suites
│  ├─ data_pipeline/             # Data ETL pipeline
│  └─ support_dashboard/         # Support dashboard frontend
├─ services/
│  └─ data_ingestion/            # TSE data ETL service
├─ ml_models/                    # Saved ML artifacts (indicator/dimension weights, classifiers)
├─ scripts/                      # Utilities: populate_last90.py, run_full_pipeline.py, migrations/maintenance
├─ docs/                         # Project documentation
└─ data/                         # SQLite databases (gitignored)
```

## Key Components
- **API (`apps/analysis_api/src/gravity_tech/api/v1`)**: `analysis.py`, `patterns.py`, `ml.py`, `tools.py`, `backtest.py`, `db_explorer.py`.
- **Services (`apps/analysis_api/src/gravity_tech/services`)**: `analysis_service.py`, `cache_service.py`, `data_ingestor_service.py`, `tool_recommendation_service.py`, `fast_indicators.py`, `signal_engine.py`.
- **Indicators (`apps/analysis_api/src/gravity_tech/core/indicators`)**: Trend, Momentum, Volume, Volatility, Cycle, Support/Resistance calculators with confidence scores.
- **ML (`apps/analysis_api/src/gravity_tech/ml`)**: `complete_analysis_pipeline.py`, `five_dimensional_decision_matrix.py`, `volume_dimension_matrix.py`, multi-horizon analyzers/feature extractors, `pattern_classifier.py`, `backtesting.py`.
- **Patterns (`apps/analysis_api/src/gravity_tech/patterns`)**: Harmonic, classical, Elliott, candlestick detection utilities.
- **Database (`apps/analysis_api/src/gravity_tech/database/`)**: Canonical schema files plus `DatabaseManager`/`HistoricalScoreManager`.
- **Scripts (`scripts/`)**: Operational helpers. `run_full_pipeline.py` runs the full TSE→analysis→TechAnalysis.db flow; other populate/maintenance scripts live here.
- **Clients**: `apps/analysis_api/src/gravity_tech/clients/data_service_client.py` for adjusted OHLCV retrieval via HTTP + Redis cache.
- **Middleware**: CORS, discovery, security, tracing, and metrics helpers.
- **Feature Flags (see `settings`)**: `enable_scenarios` toggles `/api/v1/scenarios/*`; `expose_db_explorer` toggles `/api/v1/db/*`.
- **Domain Entities (`apps/analysis_api/src/gravity_tech/core/domain/entities/`)**: Immutable dataclasses for Candle, Signal, IndicatorResult, PatternResult, etc.
- **Analysis (`apps/analysis_api/src/gravity_tech/core/analysis/`)**: Market phase analysis, scenario analysis engines.

## Notes
- API/tests assume a SQLite backend by default; configure environment variables for PostgreSQL if needed.
- Redis and ingestion flags live in `.env`; defaults are safe for local development.
- ML artifacts expected in `ml_models/` for ML/Pattern endpoints.
- **Removed files (Dec 26, 2025)**:
  - `apps/analysis_api/src/gravity_tech/core/indicators/volume_day3.py` (orphaned, replaced by volume.py)
  - `apps/analysis_api/src/gravity_tech/models/schemas_backup.py` (deprecated Phase 2.1 compat layer)
