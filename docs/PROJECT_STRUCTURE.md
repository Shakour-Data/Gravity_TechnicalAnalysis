# Project Structure

**Version:** 1.0.0  
**Last Updated:** 2025-12-10

## Directory Map
```
Gravity_TechnicalAnalysis/
├─ src/
│  └─ gravity_tech/       # FastAPI routers, services, core domain, ML, database helpers
├─ ml_models/             # Saved ML artifacts (indicator/dimension weights, classifiers)
├─ scripts/               # Utilities: populate_last90.py, run_full_pipeline.py, migrations/maintenance
├─ tests/                 # Unit/integration/slow suites
├─ docs/                  # Project documentation
└─ data/                  # SQLite databases (gitignored)
```

## Key Components
- **API (`src/gravity_tech/api/v1`)**: `analysis.py`, `patterns.py`, `ml.py`, `tools.py`, `backtest.py`, `db_explorer.py`.
- **Services (`src/gravity_tech/services`)**: `analysis_service.py`, `cache_service.py`, `data_ingestor_service.py`, `tool_recommendation_service.py`, `fast_indicators.py`, `signal_engine.py`.
- **Indicators (`src/gravity_tech/core/indicators`)**: Trend, Momentum, Volume, Volatility, Cycle, Support/Resistance calculators with confidence scores.
- **ML (`src/gravity_tech/ml`)**: `complete_analysis_pipeline.py`, `five_dimensional_decision_matrix.py`, `volume_dimension_matrix.py`, multi-horizon analyzers/feature extractors, `pattern_classifier.py`, `backtesting.py`.
- **Patterns (`src/gravity_tech/patterns`)**: Harmonic, classical, Elliott, candlestick detection utilities.
- **Database (`src/gravity_tech/database/`)**: Canonical schema files plus `DatabaseManager`/`HistoricalScoreManager`. Legacy `database/` folder removed to avoid duplication.
- **Scripts (`scripts/`)**: Operational helpers. `run_full_pipeline.py` runs the full TSE→analysis→TechAnalysis.db flow; other populate/maintenance scripts live here.
- **Clients**: `clients/data_service_client.py` for adjusted OHLCV retrieval via HTTP + Redis cache.
- **Middleware**: CORS, discovery, security, tracing, and metrics helpers.
- **Feature Flags (see `settings`)**: `enable_scenarios` toggles `/api/v1/scenarios/*`; `expose_db_explorer` toggles `/api/v1/db/*`.

## Notes
- API/tests assume a SQLite backend by default; configure environment variables for PostgreSQL if needed.
- Redis and ingestion flags live in `.env`; defaults are safe for local development.
- ML artifacts expected in `ml_models/` for ML/Pattern endpoints.
