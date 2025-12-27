# Repository Structure Guide

**Purpose:** Define the organizational structure, layer boundaries, and import guidelines for Gravity Technical Analysis.

**Audience:** Developers, Architects, Code Reviewers

---

## 📂 Directory Layout

```
gravity-ta/
│
├── apps/                              # Application services
│   ├── analysis_api/                  # FastAPI microservice
│   │   ├── src/gravity_tech/
│   │   │   ├── main.py                # FastAPI entry point
│   │   │   ├── __init__.py
│   │   │   │
│   │   │   ├── api/                   # REST API Layer
│   │   │   │   ├── v1/                # API v1 endpoints
│   │   │   │   │   ├── __init__.py    # Router aggregation
│   │   │   │   │   ├── analysis.py    # /api/v1/analyze
│   │   │   │   │   ├── patterns.py    # /api/v1/patterns
│   │   │   │   │   ├── ml.py          # /api/v1/ml
│   │   │   │   │   ├── tools.py       # /api/v1/tools
│   │   │   │   │   ├── backtest.py    # /api/v1/backtest
│   │   │   │   │   ├── scenarios.py   # /api/v1/scenarios (optional)
│   │   │   │   │   ├── db_explorer.py # /api/v1/db (optional)
│   │   │   │   │   ├── health.py      # /health endpoints
│   │   │   │   │   └── auth.py        # Authentication
│   │   │   │   └── dependencies.py    # FastAPI dependency injection
│   │   │   │
│   │   │   ├── core/                  # Business Logic Layer (no external deps)
│   │   │   │   ├── domain/            # Domain entities
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── entities.py    # Dataclasses for signals, results
│   │   │   │   │   ├── enums.py       # Signal types, timeframes
│   │   │   │   │   └── value_objects.py
│   │   │   │   │
│   │   │   │   ├── indicators/        # Technical indicators (60+)
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── trend.py       # SMA, EMA, MACD, etc
│   │   │   │   │   ├── momentum.py    # RSI, Stochastic, MACD, etc
│   │   │   │   │   ├── volatility.py  # ATR, Bollinger, Keltner, etc
│   │   │   │   │   ├── cycle.py       # Dominant cycle, phase
│   │   │   │   │   ├── volume.py      # OBV, CMF, VWAP, etc
│   │   │   │   │   └── support_resistance.py  # S/R levels
│   │   │   │   │
│   │   │   │   ├── patterns/          # Pattern detection
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── candlestick.py # Candlestick patterns
│   │   │   │   │   ├── classical.py   # Head&Shoulders, Triangles, etc
│   │   │   │   │   ├── harmonic.py    # Gartley, Butterfly, Bat, Crab
│   │   │   │   │   └── elliott.py     # Elliott wave counting
│   │   │   │   │
│   │   │   │   ├── analysis/          # Analysis orchestration
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── multi_horizon.py       # 3D, 7D, 30D analysis
│   │   │   │   │   ├── five_dimensional.py   # 5D decision matrix
│   │   │   │   │   ├── volume_matrix.py      # Volume interactions
│   │   │   │   │   └── signal_engine.py      # Signal aggregation
│   │   │   │   │
│   │   │   │   └── validators/        # Input/output validation
│   │   │   │       ├── __init__.py
│   │   │   │       ├── candles.py     # OHLCV validation
│   │   │   │       └── signals.py     # Signal validation
│   │   │   │
│   │   │   ├── services/              # Use Cases & Orchestration
│   │   │   │   ├── __init__.py
│   │   │   │   ├── analysis_service.py        # Analysis use case
│   │   │   │   ├── tool_recommendation_service.py
│   │   │   │   ├── pattern_backtest_service.py
│   │   │   │   ├── scenario_analysis_service.py
│   │   │   │   ├── data_ingestor_service.py
│   │   │   │   └── cache_service.py
│   │   │   │
│   │   │   ├── infrastructure/        # External Service Adapters
│   │   │   │   ├── __init__.py
│   │   │   │   ├── contracts.py       # Abstract interfaces
│   │   │   │   │   ├── CacheBackend
│   │   │   │   │   ├── DatabaseBackend
│   │   │   │   │   └── ExternalDataService
│   │   │   │   │
│   │   │   │   ├── adapters/          # Concrete implementations
│   │   │   │   │   ├── redis_cache.py
│   │   │   │   │   ├── memory_cache.py
│   │   │   │   │   ├── postgres_db.py
│   │   │   │   │   ├── sqlite_db.py
│   │   │   │   │   └── data_service_client.py
│   │   │   │   │
│   │   │   │   ├── container.py       # Dependency injection
│   │   │   │   ├── config.py
│   │   │   │   └── exceptions.py
│   │   │   │
│   │   │   ├── middleware/            # Cross-cutting Concerns
│   │   │   │   ├── __init__.py
│   │   │   │   ├── logging.py         # Structured logging
│   │   │   │   ├── metrics.py         # Prometheus metrics
│   │   │   │   ├── tracing.py         # Distributed tracing
│   │   │   │   ├── auth.py            # JWT authentication
│   │   │   │   ├── events.py          # Event publishing
│   │   │   │   ├── resilience.py      # Circuit breaker, retry
│   │   │   │   └── error_handlers.py  # Global exception handling
│   │   │   │
│   │   │   ├── ml/                    # Machine Learning
│   │   │   │   ├── __init__.py
│   │   │   │   ├── model_registry.py  # Model loading
│   │   │   │   ├── pattern_classifier.py
│   │   │   │   ├── weight_optimizer.py
│   │   │   │   ├── backtesting.py
│   │   │   │   └── feature_extraction.py
│   │   │   │
│   │   │   └── config/                # Configuration
│   │   │       ├── __init__.py
│   │   │       ├── unified_settings.py  # Single config source
│   │   │       ├── constants.py
│   │   │       └── defaults.py
│   │   │
│   │   ├── tests/                     # Test Suite
│   │   │   ├── unit/                  # Unit tests (no external deps)
│   │   │   ├── integration/           # Integration tests (DB, cache)
│   │   │   ├── api/                   # API endpoint tests
│   │   │   ├── fixtures/              # Test data and fixtures
│   │   │   ├── conftest.py            # pytest configuration
│   │   │   └── README.md
│   │   │
│   │   ├── ml_models/                 # ML model artifacts (local only)
│   │   │   ├── pattern_classifier_*.pkl
│   │   │   ├── config_*.json
│   │   │   └── dimension_weights_*.json
│   │   │
│   │   └── README.md                  # API service README
│   │
│   └── data_pipeline/                 # (Phase 3) Data ETL service
│       └── (structure TBD)
│
├── services/                          # Shared Services
│   ├── data_ingestion/                # Data ETL (legacy, for migration)
│   │   ├── src/
│   │   ├── scripts/
│   │   └── README.md
│   │
│   └── README.md
│
├── scripts/                           # Utility Scripts
│   ├── run_full_pipeline.py           # ⚠️ DEPRECATED (use data_pipeline)
│   ├── database/
│   │   ├── validate_schema.py
│   │   └── reset_db.py
│   ├── maintenance/
│   │   ├── cleanup_logs.py
│   │   └── optimize_indexes.py
│   ├── utils/
│   │   ├── env_validator.py
│   │   └── health_check.py
│   └── README.md
│
├── ml_models/                         # ML Model Storage (root)
│   ├── multi_horizon/
│   │   ├── config_btcusdt.json
│   │   └── dimension_weights_btcusdt.json
│   ├── pattern_classifier_btcusdt.pkl
│   └── .gitkeep
│
├── data/                              # Local Data (git ignored)
│   ├── TechAnalysis.db                # SQLite for local development
│   ├── tse_data.db                    # TSE reference data
│   ├── cache/
│   └── .gitkeep
│
├── docs/                              # Documentation
│   ├── INDEX.md                       # Documentation index
│   ├── REPOSITORY_STRUCTURE.md        # This file
│   ├── architecture/
│   │   ├── SYSTEM_ARCHITECTURE_DIAGRAMS.md
│   │   ├── DATA_SERVICE_INTEGRATION.md
│   │   └── SIGNAL_CALCULATION.md
│   ├── guides/
│   │   ├── QUICK_START.md
│   │   ├── API_REFERENCE.md
│   │   ├── CONTRIBUTING.md
│   │   ├── FIVE_DIMENSIONAL_DECISION_GUIDE.md
│   │   ├── VOLUME_MATRIX_GUIDE.md
│   │   └── TREND_ANALYSIS_SUMMARY.md
│   ├── operations/
│   │   ├── DEPLOYMENT_GUIDE.md
│   │   ├── RUNBOOK.md
│   │   ├── SECURITY.md
│   │   └── MONITORING.md
│   └── processes/
│       ├── TECHNICAL_ANALYSIS.md
│       ├── PATTERN_DETECTION.md
│       ├── ML_PREDICTION.md
│       ├── TOOL_RECOMMENDATION.md
│       ├── SCENARIO_ANALYSIS.md
│       └── BACKTEST.md
│
├── infra/                             # Infrastructure
│   ├── k8s/                           # Kubernetes
│   │   ├── base/
│   │   ├── overlays/
│   │   └── helm/
│   └── terraform/                     # Infrastructure as Code
│
├── deployment/                        # Deployment Config
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── .dockerignore
│   └── kubernetes/
│
├── experiments/                       # Experimental Code
│   ├── new_indicators/
│   └── ml_research/
│
├── alembic/                           # Database Migrations
│   ├── versions/
│   ├── env.py
│   └── script.py.mako
│
├── .github/                           # GitHub Configuration
│   └── workflows/
│       ├── ci.yml
│       ├── security.yml
│       └── deploy.yml
│
├── requirements/                      # Dependency Files
│   ├── base.txt
│   ├── dev.txt
│   ├── ml.txt
│   └── prod.txt
│
├── pyrightconfig.json                 # Type checking
├── pytest.ini                         # Test configuration
├── pyproject.toml                     # Project metadata
├── .gitignore                         # Git ignore (with security rules)
├── .env.example                       # Example environment
├── README.md                          # Main README
├── ARCHITECTURE_FIX_ROADMAP.md       # Architecture improvement plan
└── CONTRIBUTING.md                    # Contribution guidelines
```

---

## 🏗️ Layer Architecture

### Layer Definitions

| Layer | Location | Purpose | Can Import From | Cannot Import |
|-------|----------|---------|---|---|
| **Domain** | `core/domain/` | Pure business entities, enums, value objects | `stdlib` | External libraries |
| **Indicators** | `core/indicators/` | Technical analysis calculations | `domain`, `numpy`, `pandas` | `api`, `services` |
| **Patterns** | `core/patterns/` | Pattern detection algorithms | `domain`, `indicators` | `api`, `services` |
| **Analysis** | `core/analysis/` | Analysis orchestration | `domain`, `indicators`, `patterns` | `api` |
| **Services** | `services/` | Use cases, business logic | `core/*`, `infrastructure` | `api/v1` |
| **Infrastructure** | `infrastructure/` | External service adapters | `core/domain` | `api`, `services` |
| **API** | `api/v1/` | HTTP endpoints | `services`, `core`, `infrastructure` | Nothing (leaves) |
| **Middleware** | `middleware/` | Cross-cutting concerns | Any layer | N/A |

### Import Rules

#### ✅ **DO's**

```python
# ✅ API importing from services
from gravity_tech.services.analysis_service import TechnicalAnalysisService

# ✅ Services importing from core
from gravity_tech.core.indicators.trend import calculate_sma
from gravity_tech.core.domain.entities import Signal

# ✅ Indicators importing indicators
from gravity_tech.core.indicators.trend import calculate_ema

# ✅ Cross-layer imports in same layer
from gravity_tech.core.indicators import trend, momentum

# ✅ Infrastructure implementing contracts
from gravity_tech.infrastructure.contracts import CacheBackend

# ✅ Middleware anywhere (cross-cutting)
from gravity_tech.middleware.logging import logger
```

#### ❌ **DON'Ts**

```python
# ❌ Core importing from API
from gravity_tech.api.v1.analysis import AnalysisRequest  # WRONG!

# ❌ Indicators importing services
from gravity_tech.services.cache_service import cache_manager  # WRONG!

# ❌ Services importing from API
from gravity_tech.api.v1.patterns import router  # WRONG!

# ❌ Hardcoded external dependencies in core
import redis  # WRONG! Use contracts instead

# ❌ Circular imports
# api → services → infrastructure → api (BAD!)
```

---

## 🔌 Component Interactions

### Example: `/api/v1/analyze` Request Flow

```
1. HTTP Request
   ↓
2. api/v1/analysis.py::analyze()
   - Validate request (domain entities)
   - Dependency inject service
   ↓
3. services/analysis_service.py::TechnicalAnalysisService
   - Call core analysis layer
   ↓
4. core/analysis/multi_horizon.py::MultiHorizonAnalyzer
   - Call indicators
   - Call pattern detection
   ↓
5. core/indicators/*.py
   - Pure calculations (no I/O)
   ↓
6. services/analysis_service.py
   - Cache result (infrastructure)
   - Publish event (infrastructure)
   ↓
7. api/v1/analysis.py
   - Return response
```

---

## 📦 Production vs Experimental Code

### Production Code

- **Location:** `apps/analysis_api/src/gravity_tech/`
- **Status:** Tested, documented, deployed
- **Import Rule:** Only import production code in production
- **Breaking Changes:** Discussed in architecture board

### Experimental Code

- **Location:** `experiments/`
- **Status:** Draft, research, POC
- **Import Rule:** Can import production code, but not vice-versa
- **Migration:** Move to production when ready, add tests first

### Legacy/Deprecated Code

- **Location:** `services/data_ingestion/` (during migration)
- **Status:** Being migrated to `apps/data_pipeline/`
- **Import Rule:** Avoid in new code
- **Deprecation:** Documented with timeline

---

## 🧪 Testing Structure

```
tests/
├── unit/                           # No external deps
│   ├── test_indicators/
│   ├── test_patterns/
│   ├── test_domain/
│   └── conftest.py
│
├── integration/                    # With dependencies
│   ├── test_services/
│   ├── test_cache/
│   ├── test_database/
│   └── conftest.py
│
├── api/                            # Endpoint tests
│   ├── test_analysis_endpoint.py
│   ├── test_patterns_endpoint.py
│   └── conftest.py
│
├── fixtures/                       # Shared test data
│   ├── candle_fixtures.py
│   ├── symbol_fixtures.py
│   └── result_fixtures.py
│
└── conftest.py                     # Root pytest config
```

**Testing Philosophy:**
- **Unit:** Fast, isolated, in-memory (no I/O)
- **Integration:** With real DB/cache, slower but realistic
- **API:** End-to-end HTTP tests
- **Coverage:** Target 80%+

---

## 🎯 Conventions

### File Naming

- `module_name.py` - lowercase with underscores
- `ClassName` - PascalCase for classes
- `function_name` - lowercase with underscores
- `CONSTANT_NAME` - UPPERCASE for constants

### Module Structure

```python
"""
Brief module description.

Classes:
    ClassName: What it does
    
Functions:
    function_name: What it does
"""

from __future__ import annotations

# Standard library imports
import os
from typing import Optional, List

# Third-party imports
import numpy as np
from pydantic import BaseModel

# Local imports
from gravity_tech.core.domain.entities import Signal
from gravity_tech.infrastructure.contracts import CacheBackend

# Module-level constants
DEFAULT_TTL = 300

# Classes
class ClassName:
    """Documentation"""
    pass

# Functions
def function_name() -> Optional[int]:
    """Documentation"""
    pass

# Main guard (if applicable)
if __name__ == "__main__":
    pass
```

---

## 🚀 Adding New Features

### Process

1. **Determine Layer:** Where does the logic belong?
   - Pure calculation? → `core/`
   - API endpoint? → `api/v1/`
   - External service? → `infrastructure/adapters/`

2. **Implement:** Following import rules

3. **Test:** Unit + Integration + API tests

4. **Document:** Docstrings + README update

5. **PR Review:** Check layer boundaries

### Example: Add New Indicator

```
1. File: core/indicators/new_indicator.py
   - Pure calculation function
   - No external deps (except numpy/pandas)

2. Export: core/indicators/__init__.py
   from .new_indicator import calculate_new_indicator

3. Test: tests/unit/test_indicators/test_new_indicator.py
   - Test calculation
   - Test edge cases

4. Service: services/analysis_service.py
   - Call new indicator in analysis pipeline

5. API: api/v1/analysis.py
   - Include in response

6. Docs: Update indicator list in README
```

---

## 📋 Checklist Before Committing

- [ ] Code in correct layer (respecting import rules)
- [ ] Docstrings added
- [ ] Type hints complete
- [ ] Tests added/updated (80%+ coverage target)
- [ ] No circular imports
- [ ] No hardcoded paths (use config)
- [ ] No debug print statements
- [ ] Logging instead of print
- [ ] Follow naming conventions
- [ ] README/docs updated if needed

---

## 🤝 Questions?

- **Architecture questions:** See `docs/architecture/`
- **API questions:** See `docs/guides/API_REFERENCE.md`
- **Setup questions:** See `docs/guides/QUICK_START.md`
- **Contributing:** See `CONTRIBUTING.md`

---

**Document Version:** 1.0  
**Last Updated:** December 26, 2025  
**Maintainer:** Architecture Team
