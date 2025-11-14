# 📁 Gravity Technical Analysis - Project Structure

**Version:** 1.2.0  
**Last Updated:** November 14, 2025  
**Status:** Production Ready ✅

---

## 🎯 Overview

این میکروسرویس یک سیستم تحلیل تکنیکال کامل با بیش از 60 اندیکاتور، الگوهای قیمتی، و سیستم توصیه ابزار مبتنی بر Machine Learning است.

---

## 📂 Root Directory Structure

```
Gravity_TechAnalysis/
├── 📄 Configuration Files
│   ├── .editorconfig         # Editor configuration
│   ├── .env.example          # Environment variables template
│   ├── .gitignore            # Git ignore rules
│   ├── .yamllint             # YAML linting configuration
│   ├── pyproject.toml        # Python project metadata & dependencies
│   ├── requirements.txt      # Python dependencies
│   └── VERSION               # Current version (1.2.0)
│
├── 📄 Docker & Deployment
│   ├── Dockerfile            # Container image definition
│   ├── docker-compose.yml    # Multi-container setup
│   ├── k8s/                  # Kubernetes manifests (10 files)
│   └── helm/                 # Helm charts (2 files)
│
├── 📄 Documentation
│   ├── README.md             # Main documentation
│   ├── CHANGELOG.md          # Version history
│   ├── CONTRIBUTING.md       # Contribution guidelines
│   ├── LICENSE               # MIT License
│   ├── RELEASE_NOTES_v1.0.0.md
│   ├── RELEASE_NOTES_v1.1.0.md
│   ├── RELEASE_NOTES_v1.2.0.md
│   └── docs/                 # Detailed documentation (39 files)
│
├── 📁 Source Code
│   └── src/gravity_tech/     # Main application package (131 files)
│
├── 📁 Database
│   ├── database/             # Database schemas & managers (2 files)
│   └── setup_database.py     # Database initialization script
│
├── 📁 Machine Learning
│   ├── ml/                   # ML modules (1 file)
│   └── ml_models/            # Trained models (5 files)
│
├── 📁 Testing
│   └── tests/                # Test suite (31 files)
│
├── 📁 Automation
│   ├── .github/workflows/    # CI/CD pipelines (3 files)
│   └── scripts/              # Utility scripts (4 files)
│
└── 📁 Data (gitignored)
    └── data/                 # Local data storage
```

---

## 🔍 Detailed Directory Breakdown

### 1️⃣ **src/gravity_tech/** (131 files)
**Purpose:** Main application source code

```
src/gravity_tech/
├── analysis/              # Market analysis modules
│   ├── market_phase.py
│   └── scenario_analysis.py
│
├── api/                   # REST API endpoints
│   └── v1/
│       ├── analysis.py
│       ├── health.py
│       ├── indicators.py
│       └── tools.py       # Tool recommendation API
│
├── indicators/            # 60+ Technical indicators
│   ├── cycle.py
│   ├── momentum.py
│   ├── support_resistance.py
│   ├── trend.py
│   ├── volatility.py
│   └── volume.py
│
├── ml/                    # Machine Learning components
│   ├── ml_indicator_weights.py
│   ├── ml_dimension_weights.py
│   ├── ml_tool_recommender.py
│   └── scenario_weight_optimizer.py
│
├── middleware/            # HTTP middleware
│   ├── auth.py           # JWT authentication
│   ├── events.py         # Event-driven messaging
│   ├── logging.py        # Structured logging
│   ├── resilience.py     # Circuit breaker, retry
│   ├── security.py       # Rate limiting, CORS
│   └── tracing.py        # OpenTelemetry
│
├── models/                # Data models
│   ├── analysis_models.py
│   ├── indicator_models.py
│   └── response_models.py
│
├── patterns/              # Chart pattern detection
│   ├── candlestick.py
│   └── classical_patterns.py
│
├── services/              # Business logic
│   ├── analysis_service.py
│   ├── fast_indicators.py
│   ├── performance_optimizer.py
│   └── tool_recommendation_service.py
│
├── utils/                 # Utilities
│   ├── cache.py
│   ├── helpers.py
│   └── validators.py
│
├── config/                # Configuration management
│   └── settings.py
│
└── main.py               # FastAPI application entry point
```

### 2️⃣ **database/** (2 files)
- `historical_manager.py` - Historical data management
- `tool_performance_history.sql` - PostgreSQL schema (optional)
- **DatabaseManager** with auto-detection: PostgreSQL → SQLite → JSON

### 3️⃣ **ml_models/** (5 files)
- Trained LightGBM/XGBoost models
- Model metadata and versioning
- Feature extractors

### 4️⃣ **tests/** (31 files)
```
tests/
├── unit/              # Unit tests (95%+ coverage)
├── integration/       # Integration tests
├── contract/          # API contract tests (Pact)
└── load/              # Load tests (Locust)
```

### 5️⃣ **docs/** (39 files)
```
docs/
├── api/               # API documentation
├── architecture/      # System design
├── guides/            # User guides
├── operations/        # DevOps & deployment
└── team/              # Team workflows
```

### 6️⃣ **k8s/** (10 files)
- `deployment.yaml` - Application deployment
- `service.yaml` - Service definition
- `ingress.yaml` - Ingress routing
- `hpa.yaml` - Horizontal Pod Autoscaler
- `configmap.yaml`, `secret.yaml`
- `monitoring.yaml` - Prometheus/Grafana
- `redis.yaml` - Redis cache
- `namespace.yaml`, `rbac.yaml`

### 7️⃣ **scripts/** (4 files)
- Database migration scripts
- Data processing utilities
- Development helpers

---

## 🚀 Quick Start

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Setup database (optional)
python setup_database.py

# Run locally
cd src && uvicorn gravity_tech.main:app --reload
```

### Docker
```bash
docker-compose up -d
```

### Kubernetes
```bash
kubectl apply -f k8s/
# or with Helm
helm install gravity-tech helm/technical-analysis/
```

---

## 📊 Statistics

| Metric | Count |
|--------|-------|
| **Total Files** | 228 |
| **Source Files** | 131 |
| **Test Files** | 31 |
| **Documentation** | 39 |
| **Configuration** | 27 |
| **Technical Indicators** | 60+ |
| **API Endpoints** | 15+ |
| **ML Models** | 5 |

---

## 🧹 Cleanup History

**Removed in v1.2.0:**
- ✅ `deprecated/` folder (3 files)
- ✅ `examples/` folder (11 files)
- ✅ 11 `DAY_*_COMPLETION_REPORT.md` files
- ✅ 3 `CODE_REVIEW_v1.1.0_Day*.md` files
- ✅ 8 temporary/release files
- ✅ All `__pycache__/` directories
- ✅ `.pytest_cache/`, `.coverage`
- ✅ `test_complete_system.py` (moved to tests/)

**Total Removed:** 37 files, 15,345 lines of code

---

## 🎯 Production Readiness

- ✅ Clean code structure
- ✅ Comprehensive test coverage (95%+)
- ✅ Docker & Kubernetes ready
- ✅ CI/CD pipelines configured
- ✅ Security middleware (JWT, rate limiting)
- ✅ Observability (OpenTelemetry, Prometheus)
- ✅ Database fallback system (PostgreSQL → SQLite → JSON)
- ✅ ML-based tool recommendation
- ✅ 60+ technical indicators optimized with Numba
- ✅ API documentation (OpenAPI/Swagger)

---

## 📝 Version Information

- **Current Version:** 1.2.0
- **Python:** 3.11+
- **FastAPI:** Latest
- **Database:** PostgreSQL (optional), SQLite (fallback), JSON (final)
- **ML Stack:** LightGBM, XGBoost, scikit-learn
- **Deployment:** Docker, Kubernetes, Helm

---

## 📞 Support

- **Documentation:** `/docs`
- **API Docs:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`
- **GitHub:** Gravity_TechAnalysis

---

**Last Updated:** November 14, 2025  
**Maintained By:** Gravity Tech Team
