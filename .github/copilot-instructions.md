# Gravity Technical Analysis - AI Agent Instructions

This document guides AI coding agents (like GitHub Copilot) in understanding and contributing to the Gravity_TechnicalAnalysis codebase.

## Big Picture Architecture

Gravity_TechnicalAnalysis is a FastAPI-based microservice for technical analysis and ML-powered trading signals. It processes OHLCV financial data to generate indicators, detect patterns, and provide actionable recommendations.

### Core Components
- **API Layer** (`src/gravity_tech/api/v1/`): REST endpoints for analysis, patterns, ML inference, backtesting
- **Services Layer** (`src/gravity_tech/services/`): Business logic orchestration (TechnicalAnalysisService, PatternBacktester)
- **Core Domain** (`src/gravity_tech/core/`): Indicators (trend/momentum/volatility/cycle/volume/support-resistance), patterns (harmonic/candlestick/Elliott)
- **ML Layer** (`src/gravity_tech/ml/`): 5-dimensional decision matrix, pattern classification, feature extraction
- **Data Layer** (`src/gravity_tech/database/`): PostgreSQL/SQLite integration with Alembic migrations

### Data Flow
1. Raw OHLCV data → Feature extraction (indicators + patterns)
2. Multi-dimensional analysis (trend/momentum/volatility/cycle/support-resistance)
3. Volume-dimension matrix interactions
4. 5-dimensional decision matrix → Final signal + confidence
5. Optional: Results ingestion to historical database

### Key Structural Decisions
- **Clean Architecture**: Domain entities separate from infrastructure
- **Optional Dependencies**: Redis cache, external data service, message brokers (Kafka/RabbitMQ)
- **Feature Flags**: `ENABLE_SCENARIOS`, `EXPOSE_DB_EXPLORER`, `ENABLE_DATA_INGESTION`
- **Multi-horizon Analysis**: Simultaneous analysis across timeframes

## Critical Developer Workflows

### Running the Service
```bash
set PYTHONPATH=src
uvicorn gravity_tech.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive docs: `http://localhost:8000/api/docs`
- Health check: `http://localhost:8000/health`

### Testing
```bash
pytest tests/ -v --cov=src --cov-report=html  # Full test suite with coverage
pytest tests/unit/ -v                        # Unit tests only
pytest tests/api/ -v                         # API integration tests
pytest tests/tse_data/ -v                    # Real TSE data tests
```

### Database Operations
```bash
alembic revision --autogenerate -m "Migration message"  # Create migration
alembic upgrade head                                   # Apply migrations
```

### Code Quality
```bash
black src/ tests/                    # Format code
ruff check src/ tests/               # Lint code
mypy src/                            # Type check
isort src/ tests/                    # Sort imports
```

## Project-Specific Conventions

### Code Style
- **Type Hints**: Mandatory for all functions, classes, and variables
- **PEP8**: Enforced via ruff/black
- **Docstrings**: Use Google-style docstrings for complex functions
- **Naming**: snake_case for functions/variables, PascalCase for classes

### Architecture Patterns
- **Service Layer Pattern**: All business logic in dedicated service classes
- **Repository Pattern**: Data access abstracted in managers (DatabaseManager, HistoricalScoreManager)
- **Factory Pattern**: For creating complex objects (indicators, patterns)
- **Strategy Pattern**: Multiple analysis strategies (fast batch vs. standard)

### Error Handling
- Use custom exceptions from `core.domain.exceptions`
- Return structured error responses via FastAPI's HTTPException
- Log errors with structlog, not print statements

### Testing Patterns
- **Fixture-based**: Use pytest fixtures for test data (sample_candles, tse_candles)
- **Mock external deps**: Redis, external APIs in unit tests
- **Data-driven tests**: Parametrize tests with real market data
- **Coverage gate**: 70% minimum coverage required

## Integration Points & External Dependencies

### Optional Services
- **Redis**: Caching layer (configure via `REDIS_HOST`, `REDIS_PORT`)
- **External Data Service**: OHLCV data provider (`DATA_SERVICE_URL`)
- **Message Brokers**: Kafka/RabbitMQ for event-driven ingestion
- **Prometheus**: Metrics collection (`/metrics` endpoint)

### Database Integration
- **PostgreSQL primary**: For production deployments
- **SQLite fallback**: For development/testing
- **Alembic migrations**: Version-controlled schema changes
- **Connection pooling**: Via SQLAlchemy engine configuration

### ML Model Dependencies
- Models stored in `ml_models/` directory
- Pattern classifier: `pattern_classifier_*.pkl`
- Dimension weights: `dimension_weights_*.json`
- Auto-reload on model file changes

## Key Files & Directories

### Essential Reading
- `docs/architecture/SYSTEM_ARCHITECTURE_DIAGRAMS.md`: System overview with Mermaid diagrams
- `docs/guides/QUICK_START.md`: Getting started guide
- `docs/PROCESS_OVERVIEW.md`: End-to-end process flow
- `pyproject.toml`: Dependencies and tool configuration

### Code Examples
- `src/gravity_tech/api/v1/analysis.py`: Main analysis endpoint implementation
- `src/gravity_tech/services/analysis_service.py`: Core analysis orchestration
- `src/gravity_tech/core/indicators/trend.py`: Indicator implementation example
- `src/gravity_tech/patterns/harmonic.py`: Pattern detection logic

### Configuration
- `src/gravity_tech/config/settings.py`: Pydantic settings with environment variables
- `alembic.ini`: Database migration configuration
- `.env.example`: Environment variable template

## Common Patterns & Examples

### Indicator Calculation
```python
# From src/gravity_tech/core/indicators/trend.py
def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return np.full(len(prices), np.nan)
    return np.convolve(prices, np.ones(period), 'valid') / period
```

### API Endpoint Structure
```python
# From src/gravity_tech/api/v1/analysis.py
@router.post("/analyze", response_model=TechnicalAnalysisResult)
async def analyze_technical_data(request: AnalysisRequest) -> TechnicalAnalysisResult:
    """Perform complete technical analysis."""
    return await analysis_service.analyze(request)
```

### Service Layer Pattern
```python
# From src/gravity_tech/services/analysis_service.py
class TechnicalAnalysisService:
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache_service = cache_service
        self.indicator_calculator = IndicatorCalculator()
        self.pattern_detector = PatternDetector()
```

### Harmonic Pattern Detection
```python
# From src/gravity_tech/patterns/harmonic.py
def detect_gartley(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Optional[PatternResult]:
    """Detect Gartley harmonic pattern using Fibonacci ratios."""
    # Implementation checks for XA, AB, BC, CD leg ratios
    # Returns pattern details if found
```

### ML Model Usage
```python
# From src/gravity_tech/ml/pattern_classifier.py
class PatternClassifier:
    def __init__(self, model_path: str):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)
```

## Development Best Practices

### When Adding New Indicators
1. Implement in appropriate `core/indicators/` module
2. Add confidence scoring
3. Update `IndicatorCalculator` integration
4. Add unit tests with real data
5. Update API response models if needed

### When Adding New Patterns
1. Implement detection logic in `patterns/` module
2. Add ML scoring if applicable
3. Update `PatternDetector` service
4. Add comprehensive tests
5. Update documentation

### When Modifying API
1. Update Pydantic models in `core/contracts/`
2. Modify service layer logic
3. Update API endpoint and validation
4. Add integration tests
5. Update OpenAPI documentation

### Performance Considerations
- Use Numba for numerical computations
- Implement caching for expensive operations
- Batch process multiple symbols when possible
- Monitor memory usage for large datasets

## Troubleshooting

### Common Issues
- **Model missing errors**: Ensure ML model files exist in `ml_models/`
- **Cache connection failures**: Check Redis configuration or disable caching
- **Database errors**: Verify connection strings and run migrations
- **Import errors**: Ensure `PYTHONPATH=src` is set

### Debugging Tips
- Use `structlog` for structured logging
- Check `/health` endpoint for service status
- Review test failures for integration issues
- Use `--reload` flag during development for hot reloading</content>
<parameter name="filePath">e:\Shakour\GravityProjects\Gravity_TechnicalAnalysis\.github\copilot-instructions.md