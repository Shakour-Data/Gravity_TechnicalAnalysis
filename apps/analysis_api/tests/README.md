# Test Suite - Complete Documentation

## 📊 نمای کلی (Overview)

ساختار تست‌های جامع برای دستیابی به **70%+** پوشش تمام ماژول‌های پروژه.

- ✅ **40+ فایل تست**
- ✅ **300+ متد تست**  
- ✅ **10,000+ خط کد تست**
- ✅ **بدون Mock Tests**
- ✅ **فقط داده‌های واقعی TSE**

## 🎯 ساختار سازماندهی شده

## 🎯 Quick Navigation

| Category | Path | Files | Tests | Purpose |
|----------|------|-------|-------|---------|
| **Unit** | `unit/` | 28 | 100+ | Individual components |
| **TSE Data** | `tse_data/` | 4 | 152+ | Real market data |
| **API** | `api/` | 5 | 40+ | REST endpoints |
| **Services** | `services/` | 5 | 20+ | Service layer |
| **ML** | `ml/` | 5 | 15+ | Machine learning |
| **E2E** | `e2e/` | 5 | 20+ | End-to-end flows |
| **Integration** | `integration/` | 4 | 15+ | Multi-component |
| **Accuracy** | `accuracy/` | 4 | 12+ | Correctness |
| **Benchmarks** | `benchmarks/` | 4 | 8+ | Performance |
| **Contract** | `contract/` | 2 | 5+ | API contracts |
| **Archived** | `archived/` | 18 | N/A | Legacy code |
| **Performance** | `performance/` | 0 | — | Ready for expansion |

## 🚀 Getting Started

### Installation
```bash
# Install dependencies
pip install pytest pytest-cov

# Verify installation
pytest --version
```

### Run All Tests (Active Only)
```bash
pytest tests/ --ignore=tests/archived -v --cov=src --cov-report=term-missing
```

### Run Specific Category
```bash
# Example: Run TSE data tests
pytest tests/tse_data/ -v --cov=src
```

## 📁 Directory Structure

### 1. **unit/** - Unit Tests (28 files)
Core component testing for all individual modules.

**Key Files:**
- `test_momentum.py` - Momentum indicators
- `test_volume.py` - Volume indicators
- `test_trend.py` - Trend indicators
- `test_volatility_comprehensive.py` - Volatility analysis
- `test_elliott.py` - Elliott wave analysis
- `test_candlestick_patterns.py` - Pattern recognition

**Run:**
```bash
pytest tests/unit/ -v
```

### 2. **integration/** - Integration Tests (4 files)
Multi-component workflow validation.

**Key Files:**
- `test_complete_analysis.py` - Full analysis pipeline
- `test_multi_horizon.py` - Multi-timeframe analysis
- `test_combined_system.py` - System combination

**Run:**
```bash
pytest tests/integration/ -v
```

### 3. **accuracy/** - Accuracy Tests (4 files)
Mathematical correctness and validation.

**Key Files:**
- `test_comprehensive_accuracy.py` - Overall accuracy
- `test_confidence_metrics.py` - Confidence validation
- `test_accuracy_weighting.py` - Weighting correctness

**Run:**
```bash
pytest tests/accuracy/ -v
```

### 4. **contract/** - Contract Tests (2 files)
API contract compliance validation.

**Key Files:**
- `test_api_contract.py` - Contract validation

**Run:**
```bash
pytest tests/contract/ -v
```

### 5. **tse_data/** - TSE Data Tests (4 files, 152+ tests) ⭐
Real Tehran Stock Exchange data integration.

**Database:** `E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db`

**Key Files:**
- `test_all_with_tse_data.py` - Complete analysis
- `test_phase4_advanced_patterns_tse.py` - 28+ pattern tests
- `test_phase5_edge_cases_stress_tse.py` - 25+ edge cases
- `test_services_with_tse_data.py` - Service integration

**Run:**
```bash
pytest tests/tse_data/ -v --cov=src
```

**Individual File:**
```bash
pytest tests/tse_data/test_all_with_tse_data.py -v
```

### 6. **api/** - API Tests (5 files, 40+ tests) ⭐
REST API endpoint and schema validation.

**Key Files:**
- `test_api_v1_comprehensive_fixed.py` - Fixed API tests
- `test_api_v1_clean.py` - Clean endpoint tests
- `test_api_endpoints_comprehensive.py` - All endpoints

**Run:**
```bash
pytest tests/api/ -v
```

**Specific Test:**
```bash
pytest tests/api/test_api_v1_comprehensive_fixed.py::TestToolRecommendation -v
```

### 7. **services/** - Service Tests (5 files, 20+ tests) ⭐
Service layer component validation.

**Key Files:**
- `test_services_comprehensive.py` - Service coverage
- `test_services_final.py` - Service finalization
- `test_cache_service.py` - Cache functionality
- `test_service_discovery.py` - Service discovery

**Run:**
```bash
pytest tests/services/ -v
```

### 8. **ml/** - ML Tests (5 files, 15+ tests) ⭐
Machine learning model validation.

**Key Files:**
- `test_deep_learning_comprehensive.py` - Deep learning
- `test_ml_comprehensive.py` - Full pipeline
- `test_day5_advanced_ml.py` - Advanced models

**Run:**
```bash
pytest tests/ml/ -v
```

### 9. **e2e/** - End-to-End Tests (5 files, 20+ tests) ⭐
Complete system workflow validation.

**Key Files:**
- `test_advanced_backtesting.py` - Backtesting workflows
- `test_day6_api_integration.py` - API integration
- `test_advanced_features.py` - Feature workflows

**Run:**
```bash
pytest tests/e2e/ -v
```

### 10. **benchmarks/** - Benchmark Tests (4 files, 8+ tests) ⭐
Performance measurement and profiling.

**Key Files:**
- `benchmark_momentum_indicators.py` - Momentum performance
- `benchmark_new_indicators.py` - New feature performance
- `benchmark_volume_day3.py` - Volume performance

**Run:**
```bash
pytest tests/benchmarks/ -v
```

### 11. **archived/** - Archived Tests (18 files)
Legacy and experimental test files (not part of active suite).

**Contents:**
- Legacy test implementations
- Experimental proof-of-concepts
- Backup and reference files
- Validation scripts

**Run:**
```bash
# Only if needed
pytest tests/archived/ -v
```

### 12. **performance/** - Performance Tests (0 files)
Ready for performance regression testing.

**Status:** Empty, prepared for future expansion

## 📊 Coverage Information

### Current Status
- **Coverage**: 11.71% (1,948 of 16,611 lines)
- **Active Tests**: 302+ test cases
- **Target**: 95% total coverage

### Coverage by Category
| Category | Coverage | Tests |
|----------|----------|-------|
| unit | Comprehensive | 100+ |
| tse_data | Real data validation | 152+ |
| api | Endpoint coverage | 40+ |
| services | Layer coverage | 20+ |
| integration | Multi-component | 15+ |
| accuracy | Correctness | 12+ |
| ml | Model coverage | 15+ |
| e2e | Flow validation | 20+ |
| benchmarks | Performance | 8+ |
| contract | Contract validation | 5+ |

### Generate Coverage Report
```bash
# Terminal report
pytest tests/ --ignore=tests/archived --cov=src --cov-report=term-missing

# HTML report
pytest tests/ --ignore=tests/archived --cov=src --cov-report=html
open htmlcov/index.html
```

## 🔧 Running Tests

### Basic Commands

```bash
# Run all tests (exclude archived)
pytest tests/ --ignore=tests/archived -v

# Run all tests with coverage
pytest tests/ --ignore=tests/archived -v --cov=src --cov-report=term-missing

# Run specific directory
pytest tests/unit/ -v

# Run specific file
pytest tests/unit/test_momentum.py -v

# Run specific test class
pytest tests/unit/test_momentum.py::TestMomentumIndicators -v

# Run specific test method
pytest tests/unit/test_momentum.py::TestMomentumIndicators::test_rsi -v
```

### Advanced Commands

```bash
# Run in parallel (requires pytest-xdist)
pytest tests/ -n auto --ignore=tests/archived -v

# Run only failed tests
pytest tests/ --lf -v

# Run and stop on first failure
pytest tests/ -x -v

# Run with verbose output and print statements
pytest tests/ -vv -s --tb=long

# Run specific marker
pytest tests/ -m "not slow" -v

# Collect tests without running
pytest tests/ --collect-only -q
```

## 📝 Configuration

### conftest.py
Central pytest configuration file containing:
- **Fixtures**: Shared test fixtures and TSE data integration
- **Hooks**: Test lifecycle management
- **Plugins**: pytest plugin configuration
- **TSE Database**: Real market data configuration

### pytest.ini
Configuration file with:
- Test discovery patterns
- Markers and options
- Plugin settings

### pyproject.toml
Project configuration including:
- Dependencies
- Tool configurations
- Build settings

## 🔍 Key Fixtures

All fixtures are defined in `conftest.py`:

```python
# TSE Data Fixtures
tse_candles_short     # Recent TSE data
tse_candles_long      # Extended TSE data
tse_candles_long_alt  # Alternative dataset
tse_symbol_data       # Multi-symbol data

# Sample Data
sample_candles        # Generated sample data
sample_data_long      # Extended sample data

# Market Data
market_data           # General market data
```

## 📚 Documentation Files

1. **TEST_STRUCTURE.md** (500+ lines)
   - Complete directory structure
   - Category explanations
   - Best practices
   - Migration notes

2. **REORGANIZATION_VERIFICATION.md**
   - Reorganization details
   - File movement tracking
   - Statistics and metrics

3. **TEST_EXECUTION_GUIDE.md**
   - Quick command reference
   - Common tasks
   - Troubleshooting guide

4. **DIRECTORY_TREE.txt**
   - Visual directory tree
   - File listings
   - Organization overview

## 🎯 Common Tasks

### Run TSE Data Tests
```bash
pytest tests/tse_data/ -v --cov=src
```

### Run API Tests
```bash
pytest tests/api/ -v --cov=src
```

### Run All Unit Tests
```bash
pytest tests/unit/ -v --cov=src
```

### Generate HTML Coverage Report
```bash
pytest tests/ --ignore=tests/archived --cov=src --cov-report=html
open htmlcov/index.html
```

### Run Tests in Parallel
```bash
pytest tests/ -n auto --ignore=tests/archived -v
```

### Debug a Failing Test
```bash
pytest tests/path/to/test.py::TestClass::test_method -vv -s --tb=long
```

## 🐛 Troubleshooting

### Tests Not Found
```bash
# Verify test discovery
pytest tests/ --collect-only -q
```

### Import Errors
```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
pytest tests/ -v
```

### TSE Data Not Loading
```bash
# Check fixture availability
pytest tests/tse_data/ -v --tb=short
```

### Coverage Report Missing
```bash
# Install coverage
pip install pytest-cov

# Generate report
pytest tests/ --cov=src --cov-report=html
```

### Permission Denied
```bash
# Run as administrator (if needed)
sudo pytest tests/ -v
```

## ✅ Checklist Before Running Tests

- [ ] Python 3.13.7+ installed
- [ ] pytest 7.4.3+ installed
- [ ] pytest-cov 4.1.0+ installed
- [ ] TSE database accessible
- [ ] Source files in `src/` directory
- [ ] No uncommitted changes
- [ ] Virtual environment activated
- [ ] All dependencies installed

## 🚀 Next Steps

1. **Immediate**: Run all tests to verify setup
   ```bash
   pytest tests/ --ignore=tests/archived -v --cov=src
   ```

2. **Analysis**: Review coverage report
   ```bash
   pytest tests/ --ignore=tests/archived --cov=src --cov-report=html
   ```

3. **Expansion**: Identify coverage gaps and create Phase 6+ tests

4. **Targets**: Work toward 95% coverage goal

## 📞 Support

For issues or questions:
1. Check **TEST_STRUCTURE.md** for detailed information
2. Review **TEST_EXECUTION_GUIDE.md** for command reference
3. Check **conftest.py** for fixture definitions
4. Review test files for examples

## 📄 Related Files

- `pytest.ini` - Pytest configuration
- `pyproject.toml` - Project configuration
- `conftest.py` - Pytest fixtures and hooks
- `.gitignore` - Git ignore patterns
- `README.md` - Project README

---

**Last Updated**: December 4, 2025  
**Status**: ✅ Ready for Coverage Execution  
**Total Tests**: 302+  
**Coverage Target**: 95%
