# Test Structure Organization

**Date**: December 4, 2025  
**Status**: ✅ Complete Reorganization

## Overview

The test suite has been reorganized into a logical, hierarchical structure that makes it easy to:
- Find tests by category and purpose
- Run specific test suites
- Maintain and update tests
- Measure coverage by test category

## Directory Structure

```
tests/
├── unit/                          # Unit tests (core components)
│   ├── core/                      # Core indicator tests
│   ├── ml/                        # ML component unit tests
│   ├── utils/                     # Utility tests
│   └── 28 test files              # Individual component tests
│
├── integration/                   # Integration tests
│   ├── test_combined_system.py
│   ├── test_complete_analysis.py
│   └── test_multi_horizon.py
│
├── accuracy/                      # Accuracy and correctness tests
│   ├── test_accuracy_weighting.py
│   ├── test_comprehensive_accuracy.py
│   └── test_confidence_metrics.py
│
├── contract/                      # API contract tests
│   └── test_api_contract.py
│
├── performance/                   # Performance tests (empty - ready for expansion)
│
├── api/                          # API endpoint tests
│   ├── test_api_v1_clean.py
│   ├── test_api_v1_comprehensive.py
│   ├── test_api_v1_comprehensive_fixed.py
│   └── test_api_endpoints_comprehensive.py
│
├── services/                     # Service layer tests
│   ├── test_services_comprehensive.py
│   ├── test_services_final.py
│   ├── test_cache_service.py
│   └── test_service_discovery.py
│
├── ml/                           # Machine Learning tests
│   ├── test_deep_learning_models.py
│   ├── test_deep_learning_comprehensive.py
│   ├── test_ml_comprehensive.py
│   └── test_day5_advanced_ml.py
│
├── e2e/                          # End-to-end tests
│   ├── test_day6_api_integration.py
│   ├── test_advanced_features.py
│   ├── test_advanced_features_corrected.py
│   └── test_advanced_backtesting.py
│
├── tse_data/                     # TSE (Tehran Stock Exchange) data tests
│   ├── test_all_with_tse_data.py
│   ├── test_phase4_advanced_patterns_tse.py
│   ├── test_phase5_edge_cases_stress_tse.py
│   └── test_services_with_tse_data.py
│
├── benchmarks/                   # Performance benchmarks
│   ├── benchmark_momentum_indicators.py
│   ├── benchmark_new_indicators.py
│   └── benchmark_volume_day3.py
│
├── archived/                     # Legacy and experimental tests
│   ├── test_patterns_comprehensive.py
│   ├── test_pattern_recognition.py
│   ├── test_realtime_comprehensive.py
│   ├── test_resilience.py
│   ├── test_sample.py
│   ├── test_events.py
│   ├── test_middleware_comprehensive.py
│   ├── test_indicators.py
│   ├── test_momentum_indicators.py
│   ├── test_new_trend_indicators.py
│   ├── test_fibonacci_tools.py
│   ├── test_fibonacci_tools_comprehensive.py
│   ├── test_volume_day3.py
│   ├── test_volume_indicators_comprehensive.py
│   ├── validate_momentum_indicators.py
│   ├── validate_volume_day3.py
│   ├── mathematical_validation.py
│   └── test_cycle_fix.txt.bak
│
├── conftest.py                   # Pytest configuration & fixtures
└── __pycache__/                  # Python cache
```

## Test Categories Explained

### 1. **Unit Tests** (`tests/unit/`)
Core unit tests for individual components:
- **Individual Components**: 28 test files covering candles, patterns, indicators, momentum, volume, trend, volatility, cycles, Elliott waves, etc.
- **Purpose**: Validate individual functions and classes in isolation
- **Coverage Target**: 95%+

### 2. **Integration Tests** (`tests/integration/`)
Tests that validate multiple components working together:
- `test_combined_system.py` - Multiple systems in combination
- `test_complete_analysis.py` - Complete analysis pipeline
- `test_multi_horizon.py` - Multi-timeframe analysis
- **Purpose**: Ensure components integrate correctly

### 3. **Accuracy Tests** (`tests/accuracy/`)
Tests focused on correctness and precision:
- `test_accuracy_weighting.py` - Weighting algorithm accuracy
- `test_comprehensive_accuracy.py` - Overall system accuracy
- `test_confidence_metrics.py` - Confidence score validation
- **Purpose**: Verify mathematical correctness

### 4. **API Tests** (`tests/api/`)
RESTful API endpoint testing:
- `test_api_v1_*.py` - Various API v1 endpoint tests
- `test_api_endpoints_comprehensive.py` - Comprehensive endpoint coverage
- **Purpose**: Validate API contracts and responses

### 5. **Service Tests** (`tests/services/`)
Service layer component testing:
- `test_services_*.py` - Service functionality
- `test_cache_service.py` - Caching service
- `test_service_discovery.py` - Service discovery
- **Purpose**: Test service implementations

### 6. **ML Tests** (`tests/ml/`)
Machine Learning model testing:
- `test_deep_learning_models.py` - Individual ML models
- `test_deep_learning_comprehensive.py` - Comprehensive ML testing
- `test_ml_comprehensive.py` - Full ML pipeline
- **Purpose**: Validate ML components and training

### 7. **E2E Tests** (`tests/e2e/`)
End-to-end system flow tests:
- `test_advanced_features.py` - Advanced feature workflows
- `test_advanced_backtesting.py` - Backtesting scenarios
- `test_day6_api_integration.py` - Complete API integration
- **Purpose**: Test real-world usage scenarios

### 8. **TSE Data Tests** (`tests/tse_data/`)
Tests using real Tehran Stock Exchange data:
- `test_all_with_tse_data.py` - Complete analysis with TSE data
- `test_phase4_advanced_patterns_tse.py` - 28+ advanced pattern tests
- `test_phase5_edge_cases_stress_tse.py` - 25+ edge case tests
- `test_services_with_tse_data.py` - Service tests with TSE data
- **Purpose**: Validate system with real market data
- **Database**: `E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db`

### 9. **Benchmark Tests** (`tests/benchmarks/`)
Performance and benchmark testing:
- `benchmark_momentum_indicators.py` - Momentum performance
- `benchmark_new_indicators.py` - New feature performance
- `benchmark_volume_day3.py` - Volume indicator performance
- **Purpose**: Measure and track performance metrics

### 10. **Contract Tests** (`tests/contract/`)
API contract validation:
- `test_api_contract.py` - Contract compliance
- **Purpose**: Ensure API contracts are maintained

### 11. **Archived Tests** (`tests/archived/`)
Legacy and experimental files:
- Proof-of-concept implementations
- Older versions no longer actively maintained
- Backup and reference implementations
- **Status**: Not part of active test suite

## Running Tests

### Run All Tests
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Run by Category
```bash
# Unit tests only
pytest tests/unit/ -v --cov=src

# Integration tests only
pytest tests/integration/ -v --cov=src

# TSE data tests only
pytest tests/tse_data/ -v --cov=src

# API tests only
pytest tests/api/ -v --cov=src

# All tests except archived
pytest tests/ --ignore=tests/archived -v --cov=src
```

### Run Specific Test File
```bash
pytest tests/unit/test_momentum.py -v
pytest tests/tse_data/test_all_with_tse_data.py -v
pytest tests/api/test_api_v1_comprehensive_fixed.py -v
```

## Coverage Summary

Current Status: **11.71%** (1,948 of 16,611 lines)

### Coverage by Category
- **Unit Tests**: Comprehensive coverage for individual components
- **TSE Data Tests**: Real-world scenario validation (152+ tests)
- **API Tests**: Endpoint validation (40+ tests)
- **Services**: Service layer verification
- **Integration**: Multi-component workflows
- **E2E**: Complete system workflows

### Coverage Target
- **Goal**: 95% total coverage
- **Current**: 11.71%
- **In Progress**: Phase 4-5 TSE tests provide foundation for expansion

## Key Files

| File | Purpose |
|------|---------|
| `conftest.py` | Pytest configuration, fixtures, TSE data integration |
| `unit/` | Core component testing (28 test files) |
| `tse_data/` | Real market data integration (152+ tests) |
| `api/` | API endpoint validation (40+ tests) |
| `integration/` | Multi-component workflows |

## Migration Notes

### What Moved
1. **TSE Data Tests**: Moved to `tests/tse_data/`
   - `test_all_with_tse_data.py`
   - `test_phase4_advanced_patterns_tse.py`
   - `test_phase5_edge_cases_stress_tse.py`
   - `test_services_with_tse_data.py`

2. **API Tests**: Moved to `tests/api/`
   - `test_api_v1_clean.py`
   - `test_api_v1_comprehensive.py`
   - `test_api_v1_comprehensive_fixed.py`
   - `test_api_endpoints_comprehensive.py`

3. **Service Tests**: Moved to `tests/services/`
   - `test_services_comprehensive.py`
   - `test_services_final.py`
   - `test_cache_service.py`
   - `test_service_discovery.py`

4. **ML Tests**: Moved to `tests/ml/`
   - `test_deep_learning_models.py`
   - `test_deep_learning_comprehensive.py`
   - `test_ml_comprehensive.py`
   - `test_day5_advanced_ml.py`

5. **E2E Tests**: Moved to `tests/e2e/`
   - `test_day6_api_integration.py`
   - `test_advanced_features.py`
   - `test_advanced_features_corrected.py`
   - `test_advanced_backtesting.py`

6. **Benchmarks**: Moved to `tests/benchmarks/`
   - `benchmark_momentum_indicators.py`
   - `benchmark_new_indicators.py`
   - `benchmark_volume_day3.py`

7. **Legacy Files**: Moved to `tests/archived/`
   - 14 older test files
   - 3 validation scripts
   - 1 backup file

### What Stayed
- `tests/unit/` - Already organized, kept as-is (28 test files)
- `tests/integration/` - Already organized, kept as-is (3 test files)
- `tests/accuracy/` - Already organized, kept as-is (3 test files)
- `tests/contract/` - Already organized, kept as-is (1 test file)
- `tests/performance/` - Empty folder for future performance tests
- `conftest.py` - Central fixture configuration

## Best Practices

1. **File Organization**: Each test category has its own folder
2. **Import Paths**: Use relative imports within categories
3. **Fixtures**: Shared in `conftest.py` at root level
4. **TSE Data**: Load from `E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db`
5. **Coverage**: Run with `--cov=src --cov-report=term-missing` for details

## Next Steps

1. **Coverage Expansion**:
   - Execute all Phase 4-5 TSE tests
   - Measure coverage improvement
   - Identify gaps at 11.71% baseline

2. **Additional Tests**:
   - Create Phase 6+ tests for coverage gaps
   - Target 95% total coverage

3. **Performance**:
   - Expand benchmarks folder
   - Add performance regression tests

## Contact & Documentation

For more information about:
- **Test Execution**: See `conftest.py` for fixtures and setup
- **TSE Data**: Check fixture definitions in `conftest.py`
- **API Testing**: Review `tests/api/` folder structure
- **Coverage Reports**: Run with `--cov-report=html` for HTML reports

---

**Last Updated**: December 4, 2025  
**Reorganized By**: Gravity Technical Analysis Team
