#!/usr/bin/env bash
# Test Execution Quick Reference Guide

## 📋 Test Categories and Quick Commands

### 1. Run All Tests (Active Only)
```bash
pytest tests/ --ignore=tests/archived -v --cov=src --cov-report=term-missing
```

### 2. Run by Category

#### Unit Tests (Individual Components)
```bash
pytest tests/unit/ -v --cov=src
```

#### TSE Data Tests (Real Market Data)
```bash
pytest tests/tse_data/ -v --cov=src
# Test Count: 152+ tests
# Data Source: E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db
```

#### API Tests (REST Endpoints)
```bash
pytest tests/api/ -v --cov=src
# Test Count: 40+ tests
# Coverage: API v1 endpoints, schemas, contracts
```

#### Service Tests (Service Layer)
```bash
pytest tests/services/ -v --cov=src
# Coverage: Cache service, service discovery, integration
```

#### ML Tests (Machine Learning)
```bash
pytest tests/ml/ -v --cov=src
# Coverage: Deep learning models, model training, ML pipeline
```

#### E2E Tests (End-to-End)
```bash
pytest tests/e2e/ -v --cov=src
# Coverage: Complete workflows, real-world scenarios
```

#### Integration Tests (Multi-Component)
```bash
pytest tests/integration/ -v --cov=src
# Coverage: System integration, multi-horizon analysis
```

#### Accuracy Tests (Correctness)
```bash
pytest tests/accuracy/ -v --cov=src
# Coverage: Mathematical correctness, confidence metrics
```

#### Benchmark Tests (Performance)
```bash
pytest tests/benchmarks/ -v --cov=src
# Coverage: Performance profiling, indicator benchmarks
```

#### Contract Tests (API Contracts)
```bash
pytest tests/contract/ -v --cov=src
# Coverage: API contract compliance
```

### 3. Run Specific Test Files

```bash
# TSE data comprehensive tests
pytest tests/tse_data/test_all_with_tse_data.py -v

# API v1 fixed tests
pytest tests/api/test_api_v1_comprehensive_fixed.py -v

# Advanced patterns with TSE
pytest tests/tse_data/test_phase4_advanced_patterns_tse.py -v

# Edge cases and stress tests
pytest tests/tse_data/test_phase5_edge_cases_stress_tse.py -v
```

### 4. Generate Coverage Reports

#### Terminal Report (Quick)
```bash
pytest tests/ --ignore=tests/archived --cov=src --cov-report=term-missing
```

#### HTML Report (Detailed)
```bash
pytest tests/ --ignore=tests/archived --cov=src --cov-report=html
# Open: htmlcov/index.html
```

#### By Category with Coverage
```bash
pytest tests/unit/ --cov=src --cov-report=term-missing -v
pytest tests/tse_data/ --cov=src --cov-report=term-missing -v
pytest tests/api/ --cov=src --cov-report=term-missing -v
```

### 5. Quick Test Counts

```bash
# Count tests in each category
pytest tests/unit/ --collect-only -q
pytest tests/tse_data/ --collect-only -q
pytest tests/api/ --collect-only -q
pytest tests/services/ --collect-only -q
pytest tests/ml/ --collect-only -q
pytest tests/e2e/ --collect-only -q
```

### 6. Run Tests with Markers

```bash
# Run only fast tests
pytest tests/ -m "not slow" -v

# Run only integration tests
pytest tests/integration/ -v

# Run everything except archived
pytest tests/ --ignore=tests/archived -v
```

### 7. Parallel Execution (Multiple CPUs)

```bash
# Install pytest-xdist first
pip install pytest-xdist

# Run in parallel
pytest tests/ -n auto --ignore=tests/archived -v
```

## 📊 Test Statistics

| Category | Files | Tests | Status |
|----------|-------|-------|--------|
| unit | 28 | ~100+ | ✅ Active |
| tse_data | 4 | 152+ | ✅ Active |
| api | 5 | 40+ | ✅ Active |
| services | 5 | 20+ | ✅ Active |
| integration | 4 | 15+ | ✅ Active |
| accuracy | 4 | 12+ | ✅ Active |
| ml | 5 | 15+ | ✅ Active |
| e2e | 5 | 20+ | ✅ Active |
| benchmarks | 4 | 8+ | ✅ Active |
| contract | 2 | 5+ | ✅ Active |
| archived | 18 | N/A | 📦 Legacy |
| **TOTAL** | **84** | **302+** | ✅ |

## 🎯 Coverage Goals

| Target | Current | Status |
|--------|---------|--------|
| Overall | 11.71% | 🔄 In Progress |
| Phase 4-5 TSE | Ready | 🚀 Ready for Execution |
| Unit Coverage | 95%+ | 📊 Measuring |
| Goal | 95% | 🎯 Target |

## 🔍 Directory Structure

```
tests/
├── unit/                  → Individual component tests
├── integration/           → Multi-component workflows
├── accuracy/             → Correctness validation
├── contract/             → API contracts
├── tse_data/             → Real market data tests ⭐
├── api/                  → API endpoints ⭐
├── services/             → Service layer ⭐
├── ml/                   → ML models ⭐
├── e2e/                  → End-to-end flows ⭐
├── benchmarks/           → Performance tests ⭐
├── performance/          → Ready for expansion
├── archived/             → Legacy tests
├── conftest.py           → Fixtures & configuration
└── TEST_STRUCTURE.md     → Full documentation
```

## 🚀 Getting Started

1. **First Run**: Execute all active tests
   ```bash
   pytest tests/ --ignore=tests/archived -v --cov=src --cov-report=term-missing
   ```

2. **Check Coverage**: Generate HTML report
   ```bash
   pytest tests/ --ignore=tests/archived --cov=src --cov-report=html
   open htmlcov/index.html
   ```

3. **Run Specific Category**: Focus on one area
   ```bash
   pytest tests/tse_data/ -v --cov=src
   ```

4. **Troubleshoot**: Verbose output
   ```bash
   pytest tests/ -vv -s --tb=long
   ```

## 📝 Common Tasks

### Run only failing tests
```bash
pytest tests/ --lf -v
```

### Run last failed and exit on first failure
```bash
pytest tests/ --lf -x -v
```

### Show test names without running
```bash
pytest tests/ --collect-only -q
```

### Run with print statements visible
```bash
pytest tests/ -s -v
```

### Run specific test class
```bash
pytest tests/unit/test_momentum.py::TestMomentumIndicators -v
```

### Run specific test method
```bash
pytest tests/unit/test_momentum.py::TestMomentumIndicators::test_rsi -v
```

## 🔗 Related Documents

- `TEST_STRUCTURE.md` - Complete test structure documentation
- `REORGANIZATION_VERIFICATION.md` - Reorganization verification report
- `conftest.py` - Pytest configuration and fixtures
- `.github/workflows/` - CI/CD pipeline configuration

## ✅ Verification Checklist

Before running tests:
- [ ] Python 3.13.7 or higher
- [ ] pytest 7.4.3+ installed
- [ ] pytest-cov 4.1.0+ installed
- [ ] TSE database accessible: `E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db`
- [ ] conftest.py fixtures configured
- [ ] No uncommitted changes in critical files

## 🐛 Troubleshooting

### Tests not found
```bash
# Verify test discovery
pytest tests/ --collect-only
```

### Import errors
```bash
# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
pytest tests/ -v
```

### TSE data not loading
```bash
# Check fixture availability
pytest tests/tse_data/ -v --tb=short
```

### Coverage report missing
```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html
```

---

Last Updated: December 4, 2025  
Status: ✅ Ready for Coverage Execution
