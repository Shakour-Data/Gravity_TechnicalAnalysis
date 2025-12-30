# Phase 4: Testing & Quality Assurance Guide

## 📊 Overview

This guide explains Phase 4 of the Architecture Improvement Roadmap: **Testing & Quality Assurance**.

**Phase 4 Objectives:**
- Increase test coverage from 57.85% → 80%+
- Add comprehensive security testing (OWASP Top 10)
- Implement coverage gates in CI/CD
- Improve code quality and reduce bugs

---

## 🎯 Coverage Targets

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| **Overall Coverage** | 57.85% | 80%+ | 22.15%+ |
| **Unit Tests** | 50% | 65% | 15% |
| **Integration Tests** | 5% | 12% | 7% |
| **E2E Tests** | 2% | 3% | 1% |
| **Security Tests** | 0% | Comprehensive | Full |

---

## 🏗️ Test Structure

```
apps/analysis_api/tests/
├── unit/                           # Isolated component tests
│   ├── test_indicators.py         # Technical indicators (200+ tests)
│   ├── test_patterns.py           # Pattern detection (150+ tests)
│   ├── test_analysis.py           # Analysis services (100+ tests)
│   └── test_models.py             # Domain models (50+ tests)
│
├── integration/                    # Component interactions
│   ├── test_api_endpoints.py      # REST API (80+ tests)
│   ├── test_services.py           # Service layer (60+ tests)
│   └── test_database.py           # Database ops (40+ tests)
│
├── e2e/                            # Complete workflows
│   ├── test_analysis_flow.py      # Full analysis (30+ tests)
│   └── test_pipeline_flow.py      # Data pipeline (20+ tests)
│
├── security/                       # OWASP Top 10
│   ├── test_owasp_injection.py    # SQL/command injection
│   ├── test_owasp_auth.py         # Auth/authz
│   ├── test_owasp_validation.py   # Input validation
│   └── test_owasp_exposure.py     # Data exposure
│
├── test_phase4_comprehensive.py   # Comprehensive test templates
├── test_phase4_security.py        # Security test suite
└── conftest.py                     # Shared fixtures (Phase 4 enhanced)
```

---

## 🚀 Running Tests

### Run All Tests with Coverage
```bash
pytest --cov=apps/analysis_api/src --cov-report=html
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# Security tests
pytest -m security

# Exclude slow tests
pytest -m 'not slow'
```

### Run with Coverage Gate (70% minimum)
```bash
pytest --cov=apps/analysis_api/src --cov-fail-under=70
```

### Generate HTML Coverage Report
```bash
pytest --cov=apps/analysis_api/src --cov-report=html
open htmlcov/index.html
```

### Use Coverage Analysis Script
```bash
# Analyze coverage gaps
python scripts/analyze_coverage.py --analyze

# Run coverage and analyze
python scripts/analyze_coverage.py --run

# Generate test templates for gap modules
python scripts/analyze_coverage.py --templates
```

---

## 📝 Writing Tests

### Test File Template
```python
import pytest
from unittest.mock import Mock, patch, AsyncMock

@pytest.mark.unit
class TestMyComponent:
    """Tests for MyComponent"""
    
    @pytest.fixture
    def subject(self):
        """Initialize component under test"""
        return MyComponent(
            dependency1=Mock(),
            dependency2=Mock()
        )
    
    def test_basic_behavior(self, subject):
        """Test happy path"""
        result = subject.do_something()
        assert result == expected
    
    def test_error_handling(self, subject):
        """Test error cases"""
        with pytest.raises(ValueError):
            subject.do_something_invalid()
    
    @pytest.mark.asyncio
    async def test_async_behavior(self, subject):
        """Test async operations"""
        result = await subject.async_operation()
        assert result == expected
```

### Using Test Fixtures

**Built-in fixtures:**
```python
def test_with_candles(sample_candles):
    """Use pre-made sample data"""
    assert len(sample_candles) == 100

def test_with_uptrend(sample_uptrend_candles):
    """Use uptrend data"""
    assert all(c.close > c.open for c in sample_uptrend_candles)

def test_with_container(test_container):
    """Use DI container"""
    service = test_container.get("analysis_service")
    result = await service.analyze(candles)

def test_with_request(request_builder):
    """Build custom requests"""
    req = request_builder.analysis_request(symbol="ETHUSDT")
    assert req["symbol"] == "ETHUSDT"
```

### Mocking Examples

```python
# Mock async method
@pytest.mark.asyncio
async def test_with_mocked_cache(mock_cache):
    mock_cache.get = AsyncMock(return_value={"cached": "data"})
    
    result = await service.get_cached_data("key")
    
    mock_cache.get.assert_called_once_with("key")
    assert result == {"cached": "data"}

# Mock database
def test_with_mocked_db(mock_database):
    mock_database.fetch_one.return_value = {"id": 1, "name": "test"}
    
    result = service.get_record(1)
    
    mock_database.fetch_one.assert_called_once()
    assert result["name"] == "test"
```

---

## 🔐 Security Testing

### OWASP Top 10 Coverage

**1. Injection (SQL, Command, LDAP)**
```python
@pytest.mark.security
def test_sql_injection_prevention(client):
    malicious = "'; DROP TABLE symbols; --"
    response = client.post(
        "/api/v1/analyze",
        json={"symbol": malicious}
    )
    assert response.status_code == 400
```

**2. Broken Authentication**
```python
@pytest.mark.security
def test_missing_auth_rejected(client):
    response = client.post("/api/v1/protected/endpoint")
    assert response.status_code == 401
```

**3. Sensitive Data Exposure**
```python
@pytest.mark.security
def test_passwords_not_in_response(client):
    response = client.get("/api/v1/user/profile")
    assert "password" not in response.json()
```

**4-10. See [test_phase4_security.py](test_phase4_security.py) for complete suite**

---

## 📊 Coverage Reports

### HTML Report
```bash
pytest --cov=apps/analysis_api/src --cov-report=html
# Open htmlcov/index.html in browser
```

### Terminal Report
```bash
pytest --cov=apps/analysis_api/src --cov-report=term-missing
```

### JSON Report (for CI/CD)
```bash
pytest --cov=apps/analysis_api/src --cov-report=json
# Useful for tracking coverage trends
```

---

## 🔄 CI/CD Integration

### GitHub Actions Coverage Gate
```yaml
- name: Run tests with coverage gate
  run: pytest --cov=apps/analysis_api/src --cov-fail-under=70
```

### Coverage Badge
```markdown
![Coverage](https://img.shields.io/badge/coverage-75%25-yellowgreen)
```

---

## 📈 Quality Metrics

### Track These Metrics
- **Line Coverage:** % of lines executed
- **Branch Coverage:** % of conditional branches tested
- **Complexity:** Cyclomatic complexity per function (target: < 10)
- **Test Pass Rate:** Should be 100%
- **Flaky Tests:** Should be 0
- **Test Speed:** Full suite < 5 minutes

### Improve Coverage
1. Run analysis: `python scripts/analyze_coverage.py --analyze`
2. Identify gaps (< 70% coverage)
3. Write tests for gap areas
4. Verify: `pytest --cov=apps/analysis_api/src --cov-fail-under=70`
5. Commit: `git add -A && git commit -m "test(phase4): Add missing tests for <module>"`

---

## 🚦 Quality Gates

### Minimum Requirements
- ✅ Overall coverage: 70%+
- ✅ Critical paths: 100%
- ✅ All tests pass
- ✅ No flaky tests
- ✅ Security tests pass
- ✅ No new vulnerabilities

### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

pytest --cov=apps/analysis_api/src --cov-fail-under=70 || exit 1
```

---

## 🛠️ Troubleshooting

### Coverage Not Accurate?
```bash
# Clear coverage cache
rm -rf .coverage .coverage.* htmlcov/

# Rerun tests
pytest --cov=apps/analysis_api/src --cov-report=html
```

### Flaky Tests?
```bash
# Run same test multiple times
pytest tests/unit/test_something.py -v --count=10
```

### Slow Tests?
```bash
# Profile test execution time
pytest --durations=10

# Skip slow tests
pytest -m 'not slow'
```

---

## 📚 Additional Resources

- **Test Templates:** `test_phase4_comprehensive.py`
- **Security Tests:** `test_phase4_security.py`
- **Docs Index:** `docs/INDEX.md`
- **Conftest Fixtures:** `conftest.py` (Phase 4 enhanced)

---

## ✅ Phase 4 Checklist

- [ ] Create `tests/unit/test_indicators.py` (200+ tests)
- [ ] Create `tests/unit/test_patterns.py` (150+ tests)
- [ ] Create `tests/unit/test_analysis.py` (100+ tests)
- [ ] Create `tests/integration/test_api_endpoints.py` (80+ tests)
- [ ] Create `tests/integration/test_services.py` (60+ tests)
- [ ] Create `tests/integration/test_database.py` (40+ tests)
- [ ] Create `tests/e2e/test_analysis_flow.py` (30+ tests)
- [ ] Create `tests/e2e/test_pipeline_flow.py` (20+ tests)
- [ ] Create `tests/security/test_owasp_*.py` (200+ tests)
- [ ] Achieve 65%+ unit test coverage
- [ ] Achieve 12%+ integration coverage
- [ ] Achieve 3%+ E2E coverage
- [ ] Achieve 80%+ overall coverage
- [ ] All security tests pass
- [ ] Coverage gates in CI (70% minimum)
- [ ] Documentation complete

---

**Status:** 🔄 IN PROGRESS  
**Target Date:** End of Week 11  
**Questions?** See `docs/INDEX.md`
