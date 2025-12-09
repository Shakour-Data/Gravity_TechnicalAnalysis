# ⚡ Quick Reference - Test Commands

## 🎯 Most Common Commands

### Run All Tests
```bash
pytest tests/ -v
```

### Run with Coverage
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Generate HTML Coverage Report
```bash
pytest tests/ --cov=src --cov-report=html
```

---

## 🧪 Run by Category

```bash
# Security & Authentication
pytest tests/unit/middleware/ -v

# Pattern Recognition
pytest tests/unit/patterns/ -v

# Business Services
pytest tests/unit/services/ -v

# ML Models
pytest tests/unit/ml/ -v

# Utilities
pytest tests/unit/utils/ -v

# All Domain Tests
pytest tests/unit/domain/ -v

# All Analysis Tests
pytest tests/unit/analysis/ -v

# All Indicator Tests
pytest tests/unit/indicators/ -v
```

---

## 📊 Coverage Analysis

### Show Missing Lines
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Generate HTML Report
```bash
pytest tests/ --cov=src --cov-report=html
# Then open: htmlcov/index.html
```

### Coverage for Specific Module
```bash
pytest tests/ --cov=src.middleware --cov-report=term-missing
```

---

## 🔍 Debug & Troubleshoot

### Show Print Statements
```bash
pytest tests/ -s
```

### Verbose Output
```bash
pytest tests/ -vv
```

### Full Traceback
```bash
pytest tests/ --tb=long
```

### Run Failed Tests Only
```bash
pytest tests/ --lf
```

### Run Last Failed First
```bash
pytest tests/ --ff
```

---

## ⚡ Performance

### Run Tests in Parallel
```bash
pytest tests/ -n auto -v
```

### Show Slowest Tests
```bash
pytest tests/ --durations=10
```

### Quiet Mode
```bash
pytest tests/ -q
```

---

## 🎯 Specific Test Execution

### Run Specific Test File
```bash
pytest tests/unit/middleware/test_auth_comprehensive.py -v
```

### Run Specific Test Class
```bash
pytest tests/unit/middleware/test_auth_comprehensive.py::TestTokenCreation -v
```

### Run Specific Test Method
```bash
pytest tests/unit/middleware/test_auth_comprehensive.py::TestTokenCreation::test_create_access_token -v
```

### Run Tests Matching Pattern
```bash
pytest tests/ -k "auth" -v       # All tests with "auth" in name
pytest tests/ -k "not slow" -v   # All except slow tests
```

---

## 📋 List Tests Without Running

```bash
# Collect all tests
pytest tests/ --collect-only

# Count tests
pytest tests/ --collect-only -q

# Show test structure
pytest tests/ --collect-only -v
```

---

## 🚀 Full Workflow

### Step 1: Run All Tests with Coverage
```bash
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
```

### Step 2: Check Terminal Output
```
Look for coverage summary at the end
Target: 70%+ coverage
```

### Step 3: Open HTML Report
```bash
# Windows
start htmlcov/index.html

# macOS
open htmlcov/index.html

# Linux
xdg-open htmlcov/index.html
```

### Step 4: Analyze Gaps
```
View the HTML report to see which lines aren't covered
Identify patterns in missing coverage
Plan additional tests if needed
```

---

## 📊 Coverage Targets

| Module | Target |
|--------|--------|
| middleware | 95% |
| patterns | 85% |
| services | 80% |
| utils | 85% |
| ml | 75% |
| domain | 90% |
| analysis | 80% |
| indicators | 90% |
| **OVERALL** | **70%** |

---

## 💡 Pro Tips

### Faster Test Runs
```bash
# Parallel execution (4x faster)
pytest tests/ -n auto -v

# No coverage (faster, no report)
pytest tests/ -v
```

### Focused Testing
```bash
# Only authentication tests
pytest tests/unit/middleware/ -v

# Only test failures
pytest tests/ --lf -v

# Only quick tests (if marked)
pytest tests/ -m "not slow" -v
```

### Better Reports
```bash
# With line coverage details
pytest tests/ --cov=src --cov-report=term-missing:skip-covered

# JSON report for CI/CD
pytest tests/ --cov=src --cov-report=json
```

---

## 📍 VS Code Integration

### Method 1: Terminal
1. Open Terminal (Ctrl+`)
2. Run: `pytest tests/ -v`

### Method 2: Tasks
1. Ctrl+Shift+P
2. Select "Run Task"
3. Choose "Run All Tests"

### Method 3: Python Extension
1. Click "Test" icon in sidebar
2. Right-click test file
3. Select "Run Tests"

---

## 🔗 Configuration

### pytest.ini
```bash
# Contains pytest configuration
# Located at: e:\Shakour\GravityProjects\Gravity_TechnicalAnalysis\pytest.ini
```

### conftest.py
```bash
# Contains test fixtures and configuration
# Located at: tests/conftest.py
```

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| `TEST_ORGANIZATION.md` | Folder structure |
| `TEST_SUMMARY.py` | Statistics & reference |
| `EXECUTION_GUIDE.py` | How to run |
| `PROGRESS_CHECKLIST.md` | Detailed tracking |
| `README_TESTS.md` | Complete guide |
| `QUICK_REFERENCE.md` | This file |

---

## 🎓 Examples

### Example 1: Check Auth Tests
```bash
pytest tests/unit/middleware/test_auth_comprehensive.py -v --tb=short
```

### Example 2: Pattern Tests with Coverage
```bash
pytest tests/unit/patterns/test_patterns_comprehensive.py -v --cov=src.patterns
```

### Example 3: All Tests with Performance
```bash
pytest tests/ -v --durations=5 --cov=src
```

### Example 4: Failed Tests Only
```bash
pytest tests/ --lf -v
```

### Example 5: Generate Full Report
```bash
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
open htmlcov/index.html  # or: start htmlcov/index.html on Windows
```

---

## ✨ Common Workflow

```bash
# 1. Quick test run
pytest tests/ -v

# 2. If passes, run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# 3. If coverage < 70%, generate HTML report
pytest tests/ --cov=src --cov-report=html

# 4. Open report and identify gaps
start htmlcov/index.html  # Windows
# or
open htmlcov/index.html   # macOS

# 5. Add missing tests and repeat
```

---

## 🆘 Troubleshooting

### Tests Not Found
```bash
# Verify tests exist
pytest tests/ --collect-only

# Check for missing __init__.py
# Look in: tests/unit/[category]/__init__.py
```

### Import Errors
```bash
# Check Python path
# Verify conftest.py exists in tests/

# Run with verbose errors
pytest tests/ -vv --tb=long
```

### Coverage Not Generated
```bash
# Make sure coverage is installed
pip install coverage

# Run with explicit coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Slow Tests
```bash
# Check which tests are slow
pytest tests/ --durations=10

# Run only fast tests
pytest tests/ -m "not slow" -v

# Or run in parallel
pytest tests/ -n auto -v
```

---

## 📖 Complete Guide Links

- 📖 Full Setup: See `README_TESTS.md`
- 📊 Progress: See `PROGRESS_CHECKLIST.md`
- 📋 Organization: See `TEST_ORGANIZATION.md`
- 📈 Summary: See `docs/reports/FINAL_TEST_SUMMARY.md`
- ⚙️ Execution: See `EXECUTION_GUIDE.py`

---

**Ready to test! Run your first command above and check the results. 🚀**
