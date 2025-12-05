"""
Test Execution Guide

راهنمای اجرای تست‌های پروژه
"""

# 🚀 Quick Commands
# =================

"""
# اجرای تمام تست‌ها
pytest tests/ -v --cov=src --cov-report=term-missing

# اجرای unit tests فقط
pytest tests/unit/ -v

# اجرای integration tests
pytest tests/integration/ -v

# اجرای TSE data tests
pytest tests/tse_data/ -v

# اجرای تست‌های مشخصی
pytest tests/unit/middleware/test_auth_comprehensive.py -v

# اجرای با coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# اجرای parallel (سریع‌تر)
pytest tests/ -n auto -v

# اجرای specific test class
pytest tests/unit/middleware/test_auth_comprehensive.py::TestTokenCreation -v

# اجرای با فیلتر
pytest tests/ -k "auth" -v

# تست‌های شامل کلمه auth
pytest tests/ -m "not slow" -v
"""

# 📊 Expected Coverage
EXPECTED_COVERAGE = {
    "middleware": 95,  # Authentication, rate limiting, security
    "indicators": 90,  # Technical indicators
    "patterns": 85,    # Pattern recognition
    "services": 80,    # Business services
    "ml": 75,          # Machine learning
    "utils": 85,       # Utility functions
    "domain": 90,      # Domain entities
    "analysis": 80,    # Market analysis
}

# 🎯 Coverage Target
TARGET = 70  # 70% minimum

# 📈 Test Execution Plan
EXECUTION_PLAN = {
    "phase1": {
        "name": "Unit Tests",
        "commands": [
            "pytest tests/unit/ -v --cov=src",
        ],
        "duration": "~30s",
    },
    "phase2": {
        "name": "Integration Tests",
        "commands": [
            "pytest tests/integration/ -v --cov=src",
        ],
        "duration": "~20s",
    },
    "phase3": {
        "name": "Real TSE Data Tests",
        "commands": [
            "pytest tests/tse_data/ -v --cov=src",
        ],
        "duration": "~40s",
    },
    "phase4": {
        "name": "API Tests",
        "commands": [
            "pytest tests/api/ -v --cov=src",
        ],
        "duration": "~15s",
    },
    "phase5": {
        "name": "Accuracy Tests",
        "commands": [
            "pytest tests/accuracy/ -v --cov=src",
        ],
        "duration": "~10s",
    },
    "full": {
        "name": "Full Test Suite",
        "commands": [
            "pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html",
        ],
        "duration": "~2min",
    },
}

# 📋 Test Categories and Commands
TEST_COMMANDS = {
    "auth": "pytest tests/unit/middleware/test_auth_comprehensive.py -v",
    "indicators": "pytest tests/unit/indicators/ -v",
    "patterns": "pytest tests/unit/patterns/ -v",
    "services": "pytest tests/unit/services/ -v",
    "ml": "pytest tests/unit/ml/ -v",
    "domain": "pytest tests/unit/domain/ -v",
    "analysis": "pytest tests/unit/analysis/ -v",
    "utils": "pytest tests/unit/utils/ -v",
    "integration": "pytest tests/integration/ -v",
    "tse_data": "pytest tests/tse_data/ -v",
    "api": "pytest tests/api/ -v",
    "accuracy": "pytest tests/accuracy/ -v",
    "all": "pytest tests/ -v --cov=src --cov-report=term-missing",
}

# 🔍 Debugging Commands
DEBUG_COMMANDS = {
    "verbose": "pytest tests/ -vv --tb=long",
    "quiet": "pytest tests/ -q",
    "failed_only": "pytest tests/ --lf",
    "last_failed": "pytest tests/ --ff",
    "show_print": "pytest tests/ -s",
}

# 📊 Coverage Analysis
COVERAGE_ANALYSIS = {
    "generate_html": "pytest tests/ --cov=src --cov-report=html",
    "open_report": "open htmlcov/index.html",  # macOS
    "open_report_windows": "start htmlcov/index.html",  # Windows
    "coverage_missing": "pytest tests/ --cov=src --cov-report=term-missing",
}

if __name__ == "__main__":
    print("Test Execution Guide")
    print("=" * 60)
    print("\n📋 Quick Commands:")
    for name, cmd in list(TEST_COMMANDS.items())[:5]:
        print(f"  {name:15} -> {cmd}")
    print("\n... and more! See TEST_COMMANDS dictionary for complete list.")

