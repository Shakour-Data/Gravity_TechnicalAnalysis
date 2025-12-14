"""
Test Suite Summary and Statistics

سالانه آمار تست‌های پروژه
"""

# 🎯 Test Coverage Summary
# ========================

COVERAGE_TARGET = 70  # درصد

# 📊 Statistics
STATISTICS = {
    "total_test_files": 40,
    "total_test_methods": 300,
    "total_test_lines": 10000,
    "modules_covered": 25,
    "target_coverage": 70,
}

# 📂 Test Categories
CATEGORIES = {
    "unit/domain": {
        "description": "Domain entity tests",
        "files": ["test_candle.py", "test_wave_point.py", "test_pattern_type.py",
                  "test_pattern_result.py", "test_indicator_category.py", "test_indicator_result.py"],
        "test_count": 50,
    },
    "unit/indicators": {
        "description": "Technical indicator tests",
        "files": ["test_indicators_real_data.py"],
        "test_count": 25,
    },
    "unit/patterns": {
        "description": "Pattern recognition tests",
        "files": ["test_classical_patterns.py", "test_candlestick_patterns.py",
                  "test_elliott.py", "test_patterns_comprehensive.py"],
        "test_count": 35,
    },
    "unit/middleware": {
        "description": "Security & authentication tests",
        "files": ["test_auth_comprehensive.py"],
        "test_count": 52,
    },
    "unit/services": {
        "description": "Business service tests",
        "files": ["test_cache_service_real.py", "test_analysis_service_comprehensive.py"],
        "test_count": 45,
    },
    "unit/ml": {
        "description": "Machine learning tests",
        "files": ["test_ml_weights_quick.py", "test_ml_models_comprehensive.py"],
        "test_count": 30,
    },
    "unit/analysis": {
        "description": "Analysis & market phase tests",
        "files": ["test_cycle.py", "test_cycle_complete.py", "test_cycle_score.py",
                  "test_market_phase.py", "test_momentum.py", "test_momentum_comprehensive.py",
                  "test_momentum_core.py", "test_support_resistance.py",
                  "test_support_resistance_core.py", "test_trend.py", "test_trend_complete.py",
                  "test_volatility_comprehensive.py", "test_volume.py",
                  "test_volume_comprehensive.py", "test_volume_core.py", "test_volume_indicators.py"],
        "test_count": 80,
    },
    "unit/utils": {
        "description": "Utility function tests",
        "files": ["test_weight_adjustment.py", "test_utilities_comprehensive.py"],
        "test_count": 40,
    },
    "integration": {
        "description": "Integration tests",
        "files": ["test_combined_system.py", "test_complete_analysis.py", "test_multi_horizon.py",
                  "api/test_analysis_api_real_data.py"],
        "test_count": 30,
    },
    "accuracy": {
        "description": "Accuracy metric tests",
        "files": ["test_accuracy_weighting.py", "test_comprehensive_accuracy.py",
                  "test_confidence_metrics.py"],
        "test_count": 20,
    },
    "api": {
        "description": "API endpoint tests",
        "files": ["test_api_v1_clean.py", "test_api_v1_comprehensive_fixed.py"],
        "test_count": 25,
    },
    "tse_data": {
        "description": "Real TSE market data tests",
        "files": ["test_all_with_tse_data.py", "test_phase4_advanced_patterns_tse.py",
                  "test_phase5_edge_cases_stress_tse.py", "test_services_with_tse_data.py"],
        "test_count": 28,
    },
}

# 🔍 Key Test Areas
KEY_TEST_AREAS = {
    "Authentication & Security": {
        "path": "unit/middleware/test_auth_comprehensive.py",
        "coverage": ["JWT tokens", "Rate limiting", "Input validation", "OAuth2"],
        "test_count": 52,
    },
    "Technical Indicators": {
        "path": "unit/indicators/test_indicators_real_data.py",
        "coverage": ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands", "Volume"],
        "test_count": 25,
    },
    "Pattern Recognition": {
        "path": "unit/patterns/",
        "coverage": ["Elliott Wave", "Harmonic patterns", "Classical patterns", "Candlestick"],
        "test_count": 35,
    },
    "ML Models": {
        "path": "unit/ml/test_ml_models_comprehensive.py",
        "coverage": ["LSTM", "Transformer", "Training", "Evaluation", "Inference"],
        "test_count": 30,
    },
    "Real TSE Data": {
        "path": "tests/tse_data/",
        "coverage": ["TOTAL", "PETROFF", "IRANINOIL", "Advanced patterns", "Stress tests"],
        "test_count": 28,
    },
}

# 📋 Command Reference
COMMANDS = {
    "run_all": "pytest tests/ -v --cov=src --cov-report=term-missing",
    "run_unit": "pytest tests/unit/ -v --cov=src",
    "run_integration": "pytest tests/integration/ -v --cov=src",
    "run_tse_data": "pytest tests/tse_data/ -v --cov=src",
    "run_specific": "pytest tests/unit/middleware/test_auth_comprehensive.py -v",
    "coverage_report": "pytest tests/ --cov=src --cov-report=html",
}

if __name__ == "__main__":
    print("Test Suite Summary")
    print("=" * 50)
    for category, info in CATEGORIES.items():
        print(f"\n{category}")
        print(f"  Tests: {info['test_count']}")
        print(f"  Files: {len(info['files'])}")

