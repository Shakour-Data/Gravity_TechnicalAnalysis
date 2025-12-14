"""
Test Suite Organization Guide

هدف: 70%+ پوشش تست برای تمام ماژول‌ها

صورتش ساختار:
├── tests/
│   ├── accuracy/                    # Accuracy metrics tests
│   │   ├── __init__.py
│   │   ├── test_accuracy_weighting.py
│   │   ├── test_comprehensive_accuracy.py
│   │   └── test_confidence_metrics.py
│   │
│   ├── api/                         # API endpoint tests  
│   │   ├── __init__.py
│   │   ├── test_api_v1_clean.py
│   │   └── test_api_v1_comprehensive_fixed.py
│   │
│   ├── integration/                 # Integration tests
│   │   ├── api/
│   │   │   └── test_analysis_api_real_data.py
│   │   ├── __init__.py
│   │   ├── test_combined_system.py
│   │   ├── test_complete_analysis.py
│   │   └── test_multi_horizon.py
│   │
│   ├── tse_data/                    # Real TSE data tests
│   │   ├── __init__.py
│   │   ├── test_all_with_tse_data.py
│   │   ├── test_phase4_advanced_patterns_tse.py
│   │   ├── test_phase5_edge_cases_stress_tse.py
│   │   └── test_services_with_tse_data.py
│   │
│   ├── unit/                        # Unit tests organized by module
│   │   ├── domain/                  # Domain entities
│   │   │   ├── __init__.py
│   │   │   ├── test_candle.py
│   │   │   ├── test_wave_point.py
│   │   │   ├── test_pattern_type.py
│   │   │   ├── test_pattern_result.py
│   │   │   ├── test_indicator_category.py
│   │   │   └── test_indicator_result.py
│   │   │
│   │   ├── indicators/              # Technical indicators
│   │   │   ├── __init__.py
│   │   │   └── test_indicators_real_data.py
│   │   │
│   │   ├── patterns/                # Pattern recognition
│   │   │   ├── __init__.py
│   │   │   ├── test_classical_patterns.py
│   │   │   ├── test_candlestick_patterns.py
│   │   │   ├── test_elliott.py
│   │   │   └── test_patterns_comprehensive.py
│   │   │
│   │   ├── middleware/              # Security & middleware
│   │   │   ├── __init__.py
│   │   │   └── test_auth_comprehensive.py
│   │   │
│   │   ├── services/                # Business services
│   │   │   ├── __init__.py
│   │   │   ├── test_cache_service_real.py
│   │   │   └── test_analysis_service_comprehensive.py
│   │   │
│   │   ├── ml/                      # Machine learning
│   │   │   ├── __init__.py
│   │   │   ├── test_ml_weights_quick.py
│   │   │   └── test_ml_models_comprehensive.py
│   │   │
│   │   ├── analysis/                # Analysis & market phase
│   │   │   ├── __init__.py
│   │   │   ├── test_cycle.py
│   │   │   ├── test_cycle_complete.py
│   │   │   ├── test_cycle_score.py
│   │   │   ├── test_market_phase.py
│   │   │   ├── test_momentum.py
│   │   │   ├── test_momentum_comprehensive.py
│   │   │   ├── test_momentum_core.py
│   │   │   ├── test_support_resistance.py
│   │   │   ├── test_support_resistance_core.py
│   │   │   ├── test_trend.py
│   │   │   ├── test_trend_complete.py
│   │   │   ├── test_volatility_comprehensive.py
│   │   │   ├── test_volume.py
│   │   │   ├── test_volume_comprehensive.py
│   │   │   ├── test_volume_core.py
│   │   │   └── test_volume_indicators.py
│   │   │
│   │   ├── utils/                   # Utility functions
│   │   │   ├── __init__.py
│   │   │   ├── test_weight_adjustment.py
│   │   │   └── test_utilities_comprehensive.py
│   │   │
│   │   └── __init__.py
│   │
│   ├── conftest.py                  # Shared fixtures
│   ├── README.md                    # Test documentation
│   └── TEST_STRUCTURE.md            # This file
"""
