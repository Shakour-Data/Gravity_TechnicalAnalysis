# Unit Test Creation Plan for 95% Coverage (~1000 Tests)

## Overview
Create comprehensive unit tests for all Python source files in `src/` directory and subdirectories. Target: 95% test coverage with approximately 1000 total tests. Tests will be placed in `tests/unit/` directory using pytest framework.

## Source Files to Test
Based on directory structure, the following source files need unit tests:

### Core Domain Entities (src/core/domain/entities/)
- candle.py (Candle dataclass, properties, methods)
- decision.py
- elliott_wave_result.py
- indicator_category.py
- indicator_result.py
- pattern_result.py
- pattern_type.py
- signal_strength.py
- signal.py
- wave_point.py

### Core Indicators (src/core/indicators/)
- cycle.py
- momentum.py (functions + MomentumIndicators class)
- support_resistance.py
- trend.py
- volatility.py
- volume_day3.py
- volume.py

### Core Patterns (src/core/patterns/)
- candlestick.py
- classical.py
- divergence.py
- elliott_wave.py

### Core Analysis (src/core/analysis/)
- market_phase.py

### Gravity Tech Main (src/gravity_tech/)
- main.py (FastAPI app, endpoints, middleware)

### Gravity Tech API (src/gravity_tech/api/)
- response_formatters.py
- v1/ml.py
- v1/patterns.py
- v1/scenarios.py
- v1/tools.py

### Gravity Tech Clients (src/gravity_tech/clients/)
- data_service_client.py

### Gravity Tech Config (src/gravity_tech/config/)
- settings.py

### Gravity Tech Database (src/gravity_tech/database/)
- historical_manager.py
- schemas.sql (if contains Python code)

### Gravity Tech Indicators (src/gravity_tech/indicators/)
- cycle.py
- momentum.py
- support_resistance.py
- trend.py
- volatility.py
- volume.py

### Gravity Tech Middleware (src/gravity_tech/middleware/)
- auth.py
- events.py
- logging.py
- resilience.py
- security.py
- service_discovery.py
- tracing.py

### Gravity Tech ML (src/gravity_tech/ml/) - Large module
- advanced_pattern_training.py
- backtesting.py
- combined_trend_momentum_analysis.py
- complete_analysis_pipeline.py
- continuous_learning.py
- data_connector.py
- feature_extraction.py
- five_dimensional_decision_matrix.py
- integrated_multi_horizon_analysis.py
- ml_dimension_weights.py
- ml_indicator_weights.py
- model_interpretability.py
- multi_horizon_analysis.py
- multi_horizon_cycle_analysis.py
- multi_horizon_cycle_features.py
- multi_horizon_feature_extraction.py
- multi_horizon_momentum_analysis.py
- multi_horizon_momentum_features.py
- multi_horizon_support_resistance_analysis.py
- multi_horizon_support_resistance_features.py
- multi_horizon_volatility_analysis.py
- multi_horizon_volatility_features.py
- multi_horizon_weights.py
- pattern_classifier.py
- pattern_features.py
- scenario_weight_optimizer.py
- train_multi_horizon_cycle.py
- train_multi_horizon_momentum.py
- train_multi_horizon_support_resistance.py
- train_multi_horizon_volatility.py
- train_multi_horizon.py
- train_pipeline.py
- train_volume_dimension_matrix.py
- train_weights.py
- volume_dimension_matrix.py
- weight_optimizer.py

### Gravity Tech Models (src/gravity_tech/models/)
- schemas.py
- schemas_backup.py

### Gravity Tech Patterns (src/gravity_tech/patterns/)
- candlestick.py
- classical.py
- divergence.py
- elliott_wave.py
- harmonic.py

### Gravity Tech Services (src/gravity_tech/services/)
- analysis_service.py
- cache_service.py
- fast_indicators.py
- performance_optimizer.py
- tool_recommendation_service.py

### Gravity Tech Utils (src/gravity_tech/utils/)
- display_formatters.py
- sample_data.py

## Test Creation Strategy
- Create one test file per source file (e.g., `test_candle.py` for `candle.py`)
- Use pytest with fixtures, parametrized tests, and comprehensive assertions
- Cover all functions/methods with multiple test cases:
  - Normal operation
  - Edge cases
  - Error conditions
  - Boundary values
- Aim for 10-20 tests per file (adjust based on complexity)
- Use mocking for external dependencies (Redis, databases, etc.)
- Test both success and failure paths

## Implementation Steps

### Phase 1: Core Domain Entities (Priority: High)
- [x] Create test_candle.py (15-20 tests for Candle class, properties, methods)
- [x] Create test_decision.py
- [ ] Create test_elliott_wave_result.py
- [x] Create test_indicator_category.py
- [x] Create test_indicator_result.py
- [x] Create test_pattern_result.py
- [x] Create test_pattern_type.py
- [x] Create test_signal.py
- [x] Create test_signal_strength.py
- [ ] Create test_wave_point.py

### Phase 2: Core Indicators (Priority: High)
- [ ] Create test_cycle.py
- [x] Create test_momentum.py (20+ tests for functions + class methods)
- [ ] Create test_support_resistance.py
- [ ] Create test_trend.py
- [ ] Create test_volatility.py
- [ ] Create test_volume_day3.py
- [ ] Create test_volume.py

### Phase 3: Core Patterns (Priority: High)
- [ ] Create test_candlestick_patterns.py
- [ ] Create test_classical_patterns.py
- [ ] Create test_divergence_patterns.py
- [ ] Create test_elliott_wave_patterns.py

### Phase 4: Core Analysis
- [ ] Create test_market_phase.py

### Phase 5: Gravity Tech Main & API
- [ ] Create test_main.py (tests for FastAPI endpoints, middleware)
- [ ] Create test_response_formatters.py
- [ ] Create test_api_ml.py
- [ ] Create test_api_patterns.py
- [ ] Create test_api_scenarios.py
- [ ] Create test_api_tools.py

### Phase 6: Gravity Tech Infrastructure
- [ ] Create test_data_service_client.py
- [ ] Create test_settings.py
- [ ] Create test_historical_manager.py

### Phase 7: Gravity Tech Indicators
- [ ] Create test_gravity_cycle.py
- [ ] Create test_gravity_momentum.py
- [ ] Create test_gravity_support_resistance.py
- [ ] Create test_gravity_trend.py
- [ ] Create test_gravity_volatility.py
- [ ] Create test_gravity_volume.py

### Phase 8: Gravity Tech Middleware
- [ ] Create test_auth.py
- [ ] Create test_events.py
- [ ] Create test_logging.py
- [ ] Create test_resilience.py
- [ ] Create test_security.py
- [ ] Create test_service_discovery.py
- [ ] Create test_tracing.py

### Phase 9: Gravity Tech ML (Priority: Medium - Large module)
- [ ] Create test_advanced_pattern_training.py
- [ ] Create test_backtesting.py
- [ ] Create test_combined_trend_momentum_analysis.py
- [ ] Create test_complete_analysis_pipeline.py
- [ ] Create test_continuous_learning.py
- [ ] Create test_data_connector.py
- [ ] Create test_feature_extraction.py
- [ ] Create test_five_dimensional_decision_matrix.py
- [ ] Create test_integrated_multi_horizon_analysis.py
- [ ] Create test_ml_dimension_weights.py
- [ ] Create test_ml_indicator_weights.py
- [ ] Create test_model_interpretability.py
- [ ] Create test_multi_horizon_analysis.py
- [ ] Create test_multi_horizon_cycle_analysis.py
- [ ] Create test_multi_horizon_cycle_features.py
- [ ] Create test_multi_horizon_feature_extraction.py
- [ ] Create test_multi_horizon_momentum_analysis.py
- [ ] Create test_multi_horizon_momentum_features.py
- [ ] Create test_multi_horizon_support_resistance_analysis.py
- [ ] Create test_multi_horizon_support_resistance_features.py
- [ ] Create test_multi_horizon_volatility_analysis.py
- [ ] Create test_multi_horizon_volatility_features.py
- [ ] Create test_multi_horizon_weights.py
- [ ] Create test_pattern_classifier.py
- [ ] Create test_pattern_features.py
- [ ] Create test_scenario_weight_optimizer.py
- [ ] Create test_train_multi_horizon_cycle.py
- [ ] Create test_train_multi_horizon_momentum.py
- [ ] Create test_train_multi_horizon_support_resistance.py
- [ ] Create test_train_multi_horizon_volatility.py
- [ ] Create test_train_multi_horizon.py
- [ ] Create test_train_pipeline.py
- [ ] Create test_train_volume_dimension_matrix.py
- [ ] Create test_train_weights.py
- [ ] Create test_volume_dimension_matrix.py
- [ ] Create test_weight_optimizer.py

### Phase 10: Gravity Tech Models & Patterns
- [ ] Create test_schemas.py
- [ ] Create test_gravity_candlestick_patterns.py
- [ ] Create test_gravity_classical_patterns.py
- [ ] Create test_gravity_divergence_patterns.py
- [ ] Create test_gravity_elliott_wave_patterns.py
- [ ] Create test_harmonic_patterns.py

### Phase 11: Gravity Tech Services
- [ ] Create test_analysis_service.py
- [ ] Create test_cache_service.py
- [ ] Create test_fast_indicators.py
- [ ] Create test_performance_optimizer.py
- [ ] Create test_tool_recommendation_service.py

### Phase 12: Gravity Tech Utils
- [ ] Create test_display_formatters.py
- [ ] Create test_sample_data.py

## Quality Assurance
- [ ] Run pytest --cov=src --cov-report=html to verify coverage
- [ ] Ensure 95% coverage achieved
- [ ] Review test quality and completeness
- [ ] Fix any failing tests or coverage gaps

## Progress Tracking
- Total source files: ~90+
- Tests created: 9/90
- Estimated tests: 160/1000
- Coverage: ~16%/95%
