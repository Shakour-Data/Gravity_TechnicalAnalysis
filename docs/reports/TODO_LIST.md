# Execution TODO List

This list captures the next phases requested for stabilizing the Gravity Technical Analysis stack. Tasks are grouped by priority; higher priority items should be addressed first. Use the checkboxes to track progress and extend the list as work evolves.

## P0 - Unblock the Multi-Horizon Pipeline

- [x] Load trained `MultiHorizonWeightLearner` artifacts and inject them into `MultiHorizonAnalyzer`, `MultiHorizonMomentumAnalyzer`, and the other dimension analyzers inside `CompleteAnalysisPipeline` (`src/gravity_tech/ml/complete_analysis_pipeline.py`).
- [x] Replace raw candle inputs with cached feature dictionaries so analyzers receive the expected `{feature_name: value}` payloads (reusable cache/service).
- [x] Ensure `_trend_score`, `_momentum_score`, `_volatility_score`, `_cycle_score`, and `_sr_score` store the full `*Score` dataclasses (e.g., `TrendScore`) and pass those objects into the volume and 5D matrices instead of floats.
- [x] Fix `MultiHorizonMomentumAnalysis.to_dict` so it exposes `signal` (or rename the field) instead of referencing the nonexistent `trend_signal` attribute (`src/gravity_tech/ml/multi_horizon_momentum_analysis.py:68-95`).
- [x] Add an end-to-end regression test that exercises `CompleteAnalysisPipeline.analyze()` with the injected learners to guarantee the pipeline produces a decision object (`tests/integration/test_complete_analysis_pipeline.py`).
- [x] Wire `gravity_tech.cli.run_complete_pipeline` to accept trained cycle/support-resistance analyzers generated from saved artifacts (see `src/gravity_tech/ml/pipeline_factory.py`).
- [x] Extend the pipeline CLI to fetch candles on demand via `DataConnector` (`--symbol/--interval/--limit`) and cover it with regression tests.

### P0 - Code Quality (models/schemas.py)
- [x] Move all imports to the top of the file to satisfy E402.
- [x] Sort and format imports (I001) and remove unused imports (F401).
- [x] Update type annotations to modern syntax (UP006, UP007).
- [x] Fix `TechnicalAnalysisResult` redeclaration (F811) and validator import/deprecation issues (UP035).
- [x] Fix the attribute access issue around line 422.
- [x] Remove trailing whitespace/blank lines with whitespace (W291, W293).

## P1 - Feature Extraction & Model Performance

- [x] Vectorize `MultiHorizonFeatureExtractor` (indicator + dimension modes) to avoid recomputing indicators per window; follow the caching pattern used by `MultiHorizonMomentumFeatureExtractor`.
- [x] Introduce a cached series builder + dataset extractor for `MultiHorizonVolatilityFeatureExtractor`, and switch its imports to `gravity_tech.core.domain.entities.Candle`.
- [x] Replace `print` statements inside the trend/dimension/momentum/volatility extractors with structured logging that can be silenced in tests.
- [x] Document and expose tunable divergence/cycle/window parameters for all extractors so training scripts can adjust performance (momentum divergence lookback, volatility ATR span, trend/volatility lookback/horizons).

## P1 - Runtime Hardening

- [x] Update `DataConnector` so mock fallback is opt-in (not default) and emit telemetry (counters/timers) to detect remote failures (`src/gravity_tech/ml/data_connector.py`).
- [x] Add regression coverage for the CLI candle-fetch flags to ensure connector failures/logging are surfaced cleanly.
- [x] Implement concurrency/throttling for `fetch_multiple_symbols` to support large watchlists without serial bottlenecks.
- [x] Cache ML models in `api/v1/ml.py`, move inference and pickle loading off the event loop, and include model metadata (hash/version) in responses.
- [x] Implement `SupportResistanceIndicators.calculate_all(...)` (or equivalent) and enable the TODO block in `TechnicalAnalysisService` so the overall signal reflects that dimension (`src/gravity_tech/services/analysis_service.py`).

## P2 - Tooling, API Surface, and QA

- [x] Replace the placeholder logic in `api/v1/tools.py` with a real tool registry and wire it to a working `DynamicToolRecommender` (load/save/train flows in `src/gravity_tech/ml/ml_tool_recommender.py`).
- [x] Provide deterministic fixtures or cached weights for `tests/integration/test_combined_system.py` so CI does not retrain LightGBM models on every run.
- [x] Add real assertions to `tests/unit/analysis/test_cycle_score.py` (verify overall signals/confidence) and remove the print-only diagnostics.
- [x] Supply the missing `k8s/*.yaml` manifests referenced in `docs/operations/DEPLOYMENT_GUIDE.md` or update the guide to match the actual deployment assets.
- [x] Add a lightweight CLI/pytest hook around `ml/backtesting.py` to run regression backtests without mutating `sys.path`.
- [x] Wire backtesting and any remaining demo paths to real market data (DataConnector/DB), replacing synthetic data defaults.

## P3 — Documentation & Telemetry

- [ ] Keep `docs/PROCESS_DOCUMENTATION.md` in sync with the actual remediation progress (link to PRs/issues once created).
- [ ] Expand the deployment/runbook section with observability steps (metrics/alerts) once DataConnector/API telemetry is in place.
- [ ] Capture the final architecture of the feature-cache/weight-loader components so new contributors can extend the pipeline without re-learning the wiring.
