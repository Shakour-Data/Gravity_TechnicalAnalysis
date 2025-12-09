# Process Documentation & Issue Ledger

This document maps every major workflow in the Gravity Technical Analysis project, explains how each layer works today, and tracks the issues blocking a production-ready rollout. It is intended to replace the partially garbled `PROCESS_AUDIT_FA.md` and will be updated as remediation work lands.

## Latest Remediation (Dec 2025)
- Multi-horizon pipeline wired to cached features and injected learners; end-to-end regression test added.
- DataConnector: mock fallback now opt-in, telemetry counters added, concurrent multi-symbol fetch supported.
- ML API caches models (hash/version exposed) and offloads inference loading from the event loop.
- Tooling API now backed by `ToolRecommendationService` instead of stubs.
- Support/Resistance indicators are active in `TechnicalAnalysisService` via `calculate_all`.
- Backtesting aligned to real data (TSE DB or DataConnector) with a new CLI wrapper; synthetic mode is explicit only.
- Kubernetes manifests supplied under `infra/k8s` and deployment guide paths updated.

## Process Inventory

| # | Process | Purpose | Primary entrypoints | Health | Key blockers |
|---|---------|---------|---------------------|--------|--------------|
| 1 | Market data acquisition | Fetch and normalize OHLCV candles with retries/mock fallback | `src/gravity_tech/ml/data_connector.py` | Needs hardening | Mock fallback hides outages |
| 2 | Trend & dimension feature extraction | Build indicator/dimension matrices for multi-horizon training | `src/gravity_tech/ml/multi_horizon_feature_extraction.py` | Degraded | O(n^2) indicator recomputation, noisy IO |
| 3 | Momentum feature extraction | Pre-compute vectorized momentum indicators & divergences | `src/gravity_tech/ml/multi_horizon_momentum_features.py` | Stable | Only missing structured logging |
| 4 | Volatility feature extraction | Derive 8 volatility signals per horizon | `src/gravity_tech/ml/multi_horizon_volatility_features.py` | Needs rewrite | Deprecated imports, repeated indicator calls |
| 5 | Weight learning & analyzers | Train/load multi-horizon learners and expose analyzers | `src/gravity_tech/ml/multi_horizon_weights.py`, `ml/train_pipeline.py` | Broken | Learners never injected into runtime pipeline |
| 6 | Complete analysis pipeline | Orchestrate five dimensions + volume + 5D decision | `src/gravity_tech/ml/complete_analysis_pipeline.py` | Broken | Wrong inputs (candles vs features), type mismatches |
| 7 | Volume & 5D decision matrices | Fuse dimension scores with volume context | `src/gravity_tech/ml/volume_dimension_matrix.py`, `five_dimensional_decision_matrix.py` | Blocked | Receive floats instead of `*Score` dataclasses |
| 8 | ML inference API | FastAPI endpoints for pattern models | `src/gravity_tech/api/v1/ml.py` | Degraded | Model reload per request, blocking CPU work |
| 9 | Technical analysis service | Indicator orchestration & market phase outcome | `src/gravity_tech/services/analysis_service.py` | Partially complete | Support/resistance pipeline still TODO |
|10 | Tool recommendation & scenario optimizer | Recommend indicator stacks per regime | `src/gravity_tech/api/v1/tools.py`, `ml/ml_tool_recommender.py` | Not implemented | Training/saving/loading stubs |
|11 | Backtesting & evaluation | Pattern PnL/regression harness | `src/gravity_tech/ml/backtesting.py` | Needs refactor | Uses ad-hoc `sys.path`, lacks fixtures |
|12 | Deployment & QA | Runbooks, regression tests, coverage | `docs/operations/DEPLOYMENT_GUIDE.md`, `tests/*` | Mixed | Missing k8s assets, slow/print-only tests |

---

## 1. Market Data Acquisition (`DataConnector`)

**Workflow summary**
- `DataConnector.fetch_daily_candles` pulls OHLCV data from the historical-data microservice with retry/backoff and falls back to `_generate_mock_data` when the request fails (`src/gravity_tech/ml/data_connector.py:28-95`).
- `fetch_multiple_symbols` iterates `fetch_daily_candles` sequentially per symbol (`src/gravity_tech/ml/data_connector.py:96-109`).

**Key artifacts**
- `settings.DATA_SERVICE_*` defaults in `src/gravity_tech/config/settings.py:37-57`.
- Mock data generator that synthesizes candles with deterministic drift/noise.

**Open issues**
- **[High]** Mock fallback is enabled by default via `allow_mock_on_failure=True` and silently replaces remote data whenever HTTP retries are exhausted (`src/gravity_tech/ml/data_connector.py:28-94`). In production this makes outages invisible and poisons downstream training.
- **[Medium]** `fetch_multiple_symbols` processes symbols sequentially and lacks throttling or concurrency controls (`src/gravity_tech/ml/data_connector.py:96-109`), making large universes painfully slow.
- **[Low]** There is no structured telemetry (success/failure counters, latency histograms) beyond a single warning log, so SREs cannot alert on degraded data feeds.

**Remediation**
1. Require explicit `allow_mock_on_failure` in constructors/config and raise when unset in prod environments.
2. Add async/threaded multi-symbol fetching (or allow a custom executor) with rate limiting.
3. Emit Prometheus-friendly counters/timers for both remote and mock pathways.

---

## 2. Trend & Dimension Feature Extraction

**Workflow summary**
- `MultiHorizonFeatureExtractor.extract_indicator_features` computes 7 indicator bundles per window and emits signal/confidence/weighted triples (`src/gravity_tech/ml/multi_horizon_feature_extraction.py:38-134`).
- `extract_dimension_features` aggregates candlestick, Elliott, and classical patterns into four higher-level dimensions (`src/gravity_tech/ml/multi_horizon_feature_extraction.py:136-221`).
- `extract_training_dataset` slides over the candle set, recomputing indicators for each window, printing progress, and collecting returns (`src/gravity_tech/ml/multi_horizon_feature_extraction.py:244-309`).

**Open issues**
- **[High]** `extract_training_dataset` repeatedly invokes indicator/pattern functions inside the sliding loop (one full indicator pass per sample), resulting in O(n * lookback * indicator_cost) runtime and multi-minute training even on synthetic data (`src/gravity_tech/ml/multi_horizon_feature_extraction.py:244-287`).
- **[Medium]** The method uses `print` for status/warnings (`src/gravity_tech/ml/multi_horizon_feature_extraction.py:260-308`), which pollutes pytest output and prevents structured logging or suppression in production CLIs.
- **[Medium]** Candlestick and classical pattern detection operate on overlapping slices each iteration without caching deduplicated signals (`src/gravity_tech/ml/multi_horizon_feature_extraction.py:175-220`), so pattern latency grows with lookback.

**Remediation**
1. Mirror the vectorized approach used in `MultiHorizonMomentumFeatureExtractor`, precomputing pandas/numpy arrays once and slicing views per window.
2. Replace `print` with a logger that can be silenced or directed to stdout only when `verbose=True`.
3. Cache pattern lookups (e.g., preprocess pattern signals for each index) to avoid repeated detection over identical subwindows.

---

## 3. Momentum Feature Extraction

**Workflow summary**
- `MultiHorizonMomentumFeatureExtractor` builds `MomentumSeriesCache` containing RSI/Stochastic/CCI/etc. across the full dataset (`src/gravity_tech/ml/multi_horizon_momentum_features.py:32-79`).
- `extract_training_dataset` leverages the cache and a deque window to emit feature/target matrices without recomputing indicators (`src/gravity_tech/ml/multi_horizon_momentum_features.py:90-183`).

**Open issues**
- **[Low]** The extractor still emits direct `print` statements for progress/summary logs (`src/gravity_tech/ml/multi_horizon_momentum_features.py:110-165`), which makes automated training noisy.
- **[Low]** Divergence detection is hard-coded with `lookback=20` and there is no way to tune or disable it for lighter workloads.

**Remediation**
1. Gate console output behind a `verbose` flag and wire it to a logger.
2. Surface divergence settings via constructor parameters (with documented defaults) so that light-weight experiments can skip the overhead.

---

## 4. Volatility Feature Extraction

**Workflow summary**
- `MultiHorizonVolatilityFeatureExtractor` iterates all eight volatility indicators, collecting signal/confidence/normalized/percentile values (`src/gravity_tech/ml/multi_horizon_volatility_features.py:60-234`).
- Outputs are intended to feed the same multi-horizon learners, although there is no dataset extractor yet.

**Open issues**
- **[High]** The module imports `Candle` from the deprecated `gravity_tech.models.schemas` shim (`src/gravity_tech/ml/multi_horizon_volatility_features.py:15`). This bypasses domain validations and will break when the compatibility layer is removed.
- **[High]** Each indicator is recomputed from scratch for every call and there is no shared cache, making an eventual dataset extractor O(n * indicator_count) per sample (`src/gravity_tech/ml/multi_horizon_volatility_features.py:83-233`).
- **[Medium]** Error paths use `print(f"Warning: ...")` instead of structured logging (`src/gravity_tech/ml/multi_horizon_volatility_features.py:95-232`), so failures cannot be captured via log aggregation.

**Remediation**
1. Switch to `from gravity_tech.core.domain.entities import Candle` and ensure typing stays consistent with other extractors.
2. Introduce a cached series builder (akin to the momentum cache) and a training-dataset method that shares indicator arrays across windows.
3. Replace raw prints with logger warnings containing indicator names and current indices.

---

## 5. Weight Learning & Analyzer Wiring

**Workflow summary**
- `MultiHorizonWeightLearner` trains a multi-output LightGBM model and exports/imports per-horizon weights (`src/gravity_tech/ml/multi_horizon_weights.py:20-360`).
- Legacy `MLTrainingPipeline` orchestrates indicator/dimension learners using the old `FeatureExtractor` (`src/gravity_tech/ml/train_pipeline.py:20-210`).
- Tests build analyzers by training ad-hoc learners inside fixtures (`tests/integration/test_combined_system.py:50-199`).

**Open issues**
- **[Critical]** `CompleteAnalysisPipeline` constructs `MultiHorizonAnalyzer()` and `MultiHorizonMomentumAnalyzer()` without passing a `MultiHorizonWeightLearner`, resulting in an immediate `TypeError` (`src/gravity_tech/ml/complete_analysis_pipeline.py:172-205`).
- **[Critical]** The same pipeline passes raw candle lists into `.analyze(...)`, but analyzers expect precomputed feature dictionaries (`src/gravity_tech/ml/complete_analysis_pipeline.py:173-206` vs. `src/gravity_tech/ml/multi_horizon_analysis.py:142-206`), so even with weight injection it would fail.
- **[Critical]** `_trend_score`, `_momentum_score`, etc., are assigned floats (`combined_score`) yet later treated as dataclasses with `.score`/`.signal` attributes (`src/gravity_tech/ml/complete_analysis_pipeline.py:173-208`). Downstream consumers receive `float` and crash inside the decision matrices.
- **[High]** Runtime never loads the persisted weights under `ml_models/multi_horizon/*.json`, so any production call would retrain or crash due to missing models.
- **[Medium]** `MultiHorizonMomentumAnalysis.to_dict` references a nonexistent `trend_signal` attribute, breaking serialization (`src/gravity_tech/ml/multi_horizon_momentum_analysis.py:68-95`).
- **[Medium]** `ml/train_pipeline.py` still uses the legacy `FeatureExtractor` pipeline and does not export the new multi-horizon JSON weight format expected by the analyzers.

**Remediation**
1. Build a small DI/orchestrator layer that loads `MultiHorizonWeightLearner.load_weights(...)` for trend/momentum/volatility/cycle/SR and injects them into analyzers.
2. Introduce feature-service wrappers that cache the latest feature vector per candle batch and pass `dict[str, float]` into analyzers.
3. Switch `_trend_score` et al. to hold the actual `TrendScore`/`MomentumScore` objects (use `analysis.to_trend_score()` or equivalent).
4. Patch `MultiHorizonMomentumAnalysis.to_dict` to expose `.signal` instead of `.trend_signal`.
5. Modernize `ml/train_pipeline.py` (or add a new CLI) that trains multi-horizon learners and writes artifacts under `ml_models/multi_horizon`.

---

## 6. Complete Analysis Pipeline & Combined Analyzer

**Workflow summary**
- `CompleteAnalysisPipeline.analyze` runs the five base dimensions, optional volume matrix, and the `FiveDimensionalDecisionMatrix` to produce a final decision (`src/gravity_tech/ml/complete_analysis_pipeline.py:118-251`).
- `CombinedTrendMomentumAnalyzer` fuses trend and momentum outputs across horizons for human-readable recommendations (`src/gravity_tech/ml/combined_trend_momentum_analysis.py:25-167`).

**Open issues**
- **[Critical]** Because the base analyzers cannot be constructed (see section 5), the entire pipeline is unusable; `PipelineResult` never materializes.
- **[High]** Even after injection, `_calculate_volume_interactions` forwards floats to `VolumeDimensionMatrix.calculate_all_interactions`, which expects `TrendScore`, `MomentumScore`, etc. (`src/gravity_tech/ml/complete_analysis_pipeline.py:213-220` vs. `src/gravity_tech/ml/volume_dimension_matrix.py:966-1011`).
- **[Medium]** There is no coordination between trend and momentum feature extraction; `CombinedTrendMomentumAnalyzer` receives dicts built ad-hoc in tests but no production-grade feature service exists.

**Remediation**
1. Once analyzers accept real learners, wrap them behind a `FeatureCache` that produces both trend and momentum feature dicts from the same candle window.
2. Update `_trend_score` assignments to store `TrendScore` from `MultiHorizonAnalysis.to_trend_score()` and pass those dataclasses into the volume/5D matrices.
3. Add regression coverage for `CompleteAnalysisPipeline.analyze()` after the wiring fix to ensure a full end-to-end run works with saved weights.

---

## 7. Volume & Five-Dimensional Decision Matrices

**Workflow summary**
- `VolumeDimensionMatrix` derives OBV/volume metrics and determines interaction types per dimension (`src/gravity_tech/ml/volume_dimension_matrix.py:1-1042`).
- `FiveDimensionalDecisionMatrix.analyze` weights dimension states, applies optional volume adjustments, and emits a `FiveDimensionalDecision` (`src/gravity_tech/ml/five_dimensional_decision_matrix.py:1-260`).

**Open issues**
- **[High]** Because `CompleteAnalysisPipeline` sends floats, `calculate_volume_trend_interaction` and siblings cannot access `score`, `signal`, or `accuracy`, causing attribute errors before any decision is produced (`src/gravity_tech/ml/volume_dimension_matrix.py:1000-1011`).
- **[Medium]** Volume adjustments currently re-estimate OBV and squeezes on every call; once pipeline wiring is fixed, we should cache these metrics per candle batch to avoid recomputation across concurrent requests.

**Remediation**
1. Fix upstream typing (see section 6) so that the volume matrix receives full score objects.
2. Expose a `VolumeMetrics` cache that `CompleteAnalysisPipeline` can reuse when running multiple decisions on the same candle slice.

---

## 8. ML Inference API (`api/v1/ml.py`)

**Workflow summary**
- `/ml/predict` and `/ml/predict/batch` load a pickled classifier, run inference, and return probabilities (`src/gravity_tech/api/v1/ml.py:179-310`).
- `/ml/backtest` replays price arrays through the backtesting engine.

**Open issues**
- **[High]** `load_ml_model()` is invoked inside every request handler, reopening the pickle on each call and wasting 10-40 ms plus GC pressure (`src/gravity_tech/api/v1/ml.py:155-218`).
- **[Medium]** The endpoints are declared `async` but run all CPU/disk work inline, blocking the event loop and tanking throughput under load (`src/gravity_tech/api/v1/ml.py:185-218`).
- **[Medium]** There is no model cache invalidation strategy or version pinning; whichever pickle exists last wins silently.

**Remediation**
1. Cache the `(model, version)` tuple at router import time (or behind an `lru_cache`) and refresh via a background reload hook.
2. Move inference into `run_in_executor` or convert handlers to `def` so FastAPI does not expect non-blocking behavior.
3. Surface model metadata (hash/version) in responses and logs so ops can detect mismatches.

---

## 9. Technical Analysis Service

**Workflow summary**
- `TechnicalAnalysisService.analyze` executes indicator calculations (fast path + canonical), pattern detection, Elliott waves, and market phase scoring inside a background thread (`src/gravity_tech/services/analysis_service.py:24-138`).

**Open issues**
- **[Medium]** Support/resistance indicators are still disabled with a `TODO` because `SupportResistanceIndicators` lacks a `calculate_all` entrypoint (`src/gravity_tech/services/analysis_service.py:101-105`), so the overall signal never reflects that dimension.
- **[Low]** Fast indicator caching is per-request and there is no shared cache or rate limiting, meaning bursty workloads still recompute expensive metrics.

**Remediation**
1. Implement `SupportResistanceIndicators.calculate_all(...)` and enable the commented section so that S/R signals populate `result.support_resistance_indicators`.
2. Consider memoizing `FastBatchAnalyzer.analyze_all_indicators` per `(symbol, timeframe, len(candles))` for short-lived re-use (especially in integration tests).

---

## 10. Tool Recommendation & Scenario Optimizer

**Workflow summary**
- FastAPI endpoints under `/tools` expose tool catalogs, presets, and "custom analysis" hooks (`src/gravity_tech/api/v1/tools.py:1-660`).
- `ml/ml_tool_recommender.py` contains the planned ML recommender plus scenario optimizer support.

**Open issues**
- **[High]** The router is stubbed: six separate `# TODO` markers leave registry lookups, recommender integration, and scenario execution unimplemented (`src/gravity_tech/api/v1/tools.py:22,260,360,475,558,639`).
- **[High]** `DynamicToolRecommender.train_recommender/save_model/load_model` are placeholders that simply print "not implemented" and never persist artifacts (`src/gravity_tech/ml/ml_tool_recommender.py:610-650`).
- **[Medium]** There is no contract describing required training data (market regimes, tool outcomes) or how to plug it into CI/CD, so the process cannot be executed end-to-end.

**Remediation**
1. Define and persist a canonical tool registry (JSON or DB) and wire it through the router instead of placeholder dicts.
2. Implement the recommender training loop using historical tool performance and persist the model under `ml_models/tools/`.
3. Add integration tests that cover at least one `/tools` happy path with a mocked recommender so regressions are caught automatically.

---

## 11. Backtesting & Evaluation

**Workflow summary**
- `PatternBacktester` simulates trades using harmonic pattern detection and optional ML classifiers (`src/gravity_tech/ml/backtesting.py`).
- CLI wrapper `gravity_tech.cli.run_backtesting` runs backtests against real data (TSE DB or DataConnector) or explicit synthetic mode; synthetic generation is opt-in only.

**Open issues**
- **[Low]** No Prometheus-style metrics for backtest runs (latency/trade counts) and no persisted fixtures for deterministic regression.

**Remediation**
1. Add a tiny deterministic fixture (e.g., cached OHLCV slice) to exercise the CLI in CI without hitting external data.
2. Emit optional metrics or structured logs (JSON) for backtest outcomes to aid comparison across runs.

---

## 12. Deployment, Operations, and QA

**Workflow summary**
- `docs/operations/DEPLOYMENT_GUIDE.md` documents Kubernetes deployment; manifests now live under `infra/k8s/` (namespace/RBAC/redis/app/service/ingress/hpa/monitoring).
- Tests under `tests/` cover unit/slow/CLI paths; integration suites using TSE data were removed pending a lightweight replacement.

**Open issues**
- **[Medium]** Observability values in `infra/k8s/monitoring.yaml` are placeholders; alert thresholds and namespaces need environment-specific tuning.
- **[Medium]** Some regression coverage for real-data paths (e.g., TSE DB) is missing after removing heavy integration suites.
- **Low]** A few utilities still log localized strings that may render poorly on Windows consoles.

**Remediation**
1. Calibrate PrometheusRule thresholds once telemetry is live; document scrape targets and alerts in the deployment guide.
2. Reintroduce a slim real-data regression (small DB fixture or connector stub) to cover end-to-end analysis without long runtimes.
3. Normalize logs to ASCII where feasible or ensure consoles use UTF-8.

---

## 13. Observability & Telemetry

**Current state**
- DataConnector tracks success/failure/mock counts and latency in-memory; no Prometheus export yet.
- ML API caches models and returns hash/version metadata, but inference latency/cache-hit metrics are not exposed.
- ServiceMonitor/PrometheusRule stubs exist in `infra/k8s/monitoring.yaml`.

**Actions**
1. Add Prometheus counters/timers around DataConnector fetches (remote vs mock) and ML inference (latency, cache hits/misses).
2. Expose `/metrics` from the FastAPI app and validate ServiceMonitor scraping; include example `curl` in the deployment guide.
3. Tune alert thresholds for error rates/latency based on baseline traffic and document them.

---

## 14. Feature Cache & Weight Loader Architecture

**Current design**
- Multi-horizon analyzers consume cached feature dicts; weight artifacts reside under `ml_models/multi_horizon/*.json|pkl` and are injected via `pipeline_factory.build_pipeline_from_weights`.
- CLI `run_complete_pipeline` fetches candles via DataConnector (or file) and feeds cached features into analyzers; `run_backtesting` aligns pattern backtests to real data sources.

**Actions**
1. Document the end-to-end flow: candles → feature cache (trend/momentum/volatility/cycle/SR) → analyzers with injected weights → volume/5D matrices → decision.
2. Record artifact versions/hashes required for each analyzer and where to update them after retraining.
3. Add a small diagram/table mapping CLIs (`run_complete_pipeline`, `run_backtesting`) to inputs, artifacts, and outputs for onboarding.
