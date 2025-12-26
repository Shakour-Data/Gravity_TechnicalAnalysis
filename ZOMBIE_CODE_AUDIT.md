# 🧟 Zombie Code Audit Report
**Gravity Technical Analysis Project**  
**Date:** December 26, 2025  
**Scope:** Production code only (apps/analysis_api/src/, apps/data_pipeline/, services/)  
**Excluded:** Tests, documentation, frontend code

---

## Executive Summary

Found **13 zombie code items** across the codebase:
- **1 deprecated/backup file** (schemas_backup.py)
- **2 unimplemented module files** (orchestrator stubs)
- **5 TODO/FIXME stub functions** (incomplete implementations)
- **2 unreferenced utility functions** (jit_compile, benchmark)
- **1 orphaned volume indicator** (volume_day3.py - never imported)
- **2 deprecated import patterns** (20+ files importing from models.schemas)

---

## 1. DEPRECATED & BACKUP FILES

### 1.1 `apps/analysis_api/src/gravity_tech/models/schemas_backup.py`

| Property | Value |
|----------|-------|
| **Type** | Deprecated/Backup |
| **Status** | ✅ Safe to Remove |
| **Reason** | Explicit deprecation warning in file header |
| **Details** | Phase 2.1 backward compatibility layer (Nov 7, 2025). Marked for removal in Phase 2.2. Now re-exports from `core.domain.entities`. |
| **Action** | **SAFE TO DELETE** - All imports migrated to `core.domain.entities`. Monitor for any direct imports. |

**Deprecation Notice (lines 1-13):**
```python
"""
⚠️ DEPRECATION WARNING ⚠️
This module is DEPRECATED as of Phase 2.1 (November 7, 2025).
All models have been migrated to: src.core.domain.entities

Please update your imports:
OLD: from models.schemas import Candle, SignalStrength, IndicatorResult
NEW: from gravity_tech.core.domain.entities import Candle, CoreSignalStrength, IndicatorResult

This backward compatibility layer will be removed in Phase 2.2 (Day 3).
"""
```

---

## 2. UNIMPLEMENTED MODULE STUBS

### 2.1 `apps/data_pipeline/src/gravity_pipeline/orchestrator.py`

| Property | Value |
|----------|-------|
| **Type** | Unimplemented Stub (Placeholder) |
| **Status** | ⚠️ Incomplete - Not Ready |
| **Lines** | 461 total (mostly TODOs) |
| **Reason** | Pipeline orchestrator framework with empty implementations |

**Issues Found:**

| Line(s) | TODO | Status |
|---------|------|--------|
| 197 | `# TODO: Implement actual TSE extraction` | Returns empty `candles = []` |
| 246 | `# TODO: Implement actual transformation logic` | Just copies input: `transformed = candles.copy()` |
| 297 | `# TODO: Implement actual validation logic` | Placeholder implementation |
| 349 | `# TODO: Implement actual deduplication` | Not implemented |
| 399 | `# TODO: Implement actual loading logic` | Not implemented |

**Action:** 
- Either **COMPLETE THE IMPLEMENTATION** (tie into TSE data pipeline) OR
- **MOVE TO experiments/** as POC/research code OR
- **REMOVE** if not actively being developed

---

## 3. INCOMPLETE/STUB FUNCTIONS

### 3.1 `apps/analysis_api/src/gravity_tech/ml/ml_tool_recommender.py`

| Property | Value |
|----------|-------|
| **Type** | Incomplete Stub Functions |
| **Status** | ⚠️ Incomplete Implementation |
| **Filename** | 693 lines total |

**Stub Functions:**

#### A. Line 406: `_get_tool_accuracy_in_regime()`
```python
def _get_tool_accuracy_in_regime(self, tool: str) -> float:
    """دریافت دقت تاریخی ابزار در رژیم خاص"""
    # TODO: Load from database
    # این باید از جدول tool_performance_history خوانده شود
    
    # فعلاً مقادیر شبیه‌سازی شده
    base_accuracy = {...}
    return base_accuracy.get(tool, 0.70)
```
**Status:** Returns hardcoded mock data, not database-backed

#### B. Lines 632-645: `train_model()`
```python
def train_model(self, training_data, test_size=0.2):
    """Train Tool Recommender Model"""
    # TODO: Implement full training pipeline
    # این نیاز به داده واقعی تریدها دارد
    
    print("⚠️ Training pipeline not implemented yet")
    print("   Needs historical trade data with tool performance")
    return {"status": "not_implemented", "message": "Training requires..."}
```
**Status:** Not implemented - prints warning

#### C. Line 646: `save_model()`
```python
def save_model(self, filename: str = "tool_recommender.pkl"):
    """ذخیره مدل"""
    # TODO: Implement model saving
    print("💾 Model saving not implemented yet")
```
**Status:** Stub - no actual save logic

#### D. Line 652: `load_model()`
```python
def load_model(self, filename: str = "tool_recommender.pkl"):
    """بارگذاری مدل"""
    # TODO: Implement model loading
    print("📂 Model loading not implemented yet")
```
**Status:** Stub - no actual load logic

**Action:** 
- EITHER **COMPLETE IMPLEMENTATION** with database integration
- OR **MARK AS DEPRECATED** and remove from service
- OR **MOVE TO ml_models/** as experimental trainer

---

### 3.2 `apps/analysis_api/src/gravity_tech/api/sse_handler.py` (Line 371)

```python
# TODO: Implement pattern recognition when available
```
**Location:** Line 371 in `_detect_patterns()` method  
**Status:** Currently returns empty list, awaiting pattern detection implementation  
**Impact:** SSE pattern streaming not functional

---

### 3.3 `apps/analysis_api/src/gravity_tech/api/v1/tools.py` (Line 415)

```python
# TODO: Get from actual tool registry
```
**Location:** Line 415 in `/api/v1/tools/categories` endpoint  
**Status:** Returns hardcoded tool categories instead of from registry  
**Impact:** Tool registry not wired into API

---

### 3.4 `apps/analysis_api/src/gravity_tech/ml/scenario_weight_optimizer.py` (Line 172)

```python
volume_trend=1.0  # TODO: محاسبه از volume dimension
```
**Location:** Line 172  
**Status:** Hardcoded value instead of calculated from volume dimension  
**Impact:** Volume dimension not integrated into scenario weighting

---

## 4. ORPHANED/UNREFERENCED CODE

### 4.1 `apps/analysis_api/src/gravity_tech/core/indicators/volume_day3.py`

| Property | Value |
|----------|-------|
| **Type** | Orphaned Module |
| **Status** | ⚠️ Unreferenced Code |
| **Lines** | 361 total |
| **Functions** | `volume_weighted_macd()`, `ease_of_movement()`, `force_index()` |
| **Imports** | **ZERO** - never imported anywhere |

**Details:**
- Created as Day 3 volume indicators (Nov 9, 2025)
- Advanced implementation: VWMACD, EOM, Force Index
- **NOT exported in `core/indicators/__init__.py`**
- **NOT imported in any production file**
- Functionality duplicated or superseded by `core/indicators/volume.py`

**Code Quality:** ✅ Well-written, documented, type-hinted

**Action:** 
- EITHER **INTEGRATE** into `core/indicators/volume.py` 
- OR **ARCHIVE TO experiments/**
- OR **DELETE** if superseded by `volume.py`

---

### 4.2 Unused Decorator Functions: `apps/analysis_api/src/gravity_tech/utils/performance.py`

| Property | Value |
|----------|-------|
| **Type** | Orphaned Utility Functions |
| **Status** | ⚠️ Unreferenced Code |

**A. `jit_compile()` decorator (Line 10)**
```python
def jit_compile(func):
    """Decorator to JIT compile functions with Numba."""
    compiled_func = jit(nopython=True, cache=True, parallel=True)(func)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return compiled_func(*args, **kwargs)
    return wrapper
```
**Usage Count:** 0 - Never used anywhere

**B. `benchmark()` decorator (Line 20)**
```python
def benchmark(func):
    """Decorator to benchmark function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper
```
**Usage Count:** 0 - Never used anywhere

**Impact:** 
- Takes up module namespace
- Maintenance burden (Numba deprecation, etc.)
- Could be useful for future profiling but currently dormant

**Action:**
- **MOVE TO** `experiments/performance_tools.py` OR
- **DELETE** and recreate if needed later with profiler integration

---

## 5. DEPRECATED IMPORT PATTERNS

### 5.1 `models.schemas` Imports (DEPRECATED)

**Old Pattern (DEPRECATED):**
```python
from gravity_tech.models.schemas import Candle, SignalStrength, IndicatorResult
```

**New Pattern (RECOMMENDED):**
```python
from gravity_tech.core.domain.entities import Candle, CoreSignalStrength, IndicatorResult
```

**Files Still Using Old Pattern (20+ files):**

| File | Import Count | Safe? |
|------|--------------|-------|
| `patterns/divergence.py` | 1 | ⚠️ Yes - re-exports OK |
| `patterns/elliott_wave.py` | 4 | ⚠️ Yes - re-exports OK |
| `patterns/classical.py` | 4 | ⚠️ Yes - re-exports OK |
| `utils/sample_data.py` | 1 | ✅ Test data OK |
| `ml/integrated_multi_horizon_analysis.py` | 3 | ⚠️ Yes - re-exports OK |
| `ml/multi_horizon_cycle_features.py` | 1 | ⚠️ Yes - re-exports OK |
| `ml/multi_horizon_cycle_analysis.py` | 2 | ⚠️ Yes - re-exports OK |
| `ml/multi_horizon_support_resistance_features.py` | 1 | ⚠️ Yes - re-exports OK |
| `ml/train_multi_horizon_support_resistance.py` | 1 | ⚠️ Yes - re-exports OK |
| `ml/train_weights.py` | 2 | ⚠️ Yes - re-exports OK |
| `ml/weight_optimizer.py` | 2 | ⚠️ Yes - re-exports OK |
| `ml/train_volume_dimension_matrix.py` | 1 | ⚠️ Yes - re-exports OK |
| `ml/train_multi_horizon_volatility.py` | 1 | ⚠️ Yes - re-exports OK |
| `ml/models/lstm_model.py` | 3 | ⚠️ Yes - re-exports OK |
| `ml/models/transformer_model.py` | 2 | ⚠️ Yes - re-exports OK |
| `ml/feature_extraction.py` | 1 | ⚠️ Yes - re-exports OK |
| `ml/five_dimensional_decision_matrix.py` | 1 | ⚠️ Yes - re-exports OK |

**Why Safe Now:**
- `models/schemas.py` re-exports from `core.domain.entities`
- Deprecation warning issued at import time
- Full backward compatibility maintained
- Can be removed in Phase 2.2 after migration complete

**Action:**
- **DEPRECATION PHASE:** Current state ✅ OK
- **MIGRATION PHASE:** Update all imports to use `core.domain.entities`
- **REMOVAL PHASE:** Delete `models/schemas.py` and `models/schemas_backup.py`

---

## 6. SUMMARY TABLE

| Item | Type | Location | Status | Safe to Remove | Priority |
|------|------|----------|--------|----------------|----------|
| schemas_backup.py | Deprecated File | models/ | ✅ Deprecated | YES | HIGH |
| orchestrator.py | Unimplemented Stub | data_pipeline/src/ | ⚠️ Incomplete | Needs work | MEDIUM |
| ml_tool_recommender.py | Stub Functions (4) | ml/ | ⚠️ Incomplete | Needs work | LOW |
| sse_handler.py | TODO Stub | api/ | ⚠️ Incomplete | Needs work | MEDIUM |
| tools.py | TODO Stub | api/v1/ | ⚠️ Incomplete | Needs work | LOW |
| scenario_weight_optimizer.py | TODO Hardcoded | ml/ | ⚠️ Incomplete | Needs work | LOW |
| volume_day3.py | Orphaned Module | core/indicators/ | ⚠️ Unused | NEEDS DECISION | MEDIUM |
| performance.py | Unused Decorators (2) | utils/ | ⚠️ Unused | YES | LOW |
| models.schemas | Deprecated Imports | Multiple (20+) | ⚠️ Deprecated | After migration | LOW |

---

## 7. RECOMMENDED CLEANUP ROADMAP

### Phase 1: Immediate Removal (Safe, No Impact)
**Effort:** 5 minutes
```
✅ DELETE: apps/analysis_api/src/gravity_tech/models/schemas_backup.py
✅ DELETE: apps/analysis_api/src/gravity_tech/utils/performance.py (jit_compile, benchmark)
```

### Phase 2: Archive Experimental Code (Safe, Preserves History)
**Effort:** 10 minutes
```
✅ MOVE: apps/analysis_api/src/gravity_tech/core/indicators/volume_day3.py 
   → experiments/archived_volume_indicators.py
   
✅ MOVE: apps/analysis_api/src/gravity_tech/ml/ml_tool_recommender.py (4 stub functions)
   → Consider marking as WIP or moving to experiments/
   
✅ MOVE: apps/data_pipeline/src/gravity_pipeline/orchestrator.py
   → Either complete implementation or move to experiments/pipeline_stub.py
```

### Phase 3: Update Deprecated Imports (Medium Effort, ~1 hour)
**Effort:** 1 hour
```
⚠️ UPDATE: 20+ files to use core.domain.entities instead of models.schemas
  
Use bulk find/replace:
  FROM: from gravity_tech.models.schemas import
  TO:   from gravity_tech.core.domain.entities import
  
ALSO UPDATE: SignalStrength → CoreSignalStrength
```

### Phase 4: Complete TODO Stubs (High Effort, ~2-4 hours)
**Effort:** 2-4 hours

For each TODO stub, decide:
1. **IMPLEMENT** - Tie into real data pipeline
2. **DEPRECATE** - Mark @deprecated and remove from public API
3. **ARCHIVE** - Move to experiments/ for future work

Stubs to address:
- `orchestrator.py` - 5 TODOs (extract, transform, validate, deduplicate, load)
- `ml_tool_recommender.py` - 4 stubs (accuracy, train, save, load)
- `sse_handler.py` - Pattern recognition streaming
- `tools.py` - Tool registry integration

---

## 8. NOTES & OBSERVATIONS

### ✅ Good Practices Found
- Well-documented deprecation warnings
- Backward compatibility layers maintained
- Type hints throughout
- Clear TODO comments marking incomplete work

### ⚠️ Potential Issues
- **Duplicate indicators:** `volume_day3.py` may duplicate `volume.py`
  - Consider merging or clarifying separation
  
- **Import debt:** 20+ files using deprecated pattern
  - Low risk but technical debt
  - Update when doing bulk refactoring
  
- **Async/Real-time:** Several TODOs for streaming/real-time features
  - Consider if these are critical path items

### 📊 Code Quality Metrics
- **Orphaned code:** 1 file (volume_day3.py)
- **Incomplete stubs:** ~10 functions across 4 files
- **Deprecated patterns:** 20+ import usages (1 backward-compat layer)
- **Unused utilities:** 2 decorator functions
- **Documentation:** ✅ Excellent (most issues clearly marked)

---

## 9. APPENDIX: DETAILED FUNCTION LOCATIONS

### All TODO Markers Found
```
ml_tool_recommender.py:406      # TODO: Load from database
ml_tool_recommender.py:632      # TODO: Implement full training pipeline
ml_tool_recommender.py:646      # TODO: Implement model saving
ml_tool_recommender.py:652      # TODO: Implement model loading
scenario_weight_optimizer.py:172 # TODO: محاسبه از volume dimension
sse_handler.py:371              # TODO: Implement pattern recognition when available
api/v1/tools.py:415             # TODO: Get from actual tool registry
orchestrator.py:197             # TODO: Implement actual TSE extraction
orchestrator.py:246             # TODO: Implement actual transformation logic
orchestrator.py:297             # TODO: Implement actual validation logic
orchestrator.py:349             # TODO: Implement actual deduplication
orchestrator.py:399             # TODO: Implement actual loading logic
```

---

**Report Generated:** December 26, 2025  
**Status:** COMPLETE ✅  
**Reviewed By:** GitHub Copilot (Code Analysis)
