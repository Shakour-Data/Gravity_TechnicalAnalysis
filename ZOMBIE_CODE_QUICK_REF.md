# 🧟 Zombie Code - Quick Reference

## Summary Table

| # | Item | Type | Location | Status | Safe? | Action |
|---|------|------|----------|--------|-------|--------|
| 1 | `schemas_backup.py` | Deprecated File | `models/` | ✅ Deprecated | YES | DELETE |
| 2 | `volume_day3.py` | Orphaned Module | `core/indicators/` | ⚠️ Unused | MAYBE | Archive/Merge |
| 3 | `jit_compile()` | Unused Decorator | `utils/performance.py` | ❌ Dead | YES | DELETE |
| 4 | `benchmark()` | Unused Decorator | `utils/performance.py` | ❌ Dead | YES | DELETE |
| 5 | `orchestrator.py` | Stub Module (5 TODOs) | `data_pipeline/src/` | ⚠️ Incomplete | NO | Complete/Archive |
| 6 | `_get_tool_accuracy_in_regime()` | Stub Function | `ml/ml_tool_recommender.py:406` | ⚠️ Mock Data | NO | Implement DB |
| 7 | `train_model()` | Stub Function | `ml/ml_tool_recommender.py:632` | ❌ Not Impl | NO | Implement |
| 8 | `save_model()` | Stub Function | `ml/ml_tool_recommender.py:646` | ❌ Not Impl | NO | Implement |
| 9 | `load_model()` | Stub Function | `ml/ml_tool_recommender.py:652` | ❌ Not Impl | NO | Implement |
| 10 | Pattern Recognition | TODO | `api/sse_handler.py:371` | ⚠️ Empty | NO | Implement |
| 11 | Tool Registry | TODO | `api/v1/tools.py:415` | ⚠️ Hardcoded | NO | Wire Registry |
| 12 | Volume Dimension | TODO | `ml/scenario_weight_optimizer.py:172` | ⚠️ Hardcoded | NO | Calculate |
| 13 | `models.schemas` imports | Deprecated Imports | 20+ files | ⚠️ Deprecated | YES | Migrate (Phase 3) |

---

## By Category

### 🗑️ Safe to Delete Immediately (2 items)
1. `apps/analysis_api/src/gravity_tech/models/schemas_backup.py` - Deprecated backward-compat layer
2. `apps/analysis_api/src/gravity_tech/utils/performance.py` - Unused decorators (jit_compile, benchmark)

**Effort:** 5 minutes | **Risk:** None | **Impact:** None

---

### 📦 Needs Archive Decision (1 item)
1. `apps/analysis_api/src/gravity_tech/core/indicators/volume_day3.py` - Orphaned volume module (361 lines)
   - Never imported, 3 advanced indicators
   - Decision: Merge into volume.py OR move to experiments/
   
**Effort:** 20 minutes | **Risk:** Low (unused code) | **Impact:** Cleanup

---

### ⚠️ Incomplete Stubs Needing Work (10 items)

**orchestrator.py (5 TODOs)** - 461-line placeholder
- Line 197: Extract stage (returns empty)
- Line 246: Transform stage (just copies)
- Line 297: Validate stage (stub)
- Line 349: Deduplicate stage (stub)
- Line 399: Load stage (stub)

**ml_tool_recommender.py (4 TODOs)** - 693 lines
- Line 406: `_get_tool_accuracy_in_regime()` - hardcoded values, needs DB
- Line 632: `train_model()` - prints "not implemented" 
- Line 646: `save_model()` - prints "not implemented"
- Line 652: `load_model()` - prints "not implemented"

**sse_handler.py (1 TODO)** - Real-time streaming
- Line 371: `_detect_patterns()` - returns empty, needs pattern detection

**tools.py (1 TODO)** - API endpoint
- Line 415: `/tools/categories` - hardcoded instead of from registry

**scenario_weight_optimizer.py (1 TODO)** - ML weights
- Line 172: `volume_trend=1.0` - hardcoded instead of calculated

**Effort:** 2-4 hours | **Risk:** Medium (active code) | **Impact:** Feature completeness

---

### 📝 Deprecated Imports (20+ files)

**Old Pattern:** `from gravity_tech.models.schemas import ...`  
**New Pattern:** `from gravity_tech.core.domain.entities import ...`

**Also Note:** `SignalStrength` → `CoreSignalStrength`

**Current Status:** ✅ Safe (backward-compat layer exists)  
**Action Plan:** Migrate in Phase 3 bulk refactoring  
**Effort:** 1 hour | **Risk:** None (compat maintained) | **Impact:** Technical debt

---

## Priority Matrix

```
HIGH PRIORITY (Remove Now)
├─ schemas_backup.py         ✅ 5 min  → DELETE
└─ performance.py decorators ✅ 5 min  → DELETE
                             Total: 10 minutes

MEDIUM PRIORITY (Week 1)
├─ volume_day3.py            ⚠️ 20 min → Archive/Merge
└─ orchestrator.py TODOs     ⚠️ 1-2 hr → Complete/Archive
                             Total: 1.5 hours

LOW PRIORITY (Phase 3)
├─ models.schemas imports    ⚠️ 1 hr   → Migrate
├─ ml_tool_recommender stubs ⚠️ 2 hr   → Implement/Remove
└─ API/ML TODOs              ⚠️ 1 hr   → Implement/Remove
                             Total: 4 hours
```

---

## Cleanup Checklist

### Immediate Actions (DO NOW)
- [ ] Delete `models/schemas_backup.py`
- [ ] Delete `utils/performance.py` (or keep for future profiling)
- [ ] Document decision on `volume_day3.py` (merge or archive)

### Week 1 Actions
- [ ] Decide: Complete `orchestrator.py` or move to experiments/
- [ ] Review if `ml_tool_recommender` stubs are active work

### Phase 3 Refactoring
- [ ] Migrate 20+ files from `models.schemas` imports
- [ ] Complete or deprecate `ml_tool_recommender` methods
- [ ] Implement pattern recognition in `sse_handler.py`
- [ ] Wire tool registry in `tools.py` endpoint

---

**Last Updated:** December 26, 2025  
**Audit Method:** Comprehensive grep/semantic search + list_code_usages  
**Files Scanned:** 162 files (analysis_api) + 24 files (data_pipeline) + 43 files (services)
