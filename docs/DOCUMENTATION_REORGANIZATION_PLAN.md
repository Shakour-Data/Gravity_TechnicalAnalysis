# 📋 Documentation Reorganization Plan

**Date:** November 15, 2025  
**Status:** IN PROGRESS  
**Goal:** Reduce 49 markdown files to ~20 organized files

---

## 🎯 Current Problems

- **49 markdown files** scattered across project
- Multiple CHANGELOGs (3 different ones)
- Duplicate guides (TREND_ANALYSIS_GUIDE vs TREND_ANALYSIS_SUMMARY)
- No clear organization
- Persian + English mixed
- Outdated INDEX.md

---

## 📊 File Inventory & Action Plan

### Root Level Files (Action Required)

| File | Lines | Action | Reason |
|------|-------|--------|---------|
| README.md | - | **KEEP** | Main project entry point |
| CHANGELOG.md | - | **CONSOLIDATE** | Merge all CHANGELOGs into one |
| CONTRIBUTING.md | - | **KEEP** | Important for contributors |
| LICENSE | - | **KEEP** | Required |
| STRUCTURE.md | - | **MOVE** | Move to docs/architecture/ |
| VERSION | - | **KEEP** | Version tracking |
| DOCS_INDEX.md | - | **UPDATE & KEEP** | Master documentation index |

### Duplicate/Redundant Files (DELETE or MERGE)

| File | Duplicate Of | Action |
|------|--------------|---------|
| CHANGELOG_ACCURACY.md | CHANGELOG.md | **MERGE INTO CHANGELOG.md** |
| CHANGELOG_CLASSICAL_PATTERNS.md | CHANGELOG.md | **MERGE INTO CHANGELOG.md** |
| docs/guides/TREND_ANALYSIS_SUMMARY.md | TREND_ANALYSIS_GUIDE.md | **DELETE** (redundant) |
| RELEASE_NOTES_v1.0.0.md | docs/releases/ | **MOVE** to docs/releases/ |
| RELEASE_NOTES_v1.1.0.md | docs/releases/ | **MOVE** to docs/releases/ |
| RELEASE_NOTES_v1.2.0.md | docs/releases/ | **ALREADY IN** docs/releases/ |

### Day Reports (ARCHIVE)

These are historical completion reports. Move to `docs/archive/reports/`:

- DAY_1_COMPLETION_REPORT.md → **ARCHIVE**
- DAY_1_COMPLETION_REPORT_v1.1.0.md → **ARCHIVE**
- DAY_2_COMPLETION_REPORT.md → **ARCHIVE**
- DAY_2_COMPLETION_REPORT_v1.1.0.md → **ARCHIVE**
- DAY_3_COMPLETION_REPORT_v1.1.0.md → **ARCHIVE**
- DAY_4_COMPLETION_REPORT_v1.1.0.md → **ARCHIVE**
- DAY_5_COMPLETION_REPORT_v1.1.0.md → **ARCHIVE**
- DAY_6_COMPLETION_REPORT_v1.1.0.md → **ARCHIVE**
- DAY_7_COMPLETION_REPORT_v1.1.0.md → **ARCHIVE**
- CODE_REVIEW_v1.1.0_Day1.md → **ARCHIVE**
- CODE_REVIEW_v1.1.0_Day2.md → **ARCHIVE**
- CODE_REVIEW_v1.1.0_Day3.md → **ARCHIVE**
- CLEANUP_REPORT.md → **ARCHIVE**
- RELEASE_PLAN_v1.1.0.md → **ARCHIVE**
- RELEASE_v1.0.0.txt → **ARCHIVE**
- RELEASE_v1.1.0_INSTRUCTIONS.md → **ARCHIVE**
- RELEASE_SUMMARY_v1.0.0_FA.md → **ARCHIVE**

### docs/guides/ (12 files → 8 files)

| File | Action | New Location |
|------|--------|--------------|
| ACCURACY_GUIDE.md | **KEEP** | - |
| CYCLE_ANALYSIS_GUIDE.md | **KEEP** | - |
| DOW_THEORY.md | **KEEP** | - |
| FIVE_DIMENSIONAL_DECISION_GUIDE.md | **KEEP** | - |
| HISTORICAL_SYSTEM_GUIDE.md | **KEEP** | - |
| ML_FEATURES_GUIDE.md | **KEEP** | - |
| PERFORMANCE_OPTIMIZATION.md | **MOVE** | docs/operations/ |
| PROJECT_SUMMARY.md | **DELETE** | Redundant with README |
| SCORING_SYSTEM_GUIDE.md | **KEEP** | - |
| SUPPORT_RESISTANCE_GUIDE.md | **KEEP** | - |
| TREND_ANALYSIS_GUIDE.md | **KEEP** | - |
| TREND_ANALYSIS_SUMMARY.md | **DELETE** | Duplicate |
| VOLATILITY_ANALYSIS_GUIDE.md | **KEEP** | - |
| VOLUME_MATRIX_GUIDE.md | **KEEP** | - |

**Result:** 12 → **9 files** in docs/guides/

### docs/team/ (6 files → 7 files)

| File | Action |
|------|--------|
| IMPROVEMENT_TASKS.md | **KEEP** |
| README.md | **KEEP** |
| SARAH_QA_PROGRESS_DAY1.md | **ARCHIVE** (historical) |
| TEAM.md | **KEEP** |
| TEAM_PROMPTS.md | **KEEP** |
| PROJECT_ISSUES_REPORT.md | **KEEP** (NEW - just created!) |

**Result:** 6 → **5 files** (after archiving SARAH_QA_PROGRESS_DAY1.md)

### docs/architecture/ (7 files → 7 files)

All good - KEEP all:
- DATA_SERVICE_INTEGRATION.md
- FILE_IDENTITY_SYSTEM.md
- MICROSERVICES_ARCHITECTURE.md
- ML_WEIGHTS.md
- MOMENTUM_ANALYSIS_PLAN.md
- SCENARIO_ANALYSIS_DESIGN.md
- SIGNAL_CALCULATION.md
- SYSTEM_ARCHITECTURE_DIAGRAMS.md

### docs/operations/ (3 files → 4 files)

- DEPLOYMENT_GUIDE.md - **KEEP**
- RUNBOOK.md - **KEEP**
- MULTI_REGION_SETUP.md - **KEEP** (if exists)
- PERFORMANCE_OPTIMIZATION.md - **ADD** (move from guides/)

### docs/releases/ (3 files → 3 files)

- RELEASE_NOTES_v1.0.0.md - **KEEP**
- RELEASE_NOTES_v1.1.0.md - **KEEP**
- RELEASE_NOTES_v1.2.0.md - **KEEP**

### Root docs/ Files (Reorganize)

| File | Action | New Location |
|------|--------|--------------|
| API_SCORE_RANGE_CHANGE.md | **MOVE** | docs/api/ |
| CHANGELOG.md | **CONSOLIDATE** | Root (merge 3 CHANGELOGs) |
| CHANGELOG_ACCURACY.md | **MERGE** | Into CHANGELOG.md |
| CHANGELOG_CLASSICAL_PATTERNS.md | **MERGE** | Into CHANGELOG.md |
| CONTRIBUTING.md | **KEEP** | Root |
| DATABASE_SETUP.md | **MOVE** | docs/operations/ |
| INDEX.md | **UPDATE & KEEP** | docs/ |
| ML_LEARNING_SYSTEM.md | **MOVE** | docs/architecture/ |
| PROJECT_STRUCTURE.md | **MERGE** | Into STRUCTURE.md or README |
| QUICKSTART.md | **KEEP** | docs/ |
| TREND_VS_MOMENTUM.md | **MOVE** | docs/guides/ |

---

## 🎯 Target Structure (20-25 files)

```
Gravity_TechAnalysis/
├── README.md (main entry)
├── CHANGELOG.md (consolidated from 3)
├── CONTRIBUTING.md
├── LICENSE
├── VERSION
├── STRUCTURE.md
│
├── docs/
│   ├── INDEX.md (master index)
│   ├── QUICKSTART.md
│   │
│   ├── guides/ (9 files)
│   │   ├── README.md
│   │   ├── ACCURACY_GUIDE.md
│   │   ├── CYCLE_ANALYSIS_GUIDE.md
│   │   ├── DOW_THEORY.md
│   │   ├── FIVE_DIMENSIONAL_DECISION_GUIDE.md
│   │   ├── HISTORICAL_SYSTEM_GUIDE.md
│   │   ├── ML_FEATURES_GUIDE.md
│   │   ├── SCORING_SYSTEM_GUIDE.md
│   │   ├── SUPPORT_RESISTANCE_GUIDE.md
│   │   ├── TREND_ANALYSIS_GUIDE.md
│   │   ├── TREND_VS_MOMENTUM.md (moved)
│   │   ├── VOLATILITY_ANALYSIS_GUIDE.md
│   │   └── VOLUME_MATRIX_GUIDE.md
│   │
│   ├── api/ (2 files)
│   │   ├── README.md (create)
│   │   └── API_SCORE_RANGE_CHANGE.md (moved)
│   │
│   ├── architecture/ (9 files)
│   │   ├── README.md
│   │   ├── DATA_SERVICE_INTEGRATION.md
│   │   ├── FILE_IDENTITY_SYSTEM.md
│   │   ├── MICROSERVICES_ARCHITECTURE.md
│   │   ├── ML_LEARNING_SYSTEM.md (moved)
│   │   ├── ML_WEIGHTS.md
│   │   ├── MOMENTUM_ANALYSIS_PLAN.md
│   │   ├── SCENARIO_ANALYSIS_DESIGN.md
│   │   ├── SIGNAL_CALCULATION.md
│   │   └── SYSTEM_ARCHITECTURE_DIAGRAMS.md
│   │
│   ├── operations/ (4 files)
│   │   ├── README.md (create)
│   │   ├── DATABASE_SETUP.md (moved)
│   │   ├── DEPLOYMENT_GUIDE.md
│   │   ├── PERFORMANCE_OPTIMIZATION.md (moved)
│   │   └── RUNBOOK.md
│   │
│   ├── releases/ (3 files)
│   │   ├── RELEASE_NOTES_v1.0.0.md
│   │   ├── RELEASE_NOTES_v1.1.0.md
│   │   └── RELEASE_NOTES_v1.2.0.md
│   │
│   ├── team/ (5 files)
│   │   ├── README.md
│   │   ├── IMPROVEMENT_TASKS.md
│   │   ├── PROJECT_ISSUES_REPORT.md (NEW!)
│   │   ├── TEAM.md
│   │   └── TEAM_PROMPTS.md
│   │
│   └── archive/
│       ├── README.md
│       └── reports/
│           └── v1.0.0/
│               ├── MICROSERVICE_EVALUATION.md (exists)
│               ├── DAY_1_COMPLETION_REPORT.md
│               ├── DAY_2_COMPLETION_REPORT.md
│               ├── ...
│               ├── CODE_REVIEW_v1.1.0_Day1.md
│               ├── ...
│               └── RELEASE_PLAN_v1.1.0.md
```

---

## 📈 Results

### Before:
- **49 markdown files**
- Scattered, duplicated, confusing
- Multiple CHANGELOGs
- No clear structure

### After:
- **~25 markdown files** (organized)
- Clear category structure
- Single consolidated CHANGELOG
- Historical reports archived
- Easy to navigate

---

## ✅ Action Items

### Phase 1: Archive Historical Reports (DONE)
- [x] Identify all day reports and old release docs
- [ ] Move to docs/archive/reports/v1.0.0/
- [ ] Update references

### Phase 2: Consolidate Duplicates
- [ ] Merge 3 CHANGELOGs into one
- [ ] Delete TREND_ANALYSIS_SUMMARY.md
- [ ] Delete PROJECT_SUMMARY.md

### Phase 3: Reorganize by Category
- [ ] Move API_SCORE_RANGE_CHANGE to docs/api/
- [ ] Move DATABASE_SETUP to docs/operations/
- [ ] Move PERFORMANCE_OPTIMIZATION to docs/operations/
- [ ] Move ML_LEARNING_SYSTEM to docs/architecture/
- [ ] Move TREND_VS_MOMENTUM to docs/guides/

### Phase 4: Create Missing READMEs
- [ ] docs/api/README.md
- [ ] docs/operations/README.md

### Phase 5: Update Master Index
- [ ] Update docs/INDEX.md with new structure
- [ ] Add descriptions for each section
- [ ] Link to all documents

---

**Status:** Ready for execution  
**Estimated Time:** 2-3 hours  
**Owner:** Documentation Lead (Dr. Hans Mueller per team structure)
