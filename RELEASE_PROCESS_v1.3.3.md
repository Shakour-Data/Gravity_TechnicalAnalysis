# Release Process: v1.3.3

**Current Status**: Pre-Release Planning
**Version**: 1.3.3 (Next Release)
**Previous Version**: 1.3.2
**Target**: December 5, 2025

---

## 📋 Release Checklist

### Phase 1: Quality Assurance (Critical - 95%+ Test Coverage Required)

- [ ] **1.1 Test Coverage Assessment**
  - Target: 95%+ code coverage
  - Current README: 11.71% (needs update)
  - Action: Run full test suite with coverage report
  - Command: `python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing`
  - Acceptance: Coverage ≥ 95%

- [ ] **1.2 Run Unit Tests**
  - Command: `python -m pytest tests/unit/ -v --tb=short`
  - Verify: All tests passing
  - Reference: docs/releases/RELEASE_NOTES_v1.3.0.md (lines 666-691)

- [ ] **1.3 Run Integration Tests**
  - Command: `python -m pytest tests/integration/ -v --tb=short`
  - Verify: Service integration working
  - Check: Database migrations, API endpoints

- [ ] **1.4 Run ML Tests**
  - Command: `python -m pytest tests/ml/ -v --tb=short`
  - Verify: Model training/inference stable
  - Check: No flaky test failures (run 3x for stability)

- [ ] **1.5 Smoke Tests**
  - Health Check: `GET /health`
  - Version Check: `GET /version` (should return 1.3.3)
  - Basic Indicator: Test SMA, RSI, MACD endpoints

---

### Phase 2: Version Synchronization

**Current State:**
- `pyproject.toml`: 1.3.2 ✅
- `README.md`: 1.2.0 ❌ (OUTDATED)
- `configs/VERSION`: 1.3.2 ✅

- [ ] **2.1 Update README.md**
  - Line 3: Update badge version 1.2.0 → 1.3.3
  - Line 3: Update GitHub release link
  - Line 7: Update ML accuracy badge (if changed)
  - Line 9: Update test coverage badge

- [ ] **2.2 Update pyproject.toml**
  - Line 7: Ensure version = "1.3.3"
  - Verify: No version mismatches

- [ ] **2.3 Update configs/VERSION**
  - Content: 1.3.3
  - Verify: Matches pyproject.toml

- [ ] **2.4 Verification**
  - Run: `grep -r "1.3.2" docs/releases/ README.md` (should find nothing or only old versions)
  - Confirm: All version sources aligned

---

### Phase 3: Documentation Updates

**Current State:**
- CHANGELOG.md: Has [Unreleased] section (line 562)
- Release Notes: v1.3.0, v1.3.1 exist
- 10-Day Checklist: "Create release notes" is incomplete (line 126)

- [ ] **3.1 Update CHANGELOG.md**
  - Move [Unreleased] section → [1.3.3] - 2025-12-05
  - Update with all changes since v1.3.2
  - Categorize: Added, Changed, Fixed, Removed
  - Reference: Code quality, type hints, linting fixes

- [ ] **3.2 Create Release Notes v1.3.3**
  - Location: `docs/releases/RELEASE_NOTES_v1.3.3.md`
  - Template: Use v1.3.0/v1.3.1 as reference
  - Include:
    - Overview & highlights
    - New features (if any)
    - Bug fixes (code quality improvements)
    - Breaking changes (none expected)
    - Migration guide (if needed)
    - Testing & deployment steps
  - Structure: Follow docs/releases/RELEASE_NOTES_v1.3.0.md (lines 661-707)

- [ ] **3.3 Create Release Summary (Persian)**
  - Location: `docs/releases/RELEASE_SUMMARY_v1.3.3_FA.md`
  - Content: Persian summary for team
  - Include: Key changes, deployment info, team notifications

- [ ] **3.4 Update Quick Start 10-Day Checklist**
  - File: QUICK_START_10_DAYS.md (line 126)
  - Update: "Create release notes" - mark complete
  - Add: Link to Release Notes v1.3.3

---

### Phase 4: Deployment & Health Verification

**Reference**: docs/releases/RELEASE_NOTES_v1.3.0.md (lines 661-691)

- [ ] **4.1 Pre-Deployment**
  - Pull latest code: `git pull origin main`
  - Verify: No uncommitted changes
  - Check: All tests passing (from Phase 1)

- [ ] **4.2 Apply Kubernetes Manifests**
  - Location: `deployment/kubernetes/base/`
  - Command: `kubectl apply -f deployment/kubernetes/overlays/prod/`
  - Verify: Pod status RUNNING

- [ ] **4.3 Health Check Endpoints**
  - URL: `http://service-endpoint/health`
  - Expected: HTTP 200, status: "healthy"
  - URL: `http://service-endpoint/version`
  - Expected: {"version": "1.3.3", ...}

- [ ] **4.4 Smoke Tests (Functional)**
  - SMA endpoint: Test with sample data
  - RSI endpoint: Verify indicator calculation
  - MACD endpoint: Confirm signal accuracy
  - ML inference: Test weight learner predictions

- [ ] **4.5 Log Verification**
  - No ERROR or CRITICAL logs
  - Performance: Response times < 100ms
  - Coverage: All endpoints responding

---

### Phase 5: Git & GitHub Release

- [ ] **5.1 Create Git Tag**
  - Command: `git tag -a v1.3.3 -m "Release v1.3.3: Code quality improvements"`
  - Verify: `git tag --list | grep v1.3.3`

- [ ] **5.2 Push Tag**
  - Command: `git push origin v1.3.3`
  - Verify: Tag appears on GitHub

- [ ] **5.3 Create GitHub Release**
  - Title: "v1.3.3 - Code Quality & Type Safety Improvements"
  - Description: Use RELEASE_NOTES_v1.3.3.md content
  - Attach: Release notes file (optional)
  - Mark as: Latest Release (if applicable)
  - Pre-release: No (mark as stable)

- [ ] **5.4 Verify Release**
  - GitHub: Release appears on releases page
  - Tag: Visible on main repository
  - CI/CD: Any automated workflows completed

---

### Phase 6: Team Notification

- [ ] **6.1 Internal Notification**
  - Slack/Teams: Post release summary
  - Include: Version, key changes, deployment status
  - Tag: DevOps, ML Team, QA Team

- [ ] **6.2 Documentation Links**
  - Share: Release Notes URL
  - Share: Changelog URL
  - Share: Deployment guide

- [ ] **6.3 Post-Release Monitoring**
  - Duration: 24-48 hours
  - Monitor: Error rates, latency, user feedback
  - Rollback plan: If critical issues found

---

## 📊 Coverage Targets

| Category | Target | Current | Status |
|----------|--------|---------|--------|
| Overall Coverage | 95% | 11.71% | 🔴 CRITICAL |
| Unit Tests | 100% | - | ⏳ TBD |
| Integration Tests | 95% | - | ⏳ TBD |
| ML Tests | 90% | - | ⏳ TBD |

**Priority Actions:**
1. Run full test suite with coverage
2. Identify coverage gaps (reference: CRITICAL_PRIORITY_ANALYSIS.md:15-67,346)
3. Add missing tests before release
4. Update badge in README (currently 11.71%)

---

## 🔄 Version Synchronization Summary

**Files to Update for v1.3.3:**

```
pyproject.toml          version = "1.3.3"           ✅ (verify)
README.md               badge: 1.2.0 → 1.3.3        ❌ REQUIRED
configs/VERSION         1.3.3                       ✅ (verify)
docs/changelog/CHANGELOG.md  [Unreleased] → [1.3.3]  ❌ REQUIRED
docs/releases/RELEASE_NOTES_v1.3.3.md  (NEW FILE)    ❌ REQUIRED
docs/releases/RELEASE_SUMMARY_v1.3.3_FA.md (NEW)    ❌ REQUIRED
QUICK_START_10_DAYS.md  Mark "release notes" done    ❌ REQUIRED
```

---

## 📝 Release Notes Structure (Template)

Use docs/releases/RELEASE_NOTES_v1.3.0.md as reference:

```markdown
# Release Notes v1.3.3

## Overview
- Release Date: 2025-12-05
- Focus: Code Quality & Type Safety

## Key Changes
- Fixed 143 files (whitespace, imports, type hints)
- Modernized to Python 3.9+ type system
- Added matplotlib lazy-loading

## Testing & QA
- Test Coverage: [X]%
- All tests passing: ✅
- Smoke tests: ✅

## Deployment
- Kubernetes: Update overlays
- Migration: None required
- Health Checks: All passing

## Links
- Repository: https://github.com/Shakour-Data/Gravity_TechnicalAnalysis
- Release Tag: v1.3.3
- Release Notes: docs/releases/RELEASE_NOTES_v1.3.3.md
```

---

## ⚠️ Critical Blockers

1. **Test Coverage < 95%**: DO NOT release until coverage meets target
2. **Version Mismatch**: All version sources must align before tag
3. **CHANGELOG Not Updated**: [Unreleased] must migrate to [1.3.3]
4. **Health Checks Fail**: All endpoints must return 200 OK

---

## 🚀 Next Steps

1. **Immediately**: Run Phase 1 quality assurance
   - Execute: `python -m pytest tests/ -v --cov=src --cov-report=html`
   - Review: htmlcov/index.html for coverage gaps
   - Target: Reach 95%+ before proceeding

2. **Then**: Execute Phase 2-3 synchronization
   - Update version numbers
   - Update documentation
   - Create release notes

3. **Finally**: Execute Phase 4-6 deployment & release
   - Deploy to production
   - Verify health
   - Create tag & GitHub release

---

**Release Manager**: @Shakour-Data
**Approval Required**: QA Lead, DevOps Lead
**Expected Duration**: 2-4 hours

---

Last Updated: 2025-12-05
