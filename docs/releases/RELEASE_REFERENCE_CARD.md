# Release v1.3.3 - Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│          RELEASE v1.3.3 - QUICK REFERENCE CARD             │
│                    December 5, 2025                         │
└─────────────────────────────────────────────────────────────┘

📌 CRITICAL PATH

Phase 1: Coverage Test (30-45 min) ⏳ START HERE
├─ Command: pytest tests/ --cov=src --cov-report=html
├─ Target: 95%+
└─ Decision: ✅ Pass → Phase 2 | ❌ Fail → Add tests

Phase 2: Version Sync (20 min) ⏳ BLOCKED
├─ README.md: 1.2.0 → 1.3.3
├─ pyproject.toml: Verify 1.3.3
└─ configs/VERSION: Verify 1.3.3

Phase 3: Documentation (30 min) ⏳ BLOCKED
├─ Update CHANGELOG.md
├─ Create RELEASE_NOTES_v1.3.3.md
└─ Create RELEASE_SUMMARY_v1.3.3_FA.md

Phase 4: Deployment (45 min) ⏳ BLOCKED
├─ Deploy to Kubernetes
├─ Health check: /health → 200
├─ Version check: /version → 1.3.3
└─ Smoke tests: ✅ Pass

Phase 5: Git Release (15 min) ⏳ BLOCKED
├─ git tag -a v1.3.3 -m "Release v1.3.3"
├─ git push origin v1.3.3
└─ GitHub Release UI

Phase 6: Notify Team (5 min) ⏳ BLOCKED
└─ Slack/Teams announcement

───────────────────────────────────────────────────────────────

⚠️  BLOCKING CRITERIA (Must be 100% met)

✅ Test Coverage ≥ 95%
✅ All Tests Pass (1200+)
✅ Versions Synchronized (1.3.3 everywhere)
✅ CHANGELOG Updated
✅ Health Endpoints Respond
✅ Documentation Complete

───────────────────────────────────────────────────────────────

📊 VERSION TRACKING

File                    Current    Target     Status
────────────────────────────────────────────────────────────
pyproject.toml          1.3.2      1.3.3      ⏳ Verify
README.md               1.2.0      1.3.3      ❌ UPDATE
configs/VERSION         1.3.2      1.3.3      ⏳ Verify

───────────────────────────────────────────────────────────────

📋 DOCUMENTS CREATED

✅ RELEASE_PROCESS_v1.3.3.md
   → Full 6-phase detailed checklist with criteria

✅ RELEASE_STEPS_FA.md
   → Persian step-by-step guide (گام‌های ریلیز)

✅ RELEASE_QUICK_START.md
   → Fast reference with commands and timeline

✅ RELEASE_SUMMARY.md
   → Executive overview and next steps

───────────────────────────────────────────────────────────────

🎯 START COMMAND (Copy & Paste Ready)

cd e:\Shakour\GravityProjects\Gravity_TechnicalAnalysis
python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

───────────────────────────────────────────────────────────────

⏱️  TIMELINE

Phase 1 → 30-45 min
Phase 2 → +20 min  (if Phase 1 ✅)
Phase 3 → +30 min  (if Phase 2 ✅)
Phase 4 → +45 min  (if Phase 3 ✅)
Phase 5 → +15 min  (if Phase 4 ✅)
Phase 6 → +5 min   (if Phase 5 ✅)
────────────────────
TOTAL  → ~2.5 hours

───────────────────────────────────────────────────────────────

🚀 STATUS: READY FOR EXECUTION ✅

All planning complete.
Documentation prepared.
Next: Execute Phase 1 ⏳

───────────────────────────────────────────────────────────────
```

---

## 📱 Mobile Quick Reference

**Phase 1 (DO FIRST):** Coverage test - 30-45 min
```
pytest tests/ --cov=src --cov-report=html
→ Need 95%+ to proceed
```

**Phase 2:** Update versions - 20 min
```
README.md: 1.2.0→1.3.3
Verify pyproject.toml & configs/VERSION = 1.3.3
```

**Phase 3:** Update docs - 30 min
```
Update CHANGELOG [Unreleased]→[1.3.3]
Create RELEASE_NOTES_v1.3.3.md
Create RELEASE_SUMMARY_v1.3.3_FA.md
```

**Phase 4:** Deploy - 45 min
```
kubectl apply -f deployment/kubernetes/overlays/prod/
Verify: /health → 200, /version → 1.3.3
```

**Phase 5:** Git release - 15 min
```
git tag -a v1.3.3 -m "Release v1.3.3"
git push origin v1.3.3
GitHub Release UI
```

**Phase 6:** Notify - 5 min
```
Slack/Teams: Release v1.3.3 live
```

---

## 🔑 Key Decision Points

```
Phase 1 Coverage Result?
├─ ✅ ≥95%  → PROCEED to Phase 2
└─ ❌ <95%  → STOP and ADD TESTS

All Tests Pass?
├─ ✅ YES   → PROCEED to Phase 4
└─ ❌ NO    → STOP and FIX FAILURES

Health Checks OK?
├─ ✅ YES   → PROCEED to Phase 5
└─ ❌ NO    → STOP and DEBUG

All Criteria Met?
├─ ✅ YES   → CREATE GITHUB RELEASE
└─ ❌ NO    → STOP and VERIFY
```

---

## 📚 Full Documentation

| Document | Purpose | Read Time |
|----------|---------|-----------|
| RELEASE_QUICK_START.md | Fast action steps | 5 min |
| RELEASE_PROCESS_v1.3.3.md | Complete checklist | 15 min |
| RELEASE_STEPS_FA.md | Persian guide | 15 min |
| RELEASE_SUMMARY.md | Executive overview | 10 min |

---

**Ready?** → Start Phase 1: `pytest tests/ --cov=src --cov-report=html`

**Questions?** → See RELEASE_QUICK_START.md or RELEASE_SUMMARY.md
