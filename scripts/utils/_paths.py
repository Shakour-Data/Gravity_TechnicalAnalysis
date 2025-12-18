from __future__ import annotations

from pathlib import Path

# __file__ = .../scripts/utils/_paths.py -> repo root is two levels up
REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_APP = REPO_ROOT / "apps" / "analysis_api"
ANALYSIS_SRC = ANALYSIS_APP / "src"
ANALYSIS_TESTS = ANALYSIS_APP / "tests"
ANALYSIS_MODELS = ANALYSIS_APP / "ml_models"


def extend_sys_path() -> None:
    """Ensure repo + analysis src are importable when scripts run standalone."""
    import sys

    for candidate in (REPO_ROOT, ANALYSIS_SRC):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
