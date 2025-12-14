from __future__ import annotations

from pathlib import Path

# Resolve key directories relative to the analysis app layout.
_HERE = Path(__file__).resolve()
ANALYSIS_DIR = _HERE.parents[3]  # apps/analysis_api
REPO_ROOT = ANALYSIS_DIR.parent.parent
ML_MODELS_DIR = ANALYSIS_DIR / "ml_models"
DATA_DIR = REPO_ROOT / "data"


__all__ = [
    "ANALYSIS_DIR",
    "REPO_ROOT",
    "ML_MODELS_DIR",
    "DATA_DIR",
]
