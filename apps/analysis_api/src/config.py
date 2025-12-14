import os
from pathlib import Path

from gravity_tech.config.paths import ANALYSIS_DIR, REPO_ROOT

BASE_DIR = ANALYSIS_DIR  # backward-compat alias

# External TSE Database (Input Source)
# Prefer Postgres when available via env; otherwise fall back to SQLite paths.
TSE_DB_FILE = None
_TSE_DB_DSN = os.getenv("TSE_DATABASE_URL") or os.getenv("TSE_DB_URL") or os.getenv("TSE_POSTGRES_URL")
if _TSE_DB_DSN:
    TSE_DB_FILE = _TSE_DB_DSN  # DSN string for Postgres
else:
    # Try multiple possible locations for portability across development machines
    _TSE_DB_CANDIDATES = [
        # Preferred: local ingestion output inside the repo
        REPO_ROOT / 'services' / 'data_ingestion' / 'data' / 'tse_data.db',
        # Legacy Windows path (original)
        r'E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db',
        # Legacy relative path (old sibling checkout)
        REPO_ROOT / 'GravityTseHisPrice' / 'data' / 'tse_data.db',
        # Home directory variant
        Path.home() / 'GravityTseHisPrice' / 'data' / 'tse_data.db',
    ]
    for candidate in _TSE_DB_CANDIDATES:
        if isinstance(candidate, str):
            candidate = Path(candidate)
        if candidate.exists():
            TSE_DB_FILE = str(candidate)
            break
    if TSE_DB_FILE is None:
        # Fallback to first candidate (will raise error if not found at runtime)
        TSE_DB_FILE = str(_TSE_DB_CANDIDATES[0])

# Internal Application Database (Output/Operational)
# This database stores analysis results, user data, etc.
# Tests expect the operational DB to be named tool_performance.db for portability.
APP_DB_FILE = str(REPO_ROOT / 'data' / 'tool_performance.db')
