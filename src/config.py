import os
from pathlib import Path

from gravity_tech.config.settings import resolve_tse_db_path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# External TSE Database (Input Source)
_RESOLVED_TSE = resolve_tse_db_path()
if not _RESOLVED_TSE:
    raise FileNotFoundError(
        "TSE database path not configured. Set `tse_db_path` in .env or export TSE_DB_PATH."
    )
TSE_DB_FILE = _RESOLVED_TSE

# Internal Application Database (Output/Operational)
APP_DB_FILE = os.path.join(BASE_DIR, 'data', 'tool_performance.db')
