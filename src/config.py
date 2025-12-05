import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# External TSE Database (Input Source)
# This database contains the raw market data (price_data, sectors, etc.)
# as defined in docs/inputs/DATABASE.md
# Try multiple possible locations for portability across development machines
_TSE_DB_CANDIDATES = [
    # Windows path (original)
    r'E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db',
    # Relative path (relative to project root)
    BASE_DIR.parent / 'GravityTseHisPrice' / 'data' / 'tse_data.db',
    # Home directory variant
    Path.home() / 'GravityTseHisPrice' / 'data' / 'tse_data.db',
]
TSE_DB_FILE = None
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
APP_DB_FILE = os.path.join(BASE_DIR, 'data', 'tool_performance.db')
