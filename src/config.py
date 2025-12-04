import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# External TSE Database (Input Source)
# This database contains the raw market data (price_data, sectors, etc.)
# as defined in docs/inputs/DATABASE.md
TSE_DB_FILE = r'E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db'

# Internal Application Database (Output/Operational)
# This database stores analysis results, user data, etc.
APP_DB_FILE = os.path.join(BASE_DIR, 'data', 'tool_performance.db')
