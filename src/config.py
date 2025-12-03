import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Database file path
DB_FILE = os.path.join(BASE_DIR, 'data', 'tse_data.db')
