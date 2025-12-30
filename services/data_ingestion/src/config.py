import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directory
DATA_DIR = os.path.join(BASE_DIR, "data")

# Database file path
DB_FILE = os.path.join(DATA_DIR, "tse_data.db")

# Basic TSE Information directory
BASIC_INFO_DIR = os.path.join(DATA_DIR, "BasicTseInformation")

# JSON file paths
COMPANIES_FILE = os.path.join(BASIC_INFO_DIR, "companies.json")
SECTORS_FILE = os.path.join(BASIC_INFO_DIR, "sectors.json")
MARKETS_FILE = os.path.join(BASIC_INFO_DIR, "markets.json")
PANELS_FILE = os.path.join(BASIC_INFO_DIR, "panels.json")
SUBSECTORS_FILE = os.path.join(BASIC_INFO_DIR, "subsectors.json")

# Metadata file
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")

# Initial start dates (sentinel for "no data yet")
# Gregorian value is stored in last_updates; Jalali value is used when calling gravity_tse.
INITIAL_START_DATE_GREGORIAN = "2011-03-21"  # ~= 1390-01-01
INITIAL_START_DATE_JALALI = "1390-01-01"
INITIAL_START_DATE = INITIAL_START_DATE_GREGORIAN
