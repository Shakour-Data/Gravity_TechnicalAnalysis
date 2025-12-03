import sys
import os
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.database import init_price_data
from src.config import DB_FILE

def main():
    print(f"Initializing database at: {DB_FILE}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    
    try:
        init_price_data.create_tables()
        print("Tables created successfully.")
        
        init_price_data.insert_indices_info()
        print("Default indices info inserted.")
        
        print("Database initialization complete.")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
