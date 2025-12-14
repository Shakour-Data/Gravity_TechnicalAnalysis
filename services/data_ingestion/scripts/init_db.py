"""
Complete Database Initialization Script
========================================
This script performs a complete database initialization including:
1. Creating all tables with correct schema
2. Loading basic reference data (sectors, markets, panels, companies)
3. Preparing the database for price data collection

Usage: python init_db.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import init_price_data
from src.config import DB_FILE, SECTORS_FILE, MARKETS_FILE, PANELS_FILE, COMPANIES_FILE
import json

def main():
    print("\n" + "="*80)
    print("COMPLETE DATABASE INITIALIZATION")
    print("="*80)
    
    # Step 1: Create all tables
    print("\n📋 Step 1: Creating database tables...")
    init_price_data.create_tables()
    print("✓ All tables created successfully")
    
    # Step 2: Create indices tables
    print("\n📊 Step 2: Creating indices tables...")
    init_price_data.create_indices_tables()
    print("✓ Indices tables created successfully")
    
    # Step 3: Load basic data
    print("\n📦 Step 3: Loading basic reference data...")
    
    try:
        # Load sectors
        with open(SECTORS_FILE, 'r', encoding='utf-8') as f:
            sectors = json.load(f)
        init_price_data.insert_sectors(sectors)
        
        # Load markets
        with open(MARKETS_FILE, 'r', encoding='utf-8') as f:
            markets = json.load(f)
        init_price_data.insert_markets(markets)
        
        # Load panels
        with open(PANELS_FILE, 'r', encoding='utf-8') as f:
            panels = json.load(f)
        init_price_data.insert_panels(panels)
        
        # Load companies
        with open(COMPANIES_FILE, 'r', encoding='utf-8') as f:
            companies = json.load(f)
        init_price_data.insert_companies(companies)
        
        print("✓ Basic reference data loaded successfully")
        
    except FileNotFoundError as e:
        print(f"⚠️  Warning: Could not load some reference data: {e}")
        print("   You may need to run data collection first.")
    except Exception as e:
        print(f"❌ Error loading basic data: {e}")
        return False
    
    print("\n" + "="*80)
    print("✅ DATABASE INITIALIZATION COMPLETE!")
    print("="*80)
    print("\n📌 Database location:", DB_FILE)
    print("\n🎯 Next steps:")
    print("  1. Load market indices:")
    print("     python main.py load-market-indices")
    print("  2. Load sector indices:")
    print("     python main.py load-sector-indices")
    print("  3. Fetch historical price data:")
    print("     python main.py fetch")
    print("\n" + "="*80)
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
