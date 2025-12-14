import sqlite3
from src.config import DB_FILE, INITIAL_START_DATE

class init_price_data:
    @staticmethod
    def _configure_connection(conn: sqlite3.Connection):
        """Apply SQLite pragmas for integrity and performance."""
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            # Some environments (read-only or in-memory) may not accept WAL/synchronous
            pass

    @staticmethod
    def _ensure_indexes(cursor: sqlite3.Cursor):
        """Create useful indexes if tables exist."""
        def _table_exists(name: str) -> bool:
            cursor.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (name,),
            )
            return cursor.fetchone() is not None

        if _table_exists("price_data"):
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_price_ticker_date ON price_data(ticker, date)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_price_date ON price_data(date)"
            )

        if _table_exists("market_indices"):
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_index_date ON market_indices(index_code, date)"
            )

        if _table_exists("sector_indices"):
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_sector_code_date ON sector_indices(sector_code, date)"
            )

    @staticmethod
    def create_indices_tables():
        conn = init_price_data.get_connection()
        cursor = conn.cursor()
        
        # Create indices_info table to store index metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS indices_info (
                index_code TEXT PRIMARY KEY,
                index_name_fa TEXT NOT NULL,
                index_name_en TEXT,
                index_type TEXT NOT NULL CHECK(index_type IN ('market', 'sector'))
            )
        ''')
        
        # Create market_indices table with OHLC
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_indices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_code TEXT NOT NULL,
                j_date TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                FOREIGN KEY(index_code) REFERENCES indices_info(index_code),
                UNIQUE(index_code, date)
            )
        ''')
        
        # Create sector_indices table with OHLC
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sector_indices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sector_code TEXT NOT NULL,
                j_date TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                FOREIGN KEY(sector_code) REFERENCES sectors(sector_id),
                UNIQUE(sector_code, date)
            )
        ''')

        init_price_data._ensure_indexes(cursor)
        conn.commit()
        conn.close()
        print("Indices tables created successfully.")
        init_price_data.create_usd_table()
        init_price_data.insert_indices_info()

    @staticmethod
    def create_usd_table():
        conn = init_price_data.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usd_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                j_date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                final REAL,
                UNIQUE(date)
            )
        ''')
        conn.commit()
        conn.close()
        print("USD prices table created successfully.")

    @staticmethod
    def insert_indices_info():
        """Insert metadata for market indices"""
        conn = init_price_data.get_connection()
        cursor = conn.cursor()
        
        indices_info = [
            ('CWI', 'شاخص کل', 'Overall Index', 'market'),
            ('EWI', 'شاخص هم وزن', 'Equal Weight Index', 'market'),
            ('CWPI', 'شاخص کل قیمت', 'Price Index', 'market'),
            ('EWPI', 'شاخص هم وزن قیمت', 'Equal Weight Price Index', 'market'),
            ('FFI', 'شاخص مالی', 'Financial Index', 'market'),
            ('MKT1I', 'شاخص بازار اول', 'First Market Index', 'market'),
            ('MKT2I', 'شاخص بازار دوم', 'Second Market Index', 'market'),
            ('INDI', 'شاخص صنعت', 'Industry Index', 'market'),
            ('ACT50', 'شاخص 50 شرکت فعال', 'Top 50 Active Companies', 'market'),
            ('LCI30', 'شاخص 30 شرکت بزرگ', 'Top 30 Large Companies', 'market'),
        ]
        
        for info in indices_info:
            cursor.execute('''
                INSERT OR IGNORE INTO indices_info (index_code, index_name_fa, index_name_en, index_type)
                VALUES (?, ?, ?, ?)
            ''', info)
        
        conn.commit()
        conn.close()
        print("Indices info inserted successfully.")

    @staticmethod
    def insert_market_indices(index_code, index_name_fa, df):
        """Insert market index data with OHLC"""
        conn = init_price_data.get_connection()
        cursor = conn.cursor()
        
        records = []
        for idx, row in df.iterrows():
            records.append((
                index_code,
                idx,  # J-Date is the index
                str(row.get('Date')),
                row.get('Open'),
                row.get('High'),
                row.get('Low'),
                row.get('Close')
            ))
            
        cursor.executemany('''
            INSERT OR REPLACE INTO market_indices (index_code, j_date, date, open, high, low, close)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', records)
        
        conn.commit()
        conn.close()
        print(f"Inserted {len(df)} records into market_indices for {index_name_fa} ({index_code}).")

    @staticmethod
    def insert_sector_indices(sector_code, sector_name, df):
        """Insert sector index data with OHLC"""
        conn = init_price_data.get_connection()
        cursor = conn.cursor()
        
        records = []
        for idx, row in df.iterrows():
            records.append((
                sector_code,
                idx,  # J-Date is the index
                str(row.get('Date')),
                row.get('Open'),
                row.get('High'),
                row.get('Low'),
                row.get('Close')
            ))
            
        cursor.executemany('''
            INSERT OR REPLACE INTO sector_indices (sector_code, j_date, date, open, high, low, close)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', records)
        
        conn.commit()
        conn.close()
        print(f"Inserted {len(df)} records into sector_indices for {sector_name} ({sector_code}).")

    @staticmethod
    def insert_usd_data(records):
        conn = init_price_data.get_connection()
        cursor = conn.cursor()
        
        data_to_insert = []
        for record in records:
             data_to_insert.append((
                record.get('Date'),
                record.get('J-Date'),
                record.get('AdjOpen'),
                record.get('AdjHigh'),
                record.get('AdjLow'),
                record.get('AdjClose'),
                record.get('AdjFinal')
            ))
            
        cursor.executemany('''
            INSERT OR REPLACE INTO usd_prices (date, j_date, open, high, low, close, final)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        conn.close()
        print(f"Inserted {len(records)} records into usd_prices.")
    
    @staticmethod
    def get_connection():
        conn = sqlite3.connect(DB_FILE)
        init_price_data._configure_connection(conn)
        return conn

    @staticmethod
    def create_tables():
        conn = init_price_data.get_connection()
        cursor = conn.cursor()

        # Create sectors table (Independent)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sectors (
                sector_id INTEGER PRIMARY KEY, -- SectorCode
                sector_name TEXT UNIQUE,
                sector_name_en TEXT,
                us_sector TEXT
            )
        ''')

        # Create markets table (Independent)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS markets (
                market_id INTEGER PRIMARY KEY,
                market_name TEXT UNIQUE
            )
        ''')

        # Create panels table (Independent)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS panels (
                panel_id INTEGER PRIMARY KEY,
                panel_name TEXT UNIQUE
            )
        ''')

        # Create companies table (Depends on sectors, markets, panels)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS companies (
                company_id TEXT PRIMARY KEY,
                ticker TEXT UNIQUE,
                name TEXT,
                sector_id INTEGER,
                panel_id INTEGER,
                market_id INTEGER,
                FOREIGN KEY(sector_id) REFERENCES sectors(sector_id),
                FOREIGN KEY(panel_id) REFERENCES panels(panel_id),
                FOREIGN KEY(market_id) REFERENCES markets(market_id)
            )
        ''')

        # Create price_data table (Depends on companies, sectors)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                j_date TEXT,
                adj_open REAL,
                adj_high REAL,
                adj_low REAL,
                adj_close REAL,
                adj_final REAL,
                adj_volume REAL,
                sector_id INTEGER,
                ticker TEXT,
                company_id TEXT,
                FOREIGN KEY(company_id) REFERENCES companies(company_id),
                FOREIGN KEY(sector_id) REFERENCES sectors(sector_id),
                UNIQUE(ticker, date)
            )
        ''')

        # Create last_updates table (Independent)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS last_updates (
                symbol TEXT PRIMARY KEY,
                last_date TEXT
            )
        ''')


        init_price_data._ensure_indexes(cursor)
        conn.commit()
        conn.close()
        print("Database tables created successfully.")

    @staticmethod
    def insert_price_data(records):
        conn = init_price_data.get_connection()
        cursor = conn.cursor()

        data_to_insert = []
        for record in records:
            if isinstance(record, dict):
                ticker = record.get('Ticker')
                if ticker == 'USD':
                    continue

                adj_final = record.get('Adj Final') or record.get('AdjFinal')
                value = record.get('Value')
                adj_volume = (value / adj_final) if adj_final and value else (record.get('Adj Volume') or record.get('AdjVolume'))
                
                data_to_insert.append((
                    record.get('Date'),
                    record.get('J-Date'),
                    record.get('Adj Open') or record.get('AdjOpen'),
                    record.get('Adj High') or record.get('AdjHigh'),
                    record.get('Adj Low') or record.get('AdjLow'),
                    record.get('Adj Close') or record.get('AdjClose'),
                    adj_final,
                    adj_volume,
                    record.get('Ticker'),
                    record.get('CompanyID')
                ))
            elif isinstance(record, tuple):
                data_to_insert.append(record[:10])

        if data_to_insert:
            cursor.executemany('''
                INSERT OR REPLACE INTO price_data (date, j_date, adj_open, adj_high, adj_low, adj_close, adj_final, adj_volume, ticker, company_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data_to_insert)

        conn.commit()
        conn.close()
        print(f"Inserted {len(records)} records into price_data.")

    @staticmethod
    def insert_last_updates(updates):
        conn = init_price_data.get_connection()
        cursor = conn.cursor()

        for symbol, date in updates.items():
            cursor.execute('''
                INSERT OR REPLACE INTO last_updates (symbol, last_date)
                VALUES (?, ?)
            ''', (symbol, date))

        conn.commit()
        conn.close()
        print(f"Inserted {len(updates)} last updates.")

    @staticmethod
    def insert_companies(companies):
        conn = init_price_data.get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT sector_id FROM sectors')
        valid_sector_ids = {row[0] for row in cursor.fetchall()}
        cursor.execute('SELECT panel_id FROM panels')
        valid_panel_ids = {row[0] for row in cursor.fetchall()}
        cursor.execute('SELECT market_id FROM markets')
        valid_market_ids = {row[0] for row in cursor.fetchall()}

        missing_sectors = set()
        missing_panels = set()
        missing_markets = set()

        def _safe_int(value):
            if value is None:
                return None
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return None
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return None

        # Pre-scan for missing sectors and add them
        new_sectors = {}
        for company in companies:
            sector_code = _safe_int(company.get('SectorCode') or company.get('SectorID'))
            if sector_code is not None and sector_code not in valid_sector_ids:
                # Try to find a name
                name = company.get('IndustryGroupName') or company.get('IndustryName') or f"Sector {sector_code}"
                if sector_code not in new_sectors:
                    new_sectors[sector_code] = name
                elif new_sectors[sector_code].startswith("Sector ") and not name.startswith("Sector "):
                     new_sectors[sector_code] = name # Upgrade name if better one found
        
        if new_sectors:
            print(f"Found {len(new_sectors)} new sectors in companies list. Adding them...")
            for s_id, s_name in new_sectors.items():
                cursor.execute('''
                    INSERT OR IGNORE INTO sectors (sector_id, sector_name, sector_name_en, us_sector)
                    VALUES (?, ?, ?, ?)
                ''', (s_id, s_name, s_name, 'Unknown'))
            # Update valid_sector_ids
            valid_sector_ids.update(new_sectors.keys())

        for company in companies:
            sector_id = _safe_int(company.get('SectorCode') or company.get('SectorID'))
            if sector_id not in valid_sector_ids:
                if sector_id is not None:
                    missing_sectors.add(str(company.get('SectorCode') or sector_id))
                sector_id = None

            panel_id = _safe_int(company.get('PanelID') or company.get('PanelCode'))
            if panel_id not in valid_panel_ids:
                if panel_id is not None:
                    missing_panels.add(str(company.get('PanelID') or company.get('PanelCode') or panel_id))
                panel_id = None

            market_id = _safe_int(company.get('MarketID'))
            if market_id not in valid_market_ids:
                if market_id is not None:
                    missing_markets.add(str(company.get('MarketID') or market_id))
                market_id = None

            # board_id = _safe_int(company.get('BoardID'))  # unused

            cursor.execute('''
                INSERT OR REPLACE INTO companies (
                    company_id, ticker, name, sector_id,
                    panel_id, market_id
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                company.get('CompanyID'),
                company.get('Ticker'),
                company.get('Name'),
                sector_id,
                panel_id,
                market_id
            ))

        conn.commit()
        conn.close()
        print(f"Inserted {len(companies)} companies.")
        if missing_sectors:
            print(f"Warning: {len(missing_sectors)} sector codes not present in sectors table: {sorted(missing_sectors)}")
        if missing_panels:
            print(f"Warning: {len(missing_panels)} panel ids not present in panels table: {sorted(missing_panels)}")
        if missing_markets:
            print(f"Warning: {len(missing_markets)} market ids not present in markets table: {sorted(missing_markets)}")

    @staticmethod
    def insert_sectors(sectors):
        conn = init_price_data.get_connection()
        cursor = conn.cursor()

        for sector in sectors:
            cursor.execute('''
                INSERT OR REPLACE INTO sectors (sector_id, sector_name, sector_name_en, us_sector)
                VALUES (?, ?, ?, ?)
            ''', (
                int(sector.get('SectorCode')) if sector.get('SectorCode') is not None else None,
                sector.get('SectorName'),
                sector.get('SectorName_en'),
                sector.get('US_Sector')
            ))

        conn.commit()
        conn.close()
        print(f"Inserted {len(sectors)} sectors.")

    @staticmethod
    def insert_markets(markets):
        conn = init_price_data.get_connection()
        cursor = conn.cursor()

        for market in markets:
            cursor.execute('''
                INSERT OR REPLACE INTO markets (market_id, market_name)
                VALUES (?, ?)
            ''', (int(market.get('MarketID')) if market.get('MarketID') is not None else None, market.get('MarketName')))

        conn.commit()
        conn.close()
        print(f"Inserted {len(markets)} markets.")

    @staticmethod
    def insert_panels(panels):
        conn = init_price_data.get_connection()
        cursor = conn.cursor()

        for panel in panels:
            cursor.execute('''
                INSERT OR REPLACE INTO panels (panel_id, panel_name)
                VALUES (?, ?)
            ''', (int(panel.get('PanelID')) if panel.get('PanelID') is not None else None, panel.get('PanelName')))

        conn.commit()
        conn.close()
        print(f"Inserted {len(panels)} panels.")

    @staticmethod
    def get_last_update(symbol):
        conn = init_price_data.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT last_date FROM last_updates WHERE symbol = ?', (symbol,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else INITIAL_START_DATE

    @staticmethod
    def update_last_update(symbol, date_str):
        conn = init_price_data.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO last_updates (symbol, last_date)
            VALUES (?, ?)
        ''', (symbol, date_str))
        conn.commit()
        conn.close()

    @staticmethod
    def update_price_data_sectors():
        """Populate the `sector_id` column in `price_data` from `companies` table."""
        conn = init_price_data.get_connection()
        cursor = conn.cursor()
        # Update price_data.sector_id using the company->sector relationship
        cursor.execute('''
            UPDATE price_data
            SET sector_id = (
                SELECT sector_id FROM companies WHERE companies.company_id = price_data.company_id
            )
            WHERE company_id IS NOT NULL
        ''')
        conn.commit()
        conn.close()
