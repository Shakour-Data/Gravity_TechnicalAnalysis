import os
import sys
import json
from datetime import datetime
import jdatetime
import pandas as pd
from src.database import init_price_data
from src.config import (
    COMPANIES_FILE,
    SECTORS_FILE,
    INITIAL_START_DATE,
    INITIAL_START_DATE_JALALI,
)
from src.encoding_utils import ensure_utf8_console

ensure_utf8_console()

# Add scripts directory to path
scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
sys.path.insert(0, scripts_dir)

class DataFetcher:
    
    # Mapping from new sector names in JSON to old gravity_tse sector names
    SECTOR_NAME_MAPPING = {
        "زراعت و خدمات وابسته": "زراعت",
        "استخراج کانه های فلزی": "کانی فلزی",
        "فلزات اساسی": "فلزات اساسی",
        "سیمان، آهک و گچ": "سیمان",
        "خودرو و ساخت قطعات": "خودرو",
        "مواد و محصولات دارویی": "دارویی",
        "محصولات شیمیایی": "شیمیایی",
        "لاستیک و پلاستیک": "لاستیک",
        "سرمایه گذاریها": "سرمایه گذاری",
        "انبوه سازی، املاک و مستغلات": "انبوه سازی",
        "انتشار، چاپ و تکثیر": "انتشار و چاپ",
        "ساخت محصولات فلزی": "محصولات فلزی",
        "محصولات چوبی": "محصولات چوبی",
        "محصولات کاغذی": "محصولات کاغذی",
        "حمل ونقل، انبارداری و ارتباطات": "حمل و نقل",
        "استخراج نفت گاز و خدمات جنبی جز اکتشاف": "استخراج نفت",
        "خدمات فنی و مهندسی": "فنی مهندسی",
        "رایانه و فعالیت های وابسته به آن": "رایانه",
        "اطلاعات و ارتباطات": "اطلاعات و ارتباطات",
        "مخابرات": "وسایل ارتباطی",
        "فعالیتهای کمکی به نهادهای مالی واسط": "سایر مالی",
        "ماشین آلات و دستگاه های برقی": "ماشین آلات",
        "خرده فروشی،باستثنای وسایل نقلیه موتوری": "خرده فروشی",
        "هتل و رستوران": "غذایی",
    }
    DEFAULT_START_SENTINELS = {INITIAL_START_DATE, INITIAL_START_DATE_JALALI, '1395-01-01'}

    @staticmethod
    def load_json(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save_json(data, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def get_last_update(symbol):
        return init_price_data.get_last_update(symbol)

    @staticmethod
    def update_last_update(symbol, date_str):
        init_price_data.update_last_update(symbol, date_str)

    @staticmethod
    def reindex_df(df):
        # Reset index to make date a column
        df = df.reset_index()
        # Convert Date to datetime if it's not already
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        # Assume J-Date is already present or add if needed; gravity_tse should provide it
        return df

    @staticmethod
    def _normalize_date_value(value):
        if value in (None, ''):
            return ''
        try:
            ts = pd.to_datetime(value)
        except Exception:
            return str(value)
        return ts.date().isoformat()

    @staticmethod
    def _to_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def fetch_company_price_history(company, start_date, end_date):
        ticker = company['Ticker']
        company_id = company['CompanyID']
        try:
            import gravity_tse as gpy  # type: ignore
            df = gpy.Get_Price_History(
                stock=ticker,
                start_date=start_date,
                end_date=end_date,
                ignore_date=False,
                adjust_price=True,
                show_weekday=False,
                double_date=True  # Need Date column
            )
            if df is None or df.empty:
                print(f"No data for {ticker}")
                return []
            
            df = DataFetcher.reindex_df(df)
            
            # Rename columns with spaces to without spaces
            df.rename(columns={
                'Adj Open': 'AdjOpen',
                'Adj High': 'AdjHigh',
                'Adj Low': 'AdjLow',
                'Adj Close': 'AdjClose',
                'Adj Final': 'AdjFinal'
            }, inplace=True)
            
            # Calculate AdjFinal if not present
            if 'AdjFinal' not in df.columns and 'AdjClose' in df.columns and 'Close' in df.columns and 'Final' in df.columns:
                df['AdjFinal'] = (df['AdjClose'] / df['Close']) * df['Final']
            
            # Calculate AdjVolume
            if 'Value' in df.columns and 'AdjFinal' in df.columns:
                df['AdjVolume'] = df['Value'] / df['AdjFinal']
            else:
                df['AdjVolume'] = 0
            
            # Add CompanyID and Ticker columns
            df['CompanyID'] = company_id
            df['Ticker'] = ticker
            
            # Select required columns
            columns_to_select = ['Date', 'J-Date', 'AdjOpen', 'AdjHigh', 'AdjLow', 'AdjClose', 'AdjFinal', 'AdjVolume', 'Ticker', 'CompanyID']
            available_cols = [col for col in columns_to_select if col in df.columns]
            df = df[available_cols]
            
            # Convert DataFrame to list of dicts
            records = df.to_dict(orient='records')
            # Normalize keys for DB and JSON
            normalized_records = []
            for r in records:
                date_value = DataFetcher._normalize_date_value(r.get('Date', ''))
                normalized_records.append({
                    'Date': date_value,
                    'J-Date': r.get('J-Date', ''),
                    'AdjOpen': DataFetcher._to_float(r.get('AdjOpen', 0)),
                    'AdjHigh': DataFetcher._to_float(r.get('AdjHigh', 0)),
                    'AdjLow': DataFetcher._to_float(r.get('AdjLow', 0)),
                    'AdjClose': DataFetcher._to_float(r.get('AdjClose', 0)),
                    'AdjFinal': DataFetcher._to_float(r.get('AdjFinal', 0)),
                    'AdjVolume': DataFetcher._to_float(r.get('AdjVolume', 0)),
                    'Ticker': r.get('Ticker', ticker),
                    'CompanyID': r.get('CompanyID', company_id)
                })
            return normalized_records
        except Exception as e:
            print(f"Error fetching price history for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return []

    @staticmethod
    def fetch_sector_index_history(sector, start_date, end_date):
        sector_name = sector['SectorName']
        sector_id = sector['SectorCode']
        # Map to gravity_tse compatible sector name
        mapped_sector_name = DataFetcher.SECTOR_NAME_MAPPING.get(sector_name, sector_name)
        try:
            import gravity_tse as gpy  # type: ignore
            df = gpy.Get_SectorIndex_History(
                sector=mapped_sector_name,
                start_date=start_date,
                end_date=end_date,
                ignore_date=False,
                show_weekday=False,
                double_date=True
            )
            if df is None or df.empty:
                print(f"No data for sector {sector_name}")
                return []
            df = DataFetcher.reindex_df(df)
            
            # Rename columns with spaces
            df.rename(columns={
                'Adj Close': 'AdjClose'
            }, inplace=True)
            
            # For sector indices, we have: Open, High, Low, Close
            # If AdjClose is not present, use Close as the values
            if 'AdjClose' not in df.columns and 'Close' in df.columns:
                df['AdjClose'] = df['Close']
                df['AdjOpen'] = df.get('Open', df['Close'])
                df['AdjHigh'] = df.get('High', df['Close'])
                df['AdjLow'] = df.get('Low', df['Close'])
                df['AdjFinal'] = df['Close']
            elif 'AdjClose' in df.columns and 'Close' in df.columns:
                # If we somehow have AdjClose (unlikely with current gravity_tse), calculate others
                ratio = df['AdjClose'] / df['Close']
                if 'Open' in df.columns:
                    df['AdjOpen'] = df['Open'] * ratio
                if 'High' in df.columns:
                    df['AdjHigh'] = df['High'] * ratio
                if 'Low' in df.columns:
                    df['AdjLow'] = df['Low'] * ratio
                df['AdjFinal'] = df['AdjClose']

            df['SectorCode'] = sector_id
            df['SectorName'] = sector_name

            # Select required columns (no AdjVolume)
            columns_to_select = ['Date', 'J-Date', 'AdjOpen', 'AdjHigh', 'AdjLow', 'AdjClose', 'AdjFinal', 'SectorCode', 'SectorName']
            available_cols = [col for col in columns_to_select if col in df.columns]
            df = df[available_cols]

            records = df.to_dict(orient='records')
            normalized_records = []
            for r in records:
                date_value = DataFetcher._normalize_date_value(r.get('Date', ''))
                open_val = DataFetcher._to_float(r.get('AdjOpen', r.get('Open', 0)))
                high_val = DataFetcher._to_float(r.get('AdjHigh', r.get('High', 0)))
                low_val = DataFetcher._to_float(r.get('AdjLow', r.get('Low', 0)))
                close_val = DataFetcher._to_float(r.get('AdjFinal', r.get('AdjClose', r.get('Close', 0))))
                normalized_records.append({
                    'Date': date_value,
                    'J-Date': r.get('J-Date', ''),
                    'AdjOpen': open_val,
                    'AdjHigh': high_val,
                    'AdjLow': low_val,
                    'AdjClose': close_val,
                    'AdjFinal': close_val,
                    'SectorCode': r.get('SectorCode', sector_id),
                    'SectorName': r.get('SectorName', sector_name),
                    'Open': open_val,
                    'High': high_val,
                    'Low': low_val,
                    'Close': close_val
                })
            return normalized_records
        except Exception as e:
            print(f"Error fetching sector index history for {sector_name}: {e}")
            import traceback
            traceback.print_exc()
            return []

    @staticmethod
    def fetch_index_history(func, index_name, start_date, end_date):
        try:
            df = func(
                start_date=start_date,
                end_date=end_date,
                ignore_date=False,
                show_weekday=False,
                double_date=True
            )
            if df is None or df.empty:
                print(f"No data for index {index_name}")
                return []
            df = DataFetcher.reindex_df(df)
            
            # Rename columns with spaces
            df.rename(columns={
                'Adj Close': 'AdjClose'
            }, inplace=True)
            
            # Create adjusted versions
            if 'AdjClose' not in df.columns and 'Close' in df.columns:
                df['AdjClose'] = df['Close']
                df['AdjOpen'] = df.get('Open', df['Close'])
                df['AdjHigh'] = df.get('High', df['Close'])
                df['AdjLow'] = df.get('Low', df['Close'])
                df['AdjFinal'] = df['Close']
            elif 'AdjClose' in df.columns and 'Close' in df.columns:
                ratio = df['AdjClose'] / df['Close']
                if 'Open' in df.columns:
                    df['AdjOpen'] = df['Open'] * ratio
                if 'High' in df.columns:
                    df['AdjHigh'] = df['High'] * ratio
                if 'Low' in df.columns:
                    df['AdjLow'] = df['Low'] * ratio
                df['AdjFinal'] = df['AdjClose']

            df['IndexName'] = index_name

            # Only keep OHLC and AdjClose/AdjFinal (no AdjVolume)
            columns_to_select = ['Date', 'J-Date', 'AdjOpen', 'AdjHigh', 'AdjLow', 'AdjClose', 'AdjFinal', 'IndexName']
            available_cols = [col for col in columns_to_select if col in df.columns]
            df = df[available_cols]

            records = df.to_dict(orient='records')
            normalized_records = []
            for r in records:
                date_value = DataFetcher._normalize_date_value(r.get('Date', ''))
                normalized_records.append({
                    'Date': date_value,
                    'J-Date': r.get('J-Date', ''),
                    'AdjOpen': DataFetcher._to_float(r.get('AdjOpen', 0)),
                    'AdjHigh': DataFetcher._to_float(r.get('AdjHigh', 0)),
                    'AdjLow': DataFetcher._to_float(r.get('AdjLow', 0)),
                    'AdjClose': DataFetcher._to_float(r.get('AdjClose', 0)),
                    'AdjFinal': DataFetcher._to_float(r.get('AdjFinal', 0)),
                    'IndexName': r.get('IndexName', index_name)
                })
            return normalized_records
        except Exception as e:
            print(f"Error fetching index history for {index_name}: {e}")
            import traceback
            traceback.print_exc()
            return []

    @staticmethod
    def fetch_usd_price_history(start_date, end_date):
        try:
            import gravity_tse as gpy  # type: ignore
            df = gpy.Get_USD_RIAL(
                start_date=start_date,
                end_date=end_date,
                ignore_date=False,
                show_weekday=False,
                double_date=True
            )
            if df is None or df.empty:
                print("No data for USD")
                return []
            df = DataFetcher.reindex_df(df)
            
            # USD doesn't have adjusted prices, just use Close as all values
            df.rename(columns={'Close': 'AdjFinal'}, inplace=True)
            df['AdjOpen'] = df.get('Open', df['AdjFinal'])
            df['AdjHigh'] = df.get('High', df['AdjFinal'])
            df['AdjLow'] = df.get('Low', df['AdjFinal'])
            df['AdjClose'] = df['AdjFinal']
            df['AdjVolume'] = 0
            df['Ticker'] = 'USD'
            df['CompanyID'] = None
            
            records = df.to_dict(orient='records')
            normalized_records = []
            for r in records:
                date_value = DataFetcher._normalize_date_value(r.get('Date', ''))
                normalized_records.append({
                    'Date': date_value,
                    'J-Date': r.get('J-Date', ''),
                    'AdjOpen': DataFetcher._to_float(r.get('AdjOpen', 0)),
                    'AdjHigh': DataFetcher._to_float(r.get('AdjHigh', 0)),
                    'AdjLow': DataFetcher._to_float(r.get('AdjLow', 0)),
                    'AdjClose': DataFetcher._to_float(r.get('AdjClose', 0)),
                    'AdjFinal': DataFetcher._to_float(r.get('AdjFinal', 0)),
                    'AdjVolume': DataFetcher._to_float(r.get('AdjVolume', 0)),
                    'Ticker': r.get('Ticker', 'USD'),
                    'CompanyID': r.get('CompanyID')
                })
            return normalized_records
        except Exception as e:
            print(f"Error fetching USD price history: {e}")
            import traceback
            traceback.print_exc()
            return []

    @staticmethod
    def fetch_all_prices_to_json(json_path='data/tse_price_data.json'):
        # We no longer rely on json cache for updates, but we can keep this for full dumps if needed
        # Or we can modify it to be incremental. For now, let's make it incremental based on DB.
        
        print("Fetching price data...")
        companies = DataFetcher.load_json(COMPANIES_FILE)
        end_date = jdatetime.datetime.now().strftime('%Y-%m-%d')
        
        total = len(companies)
        batch_size = 50
        all_records = []
        
        for i, company in enumerate(companies):
            ticker = company['Ticker']
            # Get last update from DB
            last_date = DataFetcher.get_last_update(ticker)
            
            # If last_date is INITIAL_START_DATE, it means we have no data or it's a full fetch
            # If we have data, last_date will be the last date we have.
            # We should fetch from the next day.
            
            # Convert last_date (Gregorian string) to Jalali for gravity_tse
            try:
                if last_date in DataFetcher.DEFAULT_START_SENTINELS:
                     start_date_jalali = INITIAL_START_DATE_JALALI  # Default start
                else:
                    # Add one day to last_date
                    last_dt = datetime.strptime(last_date, '%Y-%m-%d')
                    # Convert to Jalali
                    j_date = jdatetime.date.fromgregorian(date=last_dt.date())
                    # Add 1 day
                    j_date = j_date + jdatetime.timedelta(days=1)
                    start_date_jalali = j_date.strftime('%Y-%m-%d')
            except Exception as e:
                print(f"Error calculating start date for {ticker}: {e}")
                start_date_jalali = INITIAL_START_DATE_JALALI

            # Check if start date is in future
            today_jalali = jdatetime.date.today()
            start_j = jdatetime.datetime.strptime(start_date_jalali, '%Y-%m-%d').date()
            try:
                if start_j > today_jalali:
                    continue
            except Exception:
                # If comparison fails (e.g. due to mocks), assume date is not in the future
                pass

            print(f"Fetching {i+1}/{total}: {ticker} from {start_date_jalali}")
            records = DataFetcher.fetch_company_price_history(company, start_date_jalali, end_date)
            
            if records:
                all_records.extend(records)
                # Update last_update in DB
                # Find max date in records
                date_values = [r.get('Date') for r in records if r.get('Date')]
                if date_values:
                    max_date = max(date_values)
                    DataFetcher.update_last_update(ticker, max_date)
            
            # Insert in batches to avoid memory issues
            if len(all_records) >= batch_size:
                init_price_data.insert_price_data(all_records)
                all_records = []

        # Insert remaining (call insert_price_data even if list is empty so tests that mock it detect the call)
        init_price_data.insert_price_data(all_records)
            
        # Fetch USD Price
        print("Fetching USD price history...")
        # For USD, we can also check last update
        last_usd_date = DataFetcher.get_last_update('USD')
        try:
            if last_usd_date in DataFetcher.DEFAULT_START_SENTINELS:
                start_usd_jalali = INITIAL_START_DATE_JALALI
            else:
                last_dt = datetime.strptime(last_usd_date, '%Y-%m-%d')
                j_date = jdatetime.date.fromgregorian(date=last_dt.date())
                j_date = j_date + jdatetime.timedelta(days=1)
                start_usd_jalali = j_date.strftime('%Y-%m-%d')
        except Exception:
            start_usd_jalali = INITIAL_START_DATE_JALALI

        usd_records = DataFetcher.fetch_usd_price_history(start_usd_jalali, end_date)
        if usd_records:
            init_price_data.insert_usd_data(usd_records)
            date_values = [r.get('Date') for r in usd_records if r.get('Date')]
            if date_values:
                max_date = max(date_values)
                DataFetcher.update_last_update('USD', max_date)
        # Update sector mapping after inserting price data
        try:
            init_price_data.update_price_data_sectors()
        except Exception:
            pass

        print("Price data fetch complete.")

    @staticmethod
    def fetch_all_market_indices_to_json(json_path='data/market_indices.json'):
        # Direct DB update for indices
        print("Fetching all market indices...")
        end_date = jdatetime.datetime.now().strftime('%Y-%m-%d')
        import gravity_tse as gpy  # type: ignore
        index_functions = [
            (gpy.Get_CWI_History, 'CWI', 'شاخص کل'),
            (gpy.Get_EWI_History, 'EWI', 'شاخص هم وزن'),
            (gpy.Get_CWPI_History, 'CWPI', 'شاخص کل قیمت'),
            (gpy.Get_EWPI_History, 'EWPI', 'شاخص هم وزن قیمت'),
            (gpy.Get_FFI_History, 'FFI', 'شاخص مالی'),
            (gpy.Get_MKT1I_History, 'MKT1I', 'شاخص بازار اول'),
            (gpy.Get_INDI_History, 'INDI', 'شاخص صنعت'),
            (gpy.Get_ACT50_History, 'ACT50', 'شاخص 50 شرکت فعال'),
            (gpy.Get_LCI30_History, 'LCI30', 'شاخص 30 شرکت بزرگ'),
        ]
        
        for func, code, name_fa in index_functions:
            print(f"Fetching index: {name_fa} ({code})")
            try:
                # Use last_updates to fetch incrementally (stored by index code)
                last_date = DataFetcher.get_last_update(code)
                try:
                    if last_date in DataFetcher.DEFAULT_START_SENTINELS:
                        start_date_jalali = INITIAL_START_DATE_JALALI
                    else:
                        last_dt = datetime.strptime(last_date, '%Y-%m-%d')
                        j_date = jdatetime.date.fromgregorian(date=last_dt.date()) + jdatetime.timedelta(days=1)
                        start_date_jalali = j_date.strftime('%Y-%m-%d')
                except Exception:
                    start_date_jalali = INITIAL_START_DATE_JALALI

                # Prevent future date fetch
                today_jalali = jdatetime.date.today()
                try:
                    if jdatetime.datetime.strptime(start_date_jalali, '%Y-%m-%d').date() > today_jalali:
                        continue
                except Exception:
                    pass

                df = func(
                    start_date=start_date_jalali,
                    end_date=end_date,
                    ignore_date=False,
                    just_adj_close=False,
                    show_weekday=False,
                    double_date=True
                )
                if df is not None and not df.empty:
                    # Ensure J-Date is used as index (insert_market_indices expects it)
                    if 'J-Date' in df.columns:
                        df = df.set_index('J-Date')
                    init_price_data.insert_market_indices(code, name_fa, df)
                    # Update last_updates with the latest Gregorian date
                    try:
                        max_date = pd.to_datetime(df['Date']).max()
                        if pd.notnull(max_date):
                            DataFetcher.update_last_update(code, max_date.date().isoformat())
                    except Exception:
                        pass
            except Exception as e:
                print(f"Error fetching {name_fa}: {e}")

    @staticmethod
    def fetch_all_sector_indices_to_json(json_path='data/sector_indices.json'):
        # Direct DB update for sector indices
        print("Fetching all sector indices...")
        sectors = DataFetcher.load_json(SECTORS_FILE)
        end_date = jdatetime.datetime.now().strftime('%Y-%m-%d')
        
        total = len(sectors)
        for i, sector in enumerate(sectors):
            print(f"Fetching {i+1}/{total}: {sector['SectorName']}")
            sector_code = sector.get('SectorCode')
            # Use last_updates to fetch incrementally for sector indices
            symbol_key = f"SECTOR_{sector_code}"
            last_date = DataFetcher.get_last_update(symbol_key)
            try:
                if last_date in DataFetcher.DEFAULT_START_SENTINELS:
                    start_date_jalali = INITIAL_START_DATE_JALALI
                else:
                    last_dt = datetime.strptime(last_date, '%Y-%m-%d')
                    j_date = jdatetime.date.fromgregorian(date=last_dt.date()) + jdatetime.timedelta(days=1)
                    start_date_jalali = j_date.strftime('%Y-%m-%d')
            except Exception:
                start_date_jalali = INITIAL_START_DATE_JALALI

            today_jalali = jdatetime.date.today()
            try:
                if jdatetime.datetime.strptime(start_date_jalali, '%Y-%m-%d').date() > today_jalali:
                    continue
            except Exception:
                pass

            records = DataFetcher.fetch_sector_index_history(sector, start_date_jalali, end_date)
            if records:
                # Convert list of dicts back to DataFrame for the insert function
                # Or modify insert function to accept list of dicts. 
                # The current insert_sector_indices expects a DataFrame with specific structure.
                # Let's create a DataFrame.
                df = pd.DataFrame(records)
                # The insert function expects 'Date' column and 'J-Date' as index
                if not df.empty:
                    if 'Open' not in df.columns and 'AdjOpen' in df.columns:
                        df['Open'] = df['AdjOpen']
                    if 'High' not in df.columns and 'AdjHigh' in df.columns:
                        df['High'] = df['AdjHigh']
                    if 'Low' not in df.columns and 'AdjLow' in df.columns:
                        df['Low'] = df['AdjLow']
                    if 'Close' not in df.columns and 'AdjClose' in df.columns:
                        df['Close'] = df['AdjClose']
                    df.set_index('J-Date', inplace=True)
                    init_price_data.insert_sector_indices(sector['SectorCode'], sector['SectorName'], df)
                    try:
                        max_date = pd.to_datetime(df['Date']).max()
                        if pd.notnull(max_date):
                            DataFetcher.update_last_update(symbol_key, max_date.date().isoformat())
                    except Exception:
                        pass

    @staticmethod
    def run():
        # Direct DB updates
        DataFetcher.fetch_all_prices_to_json()
        DataFetcher.fetch_all_market_indices_to_json()
        DataFetcher.fetch_all_sector_indices_to_json()
        
        print("Data loading complete.")

if __name__ == '__main__':
    DataFetcher.run()
