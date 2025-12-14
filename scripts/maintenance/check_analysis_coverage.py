import sqlite3
from datetime import datetime, timedelta

# Path to the analysis results database
ANALYSIS_DB = 'data/TechAnalysis.db'

# Path to the source market data database
SOURCE_DB = r'E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db'

# Number of days to check
DAYS = 90

# Today's date (UTC, as used in pipeline)
today = datetime.utcnow().date()
start_date = today - timedelta(days=DAYS)

# Connect to analysis results database
conn = sqlite3.connect(ANALYSIS_DB)
cursor = conn.cursor()

# Get all symbols in analysis_results
def get_symbols():
    cursor.execute('SELECT DISTINCT symbol FROM analysis_results')
    return [row[0] for row in cursor.fetchall()]

# For each symbol, check if there are at least 90 unique analysis dates in the last 90 days
def check_analysis_coverage():
    symbols = get_symbols()
    incomplete = []
    for symbol in symbols:
        cursor.execute('''SELECT COUNT(DISTINCT date(analysis_date)) FROM analysis_results WHERE symbol = ? AND date(analysis_date) >= ?''', (symbol, start_date.isoformat()))
        count = cursor.fetchone()[0]
        if count < DAYS:
            incomplete.append((symbol, count))
    return incomplete

# Check for missing days for each symbol
def check_missing_days(symbol):
    cursor.execute('''SELECT DISTINCT date(analysis_date) FROM analysis_results WHERE symbol = ? AND date(analysis_date) >= ? ORDER BY date(analysis_date)''', (symbol, start_date.isoformat()))
    days = set(row[0] for row in cursor.fetchall())
    expected_days = set((start_date + timedelta(days=i)).isoformat() for i in range(DAYS))
    return sorted(expected_days - days)

if __name__ == '__main__':
    incomplete = check_analysis_coverage()
    if not incomplete:
        print('✅ All symbols have at least 90 days of analysis results.')
    else:
        print(f'❌ {len(incomplete)} symbols missing full 90-day coverage:')
        for symbol, count in incomplete:
            print(f'  {symbol}: {count} days')
            missing = check_missing_days(symbol)
            if missing:
                print(f'    Missing days: {missing[:5]} ... (total {len(missing)})')
    conn.close()
