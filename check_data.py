import sqlite3

conn = sqlite3.connect('data/TechAnalysis.db')
cursor = conn.cursor()

print('=== DAILY CALCULATION COMPLETENESS CHECK ===\n')

# Check date ranges
cursor.execute("SELECT MIN(date(timestamp)), MAX(date(timestamp)), COUNT(DISTINCT date(timestamp)) FROM historical_indicator_scores")
min_date, max_date, total_days = cursor.fetchone()
print(f'Date range: {min_date} to {max_date} ({total_days} days)')

# Check symbols per recent date
cursor.execute("SELECT date(timestamp) as date, COUNT(DISTINCT symbol) as symbols FROM historical_indicator_scores WHERE date(timestamp) >= '2025-12-01' GROUP BY date(timestamp) ORDER BY date DESC")
print('\nRecent dates (December) - symbols per day:')
recent_data = cursor.fetchall()
for row in recent_data:
    print(f'  {row[0]}: {row[1]} symbols')

# Check if calculations are complete for each symbol on each day
if recent_data:
    latest_date = recent_data[0][0]
    cursor.execute(f"SELECT symbol, COUNT(*) as indicators FROM historical_indicator_scores WHERE date(timestamp) = '{latest_date}' GROUP BY symbol ORDER BY indicators DESC LIMIT 5")
    print(f'\nIndicators per symbol for {latest_date} (top 5):')
    for row in cursor.fetchall():
        print(f'  {row[0]}: {row[1]} indicators')

    # Check summary scores for latest date
    cursor.execute(f"SELECT COUNT(*) FROM historical_scores WHERE date(timestamp) = '{latest_date}'")
    summary_count = cursor.fetchone()[0]
    print(f'Summary scores for {latest_date}: {summary_count}')

    # Check pattern detections for latest date
    cursor.execute(f"SELECT COUNT(*) FROM pattern_detection_results WHERE date(timestamp) = '{latest_date}'")
    pattern_count = cursor.fetchone()[0]
    print(f'Pattern detections for {latest_date}: {pattern_count}')

# Check indicator categories
cursor.execute("SELECT indicator_category, COUNT(*) FROM historical_indicator_scores WHERE date(timestamp) = '2025-12-11' GROUP BY indicator_category ORDER BY indicator_category")
print('\nIndicator categories for 2025-12-11:')
categories = cursor.fetchall()
for cat, count in categories:
    print(f'  {cat}: {count} indicators')

# Check signal distribution
cursor.execute("SELECT signal, COUNT(*) FROM historical_indicator_scores WHERE date(timestamp) = '2025-12-11' GROUP BY signal ORDER BY signal")
print('\nSignal distribution for 2025-12-11:')
signals = cursor.fetchall()
for signal, count in signals:
    print(f'  {signal}: {count} indicators')

print('\n=== ASSESSMENT ===')
if len(categories) >= 6:  # TREND, MOMENTUM, VOLATILITY, VOLUME, CYCLE, SUPPORT_RESISTANCE
    print('✅ All indicator categories present')
else:
    print('❌ Missing indicator categories')

if len(signals) >= 7:  # All signal types
    print('✅ All signal types present')
else:
    print('❌ Missing signal types')

if summary_count > 0:
    print('✅ Summary scores calculated')
else:
    print('❌ No summary scores')

if pattern_count > 0:
    print('✅ Pattern detection working')
else:
    print('❌ No pattern detections')

conn.close()