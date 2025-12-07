import os
import sqlite3

db = sqlite3.connect('data/gravity_tech.db')
cur = db.cursor()

tables = [
    'tse_reference',
    'historical_scores',
    'tool_performance_history',
    'tool_performance_stats',
    'ml_weights_history',
    'tool_recommendations_log',
    'market_data_cache',
    'pattern_detection_results'
]

print('Database Record Summary:')
print('=' * 60)

total = 0
for t in tables:
    cur.execute(f'SELECT COUNT(*) FROM {t}')
    count = cur.fetchone()[0]
    total += count
    print(f'{t:35} {count:>15,}')

print('=' * 60)
print(f'{"TOTAL":35} {total:>15,}')

size = os.path.getsize('data/gravity_tech.db') / (1024*1024)
print(f'\nDatabase Size: {size:.1f} MB')

db.close()
