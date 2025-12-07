#!/usr/bin/env python
"""Simple data loading from TSE to project database"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Add root to path
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def load_real_data():
    """بارگذاری داده‌های واقعی از TSE"""

    print("=" * 70)
    print("📊 بارگذاری داده‌های واقعی")
    print("=" * 70)

    tse_db_path = r"E:\Shakour\MyProjects\GravityTseHisPrice\data\tse_data.db"
    project_db_path = "data/gravity_tech.db"

    if not Path(tse_db_path).exists():
        print("❌ TSE database not found")
        return False

    try:
        tse_conn = sqlite3.connect(tse_db_path)
        tse_cursor = tse_conn.cursor()

        project_conn = sqlite3.connect(project_db_path)
        project_cursor = project_conn.cursor()

        print("\n🔄 Loading price data from TSE...")

        # Get all price data
        tse_cursor.execute("""
            SELECT ticker, date, adj_close, adj_volume
            FROM price_data
            ORDER BY ticker, date DESC
            LIMIT 5000
        """)

        records = tse_cursor.fetchall()
        loaded_count = 0

        for ticker, date, close, _ in records:
            try:
                # Calculate simple score
                trend_score = 0.5
                momentum_score = 0.5
                combined_score = 0.5

                project_cursor.execute("""
                    INSERT OR IGNORE INTO historical_scores (
                        ticker, analysis_date, timeframe,
                        trend_score, momentum_score, combined_score,
                        trend_signal, momentum_signal, combined_signal,
                        price_at_analysis, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker, date, '1d',
                    trend_score, momentum_score, combined_score,
                    'NEUTRAL', 'NEUTRAL', 'NEUTRAL',
                    close, datetime.now().isoformat()
                ))

                loaded_count += 1

            except Exception:
                pass

        project_conn.commit()
        print(f"✓ Loaded {loaded_count} records")

        print("\n" + "=" * 70)
        print("✅ Data loading completed!")
        print("=" * 70)

        tse_cursor.close()
        tse_conn.close()
        project_cursor.close()
        project_conn.close()

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = load_real_data()
    sys.exit(0 if success else 1)
