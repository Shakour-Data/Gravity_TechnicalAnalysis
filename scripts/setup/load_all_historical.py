#!/usr/bin/env python
"""Load Complete Historical Data for All Symbols - بارگذاری کامل داده‌های تاریخی"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Add root to path
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def load_all_historical_data():
    """تمام داده‌های تاریخی برای تمام نمادها را بارگذاری کن"""

    print("=" * 70)
    print("📊 بارگذاری تمام داده‌های تاریخی")
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

        print("\n🔄 دریافت تمام نمادها...")

        # Get all tickers from tse_reference
        project_cursor.execute("SELECT ticker FROM tse_reference ORDER BY ticker")
        tickers = [row[0] for row in project_cursor.fetchall()]

        print(f"✓ تعداد نمادها: {len(tickers)}")

        total_loaded = 0
        errors = 0

        print("\n📥 بارگذاری داده‌های قیمتی...")

        for ticker_idx, ticker in enumerate(tickers, 1):
            if ticker_idx % 50 == 0:
                print(f"  Processing {ticker_idx}/{len(tickers)}...")

            try:
                # Get ALL price data for this ticker
                tse_cursor.execute("""
                    SELECT date, adj_close, adj_volume, adj_high, adj_low, adj_open
                    FROM price_data
                    WHERE ticker = ?
                    ORDER BY date ASC
                """, (ticker,))

                records = tse_cursor.fetchall()

                if not records:
                    continue

                for date, close, _, high, low, open_p in records:
                    try:
                        # Calculate scores based on price
                        close_f = float(close or 0)
                        high_f = float(high or 0)
                        low_f = float(low or 0)
                        open_f = float(open_p or 0)

                        # Calculate trend score
                        if close_f > open_f:
                            trend_score = 0.5 + (close_f - open_f) / max(open_f, 1) * 0.4
                        else:
                            trend_score = 0.5 - (open_f - close_f) / max(open_f, 1) * 0.4

                        # Calculate momentum score
                        volatility = (high_f - low_f) / max(close_f, 1) * 100 if close_f > 0 else 0
                        momentum_score = 0.5 + min(volatility / 10, 0.4)

                        # Combined score
                        combined_score = (trend_score * 0.6 + momentum_score * 0.4)

                        # Determine signal
                        if combined_score > 0.55:
                            signal = 'BULLISH'
                        elif combined_score < 0.45:
                            signal = 'BEARISH'
                        else:
                            signal = 'NEUTRAL'

                        project_cursor.execute("""
                            INSERT OR IGNORE INTO historical_scores (
                                ticker, analysis_date, timeframe,
                                trend_score, momentum_score, combined_score,
                                trend_signal, momentum_signal, combined_signal,
                                volume_score, price_at_analysis, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            ticker, date, '1d',
                            max(0, min(trend_score, 1)),
                            max(0, min(momentum_score, 1)),
                            max(0, min(combined_score, 1)),
                            signal, signal, signal,
                            max(0, min(volatility / 100, 1)),
                            close_f,
                            datetime.now().isoformat()
                        ))

                        total_loaded += 1

                    except Exception:
                        errors += 1
                        if errors < 5:
                            pass

            except Exception:
                pass

        project_conn.commit()

        print(f"\n✓ تعداد رکوردهای بارگذاری شده: {total_loaded:,}")
        print(f"⚠️ خطاهای پردازش: {errors}")

        # Show final stats
        project_cursor.execute("SELECT COUNT(*) FROM historical_scores")
        total_count = project_cursor.fetchone()[0]

        print("\n" + "=" * 70)
        print("✅ بارگذاری کامل شد!")
        print("=" * 70)
        print("\n📊 آمارهای نهایی:")
        print(f"  • کل رکوردهای historical_scores: {total_count:,}")

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
    success = load_all_historical_data()
    sys.exit(0 if success else 1)
