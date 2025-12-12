"""
Fill missing backtest records for symbols using available market data.

Reads symbols from data/symbols.txt (one per line). For each symbol, if no
record exists in backtest_runs for the given interval/source, it runs a backtest
and persists the summary.

Usage (PowerShell/Windows example):
    $env:PYTHONPATH="src"; python scripts/fill_missing_backtests.py --interval 1d --limit 1200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gravity_tech.config.settings import settings
from gravity_tech.database.database_manager import DatabaseManager, DatabaseType
from gravity_tech.ml.backtesting import run_backtest_with_real_data

SYMBOLS_PATH = Path("data/symbols.txt")


def read_symbols(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"فایل نمادها یافت نشد: {path}")
    symbols = [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not symbols:
        raise SystemExit(f"فایل نمادها خالی است: {path}")
    return symbols


def backtest_exists(dbm: DatabaseManager, symbol: str, interval: str, source: str) -> bool:
    if dbm.db_type == DatabaseType.JSON_FILE:
        runs = dbm.json_data.get("backtest_runs", [])
        return any((r.get("symbol") == symbol and (r.get("interval") or "") == interval and r.get("source") == source) for r in runs)

    conn = dbm.get_connection()
    cursor = conn.cursor()
    placeholder = dbm.get_sql_placeholder()
    query = f"SELECT COUNT(*) FROM backtest_runs WHERE symbol = {placeholder} AND COALESCE(interval,'') = {placeholder} AND source = {placeholder}"
    params = (symbol, interval, source)
    try:
        cursor.execute(query, params)
        cnt = cursor.fetchone()[0]
        return bool(cnt and cnt > 0)
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        if dbm.db_type == DatabaseType.POSTGRESQL:
            dbm.release_connection(conn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill missing backtests into backtest_runs.")
    parser.add_argument("--symbols-file", type=Path, default=SYMBOLS_PATH, help="مسیر فایل نمادها (هر خط یک نماد)")
    parser.add_argument("--interval", default="1d", help="بازه زمانی (مثال: 1d, 4h)")
    parser.add_argument("--limit", type=int, default=1200, help="تعداد کندل برای بارگذاری")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="حداقل اعتماد به الگو")
    parser.add_argument("--batch-size", type=int, default=50, help="اندازه هر بچ برای گزارش‌دادن")
    parser.add_argument("--source", default="db", help="منبع داده (db یا connector)")
    parser.add_argument("--no-fallback", action="store_true", help="عدم استفاده از fallback در صورت قطع Postgres")
    args = parser.parse_args()

    symbols = read_symbols(args.symbols_file)

    try:
        dbm = DatabaseManager(
            connection_string=settings.database_url,
            allow_fallback=not args.no_fallback,
            auto_setup=True,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        raise SystemExit(f"خطا در راه‌اندازی دیتابیس: {exc}")

    total = len(symbols)
    processed = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    for start in range(0, total, args.batch_size):
        batch = symbols[start : start + args.batch_size]
        print(f"در حال پردازش نمادهای {start+1}-{start+len(batch)} از {total} ...")
        for sym in batch:
            try:
                if backtest_exists(dbm, sym, args.interval, args.source):
                    skipped += 1
                    continue

                run_backtest_with_real_data(
                    symbol=sym,
                    source=args.source,
                    interval=args.interval,
                    limit=args.limit,
                    min_confidence=args.min_confidence,
                    persist=True,
                    db_manager=dbm,
                )
                processed += 1
            except Exception as exc:
                failed.append((sym, str(exc)))
                print(f"⚠️ خطا در {sym}: {exc}")

    print(f"تمام شد. جدید ذخیره شد: {processed}, از قبل موجود بود: {skipped}, ناموفق: {len(failed)}")
    if failed:
        print("نمادهای ناموفق:")
        for sym, err in failed:
            print(f" - {sym}: {err}")


if __name__ == "__main__":
    main()
