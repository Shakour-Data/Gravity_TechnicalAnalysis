"""
Batch backtest runner.

Reads symbols from data/symbols.txt (one per line) and runs backtests in batches of 50,
persisting results via DatabaseManager (Postgres if available, otherwise SQLite fallback).

Usage (Windows/PowerShell example):
    $env:PYTHONPATH="apps/analysis_api/src"; python scripts/batch_backtest.py --interval 1d --limit 1200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gravity_tech.config.settings import settings
from gravity_tech.database.database_manager import DatabaseManager
from gravity_tech.ml.backtesting import run_backtest_with_real_data


DEFAULT_SYMBOLS_PATH = Path("data/symbols.txt")


def read_symbols(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"فایل نمادها یافت نشد: {path}")
    symbols = [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not symbols:
        raise SystemExit(f"فایل نمادها خالی است: {path}")
    return symbols


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch backtest runner (50 symbols per batch).")
    parser.add_argument("--symbols-file", type=Path, default=DEFAULT_SYMBOLS_PATH, help="فایل نمادها (هر خط یک نماد)")
    parser.add_argument("--interval", default="1d", help="بازه زمانی (مثلا 1d,4h)")
    parser.add_argument("--limit", type=int, default=1200, help="تعداد کندل برای بارگذاری")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="حداقل اعتماد به الگو")
    parser.add_argument("--batch-size", type=int, default=50, help="اندازه هر بچ")
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

    processed = 0
    failed: list[tuple[str, str]] = []
    total = len(symbols)

    for start in range(0, total, args.batch_size):
        batch = symbols[start : start + args.batch_size]
        print(f"در حال پردازش نمادهای {start+1}-{start+len(batch)} از {total} ...")
        for sym in batch:
            try:
                run_backtest_with_real_data(
                    symbol=sym,
                    source="db",
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

    print(f"تمام شد. موفق: {processed}, ناموفق: {len(failed)}")
    if failed:
        print("نمادهای ناموفق:")
        for sym, err in failed:
            print(f" - {sym}: {err}")


if __name__ == "__main__":
    main()

