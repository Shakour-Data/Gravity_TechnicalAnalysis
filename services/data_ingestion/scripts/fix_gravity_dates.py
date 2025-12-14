"""
Script to fix date parsing issues in gravity_tse.py.
Replaces the double-assignment of Date columns with an explicit
Jalali -> Gregorian conversion so downstream ingestion stays consistent.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_SCRIPT = Path(__file__).resolve().parent / "gravity_tse.py"
TARGET_ENV = "GRAVITY_TSE_SCRIPT"


def main() -> None:
    file_path = Path(os.getenv(TARGET_ENV, DEFAULT_SCRIPT))
    if not file_path.exists():
        raise FileNotFoundError(
            f"gravity_tse.py not found at {file_path}. "
            f"Set {TARGET_ENV} to point at the correct file."
        )

    content = file_path.read_text(encoding="utf-8")
    old_pattern = (
        r"df_sector_cl\['Date'\] = df_sector_cl\.index\s+"
        r"df_sector_cl\['Date'\] = pd\.to_datetime\(df_sector_cl\['Date'\]\)"
    )
    new_text = """df_sector_cl['Date'] = df_sector_cl.index
    # Convert Jalali date strings to Gregorian date objects
    df_sector_cl['Date'] = df_sector_cl['Date'].apply(
        lambda x: jdatetime.datetime.strptime(x, '%Y-%m-%d').togregorian()
    )"""

    content_new = re.sub(old_pattern, new_text, content)
    replacements = len(re.findall(old_pattern, content))

    if replacements == 0:
        print(f"[fix_gravity_dates] ℹ No matches found in {file_path}")
        return

    file_path.write_text(content_new, encoding="utf-8")
    print(f"[fix_gravity_dates] ✓ Updated {replacements} block(s) in {file_path}")


if __name__ == "__main__":
    main()
