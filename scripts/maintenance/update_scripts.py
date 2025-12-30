"""
Update All Setup Scripts to Use New Schema
تمام اسکریپت‌های setup را به‌روز کن تا از schema جدید استفاده کنند
"""

from pathlib import Path


def update_script_files():
    """تمام اسکریپت‌های setup را به‌روز کن"""

    scripts_dir = Path("scripts/setup")
    replacements = {
        # Column name replacements
        ("'symbol'", "'ticker'"),
        ('"symbol"', '"ticker"'),
        ("'timestamp'", "'analysis_date'"),
        ('"timestamp"', '"analysis_date"'),
        # Type replacements for new scripts
        ("TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP", "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"),
        ("TEXT DEFAULT CURRENT_TIMESTAMP", "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    }

    files_to_skip = [
        "migrate_database.py",  # Already done
        "populate_indicator_values.py",  # Already done
        "verify_migration.py",
        "check_schema.py",
        "check_ind_schema.py",
    ]

    print("\n" + "=" * 80)
    print("🔄 UPDATING SETUP SCRIPTS")
    print("=" * 80)

    for script_file in scripts_dir.glob("*.py"):
        if script_file.name in files_to_skip:
            print(f"\n⏭️  Skipping: {script_file.name}")
            continue

        try:
            content = script_file.read_text(encoding="utf-8")
            original_content = content

            # Apply replacements
            for old, new in replacements:
                if old in content:
                    content = content.replace(old, new)
                    print(f"   ✓ {old} → {new}")

            # Write back if changed
            if content != original_content:
                script_file.write_text(content, encoding="utf-8")
                print(f"\n✅ Updated: {script_file.name}")
            else:
                print(f"\nⓘ No changes needed: {script_file.name}")

        except Exception as e:
            print(f"\n❌ Error updating {script_file.name}: {e}")

    print("\n" + "=" * 80)
    print("✅ SCRIPT UPDATE COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    update_script_files()
