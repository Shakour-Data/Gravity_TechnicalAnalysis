"""
Verify All Setup Scripts Are Compatible with New Schema
تمام اسکریپت‌های setup را برای سازگاری با schema جدید بررسی کن
"""
import re
from pathlib import Path


def check_scripts_compatibility():
    """بررسی سازگاری اسکریپت‌ها"""
    scripts_dir = Path("scripts/setup")
    issues = []
    print("\n" + "="*80)
    print("🔍 CHECKING SCRIPTS COMPATIBILITY")
    print("="*80)
    for script_file in sorted(scripts_dir.glob("*.py")):
        # Skip certain files
        if script_file.name in ["check_schema.py", "check_ind_schema.py", "verify_migration.py", "update_scripts.py"]:
            continue
        try:
            content = script_file.read_text(encoding='utf-8')
            # Check for issues
            script_issues = []
            if "'symbol'" in content and "'ticker'" not in content:
                script_issues.append("  ⚠️ Uses 'symbol' column (should use 'ticker')")
            if "'timestamp'" in content and "'analysis_date'" not in content:
                script_issues.append("  ⚠️ Uses 'timestamp' column (should use 'analysis_date')")
            # Check for TEXT dates (but only in CREATE TABLE statements)
            if "CREATE TABLE" in content:
                if re.search(r"created_at\s+TEXT\s+NOT\s+NULL", content):
                    script_issues.append("  ⚠️ Uses TEXT for created_at (should be DATETIME)")
                if re.search(r"updated_at\s+TEXT\s+NOT\s+NULL", content):
                    script_issues.append("  ⚠️ Uses TEXT for updated_at (should be DATETIME)")
                if re.search(r"analysis_date\s+TEXT\s+NOT\s+NULL", content):
                    script_issues.append("  ⚠️ Uses TEXT for analysis_date (should be DATETIME)")
            # Report
            if script_issues:
                print(f"\n❌ {script_file.name}:")
                for issue in script_issues:
                    print(issue)
                    issues.append((script_file.name, issue))
            else:
                print(f"\n✅ {script_file.name}")
        except Exception as e:
            print(f"\n⚠️ Error reading {script_file.name}: {e}")
    # Summary
    print("\n" + "="*80)
    if issues:
        print(f"🔴 FOUND {len(issues)} COMPATIBILITY ISSUES")
        print("="*80)
        for script, issue in issues:
            print(f"  • {script}: {issue.strip()}")
    else:
        print("✅ ALL SCRIPTS COMPATIBLE WITH NEW SCHEMA")
    print("="*80 + "\n")
    return len(issues) == 0
if __name__ == "__main__":
    success = check_scripts_compatibility()
    exit(0 if success else 1)

