#!/usr/bin/env python3
"""Remove Redis availability checks from test file."""

file_path = "tests/unit/services/test_cache_service_real.py"

with open(file_path) as f:
    lines = f.readlines()

# Filter out lines that contain pytest.skip and the if statement
new_lines = []
skip_next = False

for line in lines:
    # Check if this is the if statement
    if "if not cache._is_available:" in line:
        skip_next = True
        continue

    # Skip the pytest.skip line
    if skip_next and "pytest.skip" in line:
        skip_next = False
        continue

    new_lines.append(line)

with open(file_path, "w") as f:
    f.writelines(new_lines)

print(f"✅ Removed Redis availability checks from {file_path}")
