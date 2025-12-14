"""
Quick script to reload all indices data
"""
import subprocess
import sys

print("Reloading all indices data...")
print()

# Load market indices
print("=" * 80)
print("LOADING MARKET INDICES")
print("=" * 80)
result = subprocess.run([
    sys.executable, 
    "main.py", 
    "load-market-indices"
], capture_output=False, text=True)

if result.returncode != 0:
    print("❌ Failed to load market indices")
    sys.exit(1)

print()
print("=" * 80)
print("LOADING SECTOR INDICES")
print("=" * 80)

# Load sector indices
result = subprocess.run([
    sys.executable, 
    "main.py", 
    "load-sector-indices"
], capture_output=False, text=True)

if result.returncode != 0:
    print("❌ Failed to load sector indices")
    sys.exit(1)

print()
print("=" * 80)
print("✅ ALL INDICES LOADED SUCCESSFULLY")
print("=" * 80)
