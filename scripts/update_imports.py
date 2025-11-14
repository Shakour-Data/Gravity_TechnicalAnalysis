"""
Script to Update All Imports

Updates all imports from relative to gravity_tech.* absolute imports.

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""
import os
import re
from pathlib import Path

# Base directories - source code AND tests
src_dir = Path(r"E:\Shakour\GravityProjects\GravityTechAnalysis\Gravity_TechAnalysis\src\gravity_tech")
tests_dir = Path(r"E:\Shakour\GravityProjects\GravityTechAnalysis\Gravity_TechAnalysis\tests")

# Modules to update
modules = [
    "config", "api", "middleware", "services", "indicators", 
    "analysis", "patterns", "ml", "database", "utils", "models"
]

def update_imports_in_file(file_path: Path, base_dir: Path):
    """Update imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Update each module import
        for module in modules:
            # Pattern: from module.something import ...
            pattern = rf'^from {module}\.'
            replacement = rf'from gravity_tech.{module}.'
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
            # Pattern: from module import ...
            pattern = rf'^from {module} import'
            replacement = rf'from gravity_tech.{module} import'
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
            # Pattern: import module
            pattern = rf'^import {module}$'
            replacement = rf'import gravity_tech.{module} as {module}'
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        # Only write if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated: {file_path.relative_to(base_dir)}")
            return True
        return False
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def main():
    """Main function to update all Python files"""
    updated_count = 0
    total_count = 0
    
    # Walk through source code
    print("🔧 Updating source code imports...")
    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        
        total_count += 1
        if update_imports_in_file(py_file, src_dir):
            updated_count += 1
    
    # Walk through tests
    print("\n🧪 Updating test imports...")
    for py_file in tests_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        
        total_count += 1
        if update_imports_in_file(py_file, tests_dir):
            updated_count += 1
    
    print(f"\n📊 Summary:")
    print(f"Total Python files: {total_count}")
    print(f"Files updated: {updated_count}")
    print(f"Files unchanged: {total_count - updated_count}")

if __name__ == "__main__":
    main()
