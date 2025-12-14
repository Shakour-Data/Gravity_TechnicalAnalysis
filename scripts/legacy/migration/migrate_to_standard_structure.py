"""
Migration script to reorganize Gravity Technical Analysis project structure.

This script will:
1. Create new standard directory structure
2. Move files to their proper locations
3. Update import statements
4. Generate migration report

Usage:
    python scripts/migration/migrate_to_standard_structure.py --dry-run
    python scripts/migration/migrate_to_standard_structure.py --execute
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import re


class StructureMigration:
    """Handle project structure migration."""

    def __init__(self, project_root: Path, dry_run: bool = True):
        self.project_root = project_root
        self.dry_run = dry_run
        self.operations: List[Dict] = []
        self.errors: List[str] = []

    def create_standard_structure(self):
        """Create the standard directory structure."""
        directories = [
            # Documentation
            "docs/en/getting-started",
            "docs/en/api",
            "docs/en/architecture",
            "docs/en/deployment",
            "docs/fa/getting-started",
            "docs/fa/guides",
            "docs/fa/api",
            "docs/fa/tutorials",
            "docs/changelog/releases",
            "docs/diagrams",
            
            # Source code structure
            "src/gravity_tech/api/routers",
            "src/gravity_tech/api/middleware",
            "src/gravity_tech/core/domain",
            "src/gravity_tech/core/indicators/trend",
            "src/gravity_tech/core/indicators/momentum",
            "src/gravity_tech/core/indicators/volatility",
            "src/gravity_tech/core/indicators/volume",
            "src/gravity_tech/core/indicators/cycle",
            "src/gravity_tech/core/patterns/candlestick",
            "src/gravity_tech/core/patterns/chart",
            "src/gravity_tech/core/patterns/harmonic",
            "src/gravity_tech/core/analysis",
            "src/gravity_tech/ml/models",
            "src/gravity_tech/ml/features",
            "src/gravity_tech/ml/training",
            "src/gravity_tech/ml/inference",
            "src/gravity_tech/data/database/repositories",
            "src/gravity_tech/data/database/migrations",
            "src/gravity_tech/data/cache",
            "src/gravity_tech/data/connectors",
            "src/gravity_tech/services",
            "src/gravity_tech/config",
            "src/gravity_tech/utils",
            
            # Tests structure
            "tests/unit/core",
            "tests/unit/ml",
            "tests/unit/utils",
            "tests/integration",
            "tests/e2e",
            "tests/performance",
            "tests/accuracy",
            
            # Scripts
            "scripts/setup",
            "scripts/migration",
            "scripts/deployment",
            "scripts/maintenance",
            
            # Deployment
            "deployment/docker",
            "deployment/kubernetes/base",
            "deployment/kubernetes/overlays/dev",
            "deployment/kubernetes/overlays/staging",
            "deployment/kubernetes/overlays/production",
            "deployment/kubernetes/helm",
            "deployment/terraform",
            
            # Config
            "configs",
            
            # Data (with .gitkeep)
            "data/raw",
            "data/processed",
            "data/models",
        ]

        for directory in directories:
            dir_path = self.project_root / directory
            if not self.dry_run:
                dir_path.mkdir(parents=True, exist_ok=True)
                # Create .gitkeep for empty directories
                if "data" in directory:
                    (dir_path / ".gitkeep").touch()
            self.operations.append({
                "type": "create_dir",
                "path": str(dir_path),
                "status": "done" if not self.dry_run else "planned"
            })

    def move_documentation(self):
        """Reorganize documentation files."""
        doc_moves = [
            # Persian guides
            ("TREND_ANALYSIS_GUIDE.md", "docs/fa/guides/trend_analysis.md"),
            ("MOMENTUM_ANALYSIS_GUIDE.md", "docs/fa/guides/momentum_analysis.md"),
            ("VOLATILITY_ANALYSIS_GUIDE.md", "docs/fa/guides/volatility_analysis.md"),
            ("CYCLE_ANALYSIS_GUIDE.md", "docs/fa/guides/cycle_analysis.md"),
            ("SUPPORT_RESISTANCE_GUIDE.md", "docs/fa/guides/support_resistance.md"),
            ("VOLUME_MATRIX_GUIDE.md", "docs/fa/guides/volume_matrix.md"),
            ("FIVE_DIMENSIONAL_DECISION_GUIDE.md", "docs/fa/guides/five_dimensional_decision.md"),
            
            # Getting started
            ("docs/QUICKSTART.md", "docs/fa/getting-started/quickstart.md"),
            
            # Changelog
            ("CHANGELOG.md", "docs/changelog/CHANGELOG.md"),
            ("docs/CHANGELOG.md", "docs/changelog/CHANGELOG.md"),
        ]

        for source, destination in doc_moves:
            self._move_file(source, destination)

    def move_deployment_files(self):
        """Move deployment related files."""
        deployment_moves = [
            ("Dockerfile", "deployment/docker/Dockerfile"),
            ("docker-compose.yml", "deployment/docker/docker-compose.yml"),
        ]

        for source, destination in deployment_moves:
            self._move_file(source, destination)

        # Move k8s directory
        if (self.project_root / "k8s").exists():
            self._move_directory("k8s", "deployment/kubernetes/base")

        # Move helm directory
        if (self.project_root / "helm").exists():
            self._move_directory("helm", "deployment/kubernetes/helm")

    def move_scripts(self):
        """Reorganize scripts."""
        script_moves = [
            ("setup_database.py", "scripts/setup/init_database.py"),
            ("scripts/init_db.py", "scripts/setup/init_db.py"),
            ("scripts/backup_manager.py", "scripts/maintenance/backup.py"),
            ("scripts/optimize_database.py", "scripts/maintenance/optimize_db.py"),
        ]

        for source, destination in script_moves:
            source_path = self.project_root / source
            if source_path.exists():
                self._move_file(source, destination)

    def consolidate_source_code(self):
        """Consolidate source code to single location."""
        print("\n⚠️  Manual consolidation required:")
        print("   - Merge src/core/ into src/gravity_tech/core/")
        print("   - Merge ml/ into src/gravity_tech/ml/")
        print("   - Update all import statements")
        print("   - Remove duplicate code")

    def update_pyproject_toml(self):
        """Update pyproject.toml with new structure."""
        pyproject_path = self.project_root / "pyproject.toml"
        
        print("\n📝 Update pyproject.toml:")
        print("   - Verify [tool.setuptools.package-dir]")
        print("   - Check package discovery settings")

    def _move_file(self, source: str, destination: str):
        """Move a file from source to destination."""
        source_path = self.project_root / source
        dest_path = self.project_root / destination

        if not source_path.exists():
            self.operations.append({
                "type": "move_file",
                "source": str(source_path),
                "destination": str(dest_path),
                "status": "skipped (source not found)"
            })
            return

        if not self.dry_run:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(dest_path))

        self.operations.append({
            "type": "move_file",
            "source": str(source_path),
            "destination": str(dest_path),
            "status": "done" if not self.dry_run else "planned"
        })

    def _move_directory(self, source: str, destination: str):
        """Move entire directory."""
        source_path = self.project_root / source
        dest_path = self.project_root / destination

        if not source_path.exists():
            return

        if not self.dry_run:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_path), str(dest_path))

        self.operations.append({
            "type": "move_dir",
            "source": str(source_path),
            "destination": str(dest_path),
            "status": "done" if not self.dry_run else "planned"
        })

    def generate_report(self):
        """Generate migration report."""
        report_path = self.project_root / "MIGRATION_REPORT.md"
        
        report = [
            "# 📋 Migration Report",
            f"\nMode: {'DRY RUN' if self.dry_run else 'EXECUTION'}",
            f"\nTotal operations: {len(self.operations)}",
            "\n## Operations\n"
        ]

        # Group by type
        creates = [op for op in self.operations if op["type"] == "create_dir"]
        moves = [op for op in self.operations if op["type"].startswith("move")]

        report.append(f"### Directories Created ({len(creates)})\n")
        for op in creates[:10]:  # Show first 10
            report.append(f"- {op['path']}")
        if len(creates) > 10:
            report.append(f"- ... and {len(creates) - 10} more")

        report.append(f"\n### Files/Dirs Moved ({len(moves)})\n")
        for op in moves:
            report.append(f"- {op['source']} → {op['destination']} ({op['status']})")

        if self.errors:
            report.append("\n## ⚠️ Errors\n")
            for error in self.errors:
                report.append(f"- {error}")

        report.append("\n## 📝 Manual Steps Required\n")
        report.append("""
1. **Consolidate source code:**
   - Merge `src/core/` into `src/gravity_tech/core/`
   - Merge `ml/` into `src/gravity_tech/ml/`
   - Remove duplicates

2. **Update imports:**
   ```bash
   # Find all Python files and update imports
   find src -name "*.py" -exec sed -i 's/from ml\\./from gravity_tech.ml./g' {} +
   find src -name "*.py" -exec sed -i 's/from core\\./from gravity_tech.core./g' {} +
   ```

3. **Update tests:**
   - Reorganize tests by type (unit/integration/e2e)
   - Update import statements in tests

4. **Verify configuration:**
   - Check `pyproject.toml`
   - Update paths in `docker-compose.yml`
   - Update Kubernetes manifests

5. **Run tests:**
   ```bash
   make test
   ```

6. **Update CI/CD:**
   - Update GitHub Actions workflows
   - Update deployment scripts
""")

        report_content = "\n".join(report)
        
        if not self.dry_run:
            report_path.write_text(report_content, encoding="utf-8")
            print(f"\n✅ Report saved to: {report_path}")
        else:
            print(report_content)

    def run(self):
        """Run the migration."""
        print("🚀 Starting migration...")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'EXECUTION'}")
        print(f"Project root: {self.project_root}\n")

        try:
            self.create_standard_structure()
            print("✓ Standard structure created/planned")

            self.move_documentation()
            print("✓ Documentation reorganized")

            self.move_deployment_files()
            print("✓ Deployment files moved")

            self.move_scripts()
            print("✓ Scripts reorganized")

            self.consolidate_source_code()
            self.update_pyproject_toml()

            self.generate_report()

        except Exception as e:
            self.errors.append(str(e))
            print(f"\n❌ Error: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Gravity Technical Analysis to standard structure"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the migration (default is dry-run)"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )

    args = parser.parse_args()

    # Confirm execution
    if args.execute:
        print("⚠️  WARNING: This will modify your project structure!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("Migration cancelled.")
            return

    migration = StructureMigration(
        project_root=args.project_root,
        dry_run=not args.execute
    )
    
    migration.run()

    if not args.execute:
        print("\n💡 This was a dry run. Use --execute to apply changes.")


if __name__ == "__main__":
    main()
