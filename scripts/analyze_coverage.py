#!/usr/bin/env python
"""
Phase 4: Test Coverage Analysis & Improvement Script

This script:
1. Identifies coverage gaps
2. Generates coverage reports
3. Suggests missing tests
4. Tracks progress toward 80%+ target
"""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List


class CoverageAnalyzer:
    """Analyze test coverage and identify gaps"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.coverage_report = None
        self.gap_analysis = {}
    
    def run_coverage(self) -> bool:
        """Run pytest with coverage"""
        print("📊 Running pytest with coverage...")
        
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/",
                "--cov=apps/analysis_api/src",
                "--cov-report=json",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--tb=short",
                "-v"
            ],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        return result.returncode == 0
    
    def load_coverage_data(self) -> Dict:
        """Load coverage.json"""
        coverage_file = self.project_root / ".coverage.json"
        
        if not coverage_file.exists():
            print("❌ coverage.json not found")
            return {}
        
        try:
            with open(coverage_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading coverage: {e}")
            return {}
    
    def analyze_gaps(self) -> Dict[str, float]:
        """Identify modules with low coverage"""
        data = self.load_coverage_data()
        
        if not data:
            return {}
        
        gaps = {}
        files = data.get("files", {})
        
        for filepath, file_data in files.items():
            summary = file_data.get("summary", {})
            percent_covered = summary.get("percent_covered", 0)
            
            if percent_covered < 70:  # Target is 80%, flag < 70%
                gaps[filepath] = percent_covered
        
        # Sort by coverage percentage (lowest first)
        return dict(sorted(gaps.items(), key=lambda x: x[1]))
    
    def generate_report(self):
        """Generate improvement report"""
        print("\n" + "="*70)
        print("📋 COVERAGE ANALYSIS REPORT")
        print("="*70)
        
        # Load coverage data
        data = self.load_coverage_data()
        if not data:
            print("⚠️  No coverage data available yet")
            return
        
        # Overall coverage
        totals = data.get("totals", {})
        overall_coverage = totals.get("percent_covered", 0)
        
        print(f"\n📊 Overall Coverage: {overall_coverage:.2f}%")
        print(f"   Target: 80%+")
        print(f"   Gap: {80 - overall_coverage:.2f}%")
        
        # Coverage by metric
        print(f"\n📈 Coverage Breakdown:")
        print(f"   Line Coverage: {totals.get('covered_lines', 0)}/{totals.get('num_statements', 0)}")
        print(f"   Branch Coverage: {totals.get('percent_covered_branch', 0):.2f}%")
        
        # Modules with low coverage
        gaps = self.analyze_gaps()
        if gaps:
            print(f"\n⚠️  Modules Below 70% Coverage ({len(gaps)}):")
            for filepath, coverage in list(gaps.items())[:10]:  # Top 10
                rel_path = filepath.replace(str(self.project_root), "")
                print(f"   {rel_path}: {coverage:.2f}%")
        
        # Recommendations
        self.print_recommendations(gaps)
    
    def print_recommendations(self, gaps: Dict):
        """Print test writing recommendations"""
        print(f"\n💡 Recommendations:")
        
        # Categorize modules
        indicators = []
        patterns = []
        services = []
        api = []
        other = []
        
        for filepath in gaps.keys():
            if "indicators" in filepath:
                indicators.append(filepath)
            elif "patterns" in filepath:
                patterns.append(filepath)
            elif "services" in filepath:
                services.append(filepath)
            elif "api" in filepath:
                api.append(filepath)
            else:
                other.append(filepath)
        
        if indicators:
            print(f"\n   1. Add Tests for Technical Indicators ({len(indicators)} modules)")
            print(f"      Files: {', '.join([f.split('/')[-1] for f in indicators[:3]])}")
            print(f"      Tests needed: SMA, EMA, RSI, MACD, Bollinger Bands, ATR")
            print(f"      Estimate: ~200 tests")
        
        if patterns:
            print(f"\n   2. Add Tests for Pattern Detection ({len(patterns)} modules)")
            print(f"      Files: {', '.join([f.split('/')[-1] for f in patterns[:3]])}")
            print(f"      Tests needed: Harmonic, Candlestick, Classical patterns")
            print(f"      Estimate: ~150 tests")
        
        if services:
            print(f"\n   3. Add Tests for Services ({len(services)} modules)")
            print(f"      Files: {', '.join([f.split('/')[-1] for f in services[:3]])}")
            print(f"      Tests needed: Analysis, Tool Recommendation, Pattern services")
            print(f"      Estimate: ~100 tests")
        
        if api:
            print(f"\n   4. Add Tests for API Endpoints ({len(api)} modules)")
            print(f"      Files: {', '.join([f.split('/')[-1] for f in api[:3]])}")
            print(f"      Tests needed: All v1 endpoints with different scenarios")
            print(f"      Estimate: ~80 tests")
    
    def generate_test_templates(self):
        """Generate test templates for modules"""
        gaps = self.analyze_gaps()
        
        print("\n" + "="*70)
        print("📝 TEST TEMPLATES FOR GAP MODULES")
        print("="*70)
        
        for filepath, coverage in list(gaps.items())[:3]:  # Top 3 gaps
            module_name = filepath.split("/")[-1].replace(".py", "")
            
            template = f"""
# Generated test template for {module_name}
# Coverage gap: {coverage:.2f}%

import pytest
from unittest.mock import Mock, patch, AsyncMock

class Test{module_name.title().replace('_', '')}:
    \"\"\"Tests for {module_name}\"\"\"
    
    @pytest.fixture
    def subject(self):
        # TODO: Initialize subject
        pass
    
    def test_basic_functionality(self, subject):
        # TODO: Test basic behavior
        pass
    
    def test_error_handling(self, subject):
        # TODO: Test error cases
        pass
    
    def test_edge_cases(self, subject):
        # TODO: Test boundary conditions
        pass
    
    @pytest.mark.asyncio
    async def test_async_functionality(self, subject):
        # TODO: Test async behavior if applicable
        pass
"""
            print(template)


def main():
    """Main entry point"""
    project_root = Path(__file__).parent.parent
    
    analyzer = CoverageAnalyzer(project_root)
    
    # Run tests
    print("🚀 Phase 4: Test Coverage Analysis\n")
    
    # Option 1: Run coverage analysis
    if "--run" in sys.argv:
        success = analyzer.run_coverage()
        if not success:
            print("⚠️  Some tests failed")
    
    # Option 2: Analyze existing coverage
    if "--analyze" in sys.argv or not sys.argv[1:]:
        analyzer.generate_report()
    
    # Option 3: Generate templates
    if "--templates" in sys.argv:
        analyzer.generate_test_templates()
    
    # Print summary
    print("\n" + "="*70)
    print("📊 NEXT STEPS")
    print("="*70)
    print("""
1. Run coverage analysis:
   python scripts/analyze_coverage.py --run

2. View HTML report:
   htmlcov/index.html

3. Write tests for gap modules:
   - tests/unit/test_indicators.py (200 tests)
   - tests/unit/test_patterns.py (150 tests)
   - tests/integration/test_api_endpoints.py (80 tests)
   - tests/security/test_owasp.py (100+ tests)

4. Target: 80%+ coverage (from 57.85%)

5. Push changes:
   git add -A && git commit -m "test(phase4): Add missing unit/integration/security tests"
   git push origin main
    """)


if __name__ == "__main__":
    main()
