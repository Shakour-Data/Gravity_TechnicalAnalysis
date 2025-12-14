#!/usr/bin/env python3
"""
Compatibility wrapper for the full pipeline runner.

Use scripts/run_full_pipeline.py for the end-to-end flow from the TSE source
database to data/TechAnalysis.db.
"""

from scripts.run_full_pipeline import main


if __name__ == "__main__":
    main()
