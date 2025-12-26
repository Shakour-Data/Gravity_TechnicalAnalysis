"""
Gravity Technical Analysis - Unified Data Pipeline

This module consolidates all ETL operations for the Gravity Technical Analysis system.
It replaces scattered ETL logic from services/data_ingestion and scripts/.

Key components:
- Extractors: Pull data from TSE and other sources
- Transformers: Clean and normalize data
- Validators: Ensure data quality
- Loaders: Persist data to SQLite/PostgreSQL
- Orchestrator: Coordinate the complete pipeline
"""

from gravity_pipeline.orchestrator import DataPipeline, PipelineStage

__version__ = "1.0.0"
__all__ = ["DataPipeline", "PipelineStage"]
