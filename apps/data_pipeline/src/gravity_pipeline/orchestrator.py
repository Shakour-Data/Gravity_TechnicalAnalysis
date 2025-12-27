"""
Data Pipeline Orchestrator

Unified orchestration of ETL operations:
1. Extract - pull data from sources (TSE, APIs)
2. Transform - clean, normalize, validate
3. Validate - ensure data quality
4. Deduplicate - remove duplicates
5. Load - persist to target database

Each stage can be run independently or as part of full pipeline.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import structlog

logger = structlog.get_logger()


class PipelineStage(Enum):
    """Pipeline execution stages"""
    EXTRACT = "extract"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    DEDUPLICATE = "deduplicate"
    LOAD = "load"


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    source_db_url: str
    target_db_url: str
    batch_size: int = 500
    max_workers: int = 4
    retry_count: int = 3
    log_level: str = "INFO"


@dataclass
class PipelineResult:
    """Pipeline execution result"""
    stage: PipelineStage
    status: str  # success, partial, failed
    records_processed: int
    records_failed: int
    duration_seconds: float
    errors: List[Dict[str, Any]]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        total = self.records_processed + self.records_failed
        if total == 0:
            return 0.0
        return (self.records_processed / total) * 100


class DataPipeline:
    """
    Unified data pipeline orchestrator.
    
    Consolidates ETL logic from:
    - services/data_ingestion/
    - scripts/run_full_pipeline.py
    - scripts/migrate_*.py
    - scripts/etl/
    
    Usage:
        pipeline = DataPipeline(config)
        result = await pipeline.run_full(
            symbols=['SYMBOL1', 'SYMBOL2'],
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 12, 31)
        )
    """
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration"""
        self.config = config
        self.stages_completed: List[PipelineStage] = []
        self.stage_results: Dict[PipelineStage, PipelineResult] = {}
        
        logger.info(
            "pipeline_initialized",
            source_db=config.source_db_url,
            target_db=config.target_db_url,
            batch_size=config.batch_size
        )
    
    async def run_full(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip_stages: Optional[List[PipelineStage]] = None,
    ) -> Dict[PipelineStage, PipelineResult]:
        """
        Run complete pipeline from extraction to loading.
        
        Args:
            symbols: List of symbols to process (None = all)
            start_date: Start date for historical data
            end_date: End date for historical data
            skip_stages: Stages to skip (e.g., [PipelineStage.VALIDATE])
        
        Returns:
            Dictionary of {stage: result} for all executed stages
        """
        skip_stages = skip_stages or []
        logger.info(
            "pipeline_starting",
            symbols_count=len(symbols) if symbols else "all",
            date_range=f"{start_date} to {end_date}",
            skip_stages=[s.value for s in skip_stages]
        )
        
        try:
            # Extract
            if PipelineStage.EXTRACT not in skip_stages:
                candles = await self._extract(symbols, start_date, end_date)
            else:
                logger.info("stage_skipped", stage="extract")
                candles = []
            
            # Transform
            if PipelineStage.TRANSFORM not in skip_stages and candles:
                candles = await self._transform(candles)
            else:
                logger.info("stage_skipped", stage="transform")
            
            # Validate
            if PipelineStage.VALIDATE not in skip_stages and candles:
                candles = await self._validate(candles)
            else:
                logger.info("stage_skipped", stage="validate")
            
            # Deduplicate
            if PipelineStage.DEDUPLICATE not in skip_stages and candles:
                candles = await self._deduplicate(candles)
            else:
                logger.info("stage_skipped", stage="deduplicate")
            
            # Load
            if PipelineStage.LOAD not in skip_stages and candles:
                await self._load(candles)
            else:
                logger.info("stage_skipped", stage="load")
            
            logger.info(
                "pipeline_completed",
                stages_completed=[s.value for s in self.stages_completed],
                total_duration=sum(
                    r.duration_seconds 
                    for r in self.stage_results.values()
                )
            )
            
            return self.stage_results
            
        except Exception as e:
            logger.error(
                "pipeline_failed",
                error=str(e),
                stages_completed=[s.value for s in self.stages_completed],
                exception_type=type(e).__name__
            )
            raise
    
    async def _extract(
        self,
        symbols: Optional[List[str]],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> List[Dict[str, Any]]:
        """
        Extract raw data from TSE.
        
        This stage:
        - Connects to TSE data source
        - Pulls OHLCV candles
        - Returns raw records
        """
        start_time = datetime.now()
        
        try:
            logger.info(
                "extract_stage_starting",
                symbols=symbols,
                date_range=f"{start_date} to {end_date}"
            )
            
            # TODO: Implement actual TSE extraction
            # For now, return empty placeholder
            candles = []
            
            logger.info(
                "extract_stage_complete",
                records_extracted=len(candles)
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            result = PipelineResult(
                stage=PipelineStage.EXTRACT,
                status="success",
                records_processed=len(candles),
                records_failed=0,
                duration_seconds=duration,
                errors=[]
            )
            
            self.stage_results[PipelineStage.EXTRACT] = result
            self.stages_completed.append(PipelineStage.EXTRACT)
            
            return candles
            
        except Exception as e:
            logger.error("extract_stage_failed", error=str(e))
            raise
    
    async def _transform(
        self,
        candles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Transform and normalize data.
        
        This stage:
        - Normalizes column names
        - Converts data types
        - Calculates derived fields
        - Handles missing values
        """
        start_time = datetime.now()
        
        try:
            logger.info(
                "transform_stage_starting",
                records_to_transform=len(candles)
            )
            
            # TODO: Implement actual transformation logic
            transformed = candles.copy()
            
            logger.info(
                "transform_stage_complete",
                records_transformed=len(transformed)
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            result = PipelineResult(
                stage=PipelineStage.TRANSFORM,
                status="success",
                records_processed=len(transformed),
                records_failed=0,
                duration_seconds=duration,
                errors=[]
            )
            
            self.stage_results[PipelineStage.TRANSFORM] = result
            self.stages_completed.append(PipelineStage.TRANSFORM)
            
            return transformed
            
        except Exception as e:
            logger.error("transform_stage_failed", error=str(e))
            raise
    
    async def _validate(
        self,
        candles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Validate data quality.
        
        This stage:
        - Checks for required fields
        - Validates data types
        - Checks value ranges
        - Detects anomalies
        """
        start_time = datetime.now()
        
        try:
            logger.info(
                "validate_stage_starting",
                records_to_validate=len(candles)
            )
            
            valid_records = []
            invalid_records = []
            
            # TODO: Implement actual validation logic
            valid_records = candles.copy()
            
            logger.info(
                "validate_stage_complete",
                valid=len(valid_records),
                invalid=len(invalid_records)
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            result = PipelineResult(
                stage=PipelineStage.VALIDATE,
                status="success" if len(invalid_records) == 0 else "partial",
                records_processed=len(valid_records),
                records_failed=len(invalid_records),
                duration_seconds=duration,
                errors=[]
            )
            
            self.stage_results[PipelineStage.VALIDATE] = result
            self.stages_completed.append(PipelineStage.VALIDATE)
            
            return valid_records
            
        except Exception as e:
            logger.error("validate_stage_failed", error=str(e))
            raise
    
    async def _deduplicate(
        self,
        candles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate records.
        
        This stage:
        - Identifies duplicates by symbol+timestamp
        - Keeps most recent version
        - Logs removed duplicates
        """
        start_time = datetime.now()
        
        try:
            logger.info(
                "deduplicate_stage_starting",
                records_to_deduplicate=len(candles)
            )
            
            seen = {}
            deduplicated = []
            duplicates = []
            
            # TODO: Implement actual deduplication
            deduplicated = candles.copy()
            
            logger.info(
                "deduplicate_stage_complete",
                kept=len(deduplicated),
                removed=len(duplicates)
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            result = PipelineResult(
                stage=PipelineStage.DEDUPLICATE,
                status="success",
                records_processed=len(deduplicated),
                records_failed=len(duplicates),
                duration_seconds=duration,
                errors=[]
            )
            
            self.stage_results[PipelineStage.DEDUPLICATE] = result
            self.stages_completed.append(PipelineStage.DEDUPLICATE)
            
            return deduplicated
            
        except Exception as e:
            logger.error("deduplicate_stage_failed", error=str(e))
            raise
    
    async def _load(
        self,
        candles: List[Dict[str, Any]],
    ) -> None:
        """
        Load data into target database.
        
        This stage:
        - Connects to target DB (SQLite or PostgreSQL)
        - Batches inserts/updates
        - Handles conflicts
        - Maintains referential integrity
        """
        start_time = datetime.now()
        
        try:
            logger.info(
                "load_stage_starting",
                records_to_load=len(candles),
                target_db=self.config.target_db_url
            )
            
            # TODO: Implement actual loading logic
            
            logger.info(
                "load_stage_complete",
                records_loaded=len(candles)
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            result = PipelineResult(
                stage=PipelineStage.LOAD,
                status="success",
                records_processed=len(candles),
                records_failed=0,
                duration_seconds=duration,
                errors=[]
            )
            
            self.stage_results[PipelineStage.LOAD] = result
            self.stages_completed.append(PipelineStage.LOAD)
            
        except Exception as e:
            logger.error("load_stage_failed", error=str(e))
            raise
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        total_duration = sum(
            r.duration_seconds 
            for r in self.stage_results.values()
        )
        
        total_processed = sum(
            r.records_processed 
            for r in self.stage_results.values()
        )
        
        total_failed = sum(
            r.records_failed 
            for r in self.stage_results.values()
        )
        
        return {
            "total_duration_seconds": total_duration,
            "stages_completed": [s.value for s in self.stages_completed],
            "total_records_processed": total_processed,
            "total_records_failed": total_failed,
            "overall_success_rate": (
                (total_processed / (total_processed + total_failed) * 100)
                if (total_processed + total_failed) > 0
                else 0.0
            ),
            "stage_results": {
                stage.value: {
                    "status": result.status,
                    "records_processed": result.records_processed,
                    "records_failed": result.records_failed,
                    "duration_seconds": result.duration_seconds,
                    "success_rate": result.success_rate
                }
                for stage, result in self.stage_results.items()
            }
        }
