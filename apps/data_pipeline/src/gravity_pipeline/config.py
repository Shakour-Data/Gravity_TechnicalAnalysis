"""Pipeline-specific configuration"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineEnvironmentConfig:
    """Pipeline configuration from environment variables"""
    
    # Data sources
    TSE_API_BASE_URL: str = os.getenv(
        "TSE_API_BASE_URL",
        "https://api.tse.ir/api"
    )
    
    # Database URLs
    SOURCE_DB_URL: str = os.getenv(
        "SOURCE_DB_URL",
        "sqlite:///./data/gravity_source.db"
    )
    
    TARGET_DB_URL: str = os.getenv(
        "TARGET_DB_URL",
        "sqlite:///./data/gravity.db"
    )
    
    # Pipeline settings
    BATCH_SIZE: int = int(os.getenv("PIPELINE_BATCH_SIZE", "500"))
    MAX_WORKERS: int = int(os.getenv("PIPELINE_MAX_WORKERS", "4"))
    RETRY_COUNT: int = int(os.getenv("PIPELINE_RETRY_COUNT", "3"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("PIPELINE_LOG_LEVEL", "INFO")
    
    # Optional: API credentials
    TSE_API_KEY: Optional[str] = os.getenv("TSE_API_KEY")
    
    @property
    def as_dict(self) -> dict:
        """Convert to dictionary for pipeline initialization"""
        return {
            "source_db_url": self.SOURCE_DB_URL,
            "target_db_url": self.TARGET_DB_URL,
            "batch_size": self.BATCH_SIZE,
            "max_workers": self.MAX_WORKERS,
            "retry_count": self.RETRY_COUNT,
            "log_level": self.LOG_LEVEL
        }
