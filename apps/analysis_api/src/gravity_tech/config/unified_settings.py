"""
================================================================================
Unified Settings Management for Gravity Technical Analysis
================================================================================

Single source of truth for all configuration across environments.
Load from environment variables with sensible defaults.

Usage:
    from gravity_tech.config.unified_settings import get_settings
    settings = get_settings()
    print(settings.database.url)
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import structlog

logger = structlog.get_logger()


class Environment(str, Enum):
    """Supported environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    engine: str = "sqlite"  # sqlite, postgresql, mysql
    url: str = "sqlite:///./gravity.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    timeout: int = 30
    
    def __post_init__(self):
        if not self.url:
            raise ValueError("DATABASE_URL is required")


@dataclass
class CacheConfig:
    """Cache backend configuration"""
    enabled: bool = True
    backend: str = "memory"  # redis, memory, none
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    ttl_seconds: int = 300
    key_prefix: str = "gravity:"
    connection_timeout: int = 5
    
    def __post_init__(self):
        if self.backend not in ["redis", "memory", "none"]:
            raise ValueError(f"Invalid cache backend: {self.backend}")


@dataclass
class MLConfig:
    """Machine Learning configuration"""
    enabled: bool = True
    model_path: str = "ml_models/"
    pattern_classifier_enabled: bool = True
    pattern_classifier_path: str = "ml_models/pattern_classifier_btcusdt.pkl"
    weight_optimizer_enabled: bool = False
    gpu_enabled: bool = False
    inference_timeout: int = 30
    batch_size: int = 32


@dataclass
class FeatureFlags:
    """Feature flags - centralized"""
    # Analysis features
    enable_scenarios: bool = False
    enable_harmonic_patterns: bool = True
    enable_multi_horizon: bool = True
    enable_elliott_waves: bool = True
    
    # API features
    expose_db_explorer: bool = False
    expose_metrics: bool = True
    enable_api_caching: bool = True
    
    # Data ingestion
    enable_data_ingestion: bool = True
    enable_ingestion_validation: bool = True
    enable_data_deduplication: bool = True
    
    # ML features
    enable_ml_inference: bool = True
    enable_ml_caching: bool = True
    
    # Infrastructure
    eureka_enabled: bool = False
    kafka_enabled: bool = False
    rabbitmq_enabled: bool = False
    
    # Observability
    metrics_enabled: bool = True
    tracing_enabled: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"  # json, text
    file_enabled: bool = False
    file_path: str = "logs/gravity.log"
    file_max_bytes: int = 10485760  # 10MB
    file_backup_count: int = 5


@dataclass
class ObservabilityConfig:
    """Observability configuration"""
    metrics_enabled: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    tracing_enabled: bool = False
    tracing_service_name: str = "gravity-api"
    tracing_sample_rate: float = 0.1


@dataclass
class Settings:
    """Master settings object - single source of truth"""
    
    # Application
    environment: Environment = Environment.DEVELOPMENT
    app_name: str = "Gravity Technical Analysis"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Components
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = False
    api_title: str = "Gravity Technical Analysis API"
    api_docs_enabled: bool = True
    
    # Security
    jwt_secret: str = "dev-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 1000
    rate_limit_period_seconds: int = 60
    
    # External Services
    data_service_base_url: Optional[str] = None
    data_service_timeout: int = 30
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables"""
        
        # Environment
        env_str = os.getenv("ENVIRONMENT", "development").lower()
        try:
            environment = Environment(env_str)
        except ValueError:
            logger.warning("invalid_environment", env=env_str, default="development")
            environment = Environment.DEVELOPMENT
        
        # Database
        db_url = os.getenv("DATABASE_URL", "sqlite:///./gravity.db")
        db_engine = os.getenv("DB_ENGINE", "sqlite")
        
        database = DatabaseConfig(
            engine=db_engine,
            url=db_url,
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            timeout=int(os.getenv("DB_TIMEOUT", "30")),
        )
        
        # Cache
        cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        cache_backend = os.getenv("CACHE_BACKEND", "memory")
        cache_host = os.getenv("REDIS_HOST", "localhost")
        cache_port = int(os.getenv("REDIS_PORT", "6379"))
        cache_password = os.getenv("REDIS_PASSWORD")
        cache_db = int(os.getenv("REDIS_DB", "0"))
        
        cache = CacheConfig(
            enabled=cache_enabled,
            backend=cache_backend,
            host=cache_host,
            port=cache_port,
            password=cache_password,
            db=cache_db,
            ttl_seconds=int(os.getenv("CACHE_TTL", "300")),
            key_prefix=os.getenv("CACHE_KEY_PREFIX", "gravity:"),
        )
        
        # ML
        ml = MLConfig(
            enabled=os.getenv("ML_ENABLED", "true").lower() == "true",
            model_path=os.getenv("ML_MODEL_PATH", "ml_models/"),
            pattern_classifier_enabled=os.getenv("ML_PATTERN_CLASSIFIER_ENABLED", "true").lower() == "true",
            pattern_classifier_path=os.getenv("ML_PATTERN_CLASSIFIER_PATH", "ml_models/pattern_classifier_btcusdt.pkl"),
            gpu_enabled=os.getenv("ML_GPU_ENABLED", "false").lower() == "true",
        )
        
        # Feature Flags
        features = FeatureFlags(
            enable_scenarios=os.getenv("ENABLE_SCENARIOS", "false").lower() == "true",
            expose_db_explorer=os.getenv("EXPOSE_DB_EXPLORER", "false").lower() == "true",
            enable_harmonic_patterns=os.getenv("ENABLE_HARMONIC_PATTERNS", "true").lower() == "true",
            eureka_enabled=os.getenv("EUREKA_ENABLED", "false").lower() == "true",
            kafka_enabled=os.getenv("KAFKA_ENABLED", "false").lower() == "true",
            rabbitmq_enabled=os.getenv("RABBITMQ_ENABLED", "false").lower() == "true",
            metrics_enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
            tracing_enabled=os.getenv("TRACING_ENABLED", "false").lower() == "true",
        )
        
        # Logging
        logging_config = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "json"),
            file_enabled=os.getenv("LOG_FILE_ENABLED", "false").lower() == "true",
            file_path=os.getenv("LOG_FILE_PATH", "logs/gravity.log"),
        )
        
        # Observability
        observability = ObservabilityConfig(
            metrics_enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
            metrics_port=int(os.getenv("METRICS_PORT", "9090")),
            tracing_enabled=os.getenv("TRACING_ENABLED", "false").lower() == "true",
        )
        
        # Security
        jwt_secret = os.getenv("JWT_SECRET")
        if not jwt_secret and environment == Environment.PRODUCTION:
            raise ValueError("JWT_SECRET required in production")
        
        cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
        
        settings = cls(
            environment=environment,
            app_version=os.getenv("APP_VERSION", "1.0.0"),
            debug=environment == Environment.DEVELOPMENT,
            database=database,
            cache=cache,
            ml=ml,
            features=features,
            logging=logging_config,
            observability=observability,
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
            api_workers=int(os.getenv("API_WORKERS", "4")),
            api_reload=environment == Environment.DEVELOPMENT,
            jwt_secret=jwt_secret or "dev-secret-key-change-in-production",
            cors_origins=cors_origins,
            data_service_base_url=os.getenv("DATA_SERVICE_BASE_URL"),
            rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "1000")),
            rate_limit_period_seconds=int(os.getenv("RATE_LIMIT_PERIOD", "60")),
        )
        
        logger.info(
            "settings_loaded",
            environment=environment.value,
            db_engine=db_engine,
            cache_backend=cache_backend,
        )
        
        return settings


# Global instance (lazy loaded)
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Get singleton settings instance"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings.from_env()
    return _settings_instance


def reset_settings():
    """Reset settings instance (for testing)"""
    global _settings_instance
    _settings_instance = None


if __name__ == "__main__":
    # Test loading
    settings = Settings.from_env()
    print(f"✅ Settings loaded successfully")
    print(f"   Environment: {settings.environment.value}")
    print(f"   Database: {settings.database.engine}")
    print(f"   Cache: {settings.cache.backend}")
    print(f"   ML enabled: {settings.ml.enabled}")
