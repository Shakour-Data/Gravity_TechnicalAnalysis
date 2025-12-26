"""
================================================================================
Container Factory Functions
================================================================================

Factory functions for creating services with proper dependency injection.
These are called by the ServiceContainer during initialization.

Pattern: Each factory receives the container and resolves its own dependencies.
"""

from typing import Any
import structlog

logger = structlog.get_logger()


def create_cache_service(container: "ServiceContainer") -> Any:
    """
    Factory for cache service
    
    Args:
        container: ServiceContainer instance
        
    Returns:
        Configured cache instance
    """
    from gravity_tech.config.unified_settings import get_settings
    
    settings = get_settings()
    
    if settings.cache.backend == "redis":
        try:
            from gravity_tech.infrastructure.adapters.redis_cache import RedisCacheAdapter
            cache = RedisCacheAdapter(
                host=settings.cache.host,
                port=settings.cache.port,
                password=settings.cache.password,
                db=settings.cache.db,
            )
            logger.info("cache_created", backend="redis")
            return cache
        except Exception as e:
            logger.warning("redis_cache_failed", error=str(e), fallback="memory")
            # Fall back to memory cache
            from gravity_tech.infrastructure.adapters.memory_cache import MemoryCacheAdapter
            return MemoryCacheAdapter(default_ttl=settings.cache.ttl_seconds)
    
    elif settings.cache.backend == "memory":
        from gravity_tech.infrastructure.adapters.memory_cache import MemoryCacheAdapter
        cache = MemoryCacheAdapter(default_ttl=settings.cache.ttl_seconds)
        logger.info("cache_created", backend="memory")
        return cache
    
    else:
        logger.warning("invalid_cache_backend", backend=settings.cache.backend, default="memory")
        from gravity_tech.infrastructure.adapters.memory_cache import MemoryCacheAdapter
        return MemoryCacheAdapter()


def create_database_service(container: "ServiceContainer") -> Any:
    """
    Factory for database service
    
    Args:
        container: ServiceContainer instance
        
    Returns:
        Configured database instance
    """
    from gravity_tech.config.unified_settings import get_settings
    
    settings = get_settings()
    
    if settings.database.engine == "postgresql":
        try:
            from gravity_tech.infrastructure.adapters.postgres_db import PostgresDatabaseAdapter
            db = PostgresDatabaseAdapter(url=settings.database.url)
            logger.info("database_created", engine="postgresql")
            return db
        except Exception as e:
            logger.error("postgres_db_failed", error=str(e))
            raise
    
    elif settings.database.engine == "sqlite":
        from gravity_tech.infrastructure.adapters.sqlite_db import SqliteDatabaseAdapter
        db = SqliteDatabaseAdapter(url=settings.database.url)
        logger.info("database_created", engine="sqlite")
        return db
    
    else:
        raise ValueError(f"Unsupported database engine: {settings.database.engine}")


def create_analysis_service(container: "ServiceContainer") -> Any:
    """
    Factory for analysis service
    
    Args:
        container: ServiceContainer instance
        
    Returns:
        Configured analysis service
    """
    from gravity_tech.services.analysis_service import TechnicalAnalysisService
    
    cache = container.get("cache")
    
    service = TechnicalAnalysisService(cache=cache)
    logger.info("analysis_service_created")
    return service


def create_tool_recommendation_service(container: "ServiceContainer") -> Any:
    """
    Factory for tool recommendation service
    
    Args:
        container: ServiceContainer instance
        
    Returns:
        Configured tool recommendation service
    """
    from gravity_tech.services.tool_recommendation_service import ToolRecommendationService
    
    cache = container.get("cache")
    
    service = ToolRecommendationService(cache=cache)
    logger.info("tool_recommendation_service_created")
    return service


def create_data_ingestor_service(container: "ServiceContainer") -> Any:
    """
    Factory for data ingestor service
    
    Args:
        container: ServiceContainer instance
        
    Returns:
        Configured data ingestor service
    """
    from gravity_tech.services.data_ingestor_service import DataIngestorService
    from gravity_tech.config.unified_settings import get_settings
    
    settings = get_settings()
    database = container.get("database")
    
    service = DataIngestorService(
        database=database,
        enabled=settings.features.enable_data_ingestion
    )
    logger.info("data_ingestor_service_created")
    return service


def create_event_publisher(container: "ServiceContainer") -> Any:
    """
    Factory for event publisher
    
    Args:
        container: ServiceContainer instance
        
    Returns:
        Configured event publisher
    """
    from gravity_tech.config.unified_settings import get_settings
    
    settings = get_settings()
    
    if settings.features.kafka_enabled:
        try:
            from gravity_tech.infrastructure.adapters.kafka_publisher import KafkaPublisher
            publisher = KafkaPublisher()
            logger.info("event_publisher_created", backend="kafka")
            return publisher
        except Exception as e:
            logger.warning("kafka_publisher_failed", error=str(e))
    
    if settings.features.rabbitmq_enabled:
        try:
            from gravity_tech.infrastructure.adapters.rabbitmq_publisher import RabbitMQPublisher
            publisher = RabbitMQPublisher()
            logger.info("event_publisher_created", backend="rabbitmq")
            return publisher
        except Exception as e:
            logger.warning("rabbitmq_publisher_failed", error=str(e))
    
    # Fallback to no-op publisher
    from gravity_tech.infrastructure.adapters.noop_publisher import NoOpPublisher
    logger.info("event_publisher_created", backend="noop")
    return NoOpPublisher()
