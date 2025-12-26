"""
================================================================================
Dependency Injection Container
================================================================================

ServiceContainer provides centralized dependency injection for all services.
Replaces global singletons with a managed container.

Usage:
    container = get_container()
    service = container.get("analysis_service")
    
For testing:
    test_container = create_test_container()
    test_container.register("cache", mock_cache)
"""

from typing import Dict, Any, Callable, Optional, Type
import structlog
from abc import ABC, abstractmethod

logger = structlog.get_logger()


class ServiceFactory(ABC):
    """Base factory for service creation"""
    
    @abstractmethod
    def create(self, container: "ServiceContainer") -> Any:
        """Create service instance"""
        pass


class ServiceContainer:
    """
    Dependency Injection Container
    
    Manages service lifecycle and dependencies.
    Supports singleton and transient instances.
    """
    
    def __init__(self):
        """Initialize container"""
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._is_singleton: Dict[str, bool] = {}
        logger.info("container_created")
    
    def register(
        self,
        name: str,
        factory: Callable[["ServiceContainer"], Any],
        singleton: bool = False
    ) -> None:
        """
        Register a service factory
        
        Args:
            name: Service name
            factory: Callable that creates the service, receives container as arg
            singleton: If True, service is created once and reused
        """
        if name in self._factories:
            logger.warning("service_already_registered", name=name)
        
        self._factories[name] = factory
        self._is_singleton[name] = singleton
        logger.info("service_registered", name=name, singleton=singleton)
    
    def get(self, name: str) -> Any:
        """
        Get service instance
        
        Args:
            name: Service name
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service not registered
        """
        if name not in self._factories:
            raise ValueError(f"Service not registered: {name}")
        
        # Return cached singleton if available
        if name in self._singletons:
            logger.debug("container_get_singleton", name=name)
            return self._singletons[name]
        
        # Create new instance
        factory = self._factories[name]
        instance = factory(self)
        
        # Cache if singleton
        if self._is_singleton[name]:
            self._singletons[name] = instance
            logger.debug("container_created_singleton", name=name)
        else:
            logger.debug("container_created_transient", name=name)
        
        return instance
    
    def has(self, name: str) -> bool:
        """Check if service is registered"""
        return name in self._factories
    
    async def close(self) -> None:
        """
        Close all singleton services
        
        Called on application shutdown.
        Calls close() on any service that has it.
        """
        logger.info("container_closing")
        
        for name, instance in self._singletons.items():
            try:
                if hasattr(instance, "close"):
                    if callable(instance.close):
                        result = instance.close()
                        # Handle both sync and async close
                        if hasattr(result, "__await__"):
                            await result
                    logger.debug("service_closed", name=name)
            except Exception as e:
                logger.error("service_close_failed", name=name, error=str(e))
        
        self._singletons.clear()
        logger.info("container_closed")
    
    def reset(self) -> None:
        """
        Reset container (for testing)
        
        Closes all singletons and clears cache.
        """
        logger.info("container_resetting")
        
        for name, instance in self._singletons.items():
            try:
                if hasattr(instance, "close") and callable(instance.close):
                    instance.close()
            except Exception as e:
                logger.warning("reset_close_failed", name=name, error=str(e))
        
        self._singletons.clear()
        logger.info("container_reset")
    
    def __repr__(self) -> str:
        """String representation"""
        singleton_count = len(self._singletons)
        factory_count = len(self._factories)
        return f"ServiceContainer(factories={factory_count}, singletons={singleton_count})"


# Global container instance
_container_instance: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    """
    Get or create global container instance
    
    Lazily initializes container on first access.
    """
    global _container_instance
    
    if _container_instance is None:
        _container_instance = ServiceContainer()
        _setup_container(_container_instance)
    
    return _container_instance


def reset_global_container() -> None:
    """
    Reset global container (for testing)
    """
    global _container_instance
    if _container_instance:
        _container_instance.reset()
    _container_instance = None


def create_test_container() -> ServiceContainer:
    """
    Create isolated test container
    
    Returns fresh container without setup.
    Use for unit tests.
    """
    logger.info("creating_test_container")
    return ServiceContainer()


def _setup_container(container: ServiceContainer) -> None:
    """
    Register all production services
    
    Called once on container initialization.
    """
    from gravity_tech.config.unified_settings import get_settings
    from gravity_tech.infrastructure.container_factories import (
        create_cache_service,
        create_database_service,
        create_analysis_service,
        create_tool_recommendation_service,
        create_data_ingestor_service,
        create_event_publisher,
    )
    
    settings = get_settings()
    
    # Register core services
    container.register("settings", lambda _: settings, singleton=True)
    
    # Register cache
    container.register(
        "cache",
        create_cache_service,
        singleton=True
    )
    
    # Register database
    container.register(
        "database",
        create_database_service,
        singleton=True
    )
    
    # Register analysis service
    container.register(
        "analysis_service",
        create_analysis_service,
        singleton=True
    )
    
    # Register tool recommendation service
    container.register(
        "tool_recommendation_service",
        create_tool_recommendation_service,
        singleton=True
    )
    
    # Register data ingestor
    container.register(
        "data_ingestor",
        create_data_ingestor_service,
        singleton=True
    )
    
    # Register event publisher (optional)
    if settings.features.kafka_enabled or settings.features.rabbitmq_enabled:
        container.register(
            "event_publisher",
            create_event_publisher,
            singleton=True
        )
    
    logger.info("container_setup_complete", services_count=len(container._factories))
