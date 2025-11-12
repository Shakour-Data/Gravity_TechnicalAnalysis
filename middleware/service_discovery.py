"""
Service Discovery Integration - Eureka & Consul

ثبت خودکار سرویس در Service Registry و گزارش وضعیت سلامت
"""

import asyncio
from typing import Optional
import structlog

# Make py_eureka_client optional
try:
    from py_eureka_client import eureka_client
    EUREKA_AVAILABLE = True
except ImportError:
    EUREKA_AVAILABLE = False
    eureka_client = None

# Make consul optional
try:
    import consul.aio
    CONSUL_AVAILABLE = True
except ImportError:
    CONSUL_AVAILABLE = False
    consul = None

from config.settings import settings

logger = structlog.get_logger()


class ServiceDiscovery:
    """
    مدیریت Service Discovery با Eureka یا Consul
    
    قابلیت‌ها:
    - ثبت خودکار سرویس
    - به‌روزرسانی وضعیت سلامت
    - Heartbeat به Registry
    - De-registration هنگام shutdown
    """
    
    def __init__(self):
        self.registry_type: Optional[str] = None
        self.consul_client: Optional[object] = None  # Made generic to avoid import issues
        self._heartbeat_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """راه‌اندازی اولیه Service Discovery"""
        
        if not settings.eureka_enabled:
            logger.info("service_discovery_disabled")
            return
        
        if not EUREKA_AVAILABLE:
            logger.warning("eureka_client_not_installed", 
                          message="py_eureka_client is not installed. Service discovery disabled.")
            return
        
        # تشخیص نوع registry
        if settings.eureka_server_url:
            await self._init_eureka()
        else:
            logger.warning("service_discovery_enabled_but_no_server_configured")
    
    async def _init_eureka(self):
        """راه‌اندازی Eureka Client"""
        try:
            eureka_client.init(
                eureka_server=settings.eureka_server_url,
                app_name=settings.app_name,
                instance_port=settings.port,
                instance_host=settings.host,
                instance_id=f"{settings.app_name}:{settings.port}",
                # Health check URL
                health_check_url=f"http://{settings.host}:{settings.port}/health",
                status_page_url=f"http://{settings.host}:{settings.port}/",
                home_page_url=f"http://{settings.host}:{settings.port}/",
                # Metadata
                metadata={
                    "version": settings.app_version,
                    "environment": settings.environment,
                    "metrics": f"http://{settings.host}:{settings.port}/metrics",
                },
                # Renewal settings
                renewal_interval_in_secs=30,
                duration_in_secs=90,
            )
            
            self.registry_type = "eureka"
            logger.info(
                "eureka_client_initialized",
                server=settings.eureka_server_url,
                app=settings.app_name,
                port=settings.port
            )
            
        except Exception as e:
            logger.error("eureka_initialization_failed", error=str(e))
            raise
    
    async def _init_consul(self, consul_host: str = "localhost", consul_port: int = 8500):
        """راه‌اندازی Consul Client"""
        try:
            self.consul_client = consul.aio.Consul(host=consul_host, port=consul_port)
            
            # ثبت سرویس
            await self.consul_client.agent.service.register(
                name=settings.app_name,
                service_id=f"{settings.app_name}-{settings.port}",
                address=settings.host,
                port=settings.port,
                tags=[
                    settings.environment,
                    settings.app_version,
                    "technical-analysis",
                    "microservice"
                ],
                check={
                    "http": f"http://{settings.host}:{settings.port}/health",
                    "interval": "30s",
                    "timeout": "5s",
                    "deregister_critical_service_after": "90s"
                },
                meta={
                    "version": settings.app_version,
                    "environment": settings.environment,
                }
            )
            
            self.registry_type = "consul"
            logger.info(
                "consul_client_initialized",
                host=consul_host,
                port=consul_port,
                service=settings.app_name
            )
            
        except Exception as e:
            logger.error("consul_initialization_failed", error=str(e))
            raise
    
    async def update_status(self, status: str = "UP"):
        """
        به‌روزرسانی وضعیت سلامت سرویس
        
        Args:
            status: وضعیت ("UP", "DOWN", "STARTING", "OUT_OF_SERVICE")
        """
        try:
            if self.registry_type == "eureka":
                # Eureka به صورت خودکار با health check URL وضعیت را چک می‌کند
                logger.debug("eureka_status_check_automatic")
            
            elif self.registry_type == "consul" and self.consul_client:
                # Consul هم به صورت خودکار check می‌کند
                # اما می‌توانیم manual update هم بزنیم
                await self.consul_client.agent.check.register(
                    name=f"{settings.app_name}-health",
                    check_id=f"{settings.app_name}-{settings.port}-health",
                    http=f"http://{settings.host}:{settings.port}/health",
                    interval="30s"
                )
                logger.debug("consul_health_check_registered")
        
        except Exception as e:
            logger.error("status_update_failed", error=str(e), status=status)
    
    async def deregister(self):
        """حذف سرویس از Registry (هنگام shutdown)"""
        try:
            if self.registry_type == "eureka":
                eureka_client.stop()
                logger.info("eureka_client_stopped")
            
            elif self.registry_type == "consul" and self.consul_client:
                await self.consul_client.agent.service.deregister(
                    service_id=f"{settings.app_name}-{settings.port}"
                )
                logger.info("consul_service_deregistered")
        
        except Exception as e:
            logger.error("deregistration_failed", error=str(e))
    
    async def discover_service(self, service_name: str) -> Optional[dict]:
        """
        کشف یک سرویس دیگر
        
        Args:
            service_name: نام سرویس مورد نظر
        
        Returns:
            اطلاعات سرویس (host, port, metadata) یا None
        """
        try:
            if self.registry_type == "eureka":
                # دریافت اطلاعات از Eureka
                instances = eureka_client.get_application(service_name)
                if instances and len(instances) > 0:
                    instance = instances[0]  # اولین instance
                    return {
                        "host": instance.ipAddr,
                        "port": instance.port,
                        "metadata": instance.metadata,
                        "status": instance.status
                    }
            
            elif self.registry_type == "consul" and self.consul_client:
                # دریافت اطلاعات از Consul
                index, services = await self.consul_client.health.service(
                    service_name,
                    passing=True  # فقط سرویس‌های سالم
                )
                
                if services and len(services) > 0:
                    service = services[0]
                    return {
                        "host": service["Service"]["Address"],
                        "port": service["Service"]["Port"],
                        "metadata": service["Service"]["Meta"],
                        "tags": service["Service"]["Tags"]
                    }
            
            logger.warning("service_not_found", service=service_name)
            return None
        
        except Exception as e:
            logger.error(
                "service_discovery_failed",
                service=service_name,
                error=str(e)
            )
            return None


# Global instance
service_discovery = ServiceDiscovery()


async def startup_service_discovery():
    """راه‌اندازی Service Discovery هنگام startup"""
    await service_discovery.initialize()
    await service_discovery.update_status("UP")


async def shutdown_service_discovery():
    """خاموش کردن Service Discovery هنگام shutdown"""
    await service_discovery.update_status("OUT_OF_SERVICE")
    await service_discovery.deregister()
