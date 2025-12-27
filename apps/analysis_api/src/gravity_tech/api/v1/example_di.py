"""
Example API Endpoint with Dependency Injection

Demonstrates how to use the new ServiceContainer for dependency injection.

Before (Anti-pattern - Global singleton):
    from services import cache_manager, analysis_service

    @router.post("/analyze")
    async def analyze(request):
        result = analysis_service.analyze(request.candles)
        cache_manager.set(key, result)
        return result

After (DI pattern):
    from infrastructure.container import get_container

    @router.post("/analyze")
    async def analyze(request, container = Depends(lambda: get_container())):
        service = container.get("analysis_service")
        result = await service.analyze(request.candles)
        return result
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from gravity_tech.infrastructure.container import ServiceContainer, get_container

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["Technical Analysis"])


# ============================================================================
# Dependency Injection Functions
# ============================================================================

async def get_container_dep() -> ServiceContainer:
    """FastAPI dependency for getting container.

    Returns the global container instance.
    """
    return get_container()


async def get_analysis_service(
    container: ServiceContainer = Depends(get_container_dep),
):
    """FastAPI dependency for analysis service.

    Example:
        @router.post("/analyze")
        async def analyze(
            request: dict,
            service = Depends(get_analysis_service)
        ):
            return await service.analyze(request.get("candles"))
    """
    try:
        return container.get("analysis_service")
    except ValueError as e:
        logger.error("analysis_service_not_found", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis service not available",
        ) from e


async def get_cache_service(
    container: ServiceContainer = Depends(get_container_dep),
):
    """FastAPI dependency for cache service."""
    try:
        return container.get("cache")
    except ValueError as e:
        logger.error("cache_service_not_found", error=str(e))
        return None  # Cache is optional


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "/analyze",
    summary="Perform technical analysis",
    description="Complete technical analysis with 60+ indicators",
    responses={
        200: {"description": "Successful analysis"},
        400: {"description": "Invalid input"},
        500: {"description": "Internal server error"},
    },
)
async def analyze(
    request: dict,
    service=Depends(get_analysis_service),
    cache=Depends(get_cache_service),
) -> dict:
    """Perform complete technical analysis.

    DI enables:
    - Easy to test (inject mocks)
    - Easy to swap implementations
    - Clean dependency graph
    """
    try:
        # Generate cache key
        cache_key = None
        symbol = request.get("symbol", "unknown")
        candles = request.get("candles", [])

        if cache:
            cache_key = f"analysis:{symbol}:{len(candles)}"
            cached_result = await cache.get(cache_key)
            if cached_result:
                logger.info("analysis_cache_hit", symbol=symbol)
                return cached_result

        logger.info("analysis_starting", symbol=symbol, candles=len(candles))

        # Perform analysis
        result = await service.analyze(candles)

        # Cache result
        if cache and cache_key:
            await cache.set(cache_key, result, ttl=300)
            logger.debug("analysis_cached", key=cache_key)

        logger.info("analysis_complete", symbol=symbol)
        return result

    except ValueError as e:
        logger.error("analysis_validation_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    except Exception as e:
        logger.error("analysis_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed",
        ) from e


@router.get("/health/ready")
async def health_ready(
    container: ServiceContainer = Depends(get_container_dep),
) -> dict:
    """Readiness probe - check all dependencies.

    Example of checking service health.
    """
    checks = {"status": "ready", "services": {}}

    # Check cache
    try:
        _ = container.get("cache")
        checks["services"]["cache"] = "ok"
    except Exception as e:
        checks["services"]["cache"] = f"error: {str(e)}"

    # Check database
    try:
        _ = container.get("database")
        checks["services"]["database"] = "ok"
    except Exception as e:
        checks["services"]["database"] = f"error: {str(e)}"

    # Check analysis service
    try:
        _ = container.get("analysis_service")
        checks["services"]["analysis"] = "ok"
    except Exception as e:
        checks["services"]["analysis"] = f"error: {str(e)}"

    # Determine overall status
    errors = [v for v in checks["services"].values() if "error" in v]
    if errors:
        checks["status"] = "degraded"

    return checks


# ============================================================================
# Integration with main.py
# ============================================================================

def register_endpoints(app):
    """Register all endpoints with application.

    In main.py:
        from api.v1.example_di import register_endpoints
        app = FastAPI()
        register_endpoints(app)
    """
    app.include_router(router)
