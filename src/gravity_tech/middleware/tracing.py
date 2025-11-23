"""
Distributed Tracing Setup - OpenTelemetry

پیکربندی Distributed Tracing برای ردیابی درخواست‌ها در سراسر میکروسرویس‌ها

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
import structlog

logger = structlog.get_logger()


def setup_tracing(app, service_name: str = "technical-analysis-service"):
    """
    راه‌اندازی Distributed Tracing با Jaeger
    
    Args:
        app: FastAPI application
        service_name: نام سرویس
    
    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> setup_tracing(app)
    """
    
    # تعریف resource
    resource = Resource(attributes={
        SERVICE_NAME: service_name
    })
    
    # ایجاد tracer provider
    provider = TracerProvider(resource=resource)
    
    # Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    
    # اضافه کردن span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    provider.add_span_processor(span_processor)
    
    # تنظیم global tracer
    trace.set_tracer_provider(provider)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    
    # Instrument HTTP Client
    HTTPXClientInstrumentor().instrument()
    
    # Instrument Redis
    try:
        RedisInstrumentor().instrument()
    except Exception as e:
        logger.warning("redis_instrumentation_failed", error=str(e))
    
    logger.info("distributed_tracing_enabled", exporter="jaeger")


def get_tracer(name: str = __name__):
    """
    دریافت tracer برای استفاده در کد
    
    Example:
        >>> tracer = get_tracer(__name__)
        >>> with tracer.start_as_current_span("my-operation"):
        ...     result = do_work()
    """
    return trace.get_tracer(name)
