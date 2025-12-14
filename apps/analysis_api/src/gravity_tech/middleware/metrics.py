"""
Custom Prometheus Metrics for Technical Analysis Service

Provides detailed metrics for monitoring system performance and health.

Author: Gravity Tech Team
Date: November 20, 2025
Version: 1.0.0
License: MIT
"""

import time


from prometheus_client import Counter, Gauge, Histogram, Info

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

# Analysis metrics
ANALYSIS_COUNT = Counter(
    'analysis_requests_total',
    'Total number of analysis requests',
    ['analysis_type', 'symbol', 'timeframe']
)

ANALYSIS_DURATION = Histogram(
    'analysis_duration_seconds',
    'Analysis processing time in seconds',
    ['analysis_type']
)

ANALYSIS_ERRORS = Counter(
    'analysis_errors_total',
    'Total number of analysis errors',
    ['analysis_type', 'error_type']
)

# Cache metrics
CACHE_HITS = Counter(
    'cache_hits_total',
    'Total number of cache hits',
    ['cache_type']
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total number of cache misses',
    ['cache_type']
)

CACHE_SIZE = Gauge(
    'cache_size_bytes',
    'Current cache size in bytes',
    ['cache_type']
)

# Database metrics
DB_CONNECTIONS_ACTIVE = Gauge(
    'db_connections_active',
    'Number of active database connections',
    ['db_type']
)

DB_QUERY_DURATION = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type']
)

DB_ERRORS = Counter(
    'db_errors_total',
    'Total number of database errors',
    ['db_type', 'error_type']
)

# ML metrics
ML_PREDICTIONS = Counter(
    'ml_predictions_total',
    'Total number of ML predictions',
    ['model_type']
)

ML_PREDICTION_DURATION = Histogram(
    'ml_prediction_duration_seconds',
    'ML prediction duration in seconds',
    ['model_type']
)

ML_MODEL_ACCURACY = Gauge(
    'ml_model_accuracy',
    'Current ML model accuracy',
    ['model_type']
)

# System metrics
MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Current memory usage in bytes'
)

CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'Current CPU usage percentage'
)

# Service info
SERVICE_INFO = Info(
    'service_info',
    'Service information',
    ['version', 'environment']
)


class MetricsCollector:
    """
    Centralized metrics collection and reporting
    """

    def __init__(self):
        self.start_time = time.time()

    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

    def record_analysis(self, analysis_type: str, symbol: str, timeframe: str, duration: float, error: str | None = None):
        """Record analysis metrics"""
        ANALYSIS_COUNT.labels(analysis_type=analysis_type, symbol=symbol, timeframe=timeframe).inc()
        ANALYSIS_DURATION.labels(analysis_type=analysis_type).observe(duration)

        if error:
            ANALYSIS_ERRORS.labels(analysis_type=analysis_type, error_type=error).inc()

    def record_cache_operation(self, cache_type: str, hit: bool):
        """Record cache operation metrics"""
        if hit:
            CACHE_HITS.labels(cache_type=cache_type).inc()
        else:
            CACHE_MISSES.labels(cache_type=cache_type).inc()

    def record_db_operation(self, db_type: str, query_type: str, duration: float, error: str | None = None):
        """Record database operation metrics"""
        DB_QUERY_DURATION.labels(query_type=query_type).observe(duration)

        if error:
            DB_ERRORS.labels(db_type=db_type, error_type=error).inc()

    def record_ml_prediction(self, model_type: str, duration: float):
        """Record ML prediction metrics"""
        ML_PREDICTIONS.labels(model_type=model_type).inc()
        ML_PREDICTION_DURATION.labels(model_type=model_type).observe(duration)

    def update_system_metrics(self):
        """Update system resource metrics"""
        import psutil

        try:
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.used)

            cpu = psutil.cpu_percent(interval=1)
            CPU_USAGE.set(cpu)
        except ImportError:
            # psutil not available
            pass

    def set_service_info(self, version: str, environment: str):
        """Set service information"""
        SERVICE_INFO.info({
            'version': version,
            'environment': environment
        })


# Global metrics collector
metrics_collector = MetricsCollector()
