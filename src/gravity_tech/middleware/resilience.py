"""
Resilience Patterns - Circuit Breaker, Retry, Timeout, Bulkhead

الگوهای مقاومتی برای افزایش قابلیت اطمینان میکروسرویس

Author: Gravity Tech Team
Date: November 14, 2025
Version: 1.0.0
License: MIT
"""

import asyncio
import time
from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Optional

import structlog

logger = structlog.get_logger()


class CircuitState(Enum):
    """حالت Circuit Breaker"""
    CLOSED = "closed"  # عادی - درخواست‌ها عبور می‌کنند
    OPEN = "open"  # باز - تمام درخواست‌ها رد می‌شوند
    HALF_OPEN = "half_open"  # نیمه‌باز - تست برای بازگشت به حالت عادی


class CircuitBreaker:
    """
    Circuit Breaker Pattern

    جلوگیری از ارسال درخواست به سرویس‌های خراب و دادن زمان برای بازیابی

    Args:
        failure_threshold: تعداد خطاهای متوالی برای باز شدن circuit
        recovery_timeout: زمان انتظار قبل از تست مجدد (ثانیه)
        expected_exception: نوع exception که باید catch شود

    Example:
        >>> circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        >>> @circuit
        ... async def call_external_service():
        ...     return await service.call()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED

    def __call__(self, func: Callable) -> Callable:
        """Decorator for protecting function with circuit breaker"""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info("circuit_breaker_half_open", function=func.__name__)
                else:
                    logger.warning(
                        "circuit_breaker_open",
                        function=func.__name__,
                        time_until_retry=self.recovery_timeout - (time.time() - self.last_failure_time)
                    )
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}")

            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result

            except self.expected_exception as e:
                self._on_failure()
                logger.error(
                    "circuit_breaker_failure",
                    function=func.__name__,
                    error=str(e),
                    failure_count=self.failure_count,
                    state=self.state.value
                )
                raise

        return wrapper

    def _should_attempt_reset(self) -> bool:
        """بررسی اینکه آیا زمان تست مجدد رسیده است"""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )

    def _on_success(self):
        """موفقیت - reset کردن counter و بستن circuit"""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("circuit_breaker_closed")

    def _on_failure(self):
        """خطا - افزایش counter و احتمالاً باز کردن circuit"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                "circuit_breaker_opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Retry Decorator با Exponential Backoff و Jitter

    تلاش مجدد درخواست با تاخیر فزاینده

    Args:
        max_retries: حداکثر تعداد تلاش مجدد
        initial_delay: تاخیر اولیه (ثانیه)
        max_delay: حداکثر تاخیر (ثانیه)
        exponential_base: پایه نمایی برای backoff
        jitter: اضافه کردن نویز تصادفی

    Example:
        >>> @retry_with_backoff(max_retries=5, initial_delay=1)
        ... async def fetch_data():
        ...     return await api.get('/data')
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempts=attempt + 1,
                            error=str(e)
                        )
                        raise

                    # محاسبه تاخیر با backoff
                    delay = min(
                        initial_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # اضافه کردن jitter
                    if jitter:
                        import random
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay=f"{delay:.2f}s",
                        error=str(e)
                    )

                    await asyncio.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


def timeout(seconds: float):
    """
    Timeout Decorator

    محدود کردن زمان اجرای تابع

    Args:
        seconds: حداکثر زمان اجرا (ثانیه)

    Example:
        >>> @timeout(30)
        ... async def slow_operation():
        ...     await process_data()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                logger.error(
                    "operation_timeout",
                    function=func.__name__,
                    timeout=seconds
                )
                raise TimeoutError(
                    f"Operation {func.__name__} timed out after {seconds}s"
                )

        return wrapper
    return decorator


class Bulkhead:
    """
    Bulkhead Pattern

    محدود کردن منابع برای جلوگیری از استفاده بیش از حد

    Args:
        max_concurrent: حداکثر تعداد درخواست‌های همزمان

    Example:
        >>> bulkhead = Bulkhead(max_concurrent=10)
        >>> @bulkhead
        ... async def resource_intensive_operation():
        ...     return await heavy_task()
    """

    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.active_count = 0

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with self.semaphore:
                self.active_count += 1
                logger.debug(
                    "bulkhead_acquired",
                    function=func.__name__,
                    active=self.active_count,
                    max=self.max_concurrent
                )

                try:
                    return await func(*args, **kwargs)
                finally:
                    self.active_count -= 1
                    logger.debug(
                        "bulkhead_released",
                        function=func.__name__,
                        active=self.active_count
                    )

        return wrapper


# استفاده ترکیبی - چند pattern با هم
def resilient(
    max_retries: int = 3,
    timeout_seconds: float = 30,
    circuit_threshold: int = 5,
    max_concurrent: int = 10
):
    """
    ترکیب تمام الگوهای مقاومتی

    Example:
        >>> @resilient(max_retries=3, timeout_seconds=30)
        ... async def call_external_api():
        ...     return await api.fetch()
    """
    def decorator(func: Callable) -> Callable:
        # اعمال لایه‌های مختلف
        func = Bulkhead(max_concurrent)(func)
        func = CircuitBreaker(failure_threshold=circuit_threshold)(func)
        func = retry_with_backoff(max_retries=max_retries)(func)
        func = timeout(timeout_seconds)(func)
        return func

    return decorator
