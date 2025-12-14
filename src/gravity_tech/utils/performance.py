"""
Performance decorators for fast computations.
"""

import time
from functools import wraps

from numba import jit


def jit_compile(func):
    """Decorator to JIT compile functions with Numba."""
    compiled_func = jit(nopython=True, cache=True, parallel=True)(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        return compiled_func(*args, **kwargs)

    return wrapper

def benchmark(func):
    """Decorator to benchmark function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper