import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def compute_execution_time(func):
    """Decorator that logs execution time for both sync and async functions."""

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"[TIMING] {func.__qualname__}: {elapsed:.4f}s")
        print(f"  ⏱  {func.__qualname__}: {elapsed:.4f}s")
        return result

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"[TIMING] {func.__qualname__}: {elapsed:.4f}s")
        print(f"  ⏱  {func.__qualname__}: {elapsed:.4f}s")
        return result

    import asyncio

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper