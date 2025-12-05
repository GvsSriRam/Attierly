"""
Shared caching utilities using cachetools.
Replaces custom cache implementation with battle-tested library.
"""

from typing import Any, Optional, Callable
from cachetools import TTLCache, cached
from cachetools.keys import hashkey
import functools
import hashlib
import json


class SimpleCacheService:
    """
    Simple cache service using cachetools.
    Replaces the custom CacheService implementation.
    """

    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize cache service.

        Args:
            maxsize: Maximum number of items in cache
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self.cache[key] = value

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        self.cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self.cache

    def generate_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            MD5 hash of serialized arguments
        """
        # Serialize arguments to JSON
        data = {
            'args': args,
            'kwargs': kwargs
        }
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()


def cached_function(ttl_seconds: int = 3600, maxsize: int = 128):
    """
    Decorator for caching function results.

    Usage:
        @cached_function(ttl_seconds=300)
        async def expensive_operation(arg1, arg2):
            return result

    Args:
        ttl_seconds: Cache TTL in seconds
        maxsize: Maximum cache size
    """
    cache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            key = hashkey(*args, **kwargs)

            # Check cache
            if key in cache:
                return cache[key]

            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache[key] = result
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            key = hashkey(*args, **kwargs)

            # Check cache
            if key in cache:
                return cache[key]

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = result
            return result

        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Global cache instance for shared use
default_cache = SimpleCacheService()
