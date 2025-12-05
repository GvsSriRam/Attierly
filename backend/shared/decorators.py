"""
Shared decorators for API endpoints and common operations.
"""

import logging
from functools import wraps
from typing import Callable, Any
from fastapi import HTTPException


logger = logging.getLogger(__name__)


def api_endpoint(operation_name: str):
    """
    Decorator for standardized API endpoint error handling.

    Usage:
        @router.post("/endpoint")
        @api_endpoint("operation_name")
        async def endpoint_handler(...):
            return result

    Args:
        operation_name: Name of the operation for logging
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                logger.info(f"Starting {operation_name}")
                result = await func(*args, **kwargs)
                logger.info(f"Completed {operation_name}")
                return result
            except HTTPException:
                # Re-raise HTTP exceptions as-is
                raise
            except Exception as e:
                logger.error(f"Error in {operation_name}: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal server error in {operation_name}: {str(e)}"
                )
        return wrapper
    return decorator


def log_execution(operation_name: str = None):
    """
    Simple decorator for logging function execution.

    Usage:
        @log_execution("my_operation")
        async def my_function(...):
            pass
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            logger.debug(f"Executing {op_name}")
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Completed {op_name}")
                return result
            except Exception as e:
                logger.error(f"Failed {op_name}: {str(e)}")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            logger.debug(f"Executing {op_name}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Completed {op_name}")
                return result
            except Exception as e:
                logger.error(f"Failed {op_name}: {str(e)}")
                raise

        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
