import asyncio
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from loguru import logger

P = ParamSpec("P")
R = TypeVar("R")


def log_exception(func: Callable[P, R]) -> Callable[P, R]:
    """예외 발생 시 자동으로 로깅하는 데코레이터"""

    @wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.opt(depth=1).error(
                f"{func.__name__} 함수 예외 발생: {e}"
            )
            raise

    @wraps(func)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await func(*args, **kwargs)  # type: ignore
        except Exception as e:
            logger.opt(depth=1).error(
                f"{func.__name__} 함수 예외 발생: {e}"
            )
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper  # type: ignore


def log_flow(func: Callable[P, R]) -> Callable[P, R]:
    """함수 플로우 로깅 데코레이터"""

    @wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger.opt(depth=2).info(f"{func.__name__} 함수 시작")

        try:
            result = func(*args, **kwargs)
            logger.opt(depth=2).info(f"{func.__name__} 함수 성공")
            return result

        except Exception as e:
            logger.opt(depth=2).error(
                f"{func.__name__} 함수 예외 발생: {e}"
            )
            raise

    @wraps(func)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger.opt(depth=2).info(f"{func.__name__} 함수 시작")

        try:
            result = await func(*args, **kwargs)  # type: ignore
            logger.opt(depth=2).info(f"{func.__name__} 함수 성공")
            return result  # type: ignore

        except Exception as e:
            logger.opt(depth=2).error(
                f"{func.__name__} 함수 예외 발생: {e}"
            )
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper  # type: ignore
