import asyncio
import logging
from typing import Callable, Awaitable, Any, Optional

from app.utils.logging_decorator import log_exception, log_flow

logger = logging.getLogger(__name__)


class SerialTaskQueue:
    """
    비동기 작업을 순차적으로 처리하는 큐 클래스입니다.
    
    이 클래스는 비동기 작업들을 순차적으로 실행하며, 각 작업의 완료를 기다립니다.
    작업은 큐에 추가된 순서대로 처리됩니다.
    """

    def __init__(self) -> None:
        """
        SerialTaskQueue 인스턴스를 초기화합니다.
        
        초기화 시 빈 큐와 실행 상태 플래그를 생성합니다.
        """
        self._queue: asyncio.Queue = asyncio.Queue()
        self._is_running: bool = False
        logger.debug("SerialTaskQueue 인스턴스 초기화 완료")

    @log_flow
    def start(self) -> None:
        """
        작업 큐의 워커를 시작합니다.
        
        워커가 이미 실행 중이 아닌 경우에만 새로운 워커를 시작합니다.
        """
        if not self._is_running:
            logger.info("작업 큐 워커 시작")
            loop = asyncio.get_event_loop()
            loop.create_task(self._worker())
            self._is_running = True
            logger.debug("작업 큐 워커 생성 완료")
        else:
            logger.debug("작업 큐 워커가 이미 실행 중")

    @log_exception
    async def _worker(self) -> None:
        """
        큐에서 작업을 가져와 순차적으로 실행하는 워커 메서드입니다.
        
        큐에서 작업을 가져와 실행하고, 작업이 완료되면 큐에서 제거합니다.
        이 프로세스는 무한히 반복됩니다.
        """
        logger.debug("작업 큐 워커 시작")
        while True:
            coro_func = await self._queue.get()
            logger.debug("새로운 작업 실행 시작")
            await coro_func()
            self._queue.task_done()
            logger.debug("작업 완료 및 큐에서 제거")

    @log_exception
    async def enqueue(self, coro_func: Callable[[], Awaitable[Any]]) -> Any:
        """
        새로운 비동기 작업을 큐에 추가합니다.
        
        Args:
            coro_func: 실행할 비동기 코루틴 함수
            
        Returns:
            Any: 작업의 실행 결과
            
        Note:
            작업은 큐에 추가된 순서대로 실행되며,
            이전 작업이 완료될 때까지 대기합니다.

        """
        logger.debug("새로운 작업 큐에 추가")
        future = asyncio.get_event_loop().create_future()
        
        async def wrapper() -> None:
            """
            작업을 실행하고 결과를 future에 설정하는 래퍼 함수입니다.
            """
            try:
                result = await coro_func()
                future.set_result(result)
                logger.debug("작업 실행 완료 및 결과 설정")
            except Exception as e:
                logger.error(
                    "작업 실행 중 오류 발생",
                    extra={"error": str(e)},
                )
                future.set_exception(e)
                
        await self._queue.put(wrapper)
        logger.debug("작업이 큐에 추가됨")
        return await future
    