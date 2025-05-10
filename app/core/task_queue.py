import asyncio
from typing import Callable, Awaitable, Any

from app.utils.logging_decorator import log_exception, log_flow

class SerialTaskQueue:
    def __init__(self):
        self._queue = asyncio.Queue()
        self._is_running = False

    @log_flow
    def start(self):
        if not self._is_running:
            loop = asyncio.get_event_loop()
            loop.create_task(self._worker())
            self._is_running = True
    
    @log_exception
    async def _worker(self):
        while True:
            coro_func = await self._queue.get()
            await coro_func()
            self._queue.task_done()
            
    @log_exception
    async def enqueue(self, coro_func: Callable[[], Awaitable[Any]]) -> Any:
        future = asyncio.get_event_loop().create_future()
        
        async def wrapper():
            result = await coro_func()
            future.set_result(result)
        await self._queue.put(wrapper)
        return await future