"""实时话轮的有界 PCM 缓冲；仅由通道和适配器的事件循环访问。"""

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from uuid import uuid4

from .vad import INPUT_RATE, SAMPLE_BYTES

_MAX_BUFFER_BYTES = 2 * INPUT_RATE * SAMPLE_BYTES  # 单轮最多 2 秒尚未送达的 PCM


class StreamingTurn:
    def __init__(self, opened_at: float) -> None:
        loop = asyncio.get_running_loop()
        self.id = uuid4()
        self.opened_at = opened_at
        self.endpoint_at: float | None = None
        self.ended: asyncio.Future[None] = loop.create_future()
        self.aborted: asyncio.Future[None] = loop.create_future()
        self.started: asyncio.Future[float] = loop.create_future()
        self.exhausted = False
        self.recorded_bytes = 0
        self._ready = asyncio.Event()
        self._frames: deque[bytes] = deque()
        self._buffered_bytes = 0

    def offer(self, pcm: bytes) -> bool:
        if self._buffered_bytes + len(pcm) > _MAX_BUFFER_BYTES:
            return False
        if pcm:
            self.recorded_bytes += len(pcm)
            self._buffered_bytes += len(pcm)
            self._frames.append(pcm)
            self._ready.set()
        return True

    def finish(self, at: float) -> None:
        self.endpoint_at = at
        self.ended.set_result(None)
        self._ready.set()

    def abort(self) -> None:
        if not self.aborted.done():
            self.aborted.set_result(None)
        if not self.ended.done():
            self.ended.set_result(None)
        self._frames.clear()
        self._buffered_bytes = 0
        self._ready.set()

    async def frames(self, clock: Callable[[], float]) -> AsyncIterator[bytes]:
        while not self.aborted.done():
            if self._frames:
                pcm = self._frames.popleft()
                self._buffered_bytes -= len(pcm)
                if not self.started.done():
                    self.started.set_result(clock())
                yield pcm
            elif self.ended.done():
                self.exhausted = True
                return
            else:
                self._ready.clear()
                await self._ready.wait()
        # 缺帧或关闭不是输入正常结束；不能触发适配器发送 finish-task。
        raise asyncio.CancelledError
