"""按句调度流式 TTS；设备回调只拷贝帧并累加已交设备的进度。"""

import asyncio
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass, field
import logging
from threading import Lock
from unicodedata import category

from ..config import VoiceSettings
from ..protocol import DeliveryStatus, Receipt, ReplyStream
from .tts import StreamingTTS, TtsError

_LOG = logging.getLogger(__name__)
_MAX_SEGMENT = 72
_MAX_REPLY_TEXT = 16_000
_MAX_PCM_CHUNK = 512 * 1024
_TICK = 0.008
_BOUNDARIES = frozenset("。！？；，!?;\n")


async def _segments(reply: ReplyStream) -> AsyncGenerator[str, None]:
    pending = []
    async for delta in reply:
        for char in delta:
            pending.append(char)
            if char in _BOUNDARIES or len(pending) >= _MAX_SEGMENT:
                yield "".join(pending)
                pending.clear()
    if pending:
        yield "".join(pending)


@dataclass(eq=False, slots=True)
class _Segment:
    text: str
    received: int = 0
    played: int = 0
    complete: bool = False


@dataclass(eq=False, slots=True)
class _PlaybackReply:
    signal: asyncio.Future[DeliveryStatus]
    segments: list[_Segment] = field(default_factory=list)

    def confirmation(self, bytes_per_second: int) -> str:
        # 只在先从设备队列摘除本回复后读计数，回调不会再更改它们。
        text: list[str] = []
        played = 0
        voiced_chars = 0
        for segment in self.segments:
            if segment.complete and segment.played == segment.received:
                text.append(segment.text)
                if segment.played:
                    played += segment.played
                    voiced_chars += len(segment.text)
                continue
            if segment.played:
                # 总时长未知时先参考已播整段语速，再回退到每秒四字。
                estimate = (len(segment.text) * segment.played // segment.received if segment.complete else
                            voiced_chars * segment.played // played if played else
                            int(4 * segment.played / bytes_per_second))
                count = max(1, min(estimate, len(segment.text) - 1))
                text.append(segment.text[:count])
                played += segment.played
            break
        return "".join(text) if played else ""


class _PcmQueue:
    def __init__(self, capacity: int) -> None:
        self._lock = Lock()
        self._packets: deque[tuple[_Segment, memoryview]] = deque()
        self._active: _PlaybackReply | None = None
        self._queued = 0
        self._capacity = capacity

    def attach(self, state: _PlaybackReply) -> None:
        with self._lock:
            self._active = state

    def detach(self, state: _PlaybackReply) -> None:
        with self._lock:
            if self._active is state:
                self._active = None
                self._packets.clear()
                self._queued = 0

    def size(self, state: _PlaybackReply) -> int:
        with self._lock:
            return self._queued if self._active is state else 0

    def offer(self, state: _PlaybackReply, part: _Segment, pcm: memoryview) -> int:
        with self._lock:
            if self._active is not state:
                raise asyncio.CancelledError  # 防止旧响应在网络取消时迟到的块碰到新响应
            size = min(len(pcm), self._capacity - self._queued)
            if size:
                self._packets.append((part, pcm[:size]))
                self._queued += size
                part.received += size
            return size

    def fill(self, output: memoryview) -> None:
        output = output.cast("B")
        output[:] = bytes(len(output))
        if not self._lock.acquire(blocking=False):
            return  # 与事件循环冲突时只补静音，设备线程不等待
        try:
            offset = 0
            while offset < len(output) and self._packets:
                part, pcm = self._packets[0]
                size = min(len(output) - offset, len(pcm))
                output[offset:offset + size] = pcm[:size]
                part.played += size
                self._queued -= size
                offset += size
                if size == len(pcm):
                    self._packets.popleft()
                else:
                    self._packets[0] = (part, pcm[size:])
        finally:
            self._lock.release()


class VoiceOutput:
    """面向一个设备输出；中断只结束交付，不取消 Runtime 的模型或工具。"""

    def __init__(self, tts: StreamingTTS, settings: VoiceSettings) -> None:
        self._tts = tts
        self._settings = settings
        self._bytes_per_second = 2 * settings.playback_rate
        self._prefetch = max(1, round(settings.prefetch_seconds * self._bytes_per_second))
        self._pcm = _PcmQueue(2 * round(settings.buffer_seconds * settings.playback_rate))
        self._active: _PlaybackReply | None = None
        self._workers: set[asyncio.Task[None]] = set()
        self._closed = False

    def render(self, output: memoryview) -> None:
        self._pcm.fill(output)

    def interrupt(self) -> None:
        self._stop(DeliveryStatus.PARTIAL)

    def fail(self) -> None:
        self._closed = True
        self._stop(DeliveryStatus.FAILED)

    def _stop(self, status: DeliveryStatus) -> None:
        state = self._active
        if state is not None and not state.signal.done():
            self._pcm.detach(state)  # 现在清除待播，不等待 HTTP 或回执任务
            state.signal.set_result(status)

    def _finished(self, task: asyncio.Task[None]) -> None:
        self._workers.discard(task)
        if not task.cancelled():
            task.exception()  # 回收已脱离本回复的网络任务异常

    async def deliver(self, reply: ReplyStream) -> Receipt:
        reply_id = reply.info.reply_id
        reply.limit_text(_MAX_REPLY_TEXT)
        if reply.overflowed:
            return Receipt(reply_id, DeliveryStatus.FAILED)
        if reply.abandoned:
            return Receipt(reply_id, DeliveryStatus.ZERO)
        # 正常只有一个任务；允许一个正在退出的网络任务，不无限累积不合作的适配器。
        if self._closed or self._active is not None or len(self._workers) >= 2:
            reply.abandon()
            return Receipt(reply_id, DeliveryStatus.FAILED)
        state = _PlaybackReply(asyncio.get_running_loop().create_future())
        self._active = state
        self._pcm.attach(state)
        worker = asyncio.create_task(self._produce(reply, state))
        self._workers.add(worker)
        worker.add_done_callback(self._finished)
        status: DeliveryStatus
        try:
            await asyncio.wait((worker, state.signal), timeout=self._settings.delivery_timeout,
                               return_when=asyncio.FIRST_COMPLETED)
            if state.signal.done():
                status = state.signal.result()
            elif not worker.done():
                status = DeliveryStatus.UNKNOWN
                _LOG.warning("voice delivery deadline reached")
            else:
                cancelled = worker.cancelled()
                error = None if cancelled else worker.exception()
                if not cancelled and error is None:
                    status = DeliveryStatus.FULL
                else:
                    status = DeliveryStatus.FAILED
                    _LOG.warning("voice synthesis failed (%s); reply=%s",
                                 error.diagnostic if isinstance(error, TtsError) else "tts_failed", reply_id)
        finally:
            self._pcm.detach(state)
            self._active = None
            reply.abandon()
            if not worker.done():
                worker.cancel()
            if not state.signal.done():
                state.signal.cancel()
        prefix = state.confirmation(self._bytes_per_second)
        if not prefix and status in (DeliveryStatus.PARTIAL, DeliveryStatus.FULL):
            status = DeliveryStatus.ZERO
        return Receipt(reply_id, status, prefix)

    async def _produce(self, reply: ReplyStream, state: _PlaybackReply) -> None:
        async for text in _segments(reply):
            if state.signal.done() or reply.abandoned:
                return
            part = _Segment(text)
            state.segments.append(part)
            spoken = text.strip()
            # 一串连续句末标点或空行不单独触发一条无法合成的请求。
            if not spoken or all(char.isspace() or category(char).startswith("P") for char in spoken):
                part.complete = True
                continue
            while self._pcm.size(state) >= self._prefetch:
                await asyncio.sleep(_TICK)
            if state.signal.done() or reply.abandoned:
                return
            async with aclosing(self._tts.synthesize(spoken, sample_rate=self._settings.playback_rate)) as pcm:
                async for chunk in pcm:
                    if not chunk or len(chunk) % 2 or len(chunk) > _MAX_PCM_CHUNK:
                        raise TtsError("tts_invalid_pcm_chunk")
                    remaining = memoryview(chunk)
                    while remaining:
                        if state.signal.done() or reply.abandoned:
                            return
                        consumed = self._pcm.offer(state, part, remaining)
                        remaining = remaining[consumed:]
                        if remaining:
                            await asyncio.sleep(_TICK)
            if not part.received:
                raise TtsError("tts_empty_audio")
            part.complete = True
        while self._pcm.size(state):
            await asyncio.sleep(_TICK)
        if reply.overflowed:
            raise TtsError("voice_reply_text_limit")

    async def close(self) -> None:
        self._closed = True
        self.interrupt()
        workers = tuple(self._workers)
        for task in workers:
            task.cancel()
        if workers:
            await asyncio.wait(workers, timeout=1)
