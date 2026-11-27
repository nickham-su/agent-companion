"""按句调度流式 TTS；设备回调只拷贝帧并累加已交设备的进度。"""

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
import json
import logging
from threading import Lock
from unicodedata import category

from ..config import VoiceSettings
from ..protocol import DeliveryStatus, Receipt, ReplyStream
from .tts import StreamingTTS, TtsError

_LOG = logging.getLogger(__name__)
_MAX_SEGMENT = 200
_MIN_SEGMENT_BODY = 30
_MAX_REPLY_TEXT = 16_000
_MAX_PCM_CHUNK = 512 * 1024
_TICK = 0.008
_BOUNDARIES = frozenset("。！？.!?\n")


def _is_body_char(char: str) -> bool:
    return not char.isspace() and not category(char).startswith("P")


def _elapsed_ms(start: float | None, end: float | None) -> str:
    return "-" if start is None or end is None else f"{(end - start) * 1000:.0f}"


async def _segments(reply: ReplyStream) -> AsyncGenerator[str, None]:
    pending = []
    body_chars = 0
    first = True
    async for delta in reply:
        for char in delta:
            pending.append(char)
            body_chars += _is_body_char(char)
            # 纯标点和空行保留在原文中，但不占用首段，也不单独请求合成。
            ready = first or body_chars == 0 or body_chars >= _MIN_SEGMENT_BODY
            if len(pending) >= _MAX_SEGMENT or (char in _BOUNDARIES and ready):
                yield "".join(pending)
                if body_chars:
                    first = False
                pending.clear()
                body_chars = 0
    if pending:
        yield "".join(pending)


@dataclass(eq=False, slots=True)
class _Segment:
    text: str
    received: int = 0
    completed_at: float | None = None
    end: int = 0  # 完整合成后，所在音频在本回复中的末尾偏移


@dataclass(eq=False, slots=True)
class _PlaybackReply:
    signal: asyncio.Future[DeliveryStatus]
    clock: Callable[[], float]
    hard_deadline: float
    segments: list[_Segment] = field(default_factory=list)
    first_tts_at: float | None = None
    first_pcm_at: float | None = None
    # 由 PCM 锁保护；一旦过期，设备不能再推进本条回复的证据。
    played_bytes: int = 0
    first_output_at: float | None = None
    idle_deadline: float = float("inf")
    idle_timeout: float = 0.0
    offered_bytes: int = 0
    stopped_at: float | None = None
    worker_finished_at: float | None = None
    draining: bool = False  # 合成完毕，只剩已经排队的 PCM。

    def terminal_at(self, now: float) -> float | None:
        events = (self.worker_finished_at, self.stopped_at,
                  self.hard_deadline if now >= self.hard_deadline else None)
        return min((event for event in events if event is not None), default=None)

    def worker_finished_first(self) -> bool:
        return (self.worker_finished_at is not None and self.worker_finished_at < self.hard_deadline
                and (self.stopped_at is None or self.worker_finished_at < self.stopped_at))

    def complete(self, part: _Segment) -> None:
        self.offered_bytes += part.received
        part.end = self.offered_bytes
        part.completed_at = self.clock()

    def confirmed_prefix(self, played_bytes: int, cutoff: float) -> str:
        if not played_bytes:
            return ""
        end = 0
        for index, part in enumerate(self.segments, start=1):
            if part.completed_at is None or part.completed_at >= cutoff or part.end > played_bytes:
                break
            # 未独立合成的标点/空白只作为后续已交付段落之前的格式保留。
            if part.received:
                end = index
        return "".join(part.text for part in self.segments[:end])


class _PcmQueue:
    def __init__(self, capacity: int) -> None:
        self._lock = Lock()
        self._packets: deque[memoryview] = deque()
        self._active: _PlaybackReply | None = None
        self._queued = 0
        self._capacity = capacity

    def attach(self, state: _PlaybackReply) -> None:
        with self._lock:
            self._active = state

    def detach(self, state: _PlaybackReply, *, stopping: bool = False) -> bool:
        with self._lock:
            if self._active is not state:
                return False
            was_playing = False
            if stopping:
                now = state.stopped_at = state.clock()
                # 与设备消费原子取证；有合成任务不等于已经播出过音频。
                was_playing = (state.played_bytes > 0 and state.worker_finished_at is None
                               and now < min(state.hard_deadline, state.idle_deadline)
                               and (not state.draining or self._queued > 0))
            self._active = None
            self._packets.clear()
            self._queued = 0
            return was_playing

    def size(self, state: _PlaybackReply) -> int:
        with self._lock:
            return self._queued if self._active is state else 0

    def arm_idle(self, state: _PlaybackReply, timeout: float, deadline: float | None) -> float:
        with self._lock:
            state.idle_timeout = timeout
            state.idle_deadline = state.clock() + timeout if deadline is None else deadline
            return state.idle_deadline

    def idle_snapshot(self, state: _PlaybackReply) -> tuple[float, float, str, float | None]:
        with self._lock:
            now, deadline, played = state.clock(), state.idle_deadline, state.played_bytes
        terminal = state.terminal_at(now)
        # 扫描既有、有上限的段落表，不持 PCM 锁；段落只在同一事件循环更新。
        expired = now >= deadline and (terminal is None or terminal >= deadline)
        prefix = state.confirmed_prefix(played, deadline) if expired else ""
        return now, deadline, prefix, terminal

    def offer(self, state: _PlaybackReply, part: _Segment, pcm: memoryview) -> int:
        with self._lock:
            if self._active is not state:
                raise asyncio.CancelledError  # 防止旧响应在网络取消时迟到的块碰到新响应
            size = min(len(pcm), self._capacity - self._queued)
            if size:
                self._packets.append(pcm[:size])
                self._queued += size
                part.received += size
            return size

    def fill(self, output: memoryview) -> None:
        output = output.cast("B")
        output[:] = bytes(len(output))
        if not self._lock.acquire(blocking=False):
            return  # 与事件循环冲突时只补静音，设备线程不等待
        try:
            state = self._active
            if state is None or not self._packets:
                return
            # 一次设备回调的消费在 PCM 锁内取同一个时刻；过期后只补静音。
            now = state.clock()
            if now >= min(state.hard_deadline, state.idle_deadline):
                return
            offset = 0
            while offset < len(output) and self._packets:
                pcm = self._packets[0]
                size = min(len(output) - offset, len(pcm))
                output[offset:offset + size] = pcm[:size]
                self._queued -= size
                offset += size
                if size == len(pcm):
                    self._packets.popleft()
                else:
                    self._packets[0] = pcm[size:]
            if offset:
                if not state.played_bytes:
                    state.first_output_at = now  # 回调只记时，日志留给事件循环。
                state.played_bytes += offset
                if state.idle_timeout:
                    state.idle_deadline = now + state.idle_timeout
        finally:
            self._lock.release()


class VoiceOutput:
    """面向一个设备输出；中断只结束交付，不取消 Runtime 的模型或工具。"""

    def __init__(self, tts: StreamingTTS, settings: VoiceSettings) -> None:
        self._tts = tts
        self._settings = settings
        sample_rate = settings.output_sample_rate
        self._prefetch = max(1, round(settings.prefetch_seconds * 2 * sample_rate))
        self._pcm = _PcmQueue(2 * round(settings.buffer_seconds * sample_rate))
        self._active: _PlaybackReply | None = None
        self._workers: set[asyncio.Task[None]] = set()
        self._closed = False

    def render(self, output: memoryview) -> None:
        self._pcm.fill(output)

    def interrupt(self) -> bool:
        return self._stop(DeliveryStatus.PARTIAL)

    def fail(self) -> None:
        self._closed = True
        self._stop(DeliveryStatus.FAILED)

    def _stop(self, status: DeliveryStatus) -> bool:
        state = self._active
        if state is not None and not state.signal.done():
            was_playing = self._pcm.detach(state, stopping=True)  # 不等待 HTTP 或回执任务
            state.signal.set_result(status)
            return was_playing
        return False

    def _finished(self, task: asyncio.Task[None]) -> None:
        self._workers.discard(task)
        if not task.cancelled():
            task.exception()  # 回收已脱离本回复的网络任务异常

    async def deliver(self, reply: ReplyStream) -> Receipt:
        reply.select_voice_delivery()
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
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + self._settings.delivery_timeout
        _LOG.debug("voice playback started; reply=%s", reply_id)
        state = _PlaybackReply(loop.create_future(), loop.time, deadline)
        reply.bind_playback(lambda timeout, cutoff: self._pcm.arm_idle(state, timeout, cutoff),
                            lambda: self._pcm.idle_snapshot(state))
        self._active = state
        self._pcm.attach(state)

        async def produce() -> None:
            try:
                await self._produce(reply, state)
            finally:
                state.worker_finished_at = state.clock()

        worker = asyncio.create_task(produce())
        self._workers.add(worker)
        worker.add_done_callback(self._finished)
        status: DeliveryStatus
        try:
            while not worker.done() and not state.signal.done() and loop.time() < deadline:
                await asyncio.wait((worker, state.signal), timeout=max(0, deadline - loop.time()),
                                   return_when=asyncio.FIRST_COMPLETED)
            # 任务和停播信号可能在同一次唤醒前均已就绪；比较事件时点，不比较观察顺序。
            if state.worker_finished_first():
                cancelled = worker.cancelled()
                error = None if cancelled else worker.exception()
                if not cancelled and error is None:
                    status = DeliveryStatus.FULL
                else:
                    status = DeliveryStatus.FAILED
                    _LOG.warning("voice synthesis failed (%s); reply=%s",
                                 error.diagnostic if isinstance(error, TtsError) else "tts_failed", reply_id)
            elif state.stopped_at is not None and state.stopped_at < deadline:
                status = state.signal.result()
                if status == DeliveryStatus.PARTIAL:
                    _LOG.debug("voice playback interrupted; reply=%s", reply_id)
            else:
                status = DeliveryStatus.UNKNOWN
                _LOG.warning("voice delivery deadline reached; reply=%s", reply_id)
        finally:
            self._pcm.detach(state)
            self._active = None
            reply.abandon()
            if not worker.done() and not worker.cancelling():
                worker.cancel()
            if not state.signal.done():
                state.signal.cancel()
            # 摘除后首包设备时间不会再变；失败/打断时也保留已经到达的阶段。
            if _LOG.isEnabledFor(logging.DEBUG):
                _LOG.debug("voice first audio latency; reply=%s first_text_to_tts_ms=%s tts_to_first_pcm_ms=%s "
                           "first_pcm_to_output_ms=%s first_text_to_output_ms=%s", reply_id,
                           _elapsed_ms(reply.first_text_at, state.first_tts_at),
                           _elapsed_ms(state.first_tts_at, state.first_pcm_at),
                           _elapsed_ms(state.first_pcm_at, state.first_output_at),
                           _elapsed_ms(reply.first_text_at, state.first_output_at))
        # 摘除后设备不会再推进字节数。非完整回执只确认合成完成且已全量交给设备的连续段。
        if status == DeliveryStatus.FULL:
            prefix = "".join(part.text for part in state.segments) if state.played_bytes else ""
        else:
            cutoff = state.terminal_at(loop.time())
            assert cutoff is not None
            prefix = state.confirmed_prefix(state.played_bytes, cutoff)
        if not prefix and status in (DeliveryStatus.PARTIAL, DeliveryStatus.FULL):
            status = DeliveryStatus.ZERO
        _LOG.debug("voice playback finished; reply=%s status=%s content=%s elapsed_ms=%.0f",
                   reply_id, status, json.dumps(prefix, ensure_ascii=False), (loop.time() - started) * 1000)
        return Receipt(reply_id, status, prefix)

    async def _produce(self, reply: ReplyStream, state: _PlaybackReply) -> None:
        async for text in _segments(reply):
            if state.signal.done() or reply.abandoned:
                raise asyncio.CancelledError
            part = _Segment(text)
            state.segments.append(part)
            spoken = text.strip()
            # 一串连续句末标点或空行不单独触发一条无法合成的请求。
            if not any(_is_body_char(char) for char in spoken):
                state.complete(part)
                continue
            while self._pcm.size(state) >= self._prefetch:
                await asyncio.sleep(_TICK)
            if state.signal.done() or reply.abandoned:
                raise asyncio.CancelledError
            if state.first_tts_at is None:
                state.first_tts_at = state.clock()
            async with aclosing(self._tts.synthesize(spoken, sample_rate=self._settings.output_sample_rate)) as pcm:
                async for chunk in pcm:
                    if not chunk or len(chunk) % 2 or len(chunk) > _MAX_PCM_CHUNK:
                        raise TtsError("tts_invalid_pcm_chunk")
                    if state.first_pcm_at is None:
                        state.first_pcm_at = state.clock()
                    remaining = memoryview(chunk)
                    while remaining:
                        if state.signal.done() or reply.abandoned:
                            raise asyncio.CancelledError
                        consumed = self._pcm.offer(state, part, remaining)
                        remaining = remaining[consumed:]
                        if remaining:
                            await asyncio.sleep(_TICK)
            if not part.received:
                raise TtsError("tts_empty_audio")
            state.complete(part)
        state.draining = True
        while self._pcm.size(state):
            await asyncio.sleep(_TICK)
        if state.signal.done() or reply.abandoned:
            raise asyncio.CancelledError  # 流提前关闭或设备队列被清空都不代表生产成功。
        if reply.overflowed:
            raise TtsError("voice_reply_text_limit")

    async def close(self, *, wait_for_cleanup: bool = True) -> None:
        self._closed = True
        self.interrupt()
        workers = tuple(self._workers)
        for task in workers:
            if not task.cancelling():
                task.cancel()
        if wait_for_cleanup and workers:
            await asyncio.wait(workers, timeout=1)
