"""飞书按固定窗口轮询；通知独立发送，模型输出按启动模式路由。"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
import json
import logging
from time import time

from ...protocol import Channel, DeliveryStatus, Receipt, ReplyStream
from ...speech.asr import FileASR
from ...storage.sqlite import History
from .cli import MAX_PAGE_SIZE, MIN_PAGE_SIZE, CliError, CliStdoutLimit, FeishuCLI
from .messages import MAX_BATCH_BYTES, NORMALIZE_ERRORS, BatchMediaBudget, IncomingMessage, MessageNormalizer, incoming


_LOG = logging.getLogger(__name__)
_POLL_SECONDS = 10
_OVERLAP_SECONDS = 10
_MAX_PAGES = 200
_MAX_WINDOW_PAGES = 1_000
_MAX_WINDOW_MESSAGES = 20_000
_MAX_WINDOW_RAW_BYTES = 32 * 1024 * 1024
_VOICE_FAILURE = "语音接收失败，请重新发送。"


class FeishuChannel:
    def __init__(self, cli: FeishuCLI, *, mode: str, voice_output: Channel | None = None) -> None:
        if mode not in ("text", "voice") or (mode == "voice" and voice_output is None):
            raise ValueError("voice mode requires an audio output")
        self._cli = cli
        self._voice_output = voice_output if mode == "voice" else None

    async def deliver(self, reply: ReplyStream) -> Receipt:
        if self._voice_output is not None:
            return await self._voice_output.deliver(reply)
        text = "".join([part async for part in reply])
        if not text or reply.abandoned:
            return Receipt(reply.info.reply_id, DeliveryStatus.ZERO)
        status = await self._cli.send_text(text)
        return Receipt(reply.info.reply_id, status, text if status == DeliveryStatus.FULL else "")

    async def notify_audio_failure(self) -> None:
        if await self._cli.send_text(_VOICE_FAILURE) != DeliveryStatus.FULL:
            _LOG.warning("Feishu audio failure notice not confirmed")


@dataclass(slots=True)
class _ScanWindow:
    start: int
    end: int
    through: int
    page_size: int = MAX_PAGE_SIZE
    token: str | None = None
    tokens: set[str] = field(default_factory=set)
    messages: dict[str, IncomingMessage] = field(default_factory=dict)
    raw_bytes: int = 0
    completed: bool = False
    failures: int = 0

    def restart(self, *, end: int | None = None, page_size: int | None = None) -> "_ScanWindow":
        return _ScanWindow(
            start=self.start,
            end=self.end if end is None else end,
            through=self.through,
            page_size=self.page_size if page_size is None else page_size,
        )


class FeishuPoller:
    def __init__(self, cli: FeishuCLI, asr: FileASR, channel: FeishuChannel, *,
                 chat_id: str, sender_id: str, own_id: str, session_id: str, history: History,
                 submit: Callable[[int], None], clock: Callable[[], float] = time,
                 batch_budget: int = MAX_BATCH_BYTES) -> None:
        self._cli = cli
        self._normalizer = MessageNormalizer(cli, asr)
        self._batch_budget = batch_budget
        self._channel = channel
        self._chat_id = chat_id
        self._sender_id = sender_id
        self._own_id = own_id
        self._session_id = session_id
        self._source = json.dumps(["feishu", chat_id, sender_id, own_id])
        self._history = history
        self._submit = submit
        self._clock = clock
        self._poll_lock = asyncio.Lock()
        self._notices: set[asyncio.Task[None]] = set()
        self._window: _ScanWindow | None = None

    def _submit_pending(self) -> None:
        for batch_id in self._history.pending_inbound(self._session_id, self._source):
            self._submit(batch_id)

    def _narrow_window(self, window: _ScanWindow) -> None:
        # 只有完整读取的时间范围才提交水位；超出总状态上限时可从较短范围重读。
        base = max(window.start, window.through)
        self._window = (window.restart(end=base + (window.end - base) // 2)
                        if window.end > base + 1 else None)
        raise CliError("cli_window_limit")

    async def _collect(self, window: _ScanWindow) -> bool:
        if window.completed:
            return True
        for _ in range(_MAX_PAGES):
            try:
                items, token = await self._cli.list_page(
                    window.start, window.end, window.token, page_size=window.page_size,
                )
            except CliStdoutLimit:
                if window.page_size == MIN_PAGE_SIZE:
                    raise CliError("cli_single_result_output_limit") from None
                # 游标绑定原页大小，必须丢弃未提交的内容，从同一时间窗口首页重读。
                self._window = window.restart(page_size=max(MIN_PAGE_SIZE, window.page_size // 2))
                return False
            except CliError:
                window.failures += 1
                if window.token is not None and window.failures >= 3:
                    # 过期游标等持续错误从固定窗口开头重试；不丢已提交水位。
                    self._window = window.restart()
                raise
            window.failures = 0
            if token is not None and token in window.tokens:
                self._window = window.restart()
                raise CliError("cli_page_cycle")
            for item in items:
                message = incoming(item, chat_id=self._chat_id, sender_id=self._sender_id, own_id=self._own_id)
                if (message is not None and window.start * 1000 <= message.created_ms < window.end * 1000
                        and message.message_id not in window.messages
                        and not self._history.inbound_seen(self._session_id, self._source, message.message_id)):
                    size = 4 * len(message.content)  # Python 字符串每个码点至多占 4 字节。
                    if (len(window.messages) >= _MAX_WINDOW_MESSAGES
                            or window.raw_bytes + size > _MAX_WINDOW_RAW_BYTES):
                        self._narrow_window(window)
                    window.raw_bytes += size
                    window.messages[message.message_id] = message
            if token is None:
                window.completed = True
                return True
            if len(window.tokens) >= _MAX_WINDOW_PAGES - 1:
                self._narrow_window(window)
            window.token = token
            window.tokens.add(token)
        return False  # 本次调用到页预算，下个刻度仍查询同一窗口和后续页。

    async def poll_once(self) -> None:
        async with self._poll_lock:
            # 恢复上次进程在「接收已落库、仍未移交历史」时留下的批次。
            self._submit_pending()
            if self._window is None:
                now = int(self._clock())
                floor, through = self._history.inbound_window(self._session_id, self._source, now)
                start = max(floor, through - _OVERLAP_SECONDS)
                if now <= start:
                    return
                self._window = _ScanWindow(start, now, through)
            window = self._window
            if not await self._collect(window):
                return
            received = []
            media_budget = BatchMediaBudget(self._batch_budget)
            failed_audio = []
            # 稳定排序，时间相同则保留服务端第一次见到的相对顺序。
            for raw in sorted(window.messages.values(), key=lambda message: message.created_ms):
                try:
                    normalized = await self._normalizer.normalize(raw, media_budget)
                except NORMALIZE_ERRORS:
                    normalized = None
                    if raw.kind == "audio":
                        failed_audio.append(raw.message_id)
                    else:
                        _LOG.warning("Feishu input could not be normalized")
                received.append((raw.message_id, normalized))
            recorded = self._history.record_inbound(self._session_id, self._source, window.end, media_budget.finish(received))
            self._window = None
            self._submit_pending()
            for message_id in failed_audio:
                if message_id in recorded:
                    task = asyncio.create_task(self._notify())
                    self._notices.add(task)
                    task.add_done_callback(self._notices.discard)

    async def _notify(self) -> None:
        try:
            await self._channel.notify_audio_failure()
        except Exception:
            _LOG.warning("Feishu audio failure notice unavailable")

    async def run(self) -> None:
        """不重叠轮询，长批次完成后跳过错过的时钟刻度。"""
        loop = asyncio.get_running_loop()
        next_tick = loop.time()
        try:
            while True:
                try:
                    await self.poll_once()
                except CliError as exc:
                    _LOG.warning("Feishu poll failed (%s)", exc)
                next_tick += _POLL_SECONDS
                now = loop.time()
                if next_tick < now:
                    next_tick += (int((now - next_tick) // _POLL_SECONDS) + 1) * _POLL_SECONDS
                await asyncio.sleep(max(0, next_tick - loop.time()))
        finally:
            await self.close()

    async def close(self) -> None:
        pending = tuple(self._notices)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
