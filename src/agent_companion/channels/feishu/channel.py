"""飞书私聊按时间窗口轮询，文字回复、转写回显和通知独立发送。"""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import logging
from sqlite3 import OperationalError
from time import time

from ...protocol import ChannelUser, DeliveryStatus, Message, Receipt, ReplyDelivery, ReplyInfo, ReplyStream
from ...speech.asr import FileASR
from ...storage.sqlite import History, is_sqlite_lock_error
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
_MAX_PENDING_REPLIES = 8
_MAX_PENDING_TRANSCRIPTS = 8


class FeishuChannel:
    def __init__(self, cli: FeishuCLI, *,
                 select_voice: Callable[[ReplyStream], ReplyDelivery | None] | None = None) -> None:
        self._cli = cli
        self._select_voice = select_voice
        self._pending: deque[tuple[str, str]] = deque()
        self._sender: asyncio.Task[None] | None = None
        self._transcripts: deque[bytes] = deque()
        self._transcript_sender: asyncio.Task[None] | None = None
        self._closed = False

    def select_delivery(self, reply: ReplyStream) -> ReplyDelivery:
        if not self._closed and self._select_voice is not None:
            selected = self._select_voice(reply)
            if selected is not None:
                return selected
        return ReplyDelivery(self.deliver, after_commit=self.enqueue, require_complete_model=True)

    async def deliver(self, reply: ReplyStream) -> Receipt:
        """回执仅确认全文已收齐；实际发送必须等待历史提交。"""
        if self._closed:
            reply.abandon()
            return Receipt(reply.info.reply_id, DeliveryStatus.ZERO)
        text = "".join([segment async for segment in reply])
        if self._closed or reply.abandoned:
            return Receipt(reply.info.reply_id, DeliveryStatus.ZERO)
        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)

    def enqueue(self, info: ReplyInfo, text: str) -> None:
        """将已落库的文字正文交给发送队列，也可用于语音确认正文的副本。"""
        if self._closed or len(self._pending) >= _MAX_PENDING_REPLIES:
            _LOG.warning("Feishu text reply dropped (%s); reply=%s",
                         "closed" if self._closed else "queue_full", info.reply_id)
            return
        self._pending.append((info.reply_id, text))
        _LOG.debug("Feishu text reply queued; reply=%s chars=%d pending=%d",
                   info.reply_id, len(text), len(self._pending))
        if self._sender is None:
            self._sender = asyncio.create_task(self._send_pending())

    async def _send_pending(self) -> None:
        clock = asyncio.get_running_loop().time
        try:
            while not self._closed and self._pending:
                reply_id, text = self._pending.popleft()
                started = clock()
                try:
                    if await self._cli.send_text(text) != DeliveryStatus.FULL:
                        _LOG.warning("Feishu text send not confirmed; reply=%s", reply_id)
                    else:
                        _LOG.debug("Feishu text send confirmed; reply=%s elapsed_ms=%.0f",
                                   reply_id, (clock() - started) * 1000)
                except Exception:
                    _LOG.warning("Feishu text send failed; reply=%s", reply_id)
        finally:
            self._sender = None

    def enqueue_transcript(self, text: str) -> None:
        """回显已接收的转写，不等待发送，也不占用模型回复队列。"""
        if self._closed:
            return
        if len(self._transcripts) >= _MAX_PENDING_TRANSCRIPTS:
            _LOG.warning("Feishu transcript dropped (queue_full)")
            return
        payload = self._cli.prepare_transcript(text)
        if payload is None:
            _LOG.warning("Feishu transcript dropped (invalid_payload)")
            return
        self._transcripts.append(payload)
        if self._transcript_sender is None:
            self._transcript_sender = asyncio.create_task(self._send_transcripts())

    async def _send_transcripts(self) -> None:
        try:
            while not self._closed and self._transcripts:
                payload = self._transcripts.popleft()
                try:
                    status = await self._cli.send_prepared_transcript(payload)
                    if status != DeliveryStatus.FULL:
                        _LOG.warning("Feishu transcript send not confirmed; status=%s", status)
                except Exception:
                    _LOG.warning("Feishu transcript send failed")
        finally:
            self._transcript_sender = None

    def stop_accepting(self) -> None:
        """在让出事件循环前关停发送入口和在途任务。"""
        if self._closed:
            return
        self._closed = True
        if self._pending:
            _LOG.warning("Feishu text replies dropped at shutdown; count=%d", len(self._pending))
            self._pending.clear()
        if self._sender is not None:
            self._sender.cancel()
        if self._transcripts:
            _LOG.warning("Feishu transcripts dropped at shutdown; count=%d", len(self._transcripts))
            self._transcripts.clear()
        if self._transcript_sender is not None:
            self._transcript_sender.cancel()

    async def close(self) -> None:
        self.stop_accepting()
        senders = tuple(task for task in (self._sender, self._transcript_sender) if task is not None)
        if senders:
            await asyncio.gather(*senders, return_exceptions=True)
        self._sender = None
        self._transcript_sender = None

    async def notify_audio_failure(self) -> None:
        if self._closed:
            return
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
    prepared: tuple[list[tuple[str, Message | None]], list[str]] | None = None

    def restart(self, *, end: int | None = None, page_size: int | None = None) -> "_ScanWindow":
        return _ScanWindow(
            start=self.start,
            end=self.end if end is None else end,
            through=self.through,
            page_size=self.page_size if page_size is None else page_size,
        )


class FeishuPoller:
    def __init__(self, cli: FeishuCLI, asr: FileASR, channel: FeishuChannel, *,
                 chat_id: str, user: ChannelUser, account: ChannelUser, session_id: str, history: History,
                 submit: Callable[[int], Awaitable[None] | None], private_started_ms: int,
                 clock: Callable[[], float] = time,
                 batch_budget: int = MAX_BATCH_BYTES, before_close: Callable[[], None] | None = None) -> None:
        self._cli = cli
        self._normalizer = MessageNormalizer(cli, asr)
        self._batch_budget = batch_budget
        self._channel = channel
        self._chat_id = chat_id
        self._user = user
        self._account = account
        self._session_id = session_id
        self._private_started_ms = private_started_ms
        self._source = "feishu-private"
        self._history = history
        self._submit = submit
        self._before_close = before_close
        self._clock = clock
        self._poll_lock = asyncio.Lock()
        self._notices: set[asyncio.Task[None]] = set()
        self._window: _ScanWindow | None = None

    async def _submit_pending(self) -> None:
        for batch_id in self._history.pending_inbound(self._session_id, self._source):
            result = self._submit(batch_id)
            if result is not None:
                await result

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
                message = incoming(item, chat_id=self._chat_id, user=self._user, account=self._account)
                if (message is not None and window.start * 1000 <= message.created_ms < window.end * 1000
                        and message.created_ms >= self._private_started_ms
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

    async def _prepare(self, window: _ScanWindow) -> tuple[list[tuple[str, Message | None]], list[str]]:
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
        return media_budget.finish(received), failed_audio

    async def poll_once(self) -> None:
        async with self._poll_lock:
            # 恢复上次进程在「接收已落库、仍未移交历史」时留下的批次。
            await self._submit_pending()
            if self._window is None:
                now = int(self._clock())
                initial = self._private_started_ms // 1000
                floor, through = self._history.inbound_window(self._session_id, self._source, initial)
                start = max(floor, through - _OVERLAP_SECONDS)
                if now <= start:
                    return
                self._window = _ScanWindow(start, now, through)
            window = self._window
            if not await self._collect(window):
                return
            started = asyncio.get_running_loop().time()
            if window.prepared is None:
                # 写入遇锁不丢窗口；下轮重试同一批，不重复下载与转写。
                window.prepared = await self._prepare(window)
            normalized_items, failed_audio = window.prepared
            recorded = self._history.record_inbound(self._session_id, self._source, window.end, normalized_items)
            if normalized_items and _LOG.isEnabledFor(logging.DEBUG):
                accepted = sum(message is not None and key in recorded for key, message in normalized_items)
                _LOG.debug("Feishu input window recorded; candidates=%d accepted=%d audio_failures=%d elapsed_ms=%.0f",
                           len(normalized_items), accepted, len(failed_audio),
                           (asyncio.get_running_loop().time() - started) * 1000)
            self._window = None
            for message_id, message in normalized_items:
                if (message_id in recorded and window.messages[message_id].kind == "audio"
                        and message is not None and isinstance(message.content, str)):
                    self._channel.enqueue_transcript(message.content)
            try:
                await self._submit_pending()
            finally:
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
                except OperationalError as exc:
                    if not is_sqlite_lock_error(exc):
                        _LOG.error("Feishu private inbox storage failed; sqlite_code=%s",
                                   getattr(exc, "sqlite_errorcode", None))
                        raise
                    _LOG.warning("Feishu private inbox storage busy; retrying on next poll")
                next_tick += _POLL_SECONDS
                now = loop.time()
                if next_tick < now:
                    next_tick += (int((now - next_tick) // _POLL_SECONDS) + 1) * _POLL_SECONDS
                await asyncio.sleep(max(0, next_tick - loop.time()))
        finally:
            try:
                if self._before_close is not None:
                    self._before_close()  # 入口与发送先同步关闭，再等待固定通知的取消清理。
            finally:
                await self.close()

    async def close(self) -> None:
        pending = tuple(self._notices)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
