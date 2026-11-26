"""飞书账号统一搜索私聊，文字回复、转写回显和通知独立发送。"""

import asyncio
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import logging
from sqlite3 import OperationalError
from time import time
from typing import Literal

from ...protocol import ChannelUser, DeliveryStatus, Receipt, ReplyDelivery, ReplyInfo, ReplyStream
from ...speech.asr import FileASR
from ...storage.sqlite import FeishuCandidate, FeishuRaw, History, is_sqlite_lock_error
from .cli import (DEFAULT_SEARCH_PAGE_SIZE, MAX_MGET_SIZE, MIN_PAGE_SIZE, CliError, CliStdoutLimit, FeishuCLI,
                  FeishuDiscovery, _valid_id, parse_open_id)
from .messages import (MAX_BATCH_BYTES, NORMALIZE_ERRORS, SUPPORTED_TYPES, BatchMediaBudget, IncomingMessage,
                       MessageNormalizer, incoming)


_LOG = logging.getLogger(__name__)
_LOG_SEARCH_WINDOWS = False  # 逐轮搜索摘要只在临时排查时启用。
_POLL_SECONDS = 10
_OVERLAP_SECONDS = 120
_MAX_DETAIL_BATCHES = 10
_MAX_SCAN_SECONDS = 3600
_MAX_PAGES = 10
_MAX_WINDOW_PAGES = 1_000
_MAX_WINDOW_MESSAGES = 20_000
_MAX_WINDOW_RAW_BYTES = 32 * 1024 * 1024
_VOICE_FAILURE = "语音接收失败，请重新发送。"
_MAX_PENDING_REPLIES = 8
_MAX_PENDING_TRANSCRIPTS = 8
_MAX_PENDING_NOTICES = 8
_MAX_RECOVERY_BATCH = 20
_DENSE_WINDOW_RETRY_SECONDS = 60
_MAX_MEDIA_WORKERS = 4
_STORAGE_RETRY_SECONDS = 0.25


class FeishuChannel:
    def __init__(self, cli: FeishuCLI, *,
                 select_voice: Callable[[ReplyStream], ReplyDelivery | None] | None = None) -> None:
        self._cli = cli
        self._select_voice = select_voice
        self._pending: deque[tuple[str, str]] = deque()
        self._sender: asyncio.Task[None] | None = None
        self._transcripts: deque[bytes] = deque()
        self._transcript_sender: asyncio.Task[None] | None = None
        self._pending_notices = 0
        self._notice_sender: asyncio.Task[None] | None = None
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

    def enqueue_audio_failure(self) -> None:
        if self._closed:
            return
        if self._pending_notices >= _MAX_PENDING_NOTICES:
            _LOG.warning("Feishu audio failure notice dropped (queue_full)")
            return
        self._pending_notices += 1
        if self._notice_sender is None:
            self._notice_sender = asyncio.create_task(self._send_audio_notices())

    async def _send_audio_notices(self) -> None:
        try:
            while not self._closed and self._pending_notices:
                self._pending_notices -= 1
                try:
                    await self.notify_audio_failure()
                except Exception:
                    _LOG.warning("Feishu audio failure notice unavailable")
        finally:
            self._notice_sender = None

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
        self._pending_notices = 0
        if self._notice_sender is not None:
            self._notice_sender.cancel()

    async def close(self) -> None:
        self.stop_accepting()
        senders = tuple(task for task in (self._sender, self._transcript_sender, self._notice_sender)
                        if task is not None)
        if senders:
            await asyncio.gather(*senders, return_exceptions=True)
        self._sender = None
        self._transcript_sender = None
        self._notice_sender = None

    async def notify_audio_failure(self) -> None:
        if self._closed:
            return
        if await self._cli.send_text(_VOICE_FAILURE) != DeliveryStatus.FULL:
            _LOG.warning("Feishu audio failure notice not confirmed")


def private_session_id(user: ChannelUser) -> str:
    return "feishu-private:" + sha256(user.key.encode("ascii")).hexdigest()


def _chat_digest(chat_id: str) -> str:
    return sha256(chat_id.encode("ascii")).hexdigest()


_FilterReason = Literal["non_p2p", "unsupported_type", "bot_sender", "same_account", "out_of_window"]


def _search_hit(row: object, account: ChannelUser) -> FeishuCandidate | _FilterReason:
    meta = row.get("meta_data") if isinstance(row, dict) else None
    if not isinstance(meta, dict) or type(meta.get("is_p2p_chat")) is not bool:
        raise CliError("cli_invalid_search_hit")
    if not meta["is_p2p_chat"]:
        return "non_p2p"
    kind = meta.get("type")
    if not isinstance(kind, str) or not kind:
        raise CliError("cli_invalid_search_hit")
    # 搜索可能返回大写类型（TEXT），正文接口使用小写（text）。
    kind = kind.lower()
    if kind not in SUPPORTED_TYPES:
        return "unsupported_type"
    sender = meta.get("from_id")
    if _valid_id(sender, "cli") or _valid_id(sender, "bot"):
        return "bot_sender"
    user = parse_open_id(sender)
    if user == account:
        return "same_account"
    mid, chat, stamp = meta.get("message_id"), meta.get("chat_id"), meta.get("create_time")
    if user is None or not _valid_id(mid, "om") or not _valid_id(chat, "oc") or not isinstance(stamp, str):
        raise CliError("cli_invalid_search_hit")
    try:
        dt = datetime.fromisoformat(stamp)
        if dt.tzinfo is None:
            raise ValueError
        created = int(dt.timestamp())
    except (ValueError, OverflowError, OSError):
        raise CliError("cli_invalid_search_time") from None
    return FeishuCandidate(mid, user.key, _chat_digest(chat), kind, created)


class FeishuInput:
    """每个用户的顺序工作任务调用；媒体并发受限，纯文字不占媒体槽。"""

    def __init__(self, discovery: FeishuDiscovery, asr: FileASR, history: History, *,
                 batch_budget: int = MAX_BATCH_BYTES) -> None:
        self._normalizer = MessageNormalizer(discovery, asr)
        self._history = history
        self._batch_budget = batch_budget
        self._media_slots = asyncio.Semaphore(_MAX_MEDIA_WORKERS)

    async def process_next(self, user: ChannelUser, channel: FeishuChannel, dispatch: Callable[[], None]) -> bool:
        pending = self._history.next_feishu_raw(user.key)
        if pending is None:
            return False
        batch_id, has_media = pending
        # 持有媒体槽直到提交，避免 SQLite 争用期间积累过多已展开的批次。
        async with self._media_slots if has_media else nullcontext():
            budget = BatchMediaBudget(self._batch_budget)
            received = []
            audio = []
            for row in self._history.read_feishu_raw(user.key, batch_id):
                raw = IncomingMessage(*row)
                normalized = None
                try:
                    normalized = await self._normalizer.normalize(raw, budget)
                except NORMALIZE_ERRORS:
                    if raw.kind != "audio":
                        _LOG.warning("Feishu input could not be normalized")
                if raw.kind == "audio":
                    text = normalized.content if normalized is not None else None
                    audio.append(text if isinstance(text, str) else None)
                received.append((raw.message_id, normalized))
            messages = [msg for _, msg in budget.finish(received) if msg is not None]
            warned = False
            while True:
                try:
                    self._history.finish_feishu_raw(user.key, batch_id, private_session_id(user), messages)
                    break
                except OperationalError as exc:
                    if not is_sqlite_lock_error(exc):
                        raise
                    if not warned:
                        _LOG.warning("Feishu user input storage busy; retrying prepared batch")
                        warned = True
                    await asyncio.sleep(_STORAGE_RETRY_SECONDS)
            for text in audio:
                if text is None:
                    channel.enqueue_audio_failure()
                else:
                    channel.enqueue_transcript(text)
            dispatch()  # 已就绪 Session 在提交归一化输入的同一事件循环刻度交接。
        return True


@dataclass(slots=True, repr=False)
class _ScanWindow:
    start: int
    end: int
    page_size: int = DEFAULT_SEARCH_PAGE_SIZE
    token: str | None = None
    tokens: set[str] = field(default_factory=set)
    hits: dict[str, FeishuCandidate] = field(default_factory=dict)
    remaining: deque[FeishuCandidate] | None = None
    details: dict[str, IncomingMessage | None] = field(default_factory=dict)
    detail_batch_size: int = MAX_MGET_SIZE
    raw_bytes: int = 0
    oversized: int = 0
    pages: int = 0
    returned: int = 0
    filter_reasons: Counter[_FilterReason] = field(default_factory=Counter)
    duplicates: int = 0
    completed: bool = False
    failures: int = 0

    def restart(self, *, page_size: int | None = None) -> "_ScanWindow":
        return _ScanWindow(self.start, self.end, self.page_size if page_size is None else page_size)


class FeishuPoller:
    def __init__(self, discovery: FeishuDiscovery, *, account: ChannelUser, history: History,
                 notify_user: Callable[[ChannelUser], None], private_started_ms: int,
                 clock: Callable[[], float] = time, before_close: Callable[[], None] | None = None) -> None:
        self._discovery = discovery
        self._notify_user = notify_user
        self._account = account
        self._private_started_ms = private_started_ms
        self._history = history
        self._before_close = before_close
        self._clock = clock
        self._poll_lock = asyncio.Lock()
        self._window: _ScanWindow | None = None
        self._right_segments: list[_ScanWindow] = []
        self._scan_retry_at = 0

    def _submit_pending(self) -> None:
        for key in self._history.pending_feishu_users():
            user = parse_open_id(key)
            if user is None or user == self._account:
                raise ValueError("invalid stored Feishu inbox recipient")
            self._notify_user(user)

    def _pause_scan(self) -> None:
        self._scan_retry_at = int(self._clock()) + _DENSE_WINDOW_RETRY_SECONDS

    def _narrow_window(self, window: _ScanWindow) -> None:
        if window.end - window.start <= 1:
            self._window = window.restart()
            self._pause_scan()
            raise CliError("cli_single_second_window_limit")
        mid = (window.start + window.end) // 2
        # [start, end) 二分整个区段，包括回看部分；从左往右完成，不重扫已完成的左段。
        self._right_segments.append(_ScanWindow(mid, window.end, window.page_size))
        self._window = _ScanWindow(window.start, mid, window.page_size)
        raise CliError("cli_window_limit")

    async def _collect(self, window: _ScanWindow) -> bool:
        if window.completed:
            return True
        for _ in range(_MAX_PAGES):
            try:
                items, token = await self._discovery.search_page(
                    window.start, window.end, window.token, page_size=window.page_size)
                if token is not None and token in window.tokens:
                    self._window = window.restart()
                    raise CliError("cli_page_cycle")
                page: dict[str, FeishuCandidate] = {}
                filter_reasons: Counter[_FilterReason] = Counter()
                for row in items:
                    hit = _search_hit(row, self._account)
                    if isinstance(hit, str):
                        filter_reasons[hit] += 1
                        continue
                    if not window.start <= hit.created_second < window.end:
                        filter_reasons["out_of_window"] += 1
                        continue
                    mid = hit.message_id
                    previous = page.get(mid) or window.hits.get(mid)
                    if previous is not None and previous != hit:
                        self._window = window.restart()
                        raise CliError("cli_conflicting_search_hit")
                    if previous is None and not self._history.feishu_inbound_seen(mid):
                        page[mid] = hit
            except CliStdoutLimit:
                if window.page_size == MIN_PAGE_SIZE:
                    self._pause_scan()
                    raise CliError("cli_single_result_output_limit") from None
                self._window = window.restart(page_size=max(MIN_PAGE_SIZE, window.page_size // 2))
                return False
            except CliError:
                window.failures += 1
                if window.token is not None and window.failures >= 3:
                    self._window = window.restart()
                raise
            window.failures = 0
            if len(window.hits) + len(page) > _MAX_WINDOW_MESSAGES:
                self._narrow_window(window)
            window.hits.update(page)
            window.pages += 1
            window.returned += len(items)
            window.filter_reasons.update(filter_reasons)
            window.duplicates += len(items) - sum(filter_reasons.values()) - len(page)
            if token is None:
                window.completed = True
                return True
            if len(window.tokens) >= _MAX_WINDOW_PAGES - 1:
                self._narrow_window(window)
            window.token = token
            window.tokens.add(token)
        return False

    def _verify_details(self, expected: dict[str, FeishuCandidate], rows: list) -> dict[str, IncomingMessage | None]:
        accepted: dict[str, IncomingMessage | None] = {}
        returned: set[str] = set()
        for row in rows:
            mid = row.get("message_id") if isinstance(row, dict) else None
            if not isinstance(mid, str) or mid not in expected or mid in returned:
                raise CliError("cli_invalid_message_details")
            returned.add(mid)
            hit = expected[mid]
            if row.get("deleted") is True:
                # 显式撤回只用于跳过已知 ID；缺失详情不等于撤回。
                accepted[mid] = None
                continue
            chat = row.get("chat_id")
            user = parse_open_id(hit.peer_open_id)
            if user is None or not _valid_id(chat, "oc") or _chat_digest(chat) != hit.chat_digest:
                continue
            raw = incoming(row, user=user, account=self._account)
            if (raw is None or not raw.content or raw.kind != hit.kind
                    or raw.created_ms // 1000 != hit.created_second):
                continue
            accepted[mid] = raw if raw.created_ms >= self._private_started_ms else None
        return accepted

    async def _fetch_details(self, window: _ScanWindow, remaining: deque[FeishuCandidate]) -> bool:
        for _ in range(_MAX_DETAIL_BATCHES):
            if not remaining:
                return True
            expected = {}
            for hit in remaining:
                expected[hit.message_id] = hit
                if len(expected) == window.detail_batch_size:
                    break
            try:
                rows = await self._discovery.messages(tuple(expected))
            except CliStdoutLimit:
                if len(expected) == 1:
                    # 单条超限与缺项一样待恢复；不标为已见，后续条目恢复正常批量。
                    remaining.popleft()
                    window.oversized += 1
                    window.detail_batch_size = MAX_MGET_SIZE
                else:
                    window.detail_batch_size = max(1, len(expected) // 2)
                continue
            details = self._verify_details(expected, rows)
            size = sum(4 * len(raw.content) for raw in details.values() if raw is not None)
            if window.raw_bytes + size > _MAX_WINDOW_RAW_BYTES:
                self._narrow_window(window)
            window.raw_bytes += size
            window.details.update((mid, details[mid]) for mid in expected if mid in details)
            for _ in expected:
                remaining.popleft()
        return not remaining

    def _record(self, through: int | None, now: int, hits: dict[str, FeishuCandidate],
                details: dict[str, IncomingMessage | None]) -> None:
        verified: dict[str, list[FeishuRaw]] = defaultdict(list)
        discarded = []
        for mid, raw in sorted(details.items(), key=lambda item: item[1].created_ms if item[1] else 0):
            if raw is None:
                discarded.append(mid)
            else:
                verified[hits[mid].peer_open_id].append((mid, raw.created_ms, raw.kind, raw.content))
        retry = [hit for mid, hit in hits.items() if mid not in details]
        self._history.record_feishu_window(through, now=now, verified=verified, discarded=discarded, retry=retry)
        if hits and _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug("Feishu inbox recorded; source=%s accepted=%d discarded=%d deferred=%d users=%d",
                       "recovery" if through is None else "search", sum(map(len, verified.values())),
                       len(discarded), len(retry), len(verified))
        if retry:
            _LOG.warning("Feishu details unavailable or mismatched; deferred=%d", len(retry))
        if verified:
            self._submit_pending()

    async def _recover(self, now: int) -> None:
        hits = {hit.message_id: hit for hit in self._history.due_feishu_candidates(now, _MAX_RECOVERY_BATCH)}
        if not hits:
            return
        try:
            try:
                rows = await self._discovery.messages(tuple(hits))
            except CliStdoutLimit:
                if len(hits) == 1:
                    raise
                # 只额外探测最早到期的一条；其余保持原到期时间，下一轮优先于已退避的队首。
                head = next(iter(hits))
                hits = {head: hits[head]}
                rows = await self._discovery.messages((head,))
            details = self._verify_details(hits, rows)
        except CliError as exc:
            self._history.record_feishu_window(None, now=now, verified={}, discarded=(), retry=tuple(hits.values()))
            if isinstance(exc, CliStdoutLimit):
                _LOG.warning("Feishu single-message recovery output limit; deferred=1")
            else:
                _LOG.warning("Feishu detail recovery request failed; deferred=%d", len(hits))
            return
        self._record(None, now, hits, details)

    async def _scan(self, now: int) -> None:
        if now < self._scan_retry_at:
            _LOG.debug("Feishu search waiting; reason=backoff retry_in_s=%d", self._scan_retry_at - now)
            return
        if self._window is None:
            if self._right_segments:
                self._window = self._right_segments.pop()
            else:
                floor, through = self._history.feishu_inbound_window(self._private_started_ms // 1000)
                start = max(floor, through - _OVERLAP_SECONDS)
                end = min(now, through + _MAX_SCAN_SECONDS)
                if end <= start:
                    _LOG.debug("Feishu search waiting; reason=no_time_window")
                    return
                self._window = _ScanWindow(start, end)
        window = self._window
        debug = _LOG_SEARCH_WINDOWS and _LOG.isEnabledFor(logging.DEBUG)
        started = asyncio.get_running_loop().time() if debug else 0.0
        stage, outcome = "search", "failed"
        cli_error = None
        try:
            if not await self._collect(window):
                outcome = "pending"
                return
            stage = "reconcile"
            if window.remaining is None:
                # 只信任完整且无冲突的搜索段。刷新不推进水位，恢复总是使用最新已提交的核验依据。
                fresh = self._history.reconcile_feishu_search(window.hits.values(), now=now)
                window.hits = {hit.message_id: hit for hit in fresh}
                window.remaining = deque(fresh)
            stage = "details"
            if not await self._fetch_details(window, window.remaining):
                outcome = "pending"
                return
            stage = "record"
            self._record(window.end, now, window.hits, window.details)
            if window.oversized:
                _LOG.warning("Feishu single-message detail output limit; deferred=%d", window.oversized)
            self._window = None
            outcome = "ok"
        except CliError as exc:
            cli_error = exc
            raise
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            if debug:
                filter_reasons = ",".join(f"{reason}:{count}" for reason, count in
                                          sorted(window.filter_reasons.items())) or "none"
                _LOG.debug("Feishu search window; outcome=%s stage=%s start_utc=%s end_utc=%s "
                           "pages=%d returned=%d filtered=%d duplicates=%d elapsed_ms=%.0f category=%s api_code=%s "
                           "filter_reasons=%s",
                           outcome, stage, datetime.fromtimestamp(window.start, timezone.utc).isoformat(),
                           datetime.fromtimestamp(window.end, timezone.utc).isoformat(), window.pages,
                           window.returned, sum(window.filter_reasons.values()), window.duplicates,
                           (asyncio.get_running_loop().time() - started) * 1000,
                           cli_error.category if cli_error else None, cli_error.api_code if cli_error else None,
                           filter_reasons)

    async def poll_once(self) -> None:
        async with self._poll_lock:
            self._submit_pending()  # 重启时不用再次搜到已落库的消息。
            now = int(self._clock())
            try:
                await self._scan(now)
            except CliError:
                await self._recover(now)
                raise
            await self._recover(now)

    async def run(self) -> None:
        """不重叠轮询，长批次完成后跳过错过的时钟刻度。"""
        loop = asyncio.get_running_loop()
        next_tick = loop.time()
        _LOG.info("Feishu global search started; interval_s=%d overlap_s=%d", _POLL_SECONDS, _OVERLAP_SECONDS)
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
            if self._before_close is not None:
                self._before_close()  # 同步关闭所有用户的入口与发送队列。
