"""单 Session 的运行循环；网络、通道和工具运行在同一 asyncio 事件循环。"""

import asyncio
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass
import logging
from sqlite3 import OperationalError
from typing import Literal
from uuid import uuid4

from ..config import RuntimeSettings
from ..model_diagnostics import ModelRequestDiagnostics, failure_category
from ..protocol import (AgentReply, Channel, DelegationTask, DeliveryStatus, InputItem, InputUsage, InternalInput,
                         Message, Model, ModelClosing, ModelError, PersistedInput, Receipt, ReplayReady,
                         ReplyDelivery, ReplyInfo, ReplyStream, SelectableChannel, TextDelta, ToolCall,
                         ToolReady, validate_input_items)
from ..storage.sqlite import (Agent, AgentReplyInput, DelegatedInput, History, ReplyOutcome, RetryableInputWrite,
                              is_sqlite_lock_error)
from ..workspace import Workspace
from .compaction import Compactions
from .context import Context
from .model_deadline import ModelDeadline, model_deadline_at
from .sqlite_access import SQLiteAccess
from .tools import TOOL_CANCELLED, ToolContext, Tools


_LOG = logging.getLogger(__name__)
_DELIVERY_TIMEOUT = 120  # 无播放进展语义的回执通道（例如离线控制台）
_MAX_RETIRED_MODELS = 2
_MODEL_CLOSE_GRACE = 1.0  # 关停只等待已退役的模型任务自行完成，不再次取消。


@dataclass(frozen=True, slots=True)
class _InputBatch:
    items: tuple[InputItem, ...]
    persisted_ids: frozenset[int] = frozenset()


class _InputWriteFailed(Exception):
    """输入未确认落库，与已落库后的执行或回复失败区分。"""


class Runtime:
    """每个 Session 一个队列和一个活动运行；可共享模型、工具和 SQLite。"""

    def __init__(self, session_id: str, history: History, model: Model, channel: Channel | None, *,
                 tools: Tools | None = None, settings: RuntimeSettings | None = None,
                 thinking: bool = False, instructions: str | None = None, delivery_idle_timeout: float | None = None,
                 retry_sqlite_locks: bool = False, compactions: Compactions | None = None,
                 workspace: Workspace | None = None, agent: Agent | None = None, model_origin: bytes | None = None,
                 reply_to_parent: Callable[[AgentReply], bool] | None = None) -> None:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if agent is not None and agent.session_id != session_id:
            raise ValueError("agent session does not match runtime session")
        if (reply_to_parent is not None) != (agent is not None and agent.parent_agent_id is not None):
            raise ValueError("only a child agent requires a parent reply destination")
        if reply_to_parent is not None and channel is not None:
            raise ValueError("child agent cannot have a user channel")
        self.agent = agent
        self._reply_to_parent = reply_to_parent
        self.session_id = session_id
        self.history = history
        self._model = model
        self._channel = channel
        self._delivery_idle_timeout = delivery_idle_timeout
        self._workspace = workspace
        self._tools = tools if tools is not None else Tools()
        self._settings = settings if settings is not None else RuntimeSettings()
        self._thinking = thinking
        self._instructions = instructions
        self._origin_options = {"model_origin": model_origin} if model_origin is not None else {}
        self._pending: deque[_InputBatch] = deque()
        self._trigger: Callable[[], bool] | None = None
        self._reply_commit_failed = False
        self._queued_inbound: set[int] = set()
        self._input_blocked: Literal["retryable", "unsafe"] | None = None
        self._input_retry_requested = False
        self._worker: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._retired_deliveries: set[asyncio.Task[Receipt]] = set()
        self._retired_models: set[asyncio.Task[bool]] = set()
        self._closed = False
        self._sqlite = SQLiteAccess(retry_locks=retry_sqlite_locks, stopped=lambda: self._closed)
        self._owns_compactions = compactions is None
        self._compactions = compactions if compactions is not None else Compactions(history)
        self._compactions.check_owner(history)
        self._context = Context(history, self._settings, self._compactions, model=model, sqlite=self._sqlite)
        self._idle = asyncio.Event()
        self._idle.set()
        self.last_failure: str | None = None

    def submit_batch(self, messages: Sequence[Message]) -> None:
        """在事件循环内整批追加用户输入；不阻塞当前响应。"""
        self.submit_inputs(messages)

    def submit_persisted_batch(self, batch_id: int) -> None:
        """由持久化入口移交批次；不在安全边界之前改变模型可读历史。"""
        self.submit_inputs((PersistedInput(batch_id),))

    def submit_inputs(self, items: Sequence[InputItem]) -> None:
        """在同一个事件循环同步接收有序混合批次；不立即写入历史。"""
        if self._closed:
            raise RuntimeError("runtime is closed")
        items = tuple(items)
        validate_input_items(items)
        if self.agent is None and any(isinstance(item, InternalInput) and
                                      isinstance(item.content, (DelegationTask, AgentReply)) for item in items):
            raise ValueError("agent inputs require an agent binding")
        inputs: list[InputItem] = []
        persisted_ids: set[int] = set()
        for item in items:
            if isinstance(item, PersistedInput):
                if item.batch_id in self._queued_inbound or item.batch_id in persisted_ids:
                    continue
                persisted_ids.add(item.batch_id)
            inputs.append(item)
        if not inputs:
            return
        loop = asyncio.get_running_loop()
        self._trigger = None  # 真实或既有内部输入优先于尚未运行的空转写唤醒。
        self._pending.append(_InputBatch(tuple(inputs), frozenset(persisted_ids)))
        self._queued_inbound.update(persisted_ids)
        # 只以真正入队的输入唤醒。轮询重复移交同一个持久批次不会反复消耗重试预算。
        self._input_retry_requested = True
        if self._input_blocked == "retryable":
            self._input_blocked = None
        _LOG.debug("runtime input queued; items=%d persisted=%d pending=%d",
                   len(inputs), len(persisted_ids), len(self._pending))
        self._start_worker(loop)

    def trigger_run(self, can_start: Callable[[], bool] = lambda: True) -> None:
        """无输入地请求一次新运行；队列有内容或路由过期时让位。"""
        if (self._closed or self._reply_to_parent is not None or self._pending or self._input_blocked
                or self._reply_commit_failed):
            return
        self._trigger = can_start
        self._start_worker(asyncio.get_running_loop())

    def _has_work(self) -> bool:
        return bool(self._pending) or self._trigger is not None

    def _start_worker(self, loop: asyncio.AbstractEventLoop) -> None:
        if (self._worker is None and self._has_work() and not self._closed and not self._input_blocked
                and self._has_model_slot()):
            self._idle.clear()
            self._worker = loop.create_task(self._drain())

    def _has_model_slot(self) -> bool:
        pending = sum(not task.done() for task in self._retired_models)
        if pending < _MAX_RETIRED_MODELS:
            return True
        if self.last_failure != "model_cleanup_blocked":
            _LOG.warning("model cleanup capacity reached; pending=%d", pending)
            self.last_failure = "model_cleanup_blocked"
        return False

    async def _wait_for_model_slot(self) -> None:
        while not self._closed and not self._has_model_slot():
            # 保持同一工具链和已提交的调用结果；关闭仅取消等待者，不动旧收尾任务。
            pending = tuple(task for task in self._retired_models if not task.done())
            await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        if self.last_failure == "model_cleanup_blocked" and not self._closed:
            self.last_failure = None

    @property
    def idle(self) -> bool:
        return self._idle.is_set()

    @property
    def input_retry_forbidden(self) -> bool:
        return self._input_blocked == "unsafe"

    async def wait_idle(self) -> None:
        """等待队列停转；输入故障或模型清理达到上限时也会返回，原因见 last_failure。"""
        await self._idle.wait()

    def stop_accepting(self) -> None:
        """同步关停入口；可以在调度异步清理之前调用。"""
        self._closed = True
        if self._owns_compactions:
            self._compactions.stop()

    async def close(self) -> None:
        if self._close_task is None:
            self.stop_accepting()
            worker = self._worker
            if worker is not None:
                worker.cancel()
            self._close_task = asyncio.create_task(self._finish_close(worker))
        closing = self._close_task
        interrupted = None
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError as exc:
                # 并发或反复取消关闭调用方，均不再次取消 worker 或提前交还会话。
                interrupted = exc
        closing.result()
        if interrupted is not None:
            raise interrupted

    async def _finish_close(self, worker: asyncio.Task[None] | None) -> None:
        if worker is not None:
            await asyncio.gather(worker, return_exceptions=True)
        if self._retired_models:
            _, pending = await asyncio.wait(tuple(self._retired_models), timeout=_MODEL_CLOSE_GRACE)
            if pending:
                _LOG.warning("model cleanup still running after shutdown grace; pending=%d", len(pending))
        if self._owns_compactions:
            await self._compactions.close()
        self._pending.clear()
        self._trigger = None
        self._queued_inbound.clear()
        self._worker = None
        self._idle.set()

    async def _drain(self) -> None:
        try:
            while self._has_work() and not self._closed and not self._input_blocked and self._has_model_slot():
                trigger, self._trigger = self._trigger, None
                await self._process(uuid4().hex, trigger=trigger)
        finally:
            self._worker = None
            self._idle.set()

    async def _record_head(self, run_id: str, *, start: bool = False) -> tuple[list[int], bool, bool]:
        batch = self._pending[0]
        # 写入期间才到达的新输入，可以给这条队首再提供一次有界尝试；之前的入队不累积预算。
        self._input_retry_requested = False
        recorded = await self._record_input(batch, run_id, start=start)
        # 确认写入、空批次或旧路由作废后才能交还队首；失败时保留输入和去重。
        self._pending.popleft()
        self._queued_inbound.difference_update(batch.persisted_ids)
        return recorded

    async def _record_input(self, batch: _InputBatch, run_id: str, *, start: bool) -> tuple[list[int], bool, bool]:
        rows: list[Message | int | DelegatedInput | AgentReplyInput] = []
        has_memory_user = False
        for item in batch.items:
            if isinstance(item, PersistedInput):
                rows.append(item.batch_id)
            elif isinstance(item, InternalInput):
                content = item.content
                message = item.to_model_message()
                if isinstance(content, DelegationTask):
                    rows.append(DelegatedInput(content.delegation_id, message))
                elif isinstance(content, AgentReply):
                    rows.append(AgentReplyInput(content.agent_id, content.run_id, message))
                else:
                    rows.append(message)
            else:
                rows.append(item)
                has_memory_user = True

        def append() -> tuple[list[int], bool, bool]:
            if self.agent is not None:
                ids, consumed_user, consumed_agent = self.history.append_agent_inputs(
                    self.agent.agent_id, run_id, rows, start=start)
                return ids, has_memory_user or consumed_user, consumed_agent
            if not batch.persisted_ids:
                messages = tuple(row for row in rows if isinstance(row, Message))
                return self.history.append(self.session_id, messages, retryable_io=True), has_memory_user, False
            if len(rows) == 1 and isinstance(rows[0], int):
                ids = self.history.consume_inbound(self.session_id, rows[0])
                return ids, bool(ids), False
            ordinary_rows = [row for row in rows if isinstance(row, (Message, int))]
            ids, consumed_user = self.history.append_with_inbound(self.session_id, ordinary_rows)
            return ids, has_memory_user or consumed_user, False

        try:
            ids, has_user_input, has_agent_input = await self._sqlite.write_input(append)
        except Exception as exc:
            raise _InputWriteFailed from exc
        _LOG.debug("runtime input recorded; run=%s persisted=%d messages=%d user_input=%s agent_input=%s",
                   run_id, len(batch.persisted_ids), len(ids), has_user_input, has_agent_input)
        return ids, has_user_input, has_agent_input

    async def _process(self, run_id: str, *, trigger: Callable[[], bool] | None) -> None:
        self.last_failure = None
        clock = asyncio.get_running_loop().time
        started = clock()
        recorded = False
        result = None
        try:
            if trigger is None:
                message_ids, has_user_input, has_agent_input = await self._record_head(run_id, start=True)
                if message_ids:
                    recorded = True
                    result = await self._run(run_id, message_ids[0], has_user_input=has_user_input,
                                             has_agent_input=has_agent_input)
            else:
                prepared = await self._prepare_trigger(run_id, trigger)
                if prepared is not None:
                    protected_from, messages = prepared
                    recorded = True
                    result = await self._run(run_id, protected_from, has_user_input=False,
                                             has_agent_input=False, first_messages=messages)
        except _InputWriteFailed as exc:
            self.last_failure = "input_write_failed"
            cause = exc.__cause__
            safe_to_retry = isinstance(cause, RetryableInputWrite) or (
                isinstance(cause, OperationalError) and is_sqlite_lock_error(cause))
            # 提交结果不确定或回滚失败不能重试；到达写入边界的新输入不会覆盖这条底线。
            self._input_blocked = "retryable" if safe_to_retry else "unsafe"
            if safe_to_retry and self._input_retry_requested:
                self._input_blocked = None
            if isinstance(cause, RetryableInputWrite):
                cause = cause.__cause__
            _LOG.error("runtime input write failed; queue=%s run=%s error_type=%s sqlite_code=%s",
                       self._input_blocked or "retry_requested", run_id, type(cause).__name__,
                       getattr(cause, "sqlite_errorcode", None))
        except asyncio.CancelledError:
            _LOG.debug("runtime run cancelled; run=%s", run_id)
            raise
        except (ModelError, TimeoutError) as exc:
            self.last_failure = "context_failed"
            _LOG.warning("runtime context failed (%s); run=%s",
                         "timeout" if isinstance(exc, TimeoutError) else exc.diagnostic, run_id)
        except Exception as exc:
            self.last_failure = "runtime_failed"
            _LOG.error("session runtime could not finish the current input; run=%s error_type=%s",
                       run_id, type(exc).__name__)
        if self.agent is not None and recorded and not self._closed:
            await self._finish_agent_run(run_id, result)
        _LOG.debug("runtime run finished; run=%s status=%s elapsed_ms=%.0f",
                   run_id, self.last_failure or "complete", (clock() - started) * 1000)

    async def _finish_agent_run(self, run_id: str, result: str | None) -> None:
        error = self.last_failure
        if self._reply_to_parent is not None and error is None and (not result or not result.strip()):
            error = self.last_failure = "empty_result"
        if error is not None:
            result = None
        try:
            await self._sqlite.write_input(lambda: self.history.finish_agent_run(run_id, result=result, error=error))
        except Exception as exc:
            self.last_failure = "run_terminal_write_failed"
            _LOG.error("agent terminal write failed; run=%s error_type=%s", run_id, type(exc).__name__)
            return
        if self._reply_to_parent is None or self._closed:
            return
        assert self.agent is not None
        notice = AgentReply(self.agent.agent_id, run_id, result=result, error=error)
        if self._reply_to_parent(notice):
            _LOG.debug("agent reply queued; agent=%s run=%s", self.agent.agent_id, run_id)
            try:
                await self._sqlite.write_input(lambda: self.history.mark_agent_reply_enqueued(run_id))
            except Exception as exc:
                _LOG.warning("agent reply enqueue could not be recorded; run=%s error_type=%s",
                             run_id, type(exc).__name__)

    async def _prepare_trigger(self, run_id: str,
                               route_is_current: Callable[[], bool]) -> tuple[int, list[Message]] | None:
        def idle_for_trigger() -> bool:
            return not (self._closed or self._pending or self._reply_commit_failed)

        def can_start() -> bool:
            # 重入的窗口失效回调可能同步提交真实输入。
            return idle_for_trigger() and route_is_current() and idle_for_trigger()

        if not can_start():
            return None
        prepared = await self._prepare_run(run_id, None, can_send=idle_for_trigger)
        if prepared is None:
            return None
        agent = self.agent

        def start() -> bool:
            if not idle_for_trigger():
                return False
            if agent is None:
                return can_start()
            # 拿到写锁后再检查；锁重试期间仍可能有收件落库或旧通话失效。
            return self.history.start_agent_run(agent.agent_id, run_id, can_start=can_start)

        return prepared if await self._sqlite(start) else None

    async def _prepare_run(self, run_id: str, protected_from: int | None, *,
                           can_send: Callable[[], bool]) -> tuple[int, list[Message]] | None:
        try:
            # 无消息首轮在确认提交运行前不启动额外的摘要请求。
            last_input_tokens = (await self._sqlite(self.history.last_input_tokens, self.session_id,
                                                   **self._origin_options) if protected_from is not None else None)
            return await self._context.prepare(self.session_id, thinking=self._thinking,
                                               tools=self._tools.specs, instructions=self._instructions,
                                               protected_from=protected_from, can_send=can_send,
                                               last_input_tokens=last_input_tokens, run_id=run_id)
        except OperationalError as exc:
            if not is_sqlite_lock_error(exc):
                raise
            self.last_failure = "context_lock_timeout"
            _LOG.warning("runtime context lock retries exhausted; run=%s sqlite_code=%s",
                         run_id, getattr(exc, "sqlite_errorcode", None))
            return None

    async def _run(self, run_id: str, protected_from: int, *, has_user_input: bool,
                   has_agent_input: bool, first_messages: list[Message] | None = None) -> str | None:
        def can_send() -> bool:
            return not self._closed

        for index in range(self._settings.max_tool_rounds):
            if not can_send():
                return None
            if index == 0 and first_messages is not None:
                messages = first_messages
            else:
                prepared = await self._prepare_run(run_id, protected_from, can_send=can_send)
                if prepared is None:
                    return None
                _, messages = prepared
            model_status, continue_tools, final_text = await self._reply(
                run_id, messages, has_user_input=has_user_input, has_agent_input=has_agent_input)
            if model_status != "complete":
                if model_status != "model_interrupted" or self._reply_to_parent is not None:
                    self.last_failure = model_status
                return None
            if not continue_tools:
                return final_text
            if index + 1 == self._settings.max_tool_rounds:
                self.last_failure = "tool_round_limit"
                return None
            if not can_send():
                return None
            await self._wait_for_model_slot()
            if not can_send():
                return None
            while self._pending:
                _, recorded_user_input, recorded_agent_input = await self._record_head(run_id)
                has_user_input = has_user_input or recorded_user_input
                has_agent_input = has_agent_input or recorded_agent_input
        return None

    async def _reply(self, run_id: str, messages: Sequence[Message], *,
                     has_user_input: bool, has_agent_input: bool) -> tuple[str, bool, str | None]:
        self._reply_commit_failed = False  # 新回复有自己的结算，不沿用上一条的失败。
        loop = asyncio.get_running_loop()
        info = ReplyInfo(uuid4().hex, self.session_id, run_id, loop.time())
        diagnostics = ModelRequestDiagnostics("reply", self._settings.model_timeout, run_id, info.reply_id)
        text: list[str] = []
        replay = None
        request_locks = self._tools.request_locks()
        scheduled: dict[str, tuple[ToolCall, asyncio.Task[str]]] = {}

        def is_generated_prefix(prefix: str) -> bool:
            offset = 0
            for delta in text:
                part = delta[:len(prefix) - offset]
                if not prefix.startswith(part, offset):
                    return False
                offset += len(part)
                if offset == len(prefix):
                    return True
            return False

        idle_timeout = self._delivery_idle_timeout
        reply = ReplyStream(info, validate_progress=is_generated_prefix if self._channel is not None else None)
        channel = self._channel
        selected: ReplyDelivery | None = None
        delivery_started: asyncio.Future[None] = loop.create_future()

        async def deliver(target: ReplyDelivery) -> Receipt:
            try:
                if not delivery_started.done():
                    delivery_started.set_result(None)
                return await target.deliver(reply)
            finally:
                reply.delivery_finished_at = loop.time()

        delivery: asyncio.Task[Receipt] | None = None
        has_user_channel = self._channel is not None
        model_complete = False
        reply_closed = False
        voice_incomplete = False
        input_tokens: int | None = None
        accept_model_events = True
        reader_closing = False
        read_deadline: ModelDeadline | None = None
        stop_stream = getattr(self._model, "stop_stream", None)

        def finish_reply_stream() -> None:
            nonlocal reply_closed
            if not reply_closed:
                reply_closed = True
                if delivery is not None:
                    reply.start_delivery_wait(idle_timeout, _DELIVERY_TIMEOUT)
                reply.close()

        def voice_interrupted() -> bool:
            if not reply.voice_delivery or delivery is None or not delivery.done():
                return False
            receipt = self._read_receipt(reply, delivery)
            return (receipt is not None and receipt.status != DeliveryStatus.FULL
                    and is_generated_prefix(receipt.confirmed))

        def can_start_serial_tool() -> bool:
            # 工具拿锁可能先于 Runtime 汇合点恢复；直接检查投递任务的同步终态。
            # 汇合后也尊重投递超时等不完整结算。
            return not (self._closed or voice_incomplete or voice_interrupted())

        async def read_model() -> bool:
            nonlocal delivery, input_tokens, replay, read_deadline, reader_closing, model_complete
            nonlocal selected, idle_timeout
            async with model_deadline_at(loop.time() + self._settings.model_timeout, stop_stream) as deadline:
                read_deadline = deadline
                async with aclosing(self._model.stream(messages, self._tools.specs, thinking=self._thinking)) as events:
                    async for event in events:
                        # 即使回执与这一帧同时就绪，也不能发布它或启动新的工具。
                        # 适配器在收到取消后吐出的迟到帧也一律丢弃。
                        if not accept_model_events or voice_interrupted():
                            reader_closing = True
                            deadline.disarm()
                            return False
                        if stop_stream is not None and deadline.expired():
                            # 已在传输收尾的适配器仍可能吐出缓存的终态；超时后不启动工具。
                            raise TimeoutError
                        if isinstance(event, ModelClosing):
                            reader_closing = True
                            if event.complete:
                                deadline.disarm()
                                model_complete = True
                                if has_user_channel:
                                    # 回执可先于连接关闭结算；完整投递仍等连接收尾。
                                    finish_reply_stream()
                                return True
                        elif isinstance(event, TextDelta) and event.text:
                            if not text:
                                first_text_at = loop.time()
                                reply.first_text_at = first_text_at
                                _LOG.debug("model first text; reply=%s elapsed_ms=%.0f", info.reply_id,
                                           (first_text_at - info.started_at) * 1000)
                            text.append(event.text)
                            if channel is not None:
                                first_handoff = delivery is None
                                if delivery is None:
                                    # 首段交接同步选路；随后挂断或重拨不能改变已选通道。
                                    if isinstance(channel, SelectableChannel):
                                        selected = channel.select_delivery(reply)
                                    else:
                                        selected = ReplyDelivery(channel.deliver, idle_timeout)
                                    idle_timeout = selected.idle_timeout
                                    delivery = loop.create_task(deliver(selected))
                                if not delivery.done():
                                    reply.publish(event.text)
                                if first_handoff and reply.voice_delivery:
                                    # 先让语音投递进入同一回执接口，以便拒收先于相邻工具帧。
                                    await delivery_started
                        elif isinstance(event, InputUsage):
                            input_tokens = event.input_tokens
                        elif isinstance(event, ReplayReady):
                            if replay is not None:
                                raise ModelError("invalid_sse_event")
                            replay = event.replay
                        elif isinstance(event, ToolReady):
                            call = event.call
                            if call.id in scheduled:
                                raise ModelError("duplicate_tool_id")
                            context = ToolContext(self.session_id, run_id, call.id, has_user_input,
                                                  self.agent.agent_id if self.agent is not None else None,
                                                  has_agent_input, workspace=self._workspace, reply_id=info.reply_id)
                            scheduled[call.id] = (call, loop.create_task(
                                self._tools.execute(call, context, request_locks=request_locks,
                                                    can_start_serial=can_start_serial_tool),
                            ))
            return True

        model_task: asyncio.Task[bool] | None = None

        def retire_reader(task: asyncio.Task[bool]) -> None:
            nonlocal accept_model_events
            accept_model_events = False
            if read_deadline is not None:
                read_deadline.disarm()
            self._retired_models.add(task)
            task.add_done_callback(self._model_cleanup_finished)
            if not reader_closing or not has_user_channel:
                if (stop_stream is None or not stop_stream(task)) and not task.cancelling():
                    task.cancel()

        async def complete_model() -> str:
            nonlocal model_task
            with diagnostics.activate():
                model_task = loop.create_task(read_model())
            while not model_task.done():
                if voice_interrupted():
                    retire_reader(model_task)
                    return "complete" if model_complete else "model_interrupted"
                signal = delivery if delivery is not None else delivery_started
                # wait 不会把 Runtime worker 的取消直接传给原生读流；由 retire_reader 启动关闭预算。
                waiting = (model_task,) if not has_user_channel or signal.done() else (model_task, signal)
                await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
            if model_complete:
                self._report_model_cleanup(model_task, diagnostics=diagnostics)
                return "complete"
            if model_task.cancelled():
                # 只观察子任务的取消；真正的外层取消由 asyncio.wait 继续抛出。
                return "model_cancelled"
            return "complete" if model_task.result() else "model_interrupted"

        model_status = "complete"
        model_error = None
        _LOG.debug("model request started; run=%s reply=%s messages=%d tools=%d thinking=%s",
                   run_id, info.reply_id, len(messages), len(self._tools.specs), self._thinking)
        try:
            try:
                model_status = await complete_model()
                if model_status == "model_interrupted":
                    reply.abandon()
            except TimeoutError as exc:
                model_status = "model_timeout"
                diagnostics.failed(exc)
                _LOG.warning("model request timed out; reply=%s", info.reply_id)
            except ModelError as exc:
                model_status = "model_failed"
                model_error = exc.diagnostic
                diagnostics.failed(exc)
                _LOG.warning("model response failed (%s); reply=%s", model_error, info.reply_id)
            except Exception as exc:
                model_status = "model_failed"
                diagnostics.failed(exc)
                _LOG.error("model adapter failed")
            finally:
                if selected is not None and selected.require_complete_model and model_status != "complete":
                    reply.abandon()
                finish_reply_stream()

            generated = "".join(text)
            _LOG.debug("model request finished; reply=%s status=%s input_tokens=%s output_chars=%d tools=%d elapsed_ms=%.0f",
                       info.reply_id, model_status, input_tokens, len(generated), len(scheduled),
                       (loop.time() - info.started_at) * 1000)
            # 工具和投递一直独立运行；这个汇合点只提交成套的终态数据。
            receipt = await self._delivery_result(reply, delivery, generated, idle_timeout=idle_timeout)
            if selected is not None and selected.require_complete_model and model_status != "complete":
                # 不把收齐了一段不完整模型输出记成用户已收到的正文。
                receipt = None
            voice_incomplete = (reply.voice_delivery and receipt is not None
                                and receipt.status != DeliveryStatus.FULL)
            if model_status == "model_cancelled" and voice_interrupted():
                model_status = "model_interrupted"
            # 内部会话不经过用户投递，只保留完整模型正文。
            generated_text = generated if not has_user_channel and model_status == "complete" else ""
            confirmed_text = receipt.confirmed if receipt else ""
            calls = tuple(call for call, _ in scheduled.values())
            if scheduled:
                await asyncio.wait([task for _, task in scheduled.values()])
            results = [Message("tool", TOOL_CANCELLED if task.cancelled() else task.result(),
                                 tool_call_id=call.id) for call, task in scheduled.values()]
            # 完整模型流的协议回放元数据可保留成套工具调用；适配器仍仅从确认正文回放文本。
            if model_status != "complete":
                replay = None
            # 零投递不保留未送达正文；仍可随空正文保存完整的协议回放。
            keep_reply = (replay is not None or
                          (receipt.status != DeliveryStatus.ZERO if receipt is not None else bool(generated_text)))
            rows = ([Message("assistant", confirmed_text if receipt else generated_text, calls, replay=replay),
                       *results]
                      if keep_reply or calls else [])
            outcome = ReplyOutcome(info.reply_id, info.session_id, run_id, model_status,
                                   receipt.status if receipt else None, model_error=model_error,
                                   input_tokens=input_tokens if model_status == "complete" else None)
            try:
                await self._sqlite.write_input(lambda: self.history.finalize_reply(outcome, rows, **self._origin_options))
            except Exception:
                # 没有可靠的旧回复结算；只作废唤醒，不动真实输入。也拒绝稍后到达的同一轮空结果。
                self._reply_commit_failed = True
                self._trigger = None
                raise
            _LOG.debug("reply committed; reply=%s receipt=%s confirmed_chars=%d tool_results=%d elapsed_ms=%.0f",
                       info.reply_id, receipt.status if receipt else None, len(confirmed_text), len(results),
                       (loop.time() - info.started_at) * 1000)
            # 外部副本与主投递状态无关，只在历史事务成功后尽力入队。
            try:
                if (confirmed_text and receipt is not None and selected is not None and selected.after_commit is not None
                        and (not selected.require_complete_model or receipt.status == DeliveryStatus.FULL)):
                    selected.after_commit(info, confirmed_text)
            except Exception:
                _LOG.warning("background reply could not be queued; reply=%s", info.reply_id)
            return model_status, bool(calls) and not voice_incomplete, generated_text or confirmed_text or None
        finally:
            unfinished = [task for _, task in scheduled.values() if not task.done()]
            # 异常退出时先撤销工具（含还在等锁的调用），再等待可能延迟的模型收尾。
            for task in unfinished:
                task.cancel()
            delivery_started.cancel()
            wait_for_child_model = False
            try:
                if accept_model_events and model_task is not None:
                    if not model_task.done():
                        retire_reader(model_task)
                        wait_for_child_model = not has_user_channel
                    elif not model_task.cancelled():
                        model_task.exception()
                if delivery is not None:
                    if not delivery.done():
                        self._abandon_delivery(reply, delivery)
                    elif not delivery.cancelled():
                        delivery.exception()
                if wait_for_child_model:
                    # 子任务仍持有串行执行权，原生传输收尾或耗尽预算后才能退出。
                    await asyncio.wait((model_task,))
            finally:
                if unfinished:
                    await asyncio.gather(*unfinished, return_exceptions=True)

    def _model_cleanup_finished(self, task: asyncio.Task[bool]) -> None:
        self._retired_models.discard(task)
        self._report_model_cleanup(task)
        if self.last_failure == "model_cleanup_blocked":
            # 只在碰到清理上限后恢复保留的队首；正常路径不等待退役任务。
            self._start_worker(task.get_loop())

    @staticmethod
    def _report_model_cleanup(task: asyncio.Task[bool], *,
                              diagnostics: ModelRequestDiagnostics | None = None) -> None:
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                if diagnostics is not None and isinstance(error, Exception):
                    diagnostics.failed(error)
                detail = error.diagnostic if isinstance(error, ModelError) else failure_category(error)
                _LOG.warning("model cleanup failed (%s)", detail)

    def _abandon_delivery(self, reply: ReplyStream, task: asyncio.Task[Receipt]) -> None:
        reply.abandon()
        if task not in self._retired_deliveries:
            self._retired_deliveries.add(task)
            task.add_done_callback(self._delivery_cleanup_finished)
            task.cancel()

    def _delivery_cleanup_finished(self, task: asyncio.Task[Receipt]) -> None:
        self._retired_deliveries.discard(task)
        if not task.cancelled():
            task.exception()  # 消化迟到异常；不改变已结算回复。

    async def _wait_delivery(self, reply: ReplyStream, delivery: asyncio.Task[Receipt]) -> str | None:
        """返回超时时的确认前缀；正常结束返回 None。"""
        clock = asyncio.get_running_loop().time
        while True:
            now, deadline, confirmed, playback_terminal = reply.delivery_snapshot()
            if playback_terminal is not None and playback_terminal < deadline:
                # 播放已终止且早于空闲截止，回执只待当前投递协程恢复，无需等待网络清理。
                if not delivery.done():
                    await asyncio.wait((delivery,))
                return None
            finished = reply.delivery_finished_at
            if delivery.done() and (now if finished is None else finished) < deadline:
                return None
            if now >= deadline:
                return confirmed
            # 设备只会延后截止；按旧截止唤醒时，重新同步取证即可。
            await asyncio.wait((delivery,), timeout=max(0, deadline - clock()))

    async def _delivery_result(self, reply: ReplyStream, delivery: asyncio.Task[Receipt] | None,
                               generated: str, *, idle_timeout: float | None) -> Receipt | None:
        if delivery is None:
            return None
        info = reply.info
        failed = Receipt(info.reply_id, DeliveryStatus.UNKNOWN)
        timeout_prefix = await self._wait_delivery(reply, delivery)
        if timeout_prefix is not None:
            self._abandon_delivery(reply, delivery)
            if idle_timeout is not None:
                _LOG.warning("reply delivery idle deadline reached; reply=%s", info.reply_id)
            return Receipt(info.reply_id, DeliveryStatus.UNKNOWN, timeout_prefix if generated.startswith(timeout_prefix) else "")
        receipt = self._read_receipt(reply, delivery)
        if (receipt is None or not generated.startswith(receipt.confirmed)
                or (receipt.status == DeliveryStatus.FULL and generated != receipt.confirmed)):
            return failed
        return receipt

    @staticmethod
    def _read_receipt(reply: ReplyStream, delivery: asyncio.Task[Receipt]) -> Receipt | None:
        if delivery.cancelled() or delivery.exception() is not None:
            return None
        receipt = delivery.result()
        if (not isinstance(receipt, Receipt) or receipt.reply_id != reply.info.reply_id
                or not isinstance(receipt.status, DeliveryStatus) or not isinstance(receipt.confirmed, str)
                or (receipt.status == DeliveryStatus.ZERO and receipt.confirmed)
                or (receipt.status == DeliveryStatus.PARTIAL and not receipt.confirmed)):
            return None
        return receipt
