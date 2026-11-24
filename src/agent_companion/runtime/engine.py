"""单 Session 的运行循环；网络、通道和工具运行在同一 asyncio 事件循环。"""

import asyncio
from collections import deque
from collections.abc import Sequence
from contextlib import aclosing
from dataclasses import dataclass
import logging
from pathlib import Path
from uuid import uuid4

from ..config import RuntimeSettings
from ..protocol import (Channel, DeliveryStatus, InputItem, InputUsage, InternalInput, Message, Model, ModelClosing,
                         ModelError, PersistedInput, ReasoningDelta, Receipt, ReplayReady, ReplayScope, ReplyDelivery,
                         ReplyInfo, ReplyStream, SelectableChannel, TextDelta, ToolCall, ToolReady, validate_input_items)
from ..storage.sqlite import History, ReplyOutcome, RetryableInputWrite
from .compaction import Compactions
from .context import Context
from .instructions import InstructionsError
from .model_deadline import ModelDeadline, model_deadline_at
from .sqlite_access import SQLiteAccess
from .tools import ToolContext, Tools


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
                 thinking: bool = False, instructions: str = "", delivery_idle_timeout: float | None = None,
                 retry_sqlite_locks: bool = False, compactions: Compactions | None = None) -> None:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        self.session_id = session_id
        self.history = history
        self._model = model
        self._channel = channel
        self._delivery_idle_timeout = delivery_idle_timeout
        self._tools = tools if tools is not None else Tools()
        self._settings = settings if settings is not None else RuntimeSettings()
        self._thinking = thinking
        self._instructions = instructions
        replay_scope = getattr(model, "replay_scope", None)
        self._replay_scope = replay_scope if isinstance(replay_scope, ReplayScope) else None
        self._pending: deque[_InputBatch] = deque()
        self._queued_inbound: set[int] = set()
        self._input_blocked = False
        self._worker: asyncio.Task[None] | None = None
        self._retired_deliveries: set[asyncio.Task[Receipt]] = set()
        self._retired_models: set[asyncio.Task[bool]] = set()
        self._closed = False
        self._sqlite = SQLiteAccess(retry_locks=retry_sqlite_locks, stopped=lambda: self._closed)
        self._owns_compactions = compactions is None
        self._compactions = compactions if compactions is not None else Compactions(history, model)
        self._compactions.check_owner(history, model)
        self._context = Context(history, self._settings, self._compactions, sqlite=self._sqlite,
                                replay_scope=self._replay_scope, agents_path=Path.cwd() / "AGENTS.md")
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
        if self._channel is None:
            raise RuntimeError("internal runs must be scheduled by the subtask dispatcher")
        items = tuple(items)
        validate_input_items(items)
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
        self._pending.append(_InputBatch(tuple(inputs), frozenset(persisted_ids)))
        self._queued_inbound.update(persisted_ids)
        _LOG.debug("runtime input queued; items=%d persisted=%d pending=%d",
                   len(inputs), len(persisted_ids), len(self._pending))
        self._start_worker(loop)

    def _start_worker(self, loop: asyncio.AbstractEventLoop) -> None:
        if (self._worker is None and self._pending and not self._closed and not self._input_blocked
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

    async def wait_idle(self) -> None:
        """等待队列停转；输入故障或模型清理达到上限时也会返回，原因见 last_failure。"""
        await self._idle.wait()

    async def run_internal(self, run_id: str, prompt: str) -> str | None:
        """运行独立的子会话话轮，返回失败代码；由调用方保证同会话串行。"""
        if self._closed or self._channel is not None or self._tools.specs:
            raise RuntimeError("internal runs require a runtime without a user channel or tools")
        await self._process(Message("user", prompt), run_id)
        return self.last_failure

    def stop_accepting(self) -> None:
        """同步关停入口；可以在调度异步清理之前调用。"""
        self._closed = True
        if self._owns_compactions:
            self._compactions.stop()

    async def close(self) -> None:
        self.stop_accepting()
        task = self._worker
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self._retired_models:
            _, pending = await asyncio.wait(tuple(self._retired_models), timeout=_MODEL_CLOSE_GRACE)
            if pending:
                _LOG.warning("model cleanup still running after shutdown grace; pending=%d", len(pending))
        if self._owns_compactions:
            await self._compactions.close()
        self._pending.clear()
        self._queued_inbound.clear()
        self._worker = None
        self._idle.set()

    async def _drain(self) -> None:
        try:
            while self._pending and not self._closed and not self._input_blocked and self._has_model_slot():
                await self._process(self._pending[0], uuid4().hex)
        finally:
            self._worker = None
            self._idle.set()

    async def _record_head(self, run_id: str) -> tuple[list[int], bool]:
        batch = self._pending[0]
        recorded = await self._record_input(batch, run_id)
        # 确认写入、空批次或旧路由作废后才能交还队首；失败时保留输入和去重。
        self._pending.popleft()
        self._queued_inbound.difference_update(batch.persisted_ids)
        return recorded

    async def _record_input(self, batch: _InputBatch, run_id: str) -> tuple[list[int], bool]:
        rows: list[Message | int] = []
        has_memory_user = False
        for item in batch.items:
            if isinstance(item, PersistedInput):
                rows.append(item.batch_id)
            elif isinstance(item, InternalInput):
                rows.append(item.to_model_message())
            else:
                rows.append(item)
                has_memory_user = True

        def append() -> tuple[list[int], bool]:
            if not batch.persisted_ids:
                messages = tuple(row for row in rows if isinstance(row, Message))
                return self.history.append(self.session_id, messages, retryable_io=True), has_memory_user
            if len(rows) == 1 and isinstance(rows[0], int):
                ids = self.history.consume_inbound(self.session_id, rows[0])
                return ids, bool(ids)
            ids, consumed_user = self.history.append_with_inbound(self.session_id, rows)
            return ids, has_memory_user or consumed_user

        try:
            ids, has_user_input = await self._sqlite.write_input(append)
        except Exception as exc:
            raise _InputWriteFailed from exc
        _LOG.debug("runtime input recorded; run=%s persisted=%d messages=%d user_input=%s",
                   run_id, len(batch.persisted_ids), len(ids), has_user_input)
        return ids, has_user_input

    async def _process(self, entry: _InputBatch | Message, run_id: str) -> None:
        self.last_failure = None
        clock = asyncio.get_running_loop().time
        started = clock()
        try:
            if isinstance(entry, _InputBatch):
                message_ids, has_user_input = await self._record_head(run_id)
            else:
                # 独立子任务由 dispatcher 管理，不走队首的有界重试或重新投放。
                message_ids = await self._sqlite(self.history.append, self.session_id, (entry,))
                has_user_input = False
            if message_ids:
                await self._run(run_id, message_ids[0], has_user_input=has_user_input)
        except _InputWriteFailed as exc:
            self.last_failure = "input_write_failed"
            self._input_blocked = True
            cause = exc.__cause__
            if isinstance(cause, RetryableInputWrite):
                cause = cause.__cause__
            _LOG.error("runtime input write failed; queue blocked; run=%s error_type=%s sqlite_code=%s",
                       run_id, type(cause).__name__, getattr(cause, "sqlite_errorcode", None))
        except asyncio.CancelledError:
            _LOG.debug("runtime run cancelled; run=%s", run_id)
            raise
        except InstructionsError as exc:
            self.last_failure = exc.diagnostic
            _LOG.warning("Local system instructions unavailable (%s); run=%s", exc.diagnostic, run_id)
        except (ModelError, TimeoutError) as exc:
            self.last_failure = "context_failed"
            _LOG.warning("runtime context failed (%s); run=%s",
                         "timeout" if isinstance(exc, TimeoutError) else exc.diagnostic, run_id)
        except Exception as exc:
            self.last_failure = "runtime_failed"
            _LOG.error("session runtime could not finish the current input; run=%s error_type=%s",
                       run_id, type(exc).__name__)
        _LOG.debug("runtime run finished; run=%s status=%s elapsed_ms=%.0f",
                   run_id, self.last_failure or "complete", (clock() - started) * 1000)

    async def _run(self, run_id: str, protected_from: int, *, has_user_input: bool) -> None:
        def can_send() -> bool:
            return not self._closed

        for index in range(self._settings.max_tool_rounds):
            if not can_send():
                return
            last_input_tokens = await self._sqlite(self.history.last_input_tokens, self.session_id)
            messages = await self._context.prepare(self.session_id, thinking=self._thinking,
                                                    tools=self._tools.specs, instructions=self._instructions,
                                                    protected_from=protected_from, can_send=can_send,
                                                    last_input_tokens=last_input_tokens)
            if messages is None:
                return
            model_status, continue_tools = await self._reply(
                run_id, messages, has_user_input=has_user_input)
            if model_status != "complete":
                if model_status != "model_interrupted":
                    self.last_failure = model_status
                return
            if not continue_tools:
                return
            if index + 1 == self._settings.max_tool_rounds:
                self.last_failure = "tool_round_limit"
                return
            if not can_send():
                return
            await self._wait_for_model_slot()
            if not can_send():
                return
            while self._pending:
                _, recorded_user_input = await self._record_head(run_id)
                has_user_input = has_user_input or recorded_user_input

    async def _reply(self, run_id: str, messages: Sequence[Message], *,
                     has_user_input: bool) -> tuple[str, bool]:
        loop = asyncio.get_running_loop()
        info = ReplyInfo(uuid4().hex, self.session_id, run_id, loop.time())
        text: list[str] = []
        reasoning: list[str] = []
        replay = None
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
        queued_run = self._channel is not None
        model_complete = False
        reply_closed = False
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
                            if event.complete and queued_run:
                                deadline.disarm()
                                model_complete = True
                                finish_reply_stream()  # 回执可先于连接关闭结算；完整投递仍等连接收尾。
                                return True
                        elif isinstance(event, TextDelta) and event.text:
                            if not text:
                                _LOG.debug("model first text; reply=%s elapsed_ms=%.0f", info.reply_id,
                                           (loop.time() - info.started_at) * 1000)
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
                        elif isinstance(event, ReasoningDelta) and self._thinking:
                            reasoning.append(event.text)
                        elif isinstance(event, InputUsage):
                            input_tokens = event.input_tokens
                        elif isinstance(event, ReplayReady):
                            if replay is not None or event.replay.scope != self._replay_scope:
                                raise ModelError("invalid_sse_event")
                            replay = event.replay
                        elif isinstance(event, ToolReady):
                            if self._channel is None:
                                raise ModelError("internal_tool_call")
                            call = event.call
                            if call.id in scheduled:
                                raise ModelError("duplicate_tool_id")
                            context = ToolContext(self.session_id, run_id, call.id, has_user_input)
                            scheduled[call.id] = (call, loop.create_task(self._tools.execute(call, context)))
            return True

        model_task: asyncio.Task[bool] | None = None

        def retire_reader(task: asyncio.Task[bool]) -> None:
            nonlocal accept_model_events
            accept_model_events = False
            if read_deadline is not None:
                read_deadline.disarm()
            self._retired_models.add(task)
            task.add_done_callback(self._model_cleanup_finished)
            if not reader_closing or not queued_run:
                if (stop_stream is None or not stop_stream(task)) and not task.cancelling():
                    task.cancel()

        async def complete_model() -> str:
            nonlocal model_task
            if not queued_run and stop_stream is None:
                return "complete" if await read_model() else "model_interrupted"
            model_task = loop.create_task(read_model())
            while not model_task.done():
                if voice_interrupted():
                    retire_reader(model_task)
                    return "complete" if model_complete else "model_interrupted"
                signal = delivery if delivery is not None else delivery_started
                # wait 不会把分派 worker 的取消直接传给原生读流；由 retire_reader 启动关闭预算。
                waiting = (model_task,) if not queued_run or signal.done() else (model_task, signal)
                await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
            if model_complete:
                self._report_model_cleanup(model_task)
                return "complete"
            if queued_run and model_task.cancelled():
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
            except TimeoutError:
                model_status = "model_timeout"
                _LOG.warning("model request timed out; reply=%s", info.reply_id)
            except ModelError as exc:
                model_status = "model_failed"
                model_error = exc.diagnostic
                _LOG.warning("model response failed (%s); reply=%s", model_error, info.reply_id)
            except Exception:
                model_status = "model_failed"
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
            generated_text = generated if not queued_run and model_status == "complete" else ""
            confirmed_text = receipt.confirmed if receipt else ""
            calls = tuple(call for call, _ in scheduled.values())
            if scheduled:
                await asyncio.wait([task for _, task in scheduled.values()])
            results = [Message("tool", '{"error":"tool_cancelled"}' if task.cancelled() else task.result(),
                                 tool_call_id=call.id) for call, task in scheduled.values()]
            # 完整模型流的协议回放元数据可保留成套工具调用；适配器仍仅从确认正文回放文本。
            if model_status != "complete":
                replay = None
            # 零投递的纯文本只留投递结果，不创建模型可见的助理消息。
            keep_reply = (receipt.status != DeliveryStatus.ZERO if receipt is not None else
                          bool(generated_text or replay is not None))
            rows = ([Message("assistant", confirmed_text if receipt else generated_text, calls,
                              reasoning="".join(reasoning) if model_status == "complete" and reasoning else None,
                              replay=replay),
                       *results]
                      if keep_reply or calls else [])
            outcome = ReplyOutcome(info.reply_id, info.session_id, run_id, model_status,
                                   receipt.status if receipt else None, model_error=model_error,
                                   input_tokens=input_tokens if model_status == "complete" else None)
            await self._sqlite(self.history.finalize_reply, outcome, rows)
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
            return model_status, bool(calls) and not voice_incomplete
        finally:
            delivery_started.cancel()
            unfinished = [task for _, task in scheduled.values() if not task.done()]
            if accept_model_events and model_task is not None:
                if not model_task.done():
                    retire_reader(model_task)
                    if not queued_run:
                        # 子任务仍持有串行执行权，原生传输收尾或耗尽预算后才能退出。
                        await asyncio.wait((model_task,))
                elif not model_task.cancelled():
                    model_task.exception()
            if delivery is not None:
                if not delivery.done():
                    self._abandon_delivery(reply, delivery)
                elif not delivery.cancelled():
                    delivery.exception()
            for task in unfinished:
                task.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)

    def _model_cleanup_finished(self, task: asyncio.Task[bool]) -> None:
        self._retired_models.discard(task)
        self._report_model_cleanup(task)
        if self.last_failure == "model_cleanup_blocked":
            # 只在碰到清理上限后恢复保留的队首；正常路径不等待退役任务。
            self._start_worker(task.get_loop())

    @staticmethod
    def _report_model_cleanup(task: asyncio.Task[bool]) -> None:
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                detail = error.diagnostic if isinstance(error, ModelError) else type(error).__name__
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
