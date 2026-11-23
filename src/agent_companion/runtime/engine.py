"""单 Session 的运行循环；网络、通道和工具运行在同一 asyncio 事件循环。"""

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass
import logging
from pathlib import Path
from uuid import uuid4

from ..config import RuntimeSettings
from ..protocol import (BackgroundChannel, Channel, DeliveryStatus, InputUsage, Message, Model, ModelError,
                        ModelEvent, ReasoningDelta, Receipt, ReplayReady, ReplayScope, ReplyInfo,
                        ReplyStream, TextDelta, ToolCall, ToolReady, ToolSpec)
from ..storage.sqlite import History, ReplyOutcome
from .context import Context
from .instructions import InstructionsError
from .tools import ToolContext, Tools


_LOG = logging.getLogger(__name__)
_DELIVERY_TIMEOUT = 120  # 无播放进展语义的回执通道（例如离线控制台）


@dataclass(frozen=True, slots=True)
class _InputBatch:
    messages: tuple[Message, ...]
    has_user_input: bool
    wait_for_delivery: bool
    persisted_id: int | None = None


class Runtime:
    """每个 Session 一个队列和一个活动运行；可共享模型、工具和 SQLite。"""

    def __init__(self, session_id: str, history: History, model: Model, channel: Channel | None, *,
                 tools: Tools | None = None, settings: RuntimeSettings | None = None,
                 thinking: bool = False, instructions: str = "", delivery_idle_timeout: float | None = None,
                 on_confirmed_reply: Callable[[ReplyInfo, str], None] | None = None) -> None:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        self.session_id = session_id
        self.history = history
        self._model = model
        self._channel = channel
        self._on_confirmed_reply = on_confirmed_reply
        self._delivery_idle_timeout = delivery_idle_timeout
        self._tools = tools if tools is not None else Tools()
        self._settings = settings if settings is not None else RuntimeSettings()
        self._thinking = thinking
        self._instructions = instructions
        replay_scope = getattr(model, "replay_scope", None)
        self._replay_scope = replay_scope if isinstance(replay_scope, ReplayScope) else None
        self._context = Context(history, self._settings, self._summary_stream,
                                 replay_scope=self._replay_scope, agents_path=Path.cwd() / "AGENTS.md")
        self._pending: deque[_InputBatch] = deque()
        self._queued_inbound: set[int] = set()
        self._worker: asyncio.Task[None] | None = None
        self._retired_deliveries: set[asyncio.Task[Receipt]] = set()
        self._closed = False
        self._idle = asyncio.Event()
        self._idle.set()
        self.last_failure: str | None = None

    def submit_batch(self, messages: Sequence[Message], *, wait_for_delivery: bool = True) -> None:
        """在事件循环内整批追加用户输入；不阻塞当前响应。"""
        self._submit(messages, has_user_input=True, wait_for_delivery=wait_for_delivery)

    def submit_subtask_result(self, message: Message, *, wait_for_delivery: bool = True) -> None:
        """后台回投也走同一个队列，但不会被当成新用户输入授权委派。"""
        self._submit((message,), has_user_input=False, wait_for_delivery=wait_for_delivery)

    def _check_user_channel(self, wait_for_delivery: bool) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if self._channel is None:
            raise RuntimeError("internal runs must be scheduled by the subtask dispatcher")
        if not wait_for_delivery and not isinstance(self._channel, BackgroundChannel):
            raise ValueError("channel does not support background delivery")

    def _submit(self, messages: Sequence[Message], *, has_user_input: bool, wait_for_delivery: bool) -> None:
        self._check_user_channel(wait_for_delivery)
        batch = tuple(messages)
        if not batch:
            return
        if any(item.role != "user" or item.tool_calls or item.tool_call_id or item.reasoning is not None
               or item.replay is not None
               for item in batch):
            raise ValueError("runtime inputs must be plain user messages")
        self._enqueue(_InputBatch(batch, has_user_input, wait_for_delivery))

    def submit_persisted_batch(self, batch_id: int, *, wait_for_delivery: bool = True) -> None:
        """由持久化入口移交批次；不在安全边界之前改变模型可读历史。"""
        self._check_user_channel(wait_for_delivery)
        if batch_id not in self._queued_inbound:
            self._enqueue(_InputBatch((), True, wait_for_delivery, batch_id))
            self._queued_inbound.add(batch_id)

    def _enqueue(self, batch: _InputBatch) -> None:
        loop = asyncio.get_running_loop()
        self._pending.append(batch)
        _LOG.debug("runtime input queued; batch=%s messages=%d user_input=%s pending=%d",
                   batch.persisted_id, len(batch.messages), batch.has_user_input, len(self._pending))
        self._idle.clear()
        if self._worker is None:
            self._worker = loop.create_task(self._drain())

    async def wait_idle(self) -> None:
        await self._idle.wait()

    async def run_internal(self, run_id: str, prompt: str) -> str | None:
        """运行独立的子会话话轮，返回失败代码；由调用方保证同会话串行。"""
        if self._closed or self._channel is not None or self._tools.specs:
            raise RuntimeError("internal runs require a runtime without a user channel or tools")
        await self._process(_InputBatch((Message("user", prompt),), has_user_input=False, wait_for_delivery=False), run_id)
        return self.last_failure

    def stop_accepting(self) -> None:
        """同步关停入口；可以在调度异步清理之前调用。"""
        self._closed = True

    async def close(self) -> None:
        self.stop_accepting()
        task = self._worker
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._pending.clear()
        self._queued_inbound.clear()
        self._worker = None
        self._idle.set()

    async def _drain(self) -> None:
        try:
            while self._pending and not self._closed:
                await self._process(self._pending.popleft(), uuid4().hex)
        finally:
            self._worker = None
            self._idle.set()

    def _record_input(self, batch: _InputBatch, run_id: str) -> list[int]:
        if batch.persisted_id is None:
            ids = self.history.append(self.session_id, batch.messages)
        else:
            try:
                ids = self.history.consume_inbound(self.session_id, batch.persisted_id)
            finally:
                self._queued_inbound.discard(batch.persisted_id)
        _LOG.debug("runtime input recorded; run=%s batch=%s messages=%d user_input=%s wait_for_delivery=%s",
                   run_id, batch.persisted_id, len(ids), batch.has_user_input, batch.wait_for_delivery)
        return ids

    async def _process(self, batch: _InputBatch, run_id: str) -> None:
        self.last_failure = None
        clock = asyncio.get_running_loop().time
        started = clock()
        try:
            message_ids = self._record_input(batch, run_id)
            if message_ids:
                await self._run(run_id, message_ids[0], has_user_input=batch.has_user_input,
                                wait_for_delivery=batch.wait_for_delivery)
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

    async def _run(self, run_id: str, protected_from: int, *, has_user_input: bool,
                   wait_for_delivery: bool) -> None:
        for index in range(self._settings.max_tool_rounds):
            messages = await self._context.prepare(self.session_id, thinking=self._thinking,
                                                   tools=self._tools.specs, instructions=self._instructions,
                                                   protected_from=protected_from,
                                                   last_input_tokens=self.history.last_input_tokens(self.session_id))
            model_status, has_calls = await self._reply(run_id, messages, has_user_input=has_user_input,
                                                       wait_for_delivery=wait_for_delivery)
            if model_status != "complete":
                self.last_failure = model_status
                return
            if not has_calls:
                return
            if index + 1 == self._settings.max_tool_rounds:
                self.last_failure = "tool_round_limit"
                return
            if self._closed:
                return
            while self._pending:
                batch = self._pending.popleft()
                if self._record_input(batch, run_id):
                    has_user_input = has_user_input or batch.has_user_input

    async def _summary_stream(self, messages: Sequence[Message], tools: Sequence[ToolSpec],
                              thinking: bool) -> AsyncGenerator[ModelEvent, None]:
        async with asyncio.timeout(self._settings.model_timeout):
            async with aclosing(self._model.stream(messages, tools, thinking=thinking)) as stream:
                async for event in stream:
                    yield event

    async def _reply(self, run_id: str, messages: Sequence[Message], *,
                     has_user_input: bool, wait_for_delivery: bool) -> tuple[str, bool]:
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

        validate_progress = is_generated_prefix if wait_for_delivery and self._delivery_idle_timeout is not None else None
        reply = ReplyStream(info, validate_progress=validate_progress)
        channel = self._channel

        async def deliver() -> Receipt:
            assert channel is not None
            try:
                return await channel.deliver(reply)
            finally:
                reply.delivery_finished_at = loop.time()

        delivery: asyncio.Task[Receipt] | None = None
        model_status = "complete"
        model_error = None
        input_tokens: int | None = None
        _LOG.debug("model request started; run=%s reply=%s messages=%d tools=%d thinking=%s",
                   run_id, info.reply_id, len(messages), len(self._tools.specs), self._thinking)
        try:
            try:
                async with asyncio.timeout(self._settings.model_timeout):
                    async with aclosing(self._model.stream(messages, self._tools.specs,
                                                           thinking=self._thinking)) as events:
                        async for event in events:
                            if isinstance(event, TextDelta) and event.text:
                                if not text:
                                    _LOG.debug("model first text; reply=%s elapsed_ms=%.0f", info.reply_id,
                                               (loop.time() - info.started_at) * 1000)
                                text.append(event.text)
                                if self._channel is not None and wait_for_delivery:
                                    if delivery is None:
                                        delivery = loop.create_task(deliver())
                                    if not delivery.done():
                                        reply.publish(event.text)
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
                                context = ToolContext(self.session_id, run_id, call.id, has_user_input=has_user_input)
                                scheduled[call.id] = (call, loop.create_task(self._tools.execute(call, context)))
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
                if delivery is not None:
                    reply.start_delivery_wait(self._delivery_idle_timeout, _DELIVERY_TIMEOUT)
                reply.close()

            generated = "".join(text)
            _LOG.debug("model request finished; reply=%s status=%s input_tokens=%s output_chars=%d tools=%d elapsed_ms=%.0f",
                       info.reply_id, model_status, input_tokens, len(generated), len(scheduled),
                       (loop.time() - info.started_at) * 1000)
            # 工具和投递一直独立运行；这个汇合点只提交成套的终态数据。
            receipt = await self._delivery_result(reply, delivery, generated)
            # 内部会话与后台发送都只保存成功生成的全文，不制造回执。
            generated_text = generated if not wait_for_delivery and model_status == "complete" else ""
            confirmed_text = receipt.confirmed if receipt else ""
            calls = tuple(call for call, _ in scheduled.values())
            if scheduled:
                await asyncio.wait([task for _, task in scheduled.values()])
            results = [Message("tool", '{"error":"tool_cancelled"}' if task.cancelled() else task.result(),
                                 tool_call_id=call.id) for call, task in scheduled.values()]
            if model_status != "complete":
                replay = None
            rows = ([Message("assistant", confirmed_text if receipt else generated_text, calls,
                              reasoning="".join(reasoning) if model_status == "complete" and reasoning else None,
                              replay=replay),
                       *results]
                      if receipt is not None or calls or generated_text or replay is not None else [])
            self.history.finalize_reply(
                ReplyOutcome(info.reply_id, info.session_id, run_id, model_status,
                             receipt.status if receipt else None, model_error=model_error,
                             input_tokens=input_tokens if model_status == "complete" else None), rows)
            _LOG.debug("reply committed; reply=%s receipt=%s confirmed_chars=%d tool_results=%d elapsed_ms=%.0f",
                       info.reply_id, receipt.status if receipt else None, len(confirmed_text), len(results),
                       (loop.time() - info.started_at) * 1000)
            # 外部副本与主投递状态无关，只在历史事务成功后尽力入队。
            try:
                if confirmed_text and self._on_confirmed_reply is not None:
                    self._on_confirmed_reply(info, confirmed_text)
                elif generated_text and isinstance(self._channel, BackgroundChannel):
                    self._channel.enqueue(info, generated_text)
            except Exception:
                _LOG.warning("background reply could not be queued; reply=%s", info.reply_id)
            return model_status, bool(calls)
        finally:
            unfinished = [task for _, task in scheduled.values() if not task.done()]
            if delivery is not None:
                if not delivery.done():
                    self._abandon_delivery(reply, delivery)
                elif not delivery.cancelled():
                    delivery.exception()
            for task in unfinished:
                task.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)

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
            now, deadline, confirmed = reply.delivery_snapshot()
            finished = reply.delivery_finished_at
            if delivery.done() and (now if finished is None else finished) < deadline:
                return None
            if now >= deadline:
                return confirmed
            # 设备只会延后截止；按旧截止唤醒时，重新同步取证即可。
            await asyncio.wait((delivery,), timeout=max(0, deadline - clock()))

    async def _delivery_result(self, reply: ReplyStream, delivery: asyncio.Task[Receipt] | None,
                               generated: str) -> Receipt | None:
        if delivery is None:
            return None
        info = reply.info
        failed = Receipt(info.reply_id, DeliveryStatus.UNKNOWN)
        timeout_prefix = await self._wait_delivery(reply, delivery)
        if timeout_prefix is not None:
            self._abandon_delivery(reply, delivery)
            if self._delivery_idle_timeout is not None:
                _LOG.warning("reply delivery idle deadline reached; reply=%s", info.reply_id)
            return Receipt(info.reply_id, DeliveryStatus.UNKNOWN, timeout_prefix if generated.startswith(timeout_prefix) else "")
        if delivery.cancelled():
            return failed
        try:
            receipt = delivery.result()
        except Exception:
            return failed
        if (not isinstance(receipt, Receipt) or receipt.reply_id != info.reply_id
                or not isinstance(receipt.status, DeliveryStatus) or not isinstance(receipt.confirmed, str)
                or not generated.startswith(receipt.confirmed)
                or (receipt.status == DeliveryStatus.FULL and generated != receipt.confirmed)
                or (receipt.status == DeliveryStatus.ZERO and receipt.confirmed)
                or (receipt.status == DeliveryStatus.PARTIAL and not receipt.confirmed)):
            return failed
        return receipt
