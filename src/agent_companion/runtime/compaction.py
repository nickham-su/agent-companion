"""进程内摘要待办：生成只接收冻结输入，不读写会话。"""

import asyncio
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass
import logging
from time import monotonic

from ..images import prepare_request_images
from ..model_diagnostics import ModelRequestDiagnostics
from ..protocol import Message, Model, ModelError, TextDelta, ToolReady, ToolSpec
from ..storage.sqlite import History, Summary
from ..workspace import Workspace
from .model_deadline import model_deadline_at


_LOG = logging.getLogger(__name__)
_SUMMARY_BACKOFF_SECONDS = 30
_READY_SECONDS = 300
_MAX_REMEMBERED_SESSIONS = 32


@dataclass(frozen=True, slots=True)
class Prefix:
    base: int
    first: int
    through: int


@dataclass(frozen=True, slots=True)
class _Remembered:
    prefix: Prefix
    expires_at: float
    summary: Summary | None = None  # 无摘要表示失败冷却。


class Compactions:
    """同一 History 的应用实例最多生成一份摘要，不排队。"""

    def __init__(self, history: History) -> None:
        self._history = history
        self._generating: asyncio.Task[None] | None = None
        self._active_model: Model | None = None
        self._remembered: OrderedDict[str, _Remembered] = OrderedDict()
        self._closed = False

    def check_owner(self, history: History) -> None:
        if self._history is not history:
            raise ValueError("compactions must share the same history")

    def stop(self) -> None:
        if not self._closed:
            self._closed = True
            self._remembered.clear()
            if (reader := self._generating) is not None:
                stop_stream = getattr(self._active_model, "stop_stream", None)
                if stop_stream is None or not stop_stream(reader):
                    reader.cancel()

    async def close(self) -> None:
        self.stop()
        if self._generating is not None:
            # 适配器须配合取消并关闭响应流；槽在清理完成前不交给另一个生成任务。
            pending = asyncio.gather(self._generating, return_exceptions=True)
            interrupted = None
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError as exc:
                    # 外部取消只影响此等待方，不能打断原生传输正在进行的清理。
                    interrupted = exc
            # 被取消时任务可能还没开始，因此生成协程的 finally 不一定执行。
            self._generating = None
            self._active_model = None
            if interrupted is not None:
                raise interrupted

    def _prune(self) -> None:
        now = monotonic()
        for session_id, entry in tuple(self._remembered.items()):
            if entry.expires_at <= now:
                self.discard(session_id)

    def _remember(self, session_id: str, entry: _Remembered) -> None:
        if self._closed:
            return
        self._prune()
        self.discard(session_id)
        self._remembered[session_id] = entry
        if len(self._remembered) > _MAX_REMEMBERED_SESSIONS:
            self._remembered.popitem(last=False)

    def discard(self, session_id: str) -> None:
        self._remembered.pop(session_id, None)

    def ready(self, session_id: str, covered: int) -> tuple[Prefix, Summary] | None:
        self._prune()
        entry = self._remembered.get(session_id)
        if entry is not None:
            if entry.prefix.base != covered:
                self.discard(session_id)
            elif entry.summary is not None:
                return entry.prefix, entry.summary
        return None

    def can_start(self, session_id: str, covered: int, first: int) -> bool:
        # 在同一同步准备点中，调用方先检查 ready（同时清理过期条目）。
        entry = self._remembered.get(session_id)
        return not self._closed and self._generating is None and (
            entry is None or (entry.summary is None and (entry.prefix.base, entry.prefix.first) != (covered, first))
        )

    def start(self, session_id: str, prefix: Prefix, prompt: list[Message], tools: Sequence[ToolSpec], *,
               model: Model, thinking: bool, timeout: float, run_id: str | None = None,
               instructions: str | None = None, workspace: Workspace | None = None) -> None:
        # 与 can_start 之间无 await；冻结工具定义中可变的 schema。
        if self._closed or self._generating is not None:
            return
        try:
            frozen_tools = deepcopy(tuple(tools))
        except Exception:
            self.failed(session_id, prefix, "summary_freeze_failed")
            return
        self._active_model = model
        self._generating = asyncio.create_task(self._generate(
            session_id, prefix, tuple(prompt), frozen_tools, model, thinking, timeout, run_id, instructions, workspace))

    def failed(self, session_id: str, prefix: Prefix, reason: str) -> None:
        if not self._closed:
            self._remember(session_id, _Remembered(prefix, monotonic() + _SUMMARY_BACKOFF_SECONDS))
            _LOG.warning("model summary unavailable (%s); preserving original context", reason)

    async def _generate(self, session_id: str, prefix: Prefix, prompt: tuple[Message, ...],
                         tools: tuple[ToolSpec, ...], model: Model, thinking: bool, timeout: float, run_id: str | None,
                         instructions: str | None, workspace: Workspace | None) -> None:
        diagnostics = ModelRequestDiagnostics("summary", timeout, run_id=run_id)
        try:
            loop = asyncio.get_running_loop()
            expires = loop.time() + timeout
            stop_stream = getattr(model, "stop_stream", None)
            with diagnostics.activate():
                async with model_deadline_at(expires, stop_stream) as deadline:
                    request = await prepare_request_images(prompt, workspace)
                    if deadline.expired() or loop.time() >= expires:
                        raise TimeoutError
                    if self._closed or asyncio.current_task().cancelling():
                        raise asyncio.CancelledError
                    content = await _summarize(model, request, tools, thinking, instructions)
            if deadline.expired() or loop.time() >= expires:
                raise TimeoutError
            self._remember(session_id, _Remembered(
                prefix, monotonic() + _READY_SECONDS, Summary(prefix.through, content)))
        except TimeoutError as exc:
            diagnostics.failed(exc)
            self.failed(session_id, prefix, "timeout")
        except ModelError as exc:
            diagnostics.failed(exc)
            self.failed(session_id, prefix, exc.diagnostic)
        except Exception as exc:
            diagnostics.failed(exc)
            self.failed(session_id, prefix, "summary_model_failed")
        finally:
            self._generating = None
            self._active_model = None


async def _summarize(model: Model, messages: Sequence[Message], tools: Sequence[ToolSpec], thinking: bool,
                     instructions: str | None = None) -> str:
    parts: list[str] = []
    async with aclosing(model.stream(messages, tools, thinking=thinking, instructions=instructions)) as events:
        async for event in events:
            if isinstance(event, ToolReady):
                raise ModelError("summary_tool_call")
            if isinstance(event, TextDelta):
                parts.append(event.text)
    content = "".join(parts)
    if not content.strip():
        raise ModelError("empty_summary")
    return content
