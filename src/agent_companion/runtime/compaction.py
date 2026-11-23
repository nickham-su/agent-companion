"""进程内摘要待办：生成只接收冻结输入，不读写会话。"""

import asyncio
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass
import logging
from time import monotonic

from ..protocol import (Message, Model, ModelError, ProviderReplay, ReasoningDelta,
                        ReplayReady, TextDelta, ToolReady, ToolSpec)
from ..storage.sqlite import History, Summary


_LOG = logging.getLogger(__name__)
_SUMMARY_BACKOFF_SECONDS = 30
_READY_SECONDS = 300
_MAX_REMEMBERED_SESSIONS = 32
_PRIVATE_ECHO_MIN_CHARS = 32


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
    """同一 History 和 Model 的应用实例最多生成一份摘要，不排队。"""

    def __init__(self, history: History, model: Model) -> None:
        self._history = history
        self._model = model
        self._generating: asyncio.Task[None] | None = None
        self._remembered: OrderedDict[str, _Remembered] = OrderedDict()
        self._closed = False

    def check_owner(self, history: History, model: Model) -> None:
        if self._history is not history or self._model is not model:
            raise ValueError("compactions must share the same history and model")

    def stop(self) -> None:
        if not self._closed:
            self._closed = True
            self._remembered.clear()
            if self._generating is not None:
                self._generating.cancel()

    async def close(self) -> None:
        self.stop()
        if self._generating is not None:
            # 适配器须配合取消并关闭响应流；槽在清理完成前不交给另一个生成任务。
            await asyncio.gather(self._generating, return_exceptions=True)

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
              thinking: bool, timeout: float) -> None:
        # 与 can_start 之间无 await；冻结工具定义中可变的 schema。
        try:
            frozen_tools = deepcopy(tuple(tools))
        except Exception:
            self.failed(session_id, prefix, "summary_freeze_failed")
            return
        self._generating = asyncio.create_task(self._generate(
            session_id, prefix, tuple(prompt), frozen_tools, thinking, timeout))

    def failed(self, session_id: str, prefix: Prefix, reason: str) -> None:
        if not self._closed:
            self._remember(session_id, _Remembered(prefix, monotonic() + _SUMMARY_BACKOFF_SECONDS))
            _LOG.warning("model summary unavailable (%s); preserving original context", reason)

    async def _generate(self, session_id: str, prefix: Prefix, prompt: tuple[Message, ...],
                        tools: tuple[ToolSpec, ...], thinking: bool, timeout: float) -> None:
        try:
            loop = asyncio.get_running_loop()
            expires = loop.time() + timeout
            async with asyncio.timeout_at(expires) as deadline:
                content = await _summarize(self._model, prompt, tools, thinking)
            if deadline.expired() or loop.time() >= expires:
                raise TimeoutError
            self._remember(session_id, _Remembered(
                prefix, monotonic() + _READY_SECONDS, Summary(prefix.through, content)))
        except TimeoutError:
            self.failed(session_id, prefix, "timeout")
        except ModelError as exc:
            self.failed(session_id, prefix, exc.diagnostic)
        except Exception:
            self.failed(session_id, prefix, "summary_model_failed")
        finally:
            self._generating = None


def _private_replay_fields(replay: ProviderReplay) -> Iterator[str]:
    for item in replay.items:
        if item.kind == "reasoning":
            if item.encrypted_content:
                yield item.encrypted_content
            yield from item.summary


async def _summarize(model: Model, messages: Sequence[Message], tools: Sequence[ToolSpec], thinking: bool) -> str:
    parts: list[str] = []
    reasoning: list[str] = []
    private: list[str] = []
    for message in messages:
        if message.reasoning:
            private.append(message.reasoning)
        if message.replay is not None:
            private.extend(_private_replay_fields(message.replay))
    async with aclosing(model.stream(messages, tools, thinking=thinking)) as events:
        async for event in events:
            if isinstance(event, ToolReady):
                raise ModelError("summary_tool_call")
            if isinstance(event, TextDelta):
                parts.append(event.text)
            elif isinstance(event, ReasoningDelta):
                reasoning.append(event.text)
            elif isinstance(event, ReplayReady):
                private.extend(_private_replay_fields(event.replay))
    content = "".join(parts)
    text = content.strip()
    if not text:
        raise ModelError("empty_summary")
    if reasoning:
        private.append("".join(reasoning))
    for value in private:
        value = value.strip()
        if value and (text == value or (len(value) >= _PRIVATE_ECHO_MIN_CHARS and value in text)):
            raise ModelError("summary_private_echo")
    return content
