"""DeepSeek Chat Completions 的请求投影与 SSE 解析。"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
import json
from typing import Any

import httpx

from ..model_diagnostics import adapter_started, failure_category
from ..protocol import (InputUsage, Message, ModelClosing, ModelError, ModelEvent, PrivateData, TextDelta,
                        ToolCall, ToolReady, ToolSpec)
from ._chat_completions import ChatOutput, chat_messages, chat_reasoning
from ._http_stream import ResponseStreams
from ._request import make_replay_key, request_body


CHAT_COMPLETIONS = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-flash"
_ADAPTER_ID = "deepseek"
# 明文回放只按协议和模型区分；此常量不是凭据。
_REPLAY_NAMESPACE = b"agent-companion/deepseek-replay"


def _replay_key(model: str) -> bytes:
    return make_replay_key(_REPLAY_NAMESPACE, _ADAPTER_ID, model)


def build_request(messages: Sequence[Message], tools: Sequence[ToolSpec], *, thinking: bool,
                  model: str = MODEL_NAME, reasoning_effort: str | None = None,
                  instructions: str | None = None) -> dict[str, Any]:
    key = _replay_key(model)
    # 思考模式未声明工具时，DeepSeek 忽略历史 reasoning_content。
    result = chat_messages(messages, lambda message: chat_reasoning(message, key) if thinking and tools else None,
                           key=key, instructions=instructions)
    payload = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "messages": result,
               "thinking": {"type": "enabled" if thinking else "disabled"}}
    if thinking and reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if tools:
        payload["tools"] = [{"type": "function", "function": {"name": spec.name,
                             "description": spec.description, "parameters": spec.parameters}}
                            for spec in tools]
    return payload


async def _events(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    data = []
    async for line in lines:
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif line.startswith("data:"):
            item = line.removeprefix("data:").removeprefix(" ")
            data.append(item)
    # 按 SSE 协议，未到空行的事件不完整；不能把它当作完成。


@dataclass(slots=True)
class _ToolParts:
    id: str | None = None
    name: list[str] = field(default_factory=list)
    arguments: list[str] = field(default_factory=list)

    def append(self, delta: dict[str, Any]) -> None:
        if delta.get("type") not in (None, "function"):
            raise ModelError("unsupported_tool_type")
        identity = delta.get("id")
        if identity is not None:
            if not isinstance(identity, str) or not identity or self.id not in (None, identity):
                raise ModelError("invalid_tool_id")
            self.id = identity
        function = delta.get("function")
        if function is not None:
            if not isinstance(function, dict):
                raise ModelError("invalid_tool_delta")
            for key, target in (("name", self.name), ("arguments", self.arguments)):
                value = function.get(key)
                if value is not None:
                    if not isinstance(value, str):
                        raise ModelError("invalid_tool_delta")
                    target.append(value)

    def complete(self, index: int, key: bytes) -> ToolCall:
        name = "".join(self.name)
        if not self.id or not name:
            raise ModelError("incomplete_tool_call")
        arguments = "".join(self.arguments)
        try:
            json.loads(arguments)
        except (ValueError, RecursionError):
            raise ModelError("invalid_tool_arguments") from None
        return ToolCall(self.id, name, arguments, private=PrivateData(key, str(index).encode("ascii")))


async def parse_stream(lines: AsyncIterator[str], *, model: str = MODEL_NAME) -> AsyncIterator[ModelEvent]:
    fragments: dict[int, _ToolParts] = {}
    finish: str | None = None
    input_usage: InputUsage | None = None
    output = ChatOutput()
    key = _replay_key(model)
    async for event in _events(lines):
        if event == "[DONE]":
            calls: dict[int, ToolCall] = {}
            if finish == "tool_calls" and fragments:
                calls = {index: parts.complete(index, key) for index, parts in sorted(fragments.items())}
                if len({call.id for call in calls.values()}) != len(calls):
                    raise ModelError("duplicate_tool_id")
            elif finish != "stop" or fragments:
                raise ModelError("invalid_stream_finish")
            if ready := output.complete(key, calls):
                yield ready
            for call in calls.values():
                yield ToolReady(call)
            if input_usage is not None:
                yield input_usage
            return
        try:
            item = json.loads(event)
            if not isinstance(item, dict) or "error" in item:
                raise ValueError
            usage = item.get("usage")
            if isinstance(usage, dict):
                tokens = usage.get("prompt_tokens")
                if type(tokens) is int and 0 <= tokens < 2**63:
                    input_usage = InputUsage(tokens)
            choices = item["choices"]
            if not isinstance(choices, list) or len(choices) > 1:
                raise ValueError
            if not choices:
                continue  # 最终用量块
            choice = choices[0]
            if finish is not None or choice.get("index", 0) != 0:
                raise ValueError
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise ValueError
            reasoning, content = delta.get("reasoning_content"), delta.get("content")
            if ((reasoning is not None and not isinstance(reasoning, str)) or
                    (content is not None and not isinstance(content, str))):
                raise ValueError
            if reasoning is not None:
                output.text("reasoning", reasoning)
            if content:
                output.text("text", content)
            for call in delta.get("tool_calls") or ():
                index = call["index"]
                if type(index) is not int or index < 0:
                    raise ValueError
                fragments.setdefault(index, _ToolParts()).append(call)
                output.call(index)
            reason = choice.get("finish_reason")
            if reason is not None:
                if reason not in ("stop", "tool_calls"):
                    raise ModelError("abnormal_stream_finish")
                finish = reason
        except (ValueError, KeyError, TypeError, AttributeError):
            raise ModelError("invalid_sse_event") from None
        if content:
            yield TextDelta(content)
    raise ModelError("truncated_stream")


class DeepSeek:
    def __init__(self, client: httpx.AsyncClient, *, model: str = MODEL_NAME,
                 reasoning_effort: str | None = None) -> None:
        self._client = client
        self._responses = ResponseStreams()
        self._model = model
        self._effort = reasoning_effort
        self._replay_key = _replay_key(model)

    def _payload(self, messages: Sequence[Message], tools: Sequence[ToolSpec], thinking: bool, *,
                 instructions: str | None = None) -> dict[str, Any]:
        return build_request(messages, tools, model=self._model,
                             thinking=thinking and self._effort != "disabled", reasoning_effort=self._effort,
                             instructions=instructions)

    def history_replay_bytes(self, message: Message, tools: Sequence[ToolSpec], *, thinking: bool) -> int:
        if not (thinking and tools and self._effort != "disabled"):
            return 0
        reasoning = chat_reasoning(message, self._replay_key)
        return len(reasoning.encode("utf-8")) if reasoning is not None else 0

    async def stream(self, messages: Sequence[Message], tools: Sequence[ToolSpec], *,
                     thinking: bool, instructions: str | None = None) -> AsyncGenerator[ModelEvent, None]:
        diagnostics = adapter_started(_ADAPTER_ID)
        body = request_body(self._payload(messages, tools, thinking, instructions=instructions))
        if diagnostics is not None:
            diagnostics.request_ready(len(body))
        try:
            request = self._client.stream("POST", CHAT_COMPLETIONS, content=body,
                                          headers={"Accept": "text/event-stream", "Content-Type": "application/json"})
            async with self._responses.open(request, background_tail=True) as (response, allow_tail):
                if diagnostics is not None:
                    diagnostics.response_received(response.status_code)
                if not response.is_success:
                    raise ModelError(f"http_{response.status_code}")
                if diagnostics is not None:
                    diagnostics.stage = "sse_read"
                async with aclosing(parse_stream(response.aiter_lines(), model=self._model)) as events:
                    async for event in events:
                        yield event
                # 解析器只在完整的 [DONE] 后正常结束；此时数据和连接收尾是两个边界。
                allow_tail()
                if diagnostics is not None:
                    diagnostics.stage = "response_close"
                yield ModelClosing(complete=True)
        except httpx.RequestError as error:
            if diagnostics is not None:
                diagnostics.transport_failure = failure_category(error)
            raise ModelError("transport_error") from None

    def stop_stream(self, reader: asyncio.Task[object]) -> bool:
        return self._responses.stop(reader)

    async def aclose(self) -> None:
        await self._responses.aclose()
