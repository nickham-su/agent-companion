"""DeepSeek Chat Completions 的请求投影与 SSE 解析。"""

from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
import json
from typing import Any

import httpx

from ..protocol import (Image, Message, ModelError, ModelEvent, ReasoningDelta, RequestTooLarge,
                        Text, TextDelta, ToolCall, ToolReady, ToolSpec)
from ._request import JSON_ENCODER, request_body


CHAT_COMPLETIONS = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-flash"
# 供应商公布 48 MiB；本机更早裁剪，为请求开销预留余量。
MAX_REQUEST_BYTES = 40 * 1024 * 1024


def _content_part(part: Text | Image) -> dict[str, Any]:
    if isinstance(part, Image):
        return {"type": "image_url", "image_url": {"url": part.url}}
    return {"type": "text", "text": part.text}


def build_request(messages: Sequence[Message], tools: Sequence[ToolSpec], *, thinking: bool,
                  model: str = MODEL_NAME, reasoning_effort: str | None = None) -> dict[str, Any]:
    result: list[dict[str, Any]] = []
    for message in messages:
        content: Any = message.content
        if not isinstance(content, str):
            content = [_content_part(part) for part in content]
        item = {"role": message.role, "content": content}
        if message.tool_calls:
            item["content"] = content or None
            item["tool_calls"] = [{"type": "function", "id": call.id,
                                   "function": {"name": call.name, "arguments": call.arguments}}
                                  for call in message.tool_calls]
        if message.tool_call_id is not None:
            item["tool_call_id"] = message.tool_call_id
        if thinking and tools and message.role == "assistant":
            if message.reasoning is None:
                raise ModelError("missing_reasoning")
            item["reasoning_content"] = message.reasoning
        result.append(item)
    payload = {"model": model, "stream": True, "messages": result,
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

    def complete(self) -> ToolCall:
        name = "".join(self.name)
        if not self.id or not name:
            raise ModelError("incomplete_tool_call")
        arguments = "".join(self.arguments)
        try:
            json.loads(arguments)
        except (ValueError, RecursionError):
            raise ModelError("invalid_tool_arguments") from None
        return ToolCall(self.id, name, arguments)


async def parse_stream(lines: AsyncIterator[str]) -> AsyncIterator[ModelEvent]:
    fragments: dict[int, _ToolParts] = {}
    finish: str | None = None
    async for event in _events(lines):
        if event == "[DONE]":
            if finish == "tool_calls" and fragments:
                calls = [parts.complete() for _, parts in sorted(fragments.items())]
                if len({call.id for call in calls}) != len(calls):
                    raise ModelError("duplicate_tool_id")
                for call in calls:
                    yield ToolReady(call)
            elif finish != "stop" or fragments:
                raise ModelError("invalid_stream_finish")
            return
        try:
            item = json.loads(event)
            if not isinstance(item, dict) or "error" in item:
                raise ValueError
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
            for call in delta.get("tool_calls") or ():
                index = call["index"]
                if type(index) is not int or index < 0:
                    raise ValueError
                fragments.setdefault(index, _ToolParts()).append(call)
            reason = choice.get("finish_reason")
            if reason is not None:
                if reason not in ("stop", "tool_calls"):
                    raise ModelError("abnormal_stream_finish")
                finish = reason
        except (ValueError, KeyError, TypeError, AttributeError):
            raise ModelError("invalid_sse_event") from None
        if reasoning is not None:
            yield ReasoningDelta(reasoning)
        if content:
            yield TextDelta(content)
    raise ModelError("truncated_stream")


class DeepSeek:
    def __init__(self, client: httpx.AsyncClient, *, request_limit: int = MAX_REQUEST_BYTES,
                 model: str = MODEL_NAME, reasoning_effort: str | None = None) -> None:
        if type(request_limit) is not int or not 0 < request_limit <= MAX_REQUEST_BYTES:
            raise ValueError("invalid DeepSeek request body limit")
        self._client = client
        self._request_limit = request_limit
        self._model = model
        self._effort = reasoning_effort

    def _payload(self, messages: Sequence[Message], tools: Sequence[ToolSpec], thinking: bool) -> dict[str, Any]:
        return build_request(messages, tools, model=self._model,
                             thinking=thinking and self._effort != "disabled", reasoning_effort=self._effort)

    def request_fits(self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, thinking: bool) -> bool:
        return self.remaining_request_bytes(messages, tools, thinking=thinking) is not None

    def remaining_request_bytes(self, messages: Sequence[Message], tools: Sequence[ToolSpec], *,
                                thinking: bool) -> int | None:
        try:
            return request_body(self._payload(messages, tools, thinking), self._request_limit)
        except RequestTooLarge:
            return None

    def image_replacement_bytes(self, image: Image, placeholder: Text) -> int:
        return (len(JSON_ENCODER.encode(_content_part(image)).encode("utf-8"))
                - len(JSON_ENCODER.encode(_content_part(placeholder)).encode("utf-8")))

    async def stream(self, messages: Sequence[Message], tools: Sequence[ToolSpec], *,
                     thinking: bool) -> AsyncGenerator[ModelEvent, None]:
        # 对将要发送的准确字节做最终硬检查，避免 httpx 再序列化后改变正文大小。
        body = bytearray()
        request_body(self._payload(messages, tools, thinking), self._request_limit, output=body)
        try:
            async with self._client.stream("POST", CHAT_COMPLETIONS, content=bytes(body),
                                           headers={"Accept": "text/event-stream",
                                                    "Content-Type": "application/json"}) as response:
                if not response.is_success:
                    raise ModelError(f"http_{response.status_code}")
                async with aclosing(parse_stream(response.aiter_lines())) as events:
                    async for event in events:
                        yield event
        except httpx.RequestError:
            raise ModelError("transport_error") from None
