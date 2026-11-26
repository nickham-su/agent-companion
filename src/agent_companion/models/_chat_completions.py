"""MiniMax 与百炼共用的 Chat Completions 投影、SSE 解码和 HTTP 生命周期。"""

from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from itertools import groupby
import json
import re
from typing import Any

import httpx

from ..model_diagnostics import AdapterID, VendorResponseError, adapter_started, failure_category
from ..protocol import (AssistantPartsReady, Image, InputUsage, Message, ModelClosing, ModelError, ModelEvent,
                        Part, PrivateData, Reasoning, Text, TextDelta, ToolCall, ToolReady, ToolSpec)
from ._http_stream import ResponseStreams
from ._request import input_content, request_body


_ERROR_BODY_LIMIT = 64 * 1024
_LOG_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,256}")
_LOG_MESSAGE_LIMIT = 4096
_ECHO_SCAN_LIMIT = 128 * 1024
_ECHO_VALUE_LIMIT = 4096
_ECHO_COMPARISON_LIMIT = 2 * 1024 * 1024
_FRAGMENT_LENGTH = 8
_KEY_FRAGMENT_LENGTH = 10
_COMPLETE_FIELD_LENGTH = 24
_JSON_ESCAPE = re.compile(r'\\u[0-9a-fA-F]{4}(?:\\u[0-9a-fA-F]{4})?|\\["\\/bfnrt]')
_IDENTIFIER = re.compile(r"[A-Za-z0-9]+(?:[_.@/-][A-Za-z0-9]+)*")
_IDENTIFIER_SIGNAL = re.compile(r"[0-9]|(?<=[A-Za-z0-9])[_.@/-](?=[A-Za-z0-9])")
_QUOTED_VALUE = re.compile(r"""(?<!\w)(?:"([^"\r\n]+)"|'([^'\r\n]+)'|`([^`\r\n]+)`)(?!\w)""")
_INTRODUCED_VALUE = re.compile(r"(?<=: )[^\r\n]+")
_MARKED_TRUNCATION = re.compile(r"(?<![\w-])(\w[\w-]{3,})(?:\.\.\.|…)")


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
            for key, values in (("name", self.name), ("arguments", self.arguments)):
                piece = function.get(key)
                if piece is not None:
                    if not isinstance(piece, str):
                        raise ModelError("invalid_tool_delta")
                    values.append(piece)

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


async def _sse(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, str]]:
    kind = ""
    data: list[str] = []
    async for line in lines:
        if not line:
            if data:
                yield kind, "\n".join(data)
                data.clear()
            kind = ""
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        elif line.startswith("event:"):
            kind = line[6:].removeprefix(" ")
    # 不把未遇到空行的事件当作完整事件。


def _json_value(data: str | bytes | bytearray) -> Any:
    try:
        return json.loads(data)
    except (ValueError, RecursionError):
        return None


def _json_object(data: str | bytes | bytearray) -> dict[str, Any] | None:
    value = _json_value(data)
    return value if isinstance(value, dict) else None


async def _error_body(response: httpx.Response) -> dict[str, Any]:
    data = bytearray()
    async for chunk in response.aiter_bytes():
        if len(data) + len(chunk) > _ERROR_BODY_LIMIT:
            return {}
        data.extend(chunk)
    return _json_object(data) or {}


def _nested_values(root: Any) -> Iterator[str]:
    pending = [root]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value)  # JSON 转义键与值都可能是用户提供的。
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str) or type(value) in (int, float):
            yield str(value)


def _request_values(messages: list[dict[str, Any]]) -> Iterator[str]:
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for part in content:
                yield part["text"] if part["type"] == "text" else part["image_url"]["url"]
        if "reasoning_content" in message:
            yield message["reasoning_content"]
        for call in message.get("tool_calls", ()):
            arguments = call["function"]["arguments"]
            yield arguments
            yield from _nested_values(_json_value(arguments))


def _bounded_sources(messages: list[dict[str, Any]]) -> list[str] | None:
    remaining = _ECHO_SCAN_LIMIT
    sources = []
    for count, source in enumerate(_request_values(messages), start=1):
        remaining -= len(source)
        if remaining < 0 or count > _ECHO_VALUE_LIMIT or len(sources) >= _ECHO_VALUE_LIMIT:
            # 大历史、图片或超长工具参数不做任意量的扫描。此时不放行未经核查的厂商字段。
            return None
        source = source.strip()
        if not source:
            continue
        sources.append(source)
        # 短独立行也按完整值筛查，不因所在字段是长多行文本而忽略。
        lines = source.splitlines()
        if len(lines) > 1:
            if len(sources) + len(lines) > _ECHO_VALUE_LIMIT:
                return None
            sources.extend(line.strip() for line in lines if line.strip())
    return sources


def _json_views(message: str) -> list[str]:
    views = [message]
    # 外层响应已解码；错误消息也可能引用另一层 JSON 内的转义字符串。
    for _ in range(2):
        decoded = _JSON_ESCAPE.sub(lambda token: json.loads('"' + token[0] + '"'), views[-1])
        if decoded == views[-1]:
            break
        views.append(decoded)
    return views


def _windows(value: str, width: int, excluded: tuple[tuple[int, int], ...] = ()) -> Iterator[str]:
    start = 0
    for low, high in excluded:
        if high - low >= width:
            yield from (value[index:index + width] for index in range(start, low))
            start = high - width + 1  # 跨到型号外的窗口仍参与检查。
    yield from (value[index:index + width] for index in range(start, len(value) - width + 1))


def _overlaps(source: str, by_width: dict[int, set[str]], excluded: tuple[tuple[int, int], ...] = ()) -> bool:
    # 固定数量的长度，按哈希集合匹配；不在请求全文中逐个搜索错误片段。
    return any(piece in pieces for width, pieces in by_width.items() if pieces
               for piece in _windows(source, width, excluded))


def _bounded_value(value: str) -> re.Pattern[str]:
    # 句末标点可以分隔字段；连接两个标识符的点、短横线等不能分隔。
    return re.compile(r"(?<!\w)(?<!\w[./:@-])" + re.escape(value) + r"(?!\w|[./:@-]\w)")


def _spans(value: str, pattern: re.Pattern[str]) -> tuple[tuple[int, int], ...]:
    return tuple(match.span() for match in pattern.finditer(value))


def _full_echo(source: str, views: list[str], literals: set[str], *, token: bool) -> bool:
    if source in literals:
        return True
    if token:
        # 结构化字段沿用原来的匹配策略，不因文案的边界规则而改变。
        return len(source) >= _COMPLETE_FIELD_LENGTH and any(source in view for view in views)
    if len(source) <= 1 or not any(source in view for view in views) or not any(char.isalnum() for char in source):
        return False
    whole = _bounded_value(source)
    return any(whole.search(view) for view in views)


def _echo_evidence(views: list[str], *, token: bool) -> tuple[set[str], dict[int, set[str]]]:
    literals = set()
    fragments: dict[int, set[str]] = {4: set(), _FRAGMENT_LENGTH: set()}
    for view in views:
        literals.add(view.strip())
        literals.update(match[0].strip() for match in _INTRODUCED_VALUE.finditer(view))
        quoted = [next(part for part in match.groups() if part is not None).strip()
                  for match in _QUOTED_VALUE.finditer(view)]
        literals.update(quoted)
        for excerpt in (*quoted, view) if token else quoted:
            fragments[_FRAGMENT_LENGTH].update(piece for piece in _windows(excerpt, _FRAGMENT_LENGTH)
                                               if any(char.isalnum() for char in piece))
        # 无引用边界时，不把普通英文词组的任意子串当作回显。优先带数字、标识符分隔符的片段，
        # 或没有词间空格的连续非 ASCII 字母/数字。短数字不会成为独立的片段候选。
        for match in _IDENTIFIER.finditer(view):
            fragments[_FRAGMENT_LENGTH].update(piece for piece in _windows(match[0], _FRAGMENT_LENGTH)
                                               if _IDENTIFIER_SIGNAL.search(piece))
        fragments[4].update(piece for piece in _windows(view, 4)
                            if all(not char.isascii() and char.isalnum() for char in piece))
        for match in _MARKED_TRUNCATION.finditer(view):
            literals.add(match[1])
            width = min(len(match[1]), _FRAGMENT_LENGTH)
            fragments.setdefault(width, set()).add(match[1][:width])
    return literals, fragments


def _without_echo(value: object, api_key: str, sources: list[str] | None, *, token: bool = False,
                  public_model: str | None = None) -> str | None:
    """对厂商文案、字符串错误码和请求 ID 使用同一失败路径筛查。"""
    if (not isinstance(value, str) or (token and not _LOG_TOKEN.fullmatch(value)) or sources is None
            or len(value) > _LOG_MESSAGE_LIMIT or len(api_key) > _ECHO_SCAN_LIMIT
            or len(value) * len(sources) > _ECHO_COMPARISON_LIMIT):
        return None
    views = _json_views(value)
    literals, fragments = _echo_evidence(views, token=token)
    if api_key:
        key_pieces = {piece for view in views for piece in _windows(view, _KEY_FRAGMENT_LENGTH)}
        if (any(api_key in view for view in views) or _overlaps(api_key, {_KEY_FRAGMENT_LENGTH: key_pieces})
                or any(len(literal) >= 6 and literal in api_key for literal in literals)):
            return None
    model_pattern = None
    model_only: dict[int, set[str]] = {}
    if not token and public_model and any(public_model in view for view in views):
        model_pattern = _bounded_value(public_model)
        model_spans = [(view, _spans(view, model_pattern)) for view in views]
        if any(spans for _, spans in model_spans):
            # 只有响应和请求两边都在完整、独立的型号内时，才忽略这一片段的重合。
            for width, pieces in fragments.items():
                ordinary = {piece for view, spans in model_spans for piece in _windows(view, width, spans)
                            if piece in pieces} if pieces else set()
                model_only[width] = pieces - ordinary
                fragments[width] = ordinary
    if not any(model_only.values()):
        model_pattern = None
    for source in sources:
        if source != public_model and _full_echo(source, views, literals, token=token):
            return None
        if _overlaps(source, fragments):
            return None
        if model_pattern is not None and _overlaps(source, model_only, _spans(source, model_pattern)):
            return None
    return value


class ChatOutput:
    """保留首段原始顺序，不把每个 SSE 增量保存为独立 Part。"""

    def __init__(self) -> None:
        self._parts: list[tuple[str, list[str]] | int] = []
        self._calls: set[int] = set()

    def text(self, kind: str, text: str) -> None:
        last = self._parts[-1] if self._parts else None
        if isinstance(last, tuple) and last[0] == kind:
            last[1].append(text)
        else:
            self._parts.append((kind, [text]))

    def call(self, index: int) -> None:
        if index not in self._calls:
            self._calls.add(index)
            self._parts.append(index)

    def complete(self, key: bytes, calls: dict[int, ToolCall]) -> AssistantPartsReady | None:
        has_reasoning = any(isinstance(part, tuple) and part[0] == "reasoning" for part in self._parts)
        if not has_reasoning and not calls:
            return None
        parts: list[Part] = []
        try:
            for is_call, run in groupby(self._parts, key=lambda part: isinstance(part, int)):
                if is_call:
                    # 同一次工具列表按协议 index 排序，不按流中各调用最先收到片段的顺序。
                    parts.extend(calls[index] for index in sorted(part for part in run if isinstance(part, int)))
                else:
                    for part in run:
                        assert isinstance(part, tuple)
                        kind, fragments = part
                        text = "".join(fragments)
                        if kind == "reasoning":
                            parts.append(Reasoning(PrivateData(key, text.encode("utf-8"))))
                        elif text:
                            parts.append(Text(text))
        except UnicodeError:
            raise ModelError("invalid_sse_event") from None
        if not calls and not any(isinstance(part, Text) or part.private.payload for part in parts):
            return None
        return AssistantPartsReady(tuple(parts))


def chat_reasoning(message: Message, key: bytes) -> str | None:
    payloads = [part.private.payload for part in message.parts
                if isinstance(part, Reasoning) and part.private.compatibility_key == key]
    if not payloads:
        return None
    try:
        return b"".join(payloads).decode("utf-8")
    except UnicodeError:
        return None


def _native_calls(calls: tuple[ToolCall, ...], key: bytes) -> dict[int, ToolCall] | None:
    native: dict[int, ToolCall] = {}
    for call in calls:
        data = call.private
        if data is None or data.compatibility_key != key or not data.payload.isdigit():
            return None
        try:
            index = int(data.payload)
        except ValueError:
            return None
        if index in native:
            return None
        native[index] = call
    return native


def _ordered_calls(calls: tuple[ToolCall, ...], key: bytes) -> list[tuple[int, ToolCall]]:
    if native := _native_calls(calls, key):
        return sorted(native.items())
    # 任一 index 缺失、无效或冲突时，不混排原生 index 和推测位置。
    return list(enumerate(calls))


def chat_messages(messages: Sequence[Message], reasoning_for: Callable[[Message], str | None], *,
                  key: bytes, include_tool_index: bool = False, instructions: str | None = None) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    if instructions is not None:
        projected.append({"role": "system", "content": instructions})
    for message in messages:
        assistant = message.kind == "assistant"
        calls = _ordered_calls(message.tool_calls, key)
        reasoning = reasoning_for(message) if assistant else None
        content: Any = input_content(message)
        if assistant and not (content or calls or reasoning):
            continue
        if not isinstance(content, str):
            content = [{"type": "image_url", "image_url": {"url": part.url}} if isinstance(part, Image)
                       else {"type": "text", "text": part.text} for part in content]
        item = {"role": "assistant" if assistant else "user", "content": content}
        if calls:
            item["content"] = content or None
            item["tool_calls"] = [{"type": "function", "id": call.id,
                                   **({"index": index} if include_tool_index else {}),
                                   "function": {"name": call.name, "arguments": call.arguments}}
                                  for index, call in calls]
        if reasoning is not None:
            item["reasoning_content"] = reasoning
        projected.append(item)
        projected.extend({"role": "tool", "content": call.result, "tool_call_id": call.id}
                         for _, call in calls if call.result is not None)
    return projected


def _has_private_fields(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    calls = item.get("tool_calls")
    return "reasoning_content" in item or (isinstance(calls, list) and any(
        isinstance(call, dict) and "index" in call for call in calls))


def _has_private_response_fields(item: dict[str, Any]) -> bool:
    if _has_private_fields(item):
        return True
    choices = item.get("choices")
    return isinstance(choices, list) and any(
        _has_private_fields(choice.get(field)) for choice in choices if isinstance(choice, dict)
        for field in ("delta", "message"))


class ChatCompletions(ABC):
    adapter: AdapterID
    include_tool_index = False

    def __init__(self, client: httpx.AsyncClient, *, model: str, api_key: str, endpoint: str,
                 replay_key: bytes) -> None:
        self._client = client
        self._model = model
        self._api_key = api_key
        self._endpoint = endpoint
        self._replay_key = replay_key
        self._responses = ResponseStreams()

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def _reasoning(self, message: Message) -> str | None:
        return chat_reasoning(message, self._replay_key)

    def history_replay_bytes(self, message: Message, tools: Sequence[ToolSpec], *, thinking: bool) -> int:
        reasoning = self._reasoning(message)
        return len(reasoning.encode("utf-8")) if reasoning is not None else 0

    def _request(self, messages: Sequence[Message], tools: Sequence[ToolSpec], *,
                 instructions: str | None = None) -> dict[str, Any]:
        projected = chat_messages(messages, self._reasoning, key=self._replay_key,
                                  include_tool_index=self.include_tool_index, instructions=instructions)
        result = {"model": self._model, "stream": True, "stream_options": {"include_usage": True},
                  "messages": projected}
        if tools:
            result["tools"] = [{"type": "function", "function": {"name": spec.name,
                                "description": spec.description, "parameters": spec.parameters}} for spec in tools]
        return result

    @abstractmethod
    def _payload(self, messages: Sequence[Message], tools: Sequence[ToolSpec], thinking: bool, *,
                 instructions: str | None = None) -> dict[str, Any]: ...

    @abstractmethod
    def _validate_finish(self, finish: str | None, *, has_tools: bool) -> None: ...

    @abstractmethod
    def _error_fields(self, item: dict[str, Any]) -> tuple[int | str | None, str | None] | None: ...

    def _request_ids(self, response: httpx.Response, item: dict[str, Any]) -> tuple[object, ...]:
        return (response.headers.get("x-request-id"),)

    def _check_stopping(self) -> None:
        if self._responses.stopping():
            raise asyncio.CancelledError

    def _failure(self, response: httpx.Response, item: dict[str, Any], messages: list[dict[str, Any]],
                 reason: str, *, private: bool) -> VendorResponseError:
        if private:
            # 请求回放或本轮已收到的私有字段可被变形回显，不尝试猜测远端字段是否安全。
            return VendorResponseError(reason, adapter=self.adapter, http_status=response.status_code,
                                       request_id=None, code=None, message=None)
        code, message = self._error_fields(item) or (None, None)
        sources = _bounded_sources(messages)
        request_id = None
        for candidate in self._request_ids(response, item):
            request_id = _without_echo(candidate, self._api_key, sources, token=True)
            if request_id is not None:
                break
        if type(code) is int:
            if not -(2**63) <= code < 2**63 or _without_echo(str(code), self._api_key, sources, token=True) is None:
                code = None
        else:
            code = _without_echo(code, self._api_key, sources, token=True)
        # 与 _request() 发出的 model 一致，只用于判定 MiniMax 文案的回显，不改写日志。
        public_model = self._model if self.adapter == "minimax" else None
        message = _without_echo(message, self._api_key, sources, public_model=public_model)
        return VendorResponseError(reason, adapter=self.adapter, http_status=response.status_code,
                                   request_id=request_id, code=code, message=message)

    async def _parse_stream(self, response: httpx.Response, messages: list[dict[str, Any]], *,
                            private: bool) -> AsyncIterator[ModelEvent]:
        fragments: dict[int, _ToolParts] = {}
        output = ChatOutput()
        finish: str | None = None
        usage: InputUsage | None = None
        async for kind, data in _sse(response.aiter_lines()):
            self._check_stopping()
            if kind != "error" and data == "[DONE]":
                self._validate_finish(finish, has_tools=bool(fragments))
                calls = {index: part.complete(index, self._replay_key) for index, part in sorted(fragments.items())}
                if len({call.id for call in calls.values()}) != len(calls):
                    raise ModelError("duplicate_tool_id")
                if ready := output.complete(self._replay_key, calls):
                    yield ready
                for call in calls.values():
                    yield ToolReady(call)
                if usage is not None:
                    yield usage
                return

            item = _json_object(data) or {}
            # 同一帧也可能同时携带错误和私有增量；先收紧诊断边界。
            private = private or _has_private_response_fields(item)
            if self._error_fields(item) is not None or kind == "error" or "error" in item:
                raise self._failure(response, item, messages, "vendor_error", private=private)
            choices = item.get("choices")
            if not isinstance(choices, list) or len(choices) > 1:
                raise self._failure(response, item, messages, "invalid_sse_event", private=private)
            count = item.get("usage")
            if isinstance(count, dict):
                tokens = count.get("prompt_tokens")
                if type(tokens) is int and 0 <= tokens < 2**63:
                    usage = InputUsage(tokens)
            if not choices:
                continue  # 独立的用量帧可以出现在 finish_reason 之后。
            choice = choices[0]
            if finish is not None or not isinstance(choice, dict) or choice.get("index", 0) != 0:
                raise ModelError("invalid_sse_event")
            delta = choice.get("delta")
            if delta is None:
                delta = {}
            if not isinstance(delta, dict):
                raise ModelError("invalid_sse_event")
            thought, text = delta.get("reasoning_content"), delta.get("content")
            if ((thought is not None and not isinstance(thought, str)) or
                    (text is not None and not isinstance(text, str))):
                raise ModelError("invalid_sse_event")
            if thought is not None:
                output.text("reasoning", thought)
            if text:
                output.text("text", text)
            calls_delta = delta.get("tool_calls")
            if calls_delta is not None:
                if not isinstance(calls_delta, list):
                    raise ModelError("invalid_tool_delta")
                for call in calls_delta:
                    index = call.get("index") if isinstance(call, dict) else None
                    if type(index) is not int or index < 0:
                        raise ModelError("invalid_tool_delta")
                    fragments.setdefault(index, _ToolParts()).append(call)
                    output.call(index)
            reason = choice.get("finish_reason")
            if reason is not None:
                self._validate_finish(reason, has_tools=bool(fragments))
                finish = reason
            if text:
                yield TextDelta(text)
        self._check_stopping()
        raise ModelError("truncated_stream")

    async def stream(self, messages: Sequence[Message], tools: Sequence[ToolSpec], *,
                     thinking: bool, instructions: str | None = None) -> AsyncGenerator[ModelEvent, None]:
        diagnostics = adapter_started(self.adapter)
        payload = self._payload(messages, tools, thinking, instructions=instructions)
        projected = payload["messages"]
        private = any("reasoning_content" in item for item in projected)
        if not private and self.include_tool_index:
            # 与出站排序共用兼容性检查；仅回传原生 index 才携带私有数据，本地重新编号不算。
            private = any(_native_calls(message.tool_calls, self._replay_key) for message in messages)
        body = request_body(payload)
        if diagnostics is not None:
            diagnostics.request_ready(len(body))
        try:
            request = self._client.stream("POST", self._endpoint, content=body, follow_redirects=False,
                                          headers={"Accept": "text/event-stream", "Content-Type": "application/json",
                                                   "Authorization": f"Bearer {self._api_key}"})
            async with self._responses.open(request, background_tail=True) as (response, allow_tail):
                self._check_stopping()
                if diagnostics is not None:
                    diagnostics.response_received(response.status_code)
                content_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
                if not response.is_success or content_type == "application/json" or content_type.endswith("+json"):
                    try:
                        item = await _error_body(response)
                    except httpx.RequestError:
                        if response.is_success:
                            raise
                        item = {}  # 响应体不可读，但 HTTP 失败状态和请求 ID 仍可用。
                    self._check_stopping()
                    if not response.is_success:
                        reason = f"http_{response.status_code}"
                    else:
                        reason = "vendor_error" if self._error_fields(item) is not None else "invalid_sse_event"
                    raise self._failure(response, item, projected, reason,
                                        private=private or _has_private_response_fields(item))
                if diagnostics is not None:
                    diagnostics.stage = "sse_read"
                async with aclosing(self._parse_stream(response, projected, private=private)) as events:
                    async for event in events:
                        yield event
                allow_tail()
                if diagnostics is not None:
                    diagnostics.stage = "response_close"
                yield ModelClosing(complete=True)
        except VendorResponseError as error:
            if diagnostics is None:
                error.log_response()
            raise
        except httpx.RequestError as error:
            if diagnostics is not None:
                diagnostics.transport_failure = failure_category(error)
            raise ModelError("transport_error") from None

    def stop_stream(self, reader: asyncio.Task[object]) -> bool:
        return self._responses.stop(reader)

    async def aclose(self) -> None:
        await self._responses.aclose()
