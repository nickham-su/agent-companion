"""OpenAI Responses：无状态 SSE、原生 item 及已确认历史的回传。"""

from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import asdict
import hmac
import json
import re
from typing import Any

import httpx

from ..config import normalize_openai_base_url
from ..protocol import (MAX_REPLAY_BYTES, Image, InputUsage, Message, ModelError, ModelEvent, ProviderReplay,
                        ReplayItem, ReplayReady, ReplayScope, Text, TextDelta,
                        ToolCall, ToolReady, ToolSpec)
from ._request import JSON_ENCODER, request_body


_MAX_EVENT = 16 * 1024 * 1024
_MAX_STREAM = 64 * 1024 * 1024
_MAX_TEXT = 2 * 1024 * 1024
_MAX_ITEMS = 4096
_PHASES = (None, "commentary", "final_answer")


def _input_part(part: Text | Image) -> dict[str, Any]:
    if isinstance(part, Image):
        return {"type": "input_image", "image_url": part.url, "detail": "auto"}
    return {"type": "input_text", "text": part.text}


def _function(call: ToolCall, item_id: str | None = None) -> dict[str, Any]:
    item = {"type": "function_call", "call_id": call.id, "name": call.name, "arguments": call.arguments}
    if item_id is not None:
        item["id"] = item_id
    return item


def _common_items(message: Message) -> list[dict[str, Any]]:
    if message.role == "tool":
        return [{"type": "function_call_output", "call_id": message.tool_call_id, "output": message.content}]
    items = []
    if message.content or message.role != "assistant":
        content = message.content
        if not isinstance(content, str):
            content = [_input_part(part) for part in content]
        items.append({"role": message.role, "content": content})
    items.extend(_function(call) for call in message.tool_calls)
    return items


def _identifier(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 256 and value.isascii() and all(32 < ord(c) < 127 for c in value)


def _reject_constant(_: str) -> None:
    raise ValueError


def _replay_items(message: Message, replay: ProviderReplay) -> list[dict[str, Any]] | None:
    if not isinstance(message.content, str) or not replay.items or len(replay.items) > _MAX_ITEMS:
        return None
    items: list[dict[str, Any]] = []
    calls = iter(message.tool_calls)
    used_ids: set[str] = set()
    position = 0
    for item in replay.items:
        if not _identifier(item.item_id) or item.item_id in used_ids:
            return None
        used_ids.add(item.item_id)
        if item.kind == "reasoning":
            if (not isinstance(item.encrypted_content, str) or not item.encrypted_content
                    or item.call_id is not None or item.parts or item.phase is not None
                    or not all(isinstance(text, str) for text in item.summary)):
                return None
            items.append({"type": "reasoning", "id": item.item_id, "encrypted_content": item.encrypted_content,
                          "summary": [{"type": "summary_text", "text": text} for text in item.summary]})
        elif item.kind == "function_call":
            call = next(calls, None)
            if (call is None or not _identifier(item.call_id) or call.id != item.call_id or item.parts
                    or item.phase is not None or item.encrypted_content is not None or item.summary):
                return None
            items.append(_function(call, item.item_id))
        elif item.kind == "message":
            if (item.phase not in _PHASES or not item.parts or item.call_id is not None
                    or item.encrypted_content is not None or item.summary):
                return None
            start = position
            content: list[dict[str, Any]] = []
            for part in item.parts:
                if (len(part) != 2 or part[0] not in ("output_text", "refusal")
                        or type(part[1]) is not int or not 0 <= part[1] <= _MAX_TEXT):
                    return None
                kind, amount = part
                text = message.content[position:position + amount]
                position += amount
                content.append({"type": kind, "text": text, "annotations": []} if kind == "output_text"
                               else {"type": kind, "refusal": text})
            if position > _MAX_TEXT or position == start:
                return None
            if position <= len(message.content):
                native = {"type": "message", "id": item.item_id, "role": "assistant",
                          "status": "completed", "content": content}
                if item.phase is not None:
                    native["phase"] = item.phase
                items.append(native)
            elif start < len(message.content):
                # 仅确认前缀可见，不能用原 item ID 引用未交付的正文。
                items.append({"role": "assistant", "content": message.content[start:]})
        else:
            return None
    if next(calls, None) is not None or position < len(message.content):
        return None
    return items


def _tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [{"type": "function", "name": tool.name, "description": tool.description,
             "parameters": tool.parameters, "strict": False} for tool in tools]


def _read_output(raw: Any, scope: ReplayScope, done_reasoning: dict[int, tuple[str, str]]) -> tuple[
    list[tuple[tuple[int, int], str]], tuple[ToolCall, ...], ProviderReplay | None,
]:
    if not isinstance(raw, list) or len(raw) > _MAX_ITEMS:
        raise ModelError("invalid_sse_event")
    replay: list[ReplayItem] = []
    text_parts: list[tuple[tuple[int, int], str]] = []
    calls = []
    used_ids: set[str] = set()
    call_ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not _identifier(item.get("id")):
            raise ModelError("invalid_sse_event")
        kind, identity = item.get("type"), item["id"]
        if identity in used_ids or item.get("status") not in (None, "completed"):
            raise ModelError("invalid_sse_event")
        used_ids.add(identity)
        if kind == "reasoning":
            cipher = item.get("encrypted_content")
            # 只在匹配同一个已完成 reasoning item 时才使用 done 事件的密文。
            done = done_reasoning.get(index)
            if (cipher is None or cipher == "") and done is not None and done[0] == identity:
                cipher = done[1]
            if cipher is None or cipher == "":
                continue
            summary = item.get("summary", [])
            if (not isinstance(cipher, str) or not isinstance(summary, list) or not all(
                isinstance(part, dict) and part.get("type") == "summary_text" and isinstance(part.get("text"), str)
                for part in summary
            )):
                raise ModelError("invalid_sse_event")
            replay.append(ReplayItem(kind, identity, encrypted_content=cipher,
                                     summary=tuple(part["text"] for part in summary)))
        elif kind == "function_call":
            name, call_id, arguments = item.get("name"), item.get("call_id"), item.get("arguments")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) or not _identifier(call_id):
                raise ModelError("incomplete_tool_call")
            if call_id in call_ids:
                raise ModelError("duplicate_tool_id")
            try:
                if not isinstance(arguments, str) or not isinstance(json.loads(arguments, parse_constant=_reject_constant), dict):
                    raise ValueError
            except (ValueError, RecursionError):
                raise ModelError("invalid_tool_arguments") from None
            call_ids.add(call_id)
            replay.append(ReplayItem(kind, identity, call_id=call_id))
            calls.append(ToolCall(call_id, name, arguments))
        elif kind == "message":
            content, phase = item.get("content"), item.get("phase")
            if item.get("role") != "assistant" or not isinstance(content, list) or phase not in _PHASES:
                raise ModelError("invalid_sse_event")
            parts = []
            for content_index, part in enumerate(content):
                if not isinstance(part, dict):
                    raise ModelError("invalid_sse_event")
                part_type = part.get("type")
                if part_type not in ("output_text", "refusal"):
                    raise ModelError("responses_unsupported_output")
                text = part.get("text" if part_type == "output_text" else "refusal")
                if not isinstance(text, str):
                    raise ModelError("invalid_sse_event")
                text_parts.append(((index, content_index), text))
                parts.append((part_type, len(text)))
            if any(length for _, length in parts):
                replay.append(ReplayItem(kind, identity, phase=phase, parts=tuple(parts)))
        else:
            # 不接受可产生外部副作用的内建工具或尚未定义的 item。
            raise ModelError("responses_unsupported_output")
    size = 0
    try:
        for chunk in JSON_ENCODER.iterencode([asdict(item) for item in replay]):
            size += len(chunk.encode("utf-8"))
            if size > MAX_REPLAY_BYTES:
                raise ModelError("responses_output_limit")
    except UnicodeError:
        raise ModelError("invalid_sse_event") from None
    return text_parts, tuple(calls), ProviderReplay(scope, tuple(replay)) if replay else None


async def _sse_data(chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """原始字节限额在 JSON 和 HTTP 解压之前生效。"""
    line = bytearray()
    event: list[bytes] = []
    size = total = 0
    async for chunk in chunks:
        total += len(chunk)
        if total > _MAX_STREAM:
            raise ModelError("responses_output_limit")
        offset = 0
        while offset < len(chunk):
            newline = chunk.find(b"\n", offset)
            end = len(chunk) if newline < 0 else newline
            if len(line) + end - offset > _MAX_EVENT:
                raise ModelError("responses_event_limit")
            line.extend(memoryview(chunk)[offset:end])
            if newline < 0:
                break
            offset = end + 1
            finished = bytes(line).removesuffix(b"\r")
            line.clear()
            if not finished:
                if event:
                    try:
                        yield b"\n".join(event).decode("utf-8")
                    except UnicodeError:
                        raise ModelError("invalid_sse_event") from None
                    event.clear()
                    size = 0
            elif finished.startswith(b"data:"):
                payload = finished[5:].removeprefix(b" ")
                size += len(payload) + 1
                if size > _MAX_EVENT:
                    raise ModelError("responses_event_limit")
                event.append(payload)


async def _events(chunks: AsyncIterator[bytes], scope: ReplayScope) -> AsyncIterator[ModelEvent]:
    segments: dict[tuple[int, int], list[str]] = {}
    done_reasoning: dict[int, tuple[str, str]] = {}
    done_size = 0
    last_key = (-1, -1)
    emitted = 0
    async with aclosing(_sse_data(chunks)) as records:
        async for data in records:
            if data == "[DONE]":
                break  # 此标记不能代替 response.completed。
            try:
                event = json.loads(data)
                kind = event.get("type")
            except (ValueError, AttributeError, RecursionError):
                raise ModelError("invalid_sse_event") from None
            if not isinstance(kind, str):
                raise ModelError("invalid_sse_event")
            if kind in ("error", "response.failed", "response.incomplete", "response.cancelled"):
                raise ModelError("abnormal_stream_finish")
            if kind == "response.output_item.done":
                item, index = event.get("item"), event.get("output_index")
                if isinstance(item, dict) and item.get("type") == "reasoning":
                    cipher = item.get("encrypted_content")
                    if isinstance(cipher, str) and cipher:
                        identity = item.get("id")
                        if (not _identifier(identity) or type(index) is not int or not 0 <= index < _MAX_ITEMS
                                or index in done_reasoning or item.get("status") not in (None, "completed")):
                            raise ModelError("invalid_sse_event")
                        done_size += len(cipher)
                        if done_size > MAX_REPLAY_BYTES:
                            raise ModelError("responses_output_limit")
                        done_reasoning[index] = (identity, cipher)
            if kind in ("response.output_text.delta", "response.refusal.delta"):
                index, part = event.get("output_index"), event.get("content_index")
                delta = event.get("delta")
                if type(index) is not int or type(part) is not int or min(index, part) < 0 or not isinstance(delta, str):
                    raise ModelError("invalid_sse_event")
                key = (index, part)
                if key < last_key:
                    raise ModelError("invalid_sse_event")
                last_key = key
                if delta:
                    try:
                        delta.encode("utf-8")
                    except UnicodeError:
                        raise ModelError("invalid_sse_event") from None
                    emitted += len(delta)
                    if emitted > _MAX_TEXT or (len(segments) >= _MAX_ITEMS and key not in segments):
                        raise ModelError("responses_output_limit")
                    segments.setdefault(key, []).append(delta)
                    yield TextDelta(delta)
            if kind == "response.completed":
                response = event.get("response")
                if not isinstance(response, dict) or response.get("status") not in (None, "completed"):
                    raise ModelError("invalid_stream_finish")
                parts, calls, replay = _read_output(response.get("output"), scope, done_reasoning)
                if sum(len(text) for _, text in parts) > _MAX_TEXT:
                    raise ModelError("responses_output_limit")
                visible = "".join(text for _, text in parts)
                try:
                    visible.encode("utf-8")
                except UnicodeError:
                    raise ModelError("invalid_sse_event") from None
                seen = "".join(fragment for fragments in segments.values() for fragment in fragments)
                full_parts = dict(parts)
                if not visible.startswith(seen) or any(not full_parts.get(key, "").startswith("".join(fragments))
                                                      for key, fragments in segments.items()):
                    raise ModelError("invalid_stream_finish")
                if len(visible) > emitted:
                    yield TextDelta(visible[emitted:])
                if replay is not None:
                    yield ReplayReady(replay)
                # 先完整校验本响应所有 item，再派发任何工具。
                for call in calls:
                    yield ToolReady(call)
                usage = response.get("usage")
                if isinstance(usage, dict):
                    tokens = usage.get("input_tokens")
                    if type(tokens) is int and 0 <= tokens < 2**63:
                        yield InputUsage(tokens)
                return
    raise ModelError("truncated_stream")


class OpenAIResponses:
    def __init__(self, client: httpx.AsyncClient, *, model: str, api_key: str, replay_salt: bytes,
                 reasoning_effort: str, base_url: str | None = None) -> None:
        if not api_key or not api_key.isascii() or any(not 33 <= ord(char) <= 126 for char in api_key):
            raise ValueError("invalid OpenAI Responses API credential")
        if not isinstance(replay_salt, bytes) or len(replay_salt) != 32:
            raise ValueError("invalid OpenAI Responses replay salt")
        base_url = normalize_openai_base_url(base_url)
        # 固定用途标签及盐长度，URL 与模型用 JSON 分隔，避免拼接歧义。
        target = JSON_ENCODER.encode([base_url, model]).encode("utf-8")
        fingerprint = hmac.digest(api_key.encode("ascii"),
                                  b"agent-companion/openai-responses-replay\0" + replay_salt + target, "sha256")
        self._client = client
        self._endpoint = base_url + "/responses"
        self._authorization = f"Bearer {api_key}"
        self._model = model
        self.replay_scope = ReplayScope("openai_responses", fingerprint, reasoning_effort)

    def _payload(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        tools_data = _tools(tools)
        for message in messages:
            replay = message.replay
            projected = (_replay_items(message, replay) if message.role == "assistant" and replay is not None
                         and replay.scope == self.replay_scope else None)
            items.extend(projected if projected is not None else _common_items(message))
        payload = {"model": self._model, "input": items, "stream": True, "store": False,
                   "include": ["reasoning.encrypted_content"], "reasoning": {"effort": self.replay_scope.reasoning_effort}}
        if tools_data:
            payload["tools"] = tools_data
        return payload

    async def stream(self, messages: Sequence[Message], tools: Sequence[ToolSpec], *,
                     thinking: bool) -> AsyncGenerator[ModelEvent, None]:
        # thinking 是 DeepSeek 内部协议参数；Responses 只采用用户选定的 effort。
        body = request_body(self._payload(messages, tools))
        terminal: list[ModelEvent] = []
        try:
            async with self._client.stream("POST", self._endpoint, content=body, follow_redirects=False,
                                           auth=None, headers={
                "Authorization": self._authorization,
                "Accept": "text/event-stream", "Content-Type": "application/json", "Accept-Encoding": "identity",
            }) as response:
                if not response.is_success:
                    raise ModelError(f"http_{response.status_code}")
                if response.headers.get("Content-Encoding", "").strip().lower() not in ("", "identity"):
                    raise ModelError("responses_unsupported_encoding")
                if response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "text/event-stream":
                    raise ModelError("responses_unexpected_response")
                async with aclosing(_events(response.aiter_raw(), self.replay_scope)) as events:
                    async for event in events:
                        if isinstance(event, TextDelta):
                            yield event
                        else:
                            terminal.append(event)
        except httpx.RequestError:
            raise ModelError("transport_error") from None
        # HTTP 传输完成收尾后才提交工具和私有回放；此前仅允许文本交付。
        for event in terminal:
            yield event
