"""OpenAI Responses：无状态 SSE、原生 item 及已确认历史的回传。"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import aclosing
import json
import re
from typing import Any

import httpx

from ..config import normalize_openai_base_url
from ..model_diagnostics import adapter_started, failure_category
from ..protocol import (AssistantPartsReady, Image, InputUsage, Message, ModelClosing, ModelError, ModelEvent,
                        Part, PrivateData, Reasoning, Text, TextDelta, ToolCall, ToolReady, ToolSpec)
from ._http_stream import ResponseStreams
from ._request import JSON_ENCODER, input_content, make_replay_key, request_body


_ADAPTER_ID = "openai_responses"
_MAX_REPLAY_BYTES = 8 * 1024 * 1024
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


def _identifier(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 256 and value.isascii() and all(32 < ord(c) < 127 for c in value)


def _reject_constant(_: str) -> None:
    raise ValueError


def _private(key: bytes, **data: Any) -> PrivateData:
    try:
        payload = JSON_ENCODER.encode(data).encode("utf-8")
    except UnicodeError:
        raise ModelError("invalid_sse_event") from None
    return PrivateData(key, payload)


def _metadata(part: Text | ToolCall | Reasoning, key: bytes) -> dict[str, Any] | None:
    data = part.private
    if data is None or data.compatibility_key != key or len(data.payload) > _MAX_REPLAY_BYTES:
        return None
    try:
        item = json.loads(data.payload.decode("utf-8"))
    except (ValueError, RecursionError):
        return None
    if not isinstance(item, dict) or not _identifier(item.get("id")):
        return None
    return item


def _native_text(text: str, item: dict[str, Any]) -> dict[str, Any] | None:
    phase, shape = item.get("phase"), item.get("parts")
    if phase not in _PHASES or not isinstance(shape, list) or not shape:
        return None
    content = []
    offset = 0
    for entry in shape:
        if (not isinstance(entry, list) or len(entry) != 2 or entry[0] not in ("output_text", "refusal")
                or type(entry[1]) is not int or not 0 <= entry[1] <= len(text) - offset):
            return None
        kind, length = entry
        part = text[offset:offset + length]
        offset += length
        # 展示引用不在原生回放白名单内，不保存或回传响应中的 annotations。
        content.append({"type": kind, "text": part, "annotations": []} if kind == "output_text"
                       else {"type": kind, "refusal": part})
    if offset != len(text):
        return None
    native = {"type": "message", "id": item["id"], "role": "assistant", "status": "completed", "content": content}
    if phase is not None:
        native["phase"] = phase
    return native


def _native_reasoning(item: dict[str, Any]) -> dict[str, Any] | None:
    cipher, summary = item.get("encrypted_content"), item.get("summary")
    if not isinstance(cipher, str) or not cipher or not isinstance(summary, list):
        return None
    try:
        for text in (cipher, *summary):
            if not isinstance(text, str):
                return None
            text.encode("utf-8")
    except UnicodeError:
        return None
    return {"type": "reasoning", "id": item["id"], "encrypted_content": cipher,
            "summary": [{"type": "summary_text", "text": text} for text in summary]}


def _response_items(message: Message, key: bytes | None) -> list[dict[str, Any]]:
    if message.kind != "assistant":
        body = input_content(message)
        return [{"role": "user", "content": body if isinstance(body, str) else [_input_part(p) for p in body]}]
    items: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    ids: set[str] = set()
    for part in message.parts:
        if isinstance(part, Image):
            continue  # 模型响应目前不产生图像历史。
        native = _metadata(part, key) if key is not None else None
        if native is not None and native["id"] in ids:
            native = None
        projected: dict[str, Any] | None = None
        if isinstance(part, Reasoning):
            if native is not None:
                projected = _native_reasoning(native)
        elif isinstance(part, Text):
            if native is not None:
                projected = _native_text(part.text, native)
            if projected is None and part.text:
                projected = {"role": "assistant", "content": part.text}
        else:
            if native is not None and native.get("call_id") != part.id:
                native = None
            projected = _function(part, native["id"] if native is not None else None)
            if part.result is not None:
                output: str | list[dict[str, Any]] = part.result
                if part.result_image is not None:
                    output = [_input_part(Text(part.result)), _input_part(part.result_image)]
                outputs.append({"type": "function_call_output", "call_id": part.id, "output": output})
        if projected is not None:
            if "id" in projected:
                ids.add(projected["id"])
            items.append(projected)
    return items + outputs


def _tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [{"type": "function", "name": tool.name, "description": tool.description,
             "parameters": tool.parameters, "strict": False} for tool in tools]


def _read_output(raw: Any, replay_key: bytes, done_reasoning: dict[int, tuple[str, str]]) -> tuple[
    list[tuple[tuple[int, int], str]], tuple[ToolCall, ...], tuple[Part, ...],
]:
    if not isinstance(raw, list) or len(raw) > _MAX_ITEMS:
        raise ModelError("invalid_sse_event")
    ordered: list[Part] = []
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
            ordered.append(Reasoning(_private(replay_key, id=identity, encrypted_content=cipher,
                                              summary=[part["text"] for part in summary])))
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
            call = ToolCall(call_id, name, arguments, private=_private(replay_key, id=identity, call_id=call_id))
            ordered.append(call)
            calls.append(call)
        elif kind == "message":
            content, phase = item.get("content"), item.get("phase")
            if item.get("role") != "assistant" or not isinstance(content, list) or phase not in _PHASES:
                raise ModelError("invalid_sse_event")
            parts = []
            visible = []
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
                visible.append(text)
            if any(visible):
                # 一个原生输出条目内的正文，共用其 ID 和 content 片段形状，不记录正文副本。
                ordered.append(Text("".join(visible), _private(replay_key, id=identity, phase=phase, parts=parts)))
        else:
            # 不接受可产生外部副作用的内建工具或尚未定义的 item。
            raise ModelError("responses_unsupported_output")
    if sum(len(part.private.payload) for part in ordered if not isinstance(part, Image)
           and part.private is not None) > _MAX_REPLAY_BYTES:
        raise ModelError("responses_output_limit")
    return text_parts, tuple(calls), tuple(ordered)


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


async def _events(chunks: AsyncIterator[bytes], replay_key: bytes) -> AsyncIterator[ModelEvent]:
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
                        if done_size > _MAX_REPLAY_BYTES:
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
                parts, calls, ordered = _read_output(response.get("output"), replay_key, done_reasoning)
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
                if ordered:
                    yield AssistantPartsReady(ordered)
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
    def __init__(self, client: httpx.AsyncClient, *, model: str, api_key: str,
                 reasoning_effort: str, base_url: str | None = None) -> None:
        if not api_key or not api_key.isascii() or any(not 33 <= ord(char) <= 126 for char in api_key):
            raise ValueError("invalid OpenAI Responses API credential")
        base_url = normalize_openai_base_url(base_url)
        self._client = client
        self._responses = ResponseStreams()
        self._endpoint = base_url + "/responses"
        self._authorization = f"Bearer {api_key}"
        self._model = model
        self._effort = reasoning_effort
        self._replay_key = make_replay_key(api_key.encode("ascii"), _ADAPTER_ID, model, base_url)

    def history_replay_bytes(self, message: Message, tools: Sequence[ToolSpec], *, thinking: bool) -> int:
        if message.kind != "assistant" or not any(
                isinstance(part, (Text, ToolCall, Reasoning)) and part.private is not None
                and part.private.compatibility_key == self._replay_key for part in message.parts):
            return 0
        native = _response_items(message, self._replay_key)
        common = _response_items(message, None)
        if native == common:
            return 0
        # 正文和工具调用/结果已在基础预算中，只补计实际出站的原生增量。
        return len(request_body({"input": native})) - len(request_body({"input": common}))

    def _payload(self, messages: Sequence[Message], tools: Sequence[ToolSpec], *,
                 instructions: str | None = None) -> dict[str, Any]:
        items = [item for message in messages for item in _response_items(message, self._replay_key)]
        tools_data = _tools(tools)
        payload = {"model": self._model, "input": items, "stream": True, "store": False,
                   "include": ["reasoning.encrypted_content"], "reasoning": {"effort": self._effort}}
        if instructions is not None:
            payload["instructions"] = instructions
        if tools_data:
            payload["tools"] = tools_data
        return payload

    async def stream(self, messages: Sequence[Message], tools: Sequence[ToolSpec], *,
                     thinking: bool, instructions: str | None = None) -> AsyncGenerator[ModelEvent, None]:
        # thinking 是 DeepSeek 内部协议参数；Responses 只采用用户选定的 effort。
        diagnostics = adapter_started(_ADAPTER_ID)
        body = request_body(self._payload(messages, tools, instructions=instructions))
        if diagnostics is not None:
            diagnostics.request_ready(len(body))
        terminal: list[ModelEvent] = []
        try:
            request = self._client.stream("POST", self._endpoint, content=body, follow_redirects=False,
                                          auth=None, headers={
                "Authorization": self._authorization,
                "Accept": "text/event-stream", "Content-Type": "application/json", "Accept-Encoding": "identity",
            })
            async with self._responses.open(request) as (response, allow_tail):
                if diagnostics is not None:
                    diagnostics.response_received(response.status_code)
                if not response.is_success:
                    raise ModelError(f"http_{response.status_code}")
                if response.headers.get("Content-Encoding", "").strip().lower() not in ("", "identity"):
                    raise ModelError("responses_unsupported_encoding")
                if response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "text/event-stream":
                    raise ModelError("responses_unexpected_response")
                if diagnostics is not None:
                    diagnostics.stage = "sse_read"
                async with aclosing(_events(response.aiter_raw(), self._replay_key)) as events:
                    async for event in events:
                        if isinstance(event, TextDelta):
                            yield event
                        else:
                            terminal.append(event)
                if diagnostics is not None:
                    diagnostics.stage = "response_close"
                yield ModelClosing(complete=False)
                allow_tail()
        except httpx.RequestError as error:
            if diagnostics is not None:
                diagnostics.transport_failure = failure_category(error)
            raise ModelError("transport_error") from None
        # HTTP 传输完成收尾后才提交工具和私有回放；此前仅允许文本交付。
        if diagnostics is not None:
            diagnostics.stage = "sse_read"
        for event in terminal:
            yield event

    def stop_stream(self, reader: asyncio.Task[object]) -> bool:
        return self._responses.stop(reader)

    async def aclose(self) -> None:
        await self._responses.aclose()
