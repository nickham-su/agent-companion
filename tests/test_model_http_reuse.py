import asyncio
from contextlib import aclosing
import json
import unittest
from unittest.mock import patch

import httpcore
import httpx

from agent_companion.models import _http_stream
from agent_companion.models.deepseek import DeepSeek
from agent_companion.models.openai_responses import OpenAIResponses
from agent_companion.protocol import InputUsage, Message, ModelClosing, ModelError, TextDelta
from test_openai_responses import _answer, _completed


def _sse(*events):
    return b"".join(b"data: " + (event if isinstance(event, bytes) else json.dumps(event).encode()) + b"\n\n"
                    for event in events)


def _chunk(data):
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


def _response(adapter, *, valid=True):
    if adapter == "deepseek":
        return _sse({"choices": [{"index": 0, "delta": {"content": "ok"},
                                   "finish_reason": "stop" if valid else None}],
                     "usage": {"prompt_tokens": 3}}, b"[DONE]")
    return _sse(_completed(_answer("ok"), input_tokens=3) if valid else {"type": "response.incomplete"})


def _model(client, adapter):
    return (DeepSeek(client) if adapter == "deepseek" else
            OpenAIResponses(client, model="offline", api_key="fixture", reasoning_effort="low"))


async def _read_model(model, events):
    async with aclosing(model.stream([Message("user", "test")], (), thinking=False)) as stream:
        async for event in stream:
            events.append(event)
            if isinstance(event, ModelClosing) and event.complete:
                break  # 和 Runtime 一样，在 DeepSeek 的终态立即退出读流。


class _HTTP11:
    """真实 HTTPX/httpcore 连接池；TCP/TLS 由内存连接替身计数。"""

    def __init__(self, body, *, hold=False, tail=b"", tail_error=False):
        self.connects = self.tls = self.tail_reads = 0
        self.tail_started = asyncio.Event()
        self.response_released = asyncio.Queue()
        self.release = asyncio.Event()
        if not hold:
            self.release.set()
        self.tail_cancelled = asyncio.Event()
        head = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                + _chunk(body))
        fixture = self

        class Wire(httpcore.AsyncNetworkStream):
            async def write(self, buffer, timeout=None):
                if buffer.startswith(b"POST "):
                    self.reads = 0

            async def read(self, max_bytes, timeout=None):
                self.reads += 1
                if self.reads == 1:
                    return head
                fixture.tail_reads += 1
                fixture.tail_started.set()
                try:
                    await fixture.release.wait()
                except asyncio.CancelledError:
                    fixture.tail_cancelled.set()
                    raise
                if tail_error:
                    raise httpcore.ReadError("offline tail read error")
                return _chunk(tail) if self.reads == 2 and tail else b"0\r\n\r\n"

            async def start_tls(self, **kwargs):
                fixture.tls += 1
                return self

            async def aclose(self):
                pass

            def get_extra_info(self, info):
                return False if info == "is_readable" else None

        class Backend(httpcore.AsyncNetworkBackend):
            async def connect_tcp(self, **kwargs):
                fixture.connects += 1
                return Wire()

        async def trace(event, _info):
            if event == "http11.response_closed.complete":
                self.response_released.put_nowait(None)

        async def trace_request(request):
            request.extensions["trace"] = trace

        pool = httpcore.AsyncConnectionPool
        with patch.object(httpcore, "AsyncConnectionPool", side_effect=lambda **kw: pool(network_backend=Backend(), **kw)):
            transport = httpx.AsyncHTTPTransport(trust_env=False)
        self.client = httpx.AsyncClient(transport=transport, trust_env=False, event_hooks={"request": [trace_request]})


class ModelHTTPReuseTests(unittest.IsolatedAsyncioTestCase):
    async def test_clean_http_tail_returns_connection_to_pool_without_parsing_tail(self):
        for adapter in ("deepseek", "responses"):
            with self.subTest(adapter=adapter), patch.object(_http_stream, "_RESPONSE_TAIL_TIMEOUT", 5):
                # 结束块故意与 SSE 终态分开到达；额外的正文帧也不能当成模型事件。
                wire = _HTTP11(_response(adapter), hold=True, tail=_sse({"type": "response.failed"}))
                async with wire.client as client:
                    model = _model(client, adapter)
                    try:
                        events = []
                        first = asyncio.create_task(_read_model(model, events))
                        try:
                            await asyncio.wait_for(wire.tail_started.wait(), 2)
                            if adapter == "deepseek":
                                await asyncio.wait_for(first, 2)  # HTTP 末尾还没到，模型已完成。
                                self.assertIn(InputUsage(3), events)
                            else:
                                self.assertFalse(first.done())
                                self.assertNotIn(InputUsage(3), events)
                            self.assertEqual([event for event in events if isinstance(event, TextDelta)], [TextDelta("ok")])
                        finally:
                            wire.release.set()
                            await asyncio.wait_for(first, 2)
                        self.assertIn(InputUsage(3), events)
                        await asyncio.wait_for(wire.response_released.get(), 2)
                        await _read_model(model, [])
                        await asyncio.wait_for(wire.response_released.get(), 2)
                        self.assertEqual((wire.connects, wire.tls), (1, 1))
                    finally:
                        await model.aclose()

    async def test_tail_fallback_closes_without_invalidating_completed_response(self):
        for adapter in ("deepseek", "responses"):
            for mode in ("time", "bytes", "error"):
                with (self.subTest(adapter=adapter, mode=mode),
                      patch.object(_http_stream, "_RESPONSE_TAIL_TIMEOUT", .03),
                      patch.object(_http_stream, "_RESPONSE_TAIL_MAX_BYTES", 4)):
                    wire = _HTTP11(_response(adapter), hold=mode == "time", tail=b"too large", tail_error=mode == "error")
                    async with wire.client as client:
                        model = _model(client, adapter)
                        try:
                            for _ in range(2):
                                events = []
                                await asyncio.wait_for(_read_model(model, events), 2)
                                await asyncio.wait_for(wire.response_released.get(), 2)
                                self.assertIn(InputUsage(3), events)
                            self.assertEqual((wire.connects, wire.tls, wire.tail_reads), (2, 2, 2))
                            self.assertEqual(wire.tail_cancelled.is_set(), mode == "time")
                        finally:
                            await model.aclose()

    async def test_protocol_failure_does_not_read_tail(self):
        for adapter in ("deepseek", "responses"):
            with self.subTest(adapter=adapter):
                wire = _HTTP11(_response(adapter, valid=False), hold=True)
                async with wire.client as client:
                    model = _model(client, adapter)
                    try:
                        with self.assertRaises(ModelError):
                            await _read_model(model, [])
                        self.assertFalse(wire.tail_started.is_set())
                    finally:
                        await model.aclose()

    async def test_consumer_stops_before_model_terminal_without_reading_tail(self):
        for adapter in ("deepseek", "responses"):
            with self.subTest(adapter=adapter):
                wire = _HTTP11(_response(adapter), hold=True)
                async with wire.client as client:
                    model = _model(client, adapter)
                    try:
                        async with aclosing(model.stream([Message("user", "test")], (), thinking=False)) as stream:
                            self.assertEqual(await anext(stream), TextDelta("ok"))
                        await asyncio.wait_for(wire.response_released.get(), 2)
                        self.assertFalse(wire.tail_started.is_set())
                    finally:
                        await model.aclose()
