import asyncio
from contextlib import closing
import json
import unittest

import httpx

from agent_companion.config import RuntimeSettings
from agent_companion.models.deepseek import DeepSeek, build_request, parse_stream
from agent_companion.protocol import (DeliveryStatus, Image, InputUsage, Message, ModelError, ReasoningDelta, Receipt, Text,
                                      TextDelta, ToolCall, ToolReady, ToolSpec)
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.storage.sqlite import History


SAMPLE_IMAGE = Image("data:image/png;base64,iVBORw0KGgo=")
SPEC = ToolSpec("test", "模拟工具", {"type": "object", "properties": {"size": {"type": "integer"}}})


def chunk(delta=None, finish=None):
    return json.dumps({"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]},
                      ensure_ascii=False)


def call(index=0, **values):
    return {"tool_calls": [{"index": index, **values}]}


async def lines(*events):
    yield ": keep-alive"
    for event in events:
        yield "data: " + event
        yield ""


class Bytes(httpx.AsyncByteStream):
    def __init__(self, content):
        self.content = content

    async def __aiter__(self):
        for index in range(0, len(self.content), 11):
            yield self.content[index:index + 11]


class DeepSeekTest(unittest.IsolatedAsyncioTestCase):
    async def test_large_request_reaches_transport_as_compact_utf8_json(self):
        image = Image("data:image/png;base64," + "A" * (40 * 1024 * 1024))
        message = Message("user", (Text('汉字\n"'), image))

        def endpoint(request):
            self.assertGreater(len(request.content), 40 * 1024 * 1024)
            self.assertEqual(request.headers["Content-Type"], "application/json")
            payload = json.loads(request.content)
            self.assertEqual(payload["messages"][0]["content"][0], {"type": "text", "text": '汉字\n"'})
            self.assertEqual(payload["messages"][0]["content"][1]["image_url"]["url"], image.url)
            self.assertEqual(payload["stream_options"], {"include_usage": True})
            frames = [chunk({"content": "模拟"}), chunk(finish="stop"), "[DONE]"]
            return httpx.Response(200, text="".join(f"data: {frame}\n\n" for frame in frames))

        async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
            events = [event async for event in DeepSeek(client).stream([message], (), thinking=False)]
        self.assertEqual(events, [TextDelta("模拟")])

    async def test_usage_from_final_content_or_empty_choice_chunk(self):
        final = json.loads(chunk(finish="stop"))
        usage = {"prompt_tokens": 21, "completion_tokens": 300, "total_tokens": 321}
        for chunks in ([json.dumps({**final, "usage": usage})],
                       [json.dumps(final), json.dumps({"choices": [], "usage": usage})]):
            with self.subTest(chunks=chunks):
                self.assertEqual([event async for event in parse_stream(lines(*chunks, "[DONE]"))], [InputUsage(21)])
        for invalid in (None, True, -1, 1.5, "21", 2**63):
            frame = json.dumps({**final, "usage": {"prompt_tokens": invalid, "total_tokens": 400}})
            self.assertEqual([event async for event in parse_stream(lines(frame, "[DONE]"))], [])

    async def test_model_response_diagnostics_keep_only_safe_status(self):
        statuses = iter((401, 429))

        def endpoint(request):
            return httpx.Response(next(statuses), text="synthetic upstream body")

        with closing(History(":memory:")) as history:
            async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
                runtime = Runtime("diagnostics", history, DeepSeek(client), None)
                with self.assertLogs("agent_companion.runtime.engine", level="WARNING") as logs:
                    for i in range(2):
                        self.assertEqual(await runtime.run_internal(str(i), "synthetic user content"), "model_failed")
                self.assertEqual([row.model_error for row in history.outcomes("diagnostics")], ["http_401", "http_429"])
                lines = " ".join(logs.output)
                self.assertIn("http_401", lines)
                self.assertIn("http_429", lines)
                self.assertNotIn("synthetic", lines)
                self.assertEqual(ModelError("unrecognized adapter details").diagnostic, "model_failed")

    async def test_actual_prompt_usage_triggers_summary_with_original_image(self):
        requests = []
        old = Message("user", (Text("历史图文"), Image("data:image/png;base64," + "A" * 6000)))
        settings = RuntimeSettings(context_budget=300, tail_budget=60)

        def endpoint(request):
            content, tokens = [("初次回复", 300), ("简洁概要", 1000), ("新的回复", 1)][len(requests)]
            requests.append(json.loads(request.content))
            finish = json.dumps({**json.loads(chunk(finish="stop")), "usage": {"prompt_tokens": tokens}})
            frames = [chunk({"content": content}), finish, "[DONE]"]
            return httpx.Response(200, text="".join(f"data: {frame}\n\n" for frame in frames))

        with closing(History(":memory:")) as history:
            history.append("child", [old])
            async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
                runtime = Runtime("child", history, DeepSeek(client), None, settings=settings)
                self.assertIsNone(await runtime.run_internal("first", "首次请求"))
                self.assertIsNone(history.summary("child"))
                self.assertEqual(history.last_input_tokens("child"), 300)
                self.assertIsNone(await runtime.run_internal("next", "新的文字"))
            compressed = requests[1]["messages"]
            self.assertEqual(compressed[0], requests[0]["messages"][0])
            self.assertNotIn("新的文字", repr(compressed))
            self.assertNotIn("image_url", repr(requests[2]))
            self.assertEqual(requests[2]["messages"][-1]["content"], "新的文字")
            self.assertEqual(history.read_after("child")[0].message, old)
            self.assertEqual([outcome.input_tokens for outcome in history.outcomes("child")], [300, 1])

    async def test_image_tool_and_reasoning_round_trip_over_sse(self):
        pending = []
        handler_calls = []
        payloads = [
            [chunk({"role": "assistant", "reasoning_content": "原始"}),
             chunk({"reasoning_content": "推理", "content": "先查"}),
             chunk(call(id="call-a", type="function", function={"name": "test", "arguments": '{"size":'})),
             chunk(call(function={"arguments": " 2}"})), chunk(finish="tool_calls"),
             '{"choices":[],"usage": {}}', "[DONE]"],
            [chunk({"reasoning_content": "下一步", "content": "结果"}), chunk(finish="stop"), "[DONE]"],
        ]

        def endpoint(request):
            self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
            self.assertEqual(request.headers["Accept"], "text/event-stream")
            pending.append(json.loads(request.content))
            encoded = "".join(f"data: {event}\r\n\r\n" for event in payloads[len(pending) - 1]).encode()
            return httpx.Response(200, stream=Bytes(encoded))

        async def tool(args, ctx):
            handler_calls.append((args, ctx.call_id))
            return "尺寸为 2"

        delivered = []

        class Channel:
            async def deliver(self, reply):
                text = "".join([part async for part in reply])
                delivered.append(text)
                return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)

        history = History(":memory:")
        self.addCleanup(history.close)
        async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
            runtime = Runtime("main", history, DeepSeek(client), Channel(), thinking=True,
                              tools=Tools([Tool(SPEC, tool)]))
            self.addAsyncCleanup(runtime.close)
            runtime.submit_batch([Message("user", (Text("前文"), SAMPLE_IMAGE, Text("后文")))])
            await asyncio.wait_for(runtime.wait_idle(), 2)
            self.assertIsNone(runtime.last_failure)
        self.assertEqual(handler_calls, [({"size": 2}, "call-a")])
        self.assertEqual(delivered, ["先查", "结果"])
        self.assertEqual(pending[0]["model"], "deepseek-flash")
        self.assertEqual(pending[0]["thinking"], {"type": "enabled"})
        self.assertEqual(pending[0]["messages"][0]["content"], [
            {"type": "text", "text": "前文"}, {"type": "image_url", "image_url": {"url": SAMPLE_IMAGE.url}},
            {"type": "text", "text": "后文"},
        ])
        assistant, result = pending[1]["messages"][-2:]
        self.assertEqual(assistant["reasoning_content"], "原始推理")
        self.assertEqual(result, {"role": "tool", "content": "尺寸为 2", "tool_call_id": "call-a"})
        self.assertEqual(history.read_after("main")[1].message.reasoning, "原始推理")
        self.assertEqual(history.search("main", keyword="原始推理"), [])

    async def test_json_looking_complete_before_finish_is_not_ready(self):
        release = asyncio.Event()

        async def source():
            yield "data: " + chunk(call(id="id", function={"name": "test", "arguments": "{}"}))
            yield ""
            yield "data: " + chunk(finish="tool_calls")
            yield ""
            await release.wait()
            yield "data: [DONE]"
            yield ""

        stream = parse_stream(source())
        task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        self.assertEqual(await asyncio.wait_for(task, 2), ToolReady(ToolCall("id", "test", "{}")))
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)

    async def test_truncated_or_abnormal_stream_never_yields_tool_ready(self):
        partial = chunk(call(id="id", function={"name": "test", "arguments": "{}"}))
        for ending in ([], [chunk(finish="tool_calls")], [chunk(finish="length"), "[DONE]"],
                       ["[DONE]"], [chunk(finish="stop"), "[DONE]"]):
            with self.subTest(ending=ending):
                result = []
                with self.assertRaises(ModelError):
                    async for event in parse_stream(lines(chunk({"content": "prefix"}), partial, *ending)):
                        result.append(event)
                self.assertEqual(result, [TextDelta("prefix")])

    async def test_finish_cannot_make_truncated_arguments_ready(self):
        valid = chunk(call(id="a", function={"name": "test", "arguments": "{}"}))
        invalid = chunk(call(index=1, id="b", function={"name": "test", "arguments": '{"size":'}))
        stream = parse_stream(lines(valid, invalid, chunk(finish="tool_calls"), "[DONE]"))
        with self.assertRaisesRegex(ModelError, "invalid_tool_arguments"):
            await anext(stream)

    async def test_http_failure_does_not_surface_response_body(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                    lambda request: httpx.Response(401, text="private-server-diagnostics"))) as client:
            with self.assertRaisesRegex(ModelError, "^http_401$"):
                _ = [event async for event in DeepSeek(client).stream([Message("user", "one")], (), thinking=False)]

    async def test_tool_blocks_have_no_reasoning_when_disabled(self):
        args = [Message("assistant", "", (ToolCall("a", "test", "{}"),), reasoning="internal"),
                Message("tool", "ok", tool_call_id="a")]
        request = build_request(args, [SPEC], thinking=False)
        self.assertEqual(request["thinking"], {"type": "disabled"})
        self.assertIsNone(request["messages"][0]["content"])
        self.assertNotIn("reasoning_content", request["messages"][0])
        with self.assertRaisesRegex(ModelError, "missing_reasoning"):
            build_request([Message("assistant", "history without protocol field")], [SPEC], thinking=True)

    async def test_reasoning_and_text_are_separate_events(self):
        self.assertEqual([e async for e in parse_stream(lines(chunk({"reasoning_content": "r", "content": "t"}),
                                                                  chunk(finish="stop"), "[DONE]"))],
                         [ReasoningDelta("r"), TextDelta("t")])
