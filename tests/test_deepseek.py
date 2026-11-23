import asyncio
from contextlib import closing
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import httpx

from agent_companion.config import RuntimeSettings
from agent_companion.models.deepseek import DeepSeek, build_request, parse_stream
from agent_companion.protocol import (OMITTED_INPUT_IMAGE, DeliveryStatus, Image, Message, ModelError, ReasoningDelta, Receipt,
                                      RequestTooLarge, Text, TextDelta, ToolCall, ToolReady, ToolSpec)
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
    async def test_request_body_limit_is_exact_and_applies_before_transport(self):
        requests = []
        messages = [Message("user", '汉字\n"' * 100)]
        expected = json.dumps(build_request(messages, (), thinking=False), ensure_ascii=False,
                              separators=(",", ":")).encode("utf-8")

        def endpoint(request):
            requests.append(request)
            frames = [chunk({"content": "模拟"}), chunk(finish="stop"), "[DONE]"]
            return httpx.Response(200, content="".join(f"data: {frame}\n\n" for frame in frames).encode())

        async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
            exact = DeepSeek(client, request_limit=len(expected))
            over = DeepSeek(client, request_limit=len(expected) - 1)
            self.assertTrue(exact.request_fits(messages, (), thinking=False))
            self.assertFalse(over.request_fits(messages, (), thinking=False))
            with self.assertRaisesRegex(RequestTooLarge, "^request_body_too_large$"):
                _ = [item async for item in over.stream(messages, (), thinking=False)]
            self.assertEqual(requests, [])
            self.assertEqual([item async for item in exact.stream(messages, (), thinking=False)], [TextDelta("模拟")])
            self.assertEqual(requests[0].headers["Content-Type"], "application/json")
            self.assertEqual(requests[0].content, expected)

    async def test_oversized_current_batch_preserves_text_and_valid_image_then_projects_old_images(self):
        requests = []
        image = Image("data:image/png;base64," + "A" * 56)
        original = [Message("user", "同批文字"), *(Message("user", (image,)) for _ in range(4))]

        def endpoint(request):
            self.assertLessEqual(len(request.content), 512)
            requests.append(json.loads(request.content))
            frames = [chunk({"content": "已继续"}), chunk(finish="stop"), "[DONE]"]
            return httpx.Response(200, content="".join(f"data: {frame}\n\n" for frame in frames).encode())

        class Channel:
            async def deliver(self, reply):
                text = "".join([part async for part in reply])
                return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)

        with closing(History(":memory:")) as history:
            async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
                model = DeepSeek(client, request_limit=512)
                runtime = Runtime("main", history, model, Channel())
                self.assertTrue(all(model.request_fits([item], (), thinking=False) for item in original))
                runtime.submit_batch(original)
                await runtime.wait_idle()
                self.assertIsNone(runtime.last_failure)
                self.assertEqual(len(requests), 1)
                first = requests[0]["messages"]
                self.assertEqual(first[0]["content"], "同批文字")
                self.assertEqual(first[-1]["content"][0]["type"], "image_url")
                self.assertTrue(any(item["content"] == [{"type": "text", "text": OMITTED_INPUT_IMAGE.text}]
                                    for item in first[1:-1]))
                runtime.submit_batch([Message("user", "下一条纯文字")])
                await runtime.wait_idle()
                self.assertIsNone(runtime.last_failure)
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[1]["messages"][-1]["content"], "下一条纯文字")
                self.assertEqual([row.message for row in history.read_after("main")[:5]], original)
                await runtime.close()

    async def test_current_batch_skips_large_image_and_still_sends_small_image(self):
        requests = []
        large = Image("data:image/png;base64," + "A" * 512)
        small = SAMPLE_IMAGE
        text = Text('同批文字\n"原样"')
        original = Message("user", (large, small, large, text))
        spec = ToolSpec("test", "", {"type": "object"})

        def endpoint(request):
            self.assertLessEqual(len(request.content), 512)
            requests.append(json.loads(request.content))
            return httpx.Response(200, text="".join(f"data: {part}\n\n" for part in (
                chunk({"content": "正文可答"}), chunk(finish="stop"), "[DONE]")))

        async def tool(args, context):
            self.fail("unexpected tool call")

        class Channel:
            async def deliver(self, reply):
                return Receipt(reply.info.reply_id, DeliveryStatus.FULL, "".join([part async for part in reply]))

        with closing(History(":memory:")) as history:
            async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
                runtime = Runtime("main", history, DeepSeek(client, request_limit=512), Channel(),
                                  instructions="固定指令", tools=Tools([Tool(spec, tool)]))
                try:
                    runtime.submit_batch([original])
                    await runtime.wait_idle()
                    self.assertIsNone(runtime.last_failure)
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(requests[0], build_request([
                        Message("system", "固定指令"),
                        Message("user", (OMITTED_INPUT_IMAGE, small, OMITTED_INPUT_IMAGE, text)),
                    ], (spec,), thinking=False))
                    self.assertEqual(history.read_after("main")[0].message, original)
                finally:
                    await runtime.close()

    async def test_image_placeholders_do_not_fill_small_request_before_text(self):
        received = []

        def endpoint(request):
            received.extend(json.loads(request.content)["messages"])
            return httpx.Response(200, text="".join(f"data: {part}\n\n" for part in (
                chunk({"content": "文字可答"}), chunk(finish="stop"), "[DONE]")))

        class Channel:
            async def deliver(self, reply):
                return Receipt(reply.info.reply_id, DeliveryStatus.FULL, "".join([part async for part in reply]))

        with closing(History(":memory:")) as history:
            async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
                runtime = Runtime("main", history, DeepSeek(client, request_limit=250), Channel())
                runtime.submit_batch([Message("user", (SAMPLE_IMAGE,))] * 15 + [Message("user", "正常文字")])
                await runtime.wait_idle()
                self.assertIsNone(runtime.last_failure)
                self.assertEqual([item["content"] for item in received], [OMITTED_INPUT_IMAGE.text, "正常文字"])
                self.assertEqual(len(history.read_after("main")), 17)
                await runtime.close()

    async def test_coalesced_image_placeholders_keep_a_later_small_image_and_tool_group(self):
        requests, delivered = [], []
        large = Image("data:image/png;base64," + "A" * 2048)
        text = Message("user", "正常文字")
        original = [Message("user", (large,))] * 300 + [Message("user", (SAMPLE_IMAGE,)), text]
        tool_call = ToolCall("call-a", "test", '{"size":2}')
        frames = [
            [chunk({"reasoning_content": "模拟推理"}),
             chunk(call(id=tool_call.id, function={"name": tool_call.name, "arguments": tool_call.arguments})),
             chunk(finish="tool_calls"), "[DONE]"],
            [chunk({"content": "图文已处理"}), chunk(finish="stop"), "[DONE]"],
        ]

        def endpoint(request):
            self.assertLessEqual(len(request.content), 1024)
            events = frames[len(requests)]
            requests.append(json.loads(request.content))
            return httpx.Response(200, text="".join(f"data: {event}\n\n" for event in events))

        async def tool(args, context):
            return "尺寸为 2"

        class Channel:
            async def deliver(self, reply):
                body = "".join([part async for part in reply])
                delivered.append(body)
                return Receipt(reply.info.reply_id, DeliveryStatus.FULL, body)

        settings = RuntimeSettings(context_budget=4096, tail_budget=1024)
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "history.sqlite"
            with closing(History(path)) as history:
                async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
                    runtime = Runtime("main", history, DeepSeek(client, request_limit=1024), Channel(),
                                      settings=settings, thinking=True, tools=Tools([Tool(SPEC, tool)]))
                    try:
                        history.inbound_window("main", "feishu", 1)
                        history.record_inbound("main", "feishu", 2,
                                               [(f"om_{i}", message) for i, message in enumerate(original)])
                        batch_id, = history.pending_inbound("main", "feishu")
                        runtime.submit_persisted_batch(batch_id)
                        await asyncio.wait_for(runtime.wait_idle(), 2)
                        self.assertIsNone(runtime.last_failure)
                    finally:
                        await runtime.close()
            with closing(History(path)) as reopened:
                self.assertEqual([row.message for row in reopened.read_after("main")[:len(original)]], original)

        expected = [Message("user", OMITTED_INPUT_IMAGE.text), Message("user", (SAMPLE_IMAGE,)), text]
        continuation = [Message("assistant", "", (tool_call,), reasoning="模拟推理"),
                        Message("tool", "尺寸为 2", tool_call_id=tool_call.id)]
        self.assertEqual(requests, [build_request(expected, (SPEC,), thinking=True),
                                    build_request(expected + continuation, (SPEC,), thinking=True)])
        self.assertEqual(delivered, ["图文已处理"])

    async def test_default_request_limit_prefers_latest_of_137_images(self):
        old = Image("data:image/png;base64,AAAA")
        head, tail = Message("user", "批次开头"), Message("user", "批次结尾")
        old_image = Message("user", (old,))
        omitted_image = Message("user", (OMITTED_INPUT_IMAGE,))
        latest = Message("user", (SAMPLE_IMAGE,))
        before, between, after = Text("富文本开头"), Text("目标图之前"), Text('保留\n"原文"')
        rich = Message("user", (before, *((old,) * 135), between, SAMPLE_IMAGE, after))
        rich_projection = Message("user", (before, *((OMITTED_INPUT_IMAGE,) * 135), between, SAMPLE_IMAGE, after))
        cases = (
            ("富文本", [head, old_image, rich, tail], [head, omitted_image, rich_projection, tail]),
            ("独立图片", [head, *([old_image] * 136), latest, tail],
             [head, Message("user", OMITTED_INPUT_IMAGE.text), latest, tail]),
        )
        settings = RuntimeSettings(context_budget=4096, tail_budget=1024)
        fixed = Message("system", "保留图文顺序")

        for label, original, projected in cases:
            with self.subTest(layout=label):
                requests, delivered = [], []

                def endpoint(request):
                    self.assertLessEqual(len(request.content), 40 * 1024 * 1024)
                    requests.append(json.loads(request.content))
                    frames = [chunk({"content": "图文已处理"}), chunk(finish="stop"), "[DONE]"]
                    return httpx.Response(200, text="".join(f"data: {frame}\n\n" for frame in frames))

                class Channel:
                    async def deliver(self, reply):
                        body = "".join([part async for part in reply])
                        delivered.append(body)
                        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, body)

                with closing(History(":memory:")) as history:
                    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
                        runtime = Runtime("main", history, DeepSeek(client), Channel(), settings=settings,
                                          instructions=fixed.content)
                        try:
                            runtime.submit_batch(original)
                            await asyncio.wait_for(runtime.wait_idle(), 2)
                            self.assertIsNone(runtime.last_failure)
                            self.assertEqual([row.message for row in history.read_after("main")],
                                             [*original, Message("assistant", "图文已处理")])
                        finally:
                            await runtime.close()
                self.assertEqual(requests, [build_request([fixed, *projected], (), thinking=False)])
                self.assertEqual(delivered, ["图文已处理"])

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

    async def test_summary_projection_uses_byte_budget_for_images_and_large_text(self):
        requests = []
        limit = 1400

        def endpoint(request):
            self.assertLessEqual(len(request.content), limit)
            requests.append(json.loads(request.content))
            frames = [chunk({"content": "简洁概要"}), chunk(finish="stop"), "[DONE]"]
            return httpx.Response(200, content="".join(f"data: {frame}\n\n" for frame in frames).encode())

        first = Message("user", (Text("历史图文"), Image("data:image/png;base64," + "A" * 6000)))
        old_text = Message("user", "旧文本" * 650)
        settings = RuntimeSettings(context_budget=3000, tail_budget=300)
        with closing(History(":memory:")) as history:
            history.append("child", [first, old_text])
            async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
                runtime = Runtime("child", history, DeepSeek(client, request_limit=limit), None, settings=settings)
                self.assertIsNone(await runtime.run_internal("run", "新的文字"))
                self.assertEqual(len(requests), 2)
                serialized = json.dumps(requests[0], ensure_ascii=False)
                self.assertIn("历史图片已省略", serialized)
                self.assertNotIn("image_url", serialized)
                self.assertEqual(requests[1]["messages"][-1]["content"], "新的文字")
                self.assertEqual([row.message for row in history.read_after("child")[:2]], [first, old_text])

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
