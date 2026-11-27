import asyncio
from contextlib import closing, contextmanager
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import httpx

from agent_companion.config import RuntimeSettings
from agent_companion.images import save_image
from agent_companion.models.deepseek import MODEL_NAME, DeepSeek, _replay_key, build_request, parse_stream
from agent_companion.protocol import (DeliveryStatus, Image, InputUsage, Message, MessageDelivery, ModelClosing, ModelError,
                                      Receipt, PrivateData, Reasoning, AssistantPartsReady, Text, TextDelta, ToolCall, ToolReady, ToolSpec)
from agent_companion.runtime.compaction import Compactions, Prefix
from agent_companion.runtime.context import _SUMMARIZE
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.storage.sqlite import History
from test_message_support import PNG_BYTES, PNG_URL, assistant_message, image_workspace, reasoning_parts


SPEC = ToolSpec("test", "模拟工具", {"type": "object", "properties": {"size": {"type": "integer"}}})


@contextmanager
def _record_model_timers():
    loop = asyncio.get_running_loop()
    deadlines, call_at = [], loop.call_at

    def tracking_call_at(when, callback, *args, **kwargs):
        deadline = call_at(when, callback, *args, **kwargs)
        if getattr(callback, "__module__", None) == "agent_companion.runtime.model_deadline":
            deadlines.append(deadline)
        return deadline

    with patch.object(loop, "call_at", new=tracking_call_at):
        yield deadlines


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
        self.assertEqual(events, [TextDelta("模拟"), ModelClosing(complete=True)])

    async def test_interrupted_http_response_cleanup_survives_read_deadline_and_shutdown(self):
        reading, closing_wire, release_close, closed_wire = (asyncio.Event() for _ in range(4))
        interrupt, close_cancelled = asyncio.Event(), asyncio.Event()

        class SlowWire(httpx.AsyncByteStream):
            async def __aiter__(self):
                frame = chunk({"content": "confirmed tail", "reasoning_content": "unvalidated reasoning"})
                yield f"data: {frame}\n\n".encode()
                reading.set()
                await asyncio.Future()

            async def aclose(self):
                closing_wire.set()
                try:
                    await release_close.wait()
                except asyncio.CancelledError:
                    close_cancelled.set()
                    raise
                closed_wire.set()

        class Voice:
            async def deliver(self, reply):
                reply.select_voice_delivery()
                await anext(reply)
                await interrupt.wait()
                return Receipt(reply.info.reply_id, DeliveryStatus.PARTIAL, "confirmed")

        wire = SlowWire()
        transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=wire))
        with _record_model_timers() as deadlines, closing(History(":memory:")) as history:
            async with httpx.AsyncClient(transport=transport) as client:
                runtime = Runtime("main", history, DeepSeek(client), Voice(), thinking=True,
                                  settings=RuntimeSettings(model_timeout=.3))
                shutdown = None
                try:
                    runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
                    await asyncio.wait_for(reading.wait(), 2)
                    deadline, = deadlines
                    cutoff = deadline.when()
                    interrupt.set()
                    await asyncio.wait_for(closing_wire.wait(), 2)
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                    self.assertEqual(history.outcomes("main")[0].model_status, "model_interrupted")
                    self.assertEqual(history.read_after("main")[-1].message, Message("assistant", "confirmed"))
                    self.assertTrue(deadline.cancelled())  # 已撤销旧计时 handle，而不是只清引用。
                    shutdown = asyncio.create_task(runtime.close())
                    await asyncio.sleep(max(0, cutoff - asyncio.get_running_loop().time()) + .01)
                    self.assertFalse(close_cancelled.is_set())
                    self.assertFalse(shutdown.done())
                finally:
                    release_close.set()
                    if shutdown is None:
                        await runtime.close()
                    else:
                        await asyncio.wait_for(shutdown, 2)
                self.assertTrue(closed_wire.is_set())
                self.assertFalse(close_cancelled.is_set())

    async def test_trigger_after_partial_voice_reply_obeys_real_adapter_replay_constraints(self):
        for thinking, with_tools in ((True, True), (True, False), (False, True)):
            with self.subTest(thinking=thinking, with_tools=with_tools):
                reading = asyncio.Event()
                requests = []

                class InterruptedWire(httpx.AsyncByteStream):
                    async def __aiter__(self):
                        frame = chunk({"content": "已播放未播完", "reasoning_content": "未完成的回放"})
                        yield f"data: {frame}\n\n".encode()
                        reading.set()
                        await asyncio.Future()

                def endpoint(request):
                    requests.append(json.loads(request.content))
                    if len(requests) == 1:
                        return httpx.Response(200, stream=InterruptedWire())
                    frames = [chunk({"content": "新答案", "reasoning_content": "新推理"}), chunk(finish="stop"), "[DONE]"]
                    return httpx.Response(200, text="".join(f"data: {frame}\n\n" for frame in frames))

                class Voice:
                    first = True

                    async def deliver(self, reply):
                        reply.select_voice_delivery()
                        if self.first:
                            self.first = False
                            await anext(reply)
                            await reading.wait()
                            runtime.trigger_run()
                            return Receipt(reply.info.reply_id, DeliveryStatus.PARTIAL, "已播放")
                        text = "".join([part async for part in reply])
                        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)

                async def tool(args, context):
                    self.fail("模型未请求工具")

                with closing(History(":memory:")) as history:
                    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
                        tools = Tools([Tool(SPEC, tool)]) if with_tools else Tools()
                        runtime = Runtime("main", history, DeepSeek(client), Voice(), thinking=thinking, tools=tools)
                        try:
                            runtime.submit_messages((MessageDelivery(Message("user", "现有输入"), "trigger"),))
                            await asyncio.wait_for(runtime.wait_idle(), 2)
                            self.assertIsNone(runtime.last_failure)
                            self.assertEqual(len(requests), 2)
                            self.assertNotIn("reasoning_content", requests[1]["messages"][-1])
                            partial = history.read_after("main")[1].message
                            self.assertEqual(partial, Message("assistant", "已播放"))
                            outcome = history.outcomes("main")[0]
                            self.assertEqual((outcome.model_status, outcome.delivery_status),
                                             ("model_interrupted", DeliveryStatus.PARTIAL))
                        finally:
                            await runtime.close()

    async def test_done_commits_reasoning_usage_and_native_tool_before_http_close(self):
        closing_wire, release_close, closed_wire = (asyncio.Event() for _ in range(3))
        next_reading, release_next = asyncio.Event(), asyncio.Event()
        requests, ran = [], []

        class SlowWire(Bytes):
            async def aclose(self):
                closing_wire.set()
                await release_close.wait()
                closed_wire.set()

        class NextWire(Bytes):
            async def __aiter__(self):
                async for data in super().__aiter__():
                    yield data
                next_reading.set()
                await release_next.wait()
                yield b"data: [DONE]\n\n"

        first = [chunk({"content": "confirmed tail", "reasoning_content": "validated reasoning"}),
                 chunk(call(id="call-a", type="function", function={"name": "test", "arguments": "{}"})),
                 chunk(finish="tool_calls"), '{"choices":[],"usage":{"prompt_tokens":42}}', "[DONE]"]
        second = [chunk({"content": "next", "reasoning_content": "next reasoning"}), chunk(finish="stop")]

        def endpoint(request):
            requests.append(json.loads(request.content))
            which = len(requests)
            frames = first if which == 1 else second
            content = "".join(f"data: {frame}\n\n" for frame in frames).encode()
            wire = SlowWire(content) if which == 1 else NextWire(content)
            return httpx.Response(200, stream=wire)

        async def tool(args, context):
            ran.append(context.call_id)
            return "tool completed"

        class Voice:
            async def deliver(self, reply):
                reply.select_voice_delivery()
                content = "".join([part async for part in reply])
                if content == "confirmed tail":
                    await closing_wire.wait()
                    return Receipt(reply.info.reply_id, DeliveryStatus.PARTIAL, "confirmed")
                return Receipt(reply.info.reply_id, DeliveryStatus.FULL, content)

        with _record_model_timers() as deadlines, closing(History(":memory:")) as history:
            async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
                runtime = Runtime("main", history, DeepSeek(client), Voice(), thinking=True,
                                  tools=Tools([Tool(SPEC, tool)]), settings=RuntimeSettings(model_timeout=.25))
                try:
                    runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
                    runtime.submit_messages([MessageDelivery(Message("user", "two"), "trigger")])
                    await asyncio.wait_for(next_reading.wait(), 2)
                    self.assertFalse(closed_wire.is_set())
                    _, next_deadline = deadlines
                    cutoff = next_deadline.when()
                    self.assertFalse(next_deadline.cancelled())
                    release_close.set()
                    await asyncio.wait_for(closed_wire.wait(), 2)
                    self.assertEqual(next_deadline.when(), cutoff)
                    self.assertFalse(next_deadline.cancelled())
                    release_next.set()
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                    self.assertIsNone(runtime.last_failure)
                    self.assertEqual(ran, ["call-a"])
                    old, new = history.outcomes("main")
                    self.assertEqual((old.model_status, old.delivery_status, old.input_tokens),
                                     ("complete", DeliveryStatus.PARTIAL, 42))
                    self.assertNotEqual(old.run_id, new.run_id)
                    assistant, result, user = requests[1]["messages"][-3:]
                    self.assertEqual((assistant["content"], assistant["reasoning_content"]),
                                     ("confirmed", "validated reasoning"))
                    self.assertEqual(assistant["tool_calls"][0]["id"], "call-a")
                    self.assertEqual((result["tool_call_id"], result["content"]), ("call-a", "tool completed"))
                    self.assertEqual(user["content"], "two")
                finally:
                    release_close.set()
                    release_next.set()
                    await runtime.close()
                self.assertTrue(closed_wire.is_set())

    async def test_internal_done_preserves_reply_and_tool_continuation_on_close_failure(self):
        for blocked in (False, True):
            with self.subTest(blocked=blocked), closing(History(":memory:")) as history:
                close_started, close_cancelled, release_close, closed = (asyncio.Event() for _ in range(4))
                requests, ran = [], []

                class FailingClose(Bytes):
                    async def aclose(self):
                        close_started.set()
                        if not blocked:
                            raise httpx.ReadError("synthetic response close error")
                        try:
                            await asyncio.Future()
                        except asyncio.CancelledError:
                            close_cancelled.set()
                            await release_close.wait()
                        closed.set()

                def endpoint(request):
                    requests.append(json.loads(request.content))
                    if len(requests) == 1:
                        frames = [chunk({"content": "tool reply", "reasoning_content": "tool reasoning"}),
                                  chunk(call(id="internal-call", function={"name": "test", "arguments": "{}"})),
                                  chunk(finish="tool_calls"), "[DONE]"]
                        wire = FailingClose
                    else:
                        frames = [chunk({"content": "final answer", "reasoning_content": "final reasoning"}),
                                  chunk(finish="stop"), "[DONE]"]
                        wire = Bytes
                    return httpx.Response(200, stream=wire("".join(f"data: {frame}\n\n" for frame in frames).encode()))

                async def tool(_arguments, context):
                    ran.append(context.call_id)
                    return "tool result"

                with (patch("agent_companion.models._http_stream._RESPONSE_CLOSE_TIMEOUT", .25),
                      _record_model_timers() as deadlines):
                    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
                        runtime = Runtime("main", history, DeepSeek(client), None, thinking=True,
                                          tools=Tools([Tool(SPEC, tool)]), settings=RuntimeSettings(model_timeout=.15))
                        try:
                            with self.assertLogs("agent_companion.runtime.engine", level="WARNING") as logs:
                                runtime.submit_messages([MessageDelivery(Message("user", "internal request"), "trigger")])
                                await asyncio.wait_for(close_started.wait(), 2)
                                self.assertTrue(deadlines[0].cancelled())
                                await asyncio.wait_for(runtime.wait_idle(), 2)
                            expected = "transport_close_timeout" if blocked else "transport_error"
                            self.assertIn(f"model cleanup failed ({expected})", "\n".join(logs.output))
                            self.assertIsNone(runtime.last_failure)
                            if blocked:
                                self.assertTrue(close_cancelled.is_set())
                                self.assertFalse(closed.is_set())  # 续轮不必等待底层关闭任务恢复。
                        finally:
                            release_close.set()
                            await runtime.close()
                        if blocked:
                            await asyncio.wait_for(closed.wait(), 2)

                self.assertEqual(ran, ["internal-call"])
                self.assertEqual(len(requests), 2)
                assistant, result = requests[-1]["messages"][-2:]
                self.assertEqual((assistant["content"], assistant["reasoning_content"]),
                                 ("tool reply", "tool reasoning"))
                self.assertEqual(assistant["tool_calls"][0]["id"], "internal-call")
                self.assertEqual(result, {"role": "tool", "content": "tool result", "tool_call_id": "internal-call"})
                self.assertEqual([row.model_status for row in history.outcomes("main")], ["complete", "complete"])
                first, final = [row.message for row in history.read_after("main")][1:]
                self.assertEqual(reasoning_parts(first)[0], PrivateData(_replay_key(MODEL_NAME), b"tool reasoning"))
                self.assertEqual((final.content, reasoning_parts(final)[0]),
                                 ("final answer", PrivateData(_replay_key(MODEL_NAME), b"final reasoning")))

    async def test_internal_missing_done_never_saves_body_or_replay(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled), closing(History(":memory:")) as history:
                frames = [chunk({"content": "incomplete text", "reasoning_content": "incomplete reasoning"}),
                          chunk(finish="stop")]
                data = "".join(f"data: {frame}\n\n" for frame in frames).encode()

                class MissingDone(Bytes):
                    async def __aiter__(self):
                        async for block in super().__aiter__():
                            yield block
                        if cancelled:
                            raise asyncio.CancelledError

                transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=MissingDone(data)))
                async with httpx.AsyncClient(transport=transport) as client:
                    runtime = Runtime("main", history, DeepSeek(client), None, thinking=True)
                    try:
                        runtime.submit_messages([MessageDelivery(Message("user", "internal request"), "trigger")])
                        await asyncio.wait_for(runtime.wait_idle(), 2)
                        status = "model_cancelled" if cancelled else "model_failed"
                        self.assertEqual(runtime.last_failure, status)
                    finally:
                        await runtime.close()
                outcome, = history.outcomes("main")
                self.assertEqual((outcome.model_status, outcome.assistant_message_id), (status, None))
                self.assertEqual([row.message for row in history.read_after("main")], [Message("user", "internal request")])

    async def test_two_complete_tool_rounds_finish_after_response_closes_without_new_input(self):
        requests, tool_ids = [], []
        sent = asyncio.Queue()
        closing_wire = [asyncio.Event() for _ in range(3)]
        release_close = [asyncio.Event() for _ in range(3)]
        closed = [asyncio.Event() for _ in range(3)]

        class SlowWire(Bytes):
            def __init__(self, content, index):
                super().__init__(content)
                self.index = index

            async def aclose(self):
                closing_wire[self.index].set()
                await release_close[self.index].wait()
                closed[self.index].set()

        def endpoint(request):
            index = len(requests)
            if index:
                self.assertTrue(closed[index - 1].is_set(), "续轮不应把完整响应留在退役队列")
            requests.append(json.loads(request.content))
            if index < 2:
                frames = [chunk({"content": f"step {index}"}),
                          chunk(call(id=f"call-{index}", type="function",
                                     function={"name": "test", "arguments": "{}"})),
                          chunk(finish="tool_calls"), "[DONE]"]
            else:
                frames = [chunk({"content": "final answer"}), chunk(finish="stop"), "[DONE]"]
            data = "".join(f"data: {frame}\n\n" for frame in frames).encode()
            return httpx.Response(200, stream=SlowWire(data, index))

        async def tool(args, context):
            tool_ids.append(context.call_id)
            return "tool result"

        class Voice:
            async def deliver(self, reply):
                reply.select_voice_delivery()
                text = "".join([part async for part in reply])
                sent.put_nowait((reply.info.run_id, text))
                return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)

        with closing(History(":memory:")) as history:
            async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
                runtime = Runtime("main", history, DeepSeek(client), Voice(), tools=Tools([Tool(SPEC, tool)]))
                try:
                    runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
                    run_ids = set()
                    for index, expected in enumerate(("step 0", "step 1", "final answer")):
                        await asyncio.wait_for(closing_wire[index].wait(), 2)
                        run_id, text = await asyncio.wait_for(sent.get(), 2)
                        self.assertEqual(text, expected)
                        run_ids.add(run_id)
                        self.assertEqual(len(requests), index + 1)
                        self.assertFalse(runtime._retired_models)
                        release_close[index].set()
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                    self.assertIsNone(runtime.last_failure)
                    self.assertEqual(len(run_ids), 1)
                    self.assertEqual(tool_ids, ["call-0", "call-1"])
                    self.assertEqual(len(history.outcomes("main")), 3)
                    self.assertTrue(all(event.is_set() for event in closed))
                    self.assertFalse(runtime._retired_models)
                finally:
                    for release in release_close:
                        release.set()
                    await runtime.close()

    async def test_response_close_budget_reports_timeout_without_waiting_for_uncooperative_wire(self):
        cancellation, release_close, closed = (asyncio.Event() for _ in range(3))

        class SlowWire(Bytes):
            async def aclose(self):
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    cancellation.set()
                    await release_close.wait()
                closed.set()

        frame = f"data: {chunk({'content': 'confirmed'})}\n\n".encode()
        transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=SlowWire(frame)))
        with patch("agent_companion.models._http_stream._RESPONSE_CLOSE_TIMEOUT", .02):
            async with httpx.AsyncClient(transport=transport) as client:
                model = DeepSeek(client)
                try:
                    with self.assertLogs("agent_companion.models._http_stream", level="WARNING") as logs:
                        with self.assertRaises(ModelError) as error:
                            _ = [event async for event in model.stream([Message("user", "one")], (), thinking=False)]
                    self.assertEqual(error.exception.diagnostic, "transport_close_timeout")
                    self.assertEqual(len(logs.output), 1)
                    await asyncio.wait_for(cancellation.wait(), 2)
                    await asyncio.wait_for(model.aclose(), 2)
                    self.assertFalse(closed.is_set())
                finally:
                    release_close.set()
                await asyncio.wait_for(closed.wait(), 2)

    async def test_response_close_timeout_preserves_the_primary_error_or_cancellation(self):
        for status in (503, 200):
            with self.subTest(status=status):
                reading = asyncio.Event()

                class SlowWire(Bytes):
                    async def __aiter__(self):
                        reading.set()
                        await asyncio.Future()
                        yield b""

                    async def aclose(self):
                        await asyncio.Future()

                transport = httpx.MockTransport(lambda _: httpx.Response(status, stream=SlowWire(b"")))
                with patch("agent_companion.models._http_stream._RESPONSE_CLOSE_TIMEOUT", .02):
                    async with httpx.AsyncClient(transport=transport) as client:
                        model = DeepSeek(client)

                        async def read():
                            return [event async for event in model.stream([Message("user", "one")], (), thinking=False)]

                        with self.assertLogs("agent_companion.models._http_stream", level="WARNING") as logged:
                            if status == 503:
                                with self.assertRaises(ModelError) as error:
                                    await asyncio.wait_for(read(), 2)
                                self.assertEqual(error.exception.diagnostic, "http_503")
                            else:
                                reader = asyncio.create_task(read())
                                try:
                                    await asyncio.wait_for(reading.wait(), 2)
                                finally:
                                    reader.cancel()
                                    with self.assertRaises(asyncio.CancelledError):
                                        await asyncio.wait_for(reader, 2)
                        self.assertEqual(len(logged.output), 1)
                        await model.aclose()

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
                    for _ in range(2):
                        runtime.submit_messages((MessageDelivery(Message("user", "synthetic user content"), "trigger"),))
                        await runtime.wait_idle()
                        self.assertEqual(runtime.last_failure, "model_failed")
                self.assertEqual([row.model_error for row in history.outcomes("diagnostics")], ["http_401", "http_429"])
                lines = " ".join(logs.output)
                self.assertIn("http_401", lines)
                self.assertIn("http_429", lines)
                self.assertNotIn("synthetic", lines)
                self.assertEqual(ModelError("unrecognized adapter details").diagnostic, "model_failed")

    async def test_actual_prompt_usage_triggers_summary_with_original_image(self):
        requests = []
        workspace = image_workspace(self)
        old = Message("user", (Text("历史图文"), save_image(workspace, PNG_BYTES)))
        settings = RuntimeSettings(context_budget=300, tail_budget=60)

        def endpoint(request):
            payload = json.loads(request.content)
            content, tokens = {
                "首次请求": ("初次回复", 300),
                _SUMMARIZE.content: ("简洁概要", 1000),
                "新的文字": ("新的回复", 1),
                "应用摘要": ("下一轮回复", 2),
            }[payload["messages"][-1]["content"]]
            requests.append(payload)
            finish = json.dumps({**json.loads(chunk(finish="stop")), "usage": {"prompt_tokens": tokens}})
            frames = [chunk({"content": content}), finish, "[DONE]"]
            return httpx.Response(200, text="".join(f"data: {frame}\n\n" for frame in frames))

        with closing(History(":memory:")) as history:
            history.append("child", [old])
            async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
                runtime = Runtime("child", history, DeepSeek(client), None, settings=settings, workspace=workspace)
                try:
                    runtime.submit_messages((MessageDelivery(Message("user", "首次请求"), "trigger"),))
                    await runtime.wait_idle()
                    self.assertIsNone(runtime.last_failure)
                    self.assertIsNone(history.summary("child"))
                    self.assertEqual(history.last_input_tokens("child"), 300)
                    runtime.submit_messages((MessageDelivery(Message("user", "新的文字"), "trigger"),))
                    await runtime.wait_idle()
                    self.assertIsNone(runtime.last_failure)
                    task = runtime._compactions._generating
                    if task is not None:
                        await asyncio.wait_for(asyncio.shield(task), 2)
                    self.assertIsNone(history.summary("child"))
                    runtime.submit_messages((MessageDelivery(Message("user", "应用摘要"), "trigger"),))
                    await runtime.wait_idle()
                    self.assertIsNone(runtime.last_failure)
                finally:
                    await runtime.close()
            self.assertEqual(len(requests), 4)
            compressed = next(request["messages"] for request in requests[1:3]
                              if request["messages"][-1]["content"] == _SUMMARIZE.content)
            self.assertEqual(compressed[0], requests[0]["messages"][0])
            self.assertEqual(compressed[0]["content"][1]["image_url"]["url"], PNG_URL)
            self.assertNotIn("新的文字", repr(compressed))
            self.assertNotIn("image_url", repr(requests[-1]))
            self.assertIn("新的文字", repr(requests[-1]))
            self.assertEqual(requests[-1]["messages"][-1]["content"], "应用摘要")
            self.assertEqual(history.summary("child").text, "简洁概要")
            self.assertEqual(history.read_after("child")[0].message, old)
            self.assertEqual([outcome.input_tokens for outcome in history.outcomes("child")], [300, 1, 2])

    async def test_normal_and_summary_share_thinking_and_tools_replay_condition(self):
        requests = []

        def endpoint(request):
            requests.append(json.loads(request.content))
            frames = (chunk({"content": "synthetic reply"}), chunk(finish="stop"), "[DONE]")
            return httpx.Response(200, text="".join(f"data: {frame}\n\n" for frame in frames))

        old = [Message("user", "previous question"), Message("assistant", (
            Reasoning(PrivateData(_replay_key(MODEL_NAME), b"synthetic thought")),
            ToolCall("historic-call", SPEC.name, "{}", "historic result"),
        ))]
        for thinking, tools in ((True, (SPEC,)), (True, ()), (False, (SPEC,))):
            with self.subTest(thinking=thinking, has_tools=bool(tools)), closing(History(":memory:")) as history:
                compactions = Compactions(history)
                async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
                    model = DeepSeek(client)
                    try:
                        async for _ in model.stream([*old, Message("user", "new question")], tools, thinking=thinking):
                            pass
                        compactions.start("main", Prefix(0, 1, 2), [*old, _SUMMARIZE], tools, model=model,
                                          thinking=thinking, timeout=2)
                        await asyncio.wait_for(asyncio.shield(compactions._generating), 2)
                        self.assertIsNotNone(compactions.ready("main", 0))
                    finally:
                        await compactions.close()
                ordinary, summary = requests[-2:]
                self.assertEqual(ordinary["messages"][:-1], summary["messages"][:-1])
                assistant, result = ordinary["messages"][1:3]
                self.assertEqual(assistant.get("reasoning_content"), "synthetic thought" if thinking and tools else None)
                self.assertEqual(result, {"role": "tool", "tool_call_id": "historic-call", "content": "historic result"})

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
            workspace = image_workspace(self)
            image = save_image(workspace, PNG_BYTES)
            runtime = Runtime("main", history, DeepSeek(client), Channel(), thinking=True,
                              tools=Tools([Tool(SPEC, tool)]), workspace=workspace)
            self.addAsyncCleanup(runtime.close)
            runtime.submit_messages([MessageDelivery(Message("user", (Text("前文"), image, Text("后文"))), "trigger")])
            await asyncio.wait_for(runtime.wait_idle(), 2)
            self.assertIsNone(runtime.last_failure)
        self.assertEqual(handler_calls, [({"size": 2}, "call-a")])
        self.assertEqual(delivered, ["先查", "结果"])
        self.assertEqual(pending[0]["model"], "deepseek-flash")
        self.assertEqual(pending[0]["thinking"], {"type": "enabled"})
        self.assertEqual(pending[0]["messages"][0]["content"], [
            {"type": "text", "text": "前文"}, {"type": "image_url", "image_url": {"url": PNG_URL}},
            {"type": "text", "text": "后文"},
        ])
        assistant, result = pending[1]["messages"][-2:]
        self.assertEqual(assistant["reasoning_content"], "原始推理")
        self.assertEqual(result, {"role": "tool", "content": "尺寸为 2", "tool_call_id": "call-a"})
        self.assertEqual(reasoning_parts(history.read_after("main")[1].message)[0], PrivateData(_replay_key(MODEL_NAME), "原始推理".encode()))
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
        ready = ToolCall("id", "test", "{}", private=PrivateData(_replay_key(MODEL_NAME), b"0"))
        self.assertEqual(await asyncio.wait_for(task, 2), AssistantPartsReady((ready,)))
        self.assertEqual(await anext(stream), ToolReady(ready))
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)

    async def test_truncated_or_abnormal_stream_never_yields_tool_ready(self):
        partial = chunk(call(id="id", function={"name": "test", "arguments": "{}"}))
        for ending in ([], [chunk(finish="tool_calls")], [chunk(finish="length"), "[DONE]"],
                       ["[DONE]"], [chunk(finish="stop"), "[DONE]"]):
            with self.subTest(ending=ending):
                result = []
                with self.assertRaises(ModelError):
                    async for event in parse_stream(lines(chunk({"content": "prefix", "reasoning_content": "unfinished"}),
                                                          partial, *ending)):
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
        args = [assistant_message("", (replace(ToolCall("a", "test", "{}"), result="ok"),), reasoning=PrivateData(_replay_key(MODEL_NAME), b"internal"))]
        request = build_request(args, [SPEC], thinking=False)
        self.assertEqual(request["thinking"], {"type": "disabled"})
        self.assertIsNone(request["messages"][0]["content"])
        self.assertNotIn("reasoning_content", request["messages"][0])
        request = build_request([Message("assistant", "history without protocol field")], [SPEC], thinking=True)
        self.assertNotIn("reasoning_content", request["messages"][0])

    async def test_text_is_immediate_and_complete_reasoning_is_delivered_once(self):
        stream = parse_stream(lines(chunk({"reasoning_content": "r", "content": "t"}),
                                    chunk({"reasoning_content": "续"}), chunk(finish="stop"), "[DONE]"))
        self.assertEqual(await anext(stream), TextDelta("t"))
        self.assertEqual([event async for event in stream], [AssistantPartsReady((
            Reasoning(PrivateData(_replay_key(MODEL_NAME), b"r")), Text("t"),
            Reasoning(PrivateData(_replay_key(MODEL_NAME), "续".encode())),
        ))])

    async def test_explicit_empty_reasoning_survives_until_followup_with_tools(self):
        for reasoning in ({"reasoning_content": ""}, {}, {"reasoning_content": None}):
            with self.subTest(reasoning=reasoning), closing(History(":memory:")) as history:
                requests = []

                def endpoint(request):
                    requests.append(json.loads(request.content))
                    delta = {"content": "first answer", **reasoning} if len(requests) == 1 else {"content": "next answer"}
                    frames = [chunk(delta), chunk(finish="stop"), "[DONE]"]
                    return httpx.Response(200, text="".join(f"data: {frame}\n\n" for frame in frames))

                async def unused_tool(_arguments, _context):
                    self.fail("fixture does not request tool execution")

                async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
                    model = DeepSeek(client)
                    for tools in (None, Tools([Tool(SPEC, unused_tool)])):
                        runtime = Runtime("main", history, model, None, tools=tools, thinking=True)
                        try:
                            message = Message("user", "next prompt" if tools else "first prompt")
                            runtime.submit_messages((MessageDelivery(message, "trigger"),))
                            await asyncio.wait_for(runtime.wait_idle(), 2)
                            self.assertIsNone(runtime.last_failure)
                        finally:
                            await runtime.close()
                self.assertNotIn("tools", requests[0])
                self.assertIn("tools", requests[1])
                saved = history.read_after("main")[1].message
                previous = requests[1]["messages"][1]
                if reasoning.get("reasoning_content") == "":
                    self.assertEqual(reasoning_parts(saved), (PrivateData(_replay_key(MODEL_NAME), b""),))
                    self.assertEqual(previous, {"role": "assistant", "content": "first answer", "reasoning_content": ""})
                else:
                    self.assertFalse(reasoning_parts(saved))
                    self.assertEqual(previous["role"], "assistant")
                    self.assertIn("first answer", repr(previous))
                    self.assertNotIn("reasoning_content", previous)

    async def test_empty_reasoning_alone_has_no_replay(self):
        events = [event async for event in parse_stream(lines(chunk({"reasoning_content": ""}),
                                                              chunk(finish="stop"), "[DONE]"))]
        self.assertEqual(events, [])

    async def test_tool_round_missing_reasoning_sends_history_and_surfaces_provider_rejection(self):
        calls = []
        requests = []

        async def tool(_arguments, context):
            calls.append(context.call_id)
            return "finished tool"

        def response(request):
            requests.append(json.loads(request.content))
            if len(requests) == 2:
                return httpx.Response(400)
            frames = [chunk(call(id="protected", function={"name": "test", "arguments": "{}"})),
                      chunk(finish="tool_calls"), "[DONE]"]
            return httpx.Response(200, text="".join(f"data: {frame}\n\n" for frame in frames))

        with closing(History(":memory:")) as history:
            async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
                runtime = Runtime("main", history, DeepSeek(client), None, thinking=True,
                                  tools=Tools([Tool(SPEC, tool)]))
                try:
                    runtime.submit_messages([MessageDelivery(Message("user", "current input"), "trigger")])
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                    self.assertEqual(runtime.last_failure, "model_failed")
                finally:
                    await runtime.close()
            self.assertEqual(len(requests), 2)
            assistant, result = requests[1]["messages"][-2:]
            self.assertNotIn("reasoning_content", assistant)
            self.assertEqual(assistant["tool_calls"][0]["id"], result["tool_call_id"])
            self.assertEqual(calls, ["protected"])
            self.assertEqual([row.message for row in history.read_after("main")][1:], [
                assistant_message("", (ToolCall("protected", "test", "{}", "finished tool",
                                                PrivateData(_replay_key(MODEL_NAME), b"0")),)),
            ])

    async def test_nonmatching_deepseek_replay_is_never_decoded(self):
        bad = PrivateData(_replay_key(MODEL_NAME), b"\xff")
        entry = assistant_message("confirmed", reasoning=bad)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
            for model in (DeepSeek(client, model="other-model"), DeepSeek(client)):
                self.assertEqual(model.history_replay_bytes(entry, [SPEC], thinking=True), 0)
                self.assertEqual(model._payload([entry], [SPEC], thinking=True)["messages"],
                                 [{"role": "assistant", "content": "confirmed"}])
