import asyncio
from base64 import b64decode
from contextlib import aclosing, closing
from dataclasses import replace
import json
import traceback
import unittest
from unittest.mock import patch

import httpx

from agent_companion import images
from agent_companion.config import RuntimeSettings, bailian_llm_base_url
from agent_companion.images import VIEW_IMAGE_OK, VIEW_IMAGE_TOOL_NAME, image_request_bytes, save_image
from agent_companion.model_diagnostics import ModelRequestDiagnostics
from agent_companion.models import _http_stream
from agent_companion.models.bailian import Bailian
from agent_companion.models.deepseek import DeepSeek
from agent_companion.models.minimax import MiniMax
from agent_companion.protocol import (DeliveryStatus, Image, InputUsage, Message, MessageDelivery, ModelClosing, ModelError, Receipt,
                                      PrivateData, Reasoning, Text, TextDelta, ToolCall, ToolSpec)
from agent_companion.runtime.context import _SUMMARIZE
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.runtime.view_image import VIEW_IMAGE_TOOL
from agent_companion.storage.sqlite import History, ReplyOutcome, Summary
from test_model_http_reuse import _HTTP11, _sse
from test_message_support import PNG_BYTES, PNG_URL, assistant_message, image_workspace, reasoning_parts


_SPEC = ToolSpec("lookup", "Look up a fixture", {"type": "object"})
_INPUT = Message("user", "fixture question")
_KEY = "fixture-authorization-value"
_BAD_MINIMAX = {'base_resp': {'status_code': 2013, 'status_msg': 'thinking.type="disabled" is not allowed'}}
_BAD_BAILIAN = {'error': {'code': 'invalid_parameter', 'message': 'Unsupported enable_thinking'},
                'request_id': 'fixture-request'}
_LOGGER = 'agent_companion.model_diagnostics'


def _model(client, adapter, *, api_key=_KEY, model="fixture-model", base_url=None, preserve_thinking=None):
    if adapter == "minimax":
        return MiniMax(client, model=model, api_key=api_key, base_url=base_url, reasoning_effort="low")
    return Bailian(client, model=model, api_key=api_key, preserve_thinking=preserve_thinking,
                   base_url=base_url or bailian_llm_base_url("cn-beijing", "fixture-space"))


def _choice(delta=None, *, finish=None, usage=None):
    return {"choices": [{"index": 0, "delta": {} if delta is None else delta, "finish_reason": finish}],
            "usage": usage}


def _tool_start():
    return _choice({"tool_calls": [{"index": 0, "id": "call-fixture", "type": "function", "function": {
        "name": "lookup", "arguments": '{"city":',
    }}]})


def _tool_end():
    return _choice({"tool_calls": [{"index": 0, "function": {"arguments": '"Oslo"}'}}]}, finish="tool_calls")


def _answer(text="fixture answer"):
    return _sse(_choice({"content": text, "reasoning_content": "fixture final reasoning"}, finish="stop",
                        usage={"prompt_tokens": 13}), b"[DONE]")


def _response(*items):
    return httpx.Response(200, content=_sse(*items), headers={"content-type": "text/event-stream"})


async def _collect(model, *, messages=(_INPUT,), thinking=True, events=None, runtime_style=False):
    if events is None:
        events = []
    async with aclosing(model.stream(messages, (_SPEC,), thinking=thinking)) as stream:
        async for event in stream:
            events.append(event)
            if runtime_style and isinstance(event, ModelClosing) and event.complete:
                break
    return events


class ChatModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_tool_round_with_optional_reasoning_and_usage(self):
        for adapter in ("minimax", "bailian"):
            for thought in (True, False):
                with self.subTest(adapter=adapter, thought=thought), closing(History(":memory:")) as history:
                    requests, calls = [], []

                    async def lookup(arguments, context):
                        calls.append(arguments)
                        return "fixture tool result"

                    def wire(request):
                        requests.append(json.loads(request.content))
                        self.assertEqual(request.url.path.rsplit("/", 2)[-2:], ["chat", "completions"])
                        self.assertEqual(request.headers["Authorization"], f"Bearer {_KEY}")
                        if len(requests) == 1:
                            ahead = [_choice({"reasoning_content": "fixture "}), _choice({"reasoning_content": "plan"})]
                            return _response(*(ahead if thought else [_choice({"role": "assistant"})]),
                                             _tool_start(), _tool_end(),
                                             {"choices": [], "usage": {"prompt_tokens": 7}}, b"[DONE]")
                        return httpx.Response(200, content=_answer())

                    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
                        model = _model(client, adapter)
                        runtime = Runtime("main", history, model, None, tools=Tools([Tool(_SPEC, lookup)]), thinking=True)
                        try:
                            runtime.submit_messages([MessageDelivery(_INPUT, "trigger")])
                            await asyncio.wait_for(runtime.wait_idle(), 2)
                        finally:
                            await runtime.close()
                            await model.aclose()
                    self.assertEqual(calls, [{"city": "Oslo"}])
                    self.assertEqual(len(requests), 2)
                    for request in requests:
                        self.assertEqual(request["stream_options"], {"include_usage": True})
                        if adapter == "minimax":
                            self.assertEqual(request["thinking"], {"type": "adaptive"})
                            self.assertIs(request["reasoning_split"], True)
                            self.assertEqual(request["reasoning_effort"], "low")
                        else:
                            self.assertIs(request["enable_thinking"], True)
                            for unsupported in ("thinking_display", "preserve_thinking", "reasoning_effort"):
                                self.assertNotIn(unsupported, request)
                    assistant, result = requests[1]["messages"][-2:]
                    self.assertEqual(assistant["role"], "assistant")
                    self.assertIsNone(assistant["content"])
                    expected_call = {"type": "function", "id": "call-fixture", "function": {
                        "name": "lookup", "arguments": '{"city":"Oslo"}',
                    }}
                    if adapter == "bailian":
                        expected_call["index"] = 0
                    self.assertEqual(assistant["tool_calls"], [expected_call])
                    if thought:
                        self.assertEqual(assistant["reasoning_content"], "fixture plan")
                    else:
                        self.assertNotIn("reasoning_content", assistant)
                    self.assertEqual(result, {"role": "tool", "tool_call_id": "call-fixture",
                                              "content": "fixture tool result"})
                    self.assertEqual([row.input_tokens for row in history.outcomes("main")], [7, 13])
                    self.assertEqual([row.model_status for row in history.outcomes("main")], ["complete", "complete"])
                    stored = [row.message for row in history.read_after("main") if row.message.kind == "assistant"]
                    self.assertEqual(reasoning_parts(stored[0])[0].payload if reasoning_parts(stored[0]) else None,
                                     b"fixture plan" if thought else None)
                    self.assertEqual(stored[1].content, "fixture answer")
                    self.assertEqual(reasoning_parts(stored[1])[0].payload, b"fixture final reasoning")

    async def test_compaction_and_reply_keep_compatible_reasoning_with_bailian_opt_out(self):
        for adapter, preserve in (("minimax", None), ("bailian", True), ("bailian", False)):
            with self.subTest(adapter=adapter, preserve=preserve), closing(History(":memory:")) as history:
                requests = []

                def wire(request):
                    requests.append(json.loads(request.content))
                    summary = requests[-1]["messages"][-1]["content"] == _SUMMARIZE.text
                    return httpx.Response(200, content=_answer("compressed context" if summary else "next answer"))

                async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
                    model = _model(client, adapter, preserve_thinking=preserve)
                    _seed, = history.append("main", [Message("user", "old question " + "x" * 500)])
                    end, = history.finalize_reply(ReplyOutcome("old", "main", "old", "complete", input_tokens=500), [
                        assistant_message("old answer", reasoning=PrivateData(model._replay_key, b"prior private reasoning")),
                    ])
                    runtime = Runtime("main", history, model, None,
                                      settings=RuntimeSettings(context_budget=500, tail_budget=75),
                                      instructions="role instruction", thinking=True)
                    try:
                        runtime.submit_messages([MessageDelivery(Message("user", "next question " + "y" * 300), "trigger")])
                        await asyncio.wait_for(runtime.wait_idle(), 2)
                        if task := runtime._compactions._generating:
                            await asyncio.wait_for(asyncio.shield(task), 2)
                        self.assertEqual(len(requests), 2)
                        for request in requests:
                            self.assertEqual(request["messages"][0], {"role": "system", "content": "role instruction"})
                            projected = next(entry for entry in request["messages"] if entry["content"] == "old answer")
                            self.assertEqual(projected.get("reasoning_content"),
                                             None if preserve is False else "prior private reasoning")
                        runtime.submit_messages([MessageDelivery(Message("user", "apply summary"), "trigger")])
                        await asyncio.wait_for(runtime.wait_idle(), 2)
                        self.assertEqual(history.summary("main"), Summary(end, "compressed context"))
                        self.assertEqual(requests[-1]["messages"][1]["role"], "user")
                        self.assertIn("compressed context", requests[-1]["messages"][1]["content"])
                    finally:
                        await runtime.close()

    async def test_bailian_namespaced_model_resumes_interleaved_tool_deltas_with_ordered_indexes(self):
        requests, calls = [], []
        first, second = {"city": "Oslo"}, {"city": "Tokyo"}

        async def lookup(arguments, context):
            calls.append(arguments)
            return arguments["city"]

        def wire(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 1:
                return _response(_choice({"tool_calls": [
                    {"index": 1, "id": "call-second", "type": "function", "function": {
                        "name": "lookup", "arguments": '{"city":',
                    }},
                    {"index": 0, "id": "call-first", "type": "function", "function": {
                        "name": "lookup", "arguments": json.dumps(first),
                    }},
                ]}), _choice({"tool_calls": [
                    {"index": 1, "function": {"arguments": '"Tokyo"}'}},
                ]}, finish="tool_calls"), b"[DONE]")
            return httpx.Response(200, content=_answer())

        with closing(History(":memory:")) as history:
            async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
                model = _model(client, "bailian", model="kimi/kimi-k3")
                runtime = Runtime("main", history, model, None, tools=Tools([Tool(_SPEC, lookup)]), thinking=True)
                try:
                    runtime.submit_messages([MessageDelivery(_INPUT, "trigger")])
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                finally:
                    await runtime.close()
                    await model.aclose()
            self.assertCountEqual(calls, [first, second])
            self.assertEqual([body["model"] for body in requests], ["kimi/kimi-k3"] * 2)
            assistant, *results = requests[1]["messages"][-3:]
            self.assertIsNone(assistant["content"])
            self.assertNotIn("reasoning_content", assistant)
            self.assertEqual([(call["index"], call["id"], json.loads(call["function"]["arguments"]))
                              for call in assistant["tool_calls"]],
                             [(0, "call-first", first), (1, "call-second", second)])
            self.assertEqual([(item["tool_call_id"], item["content"]) for item in results],
                             [("call-first", "Oslo"), ("call-second", "Tokyo")])
            self.assertEqual(history.read_after("main")[-1].message.content, "fixture answer")

    async def test_interleaved_parts_keep_positions_but_replay_calls_and_results_in_native_index_order(self):
        def delta(index, identity, city):
            return {"index": index, "id": identity, "function": {
                "name": "lookup", "arguments": json.dumps({"city": city}),
            }}

        async def lookup(arguments, context):
            return arguments["city"]

        for adapter in ("minimax", "bailian", "deepseek"):
            with self.subTest(adapter=adapter), closing(History(":memory:")) as history:
                requests = []

                def wire(request):
                    requests.append(json.loads(request.content))
                    if len(requests) == 1:
                        return _response(_choice({"tool_calls": [delta(3, "call-later", "Tokyo")]}),
                                         _choice({"content": "in between"}),
                                         _choice({"reasoning_content": "synthetic reasoning"}),
                                         _choice({"tool_calls": [delta(1, "call-earlier", "Oslo")]},
                                                 finish="tool_calls"), b"[DONE]")
                    return httpx.Response(200, content=_answer())

                async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
                    model = DeepSeek(client) if adapter == "deepseek" else _model(client, adapter)
                    runtime = Runtime("main", history, model, None, tools=Tools([Tool(_SPEC, lookup)]), thinking=True)
                    try:
                        runtime.submit_messages([MessageDelivery(_INPUT, "trigger")])
                        await asyncio.wait_for(runtime.wait_idle(), 2)
                    finally:
                        await runtime.close()
                        await model.aclose()

                stored = history.read_after("main")[1].message
                self.assertEqual(tuple(type(part) for part in stored.parts), (ToolCall, Text, Reasoning, ToolCall))
                self.assertEqual([call.id for call in stored.tool_calls], ["call-later", "call-earlier"])
                assistant, *results = requests[1]["messages"][-3:]
                self.assertEqual(assistant["content"], "in between")
                self.assertEqual(assistant["reasoning_content"], "synthetic reasoning")
                self.assertEqual([(call["id"], call.get("index")) for call in assistant["tool_calls"]],
                                 [("call-earlier", 1 if adapter == "bailian" else None),
                                  ("call-later", 3 if adapter == "bailian" else None)])
                self.assertEqual([(item["tool_call_id"], item["content"]) for item in results],
                                 [("call-earlier", "Oslo"), ("call-later", "Tokyo")])

    async def test_interleaved_view_image_reserves_media_for_first_outbound_call(self):
        workspace = image_workspace(self)
        first = save_image(workspace, PNG_BYTES)
        gif = b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
        later = save_image(workspace, gif)
        one_image = image_request_bytes(len(PNG_BYTES), "image/png")
        self.assertLessEqual(image_request_bytes(len(gif), "image/gif"), one_image)

        def delta(index, call_id, image):
            return {"index": index, "id": call_id, "function": {
                "name": VIEW_IMAGE_TOOL_NAME, "arguments": json.dumps({"path": image.url}),
            }}

        for adapter in ("minimax", "bailian", "deepseek"):
            with self.subTest(adapter=adapter), closing(History(":memory:")) as history:
                requests = []

                def wire(request):
                    requests.append(json.loads(request.content))
                    if len(requests) == 1:
                        return _response(_choice({"tool_calls": [delta(1, "call-one", later)]}),
                                         _choice({"content": "between"}),
                                         _choice({"reasoning_content": "fixture reasoning"}),
                                         _choice({"tool_calls": [delta(0, "call-zero", first)]}, finish="tool_calls"),
                                         b"[DONE]")
                    return httpx.Response(200, content=_answer())

                async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
                    model = DeepSeek(client) if adapter == "deepseek" else _model(client, adapter)
                    runtime = Runtime("main", history, model, None, tools=Tools([VIEW_IMAGE_TOOL]),
                                      workspace=workspace, thinking=True)
                    try:
                        with (patch.object(images, "MAX_MEDIA_BYTES", one_image),
                              patch.object(model, "_payload", wraps=model._payload) as payload):
                            runtime.submit_messages([MessageDelivery(_INPUT, "trigger")])
                            await asyncio.wait_for(runtime.wait_idle(), 2)
                        self.assertIsNone(runtime.last_failure)
                    finally:
                        await runtime.close()
                        await model.aclose()

                stored = history.read_after("main")[1].message
                self.assertEqual(tuple(type(part) for part in stored.parts), (ToolCall, Text, Reasoning, ToolCall))
                self.assertEqual([(call.id, call.result, call.result_image) for call in stored.tool_calls],
                                 [("call-one", VIEW_IMAGE_OK, None), ("call-zero", VIEW_IMAGE_OK, None)])
                projection = payload.call_args_list[1].args[0][1]
                self.assertEqual(tuple(type(part) for part in projection.parts), (ToolCall, Text, Reasoning, ToolCall))
                self.assertEqual(projection.parts[1:3], stored.parts[1:3])
                self.assertEqual([(call.id, call.result_image) for call in projection.tool_calls],
                                 [("call-one", None), ("call-zero", Image(PNG_URL))])
                assistant, *results, visual = requests[1]["messages"][1:]
                self.assertEqual(assistant["content"], "between")
                self.assertEqual(assistant["reasoning_content"], "fixture reasoning")
                self.assertEqual([call["id"] for call in assistant["tool_calls"]], ["call-zero", "call-one"])
                self.assertEqual(results, [{"role": "tool", "tool_call_id": call_id, "content": VIEW_IMAGE_OK}
                                           for call_id in ("call-zero", "call-one")])
                self.assertEqual(visual, {"role": "user", "content": [{"type": "image_url",
                                                                     "image_url": {"url": PNG_URL}}]})

    async def test_conflicting_missing_or_incompatible_indexes_fall_back_to_part_order(self):
        async with httpx.AsyncClient() as client:
            model = _model(client, "bailian")
            key = model._replay_key
            first = ToolCall("first", "lookup", "{}", "one", PrivateData(key, b"3"))
            for data in (None, PrivateData(key, b"3"), PrivateData(key, b"-1"), PrivateData(b"x" * len(key), b"1")):
                with self.subTest(data=data):
                    second = ToolCall("second", "lookup", "{}", "two", data)
                    assistant, *results = model._payload([Message("assistant", (first, Text("gap"), second))],
                                                          (), False)["messages"]
                    self.assertEqual([(call["id"], call["index"]) for call in assistant["tool_calls"]],
                                     [("first", 0), ("second", 1)])
                    self.assertEqual([item["tool_call_id"] for item in results], ["first", "second"])
            await model.aclose()

    async def test_replay_handles_key_change_invalid_payload_explicit_empty_and_multipart_input(self):
        requests = []

        def wire(request):
            requests.append(json.loads(request.content))
            return _response(_choice({"content": "ok", "reasoning_content": ""}, finish="stop"), b"[DONE]")

        async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
            mini = _model(client, "minimax", base_url="https://api.minimax.cn/v1/")
            bai = _model(client, "bailian")
            for adapter, model in (("minimax", mini), ("bailian", bai)):
                events = await _collect(model, thinking=False)
                self.assertEqual((events[0], events[-1]), (TextDelta("ok"), ModelClosing(complete=True)))
                self.assertEqual(len(events[1].parts[0].private.compatibility_key), 16)
                self.assertEqual(events[1].parts[0].private.payload, b"")
                source_key = events[1].parts[0].private.compatibility_key
                good = PrivateData(source_key, "思考".encode())
                messages = (Message("user", (Text("fixture text"), Image("https://example.invalid/photo"))),
                            assistant_message("history", (replace(ToolCall("a", "lookup", "{}"), result="fixture result"),), reasoning=good),
                            assistant_message("bad payload", reasoning=PrivateData(source_key, b"\xff")),
                            assistant_message("explicit empty", reasoning=events[1].parts[0].private))
                case_args = ([({"base_url": "https://API.MINIMAX.CN:443/v1", "api_key": "fixture-new"}, True),
                              ({"base_url": "https://api.minimax.io/v1"}, False)] if adapter == "minimax" else
                             [({"base_url": bailian_llm_base_url("ap-southeast-1", "other-space"),
                                "api_key": "fixture-new", "preserve_thinking": True}, True),
                              ({"preserve_thinking": False}, False)])
                case_args.append(({"model": "another-model"}, False))
                for opts, matches in case_args:
                    with self.subTest(adapter=adapter, opts=opts):
                        changed = _model(client, adapter, **opts)
                        try:
                            await _collect(changed, messages=messages, thinking=False)
                            request = requests[-1]
                            self.assertEqual(changed.history_replay_bytes(messages[1], (), thinking=False),
                                             len(good.payload) if matches else 0)
                            user, assistant, tool, bad, empty = request["messages"]
                            self.assertEqual([part["type"] for part in user["content"]], ["text", "image_url"])
                            self.assertEqual(assistant["content"], "history")
                            self.assertEqual(assistant["tool_calls"][0]["id"], tool["tool_call_id"])
                            self.assertEqual(assistant.get("reasoning_content"), "思考" if matches else None)
                            self.assertEqual(empty.get("reasoning_content"), "" if matches else None)
                            self.assertNotIn("reasoning_content", bad)
                            if adapter == "minimax":
                                self.assertEqual(request["thinking"], {"type": "disabled"})
                                self.assertNotIn("reasoning_effort", request)
                            else:
                                self.assertIs(request["enable_thinking"], False)
                                self.assertEqual(request.get("preserve_thinking"), opts.get("preserve_thinking"))
                                self.assertNotIn("thinking_display", request)
                        finally:
                            await changed.aclose()
                other = bai if adapter == "minimax" else mini
                self.assertEqual(other.history_replay_bytes(messages[1], (), thinking=True), 0)
                await model.aclose()

    async def test_terminal_validation_does_not_release_tool_or_replay_on_failure(self):
        for adapter in ("minimax", "bailian"):
            vendor = _BAD_MINIMAX if adapter == "minimax" else _BAD_BAILIAN
            for terminal, reason in (((), "truncated_stream"), ((vendor, b"[DONE]"), "vendor_error"),
                                     ((_choice(finish="length"), b"[DONE]"), "invalid_sse_event")):
                with self.subTest(adapter=adapter, reason=reason), closing(History(":memory:")) as history:
                    calls = []

                    async def lookup(arguments, context):
                        calls.append(arguments)
                        return "must not run"

                    wire = httpx.MockTransport(lambda _: _response(
                        _choice({"reasoning_content": "partial thought"}), _tool_start(), _tool_end(), *terminal))
                    async with httpx.AsyncClient(transport=wire) as client:
                        model = _model(client, adapter)
                        runtime = Runtime("main", history, model, None, tools=Tools([Tool(_SPEC, lookup)]), thinking=True)
                        try:
                            with self.assertLogs("agent_companion", "WARNING") as logs:
                                runtime.submit_messages([MessageDelivery(_INPUT, "trigger")])
                                await asyncio.wait_for(runtime.wait_idle(), 2)
                            if reason == "vendor_error":
                                self.assertEqual(sum("vendor model response failed" in log for log in logs.output), 1)
                        finally:
                            await runtime.close()
                            await model.aclose()
                    self.assertEqual(calls, [])
                    self.assertEqual([row.message for row in history.read_after("main")], [_INPUT])
                    self.assertEqual([row.model_error for row in history.outcomes("main")], [reason])

    async def test_bad_finishes_and_incomplete_arguments_never_publish_terminal_events(self):
        for adapter in ("minimax", "bailian"):
            for frames, reason in (([b"[DONE]"], "invalid_stream_finish"),
                                   ([_choice(finish="length"), b"[DONE]"], "abnormal_stream_finish"),
                                   ([_tool_start(), _choice(finish="stop"), b"[DONE]"], "invalid_stream_finish"),
                                   ([_tool_start(), _choice(finish="tool_calls"), b"[DONE]"],
                                    "invalid_tool_arguments")):
                with self.subTest(adapter=adapter, reason=reason):
                    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: _response(*frames))) as client:
                        model = _model(client, adapter)
                        events = []
                        try:
                            with self.assertRaisesRegex(ModelError, reason):
                                await _collect(model, events=events)
                            self.assertEqual(events, [])
                        finally:
                            await model.aclose()

    async def test_runtime_cancel_handles_pending_wire_and_ignores_late_vendor_frame(self):
        for adapter in ("minimax", "bailian"):
            reading, draining, interrupted = (asyncio.Event() for _ in range(3))
            vendor_error = _BAD_MINIMAX if adapter == "minimax" else _BAD_BAILIAN
            calls = []

            class Wire(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield _sse(_choice({"content": "confirmed", "reasoning_content": "private thought"}),
                               _tool_start(), _tool_end())
                    reading.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        # 模拟取消时已读到的错误帧；不能污染 Runtime 已截断的结果。
                        yield _sse(vendor_error)

                async def aclose(self):
                    draining.set()

            class Voice:
                async def deliver(self, reply):
                    reply.select_voice_delivery()
                    await anext(reply)
                    await interrupted.wait()
                    return Receipt(reply.info.reply_id, DeliveryStatus.PARTIAL, "confirmed")

            async def lookup(arguments, context):
                calls.append(arguments)
                return "must not run"

            wire = httpx.MockTransport(lambda _: httpx.Response(200, stream=Wire()))
            with self.subTest(adapter=adapter), closing(History(":memory:")) as history:
                async with httpx.AsyncClient(transport=wire) as client:
                    model = _model(client, adapter)
                    runtime = Runtime("main", history, model, Voice(), tools=Tools([Tool(_SPEC, lookup)]), thinking=True)
                    with self.assertNoLogs("agent_companion", "WARNING"):
                        try:
                            runtime.submit_messages([MessageDelivery(_INPUT, "trigger")])
                            await asyncio.wait_for(reading.wait(), 2)
                            interrupted.set()
                            await asyncio.wait_for(runtime.wait_idle(), 2)
                            await asyncio.wait_for(draining.wait(), 2)
                        finally:
                            await runtime.close()
                            await model.aclose()
                    self.assertEqual(calls, [])
                    self.assertEqual([row.message for row in history.read_after("main")],
                                     [_INPUT, Message("assistant", "confirmed")])
                    self.assertEqual([row.model_status for row in history.outcomes("main")], ["model_interrupted"])

    async def test_clean_tail_reuses_http11_connection_without_processing_late_sse(self):
        for adapter in ("minimax", "bailian"):
            terminal = _BAD_MINIMAX if adapter == "minimax" else _BAD_BAILIAN
            fixture = _HTTP11(_answer(), hold=True, tail=_sse(terminal))
            with (self.subTest(adapter=adapter), self.assertNoLogs(_LOGGER, "WARNING"),
                  patch.object(_http_stream, "_RESPONSE_TAIL_TIMEOUT", 2)):
                async with fixture.client as client:
                    model = _model(client, adapter)
                    try:
                        first = await asyncio.wait_for(_collect(model, runtime_style=True), 1)
                        await asyncio.wait_for(fixture.tail_started.wait(), 1)
                        self.assertEqual(first[0], TextDelta("fixture answer"))
                        self.assertEqual(first[-2:], [InputUsage(13), ModelClosing(complete=True)])
                        fixture.release.set()
                        await asyncio.wait_for(fixture.response_released.get(), 1)
                        self.assertEqual(await _collect(model, runtime_style=True), first)
                        await asyncio.wait_for(fixture.response_released.get(), 1)
                        self.assertEqual((fixture.connects, fixture.tls), (1, 1))
                    finally:
                        fixture.release.set()
                        await model.aclose()

    async def test_protocol_error_never_drains_http_tail(self):
        for adapter in ("minimax", "bailian"):
            fixture = _HTTP11(_sse(_tool_start(), _choice(finish="stop")), hold=True)
            with self.subTest(adapter=adapter):
                async with fixture.client as client:
                    model = _model(client, adapter)
                    try:
                        with self.assertRaisesRegex(ModelError, "invalid_stream_finish"):
                            await asyncio.wait_for(_collect(model), 1)
                        self.assertFalse(fixture.tail_started.is_set())
                    finally:
                        await model.aclose()


class ChatModelErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_known_http_business_and_sse_errors_keep_vendor_code_and_message(self):
        for adapter, payload, error in (("minimax", _BAD_MINIMAX, ("2013", 'thinking.type="disabled" is not allowed')),
                                         ("bailian", _BAD_BAILIAN, ("invalid_parameter", "Unsupported enable_thinking"))):
            for status, sse in ((400, False), (200, False), (200, True)):
                with self.subTest(adapter=adapter, status=status, sse=sse):
                    requests = []

                    def wire(request):
                        requests.append(json.loads(request.content))
                        if sse:
                            return _response(payload)
                        return httpx.Response(status, json=payload, headers={"x-request-id": "fixture-request"})

                    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
                        model = _model(client, adapter)
                        try:
                            diagnostics = ModelRequestDiagnostics("reply", 30)
                            reason = "http_400" if status == 400 else "vendor_error"
                            with self.assertNoLogs(_LOGGER, "WARNING"), diagnostics.activate():
                                with self.assertRaisesRegex(ModelError, reason) as failed:
                                    await _collect(model, thinking=False)
                            with self.assertLogs(_LOGGER, "WARNING") as logs:
                                diagnostics.failed(failed.exception)
                            log, = logs.output
                            self.assertIn(f"http_status={status}", log)
                            if adapter == "bailian" or not sse:
                                self.assertIn("request_id=fixture-request", log)
                            self.assertIn(f"code={error[0]}", log)
                            self.assertIn(error[1], log)
                            self.assertNotIn(error[1], str(failed.exception))
                            self.assertEqual(len(requests), 1)
                            if adapter == "minimax":
                                self.assertEqual(requests[0]["thinking"], {"type": "disabled"})
                        finally:
                            await model.aclose()

    async def test_runtime_logs_only_local_failure_details_for_requests_with_private_reasoning(self):
        thought = "synthetic reasoning containing 427 and x7q"
        remote_id, remote_code, remote_text = "remote-x7q", "code-x7q", "reflected: X7Q"
        headers = {"x-request-id": remote_id}

        async def lookup(arguments, context):
            return "synthetic tool result"

        for adapter in ("minimax", "bailian"):
            vendor = ({"base_resp": {"status_code": 427, "status_msg": remote_text}} if adapter == "minimax" else
                      {"error": {"code": remote_code, "message": remote_text}, "request_id": remote_id})
            for mode, status, reason in (("http", 400, "http_400"), ("json", 200, "vendor_error"),
                                         ("sse", 200, "vendor_error"), ("transport", None, "transport_error")):
                with self.subTest(adapter=adapter, mode=mode), closing(History(":memory:")) as history:
                    requests = []

                    def wire(request):
                        requests.append(json.loads(request.content))
                        if len(requests) == 1:
                            return _response(_choice({"reasoning_content": thought}), _tool_start(), _tool_end(),
                                             b"[DONE]")
                        if mode == "transport":
                            try:
                                raise ValueError(remote_text)
                            except ValueError as exc:
                                raise httpx.ReadError(remote_id) from exc
                        if mode == "sse":
                            return httpx.Response(200, content=b"event: error\n" + _sse(vendor), headers=headers)
                        return httpx.Response(status, json=vendor, headers=headers)

                    async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
                        model = _model(client, adapter)
                        runtime = Runtime("main", history, model, None, tools=Tools([Tool(_SPEC, lookup)]),
                                          thinking=True)
                        try:
                            with self.assertLogs("agent_companion", "DEBUG") as logs:
                                runtime.submit_messages([MessageDelivery(_INPUT, "trigger")])
                                await asyncio.wait_for(runtime.wait_idle(), 2)
                        finally:
                            await runtime.close()
                            await model.aclose()
                    self.assertEqual(len(requests), 2)
                    self.assertEqual(requests[1]["messages"][-2]["reasoning_content"], thought)
                    output = "\n".join(logs.output)
                    self.assertIn(f"reason={reason}", output)
                    if status is not None:
                        self.assertIn(f"http_status={status}", output)
                    for forbidden in (thought, remote_id, remote_code, remote_text, "request_id=", " code=", " message="):
                        self.assertNotIn(forbidden, output)
                    self.assertTrue(all(record.exc_info is None for record in logs.records))

    async def test_first_stream_private_deltas_suppress_error_echo_without_history(self):
        request_id, code, message = "rcv-x7q", "err-x7q", "echo: X7Q"
        for adapter in ("minimax", "bailian"):
            vendor = ({"base_resp": {"status_code": 7319, "status_msg": message}} if adapter == "minimax" else
                      {"error": {"code": code, "message": message}, "request_id": request_id})
            for delta in ({"reasoning_content": "synthetic reasoning x7q"}, {"tool_calls": [{"index": 7319}]}):
                for same_frame in (False, True):
                    with self.subTest(adapter=adapter, field=next(iter(delta)), same_frame=same_frame):
                        frame = _choice(delta)
                        frames = ({**frame, **vendor},) if same_frame else (frame, vendor)

                        def wire(request):
                            assistant_history = [item for item in json.loads(request.content)["messages"]
                                                 if item["role"] == "assistant"]
                            self.assertEqual(assistant_history, [])
                            return httpx.Response(200, content=_sse(*frames),
                                                  headers={"x-request-id": request_id})

                        with closing(History(":memory:")) as history:
                            async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
                                model = _model(client, adapter)
                                runtime = Runtime("main", history, model, None, thinking=True)
                                try:
                                    with self.assertLogs("agent_companion", "DEBUG") as logs:
                                        runtime.submit_messages([MessageDelivery(_INPUT, "trigger")])
                                        await asyncio.wait_for(runtime.wait_idle(), 2)
                                finally:
                                    await runtime.close()
                                    await model.aclose()
                            self.assertEqual([row.model_error for row in history.outcomes("main")], ["vendor_error"])
                            self.assertEqual([row.message for row in history.read_after("main")], [_INPUT])
                        self._assert_only_local_failure(logs, "vendor_error", 200, request_id, code, message)

    async def test_bailian_native_tool_index_suppresses_error_echo_without_reasoning(self):
        index = 7319
        request_id, code, message = f"rcv-{index}", f"err-{index}", f"echo: {index}"
        vendor = {"error": {"code": code, "message": message}, "request_id": request_id}
        headers = {"x-request-id": request_id}

        async def lookup(arguments, context):
            return "synthetic tool result"

        for mode, status, reason in (("http", 400, "http_400"), ("json", 200, "vendor_error"),
                                     ("sse", 200, "vendor_error")):
            with self.subTest(mode=mode), closing(History(":memory:")) as history:
                requests = []

                def wire(request):
                    requests.append(json.loads(request.content))
                    if len(requests) == 1:
                        return _response(_choice({"tool_calls": [{"index": index, "id": "call-lookup",
                                                                 "function": {"name": "lookup", "arguments": "{}"}}]},
                                                 finish="tool_calls"), b"[DONE]")
                    if mode == "sse":
                        return httpx.Response(status, content=b"event: error\n" + _sse(vendor), headers=headers)
                    return httpx.Response(status, json=vendor, headers=headers)

                async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
                    model = _model(client, "bailian", preserve_thinking=False)
                    runtime = Runtime("main", history, model, None, tools=Tools([Tool(_SPEC, lookup)]), thinking=False)
                    try:
                        with self.assertLogs("agent_companion", "DEBUG") as logs:
                            runtime.submit_messages([MessageDelivery(_INPUT, "trigger")])
                            await asyncio.wait_for(runtime.wait_idle(), 2)
                    finally:
                        await runtime.close()
                        await model.aclose()
                self.assertEqual(len(requests), 2)
                assistant = requests[1]["messages"][-2]
                self.assertEqual(assistant["tool_calls"][0]["index"], index)
                self.assertFalse(any("reasoning_content" in item for item in requests[1]["messages"]))
                self.assertEqual(history.outcomes("main")[-1].model_error, reason)
                self._assert_only_local_failure(logs, reason, status, request_id, code, message)

    def _assert_only_local_failure(self, logs, reason, status, *remote_fields):
        output = "\n".join(logs.output)
        self.assertIn(f"reason={reason}", output)
        self.assertIn(f"http_status={status}", output)
        for forbidden in (*remote_fields, "request_id=", " code=", " message="):
            self.assertNotIn(forbidden, output)
        self.assertTrue(all(record.exc_info is None for record in logs.records))

    async def test_unknown_shapes_fall_back_to_status_and_request_id(self):
        for adapter in ("minimax", "bailian"):
            for status, sse in ((502, False), (200, False), (200, True)):
                with self.subTest(adapter=adapter, status=status, sse=sse):
                    unknown = {"custom_status": "private response body", "message": "not a vendor error field"}
                    headers = {"x-request-id": "fixture-unknown"}
                    if sse:
                        response = httpx.Response(status, content=b"event: error\n" + _sse(unknown), headers=headers)
                    else:
                        response = httpx.Response(status, json=unknown, headers=headers)
                    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
                        model = _model(client, adapter)
                        try:
                            with self.assertLogs(_LOGGER, "WARNING") as logs, self.assertRaises(ModelError):
                                await _collect(model)
                            log, = logs.output
                            self.assertIn(f"adapter={adapter} http_status={status} request_id=fixture-unknown", log)
                            for excluded in ("code=", "message=", "private response body", "not a vendor error field"):
                                self.assertNotIn(excluded, log)
                        finally:
                            await model.aclose()

    async def test_inaccessible_error_body_still_reports_http_status_and_request_id(self):
        class Unreadable(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"{"
                raise httpx.ReadError("broken error response")

        for adapter in ("minimax", "bailian"):
            headers = {"x-request-id": "fixture-502"}
            wire = httpx.MockTransport(lambda _: httpx.Response(502, headers=headers, stream=Unreadable()))
            async with httpx.AsyncClient(transport=wire) as client:
                model = _model(client, adapter)
                with self.subTest(adapter=adapter):
                    try:
                        with self.assertLogs(_LOGGER, "WARNING") as logs, self.assertRaisesRegex(ModelError, "http_502"):
                            await _collect(model)
                        self.assertIn("http_status=502 request_id=fixture-502", logs.output[0])
                        self.assertNotIn("broken error response", logs.output[0])
                    finally:
                        await model.aclose()

    async def test_confirmed_credential_and_input_echo_are_not_logged(self):
        for adapter in ("minimax", "bailian"):
            tool_input = "fixture-private-tool-value"
            messages = (_INPUT, assistant_message("", (ToolCall("a", "lookup",
                                                                    json.dumps({"input": tool_input})),)))
            for secret in (_KEY, _INPUT.content, tool_input):
                with self.subTest(adapter=adapter):
                    echo = f"Invalid value: {secret}"
                    item = ({"base_resp": {"status_code": 2013, "status_msg": echo}} if adapter == "minimax"
                            else {"error": {"code": "invalid_parameter", "message": echo}})
                    wire = httpx.MockTransport(lambda _: httpx.Response(400, json=item,
                                                                        headers={"x-request-id": _KEY}))
                    async with httpx.AsyncClient(transport=wire) as client:
                        model = _model(client, adapter)
                        try:
                            with self.assertLogs(_LOGGER, "WARNING") as logs, self.assertRaises(ModelError):
                                await _collect(model, messages=messages)
                            self.assertIn("code=", logs.output[0])
                            self.assertNotIn(secret, logs.output[0])
                            self.assertNotIn("Invalid value", logs.output[0])
                            self.assertNotIn("request_id=", logs.output[0])
                        finally:
                            await model.aclose()

    async def test_partial_echoes_are_not_logged_or_rendered_in_tracebacks(self):
        prompt = Message("user", "synthetic-unique-prefix then a midsection with parcel-number-48153 inside")
        arguments = json.dumps({"data": [{"notification": "工作区编号雪杉林区\n内部标记",
                                          "contact": "box-743819@example.invalid",
                                          "雪域编号密钥条目": "fixture"}]}, ensure_ascii=True)
        self.assertNotIn("雪域编号密钥条目", arguments)
        tool = assistant_message("", (ToolCall("a", "lookup", arguments),))
        cases = (
            ("synthetic-unique-pr", "Rejected input: 'synthetic-unique-pr…'"),
            ("synt", "Rejected input: 'synt...'"),
            ("parcel-number-481", "Problem near parcel-number-481"),
            ("box-743819", "Invalid local value: box-743819"),
            ("雪域编号密钥条目", "Unsupported field 雪域编号密钥条目"),
            ("编号密钥", "Unsupported field 编号密钥"),
            (r"\u96ea\u6749\u6797\u533a\n", r"Invalid local value: \u96ea\u6749\u6797\u533a\n"),
            (r"\\u96EA\\u6749\\u6797\\u533A", r"Rejected input: \\u96EA\\u6749\\u6797\\u533A"),
            ("authorization-val", "Invalid token 'authorization-val…'"),
            (r"auth\u006frization", r"Bad credential snippet: auth\u006frization"),
        )
        for adapter in ("minimax", "bailian"):
            for number, (exposed, text) in enumerate(cases):
                for status in (400, 200):
                    with self.subTest(adapter=adapter, case=number, status=status):
                        item = ({"base_resp": {"status_code": 2013, "status_msg": text}} if adapter == "minimax"
                                else {"error": {"code": "invalid_parameter", "message": text}})
                        headers = {"x-request-id": "fixture-request"}
                        response = (httpx.Response(status, json=item, headers=headers) if status != 200 else
                                    httpx.Response(status, content=_sse(item), headers=headers))
                        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
                            model = _model(client, adapter)
                            try:
                                with self.assertLogs(_LOGGER, "WARNING") as logs:
                                    try:
                                        await _collect(model, messages=(prompt, tool))
                                    except ModelError as exc:
                                        trace = "".join(traceback.format_exception(exc))
                                        self.assertEqual(exc.diagnostic, "http_400" if status == 400 else "vendor_error")
                                        self.assertNotIn(exposed, trace)
                                        self.assertNotIn(text, trace)
                                    else:
                                        self.fail("vendor failure was not raised")
                                log, = logs.output
                                self.assertIn("request_id=fixture-request", log)
                                self.assertIn("code=2013" if adapter == "minimax" else "code=invalid_parameter", log)
                                self.assertNotIn(exposed, log)
                                self.assertNotIn("message=", log)
                            finally:
                                await model.aclose()

    async def test_structured_fields_check_format_and_known_partial_credentials(self):
        cases = (
            ("minimax", {"base_resp": {"status_code": 2**70, "status_msg": "Unsupported parameter"}}),
            ("bailian", {"error": {"code": "authorization-value", "message": "Unsupported parameter"},
                         "request_id": "not an id"}),
        )
        for adapter, item in cases:
            headers = {"x-request-id": "request-authorization-value-truncated"}
            response = httpx.Response(400, json=item, headers=headers)
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
                model = _model(client, adapter)
                try:
                    with self.subTest(adapter=adapter), self.assertLogs(_LOGGER, "WARNING") as logs:
                        with self.assertRaisesRegex(ModelError, "http_400"):
                            await _collect(model)
                    self.assertIn("Unsupported parameter", logs.output[0])
                    for forbidden in ("request_id=", "code=", "authorization-value", "not an id"):
                        self.assertNotIn(forbidden, logs.output[0])
                finally:
                    await model.aclose()

    async def test_structured_strings_apply_same_partial_echo_check_as_message(self):
        arguments = json.dumps({"nested": {"values": ["prefix-segmentmarker-suffix"]}})
        messages = (_INPUT, assistant_message("", (ToolCall("a", "lookup", arguments),)))
        credential_fragment = _KEY[9:19]
        self.assertEqual(len(credential_fragment), 10)
        for fragment in ("segmentmarker", credential_fragment):
            for adapter in ("minimax", "bailian"):
                with self.subTest(adapter=adapter, credential=fragment == credential_fragment):
                    request_id = f"req-{fragment}-not-a-complete-match"
                    code = f"code.{fragment}.extension"
                    item = ({"base_resp": {"status_code": 2013, "status_msg": "Unsupported parameter"}}
                            if adapter == "minimax" else
                            {"error": {"code": code, "message": "Unsupported parameter"}, "request_id": request_id})
                    headers = {"x-request-id": request_id}
                    response = httpx.Response(400, json=item, headers=headers)
                    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
                        model = _model(client, adapter)
                        try:
                            with self.assertLogs(_LOGGER, "WARNING") as logs:
                                with self.assertRaisesRegex(ModelError, "http_400"):
                                    await _collect(model, messages=messages)
                            log, = logs.output
                            self.assertIn("Unsupported parameter", log)
                            self.assertNotIn("request_id=", log)
                            self.assertNotIn(fragment, log)
                            if adapter == "minimax":
                                self.assertIn("code=2013", log)
                            else:
                                self.assertNotIn("code=", log)
                        finally:
                            await model.aclose()

    async def test_large_input_omits_unchecked_fields_including_structured_strings(self):
        marker = "late-field-marker-R81135"
        content = Message("user", (Image("data:image/jpeg;base64," + "X" * (130 * 1024)),))
        late = Message("user", marker)
        for adapter in ("minimax", "bailian"):
            item = ({"base_resp": {"status_code": 2013, "status_msg": f"Invalid value: {marker}"}}
                    if adapter == "minimax" else {"error": {"code": marker, "message": f"Invalid value: {marker}"},
                                                    "request_id": marker})
            response = httpx.Response(400, json=item, headers={"x-request-id": marker})
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
                model = _model(client, adapter)
                try:
                    with self.subTest(adapter=adapter), self.assertLogs(_LOGGER, "WARNING") as logs:
                        with self.assertRaisesRegex(ModelError, "http_400"):
                            await _collect(model, messages=(content, late))
                    self.assertIn("http_status=400", logs.output[0])
                    for excluded in (marker, "request_id=", "code=", "message="):
                        self.assertNotIn(excluded, logs.output[0])
                finally:
                    await model.aclose()

    async def test_minimax_common_phrase_and_short_numeric_tool_argument_do_not_hide_error(self):
        messages = (Message("user", "The website says access is not allowed; can you explain?"),
                    assistant_message("", (ToolCall("a", "lookup", '{"limit":1}'),)),
                    assistant_message("", (ToolCall("b", "lookup", '{"limit":12}'),)))
        for text in ('thinking.type="disabled" is not allowed',
                     'Model MiniMax-M3.1-Flash: thinking.type="disabled" is not allowed',
                     'Model MiniMax-M3.12-Flash: thinking.type="disabled" is not allowed'):
            for history in (*((message,) for message in messages), messages):
                with self.subTest(text=text, roles=[message.kind for message in history]):
                    item = {"base_resp": {"status_code": 2013, "status_msg": text}}
                    response = httpx.Response(400, json=item, headers={"x-request-id": "fixture-request"})
                    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
                        model = _model(client, "minimax")
                        try:
                            with self.assertLogs(_LOGGER, "WARNING") as logs:
                                with self.assertRaisesRegex(ModelError, "http_400"):
                                    await _collect(model, messages=history, thinking=False)
                            log, = logs.output
                            self.assertIn(f"message={text!r}", log)
                            self.assertIn("code=2013", log)
                            self.assertIn("request_id=fixture-request", log)
                        finally:
                            await model.aclose()

    async def test_unquoted_full_recipient_from_user_or_nested_tool_value_is_not_logged(self):
        recipient = "Kavil Zorven Mondralix"
        self.assertEqual(len(recipient), 22)
        arguments = json.dumps({"delivery": [{"recipient": recipient}]})
        sources = (Message("user", recipient), assistant_message("", (ToolCall("a", "lookup", arguments),)))
        text = f"Invalid recipient {recipient} in the request"
        for adapter in ("minimax", "bailian"):
            for source in sources:
                with self.subTest(adapter=adapter, source=source.kind):
                    item = ({"base_resp": {"status_code": 2013, "status_msg": text}} if adapter == "minimax"
                            else {"error": {"code": "invalid_parameter", "message": text}})
                    response = httpx.Response(400, json=item, headers={"x-request-id": "fixture-request"})
                    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
                        model = _model(client, adapter)
                        try:
                            with self.assertLogs(_LOGGER, "WARNING") as logs, self.assertRaises(ModelError):
                                await _collect(model, messages=(source,))
                            log, = logs.output
                            self.assertIn("request_id=fixture-request", log)
                            self.assertIn("code=2013" if adapter == "minimax" else "code=invalid_parameter", log)
                            self.assertNotIn("message=", log)
                            self.assertNotIn(recipient, log)
                        finally:
                            await model.aclose()

    async def test_minimax_only_exempts_bounded_current_model_in_vendor_message(self):
        public_model = "MiniMax-M3.1-Flash"
        similar_model = public_model + "-preview"
        recipient = "Kavil Zorven Mondralix"
        prompt = Message("user", f"Does {public_model} support this switch?")
        similar_prompt = Message("user", f"Does {similar_model} support this switch?")
        normal = f'Model {public_model}: thinking.type="disabled" is not allowed'
        cases = (
            ("public name only", (prompt,), normal, _KEY, "fixture-request", True, True),
            ("request id", (prompt,), normal, _KEY, public_model, True, False),
            ("near match in response", (prompt,), normal.replace(public_model, similar_model), _KEY,
             "fixture-request", False, True),
            ("near match in request", (similar_prompt,), normal, _KEY, "fixture-request", False, True),
            ("other field", (prompt, Message("user", recipient)),
             f"Model {public_model} rejected recipient {recipient} in the request", _KEY, "fixture-request", False, True),
            ("neighboring quoted echo", (Message("user", f"Keep {public_model} separate in this example"),),
             f"Unsupported selection '{public_model} separate from primary'", _KEY, "fixture-request", False, True),
            ("credential", (prompt,), normal, public_model, "fixture-request", False, True),
        )
        for label, history, text, api_key, request_id, show_message, show_request_id in cases:
            with self.subTest(case=label):
                item = {"base_resp": {"status_code": 2013, "status_msg": text}}

                def wire(request):
                    self.assertEqual(json.loads(request.content)["model"], public_model)
                    return httpx.Response(400, json=item, headers={"x-request-id": request_id})

                async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
                    model = _model(client, "minimax", api_key=api_key, model=public_model)
                    try:
                        with self.assertLogs(_LOGGER, "WARNING") as logs, self.assertRaises(ModelError):
                            await _collect(model, messages=history, thinking=False)
                        log, = logs.output
                        self.assertIn("code=2013", log)
                        if show_message:
                            self.assertIn(f"message={text!r}", log)
                        else:
                            self.assertNotIn("message=", log)
                        self.assertEqual("request_id=" in log, show_request_id)
                        self.assertNotIn(recipient, log)
                    finally:
                        await model.aclose()
