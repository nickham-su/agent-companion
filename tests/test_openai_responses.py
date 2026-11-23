import asyncio
from contextlib import closing
from io import StringIO
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from agent_companion.__main__ import _serve, main
from agent_companion.config import FeishuSettings, LLMSettings, RuntimeSettings, Settings, load_settings
from agent_companion.models.deepseek import CHAT_COMPLETIONS, DeepSeek
from agent_companion.models.openai_responses import RESPONSES, OpenAIResponses
from agent_companion.protocol import (OMITTED_INPUT_IMAGE, DeliveryStatus, Image, Message, Receipt,
                                      ReplayReady, Text, ToolSpec)
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.storage.sqlite import History, ReplyOutcome, Summary


_CIPHER = "synthetic-opaque-block"
_PRIVATE_SUMMARY = "synthetic-provider-summary"
_IMAGE = "data:image/png;base64,YQ=="
_SPEC = ToolSpec("delegate_subtask", "synthetic local test function", {"type": "object"})


def _reason(item_id="rs_1", cipher=_CIPHER, *, summary=()):
    return {"type": "reasoning", "id": item_id, "encrypted_content": cipher,
            "summary": [{"type": "summary_text", "text": part} for part in summary]}


def _answer(body, identity="msg_1", *, phase="final_answer"):
    return {"type": "message", "id": identity, "role": "assistant", "status": "completed", "phase": phase,
            "content": [{"type": "output_text", "text": body, "annotations": []}]}


def _call(num=1, *, arguments='{"value":1}'):
    return {"type": "function_call", "id": f"fc_{num}", "call_id": f"call_{num}",
            "name": _SPEC.name, "arguments": arguments, "status": "completed"}


def _delta(body, output=0, part=0):
    return {"type": "response.output_text.delta", "output_index": output, "content_index": part, "delta": body}


def _completed(*items):
    return {"type": "response.completed", "response": {"status": "completed", "output": list(items)}}


class _Wire(httpx.AsyncByteStream):
    def __init__(self, events, *, block_before_end=None, close_error=False):
        self.events = events
        self.block_before_end = block_before_end
        self.close_error = close_error

    async def __aiter__(self):
        for index, event in enumerate(self.events):
            if self.block_before_end is not None and index == len(self.events) - 1:
                await self.block_before_end.wait()
            line = (f"event: {event['type']}\r\ndata: " + json.dumps(event, ensure_ascii=False) + "\r\n\r\n").encode()
            # 拆分 HTTP chunk，验证 SSE 不假定一次读就是一行。
            middle = len(line) - 7
            yield line[:middle]
            yield line[middle:]

    async def aclose(self):
        if self.close_error:
            raise httpx.ReadError("synthetic transport close failure")


class _Fixture:
    def __init__(self, testcase, *responses):
        self._test = testcase
        self._responses = iter(responses)
        self.requests = []
        self.transport = httpx.MockTransport(self._request)

    async def _request(self, request):
        self._test.assertEqual((request.method, str(request.url)), ("POST", RESPONSES))
        self._test.assertEqual(request.headers["accept-encoding"], "identity")
        payload = json.loads(request.content)
        self.requests.append(payload)
        response = next(self._responses)
        if callable(response):
            response = response(payload)
        if not isinstance(response, _Wire):
            response = _Wire(response)
        return httpx.Response(200, headers={"Content-Type": "text/event-stream; charset=utf-8"}, stream=response)


class _Channel:
    def __init__(self, first_confirmed=None):
        self.first_confirmed = first_confirmed
        self.replies = []
        self.first_text = asyncio.Event()

    async def deliver(self, reply):
        chunks = []
        async for chunk in reply:
            chunks.append(chunk)
            self.first_text.set()
        generated = "".join(chunks)
        is_first = not self.replies
        self.replies.append(generated)
        if is_first and self.first_confirmed is not None:
            status = DeliveryStatus.PARTIAL if self.first_confirmed else DeliveryStatus.ZERO
            return Receipt(reply.info.reply_id, status, self.first_confirmed)
        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, generated)


def _model(client, *, profile="local-a", model="fixture-model", effort="custom-effort", **kwargs):
    return OpenAIResponses(client, model=model, replay_profile=profile, reasoning_effort=effort, **kwargs)


async def _complete(fixture, history, *, prompt=None, channel=None, tools=None, settings=None, session_id="main",
                    **model_settings):
    async with httpx.AsyncClient(transport=fixture.transport) as client:
        runtime = Runtime(session_id, history, _model(client, **model_settings), channel or _Channel(),
                          tools=tools, settings=settings)
        try:
            runtime.submit_batch([prompt or Message("user", "synthetic prompt")])
            await asyncio.wait_for(runtime.wait_idle(), 2)
            return runtime.last_failure
        finally:
            await runtime.close()


class OpenAIResponsesTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_order_call_ids_and_text_stream_before_terminal(self):
        gate = asyncio.Event()
        channel = _Channel()
        counts = []

        async def tool(arguments, ctx):
            counts.append((ctx.call_id, arguments))
            return f"result {ctx.call_id}"

        reasoning_first = _reason(summary=(_PRIVATE_SUMMARY,))
        reasoning_last = _reason("rs_2", "second-opaque", summary=())
        commentary = _answer("读取中。", "msg_commentary", phase="commentary")
        final = _answer("准备完成。", "msg_more")
        first = _Wire([_delta("读取中。", 1), _delta("准备完成。", 5), _completed(
            reasoning_first, commentary, _call(), reasoning_last, _call(2, arguments='{"value":2}'), final,
        )], block_before_end=gate)
        fixture = _Fixture(self, first, [_delta("已完成。"), _completed(_answer("已完成。", "msg_done"))])
        tools = Tools([Tool(_SPEC, tool)])
        prompt = Message("user", (Text("读图"), Image(_IMAGE), Text("使用工具")))
        with closing(History(":memory:")) as history:
            async with httpx.AsyncClient(transport=fixture.transport) as client:
                runtime = Runtime("main", history, _model(client), channel, tools=tools)
                try:
                    runtime.submit_batch([prompt])
                    await asyncio.wait_for(channel.first_text.wait(), 1)
                    self.assertEqual(counts, [])
                    self.assertEqual(history.outcomes("main"), [])
                    gate.set()
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                    self.assertIsNone(runtime.last_failure)
                finally:
                    gate.set()
                    await runtime.close()
            self.assertEqual(counts, [("call_1", {"value": 1}), ("call_2", {"value": 2})])
            self.assertEqual(channel.replies, ["读取中。准备完成。", "已完成。"])
            self.assertEqual([r.message.content for r in history.search("main", keyword="synthetic-provider-summary")], [])
            self.assertTrue(all(row.message.replay is None for row in history.read_after("main")))
            self.assertIsNone(history.read_after("main")[1].message.reasoning)
            self.assertEqual(len(history.outcomes("main")), 2)
        initial, continued = fixture.requests
        self.assertTrue(initial["stream"])
        self.assertFalse(initial["store"])
        self.assertEqual(initial["include"], ["reasoning.encrypted_content"])
        self.assertEqual(initial["reasoning"], {"effort": "custom-effort"})
        self.assertNotIn("conversation", continued)
        self.assertNotIn("previous_response_id", continued)
        self.assertEqual(initial["tools"], [{"type": "function", "name": _SPEC.name, "description": _SPEC.description,
                                             "parameters": {"type": "object"}, "strict": False}])
        self.assertEqual(initial["input"], [{"role": "user", "content": [
            {"type": "input_text", "text": "读图"}, {"type": "input_image", "image_url": _IMAGE, "detail": "auto"},
            {"type": "input_text", "text": "使用工具"},
        ]}])
        expected_calls = [{k: v for k, v in call.items() if k != "status"}
                          for call in (_call(), _call(2, arguments='{"value":2}'))]
        self.assertEqual(continued["input"], initial["input"] + [
            reasoning_first, commentary, expected_calls[0], reasoning_last, expected_calls[1], final,
            {"type": "function_call_output", "call_id": "call_1", "output": "result call_1"},
            {"type": "function_call_output", "call_id": "call_2", "output": "result call_2"},
        ])

    async def test_reasoning_only_reopens_same_profile_but_not_other_scopes(self):
        for profile, model, should_replay in [("local-a", "fixture-model", True),
                                              ("local-b", "fixture-model", False),
                                              ("local-a", "other-model", False)]:
            with self.subTest(profile=profile, model=model), TemporaryDirectory(dir=Path.cwd()) as folder:
                path = Path(folder) / "fixture.sqlite"
                native = _reason(summary=(_PRIVATE_SUMMARY,))
                added = {"type": "response.output_item.added", "output_index": 0,
                         "item": _reason(cipher="must-not-return-added")}
                done = {"type": "response.output_item.done", "output_index": 0, "item": native}
                first = _Fixture(self, [added, done, _completed(_reason(cipher=None, summary=(_PRIVATE_SUMMARY,)))])
                channel = _Channel()
                with closing(History(path)) as history:
                    self.assertIsNone(await _complete(first, history, channel=channel))
                    self.assertEqual(channel.replies, [])
                    self.assertEqual(len(history.read_after("main")), 2)
                    outcome = history.outcomes("main")[0]
                    self.assertIsNone(history.run_text("main", outcome.run_id))
                    self.assertIsNone(outcome.delivery_status)
                second = _Fixture(self, [_completed(_answer("完成"))])
                with closing(History(path)) as history:
                    self.assertIsNone(await _complete(second, history, profile=profile, model=model))
                expected = ([native] if should_replay else [])
                self.assertEqual(second.requests[0]["input"], [first.requests[0]["input"][0], *expected,
                                                               {"role": "user", "content": "synthetic prompt"}])

    async def test_effort_change_keeps_common_history_without_old_native_items(self):
        fixture = _Fixture(self, [_completed(_reason(), _answer("confirmed"))], [_completed(_answer("next"))])
        with closing(History(":memory:")) as history:
            self.assertIsNone(await _complete(fixture, history, effort="low"))
            self.assertIsNone(await _complete(fixture, history, effort="another-valid-for-fixture"))
        first, second = fixture.requests
        self.assertEqual(first["reasoning"], {"effort": "low"})
        self.assertEqual(second["reasoning"], {"effort": "another-valid-for-fixture"})
        self.assertEqual(second["input"], first["input"] + [
            {"role": "assistant", "content": "confirmed"}, {"role": "user", "content": "synthetic prompt"},
        ])

    async def test_corrupt_replay_and_cross_provider_use_only_common_history(self):
        for corrupt in (True, False):
            with self.subTest(corrupt=corrupt), TemporaryDirectory(dir=Path.cwd()) as folder:
                path = Path(folder) / "fixture.sqlite"
                with closing(History(path)) as history:
                    fixture = _Fixture(self, [_completed(_reason(summary=(_PRIVATE_SUMMARY,)), _answer("通用正文"))])
                    self.assertIsNone(await _complete(fixture, history))
                if corrupt:
                    with closing(sqlite3.connect(path)) as conn, conn:
                        conn.execute("UPDATE provider_replays SET items = ?", ('{"malformed":"' + _CIPHER + '"}',))
                    second = _Fixture(self, [_completed(_answer("完成"))])
                    with closing(History(path)) as history:
                        self.assertIsNone(await _complete(second, history))
                    inputs = second.requests[0]["input"]
                else:
                    captured = []

                    async def packet(request):
                        self.assertEqual(str(request.url), CHAT_COMPLETIONS)
                        captured.append(json.loads(request.content))
                        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}]}\n\n'
                                                       'data: [DONE]\n\n')

                    async with httpx.AsyncClient(transport=httpx.MockTransport(packet)) as client:
                        with closing(History(path)) as history:
                            runtime = Runtime("main", history, DeepSeek(client), _Channel())
                            try:
                                runtime.submit_batch([Message("user", "切换")])
                                await asyncio.wait_for(runtime.wait_idle(), 2)
                                self.assertIsNone(runtime.last_failure)
                            finally:
                                await runtime.close()
                    inputs = captured[0]["messages"]
                self.assertEqual(inputs[1], {"role": "assistant", "content": "通用正文"})
                self.assertNotIn(_CIPHER, json.dumps(inputs))
                self.assertNotIn(_PRIVATE_SUMMARY, json.dumps(inputs))

    async def test_partial_and_zero_delivery_retain_required_tool_replay_without_unseen_text(self):
        async def tool(_arguments, _ctx):
            return "accepted"

        for confirmed in ("开头中", ""):
            with self.subTest(confirmed=confirmed), closing(History(":memory:")) as history:
                fixture = _Fixture(self, [_completed(_reason(), _answer("开头", "msg_a", phase="commentary"),
                                                    _call(), _answer("中间未播出", "msg_b"))],
                                   [_completed(_answer("已解决", "msg_result"))],
                                   [_completed(_answer("下一轮", "msg_next"))])
                channel = _Channel(first_confirmed=confirmed)
                tools = Tools([Tool(_SPEC, tool)])
                self.assertIsNone(await _complete(fixture, history, channel=channel, tools=tools))
                self.assertIsNone(await _complete(fixture, history, channel=channel, tools=tools))
                _, continuation, later = fixture.requests
                self.assertEqual(history.read_after("main")[1].message.content, confirmed)
                inputs = continuation["input"]
                self.assertIn(_reason(), inputs)
                self.assertEqual(inputs[-1]["call_id"], "call_1")
                self.assertNotIn("未播出", json.dumps(inputs, ensure_ascii=False))
                self.assertNotIn("msg_b", json.dumps(inputs))
                if confirmed:
                    self.assertIn(_answer("开头", "msg_a", phase="commentary"), inputs)
                    self.assertIn({"role": "assistant", "content": "中"}, inputs)
                else:
                    self.assertNotIn("msg_a", json.dumps(inputs))
                    self.assertFalse(any(item.get("role") == "assistant" for item in inputs))
                self.assertEqual(later["input"][:len(inputs)], inputs)
                self.assertNotIn("未播出", json.dumps(later, ensure_ascii=False))

    async def test_text_rewrite_cuts_later_reasoning_in_tool_group(self):
        early, late = _reason(), _reason("rs_2", "synthetic-late-opaque")
        answer = _answer("ABCDEF")
        for confirmed in ("ABC", "", None):
            with self.subTest(confirmed=confirmed), closing(History(":memory:")) as history:
                tool = AsyncMock(return_value="accepted")
                tools = Tools([Tool(_SPEC, tool)])
                channel = _Channel(first_confirmed=confirmed)
                packet = [_completed(early, answer, _call(), late, _call(2))]
                if confirmed is None:
                    packet_fixture = _Fixture(self, packet, [_completed(_answer("continued", "msg_done"))])
                    self.assertIsNone(await _complete(packet_fixture, history, channel=channel, tools=tools))
                else:
                    packet_fixture = _Fixture(self, packet)
                    self.assertEqual(await _complete(packet_fixture, history, channel=channel, tools=tools), "model_failed")
                    self.assertEqual(history.outcomes("main")[-1].model_error, "responses_required_replay_unavailable")
                    self.assertEqual(len(packet_fixture.requests), 1)
                self.assertEqual([call.args[1].call_id for call in tool.await_args_list], ["call_1", "call_2"])
                # 随后新话轮仍可使用可靠前缀和通用尾部，不重新执行历史工具。
                later = _Fixture(self, [_completed(_answer("later", "msg_later"))])
                self.assertIsNone(await _complete(later, history, tools=tools))
                calls = [{k: v for k, v in _call(n).items()
                          if k != "status" and (confirmed is None or k != "id")} for n in (1, 2)]
                ordered = ([early, answer, calls[0], late, calls[1]] if confirmed is None else
                           [early, *([{"role": "assistant", "content": confirmed}] if confirmed else []), *calls])
                expected = packet_fixture.requests[0]["input"] + ordered + [
                    {"type": "function_call_output", "call_id": f"call_{n}", "output": "accepted"} for n in (1, 2)
                ]
                if confirmed is None:
                    self.assertEqual(packet_fixture.requests[1]["input"], expected)
                self.assertEqual(later.requests[0]["input"][:len(expected)], expected)
                self.assertEqual(tool.await_count, 2)

    async def test_incomplete_and_malformed_terminal_do_not_dispatch_or_save_replay(self):
        valid = _completed(_reason(), _answer("可见前缀"), _call())
        variants = [([_delta("可见前缀"), {"type": "response.failed"}], "abnormal_stream_finish"),
                    ([_delta("可见前缀"), {"type": "response.incomplete"}], "abnormal_stream_finish"),
                    ([_delta("可见前缀"), {"type": "error"}], "abnormal_stream_finish"),
                    ([_delta("可见前缀"), {"type": "response.output_item.done", "output_index": 1, "item": _reason()}],
                     "truncated_stream"),
                    ([_delta("可见前缀"), _completed(_answer("可见前缀"), _call(), _call(2, arguments='{"value":NaN}'))],
                     "invalid_tool_arguments"),
                    ([_delta("可见前缀"), _completed(_answer("可见前缀"), _call(), {**_call(2), "call_id": "call_1"})],
                     "duplicate_tool_id"),
                    ([_delta("可见前缀"), _completed(_answer("可见前缀"), _reason(cipher=None), _call())],
                     "responses_missing_encrypted_reasoning"),
                    (_Wire([_delta("可见前缀"), valid], close_error=True), "transport_error")]
        for packet, error in variants:
            with self.subTest(error=error), closing(History(":memory:")) as history:
                tool = AsyncMock(return_value="must not call")
                fixture = _Fixture(self, packet)
                self.assertEqual(await _complete(fixture, history, tools=Tools([Tool(_SPEC, tool)])), "model_failed")
                tool.assert_not_awaited()
                self.assertEqual(history.outcomes("main")[0].model_error, error)
                self.assertEqual(history.read_after("main")[1].message.content, "可见前缀")
                self.assertEqual(history._db.execute("SELECT count(*) FROM provider_replays").fetchone()[0], 0)

    async def test_prefix_changes_and_compaction_drop_old_replay_but_keep_common_tool_pair(self):
        async def tool(_arguments, _ctx):
            return "accepted"

        tools = Tools([Tool(_SPEC, tool)])
        fixture = _Fixture(self, [_completed(_reason(), _call())], [_completed(_answer("ended", "msg_done"))],
                           [_completed(_answer("next", "msg_next"))])
        with closing(History(":memory:")) as history:
            prompt = Message("user", (Text("initial"), Image(_IMAGE)))
            self.assertIsNone(await _complete(fixture, history, prompt=prompt, tools=tools))
            first = history.read_after("main")[0].message_id
            # 模拟压缩在旧工具组前改变了实际请求前缀。该组不应被拆开。
            history.save_summary("main", Summary(first, "previous input compacted"))
            self.assertIsNone(await _complete(fixture, history, tools=tools))
            inputs = fixture.requests[-1]["input"]
            self.assertEqual([i.get("type") for i in inputs], [None, "function_call", "function_call_output", None, None])
            self.assertNotIn(_CIPHER, json.dumps(inputs))
            self.assertNotIn("fc_1", json.dumps(inputs))
            self.assertEqual(inputs[1]["call_id"], inputs[2]["call_id"])

    async def test_continuation_preserves_actual_sent_prefix_and_adds_queued_input(self):
        settings = RuntimeSettings(context_budget=240, tail_budget=75, summary_budget=45)
        result = "r" * 285
        fixture = _Fixture(self, [_completed(_reason(), _call())], [_completed(_answer("done"))])
        channel = _Channel()
        with closing(History(":memory:")) as history:
            # 原记录在估算预算内，但转义后的 JSON 在首轮放不下。
            history.append("main", [Message("user", '"' * 360)])

            async def receive_at_tool_boundary(_arguments, _ctx):
                runtime.submit_batch([Message("user", "queued")])
                return result

            tool = AsyncMock(side_effect=receive_at_tool_boundary)
            async with httpx.AsyncClient(transport=fixture.transport) as client:
                runtime = Runtime("main", history, _model(client, request_limit=1000), channel,
                                  settings=settings, tools=Tools([Tool(_SPEC, tool)]))
                try:
                    runtime.submit_batch([Message("user", "prompt")])
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                    self.assertIsNone(runtime.last_failure)
                finally:
                    await runtime.close()
            self.assertIsNone(history.summary("main"))
        tool.assert_awaited_once()
        self.assertEqual(channel.replies, ["done"])
        first, next_ = fixture.requests
        self.assertEqual(first["input"], [{"role": "user", "content": "prompt"}])
        self.assertEqual(next_["input"], first["input"] + [
            _reason(), {k: v for k, v in _call().items() if k != "status"},
            {"type": "function_call_output", "call_id": "call_1", "output": result},
            {"role": "user", "content": "queued"},
        ])

    async def test_no_reasoning_continuation_can_compact_or_replace_images(self):
        for mode in ("compact", "image"):
            with self.subTest(mode=mode), closing(History(":memory:")) as history:
                tool = AsyncMock(return_value="r" * 285)
                initial, final = [_completed(_call())], [_completed(_answer("done"))]
                if mode == "compact":
                    history.append("main", [Message("user", "x" * 360)])
                    fixture = _Fixture(self, initial, [_completed(_answer("summary"))], final)
                    settings = RuntimeSettings(context_budget=240, tail_budget=75, summary_budget=45)
                    self.assertIsNone(await _complete(fixture, history, settings=settings,
                                                      tools=Tools([Tool(_SPEC, tool)])))
                    self.assertIsNotNone(history.summary("main"))
                    self.assertEqual(len(fixture.requests), 3)
                    self.assertEqual(fixture.requests[-1]["input"][0]["role"], "system")
                    self.assertNotIn("x" * 360, json.dumps(fixture.requests[-1]))
                else:
                    image = Image("data:image/png;base64," + "YQAA" * 100)
                    prompt = Message("user", (Text("image"), image))
                    fixture = _Fixture(self, initial, final)
                    self.assertIsNone(await _complete(fixture, history, prompt=prompt, request_limit=950,
                                                      tools=Tools([Tool(_SPEC, tool)])))
                    self.assertEqual(fixture.requests[0]["input"][0]["content"][1]["image_url"], image.url)
                    self.assertNotIn("input_image", json.dumps(fixture.requests[1]))
                    self.assertEqual(history.read_after("main")[0].message, prompt)
                tool.assert_awaited_once()
                self.assertEqual(fixture.requests[-1]["input"][-2:], [
                    {k: v for k, v in _call().items() if k not in ("id", "status")},
                    {"type": "function_call_output", "call_id": "call_1", "output": "r" * 285},
                ])

    async def test_budget_downgrade_then_required_replay_with_queued_input(self):
        image = Image("data:image/png;base64," + "YQAA" * 225)
        prompt = Message("user", (Text("image"), image))
        first_result, second_result = "r" * 285, "second result"
        fixture = _Fixture(self, [_completed(_call())],
                           [_completed(_reason(), _answer("ABCDEF"), _call(2))],
                           [_completed(_answer("done", "msg_done"))])
        channel = _Channel(first_confirmed="ABC")

        async def execute(_arguments, ctx):
            if ctx.call_id == "call_1":
                return first_result
            runtime.submit_batch([Message("user", "queued")])
            return second_result

        tool = AsyncMock(side_effect=execute)
        with closing(History(":memory:")) as history:
            async with httpx.AsyncClient(transport=fixture.transport) as client:
                runtime = Runtime("main", history, _model(client, request_limit=1400), channel,
                                  tools=Tools([Tool(_SPEC, tool)]))
                try:
                    runtime.submit_batch([prompt])
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                    self.assertIsNone(runtime.last_failure)
                finally:
                    await runtime.close()
            self.assertEqual([row.message.content for row in history.read_after("main")
                              if row.message.role == "assistant"], ["", "ABC", "done"])
        self.assertEqual([call.args[1].call_id for call in tool.await_args_list], ["call_1", "call_2"])
        first, downgraded, continued = (request["input"] for request in fixture.requests)
        self.assertEqual(first[0]["content"][1]["image_url"], image.url)
        calls = [{k: v for k, v in _call(n).items() if k not in ("id", "status")} for n in (1, 2)]
        self.assertEqual(downgraded, [
            {"role": "user", "content": [{"type": "input_text", "text": text}
                                         for text in ("image", OMITTED_INPUT_IMAGE.text)]},
            calls[0], {"type": "function_call_output", "call_id": "call_1", "output": first_result},
        ])
        self.assertEqual(continued, downgraded + [
            _reason(), {"role": "assistant", "content": "ABC"}, calls[1],
            {"type": "function_call_output", "call_id": "call_2", "output": second_result},
            {"role": "user", "content": "queued"},
        ])

    async def test_required_replay_over_either_budget_never_sends_or_reruns_tools(self):
        settings = RuntimeSettings(context_budget=240, tail_budget=75, summary_budget=45)
        for old, limit, failure in [("x" * 360, 2000, "protected_context_too_large"),
                                    ("", 700, "protected_request_too_large")]:
            with self.subTest(failure=failure), closing(History(":memory:")) as history:
                if old:
                    history.append("main", [Message("user", old)])
                tool = AsyncMock(return_value="r" * 285)
                fixture = _Fixture(self, [_completed(_reason(), _call())])
                with self.assertLogs("agent_companion.runtime.engine", level="WARNING") as logs:
                    self.assertEqual(await _complete(fixture, history, settings=settings, request_limit=limit,
                                                     tools=Tools([Tool(_SPEC, tool)])), failure)
                tool.assert_awaited_once()
                self.assertEqual(len(fixture.requests), 1)
                self.assertIsNone(history.summary("main"))
                self.assertEqual(history.read_after("main")[-1].message.tool_call_id, "call_1")
                self.assertEqual([record.getMessage() for record in logs.records], [
                    f"tool continuation not sent ({failure}); run={history.outcomes('main')[0].run_id}",
                ])

    async def test_missing_or_damaged_active_replay_does_not_prove_no_reasoning(self):
        for missing in (True, False):
            with self.subTest(missing=missing), closing(History(":memory:")) as history:
                tool = AsyncMock(return_value="accepted")
                fixture = _Fixture(self, [_completed(_reason(), _call())])
                finalize = history.finalize_reply

                def damage_after_commit(outcome, rows):
                    ids = finalize(outcome, rows)
                    if ids:
                        with history._db:
                            if missing:
                                history._db.execute("DELETE FROM provider_replays WHERE assistant_message_id = ?", (ids[0],))
                            else:
                                stored = history._db.execute("SELECT items FROM provider_replays"
                                                             " WHERE assistant_message_id = ?", (ids[0],)).fetchone()
                                data = json.loads(stored[0])
                                data["items"].pop(0)  # 终态标记仍表明有推理，不能误作无推理。
                                history._db.execute("UPDATE provider_replays SET items = ? WHERE assistant_message_id = ?",
                                                    (json.dumps(data), ids[0]))
                    return ids

                with patch.object(history, "finalize_reply", side_effect=damage_after_commit):
                    with self.assertLogs("agent_companion.runtime.engine", level="WARNING") as logs:
                        self.assertEqual(await _complete(fixture, history, tools=Tools([Tool(_SPEC, tool)])), "model_failed")
                tool.assert_awaited_once()
                self.assertEqual(len(fixture.requests), 1)
                outcome = history.outcomes("main")[-1]
                self.assertEqual(outcome.model_error, "responses_required_replay_unavailable")
                self.assertEqual([record.getMessage() for record in logs.records], [
                    f"tool continuation not sent (responses_required_replay_unavailable); run={outcome.run_id}",
                ])

    async def test_summary_request_never_receives_private_fields(self):
        fixture = _Fixture(self, [_completed(_reason(summary=(_PRIVATE_SUMMARY,)), _answer("old"))],
                           [_completed(_answer("compressed"))], [_completed(_answer("new response"))])
        with closing(History(":memory:")) as history:
            self.assertIsNone(await _complete(fixture, history, prompt=Message("user", "prior " + "x" * 330)))
            settings = RuntimeSettings(context_budget=220, tail_budget=55, summary_budget=45)
            self.assertIsNone(await _complete(fixture, history, settings=settings,
                                              prompt=Message("user", "current " + "y" * 290)))
            self.assertEqual(len(fixture.requests), 3)
            self.assertIsNotNone(history.summary("main"))
            for request in fixture.requests[1:]:
                self.assertNotIn(_PRIVATE_SUMMARY, json.dumps(request))
                self.assertNotIn(_CIPHER, json.dumps(request))
                self.assertNotIn("rs_1", json.dumps(request))

    async def test_http_rejection_does_not_retry_or_change_effort(self):
        requests = []

        async def reject(request):
            requests.append(json.loads(request.content))
            return httpx.Response(400, json={"error": {"message": "synthetic unsupported effort"}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(reject)) as client:
            adapter = _model(client, effort="none")
            with self.assertRaisesRegex(Exception, "http_400"):
                _ = [event async for event in adapter.stream([Message("user", "x")], (), thinking=False)]
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["reasoning"], {"effort": "none"})

    async def test_no_reasoning_is_valid_and_request_sizing_matches_actual_body(self):
        fixture = _Fixture(self, [_completed(_answer("文本"))])
        messages = [Message("user", (Text("文字"), Image(_IMAGE)))]
        async with httpx.AsyncClient(transport=fixture.transport) as client:
            adapter = _model(client, request_limit=512)
            events = [item async for item in adapter.stream(messages, (), thinking=False)]
            payload = json.dumps(fixture.requests[0], separators=(",", ":"), ensure_ascii=False).encode()
            self.assertEqual(adapter.remaining_request_bytes(messages, (), thinking=False), 512 - len(payload))
            self.assertFalse(adapter.request_fits([Message("user", "y" * 1024)], (), thinking=False))
        replay = next(event.replay for event in events if isinstance(event, ReplayReady))
        self.assertEqual([item.kind for item in replay.items], ["message"])

    async def test_tools_without_reasoning_still_continue(self):
        tool = AsyncMock(return_value="accepted")
        tools = Tools([Tool(_SPEC, tool)])
        fixture = _Fixture(self, [_completed(_call())], [_completed(_answer("done"))])
        channel = _Channel()
        with closing(History(":memory:")) as history:
            self.assertIsNone(await _complete(fixture, history, channel=channel, tools=tools))
        tool.assert_awaited_once()
        self.assertEqual(channel.replies, ["done"])
        self.assertEqual(fixture.requests[-1]["input"], [
            {"role": "user", "content": "synthetic prompt"},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": _SPEC.name,
             "arguments": '{"value":1}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "accepted"},
        ])

    async def test_sidecar_is_session_scoped_and_rolled_back_with_reply(self):
        fixture = _Fixture(self, [_completed(_reason())], [_completed(_answer("other session"))])
        with closing(History(":memory:")) as history:
            self.assertIsNone(await _complete(fixture, history))
            self.assertIsNone(await _complete(fixture, history, session_id="other"))
            self.assertNotIn(_CIPHER, json.dumps(fixture.requests[1]))
            async with httpx.AsyncClient(transport=fixture.transport) as client:
                scope = _model(client).replay_scope
            entries = history.read_after("main", replay_scope=scope)
            before_count = history._db.execute("SELECT count(*) FROM provider_replays").fetchone()[0]
            outcome = history.outcomes("main")[0]
            duplicate = ReplyOutcome(outcome.reply_id, "main", outcome.run_id, "complete")
            with self.assertRaises(sqlite3.IntegrityError):
                history.finalize_reply(duplicate, [entries[-1].message])
            self.assertEqual(history.read_after("main", replay_scope=scope), entries)
            self.assertEqual(history._db.execute("SELECT count(*) FROM provider_replays").fetchone()[0], before_count)

    async def test_production_composes_selected_openai_adapter(self):
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            settings = Settings(database=Path(folder) / "synthetic.sqlite",
                                llm=LLMSettings("openai_responses", "none", "fixture-model", "local-a"),
                                feishu=FeishuSettings("oc_fixture", "ou_peer", "ou_self"))
            with patch("agent_companion.__main__.run_feishu", new_callable=AsyncMock) as run:
                await _serve(settings, "synthetic-model-test", "synthetic-speech-test")
                run.assert_awaited_once()
                model = run.await_args.args[2]
                self.assertIsInstance(model, OpenAIResponses)
                self.assertEqual((model.replay_scope.model, model.replay_scope.profile), ("fixture-model", "local-a"))

    async def test_deepseek_explicit_effort_and_disabled_use_existing_wire(self):
        captured = []

        async def request(packet):
            self.assertEqual(str(packet.url), CHAT_COMPLETIONS)
            captured.append(json.loads(packet.content))
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}]}\n\n'
                                           'data: [DONE]\n\n')

        async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
            for effort in ("disabled", "high", "future-value"):
                model = DeepSeek(client, model="fixture-deepseek", reasoning_effort=effort)
                _ = [event async for event in model.stream([Message("user", "test")], (), thinking=True)]
        self.assertEqual(captured[0]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", captured[0])
        self.assertEqual([body["model"] for body in captured], ["fixture-deepseek"] * 3)
        self.assertEqual([body["reasoning_effort"] for body in captured[1:]], ["high", "future-value"])
        self.assertEqual([body["thinking"] for body in captured[1:]], [{"type": "enabled"}] * 2)


class LLMSettingsTests(unittest.TestCase):
    def test_rejects_old_or_incomplete_configuration(self):
        sections = ('[deepseek]\nthinking = false', '[llm]\nadapter = "deepseek"',
                    '[llm]\nreasoning_effort = "none"', '[llm]\nadapter = "deepseek"\nreasoning_effort = " "',
                    '[llm]\nadapter = "openai_responses"\nreasoning_effort = "low"\nmodel = "synthetic"')
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            config = Path(folder) / "config.toml"
            for section in sections:
                with self.subTest(section=section):
                    config.write_text(section, encoding="utf-8")
                    with self.assertRaises((TypeError, ValueError)):
                        load_settings(config)
        with patch("sys.argv", ["agent-companion", "--run"]), patch("sys.stderr"), patch("os.getenv") as getenv:
            with self.assertRaises(SystemExit) as exc:
                main()
            self.assertEqual(exc.exception.code, 2)
            getenv.assert_not_called()

        with (patch("sys.argv", ["agent-companion", "--run"]), patch("sys.stderr", new_callable=StringIO) as stderr,
              patch("agent_companion.__main__.load_settings", side_effect=ValueError("synthetic-invalid-value")),
              patch("agent_companion.__main__.os.getenv") as getenv):
            with self.assertRaises(SystemExit) as exc:
                main()
            self.assertEqual(exc.exception.code, 2)
            self.assertIn("无法加载配置（ValueError）", stderr.getvalue())
            self.assertNotIn("synthetic-invalid-value", stderr.getvalue())
            getenv.assert_not_called()

    def test_startup_reads_only_selected_llm_environment_variable(self):
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            config = Path(folder) / "synthetic.toml"
            for adapter, variable, extra in [("deepseek", "DEEPSEEK_API_KEY", ""),
                                             ("openai_responses", "OPENAI_API_KEY",
                                              'model="synthetic"\nreplay_profile="local"\n')]:
                with self.subTest(adapter=adapter):
                    config.write_text(f'[llm]\nadapter="{adapter}"\nreasoning_effort="example"\n{extra}'
                                      '[feishu]\nchat_id="oc_test"\nsender_open_id="ou_peer"\nown_open_id="ou_self"\n',
                                      encoding="utf-8")
                    with (patch("sys.argv", ["agent-companion", "--run", "--config", str(config)]),
                          patch("agent_companion.__main__.os.getenv", return_value="synthetic-only") as getenv,
                          patch("agent_companion.__main__._serve", new_callable=AsyncMock) as serve):
                        self.assertEqual(main(), 0)
                        self.assertEqual([call.args[0] for call in getenv.call_args_list], [variable, "MINIMAX_API_KEY"])
                        self.assertEqual(serve.await_args.args[0].llm.adapter, adapter)
