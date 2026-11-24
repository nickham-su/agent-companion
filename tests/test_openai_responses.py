import asyncio
from contextlib import closing
import json
import logging
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from feishu_test_support import private_discovery

from agent_companion.__main__ import _serve
from agent_companion.app import prepare_private_feishu, running_agent
from agent_companion.config import LLMSettings, RuntimeSettings, load_settings
from agent_companion.models.deepseek import CHAT_COMPLETIONS, DeepSeek
from agent_companion.models.openai_responses import OpenAIResponses
from agent_companion.protocol import (DeliveryStatus, Image, InputUsage, Message, ModelError, Receipt,
                                      ReplayReady, Text, ToolSpec)
from agent_companion.runtime.context import _SUMMARIZE
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.storage.sqlite import History, ReplyOutcome, Summary


_RESPONSES = "https://api.openai.com/v1/responses"
_KEY = "synthetic-model-test"
_TEST_SALT = bytes(32)
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


def _completed(*items, input_tokens=None):
    response = {"status": "completed", "output": list(items)}
    if input_tokens is not None:
        response["usage"] = {"input_tokens": input_tokens, "output_tokens": 9000, "total_tokens": 9999}
    return {"type": "response.completed", "response": response}


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
    def __init__(self, testcase, *responses, endpoint=_RESPONSES, api_key=_KEY):
        self._test = testcase
        self._responses = iter(responses)
        self._endpoint = endpoint
        self._api_key = api_key
        self.requests = []
        self.transport = httpx.MockTransport(self._request)

    async def _request(self, request):
        self._test.assertEqual((request.method, str(request.url)), ("POST", self._endpoint))
        self._test.assertTrue(request.headers["authorization"] == f"Bearer {self._api_key}")
        self._test.assertEqual(request.headers["accept-encoding"], "identity")
        payload = json.loads(request.content)
        self.requests.append(payload)
        response = next(self._responses)
        if callable(response):
            response = response(payload)
        if isinstance(response, httpx.Response):
            return response
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


def _model(client, *, history=None, api_key=_KEY, model="fixture-model", effort="custom-effort", **kwargs):
    salt = _TEST_SALT if history is None else history.replay_salt()
    return OpenAIResponses(client, model=model, api_key=api_key, replay_salt=salt, reasoning_effort=effort, **kwargs)


async def _complete(fixture, history, *, prompt=None, channel=None, tools=None, settings=None, session_id="main",
                    instructions="", followup=None, **model_settings):
    async with httpx.AsyncClient(transport=fixture.transport) as client:
        runtime = Runtime(session_id, history, _model(client, history=history, **model_settings), channel or _Channel(),
                          tools=tools, settings=settings, instructions=instructions)
        try:
            runtime.submit_batch([prompt or Message("user", "synthetic prompt")])
            await asyncio.wait_for(runtime.wait_idle(), 2)
            if followup is not None and runtime.last_failure is None:
                task = runtime._compactions._generating
                if task is not None:
                    await asyncio.wait_for(asyncio.shield(task), 2)
                runtime.submit_batch([followup])
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
                runtime = Runtime("main", history, _model(client, history=history), channel, tools=tools)
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

    async def test_reasoning_only_reopens_with_equivalent_base_url(self):
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "fixture.sqlite"
            native = _reason(summary=(_PRIVATE_SUMMARY,))
            added = {"type": "response.output_item.added", "output_index": 0,
                     "item": _reason(cipher="must-not-return-added")}
            done = {"type": "response.output_item.done", "output_index": 0, "item": native}
            first = _Fixture(self, [added, done, _completed(_reason(cipher=None, summary=(_PRIVATE_SUMMARY,)))])
            channel = _Channel()
            with closing(History(path)) as history:
                salt = history.replay_salt()
                self.assertIsNone(await _complete(first, history, channel=channel))
                self.assertEqual(channel.replies, [])
                self.assertEqual(len(history.read_after("main")), 2)
                outcome = history.outcomes("main")[0]
                self.assertIsNone(history.run_text("main", outcome.run_id))
                self.assertIsNone(outcome.delivery_status)
                row = history._db.execute("SELECT fingerprint, items FROM model_replays").fetchone()
                self.assertEqual(len(row["fingerprint"]), 32)
                self.assertTrue(_KEY not in row["items"])
            self.assertTrue(_KEY.encode() not in path.read_bytes())
            second = _Fixture(self, [_completed(_answer("完成"))])
            with closing(History(path)) as history:
                self.assertTrue(history.replay_salt() == salt)
                self.assertIsNone(await _complete(second, history, base_url="HTTPS://API.OpenAI.COM:443/v1///"))
            self.assertEqual(second.requests[0]["input"], [first.requests[0]["input"][0], native,
                                                           {"role": "user", "content": "synthetic prompt"}])

    async def test_key_url_model_or_effort_change_only_sends_common_history(self):
        base = "https://responses.example.invalid/gateway/v1"
        other_path = "https://responses.example.invalid/other/v1"
        variants = ({"api_key": "synthetic-rotated"}, {"base_url": other_path},
                    {"model": "other-model"}, {"effort": "another-valid-for-fixture"})
        for change in variants:
            with self.subTest(field=next(iter(change))), closing(History(":memory:")) as history:
                first = _Fixture(self, [_completed(_reason(), _answer("confirmed"))], endpoint=base + "/responses")
                self.assertIsNone(await _complete(first, history, base_url=base))
                settings = {"base_url": base, **change}
                second = _Fixture(self, [_completed(_answer("next"))], endpoint=settings["base_url"] + "/responses",
                                  api_key=settings.get("api_key", _KEY))
                self.assertIsNone(await _complete(second, history, **settings))
                payload = second.requests[0]
                self.assertEqual(payload["reasoning"], {"effort": settings.get("effort", "custom-effort")})
                self.assertEqual(payload["model"], settings.get("model", "fixture-model"))
                self.assertEqual(payload["input"], first.requests[0]["input"] + [
                    {"role": "assistant", "content": "confirmed"}, {"role": "user", "content": "synthetic prompt"},
                ])

    async def test_sidecar_copied_to_different_database_does_not_match(self):
        with (TemporaryDirectory(dir=Path.cwd()) as folder,
              closing(History(Path(folder) / "first.sqlite")) as first,
              closing(History(Path(folder) / "second.sqlite")) as second):
            self.assertTrue(first.replay_salt() != second.replay_salt())
            source = _Fixture(self, [_completed(_reason(), _answer("confirmed"))])
            self.assertIsNone(await _complete(source, first))
            second.append("main", [row.message for row in first.read_after("main")])
            with second._db:
                row = first._db.execute("SELECT * FROM model_replays").fetchone()
                second._db.execute("INSERT INTO model_replays VALUES (?, ?, ?, ?, ?)", tuple(row))
            dest = _Fixture(self, [_completed(_answer("next"))])
            self.assertIsNone(await _complete(dest, second))
            self.assertEqual(dest.requests[0]["input"], source.requests[0]["input"] + [
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
                        conn.execute("UPDATE model_replays SET items = ?", ('{"malformed":"' + _CIPHER + '"}',))
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

    async def test_partial_and_zero_delivery_keep_native_text_only_when_fully_confirmed(self):
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

    async def test_partial_and_zero_delivery_keep_both_reasoning_items_and_tool_pairs(self):
        early, late = _reason(), _reason("rs_2", "synthetic-late-opaque")
        answer = _answer("ABCDEF")
        for confirmed in ("ABC", "", None):
            with self.subTest(confirmed=confirmed), closing(History(":memory:")) as history:
                tool = AsyncMock(return_value="accepted")
                tools = Tools([Tool(_SPEC, tool)])
                channel = _Channel(first_confirmed=confirmed)
                fixture = _Fixture(self, [_completed(early, answer, _call(), late, _call(2))],
                                   [_completed(_answer("continued", "msg_done"))])
                self.assertIsNone(await _complete(fixture, history, channel=channel, tools=tools))
                self.assertEqual([call.args[1].call_id for call in tool.await_args_list], ["call_1", "call_2"])
                later = _Fixture(self, [_completed(_answer("later", "msg_later"))])
                self.assertIsNone(await _complete(later, history, tools=tools))
                calls = [{k: v for k, v in _call(n).items() if k != "status"} for n in (1, 2)]
                visible = ([answer] if confirmed is None else
                           ([{"role": "assistant", "content": confirmed}] if confirmed else []))
                expected = fixture.requests[0]["input"] + [early, *visible, calls[0], late, calls[1]] + [
                    {"type": "function_call_output", "call_id": f"call_{n}", "output": "accepted"} for n in (1, 2)
                ]
                self.assertEqual(fixture.requests[1]["input"], expected)
                self.assertEqual(later.requests[0]["input"][:len(expected)], expected)
                self.assertEqual(history.read_after("main")[1].message.content,
                                 "ABCDEF" if confirmed is None else confirmed)
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
                     (_Wire([_delta("可见前缀", output=1), valid], close_error=True), "transport_error")]
        for packet, error in variants:
            with self.subTest(error=error), closing(History(":memory:")) as history:
                tool = AsyncMock(return_value="must not call")
                fixture = _Fixture(self, packet)
                self.assertEqual(await _complete(fixture, history, tools=Tools([Tool(_SPEC, tool)])), "model_failed")
                tool.assert_not_awaited()
                self.assertEqual(history.outcomes("main")[0].model_error, error)
                self.assertEqual(history.read_after("main")[1].message.content, "可见前缀")
                self.assertEqual(history._db.execute("SELECT count(*) FROM model_replays").fetchone()[0], 0)

    async def test_voice_interrupt_while_closing_preserves_transport_but_not_withheld_terminal(self):
        closing_wire, release_close, closed_wire = (asyncio.Event() for _ in range(3))

        class SlowWire(_Wire):
            async def aclose(self):
                closing_wire.set()
                await release_close.wait()
                closed_wire.set()

        fixture = _Fixture(self, SlowWire([_delta("confirmed tail", output=1),
                                           _completed(_reason(), _answer("confirmed tail"), _call())]),
                           [_completed(_answer("next"))])
        tool = AsyncMock(return_value="unexpected")

        class Voice:
            async def deliver(self, reply):
                reply.select_voice_delivery()
                first = await anext(reply)
                if first == "confirmed tail":
                    await closing_wire.wait()
                    return Receipt(reply.info.reply_id, DeliveryStatus.PARTIAL, "confirmed")
                text = first + "".join([part async for part in reply])
                return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)

        with closing(History(":memory:")) as history:
            async with httpx.AsyncClient(transport=fixture.transport) as client:
                runtime = Runtime("main", history, _model(client), Voice(), tools=Tools([Tool(_SPEC, tool)]))
                try:
                    runtime.submit_batch([Message("user", "one")])
                    runtime.submit_batch([Message("user", "two")])
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                    self.assertFalse(closed_wire.is_set())
                    old, new = history.outcomes("main")
                    self.assertEqual((old.model_status, old.delivery_status),
                                     ("model_interrupted", DeliveryStatus.PARTIAL))
                    self.assertNotEqual(old.run_id, new.run_id)
                    self.assertEqual(fixture.requests[1]["input"][-3:], [
                        {"role": "user", "content": "one"}, {"role": "assistant", "content": "confirmed"},
                        {"role": "user", "content": "two"},
                    ])
                    tool.assert_not_awaited()
                finally:
                    release_close.set()
                    await runtime.close()
                self.assertTrue(closed_wire.is_set())
                tool.assert_not_awaited()

    async def test_historical_replay_survives_changes_to_tools_and_input_prefix(self):
        tool = AsyncMock(return_value="accepted")
        fixture = _Fixture(self, [_completed(_reason(), _call())], [_completed(_answer("ended", "msg_done"))],
                           [_completed(_answer("next", "msg_next"))])
        with closing(History(":memory:")) as history:
            prompt = Message("user", (Text("initial"), Image(_IMAGE)))
            self.assertIsNone(await _complete(fixture, history, prompt=prompt, tools=Tools([Tool(_SPEC, tool)])))
            first = history.read_after("main")[0].message_id
            # 摘要覆盖的是旧输入，不覆盖其后的工具组。
            history.save_summary("main", Summary(first, "previous input compacted"))
            changed = ToolSpec(_SPEC.name, "new description", {"type": "object", "additionalProperties": False})
            self.assertIsNone(await _complete(fixture, history, tools=Tools([Tool(changed, tool)])))
            last = fixture.requests[-1]
            self.assertNotEqual(fixture.requests[0]["tools"], last["tools"])
            inputs = last["input"]
            self.assertEqual(inputs[0]["role"], "system")
            self.assertNotIn("input_image", json.dumps(inputs))
            self.assertEqual(inputs[1:4], [
                _reason(), {k: v for k, v in _call().items() if k != "status"},
                {"type": "function_call_output", "call_id": "call_1", "output": "accepted"},
            ])
        tool.assert_awaited_once()

    async def test_continuation_reprojects_tools_and_adds_queued_input_at_safe_boundary(self):
        settings = RuntimeSettings(context_budget=240, tail_budget=75)
        result = "r" * 285
        fixture = _Fixture(self, [_completed(_reason(), _call())], [_completed(_answer("done"))])
        channel = _Channel()
        with closing(History(":memory:")) as history:
            async def receive_at_tool_boundary(_arguments, _ctx):
                tools.specs = (ToolSpec(_SPEC.name, "updated tool", _SPEC.parameters),)
                runtime.submit_batch([Message("user", "queued"), Message("user", "same batch")])
                return result

            tool = AsyncMock(side_effect=receive_at_tool_boundary)
            tools = Tools([Tool(_SPEC, tool)])
            async with httpx.AsyncClient(transport=fixture.transport) as client:
                runtime = Runtime("main", history, _model(client, history=history), channel,
                                  settings=settings, tools=tools)
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
        self.assertNotEqual(first["tools"], next_["tools"])
        self.assertEqual(first["input"], [{"role": "user", "content": "prompt"}])
        self.assertEqual(next_["input"], first["input"] + [
            _reason(), {k: v for k, v in _call().items() if k != "status"},
            {"type": "function_call_output", "call_id": "call_1", "output": result},
            {"role": "user", "content": "queued"},
            {"role": "user", "content": "same batch"},
        ])

    async def test_reasoning_continuation_compacts_history_or_keeps_current_image(self):
        for mode in ("compact", "image"):
            with self.subTest(mode=mode), closing(History(":memory:")) as history:
                tool = AsyncMock(return_value="r" * 285)
                initial, final = [_completed(_reason(), _call(), input_tokens=270)], [_completed(_answer("done"))]
                if mode == "compact":
                    history.append("main", [Message("user", "x" * 360)])
                    def generation_or_reply(payload):
                        if payload["input"][-1].get("content") == _SUMMARIZE.content:
                            return [_completed(_answer("summary"))]
                        return final

                    fixture = _Fixture(self, initial, generation_or_reply, generation_or_reply,
                                       [_completed(_answer("followup"))])
                    settings = RuntimeSettings(context_budget=270, tail_budget=75)
                    self.assertIsNone(await _complete(fixture, history, settings=settings,
                                                      tools=Tools([Tool(_SPEC, tool)]),
                                                      followup=Message("user", "apply summary")))
                    self.assertIsNotNone(history.summary("main"))
                    self.assertEqual(len(fixture.requests), 4)
                    self.assertEqual(fixture.requests[-1]["input"][0]["role"], "system")
                    self.assertNotIn("x" * 360, json.dumps(fixture.requests[-1]))
                    continuation = next(payload for payload in fixture.requests[1:3]
                                        if payload["input"][-1].get("type") == "function_call_output")
                    self.assertIn("x" * 360, json.dumps(continuation))
                else:
                    image = Image("data:image/png;base64," + "YQAA" * 100)
                    prompt = Message("user", (Text("image"), image))
                    fixture = _Fixture(self, initial, final)
                    self.assertIsNone(await _complete(fixture, history, prompt=prompt,
                                                      tools=Tools([Tool(_SPEC, tool)])))
                    self.assertEqual(fixture.requests[0]["input"][0]["content"][1]["image_url"], image.url)
                    self.assertEqual(fixture.requests[1]["input"][0], fixture.requests[0]["input"][0])
                    self.assertEqual(history.read_after("main")[0].message, prompt)
                    continuation = fixture.requests[-1]
                tool.assert_awaited_once()
                self.assertEqual(continuation["input"][-3:], [
                    _reason(), {k: v for k, v in _call().items() if k != "status"},
                    {"type": "function_call_output", "call_id": "call_1", "output": "r" * 285},
                ])

    async def test_images_remain_in_replay_with_queued_input(self):
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
                runtime = Runtime("main", history, _model(client, history=history), channel,
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
        first, intermediate, continued = (request["input"] for request in fixture.requests)
        self.assertEqual(first[0]["content"][1]["image_url"], image.url)
        calls = [{k: v for k, v in _call(n).items() if k != "status"} for n in (1, 2)]
        self.assertEqual(intermediate, [
            first[0],
            calls[0], {"type": "function_call_output", "call_id": "call_1", "output": first_result},
        ])
        self.assertEqual(continued, intermediate + [
            _reason(), {"role": "assistant", "content": "ABC"}, calls[1],
            {"type": "function_call_output", "call_id": "call_2", "output": second_result},
            {"role": "user", "content": "queued"},
        ])

    async def test_tool_group_over_soft_threshold_is_sent_once_and_whole(self):
        settings = RuntimeSettings(context_budget=240, tail_budget=75)
        result = "r" * 1500
        tool = AsyncMock(return_value=result)
        fixture = _Fixture(self, [_completed(_reason(), _call(), input_tokens=240)], [_completed(_answer("done"))])
        with closing(History(":memory:")) as history:
            self.assertIsNone(await _complete(fixture, history, settings=settings, tools=Tools([Tool(_SPEC, tool)])))
            self.assertEqual(len(fixture.requests), 2)
            self.assertIsNone(history.summary("main"))
            self.assertEqual(fixture.requests[-1]["input"][-1],
                             {"type": "function_call_output", "call_id": "call_1", "output": result})
            self.assertEqual(history.outcomes("main")[0].input_tokens, 240)
        tool.assert_awaited_once()

    async def test_missing_or_damaged_active_replay_uses_common_tool_pair(self):
        for damage in ("missing", "call-id", "escaped-surrogate", "invalid-utf8"):
            with self.subTest(damage=damage), closing(History(":memory:")) as history:
                tool = AsyncMock(return_value="accepted")
                fixture = _Fixture(self, [_completed(_reason(), _call())], [_completed(_answer("done"))])
                finalize = history.finalize_reply

                def damage_after_commit(outcome, rows):
                    finalize(outcome, rows)
                    if rows and rows[0].tool_calls:
                        with history._db:
                            if damage == "missing":
                                history._db.execute("DELETE FROM model_replays")
                            elif damage == "invalid-utf8":
                                history._db.execute("UPDATE model_replays SET items = CAST(X'FF' AS TEXT)")
                            else:
                                stored = history._db.execute("SELECT items FROM model_replays").fetchone()
                                data = json.loads(stored[0])
                                if damage == "call-id":
                                    data[-1]["call_id"] = "corrupt-call-id"
                                else:
                                    data[0]["encrypted_content"] = "\udfff"
                                history._db.execute("UPDATE model_replays SET items = ?", (json.dumps(data),))

                with patch.object(history, "finalize_reply", side_effect=damage_after_commit):
                    self.assertIsNone(await _complete(fixture, history, tools=Tools([Tool(_SPEC, tool)])))
                tool.assert_awaited_once()
                self.assertEqual(fixture.requests[1]["input"], fixture.requests[0]["input"] + [
                    {k: v for k, v in _call().items() if k not in ("id", "status")},
                    {"type": "function_call_output", "call_id": "call_1", "output": "accepted"},
                ])

    async def test_missing_cipher_does_not_discard_other_items_or_block_tools(self):
        tool = AsyncMock(return_value="accepted")
        good = _reason("rs_2")
        answer = _answer("可见正文")
        fixture = _Fixture(self, [_completed(_reason(cipher=None), answer, good, _call())],
                           [_completed(_answer("done", "msg_done"))])
        with closing(History(":memory:")) as history:
            self.assertIsNone(await _complete(fixture, history, tools=Tools([Tool(_SPEC, tool)])))
        tool.assert_awaited_once()
        self.assertEqual(fixture.requests[1]["input"], fixture.requests[0]["input"] + [
            answer, good, {k: v for k, v in _call().items() if k != "status"},
            {"type": "function_call_output", "call_id": "call_1", "output": "accepted"},
        ])

    async def test_summary_reuses_native_prefix_and_omits_tail_and_current_input(self):
        private = _reason(summary=(_PRIVATE_SUMMARY,))
        tool = AsyncMock(return_value="historical result")
        registry = Tools([Tool(_SPEC, tool)])
        summary_prompt = {"role": "user", "content": _SUMMARIZE.content}

        def generation_or_reply(payload):
            text = "compressed" if payload["input"][-1] == summary_prompt else "new response"
            return [_completed(_answer(text))]

        fixture = _Fixture(self, [_completed(private, _call())], [_completed(_answer("retained tail"), input_tokens=2500)],
                           generation_or_reply, generation_or_reply, [_completed(_answer("followup response"))])
        channel = _Channel()
        with closing(History(":memory:")) as history:
            covered, = history.append("main", [Message("user", "prior seed")])
            history.save_summary("main", Summary(covered, "same existing summary"))
            image = Message("user", (Text("historical image"), Image(_IMAGE)))
            current = "current " + "y" * 1400
            settings = RuntimeSettings(context_budget=2500, tail_budget=510)
            async with httpx.AsyncClient(transport=fixture.transport) as client:
                runtime = Runtime("main", history, _model(client, history=history), channel, tools=registry,
                                  settings=settings, instructions="fixed synthetic instructions")
                try:
                    for prompt in (image, Message("user", current)):
                        runtime.submit_batch([prompt])
                        await asyncio.wait_for(runtime.wait_idle(), 2)
                        self.assertIsNone(runtime.last_failure)
                    task = runtime._compactions._generating
                    if task is not None:
                        await asyncio.wait_for(asyncio.shield(task), 2)
                    self.assertEqual(history.summary("main").text, "same existing summary")
                    runtime.submit_batch([Message("user", "apply summary")])
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                    self.assertIsNone(runtime.last_failure)
                finally:
                    await runtime.close()
            self.assertEqual(len(fixture.requests), 5)
            previous_reply, current_reply = fixture.requests[1], fixture.requests[-1]
            compressed = next(payload for payload in fixture.requests[2:4] if payload["input"][-1] == summary_prompt)
            self.assertEqual(compressed, {**previous_reply, "input": previous_reply["input"] + [
                {"role": "user", "content": _SUMMARIZE.content},
            ]})
            self.assertIn(private, compressed["input"])
            self.assertIn("same existing summary", repr(compressed["input"]))
            self.assertNotIn("retained tail", repr(compressed["input"]))
            self.assertNotIn(current, repr(compressed["input"]))
            self.assertIn("retained tail", repr(current_reply["input"]))
            self.assertIn(current, repr(current_reply["input"]))
            self.assertNotIn(private, current_reply["input"])
            self.assertNotIn(_SUMMARIZE, [row.message for row in history.read_after("main")])
            self.assertEqual(history.summary("main").text, "compressed")
            self.assertEqual(channel.replies, ["retained tail", "new response", "followup response"])
        tool.assert_awaited_once()

    async def test_request_larger_than_old_body_limit_is_sent(self):
        image = Image("data:image/png;base64," + "A" * (32 * 1024 * 1024))
        seen = []

        def endpoint(request):
            self.assertGreater(len(request.content), 32 * 1024 * 1024)
            payload = json.loads(request.content)
            self.assertEqual(payload["input"][0]["content"][0]["image_url"], image.url)
            seen.append(True)
            return httpx.Response(413)

        async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
            with self.assertRaisesRegex(ModelError, "^http_413$"):
                _ = [event async for event in _model(client).stream([Message("user", (image,))], (), thinking=False)]
        self.assertEqual(seen, [True])

    async def test_summary_rejects_obvious_private_replay_echo_but_allows_short_common_text(self):
        long_text = "synthetic long private reasoning token " * 2
        long_cipher = "synthetic-opaque-" + "q" * 42
        cases = (
            ("summary part", "prefix " + long_text, (), True),
            ("ciphertext", "prefix " + long_cipher, (), True),
            ("response's short field", "status", (_reason("rs_2", summary=("status",)),), True),
            ("ordinary prose", "updated status for the case", (), False),
        )
        for case, generated, output_reasoning, rejected in cases:
            with self.subTest(case=case), closing(History(":memory:")) as history:
                def generation_or_reply(payload):
                    if payload["input"][-1].get("content") == _SUMMARIZE.content:
                        return [_completed(*output_reasoning, _answer(generated))]
                    return [_completed(_answer("normal visible reply"))]

                fixture = _Fixture(self, [_completed(_reason(cipher=long_cipher, summary=("status", long_text)),
                                                     _answer("old response"), input_tokens=252)],
                                   generation_or_reply, generation_or_reply,
                                   [_completed(_answer("followup reply"))])
                covered, = history.append("main", [Message("user", "prior seed")])
                saved = Summary(covered, "existing")
                history.save_summary("main", saved)
                self.assertIsNone(await _complete(fixture, history, prompt=Message("user", "x" * 340)))
                old_end = history.read_after("main")[-1].message_id
                options = dict(settings=RuntimeSettings(context_budget=252, tail_budget=125),
                               prompt=Message("user", "y" * 420), followup=Message("user", "use candidate"))
                if rejected:
                    with self.assertLogs("agent_companion.runtime.compaction", level="WARNING") as logs:
                        self.assertIsNone(await _complete(fixture, history, **options))
                    self.assertIn("summary_private_echo", logs.output[0])
                    self.assertNotIn(generated, logs.output[0])
                    self.assertEqual(history.summary("main"), saved)
                else:
                    self.assertIsNone(await _complete(fixture, history, **options))
                    self.assertEqual(history.summary("main"), Summary(old_end, generated))
                self.assertEqual(len(fixture.requests), 4)
                summary = next(payload for payload in fixture.requests[1:3]
                               if payload["input"][-1].get("content") == _SUMMARIZE.content)
                self.assertIn(long_text, repr(summary))
                self.assertIn(long_cipher, repr(summary))
                self.assertNotIn(generated, [entry.message.content for entry in history.read_after("main")])

    async def test_http_rejection_does_not_retry_or_rerun_tools(self):
        base = "https://responses.example.invalid/fixture"
        tool = AsyncMock(return_value="accepted")
        fixture = _Fixture(self, [_completed(_reason(), _call())], httpx.Response(400, json={
            "error": {"message": f"synthetic upstream error {_KEY} {base} {_CIPHER}"},
        }), endpoint=base + "/responses")
        with (closing(History(":memory:")) as history,
              self.assertLogs("agent_companion.runtime.engine", logging.WARNING) as logs):
            self.assertEqual(await _complete(fixture, history, effort="none", base_url=base,
                                             tools=Tools([Tool(_SPEC, tool)])), "model_failed")
            self.assertEqual(history.outcomes("main")[-1].model_error, "http_400")
            self.assertEqual(history.read_after("main")[-1].message.tool_call_id, "call_1")
        tool.assert_awaited_once()
        self.assertEqual([request["reasoning"] for request in fixture.requests], [{"effort": "none"}] * 2)
        self.assertIn(_reason(), fixture.requests[-1]["input"])
        self.assertTrue(all(value not in "".join(logs.output) for value in (_KEY, base, _CIPHER)))

    async def test_redirect_is_not_followed_even_with_client_defaults(self):
        for destination in ("https://responses.example.invalid/elsewhere", "https://other.example.invalid/responses"):
            with self.subTest(destination=destination):
                base = "https://responses.example.invalid/v1"
                fixture = _Fixture(self, httpx.Response(307, headers={"Location": destination}),
                                   endpoint=base + "/responses")
                async with httpx.AsyncClient(transport=fixture.transport, follow_redirects=True,
                                             auth=("fixture", "unused"), headers={"Authorization": "unused"}) as client:
                    adapter = _model(client, base_url=base)
                    self.assertNotIn("fingerprint", repr(adapter.replay_scope))
                    with self.assertRaisesRegex(ModelError, "^http_307$"):
                        _ = [event async for event in adapter.stream([Message("user", "x")], (), thinking=False)]
                self.assertEqual(len(fixture.requests), 1)
                self.assertTrue(_KEY not in json.dumps(fixture.requests))

    async def test_legacy_replay_table_is_ignored_and_left_intact(self):
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "legacy.sqlite"
            with closing(History(path)) as history:
                assistant_id, = history.append("main", [Message("assistant", "confirmed")])
                with history._db:
                    history._db.execute("CREATE TABLE provider_replays (assistant_message_id INTEGER PRIMARY KEY, items TEXT)")
                    history._db.execute("INSERT INTO provider_replays VALUES (?, ?)", (assistant_id, _CIPHER))
            with closing(History(path)) as history:
                fixture = _Fixture(self, [_completed(_answer("next"))])
                self.assertIsNone(await _complete(fixture, history))
                self.assertEqual(fixture.requests[0]["input"], [
                    {"role": "assistant", "content": "confirmed"}, {"role": "user", "content": "synthetic prompt"},
                ])
                self.assertTrue(history._db.execute("SELECT items FROM provider_replays").fetchone()[0] == _CIPHER)

    async def test_no_reasoning_is_valid_and_input_usage_is_optional(self):
        fixture = _Fixture(self, [_completed(_answer("文本"), input_tokens=17)],
                           [_completed(_answer("正常"), input_tokens=True)])
        messages = [Message("user", (Text("文字"), Image(_IMAGE)))]
        async with httpx.AsyncClient(transport=fixture.transport) as client:
            adapter = _model(client)
            events = [item async for item in adapter.stream(messages, (), thinking=False)]
            invalid = [item async for item in adapter.stream(messages, (), thinking=False)]
        replay = next(event.replay for event in events if isinstance(event, ReplayReady))
        self.assertEqual([item.kind for item in replay.items], ["message"])
        self.assertEqual([event for event in events if isinstance(event, InputUsage)], [InputUsage(17)])
        self.assertFalse(any(isinstance(event, InputUsage) for event in invalid))

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
                scope = _model(client, history=history).replay_scope
            entries = history.read_after("main", replay_scope=scope)
            before_count = history._db.execute("SELECT count(*) FROM model_replays").fetchone()[0]
            outcome = history.outcomes("main")[0]
            duplicate = ReplyOutcome(outcome.reply_id, "main", outcome.run_id, "complete")
            with self.assertRaises(sqlite3.IntegrityError):
                history.finalize_reply(duplicate, [entries[-1].message])
            self.assertEqual(history.read_after("main", replay_scope=scope), entries)
            self.assertEqual(history._db.execute("SELECT count(*) FROM model_replays").fetchone()[0], before_count)

    async def test_production_switches_named_providers_without_changing_replay_identity_rules(self):
        endpoint = "https://responses.example.invalid/api/responses"
        default = (_KEY, "HTTPS://RESPONSES.EXAMPLE.INVALID:443/api/", "fixture-model", "low")
        profiles = {
            "initial": default,
            "renamed": (_KEY, "https://responses.example.invalid/api", "fixture-model", "low"),
            "other_key": ("synthetic-rotation", *default[1:]),
            "other_address": (_KEY, "https://other.example.invalid/api", *default[2:]),
            "other_model": (*default[:2], "other-model", "low"),
            "other_effort": (*default[:3], "future-value"),
        }
        tables = """
asr_provider = "speech"
[timeouts]
llm = 0.75
[feishu]
own_open_id = "ou_self"
[providers.speech]
adapter = "minimax"
api_key = "synthetic-speech-test"
[providers.speech.asr]
max_audio_seconds = 500
"""
        for name, (key, address, model, effort) in profiles.items():
            tables += (f'[providers.{name}]\nadapter = "openai_responses"\napi_key = "{key}"\nbase_url = "{address}"\n'
                       f'[providers.{name}.llm]\nmodel = "{model}"\nreasoning_effort = "{effort}"\n')

        async def exercise(settings, history, model, asr, tts):
            self.assertIsInstance(model, OpenAIResponses)
            self.assertIsNone(tts)
            async with running_agent(settings, history, model, _Channel(), session_id="test-openai") as (runtime, _):
                self.assertEqual(runtime._settings.model_timeout, 0.75)
                runtime.submit_batch([Message("user", "synthetic prompt")])
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertIsNone(runtime.last_failure)

        async def prepare(settings, history, *, discovery):
            await prepare_private_feishu(settings, history, discovery=discovery)

            async def run(model, asr, tts):
                await exercise(settings, history, model, asr, tts)

            return run

        actual_client = httpx.AsyncClient
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            config = Path(folder) / "synthetic.toml"
            for name, (key, _, model, effort) in profiles.items():
                with self.subTest(provider=name):
                    config.write_text(f'llm_provider = "{name}"\n' + tables, encoding="utf-8")
                    settings = load_settings(config, for_run=True)
                    answer = (_reason(), _answer("old confirmed")) if name == "initial" else (_answer("follow-up"),)
                    target = "https://other.example.invalid/api/responses" if name == "other_address" else endpoint
                    fixture = _Fixture(self, [_completed(*answer)], api_key=key, endpoint=target)
                    with (patch("agent_companion.__main__.prepare_private_feishu", side_effect=prepare),
                          patch("agent_companion.__main__.httpx.AsyncClient",
                                side_effect=lambda **opts: actual_client(transport=fixture.transport, **opts))):
                        await _serve(settings, discovery=private_discovery("oc_fixture", "ou_peer", "ou_self"))
                    request, = fixture.requests
                    self.assertEqual(request["model"], model)
                    self.assertEqual(request["reasoning"]["effort"], effort)
                    body = json.dumps(request["input"])
                    self.assertEqual(_CIPHER in body, name == "renamed")
                    self.assertEqual("old confirmed" in body, name != "initial")

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
    def test_base_url_normalization_and_validation(self):
        llm = LLMSettings("openai_responses", "low", "fixture", "HTTPS://[2001:DB8::1]:8443/model/v1///", api_key=_KEY)
        self.assertEqual(llm.base_url, "https://[2001:db8::1]:8443/model/v1")
        self.assertNotIn("base_url", repr(llm))
        for url in ("http://localhost/v1", "https://token@example.invalid/v1", "https://example.invalid/v1?",
                    "https://example.invalid/v1#", "https://example.invalid:65536/v1", "https://example.invalid:",
                    "https://example.invalid\\v1", "https://example.invalid/with space", "https:///v1",
                    "https://example.invalid/" + "路" * 300):
            with self.subTest(url=url), self.assertRaises(ValueError) as exc:
                LLMSettings("openai_responses", "low", "fixture", url, api_key=_KEY)
            self.assertTrue(url not in str(exc.exception))
