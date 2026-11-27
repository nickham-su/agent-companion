import asyncio
from contextlib import asynccontextmanager, closing
from io import StringIO
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from agent_companion.__main__ import _serve, main
from agent_companion.app import running_agent
from agent_companion.config import ConfigError, RuntimeSettings, load_settings
from agent_companion.models.deepseek import CHAT_COMPLETIONS, DeepSeek
from agent_companion.models.openai_responses import OpenAIResponses
from agent_companion.protocol import DeliveryStatus, Message, MessageDelivery, Receipt, TextDelta
from agent_companion.runtime.compaction import Compactions
from agent_companion.runtime.context import _SUMMARIZE
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.instructions import load_roles
from agent_companion.storage.sqlite import History, ReplyOutcome
from test_openai_responses import _Wire, _answer, _call, _completed
from test_message_support import tool_calls


_CONFIG = '''asr_model = "speech"
[feishu]
[providers.speech]
adapter = "minimax"
api_key = "fixture-speech"
[providers.deep]
adapter = "deepseek"
api_key = "fixture-deep"
[providers.responses]
adapter = "openai_responses"
api_key = "fixture-responses"
base_url = "https://responses.example.invalid/v1"
[models.asr.speech]
provider = "speech"
max_audio_seconds = 100
[models.llm.chat]
provider = "deep"
model = "main-model"
reasoning_effort = "high"
context_budget = 100
tail_budget = 1
max_tool_rounds = 3
[models.llm.review]
provider = "responses"
model = "child-model"
reasoning_effort = "low"
context_budget = 20
tail_budget = 1
max_tool_rounds = 2
'''
_RESPONSES = "https://responses.example.invalid/v1/responses"


def _deep_reply(*, text=None, tool=None, reason="current reasoning", usage=10):
    delta = {"reasoning_content": reason}
    if tool is not None:
        call_id, args = tool
        delta["tool_calls"] = [{"index": 0, "id": call_id, "type": "function", "function": {
            "name": "delegate_subtask", "arguments": json.dumps(args),
        }}]
    else:
        delta["content"] = text
    event = {"choices": [{"delta": delta, "finish_reason": "tool_calls" if tool else "stop"}],
             "usage": {"prompt_tokens": usage}}
    return httpx.Response(200, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")


class _Channel:
    async def deliver(self, reply):
        text = "".join([part async for part in reply])
        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)


class _Files:
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory(dir=Path.cwd())))
        self.config = self.root / "fixture.toml"
        self.agents = self.root / "agents"
        self.agents.mkdir()
        self.config.write_text(_CONFIG, encoding="utf-8")
        self.role("main", 'model_id="chat"')

    def role(self, role_id, header):
        path = self.agents / f"{role_id}.md"
        path.write_text(f'+++\n{header}\ntools=["delegate_subtask"]\ndescription="test role"\n+++\n'
                        f'{role_id} instructions\n', encoding="utf-8")

    def settings(self):
        return load_settings(self.config, for_run=True)


class ProfileValidationTests(_Files, unittest.TestCase):
    def test_invalid_roles_fail_before_any_probe_or_database(self):
        for header, field in (("", "model_id"), ('model_id="fixture-not-defined"', "model_id"),
                              ('model_id=42', "model_id"), ('model_id="chat"\ntail_budget=1', "未知字段")):
            self.role("reviewer", header)
            with (self.subTest(header=header),
                  patch("sys.argv", ["agent-companion", "--run", "--config", str(self.config)]),
                  patch("sys.stderr", new_callable=StringIO) as stderr,
                  patch("agent_companion.__main__.Path.cwd", return_value=self.root),
                  patch("agent_companion.app.FeishuDiscovery") as discovery,
                  patch("agent_companion.app.History") as history,
                  patch("agent_companion.__main__.httpx.AsyncClient") as client):
                self.assertEqual(main(), 1)
                self.assertIn("agents/reviewer.md", stderr.getvalue())
                self.assertIn(field, stderr.getvalue())
                self.assertNotIn("fixture-not-defined", stderr.getvalue())
                for effect in (discovery, history, client):
                    effect.assert_not_called()

    def test_profile_budget_validation(self):
        for update, field in (("context_budget = false", "context_budget"),
                              ("tail_budget = 100", "tail_budget"),
                              ("max_tool_rounds = 0", "max_tool_rounds"),
                              ("max_tool_rounds = 1.5", "max_tool_rounds")):
            self.config.write_text(_CONFIG.replace(next(line for line in _CONFIG.splitlines()
                                                        if line.strip().startswith(field)), update),
                                   encoding="utf-8")
            with self.subTest(field=field), self.assertRaisesRegex(ConfigError, field):
                self.settings()


class ProfileRuntimeTests(_Files, unittest.IsolatedAsyncioTestCase):
    async def serve(self, handler, drive, *, expected_clients):
        actual_client = httpx.AsyncClient
        events = []
        connection_limits = []
        deep_close, responses_close = DeepSeek.aclose, OpenAIResponses.aclose

        class Client(actual_client):
            async def __aexit__(self, *args):
                events.append("client")
                return await super().__aexit__(*args)

        def client(**kwargs):
            self.assertFalse(kwargs["follow_redirects"])
            self.assertFalse(kwargs["trust_env"])
            connection_limits.append(kwargs.get("limits"))
            return Client(transport=httpx.MockTransport(handler), **kwargs)

        async def close_deep(model):
            events.append("deep")
            await deep_close(model)

        async def close_responses(model):
            events.append("responses")
            await responses_close(model)

        @asynccontextmanager
        async def prepare(settings, *, discovery, roles):
            with closing(History(settings.database)) as history:
                async def run(bindings, asr, tts):
                    await drive(settings, history, bindings, roles)
                    events.append("runtime")

                yield history, run

        with (patch("agent_companion.__main__.prepare_private_feishu", side_effect=prepare),
              patch("agent_companion.__main__.httpx.AsyncClient", side_effect=client),
              patch.object(DeepSeek, "aclose", close_deep), patch.object(OpenAIResponses, "aclose", close_responses)):
            await _serve(self.settings(), roles=load_roles(self.root))
        model_limits = httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=300.0)
        self.assertEqual(connection_limits, [model_limits] * (expected_clients - 1) + [None])  # ASR 沿用默认
        self.assertEqual(events[:1], ["runtime"])
        self.assertEqual(events[-expected_clients:], ["client"] * expected_clients)
        return events[1:-expected_clients]

    async def test_main_and_reused_child_use_own_provider_rounds_and_summary_across_restart(self):
        self.role("reviewer", 'model_id="review"')
        self.role("shared", 'model_id="chat"')
        deep_requests, child_requests = [], []
        child_id, child_session = None, None
        child_summary = asyncio.Event()
        restarting = False

        async def respond(request):
            body = json.loads(request.content)
            url = str(request.url)
            if url == CHAT_COMPLETIONS:
                self.assertTrue(request.headers.get("Authorization") == "Bearer fixture-deep")
                self.assertEqual((body["model"], body["reasoning_effort"]), ("main-model", "high"))
                last = body["messages"][-1]
                deep_requests.append(last)
                if last["content"] == "delegate":
                    return _deep_reply(tool=("new", {"agent": "reviewer", "prompt": "first child task"}))
                if last["content"] == "reuse":
                    return _deep_reply(tool=("reuse", {"agent_id": child_id, "prompt": "reused child task"}))
                self.assertNotEqual(last["content"], _SUMMARIZE.content)
                return _deep_reply(text="main answer")
            self.assertEqual(url, _RESPONSES)
            self.assertTrue(request.headers.get("Authorization") == "Bearer fixture-responses")
            self.assertEqual((body["model"], body["reasoning"]["effort"]), ("child-model", "low"))
            last = body["input"][-1]
            child_requests.append(last)
            if last.get("content") == _SUMMARIZE.content:
                self.assertIn("first child task", json.dumps(body["input"]))
                self.assertNotIn("reused child task", json.dumps(body["input"]))
                child_summary.set()
                completed = _completed(_answer("child summary"))
            elif restarting:
                self.assertEqual(last["content"], "reused child task")
                completed = _completed(_answer("child reuse answer"), input_tokens=30)
            else:
                call = _call(len(child_requests), arguments='{"agent":"missing","prompt":"fixture"}')
                # 本用例在恢复的 Run 验证摘要；第一工具续轮的用量保持在触发线以下。
                completed = _completed(call, input_tokens=18 if len(child_requests) == 1 else 30)
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=_Wire([completed]))

        async def drive(settings, history, models, roles):
            nonlocal child_id, child_session
            self.assertIs(models["main"], models["shared"])
            self.assertEqual(models["main"].settings.max_tool_rounds, 3)
            self.assertEqual(models["reviewer"].settings.max_tool_rounds, 2)
            async with running_agent(settings, history, models, _Channel(), session_id="root", roles=roles) as (root, tree):
                root.submit_messages([MessageDelivery(Message("user", "reuse" if restarting else "delegate"), "trigger")])
                await asyncio.wait_for(tree.wait_idle(), 3)
                self.assertIsNone(root.last_failure)
                accepts = [json.loads(call.result) for call in tool_calls(row.message for row in history.read_after("root"))]
                if not restarting:
                    child_id = accepts[0]["agent_id"]
                    child_session = history.agent(child_id).session_id
                    self.assertEqual(len(history.outcomes(child_session)), 2)
                    self.assertTrue(any(row.message.async_result.payload.get("error") == "tool_round_limit"
                                        for row in history.read_after("root") if row.message.async_result is not None))
                else:
                    self.assertEqual([item["agent_id"] for item in accepts], [child_id, child_id])
                    await asyncio.wait_for(child_summary.wait(), 2)
                    self.assertEqual(len(history.outcomes(child_session)), 3)

        self.assertCountEqual(await self.serve(respond, drive, expected_clients=3), ["deep", "responses"])
        self.assertEqual(len(child_requests), 2)
        self.assertEqual(child_requests[1]["type"], "function_call_output")
        restarting = True
        self.assertCountEqual(await self.serve(respond, drive, expected_clients=3), ["deep", "responses"])
        self.assertEqual(len(child_requests), 4)  # 子角色压缩和第三次普通轮各一次。
        self.assertEqual([last["tool_call_id"] for last in deep_requests if last["role"] == "tool"], ["new", "reuse"])

    async def test_existing_database_and_same_id_model_switch_isolate_usage_and_deepseek_reasoning(self):
        self.role("unused_child", 'model_id="chat"')
        database = self.settings().database
        # 模拟新库中一条没有兼容回放和用量来源的历史回应。
        with closing(History(database)) as history:
            history.finalize_reply(ReplyOutcome("past", "root", "past", "complete", input_tokens=500),
                                   [Message("assistant", "past answer")])
        requests = []
        summary_started = asyncio.Event()
        phase = "first"

        async def respond(request):
            credential = "fixture-deep-rotated" if phase in ("rotated", "changed") else "fixture-deep"
            self.assertTrue(request.headers.get("authorization") == f"Bearer {credential}")
            body = json.loads(request.content)
            messages = body["messages"]
            requests.append((body["model"], messages))
            if messages[-1]["content"] == _SUMMARIZE.content:
                summary_started.set()
                return _deep_reply(text="safe summary", reason="summary reasoning")
            return _deep_reply(text=f"{phase} answer", reason=f"{phase} private", usage=500)

        async def drive(settings, history, models, roles):
            async with running_agent(settings, history, models, _Channel(), session_id="root", roles=roles) as (root, _):
                root.submit_messages([MessageDelivery(Message("user", f"{phase} prompt"), "trigger")])
                await asyncio.wait_for(root.wait_idle(), 2)
                self.assertIsNone(root.last_failure)
                if phase == "same":
                    # 摘要是后台请求，等其发出再退出当前启动。
                    await asyncio.wait_for(summary_started.wait(), 2)
                self.assertEqual(history.read_after("root")[0].message, Message("assistant", "past answer"))

        self.assertEqual(await self.serve(respond, drive, expected_clients=2), ["deep"])
        self.assertEqual(len(requests), 1)
        original = json.dumps(requests[0][1])
        self.assertIn("past answer", original)
        self.assertNotIn("prior unknown reasoning", original)

        phase = "same"
        requests.clear()
        # 换档案名但保留实际连接、模型和 effort：旧用量及有归属的私有推理仍可用。
        self.config.write_text(_CONFIG.replace("[models.llm.chat]", "[models.llm.renamed]"), encoding="utf-8")
        self.role("main", 'model_id="renamed"')
        self.role("unused_child", 'model_id="renamed"')
        self.assertEqual(await self.serve(respond, drive, expected_clients=2), ["deep"])
        self.assertEqual(len(requests), 2)
        for _, messages in requests:
            wire = json.dumps(messages)
            self.assertIn("first private", wire)
            self.assertNotIn("prior unknown reasoning", wire)

        phase = "rotated"
        requests.clear()
        rotated = (_CONFIG.replace("[models.llm.chat]", "[models.llm.renamed]")
                   .replace('api_key = "fixture-deep"', 'api_key = "fixture-deep-rotated"'))
        self.config.write_text(rotated, encoding="utf-8")
        self.assertEqual(await self.serve(respond, drive, expected_clients=2), ["deep"])
        self.assertEqual(len(requests), 1)  # 用量来源仍按现有规则隔离；明文推理不依赖凭据。
        wire = json.dumps(requests[0][1])
        self.assertIn("first private", wire)
        self.assertIn("same private", wire)

        phase = "changed"
        requests.clear()
        self.config.write_text(rotated.replace('model = "main-model"', 'model = "different-model"'), encoding="utf-8")
        self.assertEqual(await self.serve(respond, drive, expected_clients=2), ["deep"])
        self.assertEqual(len(requests), 1)  # 同 ID 指向新模型，旧的 500 token 不触发摘要。
        model, messages = requests[0]
        self.assertEqual(model, "different-model")
        wire = json.dumps(messages)
        for old in ("past answer", "first answer", "same answer", "rotated answer", "changed prompt"):
            self.assertIn(old, wire)
        for private in ("prior unknown reasoning", "first private", "same private", "rotated private"):
            self.assertNotIn(private, wire)

    async def test_unused_profile_credentials_are_not_required(self):
        self.config.write_text(_CONFIG.replace('api_key = "fixture-responses"', 'api_key = false'), encoding="utf-8")

        async def unused(*args):
            self.fail("startup should create no network requests")

        async def drive(settings, history, models, roles):
            self.assertEqual(set(models), {"main"})

        self.assertEqual(await self.serve(unused, drive, expected_clients=2), ["deep"])
        self.role("reviewer", 'model_id="review"')
        with (patch("agent_companion.__main__.prepare_private_feishu") as prepare,
              self.assertRaisesRegex(ConfigError, r"reviewer.md.*model_id.*api_key")):
            await _serve(self.settings(), roles=load_roles(self.root))
        prepare.assert_not_called()


class SharedSummarySlotTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_models_contend_for_one_slot_and_active_model_cancels(self):
        started = asyncio.Event()
        stopped = asyncio.Event()

        class Model:
            def __init__(self, block=False):
                self.block = block
                self.summaries = 0
                self.cancels = 0

            async def stream(self, messages, tools, *, thinking, instructions=None):
                if messages[-1] == _SUMMARIZE:
                    self.summaries += 1
                    if self.block:
                        started.set()
                        try:
                            await asyncio.Future()
                        finally:
                            stopped.set()
                yield TextDelta("response")

            def stop_stream(self, task):
                self.cancels += 1
                task.cancel()
                return True

        with closing(History(":memory:")) as history:
            models = (Model(block=True), Model())
            compactions = Compactions(history)
            runtimes = []
            try:
                for session, model in zip(("first", "second"), models):
                    history.append(session, [Message("user", "old context")])
                    history.finalize_reply(ReplyOutcome(session, session, "seed", "complete", None, input_tokens=99),
                                           [Message("assistant", "old answer")])
                    runtime = Runtime(session, history, model, _Channel(), compactions=compactions,
                                      settings=RuntimeSettings(context_budget=20, tail_budget=1))
                    runtimes.append(runtime)
                    runtime.submit_messages([MessageDelivery(Message("user", "new input"), "trigger")])
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                    await asyncio.wait_for(started.wait(), 2)
                self.assertEqual([model.summaries for model in models], [1, 0])
                self.assertFalse(stopped.is_set())
            finally:
                await asyncio.gather(*(runtime.close() for runtime in runtimes))
                await compactions.close()
            self.assertTrue(stopped.is_set())
            self.assertEqual([model.cancels for model in models], [1, 0])
            self.assertIsNone(history.summary("first"))
