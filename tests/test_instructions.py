import asyncio
from contextlib import closing
from inspect import Parameter, signature
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from agent_companion.__main__ import main
from agent_companion.app import prepare_private_feishu, run_feishu, running_agent
from agent_companion.config import ConfigError, FeishuSettings, RuntimeSettings, Settings
from agent_companion.models.deepseek import DeepSeek, build_request
from agent_companion.models.openai_responses import OpenAIResponses
from agent_companion.protocol import (DeliveryStatus, InputUsage, Message, MessageDelivery, ModelError, Receipt,
                                      TextDelta, ToolCall, ToolReady)
from agent_companion.runtime.context import _SUMMARIZE
from agent_companion.runtime.instructions import MAX_ROLE_BYTES, load_roles
from agent_companion.runtime.subtasks import DELEGATE_TOOL_NAME
from agent_companion.storage.sqlite import History
from test_message_support import tool_calls


class Model:
    def __init__(self, respond=None):
        self.requests = []
        self.respond = respond or (lambda messages: [TextDelta("answer")])

    async def stream(self, messages, tools, *, thinking, instructions=None):
        self.requests.append((tuple(messages), tuple(tools), instructions))
        for event in self.respond(messages):
            yield event


class Sink:
    async def deliver(self, reply):
        text = "".join([chunk async for chunk in reply])
        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)


def _call(call_id, **arguments):
    return ToolReady(ToolCall(call_id, DELEGATE_TOOL_NAME, json.dumps(arguments)))


class _RoleFiles:
    def setUp(self):
        parent = Path(self.enterContext(TemporaryDirectory(dir=Path.cwd())))
        self.workdir = parent / "working"
        self.agents = self.workdir / "agents"
        self.agents.mkdir(parents=True)
        (self.workdir / "AGENTS.md").write_text("排除旧指令", encoding="utf-8")

    def write_role(self, role, body, *, tools=(), description=None, newline="\n"):
        lines = ["+++", f"tools = {json.dumps(tools)}", 'model_id = "chat"']
        if description is not None:
            lines.append(f"description = {json.dumps(description, ensure_ascii=False)}")
        lines.extend(("+++", ""))
        path = self.agents / f"{role}.md"
        path.write_bytes((newline.join(lines) + body).encode("utf-8"))
        return path


class RoleFileTests(_RoleFiles, unittest.TestCase):
    def test_silent_end_requires_each_role_to_opt_in(self):
        self.write_role("main", "主角色")
        self.write_role("review", "子角色", tools=("silent_end",), description="审阅")
        roles = load_roles(self.workdir)
        self.assertEqual(roles["main"].tools, ())
        self.assertEqual(roles["review"].tools, ("silent_end",))

        self.write_role("main", "主角色", tools=("silent_end",))
        self.write_role("review", "子角色", description="审阅")
        roles = load_roles(self.workdir)
        self.assertEqual((roles["main"].tools, roles["review"].tools), (("silent_end",), ()))
    def test_application_entry_points_require_keyword_roles(self):
        for entry in (prepare_private_feishu, run_feishu, running_agent):
            with self.subTest(entry=entry.__name__):
                roles = signature(entry).parameters["roles"]
                self.assertIs(roles.kind, Parameter.KEYWORD_ONLY)
                self.assertIs(roles.default, Parameter.empty)

    def test_missing_main_stops_startup_before_database_and_network_at_critical_log_level(self):
        settings = Settings(feishu=FeishuSettings(), log_level="CRITICAL")
        (self.agents / "main.example.md").write_bytes(b"template is not a configuration")
        # 上级目录即使有 main.md 也不搜索。
        other_agents = self.workdir.parent / "agents"
        other_agents.mkdir()
        (other_agents / "main.md").write_bytes(b"+++\ntools=[]\n+++\n")
        for with_directory in (True, False):
            if not with_directory:
                (self.agents / "main.example.md").unlink()
                self.agents.rmdir()
            with self.subTest(with_directory=with_directory):
                with (patch("sys.argv", ["agent-companion", "--run"]),
                      patch("sys.stderr", new_callable=StringIO) as stderr,
                      patch("agent_companion.__main__.load_settings", return_value=settings),
                      patch("agent_companion.__main__.Path.cwd", return_value=self.workdir),
                      patch("agent_companion.app.History") as history,
                      patch("agent_companion.app.FeishuDiscovery") as network):
                    self.assertEqual(main(), 1)
                history.assert_not_called()
                network.assert_not_called()
                self.assertIn("缺少 agents/main.md，请从 agents/main.example.md", stderr.getvalue())

    def test_invalid_role_fields_and_delimiters_have_fixed_diagnostics(self):
        self.write_role("main", "")
        cases = [
            ("model_id", "+++\ndescription='fixture'\ntools=[]\n+++\n"),
            ("首行", "+++ \ndescription='fixture'\ntools=[]\n+++\n"),
            ("结束分隔线", "+++\ndescription='fixture'\ntools=[]\n +++\n"),
            ("TOML 格式错误（第 2 行", "+++\ndescription=@synthetic-detail\ntools=[]\n+++\n"),
            ("TOML 格式错误", "+++\ndescription='synthetic-detail\ntools=[]\n+++\n"),
            ("未知字段", "+++\ndescription='fixture'\ntools=[]\nsynthetic-detail='fixture'\n+++\n"),
            ("description", "+++\ntools=[]\n+++\n"),
            ("description", "+++\ndescription='  '\ntools=[]\n+++\n"),
            ("description", "+++\ndescription=42\ntools=[]\n+++\n"),
            ("tools 必须", "+++\ndescription='fixture'\n+++\n"),
            ("tools 必须", "+++\ndescription='fixture'\ntools='synthetic-detail'\n+++\n"),
            ("tools 必须", "+++\ndescription='fixture'\ntools=[1]\n+++\n"),
            ("未知工具名", "+++\ndescription='fixture'\ntools=['synthetic-detail']\n+++\n"),
            ("不得重复", "+++\ndescription='fixture'\ntools=['set_timer', 'set_timer']\n+++\n"),
        ]
        for diagnosis, text in cases:
            with self.subTest(diagnosis=diagnosis, text=text):
                (self.agents / "review.md").write_text(text if diagnosis == "model_id" else
                                                     text.replace("\n+++\n", '\nmodel_id="chat"\n+++\n'),
                                                     encoding="utf-8")
                self.assert_config_error(diagnosis, name="agents/review.md")

    def test_templates_are_valid_when_copied_to_real_role_filenames(self):
        templates = Path(__file__).resolve().parent.parent / "agents"
        (self.agents / "main.md").write_bytes((templates / "main.example.md").read_bytes())
        (self.agents / "review.md").write_bytes((templates / "sub.example.md").read_bytes())
        roles = load_roles(self.workdir)
        self.assertIn(DELEGATE_TOOL_NAME, roles["main"].tools)
        self.assertTrue(roles["review"].description)

    def assert_config_error(self, diagnosis, *, name):
        with self.assertRaises(ConfigError) as caught:
            load_roles(self.workdir)
        self.assertIn(diagnosis, str(caught.exception))
        self.assertIn(name, str(caught.exception))
        self.assertNotIn(str(self.workdir), str(caught.exception))
        self.assertNotIn("synthetic-detail", str(caught.exception))

    def test_role_files_are_bounded_utf8_regular_files_without_symlinks(self):
        path = self.agents / "main.md"
        for diagnosis, create, remove in [
            ("超过 32 KiB", lambda: path.write_bytes(b"x" * (MAX_ROLE_BYTES + 1)), path.unlink),
            ("UTF-8", lambda: path.write_bytes(b"synthetic-detail\xff"), path.unlink),
            ("普通文件", lambda: path.symlink_to(self.workdir / "AGENTS.md"), path.unlink),
            ("普通文件", path.mkdir, path.rmdir),
            ("普通文件", lambda: os.mkfifo(path), path.unlink),
        ]:
            with self.subTest(diagnosis=diagnosis, create=create):
                create()
                try:
                    self.assert_config_error(diagnosis, name="agents/main.md")
                finally:
                    remove()
        self.write_role("main", "测试")
        with patch("agent_companion.runtime.instructions.open", create=True,
                   side_effect=PermissionError("synthetic-detail")):
            self.assert_config_error("无法读取文件", name="agents/main.md")
        actual = self.workdir / "actual"
        self.agents.rename(actual)
        self.agents.symlink_to(actual, target_is_directory=True)
        self.assert_config_error("不接受符号链接", name="agents/")


class RoleRuntimeTests(_RoleFiles, unittest.IsolatedAsyncioTestCase):
    async def turn(self, runtime, tree, text):
        runtime.submit_messages([MessageDelivery(Message("user", text), "trigger")])
        await asyncio.wait_for(tree.wait_idle(), 3)
        self.assertIsNone(runtime.last_failure)

    async def test_role_snapshots_and_tool_candidates_reach_model_requests(self):
        main_body = "\r\n# 主提示\r\n\r\n---\r\n+++\r\n尾行\r\n"
        self.write_role("main", main_body, tools=(DELEGATE_TOOL_NAME,), newline="\r\n")
        self.write_role("research", "调研提示\n", tools=(DELEGATE_TOOL_NAME,), description="收集证据")
        self.write_role("review", "审阅提示\r\n", description="检查缺遗漏")
        (self.agents / "ignored.example.md").write_bytes(b"\xff")
        (self.agents / "notes.txt").write_bytes(b"\xff")
        (self.agents / "nested").mkdir()
        (self.agents / "nested" / "hidden.md").write_bytes(b"\xff")

        def respond(messages):
            match messages[-1].content:
                case "开始":
                    # 已加载的正文在工具续轮不重新读取。
                    self.write_role("main", "新正文", tools=(DELEGATE_TOOL_NAME,))
                    return [_call("create-research", agent="research", prompt="调研")]
                case "调研":
                    return [_call("create-review", agent="review", prompt="审阅")]
                case "审阅":
                    return [_call("forbidden", agent="research", prompt="不应受理")]
            return [TextDelta("结果")]

        model = Model(respond)
        with closing(History(":memory:")) as history:
            async with running_agent(Settings(), history, model, Sink(), session_id="main",
                                     roles=load_roles(self.workdir)) as (runtime, tree):
                await self.turn(runtime, tree, "开始")
        for messages, tools, instructions in model.requests:
            self.assertEqual(messages[0].kind, "user")
            self.assertIn(instructions, {main_body, "调研提示\n", "审阅提示\r\n"})
            if instructions == "审阅提示\r\n":
                self.assertEqual(tools, ())
            else:
                delegate, = tools
                self.assertEqual(delegate.parameters["properties"]["agent"]["enum"], ["research", "review"])
                self.assertIn("收集证据", delegate.description)
                self.assertIn("检查缺遗漏", delegate.description)
        self.assertEqual({instructions for _, _, instructions in model.requests},
                         {main_body, "调研提示\n", "审阅提示\r\n"})
        review_tools = [call for messages, _, instructions in model.requests if instructions == "审阅提示\r\n"
                        for call in tool_calls(messages)]
        self.assertEqual(json.loads(review_tools[-1].result), {"error": "unknown_tool"})
        await self.check_wire_payloads(*model.requests[0])

    async def check_wire_payloads(self, messages, tools, main_body):
        requests = []

        def receive(request):
            requests.append(json.loads(request.content))
            return httpx.Response(400)

        async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as client:
            adapters = (DeepSeek(client), OpenAIResponses(client, model="fixture", api_key="fixture",
                                                          reasoning_effort="low"))
            for adapter in adapters:
                try:
                    with self.assertRaises(ModelError):
                        _ = [part async for part in adapter.stream(messages, tools, thinking=False, instructions=main_body)]
                finally:
                    await adapter.aclose()
        deepseek, openai = requests
        self.assertEqual(deepseek["messages"][0], {"role": "system", "content": main_body})
        self.assertEqual(openai["instructions"], main_body)
        for spec in (deepseek["tools"][0]["function"], openai["tools"][0]):
            self.assertEqual(spec["parameters"]["properties"]["agent"]["enum"], ["research", "review"])
            self.assertIn("research：收集证据", spec["description"])
            self.assertIn("review：检查缺遗漏", spec["description"])

    async def test_main_only_exposes_configured_delegation_without_empty_enum_or_legacy_prompt(self):
        self.write_role("main", "", tools=(DELEGATE_TOOL_NAME,))

        def respond(messages):
            if messages[-1].content == "运行":
                return [_call("create", agent="missing", prompt="查找"),
                        _call("reuse", agent_id="missing", prompt="复用")]
            return [TextDelta("完成")]

        model = Model(respond)
        with closing(History(":memory:")) as history:
            async with running_agent(Settings(), history, model, Sink(), session_id="main",
                                      roles=load_roles(self.workdir)) as (runtime, tree):
                await self.turn(runtime, tree, "运行")
            results = {call.id: json.loads(call.result) for call in tool_calls(row.message for row in history.tail("main", 10))}
        self.assertEqual(results, {"create": {"error": "unknown_agent_role"}, "reuse": {"error": "unknown_child_agent"}})
        messages, tools, instructions = model.requests[0]
        self.assertEqual((messages, instructions), ((Message("user", "运行"),), ""))
        (projected,) = build_request(messages, tools, thinking=False)["tools"]
        spec = projected["function"]
        self.assertEqual(spec["name"], DELEGATE_TOOL_NAME)
        self.assertNotIn("enum", spec["parameters"]["properties"]["agent"])
        self.assertIn("当前没有可用于创建子 Agent 的角色", spec["description"])

    async def test_role_changes_and_missing_file_take_effect_after_restart(self):
        self.write_role("main", "主提示", tools=(DELEGATE_TOOL_NAME,))
        review = self.write_role("review", "旧角色提示", description="审阅")
        database = self.workdir / "history.sqlite3"
        agent_id = None
        child_session_id = None

        def respond(messages):
            match messages[-1].content:
                case "创建":
                    return [_call("new", agent="review", prompt="历史输入")]
                case "继续":
                    return [_call("reuse", agent_id=agent_id, prompt="继续审阅")]
            return [TextDelta("完成")]

        for phase in ("create", "change", "remove", "without_permission", "restore"):
            with self.subTest(phase=phase):
                if phase == "change":
                    self.write_role("review", "修改后的提示", tools=("cancel_timer",), description="审阅")
                elif phase == "remove":
                    review.unlink()
                elif phase == "without_permission":
                    self.write_role("main", "主提示")
                elif phase == "restore":
                    self.write_role("main", "主提示", tools=(DELEGATE_TOOL_NAME,))
                    self.write_role("review", "恢复后的提示", description="审阅")
                model = Model(respond)
                with closing(History(database)) as history:
                    before = history._db.execute("SELECT count(*) FROM delegations").fetchone()[0]
                    async with running_agent(Settings(), history, model, Sink(), session_id="main",
                                             roles=load_roles(self.workdir)) as (runtime, tree):
                        await self.turn(runtime, tree, "创建" if phase == "create" else "继续")
                    tool_result = tool_calls(entry.message for entry in history.tail("main", 30))[-1].result
                    body = json.loads(tool_result)
                    if phase == "create":
                        agent_id = body["agent_id"]
                        child_session_id = history.agent(agent_id).session_id
                    elif phase in ("remove", "without_permission"):
                        expected = "agent_role_unavailable" if phase == "remove" else "unknown_tool"
                        self.assertEqual(body, {"error": expected})
                        self.assertEqual(history._db.execute("SELECT count(*) FROM delegations").fetchone()[0],
                                         before)
                        expected_tools = (DELEGATE_TOOL_NAME,) if phase == "remove" else ()
                        for _, tools, instructions in model.requests:
                            self.assertEqual(instructions, "主提示")
                            self.assertEqual(tuple(tool.name for tool in tools), expected_tools)
                    else:
                        self.assertEqual(body["agent_id"], agent_id)
                        expected = "修改后的提示" if phase == "change" else "恢复后的提示"
                        (messages, tools, _), = [request for request in model.requests if request[2] == expected]
                        self.assertIn("历史输入", [message.text for message in messages])
                        self.assertEqual([tool.name for tool in tools], ["cancel_timer"] if phase == "change" else [])
                    stored = next(entry for entry in history.tail(child_session_id, 30)
                                  if entry.message.text == "历史输入")
                    self.assertEqual(stored.message.source.kind, "delegation")
                    source, = history._db.execute("SELECT source FROM messages WHERE message_id = ?",
                                                  (stored.message_id,)).fetchone()
                    self.assertEqual(json.loads(source), {"kind": "delegation"})
                    delegation_id, = history._db.execute("SELECT delegation_id FROM delegations WHERE agent_id = ?"
                                                         " ORDER BY accepted_at LIMIT 1", (agent_id,)).fetchone()
                    delegation = history.delegation(delegation_id)
                    self.assertEqual((delegation.parent_agent_id, delegation.status,
                                      history.agent_run(delegation.run_id).agent_id),
                                     (history.agent(agent_id).parent_agent_id, "consumed", agent_id))

    async def test_normal_and_compaction_requests_use_same_config_snapshot(self):
        def respond(messages):
            if messages[-1] == _SUMMARIZE:
                self.write_role("main", "新指令")
                return [TextDelta("摘要")]
            return [TextDelta("answer"), InputUsage(230)]

        self.write_role("main", "固定指令")
        model = Model(respond)
        settings = Settings(runtime=RuntimeSettings(context_budget=230, tail_budget=70))
        with closing(History(":memory:")) as history:
            history.append("main", [Message("user", f"old {i} " + "x" * 258) for i in range(3)])
            async with running_agent(settings, history, model, Sink(), session_id="main",
                                     roles=load_roles(self.workdir)) as (runtime, tree):
                await self.turn(runtime, tree, "首次请求")
                await self.turn(runtime, tree, "新" * 120)
                task = runtime._compactions._generating
                if task is not None:
                    await asyncio.wait_for(asyncio.shield(task), 2)
                self.assertIsNone(history.summary("main"))
                await self.turn(runtime, tree, "应用摘要")
                self.assertEqual(history.summary("main").text, "摘要")
        self.assertTrue(any(messages[-1] == _SUMMARIZE for messages, _, _ in model.requests))
        self.assertTrue(all(instructions == "固定指令" for _, _, instructions in model.requests))
