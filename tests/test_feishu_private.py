import asyncio
from collections import defaultdict
from contextlib import ExitStack, asynccontextmanager, closing
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent_companion.__main__ import _serve
from agent_companion.app import private_feishu_identity, run_feishu, running_agent
from agent_companion.channels.feishu.cli import CliError, CommandResult, FeishuDiscovery
from agent_companion.config import (ASRSettings, FeishuSettings, LLMSettings, MiniMaxConnection, Settings,
                                    load_settings)
from agent_companion.protocol import AGENT_REPLY_PREFIX, Message, TextDelta, ToolCall, ToolReady
from agent_companion.runtime.subtasks import AgentTree, DELEGATE_TOOL_NAME, demo_roles
from agent_companion.storage.sqlite import AgentReplyInput, DelegatedInput, FeishuBindingError, History, is_sqlite_lock_error


_OWN = "ou_private_service"
_ACCOUNT_QUERY = ("api", "GET", "/open-apis/authen/v1/user_info")
_CHATS = ("oc_private_a", "oc_private_b", "oc_private_c")
_SENDERS = ("ou_private_a", "ou_private_b", "ou_private_c")


def _chat(chat, sender, *, mode="p2p", peer="user"):
    return {"chat_id": chat, "p2p_target_id": sender, "chat_mode": mode, "p2p_target_type": peer}


def _message(id_, chat, sender, body, *, stamp=100_250, sender_type="user", id_type="open_id", kind="text"):
    return {"message_id": "om_" + id_, "chat_id": chat, "create_time": str(stamp), "msg_type": kind,
            "sender": {"id": sender, "id_type": id_type, "sender_type": sender_type},
            "body": {"content": json.dumps(body if isinstance(body, dict) else {"text": body})}}


class _Runner:
    def __init__(self):
        self.account = _OWN
        self.account_failed = False
        self.checking_account = asyncio.Event()
        self.release_account = asyncio.Event()
        self.release_account.set()
        self.pages = {None: (None, None)}
        self.discovery_failed = False
        self.queries = []
        self.messages = defaultdict(list)
        self.windows = defaultdict(list)
        self.fail = set()
        self.sent = defaultdict(list)
        self.posts = defaultdict(list)
        self.hold_sends = False
        self.hold_destinations = set()
        self.closed_sends = set()
        self.commands = []
        self.downloads = 0

    async def __call__(self, argv, data, timeout):
        self.commands.append(argv[1:4])
        assert argv[-4:] == ("--as", "user", "--format", "json")
        op = argv[1:4]
        if op == _ACCOUNT_QUERY:
            self.checking_account.set()
            await self.release_account.wait()
            if self.account_failed:
                return CommandResult(1, b"", b"")
            result = {"open_id": self.account}
        elif op[:2] == ("im", "+chat-list"):
            assert argv[argv.index("--types") + 1] == "p2p"
            assert argv[argv.index("--user-id-type") + 1] == "open_id"
            cursor = argv[argv.index("--page-token") + 1] if "--page-token" in argv else None
            self.queries.append(cursor)
            if self.discovery_failed:
                return CommandResult(1, b"", b"")
            rows, next_cursor = self.pages[cursor]
            result = {"chats": rows, "has_more": next_cursor is not None, "page_token": next_cursor}
        elif op == ("api", "GET", "/open-apis/im/v1/messages"):
            query = json.loads(argv[argv.index("--params") + 1])
            chat = query["container_id"]
            self.windows[chat].append((int(query["start_time"]), int(query["end_time"])))
            if chat in self.fail:
                return CommandResult(1, b"", b"")
            result = {"has_more": False, "items": self.messages[chat]}
        elif op == ("api", "POST", "/open-apis/im/v1/messages"):
            msg = json.loads(data)
            chat = msg["receive_id"]
            content = json.loads(msg["content"])
            if msg["msg_type"] == "post":
                self.posts[chat].append(content)
                return CommandResult(0, json.dumps({"ok": True, "identity": "user", "data": {
                    "chat_id": chat, "message_id": "om_post"}}).encode(), b"")
            self.sent[chat].append(content["text"])
            if self.hold_sends or chat in self.hold_destinations:
                try:
                    await asyncio.Event().wait()
                finally:
                    self.closed_sends.add(chat)
            result = {"chat_id": chat, "message_id": "om_sent"}
        else:
            assert "+messages-resources-download" in argv
            self.downloads += 1
            return CommandResult(1, b"", b"")
        return CommandResult(0, json.dumps({"ok": True, "identity": "user", "data": result}).encode(), b"")

    def discovery(self):
        return FeishuDiscovery("mock-cli", timeout=1, runner=self)


class _Model:
    def __init__(self):
        self.a_waiting = asyncio.Event()
        self.release_a = asyncio.Event()
        self.calls = []

    async def stream(self, messages, tools, *, thinking):
        last = messages[-1]
        self.calls.append((bool(tools), tuple(m.content for m in messages if m.role == "user")))
        if last.content == "first A":
            self.a_waiting.set()
            await self.release_a.wait()
        if last.content == "first B" and tools:
            yield ToolReady(ToolCall("delegate_b", DELEGATE_TOOL_NAME, '{"agent":"general","prompt":"child B"}'))
        elif last.role == "tool":
            yield TextDelta("delegated B")
        elif isinstance(last.content, str) and last.content.startswith(AGENT_REPLY_PREFIX):
            yield TextDelta("returned B")
        else:
            yield TextDelta("reply: " + last.content)


class _SlowRetiringChild(_Model):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.retiring = asyncio.Event()
        self.release_retirement = asyncio.Event()

    async def stream(self, messages, tools, *, thinking):
        last = messages[-1]
        if tools and last.content == "delegate slow A":
            yield ToolReady(ToolCall("slow-child-a", DELEGATE_TOOL_NAME, '{"agent":"general","prompt":"slow child A"}'))
        elif not tools and last.content == "slow child A":
            self.started.set()
            try:
                await asyncio.Future()
            finally:
                self.retiring.set()
                await self.release_retirement.wait()
        else:
            async for event in super().stream(messages, tools, thinking=thinking):
                yield event


@asynccontextmanager
async def _serve_private(settings, history, model, runner, clock):
    with (patch("agent_companion.app.time", side_effect=lambda: clock[0]),
          patch("agent_companion.feishu_private.time", side_effect=lambda: clock[0]),
          patch("agent_companion.feishu_private._DISCOVERY_SECONDS", 0.02),
          patch("agent_companion.channels.feishu.channel._POLL_SECONDS", 0.02),
          patch("agent_companion.app.running_agent", wraps=running_agent) as agents):
        task = asyncio.create_task(run_feishu(settings, history, model, object(), discovery=runner.discovery(),
                                              roles=demo_roles()))
        try:
            yield agents
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("private chat service exited without cancellation")


async def _eventually(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


class FeishuPrivateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.runner = _Runner()
        self.settings = Settings(feishu=FeishuSettings())

    async def test_configuration_and_database_ownership(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "synthetic.toml"
            config.write_text('[feishu]\n', encoding="utf-8")
            self.assertEqual(load_settings(config).feishu, self.settings.feishu)
            self.assertEqual(self.settings.feishu.mode, "text")
            config.write_text('[feishu]\nmode = "voice"\n', encoding="utf-8")
            self.assertEqual(load_settings(config).feishu, FeishuSettings(mode="voice"))

            # 已有库的账号指纹不随配置项移除而改变。
            owner = sha256(f'["private","{_OWN}"]'.encode()).hexdigest()
            self.assertEqual(private_feishu_identity(_OWN), owner)
            with closing(History(Path(tmp) / "private.db", feishu_private_owner=owner)) as history:
                history.bind_feishu_private(owner, started_ms=99_000)
                self.runner.account = "ou_unexpected_account"
                with self.assertRaisesRegex(FeishuBindingError, "服务账号与数据库绑定不一致"):
                    await run_feishu(self.settings, history, _Model(), object(), discovery=self.runner.discovery(),
                                     roles=demo_roles())
                self.assertEqual(self.runner.commands, [_ACCOUNT_QUERY])
                self.assertEqual(history.check_feishu_private(owner), 99_000)
                self.assertEqual(self.runner.queries, [])
                self.assertEqual(self.runner.sent, {})
            unbound = Path(tmp) / "unbound.db"
            with closing(History(unbound)) as history:
                history.append("main", [Message("user", "unbound history")])
            original = unbound.read_bytes()
            with self.assertRaisesRegex(FeishuBindingError, "现有数据库有历史"):
                History(unbound, feishu_private_owner=owner)
            self.assertEqual(unbound.read_bytes(), original)

    async def test_lazy_sessions_parallel_delivery_and_restart(self):
        a, b, c = _CHATS
        sa, sb, sc = _SENDERS
        runner, model, clock = self.runner, _Model(), [100.25]
        with TemporaryDirectory() as tmp:
            database = Path(tmp) / "private.db"
            with closing(History(database)) as history:
                async with _serve_private(self.settings, history, model, runner, clock) as agents:
                    await _eventually(lambda: runner.queries == [None])
                    agents.assert_not_called()
                    runner.pages = {None: ([_chat(a, sa), _chat("oc_group", sa, mode="group"),
                                            _chat("oc_self", _OWN), _chat("oc_bot", "ou_bot", peer="bot")], "next"),
                                    "next": ([_chat(b, sb), _chat(a, sa), _chat("oc_empty", "ou_empty")], None)}
                    runner.messages[a] = [_message("old", a, sa, "old A", stamp=100_249),
                                          _message("first_a", a, sa, "first A"),
                                          _message("wrong_peer", a, sb, "wrong peer"),
                                          _message("own", a, _OWN, "own reply"),
                                          _message("wrong_chat", b, sa, "wrong chat"),
                                          _message("not_user", a, sa, "bot", sender_type="app"),
                                          _message("wrong_id_type", a, sa, "wrong ID type", id_type="user_id")]
                    runner.messages[b] = [_message("first_b", b, sb, "first B"),
                                          _message("bad_voice", b, sb, {"file_key": "file_synthetic"}, kind="audio")]
                    clock[0] = 101.1
                    await asyncio.wait_for(model.a_waiting.wait(), 3)
                    await _eventually(lambda: "returned B" in runner.sent[b])
                    self.assertEqual(runner.sent[b].count("语音接收失败，请重新发送。"), 1)
                    self.assertEqual(runner.sent[a], [])
                    self.assertEqual(set(runner.windows), {a, b, "oc_empty"})
                    self.assertIn("next", runner.queries)
                    self.assertEqual(agents.call_count, 2)
                    session_a, session_b = [next(call.kwargs["session_id"] for call in agents.call_args_list
                                                  if call.args[-1]._cli._chat_id == chat) for chat in (a, b)]
                    model.release_a.set()
                    await _eventually(lambda: runner.sent[a] == ["reply: first A"])
                    self.assertEqual([e.message.content for e in history.read_after(session_a)], ["first A", "reply: first A"])
                    self.assertIn((False, ("child B",)), model.calls)
                    self.assertTrue(all("first A" not in inputs or inputs == ("first A",) for _, inputs in model.calls))

                # 模拟进程在已保存入站但尚未移交 Runtime 时退出。
                history.record_inbound(session_b, "feishu-private", 101,
                                       [("om_pending", Message("user", "pending B"))])

            resumed = _Model()
            runner.pages["next"] = ([_chat(b, sb), _chat(c, sc), _chat("oc_empty", "ou_empty")], None)
            runner.messages[c] = [_message("early_c", c, sc, "early C", stamp=100_251)]
            clock[0] = 105.0
            with closing(History(database)) as history:
                async with _serve_private(self.settings, history, resumed, runner, clock) as agents:
                    await _eventually(lambda: "reply: pending B" in runner.sent[b] and "reply: early C" in runner.sent[c])
                    self.assertEqual(runner.sent[a], ["reply: first A"])
                    self.assertEqual(runner.sent[b].count("returned B"), 1)
                    self.assertEqual(len(resumed.calls), 2)
                    self.assertEqual(agents.call_count, 2)
                    self.assertIn(session_b, [call.kwargs["session_id"] for call in agents.call_args_list])
                    b_inputs = next(inputs for _, inputs in resumed.calls if inputs[-1] == "pending B")
                    self.assertEqual(b_inputs[0], "first B")
                    self.assertIn("reply: pending B", [e.message.content for e in history.read_after(session_b)])
                    self.assertIn((100, 105), runner.windows[c])
                    self.assertEqual(history.pending_inbound(session_b, "feishu-private"), [])

    async def test_discovery_cycle_error_does_not_stop_routes_and_conflicts_do_not_rebind(self):
        a, b, c = _CHATS
        sa, sb, sc = _SENDERS
        runner, model, clock = self.runner, _Model(), [100.25]
        runner.pages = {None: ([_chat(a, sa), _chat(b, sb)], None)}
        with closing(History(":memory:")) as history:
            async with _serve_private(self.settings, history, model, runner, clock) as agents:
                await _eventually(lambda: runner.queries)
                runner.pages = {None: ([], "loop"), "loop": ([_chat(c, sc)], "loop")}
                with self.assertLogs("agent_companion.feishu_private", level="WARNING") as logs:
                    await _eventually(lambda: runner.queries.count("loop") >= 2)
                    clock[0] = 102.0
                    runner.fail.add(a)
                    runner.messages[b] = [_message("independent", b, sb, "independent B")]
                    await _eventually(lambda: runner.sent[b] == ["reply: independent B"])
                    runner.pages = {None: ([_chat(a, sc), _chat(c, sa), _chat(b, sb),
                                            _chat("oc_conflict1", "ou_duplicate"),
                                            _chat("oc_conflict2", "ou_duplicate")], None)}
                    await _eventually(lambda: any("route conflict" in text for text in logs.output))
                self.assertTrue(any("cli_chat_page_cycle" in text for text in logs.output))
                self.assertNotIn(a, "\n".join(logs.output))
                self.assertNotIn(c, runner.windows)
                self.assertNotIn("oc_conflict1", runner.windows)
                self.assertEqual(agents.call_count, 1)

    async def test_malformed_discovery_protocol_is_rejected(self):
        a, sa = _CHATS[0], _SENDERS[0]
        chat = _chat(a, sa)
        bad_rows = [dict(chat, **{field: value}) for field, values in (
            ("p2p_target_id", (None, "oc_wrong_namespace", [])),
            ("chat_mode", (None, "unknown", [])),
            ("p2p_target_type", (None, "unknown", {})),
        ) for value in values]
        for rows, cursor in (([_chat(_CHATS[0], _SENDERS[0])], ["not a cursor"]), ([None], None),
                             ([{}], None), ([_chat("bad_chat_id", _SENDERS[1])], None),
                             *(([row], None) for row in bad_rows)):
            with self.subTest(rows=rows, cursor=cursor):
                self.runner.pages = {None: (rows, cursor)}
                with self.assertRaisesRegex(CliError, "cli_invalid_chat_page"):
                    await self.runner.discovery().private_chats(_OWN)
        self.runner.pages = {None: (None, None)}
        self.assertEqual(await self.runner.discovery().private_chats(_OWN), set())

    async def test_shutdown_awaits_outbound_closure_in_every_route(self):
        a, b, _ = _CHATS
        sa, sb, _ = _SENDERS
        runner, model, clock = self.runner, _Model(), [101.0]
        runner.pages = {None: ([_chat(a, sa), _chat(b, sb)], None)}
        runner.hold_sends = True
        runner.messages[a] = [_message("shutdown_a", a, sa, "shutdown A", stamp=101_001)]
        runner.messages[b] = [_message("shutdown_b", b, sb, "shutdown B", stamp=101_002)]
        with closing(History(":memory:")) as history:
            async with _serve_private(self.settings, history, model, runner, clock):
                await _eventually(lambda: runner.queries)
                clock[0] = 102.0
                await _eventually(lambda: set(runner.sent) == {a, b})
            self.assertEqual(runner.closed_sends, {a, b})

    async def test_slow_retirement_does_not_delay_other_chat_activation_or_revocation(self):
        a, b, _ = _CHATS
        sa, sb, _ = _SENDERS
        runner, model, clock = self.runner, _SlowRetiringChild(), [100.25]
        runner.pages = {None: ([_chat(a, sa)], None)}
        runner.messages[a] = [_message("delegate_a", a, sa, "delegate slow A")]
        runner.hold_destinations.add(b)
        session_a = "feishu-private:" + sha256(sa.encode()).hexdigest()
        close = AgentTree.close

        async def close_with_fault(subtasks):
            await close(subtasks)
            if subtasks._root.session_id == session_a:
                raise RuntimeError("synthetic exception body not for logs")

        with (closing(History(":memory:")) as history, patch.object(AgentTree, "close", close_with_fault)):
            async with _serve_private(self.settings, history, model, runner, clock):
                try:
                    await _eventually(lambda: runner.queries)
                    clock[0] = 101.0
                    await asyncio.wait_for(model.started.wait(), 3)
                    runner.messages[b] = [_message("independent_b", b, sb, "independent B", stamp=101_100)]
                    clock[0] = 102.0
                    runner.pages = {None: ([_chat(b, sb)], None)}
                    await asyncio.wait_for(model.retiring.wait(), 3)
                    await _eventually(lambda: runner.sent[b] == ["reply: independent B"])
                    runner.pages = {None: ([], None)}
                    await _eventually(lambda: b in runner.closed_sends)
                    with self.assertLogs("agent_companion.feishu_private", level="WARNING") as logs:
                        model.release_retirement.set()
                        await _eventually(lambda: any("retirement failed" in line for line in logs.output))
                    self.assertEqual(len(logs.records), 1)
                    self.assertTrue(logs.records[0].getMessage().endswith("error_type=RuntimeError"))
                    self.assertNotIn("synthetic exception body", logs.records[0].getMessage())
                    self.assertIsNone(logs.records[0].exc_info)
                    count = len(runner.queries)
                    await _eventually(lambda: len(runner.queries) > count)
                finally:
                    model.release_retirement.set()

    async def test_retiring_session_requires_a_new_verified_snapshot_after_old_exit(self):
        a, sa = _CHATS[0], _SENDERS[0]
        runner, model, clock = self.runner, _SlowRetiringChild(), [100.25]
        verified = {None: ([_chat(a, sa)], None)}
        runner.pages = verified
        runner.messages[a] = [_message("delegate_a", a, sa, "delegate slow A")]
        with closing(History(":memory:")) as history:
            with patch.object(history, "bind_feishu_private_agent", wraps=history.bind_feishu_private_agent) as bind:
                async with _serve_private(self.settings, history, model, runner, clock) as agents:
                    try:
                        await _eventually(lambda: runner.queries)
                        clock[0] = 101.0
                        await asyncio.wait_for(model.started.wait(), 3)
                        session = agents.call_args.kwargs["session_id"]
                        await _eventually(lambda: any(e.message.role == "tool" for e in history.read_after(session)))
                        accepted = next(json.loads(e.message.content) for e in history.read_after(session)
                                        if e.message.role == "tool")
                        runner.pages = {None: ([], None)}
                        await asyncio.wait_for(model.retiring.wait(), 3)

                        runner.messages[a].append(_message("after_a", a, sa, "fresh A", stamp=101_100))
                        clock[0] = 102.0
                        runner.pages = verified
                        count = len(runner.queries)
                        await _eventually(lambda: len(runner.queries) >= count + 2)
                        self.assertEqual((bind.call_count, agents.call_count), (1, 1))

                        runner.pages = {None: ([], None)}
                        model.release_retirement.set()
                        run_id = history.delegation(accepted["delegation_id"]).run_id
                        await _eventually(lambda: history.agent_run(run_id).status == "interrupted")
                        count = len(runner.queries)
                        await _eventually(lambda: len(runner.queries) >= count + 2)
                        self.assertEqual((bind.call_count, agents.call_count), (1, 1))

                        runner.pages = verified
                        await _eventually(lambda: "reply: fresh A" in runner.sent[a])
                        self.assertEqual((bind.call_count, agents.call_count), (2, 2))
                    finally:
                        model.release_retirement.set()

    async def test_revoked_route_cancels_queued_and_inflight_work_then_restores_only_pending_input(self):
        class PausedModel(_Model):
            def __init__(self):
                super().__init__()
                self.waiting = {label: asyncio.Event() for label in ("hold main A", "hold child A")}
                self.cancelled = {label: asyncio.Event() for label in self.waiting}

            async def stream(self, messages, tools, *, thinking):
                last = messages[-1]
                if last.content in self.waiting:
                    self.waiting[last.content].set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        self.cancelled[last.content].set()
                if last.content == "start child A" and tools:
                    yield ToolReady(ToolCall("delegate_a", DELEGATE_TOOL_NAME, '{"agent":"general","prompt":"hold child A"}'))
                elif last.tool_call_id == "delegate_a":
                    yield TextDelta("queued reply A")
                else:
                    async for part in super().stream(messages, tools, thinking=thinking):
                        yield part

        a, b, _ = _CHATS
        sa, sb, _ = _SENDERS
        runner, model, clock = self.runner, PausedModel(), [100.25]
        chats = [_chat(a, sa), _chat(b, sb)]
        runner.pages = {None: (chats, None)}
        runner.hold_destinations.add(a)
        runner.messages[a] = [_message("initial_a", a, sa, "initial A")]
        with closing(History(":memory:")) as history:
            async with _serve_private(self.settings, history, model, runner, clock) as agents:
                await _eventually(lambda: runner.queries)
                clock[0] = 101.0
                await _eventually(lambda: runner.sent[a] == ["reply: initial A"])
                session_a = agents.call_args.kwargs["session_id"]
                channel_a = agents.call_args.args[-1]

                runner.messages[a].append(_message("start_child", a, sa, "start child A", stamp=101_100))
                clock[0] = 102.0
                await _eventually(lambda: model.waiting["hold child A"].is_set() and len(channel_a._pending) == 1)
                tool = next(e.message for e in history.read_after(session_a) if e.message.role == "tool")
                child = json.loads(tool.content)
                runner.messages[a].append(_message("hold_main", a, sa, "hold main A", stamp=102_100))
                clock[0] = 103.0
                await asyncio.wait_for(model.waiting["hold main A"].wait(), 3)
                runner.messages[a].append(_message("queued_a", a, sa, "pending A", stamp=103_100))
                clock[0] = 104.0
                await _eventually(lambda: len(history.pending_inbound(session_a, "feishu-private")) == 1)

                # 普通 CLI 错误、坏页（首个分页已遗漏 A）和循环游标都不能撤销旧路由。
                runner.discovery_failed = True
                count = len(runner.queries)
                await _eventually(lambda: len(runner.queries) > count)
                runner.pages = {None: ([_chat(b, sb)], "bad"), "bad": ([None], None)}
                runner.discovery_failed = False
                await _eventually(lambda: "bad" in runner.queries)
                runner.pages = {None: ([_chat(b, sb)], "missing-id"), "missing-id": ([{}], None)}
                await _eventually(lambda: "missing-id" in runner.queries)
                for field, value in (("p2p_target_id", "invalid_id"), ("p2p_target_id", None),
                                     ("chat_mode", None), ("p2p_target_type", [])):
                    row = _chat(a, sa)
                    if value is None:
                        row.pop(field)
                    else:
                        row[field] = value
                    with self.assertLogs("agent_companion.feishu_private", level="WARNING") as logs:
                        runner.pages = {None: (chats, "ambiguous"), "ambiguous": ([row], None)}
                        await _eventually(lambda: any("cli_invalid_chat_page" in line for line in logs.output))
                    self.assertNotIn(a, runner.closed_sends)
                runner.pages = {None: ([_chat(b, sb)], "cycle"), "cycle": ([], "cycle")}
                await _eventually(lambda: runner.queries.count("cycle") >= 2)
                self.assertFalse(runner.closed_sends)
                self.assertFalse(any(event.is_set() for event in model.cancelled.values()))
                self.assertEqual(len(channel_a._pending), 1)

                runner.pages = {None: ([_chat(a, sa, mode="group"), _chat(b, sb)], None)}
                await _eventually(lambda: a in runner.closed_sends and all(event.is_set() for event in model.cancelled.values()))
                self.assertFalse(channel_a._pending)
                run_id = history.delegation(child["delegation_id"]).run_id
                await _eventually(lambda: history.agent_run(run_id).status == "interrupted")
                self.assertIsNone(history.agent_run(run_id).reply_enqueued_at)
                before = len(runner.windows[a])
                runner.messages[b].append(_message("revoked_a_b", b, sb, "first B", stamp=104_100))
                clock[0] = 105.0
                await _eventually(lambda: "returned B" in runner.sent[b])
                self.assertEqual(len(runner.windows[a]), before)
                self.assertEqual(runner.sent[a], ["reply: initial A"])
                self.assertEqual(len(history.pending_inbound(session_a, "feishu-private")), 1)

                runner.hold_destinations.remove(a)
                runner.pages = {None: (chats, None)}
                clock[0] = 106.0
                await _eventually(lambda: runner.sent[a] == ["reply: initial A", "reply: pending A"])
                self.assertEqual(agents.call_args.kwargs["session_id"], session_a)
                self.assertEqual(agents.call_count, 3)
                self.assertEqual(history.pending_inbound(session_a, "feishu-private"), [])
                a_user = [entry.message.content for entry in history.read_after(session_a) if entry.message.role == "user"]
                self.assertEqual(a_user, ["initial A", "start child A", "hold main A", "pending A"])
                self.assertEqual(sum(inputs[-1] == "pending A" for _, inputs in model.calls), 1)

    async def test_successful_snapshot_pauses_disappeared_mismatched_and_conflicting_routes(self):
        a, b, c = _CHATS
        sa, sb, sc = _SENDERS
        for label, invalid in (
            ("gone", []), ("new_peer_same_chat", [_chat(a, sc)]),
            ("conflicting_chat_type", [_chat(a, sa), _chat(a, sa, mode="group")]),
            ("topic", [{"chat_id": a, "chat_mode": "topic"}]),
            ("bot", [{"chat_id": a, "chat_mode": "p2p", "p2p_target_type": "bot"}]),
            ("self", [_chat(a, _OWN)]),
            ("conflicting", [_chat(a, sa), _chat(c, sa)]),
        ):
            with self.subTest(label=label), closing(History(":memory:")) as history:
                runner, model, clock = _Runner(), _Model(), [100.25]
                runner.pages = {None: ([_chat(a, sa), _chat(b, sb)], None)}
                runner.messages[a] = [_message("inflight", a, sa, "inflight A")]
                runner.hold_destinations.add(a)
                async with _serve_private(self.settings, history, model, runner, clock):
                    await _eventually(lambda: runner.queries)
                    clock[0] = 101.0
                    await _eventually(lambda: runner.sent[a])
                    runner.pages = {None: (invalid + [_chat(b, sb)], None)}
                    await _eventually(lambda: a in runner.closed_sends)
                    before = len(runner.windows[a])
                    runner.messages[b] = [_message("unaffected", b, sb, "unaffected B", stamp=101_100)]
                    clock[0] = 102.0
                    await _eventually(lambda: runner.sent[b] == ["reply: unaffected B"])
                    self.assertEqual(len(runner.windows[a]), before)
                    self.assertNotIn(c, runner.windows)

    async def test_failed_account_verification_leaves_new_database_available_and_keeps_startup_message(self):
        runner, model, clock = self.runner, _Model(), [100.25]
        owner = private_feishu_identity(_OWN)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "new.db"
            with closing(History(path)) as history:
                runner.account = "invalid"
                with self.assertRaisesRegex(CliError, "cli_invalid_account"):
                    await run_feishu(self.settings, history, model, object(), discovery=runner.discovery(),
                                     roles=demo_roles())
                self.assertIsNone(history.check_feishu_private(owner))
                runner.account_failed = True
                with self.assertRaisesRegex(CliError, "cli_result_unknown"):
                    await run_feishu(self.settings, history, model, object(), discovery=runner.discovery(),
                                     roles=demo_roles())
                self.assertIsNone(history.check_feishu_private(owner))
                self.assertFalse(runner.queries)
            with closing(History(path, feishu_private_owner=owner)) as history:
                a, sa = _CHATS[0], _SENDERS[0]
                runner.account = _OWN
                runner.account_failed = False
                runner.checking_account.clear()
                runner.release_account.clear()
                self_chat = _CHATS[1]
                runner.pages = {None: ([_chat(a, sa), _chat(self_chat, _OWN)], None)}
                runner.messages[a] = [_message("too_early", a, sa, "old A", stamp=100_249),
                                      _message("from_self", a, _OWN, "ignore self", stamp=100_251),
                                      _message("during_auth", a, sa, "during verification", stamp=100_251)]
                async with _serve_private(self.settings, history, model, runner, clock):
                    await asyncio.wait_for(runner.checking_account.wait(), 3)
                    self.assertIsNone(history.check_feishu_private(owner))
                    self.assertFalse(runner.queries)
                    clock[0] = 102.0
                    runner.release_account.set()
                    await _eventually(lambda: runner.sent[a] == ["reply: during verification"])
                    self.assertNotIn(self_chat, runner.windows)
                    self.assertEqual(history.check_feishu_private(owner), 100_250)

    async def test_production_checks_owner_and_uses_single_cli_verification_before_providers(self):
        runner = self.runner
        owner = private_feishu_identity(_OWN)
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            workdir = Path(tmp)
            agents = workdir / "agents"
            agents.mkdir()
            (agents / "main.md").write_text("+++\ntools=[]\n+++\n离线角色", encoding="utf-8")
            settings = replace(self.settings, database=workdir / "production.db",
                               llm=LLMSettings("openai_responses", "disabled", model="synthetic", api_key="synthetic"),
                               asr=ASRSettings(MiniMaxConnection("synthetic"), max_audio_seconds=120))
            with (patch("agent_companion.__main__.Path.cwd", return_value=workdir),
                  patch("agent_companion.app.FeishuDiscovery", return_value=runner.discovery()) as discover,
                  patch("agent_companion.app.run_private_chats") as run_chats,
                  patch("agent_companion.__main__.httpx.AsyncClient") as client,
                  patch.object(History, "replay_salt", autospec=True) as get_salt):
                runner.account_failed = True
                with self.assertRaisesRegex(CliError, "cli_result_unknown"):
                    await _serve(settings)
                client.assert_not_called()
                get_salt.assert_not_called()
                run_chats.assert_not_awaited()
                self.assertFalse(settings.database.exists())

                def check_before_salt(history):
                    self.assertIsNotNone(history.check_feishu_private(owner))
                    self.assertEqual(runner.commands, [_ACCOUNT_QUERY])
                    return bytes(32)

                runner.account_failed = False
                runner.commands.clear()
                get_salt.side_effect = check_before_salt
                await _serve(settings)
                self.assertEqual(runner.commands, [_ACCOUNT_QUERY])
                run_chats.assert_awaited_once()
                self.assertEqual(run_chats.await_args.kwargs["own_id"], _OWN)
                original = settings.database.read_bytes()
                client.reset_mock()
                get_salt.reset_mock()
                discover.reset_mock()
                run_chats.reset_mock()
                runner.commands.clear()
                runner.account = "ou_wrong_service"
                with self.assertRaisesRegex(FeishuBindingError, "服务账号与数据库绑定不一致"):
                    await _serve(settings)
                self.assertEqual(runner.commands, [_ACCOUNT_QUERY])
                self.assertEqual(settings.database.read_bytes(), original)
                discover.assert_called_once()
                run_chats.assert_not_awaited()
                client.assert_not_called()
                get_salt.assert_not_called()

    async def test_inbox_and_registration_busy_retry_without_blocking_other_routes(self):
        a, b, c = _CHATS
        sa, sb, sc = _SENDERS
        session_a = "feishu-private:" + sha256(sa.encode()).hexdigest()
        session_c = "feishu-private:" + sha256(sc.encode()).hexdigest()
        runner, model, clock = self.runner, _Model(), [100.25]
        runner.pages = {None: ([_chat(a, sa), _chat(b, sb)], None)}
        a_attempts, c_attempts = [], []
        busy_a = busy_c = True
        with closing(History(":memory:")) as history:
            store, bind = history.record_inbound, history.bind_feishu_private_agent

            def save_window(session, source, through, items):
                if session == session_a:
                    a_attempts.append((through, tuple(items)))
                    if busy_a:
                        exc = sqlite3.OperationalError("synthetic storage contention")
                        exc.sqlite_errorcode = sqlite3.SQLITE_BUSY_SNAPSHOT
                        raise exc
                return store(session, source, through, items)

            def register(session, chat):
                if session == session_c:
                    c_attempts.append(session)
                    if busy_c:
                        raise sqlite3.OperationalError("database table is locked")
                return bind(session, chat)

            with (patch.object(history, "record_inbound", side_effect=save_window),
                  patch.object(history, "bind_feishu_private_agent", side_effect=register),
                  self.assertLogs("agent_companion", level="WARNING") as logs):
                async with _serve_private(self.settings, history, model, runner, clock):
                    await _eventually(lambda: runner.queries)
                    runner.messages[a] = [_message("a_once", a, sa, "saved once A"),
                                          _message("a_failed_audio", a, sa, {"file_key": "file_a"}, kind="audio")]
                    runner.messages[b] = [_message("b_first", b, sb, "first B")]
                    runner.messages[c] = [_message("c_first", c, sc, "late route C")]
                    runner.pages = {None: ([_chat(a, sa), _chat(b, sb), _chat(c, sc)], None)}
                    clock[0] = 101.0
                    await _eventually(lambda: len(a_attempts) >= 2 and len(c_attempts) >= 2 and "returned B" in runner.sent[b])
                    self.assertEqual(runner.windows[a], [(100, 101)])
                    self.assertFalse(runner.sent[a])
                    self.assertNotIn(c, runner.windows)
                    self.assertFalse(history.inbound_seen(session_a, "feishu-private", "om_a_once"))
                    self.assertFalse(history.read_after(session_a))
                    self.assertEqual(history.inbound_window(session_a, "feishu-private", 100), (100, 100))
                    fetched, runner.messages[a] = runner.messages[a], []  # 不依赖 CLI 再返回已拉取的窗口。
                    busy_c = False
                    await _eventually(lambda: runner.sent[c] == ["reply: late route C"])
                    self.assertFalse(runner.sent[a])
                    busy_a = False
                    await _eventually(lambda: "reply: saved once A" in runner.sent[a]
                                      and "语音接收失败，请重新发送。" in runner.sent[a])
                    runner.messages[a] = fetched  # 下一窗口再次收到同一事件，也只能处理一次。
                    clock[0] = 102.0
                    await _eventually(lambda: (100, 102) in runner.windows[a])
                    self.assertTrue(all(item == a_attempts[0] for item in a_attempts if item[0] == 101))
                    self.assertEqual(runner.sent[a].count("reply: saved once A"), 1)
                    self.assertEqual(runner.sent[a].count("语音接收失败，请重新发送。"), 1)
                    self.assertEqual(sum(inputs[-1] == "saved once A" for _, inputs in model.calls), 1)
                    self.assertEqual(runner.downloads, 1)
                self.assertTrue(any("inbox storage busy" in line for line in logs.output))
                self.assertTrue(any("route storage busy" in line for line in logs.output))

    async def test_runtime_consumes_pending_batches_in_order_while_other_route_progresses(self):
        a, b, _ = _CHATS
        sa, sb, _ = _SENDERS
        session_a = "feishu-private:" + sha256(sa.encode()).hexdigest()
        runner, model, clock = self.runner, _Model(), [100.25]
        runner.pages = {None: ([_chat(a, sa), _chat(b, sb)], None)}
        blocked, attempts, committed = True, [], []
        with closing(History(":memory:")) as history:
            consume = history._consume_inbound

            def consume_batch(session, batch):
                if session != session_a:
                    return consume(session, batch)
                attempts.append(batch)
                if blocked:
                    error = sqlite3.OperationalError("synthetic contention")
                    error.sqlite_errorcode = sqlite3.SQLITE_BUSY_SNAPSHOT
                    raise error
                ids = consume(session, batch)
                committed.append(batch)
                return ids

            with patch.object(history, "_consume_inbound", side_effect=consume_batch):
                async with _serve_private(self.settings, history, model, runner, clock):
                    await _eventually(lambda: runner.queries)
                    runner.messages[a] = [_message("a1", a, sa, "queued A1")]
                    clock[0] = 101.0
                    await _eventually(lambda: attempts)
                    first = attempts[0]
                    runner.messages[a].append(_message("a2", a, sa, "queued A2", stamp=101_100))
                    runner.messages[b] = [_message("b1", b, sb, "independent B", stamp=101_100)]
                    clock[0] = 102.0
                    await _eventually(lambda: len(attempts) >= 2 and runner.sent[b] == ["reply: independent B"])
                    pending = history.pending_inbound(session_a, "feishu-private")
                    self.assertEqual(len(pending), 2)
                    self.assertEqual(pending[0], first)
                    self.assertEqual(set(attempts), {first})
                    self.assertEqual(history.read_after(session_a), [])
                    self.assertEqual(runner.sent[a], [])

                    blocked = False
                    await _eventually(lambda: len(runner.sent[a]) >= 2)
                    self.assertEqual(runner.sent[a], ["reply: queued A1", "reply: queued A2"])
                    self.assertEqual(committed, pending)
                    self.assertEqual(history.pending_inbound(session_a, "feishu-private"), [])
                    self.assertEqual(history.inbound_window(session_a, "feishu-private", 100), (100, 102))
                    self.assertEqual([e.message.content for e in history.read_after(session_a)],
                                     ["queued A1", "reply: queued A1", "queued A2", "reply: queued A2"])
                    self.assertEqual([inputs for _, inputs in model.calls if inputs[0] == "queued A1"],
                                     [("queued A1",), ("queued A1", "queued A2")])

    async def test_runtime_retries_reply_and_subtask_transactions_without_repeating_work(self):
        class Model(_Model):
            async def stream(self, messages, tools, *, thinking):
                if messages[-1].content == "first B":
                    yield TextDelta("tool-stage B")
                async for event in super().stream(messages, tools, thinking=thinking):
                    yield event

        b, sb = _CHATS[1], _SENDERS[1]
        session_b = "feishu-private:" + sha256(sb.encode()).hexdigest()
        runner, model, clock = self.runner, Model(), [100.25]
        runner.pages = {None: ([_chat(b, sb)], None)}
        blocked = {"accept", "start", "tool_reply", "tool_input", "finish", "mark", "result_input"}
        attempts, committed = defaultdict(list), defaultdict(list)
        with closing(History(":memory:")) as history, ExitStack() as patches:
            def intercept(method, stage_for):
                write = getattr(history, method)

                def attempt(*args, **kwargs):
                    stage = stage_for(*args, **kwargs)
                    if stage:
                        attempts[stage].append((args, kwargs))
                        if stage in blocked:
                            raise sqlite3.OperationalError("database table is locked")
                    result = write(*args, **kwargs)
                    if stage:
                        committed[stage].append((args, kwargs))
                    return result

                patches.enter_context(patch.object(history, method, side_effect=attempt))

            def input_stage(agent_id, run_id, rows, *, start):
                if any(isinstance(row, DelegatedInput) for row in rows):
                    return "start"
                if any(isinstance(row, AgentReplyInput) for row in rows):
                    return "result_input"
                return "tool_input" if not start else None

            def finish_stage(run_id, *, result, error):
                return "finish" if history.agent(history.agent_run(run_id).agent_id).parent_agent_id else None

            intercept("accept_delegation", lambda *args, **kwargs: "accept")
            intercept("append_agent_inputs", input_stage)
            intercept("finish_agent_run", finish_stage)
            intercept("mark_agent_reply_enqueued", lambda _: "mark")
            intercept("finalize_reply", lambda outcome, rows: "tool_reply" if rows and rows[0].tool_calls else None)
            delegate = patches.enter_context(patch.object(AgentTree, "_delegate", autospec=True,
                                                           side_effect=AgentTree._delegate))
            async with _serve_private(self.settings, history, model, runner, clock):
                await _eventually(lambda: runner.queries)
                runner.messages[b] = [_message("tool", b, sb, "first B")]
                clock[0] = 101.0
                await _eventually(lambda: len(attempts["accept"]) >= 2)
                blocked.remove("accept")
                await _eventually(lambda: len(attempts["tool_reply"]) >= 2 and len(attempts["start"]) >= 2)
                self.assertEqual([e.message.role for e in history.read_after(session_b)], ["user"])
                self.assertEqual(runner.sent[b], [])
                self.assertEqual(model.calls, [(True, ("first B",))])
                # 工具轮次之间吸收排队的用户输入时也只等当前事务。
                runner.messages[b].append(_message("during_tool", b, sb, "during tool B", stamp=101_100))
                clock[0] = 102.0
                await _eventually(lambda: history.pending_inbound(session_b, "feishu-private"))
                blocked.remove("tool_reply")
                await _eventually(lambda: len(attempts["tool_input"]) >= 2)
                self.assertEqual(model.calls, [(True, ("first B",))])
                blocked.remove("tool_input")
                await _eventually(lambda: len(runner.sent[b]) >= 2)
                tool = next(e.message for e in history.read_after(session_b) if e.message.role == "tool")
                accepted = json.loads(tool.content)
                delegation_id = accepted["delegation_id"]
                child = history.agent(accepted["agent_id"]).session_id
                self.assertEqual(history.delegation(delegation_id).status, "queued")
                self.assertIsNone(history.delegation(delegation_id).run_id)

                blocked.remove("start")
                await _eventually(lambda: len(attempts["finish"]) >= 4)
                run_id = history.delegation(delegation_id).run_id
                self.assertEqual(history.agent_run(run_id).status, "running")
                self.assertEqual(attempts["result_input"], [])
                blocked.remove("finish")
                await _eventually(lambda: len(attempts["result_input"]) >= 2 and len(attempts["mark"]) >= 2)
                self.assertEqual(history.agent_run(run_id).status, "completed")
                self.assertIsNone(history.agent_run(run_id).reply_enqueued_at)
                self.assertEqual(runner.sent[b], ["tool-stage B", "reply: during tool B"])
                blocked.remove("result_input")
                await _eventually(lambda: "returned B" in runner.sent[b])
                # 标记仍锁住时，主 Runtime 已能消费回投；释放标记不能重复回投。
                self.assertIsNone(history.agent_run(run_id).reply_enqueued_at)
                blocked.remove("mark")
                await _eventually(lambda: history.agent_run(run_id).reply_enqueued_at is not None)

                delegate.assert_awaited_once()
                self.assertEqual(len(model.calls), 4)  # 主会话三轮、子会话一轮
                self.assertEqual(runner.sent[b], ["tool-stage B", "reply: during tool B", "returned B"])
                self.assertEqual([e.message.role for e in history.read_after(session_b)],
                                 ["user", "assistant", "tool", "user", "assistant", "user", "assistant"])
                self.assertEqual([e.message.content for e in history.read_after(child)], ["child B", "reply: child B"])
                for stage in attempts:
                    self.assertEqual(committed[stage], [attempts[stage][0]], stage)
                    self.assertTrue(all(args == committed[stage][0] for args in attempts[stage]), stage)
                outcome, rows = attempts["tool_reply"][0][0]
                self.assertTrue(all(args[0] is outcome and args[1] is rows for args, _ in attempts["tool_reply"]))

    async def test_route_revocation_cancels_uncommitted_reply_even_if_storage_stays_locked(self):
        a, b, _ = _CHATS
        sa, sb, _ = _SENDERS
        session_a = "feishu-private:" + sha256(sa.encode()).hexdigest()
        runner, model, clock = self.runner, _Model(), [100.25]
        chats = [_chat(a, sa), _chat(b, sb)]
        runner.pages = {None: (chats, None)}
        attempts, shutdowns = [], []
        with closing(History(":memory:")) as history:
            finalize, interrupt = history.finalize_reply, history.interrupt_agent_tree

            def commit(outcome, rows):
                if outcome.session_id == session_a and rows[0].content == "reply: cancelled A":
                    attempts.append(outcome.reply_id)
                    raise sqlite3.OperationalError("database is locked")
                return finalize(outcome, rows)

            def interrupt_jobs(agent_id, *, reason):
                if history.agent(agent_id).session_id == session_a and reason == "shutdown":
                    shutdowns.append(agent_id)
                    raise sqlite3.OperationalError("database is locked")
                return interrupt(agent_id, reason=reason)

            with (patch.object(history, "finalize_reply", side_effect=commit),
                  patch.object(history, "interrupt_agent_tree", side_effect=interrupt_jobs),
                  patch("agent_companion.runtime.sqlite_access._LOCK_RETRIES", 5),
                  self.assertLogs("agent_companion.feishu_private", level="WARNING") as logs):
                async with _serve_private(self.settings, history, model, runner, clock):
                    await _eventually(lambda: runner.queries)
                    runner.messages[a] = [_message("cancelled", a, sa, "cancelled A")]
                    clock[0] = 101.0
                    await _eventually(lambda: len(attempts) >= 2)
                    runner.messages[a].append(_message("pending", a, sa, "pending A", stamp=101_100))
                    clock[0] = 102.0
                    await _eventually(lambda: history.pending_inbound(session_a, "feishu-private"))
                    runner.pages = {None: ([_chat(b, sb)], None)}
                    await _eventually(lambda: shutdowns)
                    stopped = len(attempts)
                    runner.messages[b] = [_message("survived", b, sb, "surviving B", stamp=102_100)]
                    clock[0] = 103.0
                    await _eventually(lambda: runner.sent[b] == ["reply: surviving B"])
                    self.assertFalse(runner.sent[a])
                    # 再验证成功证明旧路由已收尾；重开只处理仍待移交的批次。
                    runner.pages = {None: (chats, None)}
                    await _eventually(lambda: runner.sent[a] == ["reply: pending A"])
                    self.assertEqual(len(attempts), stopped)
                    self.assertEqual([e.message.content for e in history.read_after(session_a)],
                                     ["cancelled A", "pending A", "reply: pending A"])
                self.assertTrue(any("cleanup storage busy" in line for line in logs.output))

    async def test_real_sqlite_read_lock_retries_only_completed_child_result(self):
        release_child = asyncio.Event()

        class PausedChild(_Model):
            async def stream(self, messages, tools, *, thinking):
                if not tools:
                    await release_child.wait()
                async for event in super().stream(messages, tools, thinking=thinking):
                    yield event

        b, sb = _CHATS[1], _SENDERS[1]
        session = "feishu-private:" + sha256(sb.encode()).hexdigest()
        runner, model, clock = self.runner, PausedChild(), [100.25]
        runner.pages = {None: ([_chat(b, sb)], None)}
        failures = []
        with TemporaryDirectory() as tmp:
            database = Path(tmp) / "result.db"
            with closing(History(database)) as history, closing(sqlite3.connect(database, timeout=0)) as blocker:
                finalize, read_links = history.finalize_reply, history.run_delegation_ids

                def commit(outcome, rows):
                    ids = finalize(outcome, rows)
                    if outcome.session_id != session:
                        # 子回复已经提交；另一个连接现在锁住数据库，专门阻断下一步的结果读取。
                        blocker.execute("BEGIN EXCLUSIVE")
                    return ids

                def read_result(run_id):
                    try:
                        return read_links(run_id)
                    except sqlite3.OperationalError as exc:
                        failures.append((run_id, exc.sqlite_errorcode))
                        raise

                with (patch.object(history, "finalize_reply", side_effect=commit),
                      patch.object(history, "run_delegation_ids", side_effect=read_result)):
                    async with _serve_private(self.settings, history, model, runner, clock):
                        await _eventually(lambda: runner.queries)
                        runner.messages[b] = [_message("delegate", b, sb, "first B")]
                        clock[0] = 101.0
                        await _eventually(lambda: runner.sent[b] == ["delegated B"])
                        tool = next(e.message for e in history.read_after(session) if e.message.role == "tool")
                        accepted = json.loads(tool.content)
                        child = history.agent(accepted["agent_id"]).session_id
                        delegation_id = accepted["delegation_id"]
                        run_id = history.delegation(delegation_id).run_id
                        try:
                            release_child.set()
                            await _eventually(lambda: len(failures) >= 2)
                            self.assertEqual(set(failures), {(run_id, sqlite3.SQLITE_BUSY)})
                            self.assertEqual(runner.sent[b], ["delegated B"])
                        finally:
                            blocker.rollback()
                        await _eventually(lambda: "returned B" in runner.sent[b])
                        self.assertEqual(runner.sent[b], ["delegated B", "returned B"])
                        self.assertEqual(len(model.calls), 4)
                        self.assertEqual(sum(not tools for tools, _ in model.calls), 1)
                        self.assertEqual([outcome.run_id for outcome in history.outcomes(child)], [run_id])
                        payloads = [json.loads(e.message.content.removeprefix(AGENT_REPLY_PREFIX))
                                    for e in history.read_after(session) if e.message.content.startswith(AGENT_REPLY_PREFIX)]
                        run = history.agent_run(run_id)
                        self.assertEqual(payloads, [{"agent_id": accepted["agent_id"], "run_id": run_id,
                                                     "status": "completed", "result": "reply: child B",
                                                     "delegation_ids": [delegation_id]}])
                        self.assertEqual(run.status, "completed")
                        self.assertIsNotNone(run.reply_enqueued_at)

    async def test_runtime_retries_real_sqlite_commit_contention_after_rollback(self):
        a, sa = _CHATS[0], _SENDERS[0]
        session = "feishu-private:" + sha256(sa.encode()).hexdigest()
        runner, model, clock = self.runner, _Model(), [100.25]
        runner.pages = {None: ([_chat(a, sa)], None)}
        failures = []
        with TemporaryDirectory() as tmp:
            database = Path(tmp) / "commit.db"
            with closing(History(database)) as history, closing(sqlite3.connect(database, timeout=0)) as reader:
                finalize = history.finalize_reply

                def commit(outcome, rows):
                    try:
                        return finalize(outcome, rows)
                    except sqlite3.OperationalError as exc:
                        self.assertFalse(history._db.in_transaction)
                        failures.append((outcome.reply_id, exc.sqlite_errorcode))
                        raise

                with patch.object(history, "finalize_reply", side_effect=commit):
                    async with _serve_private(self.settings, history, model, runner, clock):
                        await _eventually(lambda: runner.queries)
                        runner.messages[a] = [_message("held", a, sa, "first A")]
                        clock[0] = 101.0
                        await asyncio.wait_for(model.a_waiting.wait(), 3)
                        # 在已落入用户历史后持有另一个连接的读锁，阻止回复事务提交。
                        reader.execute("BEGIN")
                        reader.execute("SELECT count(*) FROM messages").fetchone()
                        try:
                            model.release_a.set()
                            await _eventually(lambda: len(failures) >= 2)
                            self.assertEqual([e.message.content for e in history.read_after(session)], ["first A"])
                            self.assertEqual(runner.sent[a], [])
                        finally:
                            reader.rollback()
                        await _eventually(lambda: runner.sent[a] == ["reply: first A"])
                        self.assertEqual([e.message.content for e in history.read_after(session)],
                                         ["first A", "reply: first A"])
                        self.assertEqual(model.calls, [(True, ("first A",))])
                        self.assertEqual(set(failures), {(failures[0][0], sqlite3.SQLITE_BUSY)})
                        self.assertEqual([row[0] for row in history._db.execute("SELECT reply_id FROM reply_outcomes")],
                                         [failures[0][0]])

    async def test_runtime_preserves_the_non_lock_failure_path(self):
        a, sa = _CHATS[0], _SENDERS[0]
        runner, model, clock = self.runner, _Model(), [100.25]
        runner.pages = {None: ([_chat(a, sa)], None)}
        error = sqlite3.OperationalError("database is locked")
        error.sqlite_errorcode = sqlite3.SQLITE_IOERR
        failures = []
        with closing(History(":memory:")) as history:
            finalize = history.finalize_reply

            def commit(outcome, rows):
                if rows[0].content == "reply: failing A":
                    failures.append(outcome.reply_id)
                    raise error
                return finalize(outcome, rows)

            with (patch.object(history, "finalize_reply", side_effect=commit),
                  self.assertLogs("agent_companion.runtime.engine", level="ERROR") as logs):
                async with _serve_private(self.settings, history, model, runner, clock):
                    await _eventually(lambda: runner.queries)
                    runner.messages[a] = [_message("failing", a, sa, "failing A")]
                    clock[0] = 101.0
                    await _eventually(lambda: logs.output)
                    runner.messages[a].append(_message("next", a, sa, "next A", stamp=101_100))
                    clock[0] = 102.0
                    await _eventually(lambda: runner.sent[a] == ["reply: next A"])
                    self.assertEqual(len(failures), 1)
                self.assertTrue(any("OperationalError" in line for line in logs.output))

    async def test_non_lock_sqlite_error_is_not_retried_as_poll_contention(self):
        a, sa = _CHATS[0], _SENDERS[0]
        runner = self.runner
        runner.pages = {None: ([_chat(a, sa)], None)}
        runner.messages[a] = [_message("fatal", a, sa, "no retry")]
        error = sqlite3.OperationalError("database is locked")
        error.sqlite_errorcode = sqlite3.SQLITE_IOERR
        self.assertFalse(is_sqlite_lock_error(error))
        self.assertFalse(is_sqlite_lock_error(sqlite3.OperationalError("unrelated locked route")))
        locked = sqlite3.OperationalError("opaque")
        locked.sqlite_errorcode = sqlite3.SQLITE_LOCKED_SHAREDCACHE
        self.assertTrue(is_sqlite_lock_error(locked))
        with (closing(History(":memory:")) as history,
              patch.object(history, "record_inbound", side_effect=error) as record,
              patch("agent_companion.app.time", return_value=100.25),
              patch("agent_companion.feishu_private.time", return_value=101.0),
              self.assertLogs("agent_companion.channels.feishu.channel", level="ERROR") as logs):
            with self.assertRaises(ExceptionGroup) as caught:
                await asyncio.wait_for(run_feishu(self.settings, history, _Model(), object(),
                                                  discovery=runner.discovery(), roles=demo_roles()), 3)
        self.assertEqual(caught.exception.exceptions, (error,))
        record.assert_called_once()
        self.assertTrue(any("inbox storage failed" in line for line in logs.output))
