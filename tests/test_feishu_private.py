import asyncio
from collections import defaultdict
from contextlib import ExitStack, asynccontextmanager, closing
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from test_agent_support import stored_delegation

from agent_companion.__main__ import _serve
from agent_companion.app import private_feishu_identity, run_feishu, running_agent
from agent_companion.channels.feishu.cli import CliError, CommandResult, FeishuDiscovery
from agent_companion.config import (ASRSettings, FeishuSettings, LLMProvider, MiniMaxConnection, ModelProfile, RuntimeSettings, Settings,
                                    load_settings)
from agent_companion.protocol import ChannelUser, Message, TextDelta, ToolCall, ToolReady
from agent_companion.runtime.subtasks import AgentTree, DELEGATE_TOOL_NAME, demo_roles
from agent_companion.storage.sqlite import AgentReplyInput, DelegatedInput, FeishuBindingError, History, is_sqlite_lock_error
from test_message_support import tool_calls


_OWN = "ou_private_service"
_ACCOUNT_QUERY = ("api", "GET", "/open-apis/authen/v1/user_info")
_CHATS = ("oc_private_a", "oc_private_b", "oc_private_c")
_SENDERS = ("ou_private_a", "ou_private_b", "ou_private_c")


def _message(id_, chat, sender, body, *, stamp=100_250, sender_type="user", id_type="open_id", kind="text"):
    return {"message_id": "om_" + chat + "_" + id_, "chat_id": chat, "create_time": str(stamp), "msg_type": kind,
            "sender": {"id": sender, "id_type": id_type, "sender_type": sender_type},
            "body": {"content": json.dumps(body if isinstance(body, dict) else {"text": body})}}


class _Runner:
    def __init__(self):
        self.account = _OWN
        self.account_failed = False
        self.checking_account = asyncio.Event()
        self.release_account = asyncio.Event()
        self.release_account.set()
        self.verified = asyncio.Event()
        self.queries = []
        self.messages = defaultdict(list)
        self.details = []
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
            self.verified.set()
            result = {"open_id": self.account}
        elif op == ("api", "POST", "/open-apis/im/v1/messages/search"):
            search = json.loads(data)
            assert search["query"] == "" and search["filter"]["chat_type"] == "p2p"
            dates = search["filter"]["time_range"]
            start, end = (datetime.fromisoformat(dates[k]).timestamp() for k in ("start_time", "end_time"))
            self.queries.append((start, end))
            hits = []
            for rows in self.messages.values():
                for row in rows:
                    if row["sender"]["sender_type"] == "user" and start * 1000 <= int(row["create_time"]) < end * 1000:
                        hits.append({"meta_data": {"message_id": row["message_id"], "chat_id": row["chat_id"],
                                                   "type": row["msg_type"], "from_id": row["sender"]["id"],
                                                   "create_time": datetime.fromtimestamp(int(row["create_time"]) / 1000,
                                                                                         timezone.utc).isoformat(),
                                                   "is_p2p_chat": True}})
            result = {"items": hits, "has_more": False}
        elif op == ("api", "GET", "/open-apis/im/v1/messages/mget"):
            ids = json.loads(argv[argv.index("--params") + 1])["message_ids"]
            self.details.append(ids)
            result = {"items": [row for rows in self.messages.values() for row in rows if row["message_id"] in ids]}
        elif op == ("api", "POST", "/open-apis/im/v1/messages"):
            msg = json.loads(data)
            assert json.loads(argv[argv.index("--params") + 1]) == {"receive_id_type": "open_id"}
            peer = msg["receive_id"]
            assert peer.startswith("ou_")
            content = json.loads(msg["content"])
            if msg["msg_type"] == "post":
                self.posts[peer].append(content)
                return CommandResult(0, json.dumps({"ok": True, "identity": "user", "data": {
                    "chat_id": "oc_synthetic_sent", "message_id": "om_post"}}).encode(), b"")
            self.sent[peer].append(content["text"])
            if self.hold_sends or peer in self.hold_destinations:
                try:
                    await asyncio.Event().wait()
                finally:
                    self.closed_sends.add(peer)
            result = {"chat_id": "oc_synthetic_sent", "message_id": "om_sent"}
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

    async def stream(self, messages, tools, *, thinking, instructions=None):
        last = messages[-1]
        self.calls.append((bool(tools), tuple(m.content for m in messages if m.kind == "user")))
        if last.content == "first A":
            self.a_waiting.set()
            await self.release_a.wait()
        if last.content == "first B" and tools:
            yield ToolReady(ToolCall("delegate_b", DELEGATE_TOOL_NAME, '{"agent":"general","prompt":"child B"}'))
        elif last.tool_calls:
            yield TextDelta("delegated B")
        elif last.async_result is not None and last.async_result.kind == "agent_reply":
            yield TextDelta("returned B")
        else:
            yield TextDelta("reply: " + last.content)


@asynccontextmanager
async def _serve_private(settings, history, model, runner, clock):
    with (patch("agent_companion.app.time", side_effect=lambda: clock[0]),
          patch("agent_companion.feishu_private.time", side_effect=lambda: clock[0]),
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
            self.assertEqual(private_feishu_identity(ChannelUser(_OWN)), owner)
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

    async def test_shutdown_awaits_outbound_closure_in_every_route(self):
        a, b, _ = _CHATS
        sa, sb, _ = _SENDERS
        runner, model, clock = self.runner, _Model(), [101.0]
        runner.hold_sends = True
        runner.messages[a] = [_message("shutdown_a", a, sa, "shutdown A", stamp=101_001)]
        runner.messages[b] = [_message("shutdown_b", b, sb, "shutdown B", stamp=101_002)]
        with closing(History(":memory:")) as history:
            async with _serve_private(self.settings, history, model, runner, clock):
                await runner.verified.wait()
                clock[0] = 102.0
                await _eventually(lambda: set(runner.sent) == {sa, sb})
            self.assertEqual(runner.closed_sends, {sa, sb})

    async def test_failed_account_verification_leaves_new_database_available_and_keeps_startup_message(self):
        runner, model, clock = self.runner, _Model(), [100.25]
        owner = private_feishu_identity(ChannelUser(_OWN))
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
                runner.messages[a] = [_message("too_early", a, sa, "old A", stamp=100_249),
                                      _message("from_self", a, _OWN, "ignore self", stamp=100_251),
                                      _message("during_auth", a, sa, "during verification", stamp=100_251)]
                async with _serve_private(self.settings, history, model, runner, clock):
                    await asyncio.wait_for(runner.checking_account.wait(), 3)
                    self.assertIsNone(history.check_feishu_private(owner))
                    self.assertFalse(runner.queries)
                    clock[0] = 102.0
                    runner.release_account.set()
                    await _eventually(lambda: runner.sent[sa] == ["reply: during verification"])
                    self.assertFalse(history.feishu_inbound_seen(f"om_{a}_from_self"))
                    self.assertEqual(history.check_feishu_private(owner), 100_250)

    async def test_production_checks_owner_and_uses_single_cli_verification_before_providers(self):
        runner = self.runner
        owner = private_feishu_identity(ChannelUser(_OWN))
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            workdir = Path(tmp)
            agents = workdir / "agents"
            agents.mkdir()
            (agents / "main.md").write_text('+++\ntools=[]\nmodel_id="chat"\n+++\n离线角色', encoding="utf-8")
            settings = replace(self.settings, database=workdir / "production.db",
                               models={"chat": ModelProfile("chat", "synthetic", "disabled", RuntimeSettings())},
                               llm_providers={"chat": LLMProvider("openai_responses", api_key="synthetic")},
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
                self.assertEqual(run_chats.await_args.kwargs["account"], ChannelUser(_OWN))
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

    async def test_restart_recovers_each_user_without_search_hits(self):
        runner, clock = self.runner, [450]
        users = _SENDERS[:2]
        sessions = tuple("feishu-private:" + sha256(peer.encode()).hexdigest() for peer in users)
        with TemporaryDirectory(dir=Path.cwd()) as root:
            database = Path(root) / "search.sqlite3"
            with closing(History(database)) as history:
                history.bind_feishu_private(private_feishu_identity(ChannelUser(_OWN)), started_ms=100_000)
                history.feishu_inbound_window(100)
                history.record_feishu_window(115, now=115, verified={
                    peer: [(f"om_pending_{i}", 110_000 + i, "text", json.dumps({"text": f"pending {i}"}))]
                    for i, peer in enumerate(users)}, discarded=(), retry=())
            with closing(History(database)) as history:
                async with _serve_private(self.settings, history, _Model(), runner, clock) as agents:
                    await _eventually(lambda: all(len(runner.sent[peer]) == 1 for peer in users))
                    self.assertEqual(agents.call_count, 2)
                    self.assertEqual(runner.details, [])
                    self.assertEqual(history.pending_feishu_users(), [])
                    for i, (session, peer) in enumerate(zip(sessions, users)):
                        self.assertEqual(runner.sent[peer], [f"reply: pending {i}"])
                        self.assertEqual([r.message.content for r in history.tail(session, 5)],
                                         [f"pending {i}", f"reply: pending {i}"])

    async def test_blocked_media_does_not_delay_another_user_or_the_next_search(self):
        runner, clock = self.runner, [100.25]
        a, b = _CHATS[:2]
        sa, sb = _SENDERS[:2]
        session_a, session_b = ("feishu-private:" + sha256(u.encode()).hexdigest() for u in (sa, sb))
        downloading, release = asyncio.Event(), asyncio.Event()

        async def download(*args, **kwargs):
            downloading.set()
            await release.wait()
            raise CliError("synthetic_media_unavailable")

        with (closing(History(":memory:")) as history,
              patch("agent_companion.channels.feishu.channel._MAX_MEDIA_WORKERS", 1),
              patch.object(FeishuDiscovery, "download", side_effect=download) as media):
            async with _serve_private(self.settings, history, _Model(), runner, clock):
                await runner.verified.wait()
                runner.messages[a].extend([
                    _message("voice", a, sa, {"file_key": "file_synthetic"}, kind="audio"),
                    _message("text", a, sa, "A 同批文字", stamp=100_300),
                ])
                runner.messages[b].append(_message("b1", b, sb, "B 第一条"))
                clock[0] = 102
                await asyncio.wait_for(downloading.wait(), 1)
                await _eventually(lambda: runner.sent[sb] == ["reply: B 第一条"])
                self.assertEqual(history.feishu_inbound_window(0), (100, 102))
                self.assertIsNotNone(history.next_feishu_raw(sa))
                self.assertEqual(history.tail(session_a, 10), [])
                runner.messages[a].append(_message("a2", a, sa, "A 后续文字", stamp=102_100))
                runner.messages[b].append(_message("b2", b, sb, "B 后续文字", stamp=102_100))
                clock[0] = 104
                await _eventually(lambda: runner.sent[sb] == ["reply: B 第一条", "reply: B 后续文字"])
                self.assertIn((100.0, 104.0), runner.queries)
                self.assertEqual(history.feishu_inbound_window(0), (100, 104))
                self.assertEqual([row.message.content for row in history.tail(session_b, 10) if row.message.kind == "user"],
                                 ["B 第一条", "B 后续文字"])
                self.assertEqual(history.tail(session_a, 10), [])
                release.set()
                await _eventually(lambda: "reply: A 后续文字" in runner.sent[sa])
                self.assertEqual([row.message.content for row in history.tail(session_a, 10) if row.message.kind == "user"],
                                 ["A 同批文字", "A 后续文字"])
                media.assert_awaited_once()

    async def test_busy_user_registration_does_not_block_another_user(self):
        runner, clock = self.runner, [100.25]
        a, b = _CHATS[:2]
        sa, sb = _SENDERS[:2]
        session_a = "feishu-private:" + sha256(sa.encode()).hexdigest()
        blocked = True
        with closing(History(":memory:")) as history:
            register = history.ensure_main_agent

            def ensure(session):
                if blocked and session == session_a:
                    raise sqlite3.OperationalError("database is locked")
                return register(session)

            with patch.object(history, "ensure_main_agent", side_effect=ensure):
                async with _serve_private(self.settings, history, _Model(), runner, clock):
                    await runner.verified.wait()
                    runner.messages[a].append(_message("a", a, sa, "等待 A"))
                    runner.messages[b].append(_message("b", b, sb, "独立 B"))
                    clock[0] = 102
                    await _eventually(lambda: runner.sent[sb] == ["reply: 独立 B"])
                    self.assertEqual(runner.sent[sa], [])
                    self.assertEqual(history.feishu_inbound_window(0), (100, 102))
                    self.assertEqual(history.pending_feishu_users(), [sa])
                    blocked = False
                    await _eventually(lambda: runner.sent[sa] == ["reply: 等待 A"])
                    self.assertEqual(len(runner.details), 1)

    async def test_runtime_consumes_pending_batches_in_order_while_other_route_progresses(self):
        a, b, _ = _CHATS
        sa, sb, _ = _SENDERS
        session_a = "feishu-private:" + sha256(sa.encode()).hexdigest()
        runner, model, clock = self.runner, _Model(), [100.25]
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
                    await runner.verified.wait()
                    runner.messages[a] = [_message("a1", a, sa, "queued A1")]
                    clock[0] = 101.0
                    await _eventually(lambda: attempts)
                    first = attempts[0]
                    runner.messages[a].append(_message("a2", a, sa, "queued A2", stamp=101_100))
                    runner.messages[b] = [_message("b1", b, sb, "independent B", stamp=101_100)]
                    clock[0] = 102.0
                    await _eventually(lambda: len(attempts) >= 2 and runner.sent[sb] == ["reply: independent B"])
                    pending = history.pending_inbound(session_a, "feishu-private")
                    self.assertEqual(len(pending), 2)
                    self.assertEqual(pending[0], first)
                    self.assertEqual(set(attempts), {first})
                    self.assertEqual(history.read_after(session_a), [])
                    self.assertEqual(runner.sent[sa], [])

                    blocked = False
                    await _eventually(lambda: len(runner.sent[sa]) >= 2)
                    self.assertEqual(runner.sent[sa], ["reply: queued A1", "reply: queued A2"])
                    self.assertEqual(committed, pending)
                    self.assertEqual(history.pending_inbound(session_a, "feishu-private"), [])
                    self.assertEqual(history.feishu_inbound_window(100), (100, 102))
                    self.assertEqual([e.message.content for e in history.read_after(session_a)],
                                     ["queued A1", "reply: queued A1", "queued A2", "reply: queued A2"])
                    self.assertEqual([inputs for _, inputs in model.calls if inputs[0] == "queued A1"],
                                     [("queued A1",), ("queued A1", "queued A2")])

    async def test_runtime_retries_reply_and_subtask_transactions_without_repeating_work(self):
        class Model(_Model):
            async def stream(self, messages, tools, *, thinking, instructions=None):
                if messages[-1].content == "first B":
                    yield TextDelta("tool-stage B")
                async for event in super().stream(messages, tools, thinking=thinking):
                    yield event

        b, sb = _CHATS[1], _SENDERS[1]
        session_b = "feishu-private:" + sha256(sb.encode()).hexdigest()
        runner, model, clock = self.runner, Model(), [100.25]
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
                await runner.verified.wait()
                runner.messages[b] = [_message("tool", b, sb, "first B")]
                clock[0] = 101.0
                await _eventually(lambda: len(attempts["accept"]) >= 2)
                blocked.remove("accept")
                await _eventually(lambda: len(attempts["tool_reply"]) >= 2 and len(attempts["start"]) >= 2)
                _, snapshot = history.context_snapshot(session_b)
                self.assertEqual([e.message.kind for e in snapshot], ["user", "assistant"])
                self.assertTrue(snapshot[-1].active_reply)
                self.assertEqual(snapshot[-1].message.content, "")
                self.assertIsNotNone(snapshot[-1].message.tool_calls[0].result)
                self.assertEqual(runner.sent[sb], [])
                self.assertEqual(model.calls, [(True, ("first B",))])
                # 工具轮次之间吸收排队的用户输入时也只等当前事务。
                runner.messages[b].append(_message("during_tool", b, sb, "during tool B", stamp=101_100))
                clock[0] = 102.0
                await _eventually(lambda: history.pending_inbound(session_b, "feishu-private"))
                blocked.remove("tool_reply")
                await _eventually(lambda: len(attempts["tool_input"]) >= 2)
                self.assertEqual(model.calls, [(True, ("first B",))])
                blocked.remove("tool_input")
                await _eventually(lambda: len(runner.sent[sb]) >= 2)
                tool = tool_calls(e.message for e in history.read_after(session_b))[0]
                accepted = json.loads(tool.result)
                delegation_id = stored_delegation(history, session_b, tool.id).delegation_id
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
                self.assertEqual(runner.sent[sb], ["tool-stage B", "reply: during tool B"])
                blocked.remove("result_input")
                await _eventually(lambda: "returned B" in runner.sent[sb])
                # 标记仍锁住时，主 Runtime 已能消费回投；释放标记不能重复回投。
                self.assertIsNone(history.agent_run(run_id).reply_enqueued_at)
                blocked.remove("mark")
                await _eventually(lambda: history.agent_run(run_id).reply_enqueued_at is not None)

                delegate.assert_awaited_once()
                self.assertEqual(len(model.calls), 4)  # 主会话三轮、子会话一轮
                self.assertEqual(runner.sent[sb], ["tool-stage B", "reply: during tool B", "returned B"])
                self.assertEqual([e.message.kind for e in history.read_after(session_b)],
                                 ["user", "assistant", "user", "assistant", "async_tool_result", "assistant"])
                self.assertEqual([e.message.content for e in history.read_after(child)], ["child B", "reply: child B"])
                for stage in attempts:
                    self.assertEqual(committed[stage], [attempts[stage][0]], stage)
                    self.assertTrue(all(args == committed[stage][0] for args in attempts[stage]), stage)
                outcome, rows = attempts["tool_reply"][0][0]
                self.assertTrue(all(args[0] is outcome and args[1] is rows for args, _ in attempts["tool_reply"]))

    async def test_real_sqlite_lock_retries_only_child_terminal_write(self):
        release_child = asyncio.Event()

        class PausedChild(_Model):
            async def stream(self, messages, tools, *, thinking, instructions=None):
                if not tools:
                    await release_child.wait()
                async for event in super().stream(messages, tools, thinking=thinking):
                    yield event

        b, sb = _CHATS[1], _SENDERS[1]
        session = "feishu-private:" + sha256(sb.encode()).hexdigest()
        runner, model, clock = self.runner, PausedChild(), [100.25]
        failures = []
        with TemporaryDirectory() as tmp:
            database = Path(tmp) / "result.db"
            with closing(History(database)) as history, closing(sqlite3.connect(database, timeout=0)) as blocker:
                finalize, finish = history.finalize_reply, history.finish_agent_run

                def commit(outcome, rows):
                    ids = finalize(outcome, rows)
                    if outcome.session_id != session:
                        # 子回复已经提交；另一个连接现在锁住数据库，专门阻断下一步的终态写入。
                        blocker.execute("BEGIN EXCLUSIVE")
                    return ids

                def finish_result(run_id, **kwargs):
                    try:
                        return finish(run_id, **kwargs)
                    except sqlite3.OperationalError as exc:
                        failures.append((run_id, exc.sqlite_errorcode))
                        raise

                with (patch.object(history, "finalize_reply", side_effect=commit),
                      patch.object(history, "finish_agent_run", side_effect=finish_result)):
                    async with _serve_private(self.settings, history, model, runner, clock):
                        await runner.verified.wait()
                        runner.messages[b] = [_message("delegate", b, sb, "first B")]
                        clock[0] = 101.0
                        await _eventually(lambda: runner.sent[sb] == ["delegated B"])
                        tool = tool_calls(e.message for e in history.read_after(session))[0]
                        accepted = json.loads(tool.result)
                        child = history.agent(accepted["agent_id"]).session_id
                        run_id = stored_delegation(history, session, tool.id).run_id
                        try:
                            release_child.set()
                            await _eventually(lambda: len(failures) >= 2)
                            self.assertEqual(set(failures), {(run_id, sqlite3.SQLITE_BUSY)})
                            self.assertEqual(runner.sent[sb], ["delegated B"])
                        finally:
                            blocker.rollback()
                        await _eventually(lambda: "returned B" in runner.sent[sb])
                        self.assertEqual(runner.sent[sb], ["delegated B", "returned B"])
                        self.assertEqual(len(model.calls), 4)
                        self.assertEqual(sum(not tools for tools, _ in model.calls), 1)
                        self.assertEqual([outcome.run_id for outcome in history.outcomes(child)], [run_id])
                        payloads = [e.message.async_result.payload for e in history.read_after(session)
                                    if e.message.async_result is not None and e.message.async_result.kind == "agent_reply"]
                        run = history.agent_run(run_id)
                        self.assertEqual(payloads, [{"agent_id": accepted["agent_id"],
                                                     "status": "completed", "result": "reply: child B"}])
                        self.assertEqual(run.status, "completed")
                        self.assertIsNotNone(run.reply_enqueued_at)

    async def test_runtime_retries_real_sqlite_commit_contention_after_rollback(self):
        a, sa = _CHATS[0], _SENDERS[0]
        session = "feishu-private:" + sha256(sa.encode()).hexdigest()
        runner, model, clock = self.runner, _Model(), [100.25]
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
                        await runner.verified.wait()
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
                            self.assertEqual(runner.sent[sa], [])
                        finally:
                            reader.rollback()
                        await _eventually(lambda: runner.sent[sa] == ["reply: first A"])
                        self.assertEqual([e.message.content for e in history.read_after(session)],
                                         ["first A", "reply: first A"])
                        self.assertEqual(model.calls, [(True, ("first A",))])
                        self.assertEqual(set(failures), {(failures[0][0], sqlite3.SQLITE_BUSY)})
                        self.assertEqual([row[0] for row in history._db.execute("SELECT reply_id FROM reply_outcomes")],
                                         [failures[0][0]])

    async def test_runtime_preserves_the_non_lock_failure_path(self):
        a, sa = _CHATS[0], _SENDERS[0]
        runner, model, clock = self.runner, _Model(), [100.25]
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
                    await runner.verified.wait()
                    runner.messages[a] = [_message("failing", a, sa, "failing A")]
                    clock[0] = 101.0
                    await _eventually(lambda: logs.output)
                    runner.messages[a].append(_message("next", a, sa, "next A", stamp=101_100))
                    clock[0] = 102.0
                    await _eventually(lambda: runner.sent[sa] == ["reply: next A"])
                    self.assertEqual(len(failures), 1)
                self.assertTrue(any("OperationalError" in line for line in logs.output))

    async def test_non_lock_sqlite_error_is_not_retried_as_poll_contention(self):
        a, sa = _CHATS[0], _SENDERS[0]
        runner = self.runner
        runner.messages[a] = [_message("fatal", a, sa, "no retry")]
        error = sqlite3.OperationalError("database is locked")
        error.sqlite_errorcode = sqlite3.SQLITE_IOERR
        self.assertFalse(is_sqlite_lock_error(error))
        self.assertFalse(is_sqlite_lock_error(sqlite3.OperationalError("unrelated locked route")))
        locked = sqlite3.OperationalError("opaque")
        locked.sqlite_errorcode = sqlite3.SQLITE_LOCKED_SHAREDCACHE
        self.assertTrue(is_sqlite_lock_error(locked))
        with (closing(History(":memory:")) as history,
              patch.object(history, "record_feishu_window", side_effect=error) as record,
              patch("agent_companion.app.time", return_value=100.25),
              patch("agent_companion.feishu_private.time", return_value=101.0),
              self.assertLogs("agent_companion.channels.feishu.channel", level="ERROR") as logs):
            with self.assertRaises(ExceptionGroup) as caught:
                await asyncio.wait_for(run_feishu(self.settings, history, _Model(), object(),
                                                  discovery=runner.discovery(), roles=demo_roles()), 3)
        self.assertEqual(caught.exception.exceptions, (error,))
        record.assert_called_once()
        self.assertTrue(any("inbox storage failed" in line for line in logs.output))
