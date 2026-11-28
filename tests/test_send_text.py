import asyncio
from contextlib import closing
import json
from pathlib import Path
from sqlite3 import OperationalError
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch

from agent_companion.app import running_agent
from agent_companion.config import Settings
from agent_companion.protocol import (DeliveryStatus, Message, MessageDelivery, Receipt, TextDelta, TextSendResult,
                                      ToolCall, ToolReady)
from agent_companion.runtime.instructions import load_roles
from agent_companion.runtime.send_text import SEND_TEXT_TOOL_NAME
from agent_companion.runtime.subtasks import AgentRole, AgentTree, DELEGATE_TOOL_NAME
from agent_companion.runtime.tools import ToolContext
from agent_companion.storage.sqlite import History
from test_agent_support import stored_delegation
from test_message_support import tool_calls


class _ReplyChannel:
    def __init__(self):
        self.replies = []

    async def deliver(self, reply):
        text = "".join([part async for part in reply])
        self.replies.append(text)
        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)


def _send_call(call_id, arguments):
    return ToolCall(call_id, SEND_TEXT_TOOL_NAME, arguments)


def _submit(runtime):
    runtime.submit_messages((MessageDelivery(Message("user", "测试请求"), "trigger"),))


def _stored_calls(history, session_id):
    return tool_calls(row.message for row in history.read_after(session_id))


class _SendModel:
    def __init__(self, *calls):
        self.calls = calls
        self.requests = []

    async def stream(self, messages, tools, *, thinking, instructions=None):
        self.requests.append(tuple(tools))
        if messages[-1].kind == "user":
            for call in self.calls:
                yield ToolReady(call)
        else:
            yield TextDelta("普通回复")


class SendTextTests(unittest.IsolatedAsyncioTestCase):
    async def test_roles_independently_enable_tool_and_use_the_same_tree_sender(self):
        for main_enabled, child_enabled in ((False, True), (True, False), (True, True)):
            with (self.subTest(main=main_enabled, child=child_enabled), closing(History(":memory:")) as history,
                  TemporaryDirectory(dir=Path.cwd()) as directory):
                agents = Path(directory) / "agents"
                agents.mkdir()
                main_tools = [DELEGATE_TOOL_NAME] + ([SEND_TEXT_TOOL_NAME] if main_enabled else [])
                child_tools = [SEND_TEXT_TOOL_NAME] if child_enabled else []
                for role, tools in (("main", main_tools), ("child", child_tools)):
                    (agents / f"{role}.md").write_text(
                        f'+++\nmodel_id = "chat"\ndescription = "{role}"\ntools = {json.dumps(tools)}\n+++\n{role}',
                        encoding="utf-8")
                roles = load_roles(Path(directory))
                self.assertEqual(roles["main"].tools, tuple(main_tools))
                self.assertEqual(roles["child"].tools, tuple(child_tools))
                observations = {}

                class Model:
                    async def stream(self, messages, tools, *, thinking, instructions=None):
                        last = messages[-1]
                        if last.kind != "user":
                            yield TextDelta("普通回复")
                            return
                        role = instructions.strip()
                        observations[role] = tuple(tool.name for tool in tools)
                        if role == "main":
                            yield ToolReady(ToolCall("delegate", DELEGATE_TOOL_NAME,
                                                     '{"agent":"child","prompt":"测试子任务"}'))
                        if SEND_TEXT_TOOL_NAME in observations[role]:
                            yield ToolReady(_send_call(role, json.dumps({"text": role})))

                sender = AsyncMock(return_value=TextSendResult.SENT)
                channel = _ReplyChannel()
                async with running_agent(Settings(), history, Model(), channel, session_id="main", roles=roles,
                                         text_sender=sender) as (runtime, tree):
                    _submit(runtime)
                    await asyncio.wait_for(tree.wait_idle(), 3)
                    self.assertEqual(observations, {"main": tuple(main_tools), "child": tuple(child_tools)})
                    child = history.agent(stored_delegation(history, "main", "delegate").agent_id)
                    for session, allowed in (("main", main_enabled), (child.session_id, child_enabled)):
                        calls = [call for call in _stored_calls(history, session) if call.name == SEND_TEXT_TOOL_NAME]
                        self.assertEqual([json.loads(call.result) for call in calls], [{"ok": True}] if allowed else [])
                    bodies = [item.args[0] for item in sender.await_args_list]
                    self.assertCountEqual(bodies, [role for role, enabled in (
                        ("main", main_enabled), ("child", child_enabled)) if enabled])
                    self.assertTrue(channel.replies)
                    self.assertEqual(set(channel.replies), {"普通回复"})

    async def test_character_limit_and_complete_history_before_and_after_send(self):
        accepted = _send_call("accepted", json.dumps({"text": "🧪" * 1000}, ensure_ascii=True))
        oversized = _send_call("oversized", json.dumps({"text": "🧪" * 1001}, ensure_ascii=False, indent=2))
        entered, release = asyncio.Event(), asyncio.Event()
        seen = []

        async def send(text):
            seen.append(text)
            entered.set()
            await release.wait()
            return TextSendResult.SENT

        roles = {"main": AgentRole(None, (SEND_TEXT_TOOL_NAME,))}
        channel = _ReplyChannel()
        with closing(History(":memory:")) as history:
            async with running_agent(Settings(), history, _SendModel(accepted, oversized), channel,
                                     session_id="main", roles=roles, text_sender=send) as (runtime, tree):
                _submit(runtime)
                await asyncio.wait_for(entered.wait(), 3)
                staged = {call.id: call for call in _stored_calls(history, "main")}
                self.assertEqual(staged[accepted.id].arguments, accepted.arguments)
                self.assertIsNone(staged[accepted.id].result)
                self.assertEqual(channel.replies, [])
                release.set()
                await asyncio.wait_for(tree.wait_idle(), 3)
                stored = {call.id: call for call in _stored_calls(history, "main")}
                self.assertEqual({key: call.arguments for key, call in stored.items()},
                                 {call.id: call.arguments for call in (accepted, oversized)})
                self.assertEqual(json.loads(stored[accepted.id].result), {"ok": True})
                self.assertEqual(json.loads(stored[oversized.id].result), {"ok": False, "reason": "text_too_long"})
                self.assertEqual(seen, ["🧪" * 1000])
                self.assertEqual(channel.replies, ["普通回复"])

    async def test_enabled_without_binding_is_visible_and_returns_unavailable(self):
        call = _send_call("unbound", '{"text":"未绑定时保留原参数"}')
        roles = {"main": AgentRole(None, (SEND_TEXT_TOOL_NAME,))}
        model = _SendModel(call)
        with closing(History(":memory:")) as history:
            async with running_agent(Settings(), history, model, _ReplyChannel(), session_id="main",
                                     roles=roles) as (runtime, tree):
                _submit(runtime)
                await asyncio.wait_for(tree.wait_idle(), 3)
                spec, = model.requests[0]
                self.assertEqual(spec.name, SEND_TEXT_TOOL_NAME)
                self.assertEqual(tuple(spec.parameters["properties"]), ("text",))
                self.assertEqual(spec.parameters["required"], ["text"])
                self.assertEqual(spec.parameters["properties"]["text"]["maxLength"], 1000)
                self.assertFalse(spec.parameters["additionalProperties"])
                stored, = _stored_calls(history, "main")
                self.assertEqual(stored.arguments, call.arguments)
                self.assertEqual(json.loads(stored.result), {"ok": False, "reason": "unavailable"})

    async def test_shutdown_revokes_old_reference_without_affecting_other_session_or_reopen(self):
        roles = {"main": AgentRole(None, (SEND_TEXT_TOOL_NAME,))}
        model, channel, settings = _SendModel(), _ReplyChannel(), Settings()
        sender_a, sender_b = (AsyncMock(return_value=TextSendResult.SENT) for _ in range(2))

        async def execute(runtime, text):
            call = _send_call("send", json.dumps({"text": text}))
            context = ToolContext(runtime.session_id, "run", call.id)
            return json.loads(await runtime._tools.execute(call, context))

        with closing(History(":memory:")) as history:
            async with running_agent(settings, history, model, channel, session_id="b", roles=roles,
                                     text_sender=sender_b) as (other, _):
                async with running_agent(settings, history, model, channel, session_id="a", roles=roles,
                                         text_sender=sender_a) as (old, tree):
                    self.assertEqual(await execute(old, "first"), {"ok": True})
                    tree.stop_accepting()
                    self.assertEqual(await execute(old, "after stop"), {"ok": False, "reason": "unavailable"})
                async with running_agent(settings, history, model, channel, session_id="a", roles=roles,
                                         text_sender=sender_a) as (current, _):
                    self.assertEqual(await execute(old, "after reopen"), {"ok": False, "reason": "unavailable"})
                    self.assertEqual(await execute(current, "reopened"), {"ok": True})
                    self.assertEqual(await execute(other, "other"), {"ok": True})
        self.assertEqual([item.args for item in sender_a.await_args_list], [("first",), ("reopened",)])
        sender_b.assert_awaited_once_with("other")

    async def test_invalid_parameters_never_reach_sender(self):
        args = ('{}', '{"text":5}', '{"text":"测试","recipient":"ou_synthetic"}')
        model = _SendModel(*(_send_call(str(index), raw) for index, raw in enumerate(args)))
        sender = AsyncMock(return_value=TextSendResult.SENT)
        roles = {"main": AgentRole(None, (SEND_TEXT_TOOL_NAME,))}
        with closing(History(":memory:")) as history:
            async with running_agent(Settings(), history, model, _ReplyChannel(), session_id="main",
                                     roles=roles, text_sender=sender) as (runtime, tree):
                _submit(runtime)
                await asyncio.wait_for(tree.wait_idle(), 3)
                stored = _stored_calls(history, "main")
                self.assertEqual([call.arguments for call in stored], list(args))
                self.assertEqual([json.loads(call.result) for call in stored],
                                 [{"ok": False, "reason": "invalid_text"}] * len(args))
        sender.assert_not_awaited()

    async def test_startup_failure_revokes_binding(self):
        sender = AsyncMock(return_value=TextSendResult.SENT)
        roles = {"main": AgentRole(None, (SEND_TEXT_TOOL_NAME,))}
        tools = []

        def fail_start(tree, channel, owner):
            tools.append(tree._send_text_tool)
            raise RuntimeError("synthetic startup failure")

        with (closing(History(":memory:")) as history,
              patch.object(AgentTree, "bind_root", fail_start),
              self.assertRaisesRegex(RuntimeError, "synthetic startup failure")):
            async with running_agent(Settings(), history, _SendModel(), _ReplyChannel(), session_id="main",
                                     roles=roles, text_sender=sender):
                self.fail("startup unexpectedly succeeded")
        tool, = tools
        result = await tool.handler({"text": "failed startup"}, ToolContext("main", "run", "send"))
        self.assertEqual(json.loads(result), {"ok": False, "reason": "unavailable"})
        sender.assert_not_awaited()

    async def test_duplicate_text_keys_are_rejected_without_sending_and_preserved_in_history(self):
        calls = (
            _send_call("duplicate", '{"text":"first", "text":"second"}'),
            _send_call("escaped", r'{"text":"first", "te\u0078t":"second"}'),
        )
        sender = AsyncMock(return_value=TextSendResult.SENT)
        roles = {"main": AgentRole(None, (SEND_TEXT_TOOL_NAME,))}
        with closing(History(":memory:")) as history:
            async with running_agent(Settings(), history, _SendModel(*calls), _ReplyChannel(),
                                     session_id="main", roles=roles, text_sender=sender) as (runtime, tree):
                _submit(runtime)
                await asyncio.wait_for(tree.wait_idle(), 3)
                stored = _stored_calls(history, "main")
                self.assertEqual([call.arguments for call in stored], [call.arguments for call in calls])
                self.assertEqual([json.loads(call.result) for call in stored],
                                 [{"error": "invalid_arguments"}] * len(calls))
        sender.assert_not_awaited()

    async def test_shutdown_records_unconfirmed_sends_in_main_and_child_history(self):
        calls = {role: _send_call(role, json.dumps({"text": role})) for role in ("main", "child")}
        entered = {role: asyncio.Event() for role in calls}
        cancelled = []

        async def send(text):
            entered[text].set()
            try:
                await asyncio.Future()  # Never returns a CLI receipt.
            finally:
                cancelled.append(text)

        class Model:
            async def stream(self, messages, tools, *, thinking, instructions=None):
                if messages[-1].kind == "user":
                    if instructions == "main":
                        yield ToolReady(ToolCall("delegate", DELEGATE_TOOL_NAME,
                                                 '{"agent":"child","prompt":"测试子任务"}'))
                    yield ToolReady(calls[instructions])

        roles = {
            "main": AgentRole("main", (DELEGATE_TOOL_NAME, SEND_TEXT_TOOL_NAME)),
            "child": AgentRole("child", (SEND_TEXT_TOOL_NAME,), "测试子角色"),
        }
        with closing(History(":memory:")) as history:
            async def serve():
                async with running_agent(Settings(), history, Model(), _ReplyChannel(), session_id="main",
                                         roles=roles, text_sender=send) as (runtime, _):
                    _submit(runtime)
                    await asyncio.gather(*(event.wait() for event in entered.values()))

            await asyncio.wait_for(serve(), 3)
            child = history.agent(stored_delegation(history, "main", "delegate").agent_id)
            for role, session in (("main", "main"), ("child", child.session_id)):
                stored, = [call for call in _stored_calls(history, session) if call.name == SEND_TEXT_TOOL_NAME]
                self.assertEqual(stored.arguments, calls[role].arguments)
                self.assertEqual(json.loads(stored.result), {"ok": False, "reason": "send_unconfirmed"})
        self.assertCountEqual(cancelled, ["main", "child"])

    async def test_shutdown_preserves_confirmed_send_while_its_result_waits_for_sqlite(self):
        call = _send_call("confirmed", '{"text":"测试已送达"}')
        sender = AsyncMock(return_value=TextSendResult.SENT)
        roles = {"main": AgentRole(None, (SEND_TEXT_TOOL_NAME,))}
        write_started = asyncio.Event()
        with closing(History(":memory:")) as history:
            record = history.record_tool_result

            def lock_once(*args):
                if not write_started.is_set():
                    write_started.set()
                    raise OperationalError("database is locked")
                return record(*args)

            with patch.object(history, "record_tool_result", side_effect=lock_once):
                async with running_agent(Settings(), history, _SendModel(call), _ReplyChannel(), session_id="main",
                                         roles=roles, text_sender=sender, retry_sqlite_locks=True) as (runtime, _):
                    _submit(runtime)
                    await asyncio.wait_for(write_started.wait(), 3)
            stored, = _stored_calls(history, "main")
            self.assertEqual(stored.arguments, call.arguments)
            self.assertEqual(json.loads(stored.result), {"ok": True})
        sender.assert_awaited_once_with("测试已送达")
