import asyncio
from contextlib import closing, nullcontext, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
from sqlite3 import SQLITE_BUSY, SQLITE_CORRUPT, SQLITE_IOERR, OperationalError
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpcore
import httpx

from test_agent_support import stored_delegation

from agent_companion.app import running_agent
from agent_companion.config import Settings
from agent_companion.models import _http_stream
from agent_companion.models.deepseek import DeepSeek
from agent_companion.protocol import (AgentReply, DeliveryStatus, Message, MessageDelivery, ModelError,
                                      Receipt, TextDelta, ToolCall, ToolReady, UserSource)
from agent_companion.runtime.silent_end import SILENT_END_TOOL_NAME
from agent_companion.runtime.subtasks import AgentRole, DELEGATE_TOOL_NAME, demo_roles
from agent_companion.runtime.timers import CANCEL_TIMER_TOOL_NAME, SET_TIMER_TOOL_NAME, Timers
from agent_companion.runtime.tools import ToolContext
from agent_companion.session_scheduler import SessionSchedulers
from agent_companion.storage.sqlite import History
from test_message_support import tool_calls as all_tool_calls


async def _wait(awaitable):
    return await asyncio.wait_for(awaitable, timeout=2)


async def _eventually(condition):
    async with asyncio.timeout(2):
        while not condition():
            await asyncio.sleep(0.005)


def _call(call_id, prompt, **extra):
    if not any(field in extra for field in ("agent", "agent_id")):
        extra["agent"] = "general"
    return ToolCall(call_id, DELEGATE_TOOL_NAME, json.dumps({"prompt": prompt, **extra}))


def _result_inputs(messages):
    return [message.async_result.payload
            for message in messages if message.async_result is not None and message.async_result.kind == "agent_reply"]


class _Model:
    def __init__(self, answer, *, tool_names=False):
        self.answer = answer
        self.requests = []
        self.tool_names = tool_names

    async def stream(self, messages, tools, *, thinking, instructions=None):
        names = tuple(tool.name for tool in tools)
        self.requests.append((tuple(messages), names))
        async for event in self.answer(messages, names if self.tool_names else bool(names)):
            yield event


class _Channel:
    def __init__(self):
        self.replies = []
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()

    async def deliver(self, reply):
        content = "".join([part async for part in reply])
        self.replies.append((reply.info.session_id, reply.info.run_id, content))
        if content == "hold-delivery":
            self.blocked.set()
            await self.release.wait()
        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, content)


class DemoIsolationTest(unittest.TestCase):
    def test_cli_demo_config_cannot_write_production_db_or_interrupt_its_jobs(self):
        from agent_companion.__main__ import main

        with TemporaryDirectory(dir=Path.cwd()) as folder:
            database = Path(folder) / "synthetic.sqlite"
            config = Path(folder) / "settings.toml"
            config.write_text('database = "synthetic.sqlite"\n', encoding="utf-8")
            with closing(History(database)) as history:
                history.append("main", (Message("user", "existing synthetic data"),))
                main_agent = history.ensure_main_agent("main")
                fixture = MessageDelivery(Message("user", "fixture"), "trigger")
                history.append_agent_inputs(main_agent.agent_id, "parent", (fixture,), start=True)
                _, job = history.accept_delegation(main_agent.agent_id, "parent", "tool", role_id="general")
            with patch("sys.argv", ["agent-companion", "--config", str(config), "--demo-subtasks"]), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            with closing(History(database)) as history:
                self.assertEqual([row.message.content for row in history.read_after("main")],
                                 ["existing synthetic data", "fixture"])
                self.assertEqual(history.delegation(job.delegation_id).status, "queued")


class SubtaskTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = Settings()
        self.history = History(":memory:")
        self.addCleanup(self.history.close)
        self.channel = _Channel()

    def _messages(self, session):
        return [entry.message for entry in self.history.read_after(session)]

    def _acceptances(self, session="main"):
        return {call.id: json.loads(call.result) for call in all_tool_calls(self._messages(session))}

    def _delegation(self, call_id, session="main"):
        return stored_delegation(self.history, session, call_id)

    def _run(self, call_id, session="main"):
        return self.history.agent_run(self._delegation(call_id, session).run_id)

    def _agent_runs(self, agent_id):
        return [self.history.agent_run(run_id) for run_id, in self.history._db.execute(
            "SELECT run_id FROM agent_runs WHERE agent_id = ? ORDER BY rowid", (agent_id,))]

    def _session(self, acceptance):
        return self.history.agent(acceptance["agent_id"]).session_id

    async def _open_native_job(self, runtime, tree, call_id):
        agent_id = runtime.agent.agent_id
        run_id = f"parent-{call_id}"
        fixture = MessageDelivery(Message("user", "fixture"), "trigger")
        self.history.append_agent_inputs(agent_id, run_id, (fixture,), start=True)
        call = _call(call_id, "fixture task")
        context = ToolContext("main", run_id, call_id, agent_id=agent_id)
        accepted = json.loads(await tree._delegate(json.loads(call.arguments), context))
        self.assertEqual(accepted, {"status": "accepted", "agent_id": self._delegation(call_id).agent_id})
        return call_id

    async def test_native_subtask_transport_finishes_or_expires_before_releasing_parent(self):
        for mode in ("complete", "graceful", "timeout"):
            with self.subTest(mode=mode):
                await self._check_native_subtask_close(mode)

    async def _check_native_subtask_close(self, mode):
        read_waiting, closing_wire, release, closed, cancelled = (asyncio.Event() for _ in range(5))
        body = b'data: {"choices":[{"index":0,"delta":{"content":"child result"}}]}\n\n'
        if mode == "complete":
            body += b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        response = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n"
                    + f"Content-Length: {len(body) + (0 if mode == 'complete' else 100)}\r\n\r\n".encode() + body)

        class Wire(httpcore.AsyncNetworkStream):
            sent = False

            async def read(self, max_bytes, timeout=None):
                if not self.sent:
                    self.sent = True
                    return response
                read_waiting.set()
                await asyncio.Future()

            async def write(self, buffer, timeout=None):
                pass

            async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
                return self

            async def aclose(self):
                closing_wire.set()
                try:
                    if mode != "complete":
                        await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                closed.set()

        class Backend(httpcore.AsyncNetworkBackend):
            async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
                return Wire()

        pool = httpcore.AsyncConnectionPool
        with patch.object(httpcore, "AsyncConnectionPool", side_effect=lambda **kw: pool(network_backend=Backend(), **kw)):
            transport = httpx.AsyncHTTPTransport(trust_env=False)

        schedulers = SessionSchedulers()
        scheduler = schedulers.get_or_create("main")
        budget = 0.1
        loop = asyncio.get_running_loop()
        with (patch.object(_http_stream, "_RESPONSE_CLOSE_TIMEOUT", budget),
              patch.object(_http_stream._LOG, "warning", wraps=_http_stream._LOG.warning) as warning):
            async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
                model = DeepSeek(client)
                agent = running_agent(self.settings, self.history, model, self.channel,
                                      session_id="main", schedulers=schedulers, roles=demo_roles())
                runtime, subtasks = await agent.__aenter__()
                stopping = None
                try:
                    job = await self._open_native_job(runtime, subtasks, f"native-{mode}")
                    if mode == "complete":
                        await _wait(subtasks.wait_idle())
                        await _wait(runtime.wait_idle())
                        result = self._run(job)
                        self.assertEqual((result.status, result.result), ("completed", "child result"))
                        self.assertEqual(_result_inputs(self._messages("main")), [
                            {"agent_id": result.agent_id, "status": "completed", "result": "child result"}])
                        self.assertTrue(closed.is_set())
                        return

                    await _wait(read_waiting.wait())
                    started = loop.time()
                    stopping = asyncio.create_task(agent.__aexit__(None, None, None))
                    await _wait(closing_wire.wait())
                    self.assertFalse(stopping.done())
                    with self.assertRaisesRegex(RuntimeError, "already bound or owned"):
                        scheduler.reserve_agent()
                    if mode == "graceful":
                        release.set()
                    await asyncio.wait_for(asyncio.shield(stopping), 1)
                    self.assertEqual(closed.is_set(), mode == "graceful")
                    self.assertEqual(cancelled.is_set(), mode == "timeout")
                    self.assertEqual(warning.called, mode == "timeout")
                    if mode == "timeout":
                        self.assertGreaterEqual(loop.time() - started, budget)
                        self.assertIn("transport_close_timeout", warning.call_args.args[0])
                    result = self._run(job)
                    self.assertEqual((result.status, result.error), ("interrupted", "shutdown"))
                    self.assertNotIn(result.agent_id,
                                     [notice["agent_id"] for notice in _result_inputs(self._messages("main"))])
                    scheduler.release_agent(scheduler.reserve_agent())
                finally:
                    release.set()
                    if stopping is None:
                        await agent.__aexit__(None, None, None)
                    else:
                        await stopping
                    await model.aclose()

    async def test_late_results_wake_each_level_and_enable_more_delegation(self):
        started_b, started_c, release_b, release_c = (asyncio.Event() for _ in range(4))
        roles = demo_roles()
        roles["research"] = AgentRole("research", (DELEGATE_TOOL_NAME,), "prepare report")

        async def answer(messages, tools):
            last = messages[-1]
            if last.content == "begin":
                yield ToolReady(_call("start-a", "start A", agent="research"))
            elif last.content == "start A":
                yield ToolReady(_call("start-b", "start B"))
            elif last.content == "start B":
                started_b.set()
                await release_b.wait()
                yield TextDelta("B done")
            elif last.content == "start C":
                started_c.set()
                await release_c.wait()
                yield TextDelta("C done")
            elif last.content == "main followup":
                yield TextDelta("handled followup")
            elif last.tool_calls:
                yield TextDelta({"start-a": "main acknowledged", "start-b": "A pending B",
                                 "start-c": "A pending C", "start-d": "main pending D"}[(last.tool_calls[-1].id if last.tool_calls else None)])
            elif last.content == "start D":
                yield TextDelta("D done")
            else:
                result = _result_inputs([last])[0]["result"]
                if result == "B done":
                    yield ToolReady(_call("start-c", "start C"))
                elif result == "C done":
                    yield TextDelta("A final")
                elif result == "A final":
                    yield ToolReady(_call("start-d", "start D"))
                else:
                    yield TextDelta(f"main received {result}")

        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel, session_id="main", roles=roles) as (main, tree):
            main.submit_messages((MessageDelivery(Message("user", "begin"), "trigger"),))
            await _wait(started_b.wait())
            await _eventually(lambda: "main received A pending B" in [text for _, _, text in self.channel.replies])
            a = self._acceptances()["start-a"]
            a_session = self._session(a)
            a_first = self._run("start-a")
            self.assertEqual(a_first.result, "A pending B")
            self.assertEqual(set(a), {"status", "agent_id"})
            main.submit_messages((MessageDelivery(Message("user", "main followup"), "trigger"),))
            await _wait(main.wait_idle())
            self.assertIn("handled followup", [content for _, _, content in self.channel.replies])

            release_b.set()
            await _wait(started_c.wait())
            await _eventually(lambda: "main received A pending C" in [text for _, _, text in self.channel.replies])
            self.assertNotIn("start-d", self._acceptances())
            c_source = self._delegation("start-c", a_session).parent_run_id
            self.assertNotEqual(c_source, a_first.run_id)
            self.assertEqual(self.history.agent_run(c_source).result, "A pending C")
            self.assertEqual(self.history.run_delegation_ids(c_source), ())
            release_c.set()
            await _wait(tree.wait_idle())

        main_notices = _result_inputs(self._messages("main"))
        a_notices = [notice for notice in main_notices if notice["agent_id"] == a["agent_id"]]
        self.assertEqual([notice["result"] for notice in a_notices], ["A pending B", "A pending C", "A final"])
        a_runs = self._agent_runs(a["agent_id"])
        self.assertEqual([run.result for run in a_runs], [notice["result"] for notice in a_notices])
        self.assertEqual([run.status for run in a_runs], ["completed"] * 3)
        self.assertEqual([self.history.run_delegation_ids(run.run_id) for run in a_runs],
                         [(self._delegation("start-a").delegation_id,), (), ()])
        self.assertEqual([notice["result"] for notice in _result_inputs(self._messages(a_session))], ["B done", "C done"])
        self.assertEqual([notice["result"] for notice in main_notices if notice["agent_id"] != a["agent_id"]], ["D done"])

    async def test_parent_records_reply_content_without_source_lookup_or_deduplication(self):
        async def answer(messages, tools):
            match messages[-1].content:
                case "delegate":
                    yield ToolReady(_call("child", "work"))
                case "work":
                    yield TextDelta("persisted result")
                case _:
                    yield TextDelta("acknowledged")

        schedulers = SessionSchedulers()
        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel,
                                 session_id="main", roles=demo_roles(), schedulers=schedulers) as (main, tree):
            main.submit_messages((MessageDelivery(Message("user", "delegate"), "trigger"),))
            await _wait(tree.wait_idle())
            child_run = self._run("child")
            self.assertEqual((child_run.status, child_run.result), ("completed", "persisted result"))
            parent = schedulers.get_or_create("main")
            changed = AgentReply(child_run.agent_id, child_run.run_id, result="delivered result")
            without_run = AgentReply(child_run.agent_id, "unknown-run", error="delivered error")
            parent.session_input(tuple(MessageDelivery(reply.to_history_message(), "trigger") for reply in (changed, changed, without_run)))
            await _wait(tree.wait_idle())

        expected = [reply.to_history_message().async_result.payload for reply in (
            AgentReply(child_run.agent_id, child_run.run_id, result="persisted result"), changed, changed, without_run)]
        self.assertEqual(_result_inputs(self._messages("main")), expected)
        self.assertEqual(_result_inputs(model.requests[-1][0]), expected)
        self.assertEqual(self._run("child"), child_run)

    async def test_safe_boundary_merges_queued_delegations_and_subresult(self):
        a_waiting, release_a, b_waiting, release_b = (asyncio.Event() for _ in range(4))
        roles = demo_roles()
        roles["research"] = AgentRole("research", (DELEGATE_TOOL_NAME,), "combine delegated work")

        async def answer(messages, tools):
            last = messages[-1]
            match last.content:
                case "begin":
                    yield ToolReady(_call("first", "A first", agent="research"))
                case "A first":
                    yield TextDelta("intermediate tool prose")
                    yield ToolReady(_call("start-b", "B work"))
                    a_waiting.set()
                    await release_a.wait()
                case "B work":
                    b_waiting.set()
                    await release_b.wait()
                    yield TextDelta("B ready")
                case "enqueue more":
                    target = self._acceptances()["first"]["agent_id"]
                    yield ToolReady(_call("second", "A second", agent_id=target))
                    yield ToolReady(_call("third", "A third", agent_id=target))
                case _:
                    if _result_inputs([last]) and _result_inputs([last])[0]["result"] == "B ready":
                        yield TextDelta("A merged final")
                    else:
                        yield TextDelta("acknowledged")

        schedulers = SessionSchedulers()
        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel, session_id="main",
                                 roles=roles, schedulers=schedulers) as (main, tree):
            main.submit_messages((MessageDelivery(Message("user", "begin"), "trigger"),))
            await _wait(a_waiting.wait())
            await _wait(b_waiting.wait())
            await _wait(main.wait_idle())
            main.submit_messages((MessageDelivery(Message("user", "enqueue more"), "trigger"),))
            await _wait(main.wait_idle())
            call_ids = ("first", "second", "third")
            a_session = self._session(self._acceptances()[call_ids[0]])
            self.assertEqual([self._delegation(call_id).status for call_id in call_ids],
                             ["consumed", "queued", "queued"])
            a_history = self._messages(a_session)
            self.assertEqual([row.content for row in a_history], ["A first", ""])
            self.assertEqual([call.id for call in a_history[-1].tool_calls], ["start-b"])
            scheduler = schedulers.get_or_create(a_session)
            with patch.object(scheduler, "submit_internal", wraps=scheduler.submit_internal) as submit:
                release_b.set()
                await _eventually(lambda: submit.called)
                submit.assert_called_once()
                self.assertEqual(self._messages(a_session), a_history)
            release_a.set()
            await _wait(tree.wait_idle())

        replies = _result_inputs(self._messages("main"))
        reply, = replies
        delegations = [self._delegation(call_id) for call_id in call_ids]
        merged_run_id, = {item.run_id for item in delegations}
        self.assertEqual(reply, {"agent_id": delegations[0].agent_id, "status": "completed", "result": "A merged final"})
        self.assertEqual(self.history.run_delegation_ids(merged_run_id), tuple(item.delegation_id for item in delegations))
        a_messages = self._messages(a_session)
        self.assertEqual(len(_result_inputs(a_messages)), 1)
        self.assertEqual([row.content for row in a_messages if row.kind == "user"][:3], ["A first", "A second", "A third"])
        self.assertEqual([row.content for row in a_messages if row.kind == "assistant"],
                         ["intermediate tool prose", "A merged final"])
        self.assertEqual({row.run_id for row in self.history.outcomes(a_session)}, {merged_run_id})

    async def test_busy_parent_does_not_admit_result_until_current_delivery_settles(self):
        release_child = asyncio.Event()

        async def answer(messages, tools):
            last = messages[-1]
            if last.content == "delegate and speak":
                yield ToolReady(_call("child", "work"))
            elif last.content == "work":
                await release_child.wait()
                yield TextDelta("child ready")
            elif last.tool_calls:
                yield TextDelta("hold-delivery")
            else:
                yield TextDelta("parent got result")

        schedulers = SessionSchedulers()
        scheduler = schedulers.get_or_create("main")
        with patch.object(scheduler, "submit_internal", wraps=scheduler.submit_internal) as submit:
            async with running_agent(self.settings, self.history, _Model(answer), self.channel,
                                     session_id="main", roles=demo_roles(), schedulers=schedulers) as (main, tree):
                main.submit_messages((MessageDelivery(Message("user", "delegate and speak"), "trigger"),))
                await _wait(self.channel.blocked.wait())
                release_child.set()
                await _eventually(lambda: submit.called)
                submit.assert_called_once()
                self.assertEqual(_result_inputs(self._messages("main")), [])
                self.channel.release.set()
                await _wait(tree.wait_idle())

        self.assertEqual([content for _, _, content in self.channel.replies], ["hold-delivery", "parent got result"])
        self.assertNotEqual(self.channel.replies[0][1], self.channel.replies[1][1])

    async def test_reuse_is_serial_and_many_distinct_agents_can_run(self):
        started = set()
        changed, release = asyncio.Event(), asyncio.Event()

        async def answer(messages, tools):
            last = messages[-1]
            if last.content == "start ten":
                for i in range(10):
                    yield ToolReady(_call(f"job-{i}", f"job {i}"))
            elif last.content == "reuse one":
                target = self._acceptances()["job-0"]["agent_id"]
                yield ToolReady(_call("again", "job again", agent_id=target))
            elif not tools:
                started.add(last.content)
                changed.set()
                await release.wait()
                yield TextDelta(f"done {last.content}")
            else:
                yield TextDelta("parent processed queue")

        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel, session_id="main",
                                 roles=demo_roles()) as (main, tree):
            main.submit_messages((MessageDelivery(Message("user", "start ten"), "trigger"),))
            async with asyncio.timeout(2):
                while len(started) < 10:
                    changed.clear()
                    await changed.wait()
            await _wait(main.wait_idle())
            first = self._acceptances()["job-0"]
            self.assertEqual(len({item["agent_id"] for item in self._acceptances().values()}), 10)
            main.submit_messages((MessageDelivery(Message("user", "reuse one"), "trigger"),))
            await _wait(main.wait_idle())
            again = self._acceptances()["again"]
            self.assertEqual(again["agent_id"], first["agent_id"])
            queued = self._delegation("again")
            self.assertNotEqual(queued.delegation_id, self._delegation("job-0").delegation_id)
            self.assertEqual(queued.status, "queued")
            self.assertIsNone(queued.run_id)
            self.assertNotIn("job again", started)
            self.assertEqual([row.content for row in self._messages(self._session(first))], ["job 0"])
            release.set()
            await _wait(tree.wait_idle())

        self.assertEqual(len(_result_inputs(self._messages("main"))), 11)
        self.assertNotEqual(self._run("job-0").run_id, self._run("again").run_id)
        self.assertEqual([row.content for row in self._messages(self._session(first))],
                         ["job 0", "done job 0", "job again", "done job again"])

    async def test_child_model_self_cancellation_fails_once_and_drains_queued_delegation(self):
        started, cancel_model = asyncio.Event(), asyncio.Event()

        async def answer(messages, tools):
            content = messages[-1].content
            if content == "first user":
                yield ToolReady(_call("first", "cancel this"))
            elif content == "second user":
                yield ToolReady(_call("second", "next job", agent_id=self._acceptances()["first"]["agent_id"]))
            elif content == "cancel this":
                yield TextDelta("unfinished partial text")
                started.set()
                await cancel_model.wait()
                raise asyncio.CancelledError
            elif content == "next job":
                yield TextDelta("finished next job")
            else:
                yield TextDelta("ack")

        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel,
                                 session_id="main", roles=demo_roles()) as (main, tree):
            main.submit_messages((MessageDelivery(Message("user", "first user"), "trigger"),))
            await _wait(started.wait())
            await _wait(main.wait_idle())
            main.submit_messages((MessageDelivery(Message("user", "second user"), "trigger"),))
            await _wait(main.wait_idle())
            self.assertEqual(self._run("first").status, "running")
            self.assertEqual(self._delegation("second").status, "queued")
            cancel_model.set()
            await _wait(tree.wait_idle())

            failed, completed = self._run("first"), self._run("second")
            self.assertEqual((failed.status, failed.result, failed.error), ("failed", None, "model_cancelled"))
            self.assertEqual((completed.status, completed.result), ("completed", "finished next job"))
            expected_replies = [
                {"agent_id": failed.agent_id, "status": "failed", "error": failed.error},
                {"agent_id": completed.agent_id, "status": "completed", "result": completed.result},
            ]
            stored = self._messages("main")
            self.assertEqual(_result_inputs(stored), expected_replies)
            requests = [message for messages, _ in model.requests for message in messages]
            request_replies = _result_inputs(requests)
            for reply in expected_replies:
                self.assertIn(reply, request_replies)
            expected_acceptance = {"status": "accepted", "agent_id": completed.agent_id}
            jobs = [self._delegation(call_id) for call_id in ("first", "second")]
            internal_ids = {run.run_id for run in (failed, completed)}
            internal_ids.update(job.delegation_id for job in jobs)
            internal_ids.update(job.parent_run_id for job in jobs)
            for messages in (stored, requests):
                tools = all_tool_calls(messages)
                self.assertEqual({call.id for call in tools}, {"first", "second"})
                for message in tools:
                    self.assertEqual(json.loads(message.result), expected_acceptance)
                for message in messages:
                    for internal_id in internal_ids:
                        self.assertNotIn(internal_id, str(message.content))

    async def test_role_tools_apply_to_delegated_tasks_and_child_replies(self):
        roles = demo_roles()
        roles["general"] = AgentRole("worker", (SET_TIMER_TOOL_NAME, CANCEL_TIMER_TOOL_NAME), "analysis")

        async def answer(messages, tools):
            last = messages[-1]
            if last.content == "user work":
                yield ToolReady(_call("authorized", "child work"))
            elif last.content == "child work":
                yield ToolReady(_call("forbidden-tool", "never"))
                yield ToolReady(ToolCall("child-timer", SET_TIMER_TOOL_NAME, '{"interval_seconds":100}'))
                yield ToolReady(ToolCall("child-cancel", CANCEL_TIMER_TOOL_NAME, '{"timer_id":"missing"}'))
            elif _result_inputs([last]):
                yield ToolReady(ToolCall("result-timer", SET_TIMER_TOOL_NAME, '{"interval_seconds":100}'))
            else:
                yield TextDelta("final response")

        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel, session_id="main", roles=roles) as (main, tree):
            main.submit_messages((MessageDelivery(Message("user", "user work"), "trigger"),))
            await _wait(tree.wait_idle())

        job = self._acceptances()["authorized"]
        results = self._acceptances(self._session(job))
        self.assertEqual(results["forbidden-tool"], {"error": "unknown_tool"})
        self.assertEqual(results["child-timer"]["status"], "accepted")
        self.assertEqual(results["child-cancel"], {"status": "not_active"})
        self.assertEqual(self._acceptances()["result-timer"]["status"], "accepted")
        self.assertIn((SET_TIMER_TOOL_NAME, CANCEL_TIMER_TOOL_NAME), [tools for _, tools in model.requests])
        notice, = _result_inputs(self._messages("main"))
        self.assertEqual(notice["result"], "final response")

    async def test_invalid_targets_and_arguments_never_create_records(self):
        root = self.history.ensure_main_agent("main")
        other = self.history.ensure_main_agent("other")
        fixture = MessageDelivery(Message("user", "outside fixture"), "trigger")
        self.history.append_agent_inputs(other.agent_id, "outside", (fixture,), start=True)
        outsider, _ = self.history.accept_delegation(other.agent_id, "outside", "outside-child", role_id="general")
        roles = demo_roles()
        roles["research"] = AgentRole("research", (DELEGATE_TOOL_NAME,), "review ownership")

        async def answer(messages, tools):
            last = messages[-1]
            if last.content == "setup":
                yield ToolReady(_call("setup-sibling", "sibling"))
            elif last.content == "test root":
                yield ToolReady(_call("self", "deny", agent_id=root.agent_id))
                yield ToolReady(_call("outside", "deny", agent_id=outsider.agent_id))
                yield ToolReady(_call("unknown", "deny", agent_id="not-an-agent"))
                yield ToolReady(_call("invalid-role", "deny", agent="main"))
                yield ToolReady(_call("unknown-role", "deny", agent="missing"))
                yield ToolReady(_call("both", "deny", agent="general", agent_id=root.agent_id))
                yield ToolReady(_call("blank", "   "))
                yield ToolReady(ToolCall("missing-prompt", DELEGATE_TOOL_NAME, '{"agent":"general"}'))
                yield ToolReady(ToolCall("no-target", DELEGATE_TOOL_NAME, '{"prompt":"deny"}'))
                yield ToolReady(_call("research", "test child", agent="research"))
            elif last.content == "test child":
                sibling = self._acceptances()["setup-sibling"]["agent_id"]
                yield ToolReady(_call("parent", "deny", agent_id=root.agent_id))
                yield ToolReady(_call("sibling", "deny", agent_id=sibling))
            else:
                yield TextDelta("done")

        async with running_agent(self.settings, self.history, _Model(answer), self.channel,
                                 session_id="main", roles=roles) as (main, tree):
            main.submit_messages((MessageDelivery(Message("user", "setup"), "trigger"),))
            await _wait(tree.wait_idle())
            initial_count = self.history._db.execute("SELECT count(*) FROM delegations").fetchone()[0]
            main.submit_messages((MessageDelivery(Message("user", "test root"), "trigger"),))
            await _wait(tree.wait_idle())

        results = self._acceptances()
        for key in ("self", "outside", "unknown"):
            self.assertEqual(results[key], {"error": "unknown_child_agent"})
        for key in ("invalid-role", "unknown-role"):
            self.assertEqual(results[key], {"error": "unknown_agent_role"})
        for key in ("both", "blank", "missing-prompt", "no-target"):
            self.assertEqual(results[key], {"error": "invalid_arguments"})
        self.assertEqual(self._acceptances(self._session(results["research"])),
                         {"parent": {"error": "unknown_child_agent"}, "sibling": {"error": "unknown_child_agent"}})
        self.assertEqual(self.history._db.execute("SELECT count(*) FROM delegations").fetchone()[0], initial_count + 1)

    async def test_empty_or_failed_child_returns_error_without_leaking_earlier_tool_text(self):
        roles = demo_roles()
        roles["general"] = AgentRole("worker", (CANCEL_TIMER_TOOL_NAME,), "test responses")

        async def answer(messages, tools):
            last = messages[-1]
            if last.content == "test errors":
                yield ToolReady(_call("fail", "failing work"))
                yield ToolReady(_call("empty", "empty work"))
            elif last.content == "failing work":
                yield TextDelta("uncommitted partial output")
                raise ModelError("truncated_stream")
            elif last.content == "empty work":
                yield TextDelta("tool text only")
                yield ToolReady(ToolCall("cancel", CANCEL_TIMER_TOOL_NAME, '{"timer_id":"missing"}'))
            elif (last.tool_calls[-1].id if last.tool_calls else None) != "cancel":
                yield TextDelta("parent reply")

        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel,
                                 session_id="main", roles=roles) as (main, tree):
            main.submit_messages((MessageDelivery(Message("user", "test errors"), "trigger"),))
            await _wait(tree.wait_idle())
        cases = {notice["agent_id"]: notice for notice in _result_inputs(self._messages("main"))}
        for call_id, error in (("fail", "model_failed"), ("empty", "empty_result")):
            run = self._run(call_id)
            self.assertEqual(run.error, error)
            self.assertEqual(cases[run.agent_id], {"agent_id": run.agent_id, "status": "failed", "error": error})
        empty = self._acceptances()["empty"]
        self.assertIn("tool text only", [row.content for row in self._messages(self._session(empty))])

    async def test_opted_in_child_can_finish_silently_without_parent_wakeup(self):
        roles = demo_roles()
        roles["review"] = AgentRole("审阅任务", (SILENT_END_TOOL_NAME,), "支持静默的子角色")

        async def answer(messages, tools):
            last = messages[-1]
            if last.content in ("请求静默任务", "请求正文任务"):
                self.assertNotIn(SILENT_END_TOOL_NAME, tools)
                call_id, prompt = ("silent-job", "静默任务") if last.content == "请求静默任务" else ("text-job", "正文任务")
                yield ToolReady(_call(call_id, prompt, agent="review"))
            elif last.content in ("静默任务", "正文任务"):
                self.assertEqual(tools, (SILENT_END_TOOL_NAME,))
                yield ToolReady(ToolCall("finish-child", SILENT_END_TOOL_NAME, "{}"))
                if last.content == "正文任务":
                    yield TextDelta("需要汇报的子结果")
            elif _result_inputs([last]):
                yield TextDelta("父级收到结果")
            elif last.tool_calls:
                yield TextDelta("父级已派发")

        model = _Model(answer, tool_names=True)
        async with running_agent(self.settings, self.history, model, self.channel,
                                 session_id="main", roles=roles) as (main, tree):
            main.submit_messages((MessageDelivery(Message("user", "请求静默任务"), "trigger"),))
            await _wait(tree.wait_idle())
            run = self._run("silent-job")
            self.assertEqual((run.status, run.result, run.error), ("completed", None, None))
            self.assertEqual(_result_inputs(self._messages("main")), [])
            self.assertEqual(len(self._agent_runs(main.agent.agent_id)), 1)
            self.assertEqual(len(model.requests), 3)
            child_session = self._session(self._acceptances()["silent-job"])
            self.assertEqual(self._acceptances(child_session)["finish-child"], {"status": "accepted"})

            main.submit_messages((MessageDelivery(Message("user", "请求正文任务"), "trigger"),))
            await _wait(tree.wait_idle())
            run = self._run("text-job")
            self.assertEqual((run.status, run.result, run.error), ("completed", "需要汇报的子结果", None))
            self.assertEqual(_result_inputs(self._messages("main")),
                             [{"agent_id": run.agent_id, "status": "completed", "result": "需要汇报的子结果"}])
            child_session = self._session(self._acceptances()["text-job"])
            self.assertEqual(self._acceptances(child_session)["finish-child"],
                             {"status": "ignored", "reason": "text_present"})

    async def test_round_budget_is_per_run_and_late_results_survive_exhaustion(self):
        release_children, failed = asyncio.Event(), asyncio.Event()
        roles = demo_roles()
        roles["research"] = AgentRole("research", (DELEGATE_TOOL_NAME,), "test round budget")
        settings = replace(self.settings, runtime=replace(self.settings.runtime, max_tool_rounds=2))

        async def answer(messages, tools):
            last = messages[-1]
            if last.content == "root":
                yield ToolReady(_call("start-a", "work A", agent="research"))
            elif last.content == "work A":
                yield ToolReady(_call("start-b", "work B"))
            elif (last.tool_calls[-1].id if last.tool_calls else None) == "start-b":
                yield ToolReady(_call("start-c", "work C"))
            elif last.content in ("work B", "work C"):
                await release_children.wait()
                yield TextDelta(f"done {last.content}")
            elif tools == (DELEGATE_TOOL_NAME,) and _result_inputs([last]):
                yield TextDelta("A processes late result")
            else:
                notices = _result_inputs([last])
                if notices and notices[0].get("error") == "tool_round_limit":
                    failed.set()
                yield TextDelta("root response")

        model = _Model(answer, tool_names=True)
        async with running_agent(settings, self.history, model, self.channel, session_id="main", roles=roles) as (main, tree):
            main.submit_messages((MessageDelivery(Message("user", "root"), "trigger"),))
            await _wait(failed.wait())
            a = self._acceptances()["start-a"]
            a_session = self._session(a)
            a_run = self._run("start-a")
            self.assertEqual((a_run.status, a_run.error), ("failed", "tool_round_limit"))
            self.assertEqual([row.run_id for row in self.history.outcomes(a_session)], [a_run.run_id, a_run.run_id])
            self.assertEqual({self._delegation(call_id, a_session).parent_run_id for call_id in self._acceptances(a_session)},
                             {a_run.run_id})
            release_children.set()
            await _wait(tree.wait_idle())

        notices = [notice for notice in _result_inputs(self._messages("main")) if notice["agent_id"] == a["agent_id"]]
        self.assertEqual([notice.get("result") for notice in notices], [None, "A processes late result", "A processes late result"])
        a_runs = self._agent_runs(a["agent_id"])
        self.assertEqual([run.result for run in a_runs], [notice.get("result") for notice in notices])
        self.assertTrue(all(self.history.run_delegation_ids(run.run_id) == () for run in a_runs[1:]))

    async def test_trusted_reply_wakes_failed_input_batch_in_fifo_without_reexecuting_tool(self):
        for timing in ("before failure is handled", "while worker is exiting", "after worker stops"):
            with self.subTest(timing=timing), closing(History(":memory:")) as history:
                child_started, allow_child, allow_tool_round_to_end = (asyncio.Event() for _ in range(3))
                at_boundary, resume_worker = asyncio.Event(), asyncio.Event()
                attempts = 0

                async def answer(messages, tools):
                    content = messages[-1].content
                    if content == "delegate once":
                        yield ToolReady(_call("only", "child task"))
                        await allow_tool_round_to_end.wait()
                    elif content == "child task":
                        child_started.set()
                        await allow_child.wait()
                        yield TextDelta("trusted result")
                    else:
                        yield TextDelta("ack")

                model = _Model(answer)
                error = OperationalError("simulated statement IOERR")
                error.sqlite_errorcode = SQLITE_IOERR
                append = history._append
                speech = Message("user", "head speech")
                schedulers = SessionSchedulers()
                scheduler = schedulers.get_or_create("main")

                def fail_twice(session_id, messages, **kwargs):
                    nonlocal attempts
                    rows = tuple(messages)
                    ids = append(session_id, rows, **kwargs)
                    if rows == (speech,):
                        attempts += 1
                        if attempts <= 2:
                            raise error
                    return ids

                async with running_agent(self.settings, history, model, _Channel(),
                                         session_id="main", roles=demo_roles(), schedulers=schedulers) as (main, tree):
                    action = "_record_input" if timing == "before failure is handled" else "_finish_agent_run"
                    original = getattr(main, action)

                    async def pause_worker(*args, **kwargs):
                        try:
                            return await original(*args, **kwargs)
                        finally:
                            if attempts == 2 and not at_boundary.is_set() and timing != "after worker stops":
                                at_boundary.set()
                                await resume_worker.wait()

                    with (patch.object(history, "_append", side_effect=fail_twice),
                          patch.object(main, action, side_effect=pause_worker),
                          self.assertLogs("agent_companion.runtime.engine", "ERROR")):
                        main.submit_messages((MessageDelivery(Message("user", "delegate once"), "trigger"),))
                        await _wait(child_started.wait())
                        head = MessageDelivery(Message("user", "head record"), "record")
                        main.submit_messages((head, MessageDelivery(speech, "trigger")))
                        main.submit_messages((MessageDelivery(Message("user", "queued later"), "trigger"),))
                        allow_tool_round_to_end.set()
                        if timing == "after worker stops":
                            await _wait(main.wait_idle())
                            self.assertEqual(main.last_failure, "input_write_failed")
                            self.assertIsNone(main._worker)
                        else:
                            await _wait(at_boundary.wait())
                            self.assertFalse(main.idle)
                        self.assertTrue(main.has_pending_messages)
                        (tool_result,) = all_tool_calls(r.message for r in history.read_after("main"))
                        with patch.object(scheduler, "submit_internal", wraps=scheduler.submit_internal) as submit:
                            allow_child.set()
                            await _eventually(lambda: submit.called)
                        resume_worker.set()
                        await _wait(tree.wait_idle())

                    self.assertEqual(attempts, 3)
                    self.assertFalse(main.has_pending_messages)
                    messages = [r.message for r in history.read_after("main")]
                    self.assertEqual([r.content for r in messages if r.kind == "user"],
                                     ["delegate once", "head record", "head speech", "queued later"])
                    self.assertEqual(all_tool_calls(messages), [tool_result])
                    self.assertEqual([r.get("result") for r in _result_inputs(messages)], ["trusted result"])
                    self.assertEqual(len(model.requests), 5)  # 初始请求、一次子运行、队首、后续输入、回投。

    async def test_storage_failure_before_acceptance_does_not_queue_child(self):
        async def answer(messages, tools):
            if messages[-1].content == "run tool":
                yield ToolReady(_call("fault", "unreachable child"))
            else:
                yield TextDelta("parent still working")

        model = _Model(answer)
        with patch.object(self.history, "accept_delegation", side_effect=OSError("synthetic")):
            async with running_agent(self.settings, self.history, model, self.channel,
                                     session_id="main", roles=demo_roles()) as (main, tree):
                main.submit_messages((MessageDelivery(Message("user", "run tool"), "trigger"),))
                await _wait(tree.wait_idle())
        self.assertEqual(self._acceptances()["fault"], {"error": "tool_failed"})
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(self.history._db.execute("SELECT count(*) FROM agents").fetchone()[0], 1)
        self.assertEqual(_result_inputs(self._messages("main")), [])

    async def test_context_lock_retries_are_bounded_without_replaying_the_child_tool(self):
        for locked_reads in (1, 3):
            with self.subTest(locked_reads=locked_reads), closing(History(":memory:")) as history:
                roles = demo_roles()
                roles["general"] = replace(roles["general"], tools=(CANCEL_TIMER_TOOL_NAME,))
                child_requests = read_failures = 0
                snapshot = history.context_snapshot

                def read_context(session_id, **kwargs):
                    nonlocal read_failures
                    if session_id != "main" and child_requests == 1 and read_failures < locked_reads:
                        read_failures += 1
                        raise OperationalError("database is locked")
                    return snapshot(session_id, **kwargs)

                async def answer(messages, tools):
                    nonlocal child_requests
                    if tools == (CANCEL_TIMER_TOOL_NAME,):
                        child_requests += 1
                        if child_requests == 1:
                            yield ToolReady(ToolCall("child-tool", CANCEL_TIMER_TOOL_NAME, '{"timer_id":"fixture"}'))
                        else:
                            yield TextDelta("child finished")
                    elif messages[-1].content == "run child":
                        yield ToolReady(_call("start", "child work"))
                    else:
                        yield TextDelta("root response")

                model = _Model(answer, tool_names=True)
                with (patch.object(history, "context_snapshot", side_effect=read_context),
                      patch("agent_companion.runtime.sqlite_access._LOCK_RETRIES", 2)):
                    async with running_agent(self.settings, history, model, self.channel, session_id="main",
                                             retry_sqlite_locks=True, roles=roles) as (main, tree):
                        main.submit_messages((MessageDelivery(Message("user", "run child"), "trigger"),))
                        await _wait(tree.wait_idle())
                        rows = [row.message for row in history.read_after("main")]
                        accepted = json.loads(all_tool_calls(rows)[0].result)
                        run = history.agent_run(stored_delegation(history, "main", "start").run_id)
                        child = history.agent(accepted["agent_id"])
                        tool_results = all_tool_calls(row.message for row in history.read_after(child.session_id))
                        notice, = _result_inputs(rows)
                        self.assertEqual(notice["agent_id"], run.agent_id)
                        self.assertEqual([call.id for call in tool_results], ["child-tool"])
                        self.assertEqual(read_failures, locked_reads)
                        if locked_reads == 1:
                            self.assertEqual(child_requests, 2)
                            self.assertEqual((run.status, notice["result"]), ("completed", "child finished"))
                        else:
                            self.assertEqual(child_requests, 1)
                            self.assertEqual((run.status, run.error), ("failed", "context_lock_timeout"))
                            self.assertEqual(notice["error"], run.error)

    async def test_reply_archive_lock_is_not_a_context_failure_and_does_not_replay(self):
        roles = demo_roles()
        roles["general"] = replace(roles["general"], tools=(CANCEL_TIMER_TOOL_NAME,))
        archive_attempts = []
        finalize = self.history.finalize_reply

        def archive(outcome, rows):
            if outcome.session_id != "main":
                archive_attempts.append(outcome.reply_id)
                error = OperationalError("database is locked")
                error.sqlite_errorcode = SQLITE_BUSY
                raise error
            return finalize(outcome, rows)

        async def answer(messages, tools):
            if tools == (CANCEL_TIMER_TOOL_NAME,):
                yield ToolReady(ToolCall("child-tool", CANCEL_TIMER_TOOL_NAME, '{"timer_id":"fixture"}'))
            elif messages[-1].content == "run child":
                yield ToolReady(_call("start", "child work"))
            else:
                yield TextDelta("root response")

        model = _Model(answer, tool_names=True)
        with (patch.object(self.history, "finalize_reply", side_effect=archive),
              patch.object(Timers, "_cancel", autospec=True, return_value='{"status":"not_active"}') as tool,
              patch.object(self.history, "finish_agent_run", wraps=self.history.finish_agent_run) as finish,
              patch("agent_companion.runtime.sqlite_access._LOCK_RETRIES", 2),
              self.assertLogs("agent_companion.runtime.engine", level="ERROR")):
            async with running_agent(self.settings, self.history, model, self.channel, session_id="main",
                                     retry_sqlite_locks=True, roles=roles) as (main, tree):
                main.submit_messages((MessageDelivery(Message("user", "run child"), "trigger"),))
                await _wait(tree.wait_idle())

        run = self._run("start")
        notice, = _result_inputs(self._messages("main"))
        self.assertEqual((run.status, run.error), ("failed", "runtime_failed"))
        self.assertEqual(notice, {"agent_id": run.agent_id, "status": "failed", "error": run.error})
        self.assertEqual([call.kwargs for call in finish.call_args_list if call.args == (run.run_id,)],
                         [{"result": None, "error": run.error}])
        self.assertEqual(len(archive_attempts), 3)
        self.assertEqual(len(set(archive_attempts)), 1)
        self.assertEqual(self.history.outcomes(self.history.agent(run.agent_id).session_id), [])
        self.assertEqual(sum(tools == (CANCEL_TIMER_TOOL_NAME,) for _, tools in model.requests), 1)
        self.assertEqual(tool.await_count, 1)

    async def test_retiring_a_tree_reclaims_its_child_schedulers_and_keeps_history(self):
        schedulers = SessionSchedulers()
        channel = self.channel
        reuse = None

        async def answer(messages, tools):
            match messages[-1].content:
                case "start":
                    yield ToolReady(_call("create", "original task"))
                case "resume":
                    yield ToolReady(_call("resume", "follow-up task", agent_id=reuse))
                case _:
                    yield TextDelta("child response" if not tools else "root response")

        model = _Model(answer)

        def open_tree(session_id):
            return running_agent(self.settings, self.history, model, channel, session_id=session_id,
                                 schedulers=schedulers, roles=demo_roles())

        async with open_tree("other") as (other, other_tree):
            other.submit_messages((MessageDelivery(Message("user", "start"), "trigger"),))
            await _wait(other_tree.wait_idle())
            other_session = self._session(self._acceptances("other")["create"])
            other_scheduler = schedulers.get_or_create(other_session)
            previous_scheduler = None
            for text in ("start", "resume", "resume"):
                async with open_tree("main") as (main, tree):
                    main.submit_messages((MessageDelivery(Message("user", text), "trigger"),))
                    await _wait(tree.wait_idle())
                    accepted = self._acceptances()["create" if text == "start" else "resume"]
                    if reuse is None:
                        reuse = accepted["agent_id"]
                    self.assertEqual(accepted["agent_id"], reuse)
                    child_session = self._session(accepted)
                    current_scheduler = schedulers.get_or_create(child_session)
                    self.assertIsNot(current_scheduler, previous_scheduler)
                    previous_scheduler = current_scheduler
                self.assertNotIn(child_session, schedulers._sessions)
                self.assertIs(schedulers._sessions[other_session], other_scheduler)
            self.assertEqual([row.content for row in self._messages(child_session) if row.kind == "user"],
                             ["original task", "follow-up task", "follow-up task"])
        self.assertEqual(set(schedulers._sessions), {"main", "other"})

    async def test_terminal_write_failure_has_bounded_retry_and_no_replay(self):
        for writes_blocked, child_error in ((1, None), (4, None), (1, "model_failed"), (4, "model_failed")):
            with self.subTest(writes_blocked=writes_blocked, child_error=child_error), closing(History(":memory:")) as history:
                calls = 0
                session = "fault fixture"
                schedulers = SessionSchedulers()
                scheduler = schedulers.get_or_create(session)
                observed_at_handoff = []
                attempted_run = None

                async def answer(messages, tools):
                    last = messages[-1]
                    if last.content == "try child":
                        yield ToolReady(_call("child", "child work"))
                    elif last.content == "child work":
                        if child_error:
                            raise ModelError("truncated_stream")
                        yield TextDelta("child result")
                    else:
                        yield TextDelta("root response")

                model = _Model(answer)
                finish = history.finish_agent_run
                submit = scheduler.submit_internal

                def attempt(run_id, *, result, error):
                    nonlocal calls, attempted_run
                    if result == "child result" or error == "model_failed":
                        attempted_run = run_id
                        calls += 1
                        if calls <= writes_blocked:
                            raise OperationalError("database is locked")
                    finish(run_id, result=result, error=error)

                def observe(item):
                    self.assertEqual(item.intent, "trigger")
                    observed_at_handoff.append(history.agent_run(attempted_run))
                    return submit(item)

                with (patch.object(history, "finish_agent_run", side_effect=attempt),
                      patch.object(scheduler, "submit_internal", side_effect=observe),
                      patch("agent_companion.runtime.sqlite_access._LOCK_RETRIES", 2)):
                    async with running_agent(self.settings, history, model, self.channel, session_id=session,
                                             retry_sqlite_locks=True, roles=demo_roles(),
                                             schedulers=schedulers) as (main, tree):
                        main.submit_messages((MessageDelivery(Message("user", "try child"), "trigger"),))
                        if writes_blocked > 2:
                            with self.assertLogs("agent_companion.runtime.engine", level="ERROR") as logs:
                                await _wait(tree.wait_idle())
                            self.assertIn("agent terminal write failed", logs.output[0])
                        else:
                            await _wait(tree.wait_idle())
                        messages = [item.message for item in history.read_after(session)]
                        notices = _result_inputs(messages)
                        (tool,) = all_tool_calls(messages)
                        run_id = stored_delegation(history, session, tool.id).run_id
                    persisted = history.agent_run(run_id)

                self.assertEqual(calls, 2 if writes_blocked == 1 else 3)
                self.assertEqual(sum(not tools for _, tools in model.requests), 1)
                if writes_blocked > 2:
                    self.assertEqual(observed_at_handoff, [])
                    self.assertEqual(notices, [])
                    self.assertEqual(persisted.status, "interrupted")
                else:
                    self.assertEqual(observed_at_handoff, [persisted])
                    notice = AgentReply(persisted.agent_id, run_id, error=child_error,
                                       result=None if child_error else "child result")
                    payload = notice.to_history_message().async_result.payload
                    self.assertEqual(notices, [payload])
                    self.assertEqual(persisted.status, payload["status"])
                    self.assertEqual((persisted.result, persisted.error), (notice.result, notice.error))

    async def test_shutdown_interrupts_running_and_queued_work_and_reopens_without_replay(self):
        schedulers = SessionSchedulers()
        entered, cancellation, finish_cleanup, exit_requested = (asyncio.Event() for _ in range(4))
        roles = demo_roles()

        async def answer(messages, tools):
            last = messages[-1]
            if last.content == "start":
                yield ToolReady(_call("hold", "wait for shutdown"))
            elif last.content == "queue":
                yield ToolReady(_call("queue", "must not start", agent_id=self._acceptances()["hold"]["agent_id"]))
            elif last.content == "wait for shutdown":
                entered.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    cancellation.set()
                    await finish_cleanup.wait()
                    raise
            else:
                yield TextDelta("parent running")

        model = _Model(answer)

        async def serve():
            async with running_agent(self.settings, self.history, model, self.channel,
                                     session_id="main", schedulers=schedulers, roles=roles) as (main, _):
                main.submit_messages((MessageDelivery(Message("user", "start"), "trigger"),))
                await entered.wait()
                await main.wait_idle()
                main.submit_messages((MessageDelivery(Message("user", "queue"), "trigger"),))
                await main.wait_idle()
                exit_requested.set()
                await asyncio.Future()

        serving = asyncio.create_task(serve())
        try:
            await _wait(exit_requested.wait())
            accepted = self._acceptances()
            main_before = self.history.ensure_main_agent("main")
            self.assertEqual(self._delegation("queue").status, "queued")
            with self.assertRaisesRegex(RuntimeError, "already bound or owned"):
                async with running_agent(self.settings, self.history, model, self.channel,
                                         session_id="main", schedulers=schedulers, roles=demo_roles()):
                    self.fail("the original route still owns its running child")
            self.assertEqual(self._run("hold").status, "running")

            serving.cancel()
            await _wait(cancellation.wait())
            serving.cancel()
            self.assertFalse(schedulers.submit_internal("main", MessageDelivery(Message("user", "late"), "trigger")))
            with self.assertRaisesRegex(RuntimeError, "already bound or owned"):
                schedulers.get_or_create("main").reserve_agent()
            finish_cleanup.set()
            with self.assertRaises(asyncio.CancelledError):
                await _wait(asyncio.shield(serving))
        finally:
            finish_cleanup.set()
            if not serving.done():
                serving.cancel()
                await asyncio.gather(serving, return_exceptions=True)

        self.assertEqual(self._run("hold").status, "interrupted")
        self.assertEqual(_result_inputs(self._messages("main")), [])
        self.assertEqual(self._delegation("queue").status, "interrupted")
        self.assertIsNone(self._delegation("queue").run_id)
        old_child_id = accepted["hold"]["agent_id"]

        async def reopened_answer(messages, tools):
            if messages[-1].content == "reuse":
                yield ToolReady(_call("reuse", "fresh work", agent_id=old_child_id))
            else:
                yield TextDelta("restart reply")

        next_model = _Model(reopened_answer)
        async with running_agent(self.settings, self.history, next_model, self.channel,
                                 session_id="main", schedulers=schedulers, roles=roles) as (main, tree):
            self.assertEqual(main.agent, main_before)
            await _wait(tree.wait_idle())
            self.assertEqual(next_model.requests, [])
            main.submit_messages((MessageDelivery(Message("user", "reuse"), "trigger"),))
            await _wait(tree.wait_idle())
        again = self._acceptances()["reuse"]
        self.assertEqual(again["agent_id"], old_child_id)
        self.assertEqual(self._run("reuse").status, "completed")
        self.assertEqual([row.content for row in self._messages(self._session(again))],
                         ["wait for shutdown", "fresh work", "restart reply"])

    async def test_user_input_joins_running_internal_turn_only_at_safe_boundary(self):
        blocked, release = asyncio.Event(), asyncio.Event()

        async def answer(messages, tools):
            last = messages[-1]
            if last.content == "synthetic internal event":
                yield ToolReady(_call("early", "invalid creation", agent="missing"))
                blocked.set()
                await release.wait()
            elif last.content == "new user message":
                yield ToolReady(_call("after-boundary", "child input"))
            else:
                yield TextDelta("reply")

        async with running_agent(self.settings, self.history, _Model(answer), self.channel,
                                 session_id="main", roles=demo_roles()) as (main, tree):
            message = Message("user", "synthetic internal event", source=UserSource("internal"))
            main.submit_messages((MessageDelivery(message, "trigger"),))
            await _wait(blocked.wait())
            main.submit_messages((MessageDelivery(Message("user", "new user message"), "trigger"),))
            history = self._messages("main")
            self.assertEqual([message.content for message in history], ["synthetic internal event", ""])
            self.assertEqual([call.id for call in history[-1].tool_calls], ["early"])
            release.set()
            await _wait(tree.wait_idle())
        self.assertEqual(self._acceptances()["early"], {"error": "unknown_agent_role"})
        source = self._delegation("after-boundary").parent_run_id
        self.assertEqual([outcome.run_id for outcome in self.history.outcomes("main")[:2]], [source, source])
        self.assertEqual(self._run("after-boundary").result, "reply")

    async def test_child_reply_is_not_enqueued_for_unsafe_parent_but_recovers_after_lock(self):
        for sqlite_code in (SQLITE_CORRUPT, SQLITE_BUSY):
            with self.subTest(sqlite_code=sqlite_code), closing(History(":memory:")) as history:
                assigned, release = asyncio.Event(), asyncio.Event()
                schedulers = SessionSchedulers()
                scheduler = schedulers.get_or_create("main")
                unsafe = sqlite_code == SQLITE_CORRUPT
                blocked_input = Message("user", "blocked input")
                error = OperationalError("simulated input failure")
                error.sqlite_errorcode = sqlite_code
                append = history._append

                def fail_input(session_id, messages, **kwargs):
                    rows = tuple(messages)
                    ids = append(session_id, rows, **kwargs)
                    if session_id == "main" and rows == (blocked_input,):
                        raise error
                    return ids

                async def answer(messages, tools):
                    if messages[-1].content == "delegate":
                        yield ToolReady(_call("start", "child work"))
                    elif messages[-1].content == "child work":
                        assigned.set()
                        await release.wait()
                        yield TextDelta("child finished")
                    else:
                        yield TextDelta("ack")

                model = _Model(answer)
                async with running_agent(self.settings, history, model, _Channel(), session_id="main",
                                         retry_sqlite_locks=True, schedulers=schedulers,
                                         roles=demo_roles()) as (main, tree):
                    scheduler.session_input((MessageDelivery(Message("user", "delegate"), "trigger"),))
                    await _wait(assigned.wait())
                    await _wait(main.wait_idle())
                    [tool] = all_tool_calls(row.message for row in history.read_after("main"))
                    delegation = stored_delegation(history, "main", tool.id)
                    with (patch.object(history, "_append", side_effect=fail_input),
                          patch("agent_companion.runtime.sqlite_access._LOCK_RETRIES", 1),
                          self.assertLogs("agent_companion.runtime.engine", "ERROR")):
                        scheduler.session_input((MessageDelivery(blocked_input, "trigger"),))
                        await _wait(main.wait_idle())
                    self.assertEqual(main.last_failure, "input_write_failed")
                    self.assertEqual(main.input_retry_forbidden, unsafe)

                    scheduler.session_input(input_active=True)
                    diagnostics = (self.assertLogs("agent_companion.runtime.subtasks", "WARNING")
                                   if unsafe else nullcontext())
                    with (patch.object(scheduler, "submit_internal", wraps=scheduler.submit_internal) as submit,
                          diagnostics as logs):
                        release.set()
                        await _wait(tree.wait_idle())
                    record = history.agent_run(delegation.run_id)
                    self.assertEqual((record.status, record.result, record.error), ("completed", "child finished", None))
                    self.assertEqual(_result_inputs([row.message for row in history.read_after("main")]), [])
                    if unsafe:
                        submit.assert_not_called()
                        self.assertIn("reason=parent_input_retry_forbidden", logs.output[0])
                    else:
                        submit.assert_called_once()

                    scheduler.session_input(input_active=False)
                    await _wait(tree.wait_idle())
                    replies = _result_inputs([row.message for row in history.read_after("main")])
                    if unsafe:
                        self.assertEqual(replies, [])
                    else:
                        self.assertEqual([reply["result"] for reply in replies], ["child finished"])

                if unsafe:
                    async with running_agent(self.settings, history, model, _Channel(),
                                             session_id="main", schedulers=schedulers, roles=demo_roles()) as (_, tree):
                        await _wait(tree.wait_idle())
                    self.assertEqual(_result_inputs([row.message for row in history.read_after("main")]), [])

    async def test_unbound_parent_keeps_persisted_terminal_without_replaying(self):
        assigned, release = asyncio.Event(), asyncio.Event()
        schedulers = SessionSchedulers()

        async def answer(messages, tools):
            last = messages[-1]
            if last.content == "parent input":
                yield ToolReady(_call("start", "child work"))
            elif last.content == "child work":
                assigned.set()
                await release.wait()
                yield TextDelta("child finished while parent was unavailable")
            else:
                yield TextDelta("ack")

        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel,
                                 session_id="main", schedulers=schedulers, roles=demo_roles()) as (main, tree):
            main.submit_messages((MessageDelivery(Message("user", "parent input"), "trigger"),))
            await _wait(assigned.wait())
            await _wait(main.wait_idle())
            schedulers.get_or_create("main").input_revoker()()
            release.set()
            await _wait(tree.wait_idle())
            record = self._run("start")
            self.assertEqual((record.status, record.result), ("completed", "child finished while parent was unavailable"))
            self.assertEqual(_result_inputs(self._messages("main")), [])
        async with running_agent(self.settings, self.history, model, self.channel,
                                 session_id="main", schedulers=schedulers, roles=demo_roles()) as (_, tree):
            await _wait(tree.wait_idle())
        self.assertEqual(_result_inputs(self._messages("main")), [])
        self.assertEqual(len(model.requests), 3)

    async def test_handoff_after_committed_acceptance_is_interrupted_without_model_work(self):
        async def answer(messages, tools):
            if messages[-1].content == "delegate":
                yield ToolReady(_call("child", "must not start"))
            else:
                yield TextDelta("parent response")

        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel, session_id="main",
                                 roles=demo_roles()) as (main, tree):
            with (patch.object(tree, "_bind", side_effect=RuntimeError("synthetic closed entry")),
                  self.assertLogs("agent_companion.runtime.subtasks", level="WARNING") as logs):
                main.submit_messages((MessageDelivery(Message("user", "delegate"), "trigger"),))
                await _wait(tree.wait_idle())
            self.assertEqual(self._acceptances()["child"], {"error": "delegation_handoff_failed"})
            record, = self.history._db.execute("SELECT status, error, run_id FROM delegations")
            self.assertEqual(tuple(record), ("interrupted", "delegation_handoff_failed", None))
        self.assertIn("delegation handoff failed", logs.output[0])
        self.assertNotIn("synthetic closed entry", logs.output[0])
        self.assertEqual(len(model.requests), 2)

    async def test_route_can_reopen_after_startup_or_cleanup_failure(self):
        schedulers = SessionSchedulers()

        async def answer(messages, tools):
            yield TextDelta("new route reply")

        model = _Model(answer)
        for mode in ("startup", "shutdown"):
            with self.subTest(mode=mode):
                interrupt = self.history.interrupt_agent_tree

                def fail_once(agent_id, *, reason):
                    if reason == {"startup": "restart", "shutdown": "shutdown"}[mode]:
                        raise OSError("synthetic fixture")
                    interrupt(agent_id, reason=reason)

                with (patch.object(self.history, "interrupt_agent_tree", side_effect=fail_once),
                      self.assertRaises(OSError)):
                    async with running_agent(self.settings, self.history, model, self.channel,
                                             session_id="main", schedulers=schedulers, roles=demo_roles()):
                        if mode == "startup":
                            self.fail("startup failure should not open a route")
                async with running_agent(self.settings, self.history, model, self.channel,
                                         session_id="main", schedulers=schedulers, roles=demo_roles()) as (main, tree):
                    main.submit_messages((MessageDelivery(Message("user", "restart cleanly"), "trigger"),))
                    await _wait(tree.wait_idle())
        self.assertEqual([content for _, _, content in self.channel.replies], ["new route reply", "new route reply"])
