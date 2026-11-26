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
from agent_companion.protocol import (AGENT_REPLY_PREFIX, AgentReply, DeliveryStatus, InternalInput, Message, ModelError,
                                      PersistedInput, Receipt, TextDelta, ToolCall, ToolReady)
from agent_companion.runtime.subtasks import AgentRole, DELEGATE_TOOL_NAME, demo_roles
from agent_companion.runtime.timers import CANCEL_TIMER_TOOL_NAME, SET_TIMER_TOOL_NAME, Timers
from agent_companion.runtime.tools import ToolContext
from agent_companion.session_scheduler import SessionSchedulers
from agent_companion.storage.sqlite import History


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
    return [json.loads(message.content.removeprefix(AGENT_REPLY_PREFIX)) for message in messages
            if message.role == "user" and isinstance(message.content, str) and message.content.startswith(AGENT_REPLY_PREFIX)]


class _Model:
    def __init__(self, answer, *, tool_names=False):
        self.answer = answer
        self.requests = []
        self.tool_names = tool_names

    async def stream(self, messages, tools, *, thinking):
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
                history.append_agent_inputs(main_agent.agent_id, "parent", [Message("user", "fixture")], start=True)
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
        return {message.tool_call_id: json.loads(message.content)
                for message in self._messages(session) if message.role == "tool"}

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
        self.history.append_agent_inputs(agent_id, run_id, [Message("user", "fixture")], start=True)
        call = _call(call_id, "fixture task")
        context = ToolContext("main", run_id, call_id, has_user_input=True, agent_id=agent_id)
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
                        self.assertIsNotNone(result.reply_enqueued_at)
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
                    self.assertEqual((result.status, result.error, result.reply_enqueued_at), ("interrupted", "shutdown", None))
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
            elif last.role == "tool":
                yield TextDelta({"start-a": "main acknowledged", "start-b": "A pending B",
                                 "start-c": "A pending C", "start-d": "main pending D"}[last.tool_call_id])
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
            main.submit_batch((Message("user", "begin"),))
            await _wait(started_b.wait())
            await _eventually(lambda: "main received A pending B" in [text for _, _, text in self.channel.replies])
            a = self._acceptances()["start-a"]
            a_session = self._session(a)
            a_first = self._run("start-a")
            self.assertEqual(a_first.result, "A pending B")
            self.assertEqual(set(a), {"status", "agent_id"})
            main.submit_batch((Message("user", "main followup"),))
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
        self.assertTrue(all(run.reply_consumed_run_id is not None for run in a_runs))
        self.assertEqual([self.history.run_delegation_ids(run.run_id) for run in a_runs],
                         [(self._delegation("start-a").delegation_id,), (), ()])
        self.assertEqual([notice["result"] for notice in _result_inputs(self._messages(a_session))], ["B done", "C done"])
        self.assertEqual([notice["result"] for notice in main_notices if notice["agent_id"] != a["agent_id"]], ["D done"])

    async def test_safe_boundary_merges_queued_delegations_and_subresult_once(self):
        a_waiting, release_a, b_waiting, release_b, b_enqueued = (asyncio.Event() for _ in range(5))
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

        mark = self.history.mark_agent_reply_enqueued

        def mark_and_signal(run_id):
            mark(run_id)
            if self.history.agent_run(run_id).result == "B ready":
                b_enqueued.set()

        schedulers = SessionSchedulers()
        model = _Model(answer)
        with patch.object(self.history, "mark_agent_reply_enqueued", side_effect=mark_and_signal):
            async with running_agent(self.settings, self.history, model, self.channel, session_id="main",
                                     roles=roles, schedulers=schedulers) as (main, tree):
                main.submit_batch((Message("user", "begin"),))
                await _wait(a_waiting.wait())
                await _wait(b_waiting.wait())
                await _wait(main.wait_idle())
                main.submit_batch((Message("user", "enqueue more"),))
                await _wait(main.wait_idle())
                call_ids = ("first", "second", "third")
                a_session = self._session(self._acceptances()[call_ids[0]])
                self.assertEqual([self._delegation(call_id).status for call_id in call_ids],
                                 ["consumed", "queued", "queued"])
                self.assertEqual([row.content for row in self._messages(a_session)], ["A first"])
                release_b.set()
                await _wait(b_enqueued.wait())
                source = self.history._db.execute("SELECT run_id, agent_id FROM agent_runs WHERE result = 'B ready'").fetchone()
                duplicate = AgentReply(source["agent_id"], source["run_id"], result="B ready")
                scheduler = schedulers.get_or_create(a_session)
                self.assertTrue(scheduler.submit_internal(InternalInput(duplicate)))
                self.assertEqual([row.content for row in self._messages(a_session)], ["A first"])
                release_a.set()
                await _wait(tree.wait_idle())
                self.assertTrue(scheduler.submit_internal(InternalInput(duplicate)))
                await _wait(tree.wait_idle())

        replies = _result_inputs(self._messages("main"))
        reply, = replies
        delegations = [self._delegation(call_id) for call_id in call_ids]
        merged_run_id, = {item.run_id for item in delegations}
        self.assertEqual(reply, {"agent_id": delegations[0].agent_id, "status": "completed", "result": "A merged final"})
        self.assertEqual(self.history.run_delegation_ids(merged_run_id), tuple(item.delegation_id for item in delegations))
        self.assertEqual(self.history.agent_run(source["run_id"]).reply_consumed_run_id, merged_run_id)
        a_messages = self._messages(a_session)
        self.assertEqual(len(_result_inputs(a_messages)), 1)
        self.assertEqual([row.content for row in a_messages if row.role == "user"][:3], ["A first", "A second", "A third"])
        self.assertEqual([row.content for row in a_messages if row.role == "assistant"],
                         ["intermediate tool prose", "A merged final"])
        self.assertEqual({row.run_id for row in self.history.outcomes(a_session)}, {merged_run_id})

    async def test_busy_parent_does_not_admit_result_until_current_delivery_settles(self):
        release_child, child_posted = asyncio.Event(), asyncio.Event()

        async def answer(messages, tools):
            last = messages[-1]
            if last.content == "delegate and speak":
                yield ToolReady(_call("child", "work"))
            elif last.content == "work":
                await release_child.wait()
                yield TextDelta("child ready")
            elif last.role == "tool":
                yield TextDelta("hold-delivery")
            else:
                yield TextDelta("parent got result")

        mark = self.history.mark_agent_reply_enqueued

        def signal(run_id):
            mark(run_id)
            child_posted.set()

        with patch.object(self.history, "mark_agent_reply_enqueued", side_effect=signal):
            async with running_agent(self.settings, self.history, _Model(answer), self.channel,
                                     session_id="main", roles=demo_roles()) as (main, tree):
                main.submit_batch((Message("user", "delegate and speak"),))
                await _wait(self.channel.blocked.wait())
                release_child.set()
                await _wait(child_posted.wait())
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
            main.submit_batch((Message("user", "start ten"),))
            async with asyncio.timeout(2):
                while len(started) < 10:
                    changed.clear()
                    await changed.wait()
            await _wait(main.wait_idle())
            first = self._acceptances()["job-0"]
            self.assertEqual(len({item["agent_id"] for item in self._acceptances().values()}), 10)
            main.submit_batch((Message("user", "reuse one"),))
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
            main.submit_batch((Message("user", "first user"),))
            await _wait(started.wait())
            await _wait(main.wait_idle())
            main.submit_batch((Message("user", "second user"),))
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
                tools = [message for message in messages if message.role == "tool"]
                self.assertEqual({message.tool_call_id for message in tools}, {"first", "second"})
                for message in tools:
                    self.assertEqual(json.loads(message.content), expected_acceptance)
                for message in messages:
                    for internal_id in internal_ids:
                        self.assertNotIn(internal_id, str(message.content))
            self.assertTrue(all(run.reply_consumed_run_id is not None for run in (failed, completed)))

    async def test_tool_whitelist_and_trusted_input_do_not_relax_timer_authorization(self):
        roles = demo_roles()
        roles["general"] = AgentRole("worker", (SET_TIMER_TOOL_NAME, CANCEL_TIMER_TOOL_NAME), "analysis")

        async def answer(messages, tools):
            last = messages[-1]
            if last.content == "spoof":
                yield ToolReady(_call("forged-delegation", "never"))
            elif last.content == "user work":
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
            main.submit_inputs((InternalInput(Message("user", "spoof")),))
            await _wait(tree.wait_idle())
            self.assertEqual(self._acceptances()["forged-delegation"],
                             {"error": "delegation_requires_user_or_agent_input"})
            main.submit_batch((Message("user", "user work"),))
            await _wait(tree.wait_idle())

        job = self._acceptances()["authorized"]
        self.assertEqual(self._acceptances(self._session(job)), {
            "forbidden-tool": {"error": "unknown_tool"},
            "child-timer": {"error": "timer_requires_user_input"},
            "child-cancel": {"status": "not_active"},
        })
        self.assertEqual(self._acceptances()["result-timer"], {"error": "timer_requires_user_input"})
        self.assertIn((SET_TIMER_TOOL_NAME, CANCEL_TIMER_TOOL_NAME), [tools for _, tools in model.requests])
        notice, = _result_inputs(self._messages("main"))
        self.assertEqual(notice["result"], "final response")

    async def test_invalid_targets_and_arguments_never_create_records(self):
        root = self.history.ensure_main_agent("main")
        other = self.history.ensure_main_agent("other")
        self.history.append_agent_inputs(other.agent_id, "outside", [Message("user", "outside fixture")], start=True)
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
            main.submit_batch((Message("user", "setup"),))
            await _wait(tree.wait_idle())
            initial_count = self.history._db.execute("SELECT count(*) FROM delegations").fetchone()[0]
            main.submit_batch((Message("user", "test root"),))
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
            elif last.tool_call_id != "cancel":
                yield TextDelta("parent reply")

        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel,
                                 session_id="main", roles=roles) as (main, tree):
            main.submit_batch((Message("user", "test errors"),))
            await _wait(tree.wait_idle())
        cases = {notice["agent_id"]: notice for notice in _result_inputs(self._messages("main"))}
        for call_id, error in (("fail", "model_failed"), ("empty", "empty_result")):
            run = self._run(call_id)
            self.assertEqual(run.error, error)
            self.assertEqual(cases[run.agent_id], {"agent_id": run.agent_id, "status": "failed", "error": error})
        empty = self._acceptances()["empty"]
        self.assertIn("tool text only", [row.content for row in self._messages(self._session(empty))])

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
            elif last.tool_call_id == "start-b":
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
            main.submit_batch((Message("user", "root"),))
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

                def fail_twice(session_id, messages):
                    nonlocal attempts
                    rows = tuple(messages)
                    ids = append(session_id, rows)
                    if rows == (speech,):
                        attempts += 1
                        if attempts <= 2:
                            raise error
                    return ids

                async with running_agent(self.settings, history, model, _Channel(),
                                         session_id="main", roles=demo_roles()) as (main, tree):
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
                        main.submit_batch((Message("user", "delegate once"),))
                        await _wait(child_started.wait())
                        history.record_inbound("main", "test", 0, [("event", Message("user", "head persisted"))])
                        [batch_id] = history.pending_inbound("main", "test")
                        main.submit_inputs((PersistedInput(batch_id), speech))
                        main.submit_batch((Message("user", "queued later"),))
                        allow_tool_round_to_end.set()
                        if timing == "after worker stops":
                            await _wait(main.wait_idle())
                            self.assertEqual(main.last_failure, "input_write_failed")
                            self.assertIsNone(main._worker)
                        else:
                            await _wait(at_boundary.wait())
                            self.assertFalse(main.idle)
                        self.assertEqual(history.pending_inbound("main", "test"), [batch_id])
                        [tool_result] = [r.message for r in history.read_after("main") if r.message.role == "tool"]
                        run_id = stored_delegation(history, "main", tool_result.tool_call_id).run_id
                        allow_child.set()
                        await _eventually(lambda: history.agent_run(run_id).reply_enqueued_at is not None)
                        resume_worker.set()
                        await _wait(tree.wait_idle())

                    self.assertEqual(attempts, 3)
                    self.assertEqual(history.pending_inbound("main", "test"), [])
                    messages = [r.message for r in history.read_after("main")]
                    self.assertEqual([r.content for r in messages if r.role == "user"][:-1],
                                     ["delegate once", "head persisted", "head speech", "queued later"])
                    self.assertEqual([r for r in messages if r.role == "tool"], [tool_result])
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
                main.submit_batch((Message("user", "run tool"),))
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
                        main.submit_batch((Message("user", "run child"),))
                        await _wait(tree.wait_idle())
                        rows = [row.message for row in history.read_after("main")]
                        accepted = next(json.loads(row.content) for row in rows if row.role == "tool")
                        run = history.agent_run(stored_delegation(history, "main", "start").run_id)
                        child = history.agent(accepted["agent_id"])
                        tool_results = [row.message for row in history.read_after(child.session_id)
                                        if row.message.role == "tool"]
                        notice, = _result_inputs(rows)
                        self.assertEqual(notice["agent_id"], run.agent_id)
                        self.assertIsNotNone(run.reply_consumed_run_id)
                        self.assertEqual([row.tool_call_id for row in tool_results], ["child-tool"])
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
                main.submit_batch((Message("user", "run child"),))
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
            other.submit_batch((Message("user", "start"),))
            await _wait(other_tree.wait_idle())
            other_session = self._session(self._acceptances("other")["create"])
            other_scheduler = schedulers.get_or_create(other_session)
            previous_scheduler = None
            for text in ("start", "resume", "resume"):
                async with open_tree("main") as (main, tree):
                    main.submit_batch((Message("user", text),))
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
            self.assertEqual([row.content for row in self._messages(child_session) if row.role == "user"],
                             ["original task", "follow-up task", "follow-up task"])
        self.assertEqual(set(schedulers._sessions), {"main", "other"})

    async def test_terminal_write_failure_has_bounded_retry_and_no_replay(self):
        for count in (1, 4):
            with self.subTest(count=count), closing(History(":memory:")) as history:
                calls = 0
                session = "fault fixture"

                async def answer(messages, tools):
                    last = messages[-1]
                    if last.content == "try child":
                        yield ToolReady(_call("child", "child work"))
                    else:
                        yield TextDelta("child result" if not tools else "root response")

                model = _Model(answer)
                finish = history.finish_agent_run

                def attempt(run_id, *, result, error):
                    nonlocal calls
                    if result == "child result":
                        calls += 1
                        if calls <= count:
                            raise OperationalError("database is locked")
                    finish(run_id, result=result, error=error)

                with (patch.object(history, "finish_agent_run", side_effect=attempt),
                      patch("agent_companion.runtime.sqlite_access._LOCK_RETRIES", 2)):
                    async with running_agent(self.settings, history, model, self.channel, session_id=session,
                                             retry_sqlite_locks=True, roles=demo_roles()) as (main, tree):
                        main.submit_batch((Message("user", "try child"),))
                        if count == 4:
                            with self.assertLogs("agent_companion.runtime.engine", level="ERROR") as logs:
                                await _wait(tree.wait_idle())
                            self.assertIn("agent terminal write failed", logs.output[0])
                        else:
                            await _wait(tree.wait_idle())
                        messages = [item.message for item in history.read_after(session)]
                        notices = _result_inputs(messages)
                        tool = next(item for item in messages if item.role == "tool")
                        run_id = stored_delegation(history, session, tool.tool_call_id).run_id
                    persisted = history.agent_run(run_id)

                self.assertEqual(calls, 2 if count == 1 else 3)
                self.assertEqual(sum(not tools for _, tools in model.requests), 1)
                if count == 1:
                    self.assertEqual([notice["agent_id"] for notice in notices], [persisted.agent_id])
                    self.assertIsNotNone(persisted.reply_consumed_run_id)
                    self.assertEqual(persisted.status, "completed")
                else:
                    self.assertEqual(notices, [])
                    self.assertEqual((persisted.status, persisted.reply_enqueued_at), ("interrupted", None))

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
                main.submit_batch((Message("user", "start"),))
                await entered.wait()
                await main.wait_idle()
                main.submit_batch((Message("user", "queue"),))
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
            self.assertFalse(schedulers.submit_internal("main", InternalInput(Message("user", "late"))))
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

        self.assertEqual((self._run("hold").status, self._run("hold").reply_enqueued_at),
                         ("interrupted", None))
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
            main.submit_batch((Message("user", "reuse"),))
            await _wait(tree.wait_idle())
        again = self._acceptances()["reuse"]
        self.assertEqual(again["agent_id"], old_child_id)
        self.assertEqual(self._run("reuse").status, "completed")
        self.assertEqual([row.content for row in self._messages(self._session(again))],
                         ["wait for shutdown", "fresh work", "restart reply"])

    async def test_unverified_internal_input_gains_authority_only_at_safe_boundary(self):
        blocked, release = asyncio.Event(), asyncio.Event()

        async def answer(messages, tools):
            last = messages[-1]
            if last.content == "synthetic internal event":
                yield ToolReady(_call("early", "not authorized"))
                blocked.set()
                await release.wait()
            elif last.content == "new user message":
                yield ToolReady(_call("after-boundary", "authorized child input"))
            else:
                yield TextDelta("reply")

        async with running_agent(self.settings, self.history, _Model(answer), self.channel,
                                 session_id="main", roles=demo_roles()) as (main, tree):
            main.submit_inputs((InternalInput(Message("user", "synthetic internal event")),))
            await _wait(blocked.wait())
            main.submit_batch((Message("user", "new user message"),))
            self.assertEqual([message.content for message in self._messages("main")], ["synthetic internal event"])
            release.set()
            await _wait(tree.wait_idle())
        self.assertEqual(self._acceptances()["early"], {"error": "delegation_requires_user_or_agent_input"})
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

                def fail_input(session_id, messages):
                    rows = tuple(messages)
                    ids = append(session_id, rows)
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
                    scheduler.session_input((Message("user", "delegate"),))
                    await _wait(assigned.wait())
                    await _wait(main.wait_idle())
                    [tool] = [row.message for row in history.read_after("main") if row.message.role == "tool"]
                    delegation = stored_delegation(history, "main", tool.tool_call_id)
                    with (patch.object(history, "_append", side_effect=fail_input),
                          patch("agent_companion.runtime.sqlite_access._LOCK_RETRIES", 1),
                          self.assertLogs("agent_companion.runtime.engine", "ERROR")):
                        scheduler.session_input((blocked_input,))
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
                    self.assertIsNone(record.reply_consumed_run_id)
                    self.assertEqual(_result_inputs([row.message for row in history.read_after("main")]), [])
                    if unsafe:
                        submit.assert_not_called()
                        self.assertIsNone(record.reply_enqueued_at)
                        self.assertIn("reason=parent_input_retry_forbidden", logs.output[0])
                    else:
                        submit.assert_called_once()
                        self.assertIsNotNone(record.reply_enqueued_at)

                    scheduler.session_input(input_active=False)
                    await _wait(tree.wait_idle())
                    replies = _result_inputs([row.message for row in history.read_after("main")])
                    if unsafe:
                        self.assertEqual(replies, [])
                        self.assertIsNone(history.agent_run(delegation.run_id).reply_enqueued_at)
                    else:
                        self.assertEqual([reply["result"] for reply in replies], ["child finished"])
                        self.assertIsNotNone(history.agent_run(delegation.run_id).reply_consumed_run_id)

                if unsafe:
                    async with running_agent(self.settings, history, model, _Channel(),
                                             session_id="main", schedulers=schedulers, roles=demo_roles()) as (_, tree):
                        await _wait(tree.wait_idle())
                    self.assertEqual(_result_inputs([row.message for row in history.read_after("main")]), [])
                    self.assertIsNone(history.agent_run(delegation.run_id).reply_enqueued_at)

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
            main.submit_batch((Message("user", "parent input"),))
            await _wait(assigned.wait())
            await _wait(main.wait_idle())
            schedulers.get_or_create("main").input_revoker()()
            release.set()
            await _wait(tree.wait_idle())
            record = self._run("start")
            self.assertEqual((record.status, record.reply_enqueued_at), ("completed", None))
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
                main.submit_batch((Message("user", "delegate"),))
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
                    main.submit_batch((Message("user", "restart cleanly"),))
                    await _wait(tree.wait_idle())
        self.assertEqual([content for _, _, content in self.channel.replies], ["new route reply", "new route reply"])
