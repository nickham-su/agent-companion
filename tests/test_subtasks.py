import asyncio
from contextlib import closing, redirect_stdout
import io
import json
from pathlib import Path
from sqlite3 import OperationalError
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpcore
import httpx

from agent_companion.app import running_agent
from agent_companion.config import Settings
from agent_companion.models import _http_stream
from agent_companion.models.deepseek import DeepSeek
from agent_companion.protocol import (DeliveryStatus, InternalInput, Message, ModelError, Receipt, TextDelta,
                                      ToolCall, ToolReady)
from agent_companion.runtime.subtasks import DELEGATE_TOOL_NAME, RESULT_PREFIX
from agent_companion.runtime.timers import CANCEL_TIMER_TOOL_NAME, SET_TIMER_TOOL_NAME
from agent_companion.runtime.tools import ToolContext
from agent_companion.session_scheduler import SessionSchedulers
from agent_companion.storage.sqlite import History


async def _wait(awaitable):
    return await asyncio.wait_for(awaitable, timeout=2)


def _call(call_id, prompt, **extra):
    return ToolCall(call_id, DELEGATE_TOOL_NAME, json.dumps({"prompt": prompt, **extra}))


def _result_inputs(messages):
    return [json.loads(message.content.removeprefix(RESULT_PREFIX)) for message in messages
            if message.role == "user" and isinstance(message.content, str) and message.content.startswith(RESULT_PREFIX)]


class _Model:
    def __init__(self, answer):
        self.answer = answer
        self.requests = []

    async def stream(self, messages, tools, *, thinking):
        self.requests.append((tuple(messages), tuple(tool.name for tool in tools)))
        async for event in self.answer(messages, bool(tools)):
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


class _AcceptUnavailable(History):
    def accept_subtask(self, *args):
        raise OSError("synthetic acceptance error")


class _FinishUnavailable(History):
    def __init__(self, failures):
        super().__init__(":memory:")
        self.failures = failures
        self.attempts = 0

    def finish_subtask(self, run_id, *, result, error):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise OperationalError("synthetic write failure")
        super().finish_subtask(run_id, result=result, error=error)


class DemoIsolationTest(unittest.TestCase):
    def test_cli_demo_config_cannot_write_production_db_or_interrupt_its_jobs(self):
        from agent_companion.__main__ import main

        with TemporaryDirectory(dir=Path.cwd()) as folder:
            database = Path(folder) / "synthetic.sqlite"
            config = Path(folder) / "settings.toml"
            config.write_text('database = "synthetic.sqlite"\n', encoding="utf-8")
            with closing(History(database)) as history:
                history.append("main", (Message("user", "existing synthetic data"),))
                history.accept_subtask("child", "job", "main", "parent", "tool")
            with patch("sys.argv", ["agent-companion", "--config", str(config), "--demo-subtasks"]), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            with closing(History(database)) as history:
                self.assertEqual([row.message.content for row in history.read_after("main")], ["existing synthetic data"])
                self.assertEqual(history.subtask_run("child", "job").status, "queued")


class SubtaskTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = Settings()
        self.history = History(":memory:")
        self.addCleanup(self.history.close)
        self.channel = _Channel()

    def _messages(self, session):
        return [entry.message for entry in self.history.read_after(session)]

    def _acceptance(self):
        return next(json.loads(message.content) for message in self._messages("main") if message.role == "tool")

    def _run(self, acceptance):
        return self.history.subtask_run(acceptance["child_session_id"], acceptance["run_id"])

    async def _delegate(self, subtasks, call_id, prompt, **extra):
        call = _call(call_id, prompt, **extra)
        context = ToolContext("main", "parent-run", call_id, has_user_input=True)
        return json.loads(await subtasks.tools.execute(call, context))

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
                                      session_id="main", schedulers=schedulers)
                runtime, subtasks = await agent.__aenter__()
                stopping = None
                try:
                    job = await self._delegate(subtasks, f"native-{mode}", "fixture task")
                    if mode == "complete":
                        await _wait(subtasks.wait_idle())
                        await _wait(runtime.wait_idle())
                        result = self._run(job)
                        self.assertEqual((result.status, result.result), ("completed", "child result"))
                        self.assertIsNotNone(result.enqueued_at)
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
                    self.assertEqual((result.status, result.error, result.enqueued_at), ("interrupted", "shutdown", None))
                    scheduler.release_agent(scheduler.reserve_agent())
                finally:
                    release.set()
                    if stopping is None:
                        await agent.__aexit__(None, None, None)
                    else:
                        await stopping
                    await model.aclose()

    async def test_locked_admissions_reserve_capacity_and_cancellation_releases_it(self):
        async def answer(messages, main):
            yield TextDelta("完成")

        accept = self.history.accept_subtask
        blocked = True
        attempts = set()
        saturated, replacing = asyncio.Event(), asyncio.Event()

        def accept_job(child, run_id, parent, parent_run, call_id):
            attempts.add(call_id)
            if len(attempts) == 8:
                saturated.set()
            if call_id == "replacement":
                replacing.set()
            if blocked:
                raise OperationalError("database is locked")
            accept(child, run_id, parent, parent_run, call_id)

        with patch.object(self.history, "accept_subtask", side_effect=accept_job):
            async with running_agent(self.settings, self.history, _Model(answer), self.channel,
                                     session_id="main", retry_sqlite_locks=True) as (_, subtasks):
                tasks = [asyncio.create_task(self._delegate(subtasks, f"call-{i}", "job")) for i in range(10)]
                try:
                    await _wait(saturated.wait())
                    self.assertEqual(await _wait(asyncio.gather(*tasks[8:])),
                                     [{"error": "delegation_capacity_reached"}] * 2)
                    tasks[0].cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await tasks[0]
                    tasks.append(asyncio.create_task(self._delegate(subtasks, "replacement", "job")))
                    await _wait(replacing.wait())
                    blocked = False
                    accepted = await _wait(asyncio.gather(*tasks[1:8], tasks[-1]))
                    self.assertEqual(len({r["run_id"] for r in accepted}), 8)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

    async def test_delegation_caps_concurrency_and_reports_each_rejected_tool_id(self):
        started, active = [], 0
        saturated, release = asyncio.Event(), asyncio.Event()
        peak = 0

        async def answer(messages, main):
            nonlocal active, peak
            if not main:
                started.append(messages[-1].content)
                active += 1
                peak = max(peak, active)
                if active == 2:
                    saturated.set()
                try:
                    await release.wait()
                    yield TextDelta("已完成")
                finally:
                    active -= 1
            elif messages[-1].content == "fan-out":
                for i in range(10):
                    yield ToolReady(_call(f"call-{i}", f"job-{i}"))
            else:
                yield TextDelta("已处理")

        async with running_agent(self.settings, self.history, _Model(answer), self.channel,
                                 session_id="main") as (main, subtasks):
            main.submit_batch([Message("user", "fan-out")])
            await _wait(saturated.wait())
            await _wait(main.wait_idle())
            results = {m.tool_call_id: json.loads(m.content) for m in self._messages("main") if m.role == "tool"}
            accepted = [r for r in results.values() if r.get("status") == "accepted"]
            self.assertEqual(len(accepted), 8)
            self.assertEqual([k for k, v in results.items() if v.get("error") == "delegation_capacity_reached"],
                             ["call-8", "call-9"])
            self.assertEqual(sum(self._run(r).status == "running" for r in accepted), 2)
            self.assertEqual(sum(self._run(r).status == "queued" for r in accepted), 6)
            self.assertEqual(len(started), 2)
            release.set()
            await _wait(subtasks.wait_idle())
            await _wait(main.wait_idle())
            self.assertEqual((len(started), peak), (8, 2))
            self.assertEqual(set(started), {f"job-{i}" for i in range(8)})
            fresh = await self._delegate(subtasks, "extra", "job-extra")
            self.assertEqual(fresh["status"], "accepted")
            await _wait(subtasks.wait_idle())

    async def _return_during_parent_state(self, *, busy):
        child_started = asyncio.Event()
        child_release = asyncio.Event()

        async def answer(messages, main):
            last = messages[-1]
            if not main:
                child_started.set()
                await child_release.wait()
                yield TextDelta("child-finished-output")
            elif last.content == "delegate":
                yield ToolReady(_call("delegate-1", "child-only-input"))
            elif last.role == "tool":
                yield TextDelta("accepted-first")
            elif last.content == "second-input":
                yield TextDelta("hold-delivery")
            else:
                yield TextDelta("main-used-result")

        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel,
                                 session_id="main") as (main, subtasks):
            main.submit_batch((Message("user", "delegate"),))
            await _wait(child_started.wait())
            await _wait(main.wait_idle())
            self.assertEqual([row[2] for row in self.channel.replies], ["accepted-first"])
            accepted = self._acceptance()
            self.assertEqual(accepted["status"], "accepted")
            run = self._run(accepted)
            self.assertEqual((run.status, run.tool_call_id), ("running", "delegate-1"))
            self.assertEqual(run.parent_run_id, self.channel.replies[0][1])
            if busy:
                main.submit_batch((Message("user", "second-input"),))
                await _wait(self.channel.blocked.wait())
                self.assertEqual(self.channel.replies[-1][2], "hold-delivery")
            child_release.set()
            await _wait(subtasks.wait_idle())
            run = self._run(accepted)
            self.assertEqual((run.status, run.result), ("completed", "child-finished-output"))
            self.assertLessEqual(run.accepted_at, run.started_at)
            self.assertLessEqual(run.started_at, run.finished_at)
            self.assertLessEqual(run.finished_at, run.enqueued_at)
            if busy:
                self.assertFalse(_result_inputs(self._messages("main")))
                self.assertFalse(any(_result_inputs(messages) for messages, _ in model.requests))
                self.channel.release.set()
            # 不需要再提交用户消息，子任务回投即可触发处理。
            await _wait(main.wait_idle())
            self.assertEqual(_result_inputs(self._messages("main")), [
                {"child_session_id": run.child_session_id, "run_id": run.run_id,
                 "status": "completed", "result": "child-finished-output",
                 "tool_call_id": "delegate-1", "tool_name": DELEGATE_TOOL_NAME},
            ])
            tool_rows = [message for message in self._messages("main") if message.role == "tool"]
            self.assertEqual([(row.tool_call_id, json.loads(row.content)) for row in tool_rows], [(run.tool_call_id, accepted)])
            self.assertEqual(_result_inputs(model.requests[-1][0]), _result_inputs(self._messages("main")))
            self.assertEqual(self._messages(run.child_session_id), [
                Message("user", "child-only-input"), Message("assistant", "child-finished-output"),
            ])
            self.assertNotIn(Message("user", "child-only-input"), self._messages("main"))
            child_outcome, = self.history.outcomes(run.child_session_id)
            self.assertEqual(child_outcome.run_id, run.run_id)
            self.assertIsNone(child_outcome.delivery_status)
            self.assertEqual({row[0] for row in self.channel.replies}, {"main"})
            self.assertEqual(self.channel.replies[-1][2], "main-used-result")
            self.assertNotEqual(self.channel.replies[-1][1], self.channel.replies[0][1])
            main_tools = (DELEGATE_TOOL_NAME, SET_TIMER_TOOL_NAME, CANCEL_TIMER_TOOL_NAME)
            self.assertTrue(all(names == main_tools for _, names in model.requests if names))
            self.assertEqual(sum(not names for _, names in model.requests), 1)

    async def test_busy_main_waits_for_delivery_before_admitting_result(self):
        await self._return_during_parent_state(busy=True)

    async def test_result_wakes_idle_main(self):
        await self._return_during_parent_state(busy=False)

    async def test_busy_mixed_response_admits_result_into_same_run_after_boundary(self):
        child_release = asyncio.Event()

        async def answer(messages, main):
            last = messages[-1]
            if not main:
                await child_release.wait()
                yield TextDelta("child-result")
            elif last.content == "mixed":
                yield TextDelta("hold-delivery")
                yield ToolReady(_call("rejected-mixed", "prompt", child_session_id="nonexistent"))
            else:
                yield TextDelta("continued")

        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel,
                                 session_id="main") as (main, subtasks):
            accepted = await self._delegate(subtasks, "background", "child-task")
            main.submit_batch((Message("user", "mixed"),))
            await _wait(self.channel.blocked.wait())
            child_release.set()
            await _wait(subtasks.wait_idle())
            self.assertIsNotNone(self._run(accepted).enqueued_at)
            self.assertFalse(_result_inputs(self._messages("main")))
            self.channel.release.set()
            await _wait(main.wait_idle())
            followup, _ = model.requests[-1]
            self.assertEqual(followup[-2].tool_call_id, "rejected-mixed")
            self.assertEqual(_result_inputs(followup)[0]["run_id"], accepted["run_id"])
            self.assertEqual([row[2] for row in self.channel.replies], ["hold-delivery", "continued"])
            self.assertEqual(self.channel.replies[0][1], self.channel.replies[1][1])

    async def test_locked_child_ownership_lookup_still_rejects_unknown_child(self):
        async def answer(messages, main):
            yield TextDelta("完成")

        async with running_agent(self.settings, self.history, _Model(answer), self.channel,
                                 session_id="main", retry_sqlite_locks=True) as (main, subtasks):
            first = await self._delegate(subtasks, "first", "job")
            child = first["child_session_id"]
            await _wait(subtasks.wait_idle())
            lookup = self.history.can_resume_subtask
            attempts = dict.fromkeys((child, "unknown"), 0)

            def read(parent, requested):
                attempts[requested] += 1
                if attempts[requested] < 3:
                    raise OperationalError("database table is locked")
                return lookup(parent, requested)

            with patch.object(self.history, "can_resume_subtask", side_effect=read):
                resumed = await _wait(self._delegate(subtasks, "resume", "resume job", child_session_id=child))
                refused = await _wait(self._delegate(subtasks, "unknown", "unknown job", child_session_id="unknown"))
            self.assertEqual(attempts, {child: 3, "unknown": 3})
            self.assertEqual(resumed["child_session_id"], child)
            self.assertNotEqual(resumed["run_id"], first["run_id"])
            self.assertEqual(refused, {"error": "unknown_child_session"})
            await _wait(subtasks.wait_idle())
            await _wait(main.wait_idle())

    async def test_reuse_is_serial_while_other_child_runs_independently(self):
        first_started, first_release = asyncio.Event(), asyncio.Event()
        second_started, other_started = asyncio.Event(), asyncio.Event()

        async def answer(messages, main):
            prompt = messages[-1].content
            if main:
                yield TextDelta("main-result-handled")
            else:
                if prompt == "first":
                    first_started.set()
                    await first_release.wait()
                elif prompt == "second":
                    second_started.set()
                else:
                    other_started.set()
                yield TextDelta(f"answer-{prompt}")

        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel,
                                 session_id="main") as (main, subtasks):
            first = await self._delegate(subtasks, "first-call", "first")
            await _wait(first_started.wait())
            child = first["child_session_id"]
            second = await self._delegate(subtasks, "second-call", "second", child_session_id=child)
            other = await self._delegate(subtasks, "other-call", "other")
            await _wait(other_started.wait())
            self.assertFalse(second_started.is_set())
            self.assertEqual(self._run(second).status, "queued")
            self.assertEqual(second["child_session_id"], child)
            self.assertNotEqual(other["child_session_id"], child)
            self.assertEqual(len({item["run_id"] for item in (first, second, other)}), 3)
            first_release.set()
            await _wait(subtasks.wait_idle())
            await _wait(main.wait_idle())
            self.assertTrue(second_started.is_set())
            child_history = [message.content for message in self._messages(child)]
            self.assertEqual(child_history, ["first", "answer-first", "second", "answer-second"])
            second_request = next(messages for messages, tools in model.requests
                                  if not tools and messages[-1] == Message("user", "second"))
            self.assertIn(Message("assistant", "answer-first"), second_request)
            self.assertEqual([message.content for message in self._messages(other["child_session_id"])],
                             ["other", "answer-other"])
            self.assertEqual({item["run_id"] for item in _result_inputs(self._messages("main"))},
                             {item["run_id"] for item in (first, second, other)})

    async def test_submit_failure_preserves_tool_call_ids_and_keeps_main_running(self):
        async def answer(messages, main):
            self.assertTrue(main)
            if messages[-1].content == "delegate":
                yield ToolReady(_call("unknown", "prompt", child_session_id="nonexistent"))
                yield ToolReady(_call("storage", "prompt"))
            else:
                yield TextDelta("main-continued")

        model = _Model(answer)
        with closing(_AcceptUnavailable(":memory:")) as history:
            async with running_agent(self.settings, history, model, self.channel,
                                     session_id="main") as (main, subtasks):
                main.submit_batch((Message("user", "delegate"),))
                await _wait(main.wait_idle())
                await _wait(subtasks.wait_idle())
                expected = {"unknown": {"error": "unknown_child_session"}, "storage": {"error": "tool_failed"}}
                self.assertEqual({item.tool_call_id: json.loads(item.content) for item in model.requests[-1][0]
                                  if item.role == "tool"}, expected)
                self.assertEqual({entry.message.tool_call_id: json.loads(entry.message.content)
                                  for entry in history.read_after("main") if entry.message.role == "tool"}, expected)
                self.assertFalse(_result_inputs(entry.message for entry in history.read_after("main")))
                self.assertIsNone(main.last_failure)
                self.assertEqual(self.channel.replies[-1][2], "main-continued")
                forbidden = await subtasks.tools.execute(_call("no-recursion", "prompt"),
                                                         ToolContext("child", "x", "no-recursion", has_user_input=False))
                self.assertEqual(json.loads(forbidden), {"error": "not_parent_session"})

    async def test_failed_or_tool_calling_child_returns_failure_without_direct_delivery(self):
        async def answer(messages, main):
            if main:
                yield TextDelta("main-handled-failure")
            else:
                yield TextDelta("unconfirmed-child-prefix")
                if messages[-1].content == "attempt-recursion":
                    yield ToolReady(_call("nested", "must not run"))
                raise ModelError("simulated_disconnect")

        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel,
                                 session_id="main") as (main, subtasks):
            accepted = [await self._delegate(subtasks, prompt, prompt)
                        for prompt in ("fail", "attempt-recursion")]
            await _wait(subtasks.wait_idle())
            await _wait(main.wait_idle())
            child_requests = [messages for messages, names in model.requests if not names]
            self.assertEqual(len(child_requests), 2)
            inputs = _result_inputs(self._messages("main"))
            self.assertEqual({previous["run_id"] for previous in inputs}, {item["run_id"] for item in accepted})
            for item in accepted:
                run = self._run(item)
                self.assertEqual((run.status, run.error, run.result), ("failed", "model_failed", None))
                notice = next(notice for notice in inputs if notice["run_id"] == run.run_id)
                self.assertEqual((notice["tool_call_id"], notice["tool_name"]), (run.tool_call_id, DELEGATE_TOOL_NAME))
                self.assertEqual(len(self._messages(run.child_session_id)), 1)
            self.assertTrue(all(item["error"] == "model_failed" and "result" not in item for item in inputs))
            self.assertTrue(self.channel.replies)
            self.assertEqual({(session, text) for session, _, text in self.channel.replies},
                             {("main", "main-handled-failure")})

    async def _finish_with_write_failures(self, *, failures, expected_attempts, completed):
        async def answer(messages, main):
            yield TextDelta("main-read-result" if main else "child-result")

        model = _Model(answer)
        with closing(_FinishUnavailable(failures)) as history:
            async with running_agent(self.settings, history, model, self.channel,
                                     session_id="main") as (main, subtasks):
                accepted = await self._delegate(subtasks, "finish", "run this once")
                run_id, child = accepted["run_id"], accepted["child_session_id"]
                if completed:
                    await _wait(subtasks.wait_idle())
                else:
                    with self.assertLogs("agent_companion.runtime.subtasks", level="ERROR") as logged:
                        await _wait(subtasks.wait_idle())
                    self.assertEqual(len(logged.output), 1)
                    self.assertIn(run_id, logged.output[0])
                    self.assertIn("attempts=3", logged.output[0])
                    self.assertIn("OperationalError; result not enqueued", logged.output[0])
                    self.assertNotIn("synthetic write failure", logged.output[0])
                await _wait(main.wait_idle())
                self.assertEqual(history.attempts, expected_attempts)
                self.assertEqual(sum(not tools for _, tools in model.requests), 1)
                self.assertEqual(len(history.outcomes(child)), 1)
                run = history.subtask_run(child, run_id)
                inputs = _result_inputs(entry.message for entry in history.read_after("main"))
                if completed:
                    self.assertEqual((run.status, run.result), ("completed", "child-result"))
                    self.assertLessEqual(run.finished_at, run.enqueued_at)
                    self.assertEqual([item["run_id"] for item in inputs], [run_id])
                    self.assertEqual(len(self.channel.replies), 1)
                else:
                    self.assertEqual((run.status, run.result, run.enqueued_at), ("running", None, None))
                    self.assertFalse(inputs)
                    self.assertFalse(self.channel.replies)
            if not completed:
                self.assertEqual(history.subtask_run(child, run_id).status, "interrupted")

    async def test_terminal_write_failure_is_retried_without_rerunning_or_reenqueuing(self):
        await self._finish_with_write_failures(failures=1, expected_attempts=2, completed=True)

    async def test_persistent_terminal_write_failure_is_diagnosed_without_enqueuing(self):
        await self._finish_with_write_failures(failures=10, expected_attempts=3, completed=False)

    async def test_shutdown_stops_delegation_and_return_before_yielding(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def answer(messages, main):
            self.assertFalse(main)
            started.set()
            await release.wait()
            yield TextDelta("finishes-as-app-exits")

        async with running_agent(self.settings, self.history, _Model(answer), self.channel,
                                 session_id="main") as (main, subtasks):
            accepted = await self._delegate(subtasks, "first", "pending")
            await _wait(started.wait())

            async def compete_with_shutdown():
                late = await self._delegate(subtasks, "late", "must not be accepted")
                with self.assertRaisesRegex(RuntimeError, "runtime is closed"):
                    main.submit_inputs((InternalInput(Message("user", "late internal result")),))
                return late

            # 两条路径比异步 close 更早就绪；退出必须先同步封住入口。
            release.set()
            late_tool = asyncio.create_task(compete_with_shutdown())
        self.assertEqual(await _wait(late_tool), {"error": "delegation_closed"})
        run = self._run(accepted)
        self.assertEqual(run.status, "completed")
        self.assertIsNone(run.enqueued_at)
        self.assertFalse(self._messages("main"))
        self.assertFalse(self.channel.replies)

    async def test_registered_parent_releases_typed_notice_with_real_user_only(self):
        schedulers = SessionSchedulers()

        async def answer(messages, main):
            last = messages[-1]
            if not main:
                yield TextDelta("completed:" + last.content)
            elif last.content == "real-user":
                self.assertEqual(_result_inputs(messages)[-1]["tool_call_id"], "original-call")
                yield ToolReady(_call("allowed", "second-child"))
            else:
                yield TextDelta("done")

        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel,
                                 session_id="other", schedulers=schedulers):
            async with running_agent(self.settings, self.history, model, self.channel,
                                     session_id="main", schedulers=schedulers) as (main, subtasks):
                parent = schedulers.get_or_create("main")
                parent.session_input(input_active=True)
                accepted = await self._delegate(subtasks, "original-call", "first-child")
                await _wait(subtasks.wait_idle())
                self.assertEqual(self._run(accepted).status, "completed")
                self.assertFalse(self._messages("main"))
                self.assertFalse(self._messages("other"))
                parent.session_input((Message("user", "real-user"),), input_active=False)
                await _wait(main.wait_idle())
                await _wait(subtasks.wait_idle())
                await _wait(main.wait_idle())
                self.assertFalse(self._messages("other"))
                rows = self._messages("main")
                self.assertEqual([notice["tool_call_id"] for notice in _result_inputs(rows)], ["original-call", "allowed"])
                self.assertEqual(rows[1], Message("user", "real-user"))
                accepted_row, = (row for row in rows if row.role == "tool")
                self.assertEqual((accepted_row.tool_call_id, json.loads(accepted_row.content)["status"]),
                                 ("allowed", "accepted"))

    async def test_parent_without_active_binding_keeps_terminal_without_replaying(self):
        schedulers = SessionSchedulers()
        child_started, child_release = asyncio.Event(), asyncio.Event()

        async def answer(messages, main):
            if not main:
                child_started.set()
                await child_release.wait()
                yield TextDelta("child completed offline")
            else:
                self.fail("an offline completion must not reach a model or send channel")

        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel,
                                 session_id="main", schedulers=schedulers) as (main, subtasks):
            accepted = await self._delegate(subtasks, "offline", "child-task")
            await _wait(child_started.wait())
            parent = schedulers.get_or_create("main")
            revoke = parent.input_revoker()
            revoke()
            child_release.set()
            await _wait(subtasks.wait_idle())
            await _wait(main.wait_idle())
            run = self._run(accepted)
            self.assertEqual((run.status, run.result, run.enqueued_at), ("completed", "child completed offline", None))
        async with running_agent(self.settings, self.history, model, self.channel,
                                 session_id="main", schedulers=schedulers) as (main, subtasks):
            self.assertIs(schedulers.get_or_create("main"), parent)
            await _wait(main.wait_idle())
            await _wait(subtasks.wait_idle())
            self.assertEqual(len(model.requests), 1)
            self.assertFalse(self._messages("main"))
            self.assertFalse(self.channel.replies)

    async def test_result_only_run_cannot_delegate_even_after_tool_failure(self):
        async def answer(messages, main):
            last = messages[-1]
            if not main:
                yield TextDelta("finished")
            elif _result_inputs((last,)):
                yield ToolReady(_call("repeat-1", "must not run"))
            elif last.tool_call_id == "repeat-1":
                yield ToolReady(_call("repeat-2", "also must not run"))
            else:
                yield TextDelta("main-used-result")

        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel,
                                 session_id="main") as (main, subtasks):
            await self._delegate(subtasks, "initial", "original child")
            await _wait(subtasks.wait_idle())
            await _wait(main.wait_idle())
            expected = {call_id: {"error": "delegation_requires_user_input"} for call_id in ("repeat-1", "repeat-2")}
            self.assertEqual({message.tool_call_id: json.loads(message.content) for message in self._messages("main")
                              if message.role == "tool"}, expected)
            self.assertEqual({message.tool_call_id: json.loads(message.content) for message in model.requests[-1][0]
                              if message.role == "tool"}, expected)
            self.assertEqual(sum(not tools for _, tools in model.requests), 1)
            self.assertEqual(self.channel.replies[-1][2], "main-used-result")
            self.assertIsNone(main.last_failure)

    async def test_merged_user_input_allows_delegation_only_after_safe_boundary(self):
        result_started, release = asyncio.Event(), asyncio.Event()

        async def answer(messages, main):
            last = messages[-1]
            if not main:
                yield TextDelta("completed-" + last.content)
            elif last.content == "new-user-input":
                yield ToolReady(_call("allowed", "with-user-approval"))
            elif any(result.get("result") == "completed-initial" for result in _result_inputs((last,))):
                result_started.set()
                await release.wait()
                yield ToolReady(_call("blocked", "must not run"))
            else:
                yield TextDelta("main-finished")

        model = _Model(answer)
        async with running_agent(self.settings, self.history, model, self.channel,
                                 session_id="main") as (main, subtasks):
            await self._delegate(subtasks, "initial", "initial")
            await _wait(result_started.wait())
            main.submit_batch((Message("user", "new-user-input"),))
            release.set()
            await _wait(main.wait_idle())
            await _wait(subtasks.wait_idle())
            await _wait(main.wait_idle())
            tools = {message.tool_call_id: json.loads(message.content) for message in self._messages("main")
                     if message.role == "tool"}
            self.assertEqual(tools["blocked"], {"error": "delegation_requires_user_input"})
            self.assertEqual(tools["allowed"]["status"], "accepted")
            run = self._run(tools["allowed"])
            self.assertEqual(run.result, "completed-with-user-approval")
            outcomes = self.history.outcomes("main")
            self.assertEqual({outcome.run_id for outcome in outcomes[:2]}, {run.parent_run_id})
            self.assertEqual(sum(not names for _, names in model.requests), 2)

    async def test_duplicate_open_does_not_interrupt_the_running_child(self):
        schedulers = SessionSchedulers()
        child_started, release = asyncio.Event(), asyncio.Event()

        async def answer(messages, main):
            if not main:
                child_started.set()
                await release.wait()
            yield TextDelta("done")

        model = _Model(answer)
        with patch.object(self.history, "interrupt_subtasks", wraps=self.history.interrupt_subtasks) as interrupt:
            async with running_agent(self.settings, self.history, model, self.channel,
                                     session_id="main", schedulers=schedulers) as (main, subtasks):
                accepted = await self._delegate(subtasks, "call", "child")
                await _wait(child_started.wait())
                with self.assertRaisesRegex(RuntimeError, "already bound or owned"):
                    async with running_agent(self.settings, self.history, model, self.channel,
                                             session_id="main", schedulers=schedulers):
                        self.fail("a duplicate agent entered the active session")
                interrupt.assert_called_once_with("main", reason="restart")
                self.assertEqual(self._run(accepted).status, "running")
                release.set()
                await _wait(subtasks.wait_idle())
                await _wait(main.wait_idle())
                self.assertEqual(self._run(accepted).status, "completed")

    async def test_shutdown_retains_ownership_during_slow_child_close_and_repeated_cancellation(self):
        schedulers = SessionSchedulers()
        child_started, child_closing, release_close = asyncio.Event(), asyncio.Event(), asyncio.Event()
        previous = []

        async def answer(messages, main):
            if not main and messages[-1].content == "old-child":
                child_started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    child_closing.set()
                    await release_close.wait()
                    raise
            yield TextDelta("done")

        model = _Model(answer)

        async def run_old():
            async with running_agent(self.settings, self.history, model, self.channel,
                                     session_id="main", schedulers=schedulers) as (_, subtasks):
                previous.append(await self._delegate(subtasks, "old-call", "old-child"))
                await asyncio.Future()

        old_task = asyncio.create_task(run_old())
        try:
            await _wait(child_started.wait())
            parent = schedulers.get_or_create("main")
            revoke_old = parent.input_revoker()
            old_task.cancel()
            await _wait(child_closing.wait())
            old_task.cancel()  # 不能中断正在执行的 shutdown，也不能提前放开所有权。
            await asyncio.sleep(0)
            self.assertFalse(old_task.done())
            self.assertEqual(self._run(previous[0]).status, "running")
            with self.assertRaisesRegex(RuntimeError, "not bound"):
                parent.session_input((Message("user", "late user"),))
            self.assertFalse(schedulers.submit_internal("main", InternalInput(Message("user", "late result"))))
            with self.assertRaisesRegex(RuntimeError, "already bound or owned"):
                async with running_agent(self.settings, self.history, model, self.channel,
                                         session_id="main", schedulers=schedulers):
                    self.fail("a successor entered before old shutdown finished")
            release_close.set()
            with self.assertRaises(asyncio.CancelledError):
                await _wait(old_task)
            run = self._run(previous[0])
            self.assertEqual((run.status, run.error, run.enqueued_at), ("interrupted", "shutdown", None))

            async with running_agent(self.settings, self.history, model, self.channel,
                                     session_id="main", schedulers=schedulers) as (main, subtasks):
                revoke_old()
                accepted = await self._delegate(subtasks, "new-call", "new-child")
                await _wait(subtasks.wait_idle())
                await _wait(main.wait_idle())
                self.assertEqual(self._run(accepted).status, "completed")
                self.assertEqual([item["tool_call_id"] for item in _result_inputs(self._messages("main"))], ["new-call"])
        finally:
            release_close.set()
            if not old_task.done():
                old_task.cancel()
            await asyncio.gather(old_task, return_exceptions=True)

    async def test_startup_or_shutdown_failure_releases_session_ownership(self):
        schedulers = SessionSchedulers()
        model = _Model(None)
        interrupt = self.history.interrupt_subtasks
        for failure in ("restart", "shutdown"):
            with self.subTest(failure=failure):
                def fail_phase(session_id, *, reason):
                    if reason == failure:
                        raise OperationalError("injected lifecycle failure")
                    return interrupt(session_id, reason=reason)

                with patch.object(self.history, "interrupt_subtasks", side_effect=fail_phase):
                    with self.assertRaisesRegex(OperationalError, "injected lifecycle failure"):
                        async with running_agent(self.settings, self.history, model, self.channel,
                                                 session_id="main", schedulers=schedulers):
                            self.assertEqual(failure, "shutdown")
                async with running_agent(self.settings, self.history, model, self.channel,
                                         session_id="main", schedulers=schedulers):
                    self.assertIsNotNone(schedulers.get_or_create("main").input_revoker())

    async def test_close_persists_interruption_and_restart_does_not_resume_or_replay(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocked(messages, main):
            self.assertFalse(main)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            yield TextDelta("must-not-return")

        with TemporaryDirectory(dir=Path.cwd()) as temp:
            path = Path(temp) / "history.sqlite3"
            with closing(History(path)) as history:
                async with running_agent(self.settings, history, _Model(blocked), self.channel,
                                         session_id="main") as (_, subtasks):
                    first = await self._delegate(subtasks, "first", "active")
                    await _wait(started.wait())
                    child = first["child_session_id"]
                    queued = await self._delegate(subtasks, "second", "queued", child_session_id=child)
                self.assertTrue(cancelled.is_set())
                for item in (first, queued):
                    run = history.subtask_run(child, item["run_id"])
                    self.assertEqual((run.status, run.error, run.enqueued_at), ("interrupted", "shutdown", None))
                history.accept_subtask(child, "crashed", "main", "old-parent-run", "old-call")
                history.start_subtask("crashed")
                history.accept_subtask("foreign-child", "foreign-run", "foreign-parent", "x", "foreign-call")

            async def resumed(messages, main):
                yield TextDelta("main-handled" if main else "fresh-child-result")

            model = _Model(resumed)
            with closing(History(path)) as history:
                async with running_agent(self.settings, history, model, self.channel,
                                         session_id="main") as (main, subtasks):
                    await _wait(subtasks.wait_idle())
                    await _wait(main.wait_idle())
                    self.assertFalse(model.requests)
                    self.assertFalse(self.channel.replies)
                    crashed = history.subtask_run(child, "crashed")
                    self.assertEqual((crashed.status, crashed.error, crashed.enqueued_at),
                                     ("interrupted", "restart", None))
                    self.assertEqual(history.subtask_run("foreign-child", "foreign-run").status, "queued")
                    forbidden = await self._delegate(subtasks, "foreign", "denied", child_session_id="foreign-child")
                    self.assertEqual(forbidden, {"error": "unknown_child_session"})
                    fresh = await self._delegate(subtasks, "fresh", "resume explicitly", child_session_id=child)
                    self.assertEqual(fresh["child_session_id"], child)
                    self.assertNotEqual(fresh["run_id"], first["run_id"])
                    await _wait(subtasks.wait_idle())
                    await _wait(main.wait_idle())
                    self.assertEqual(history.subtask_run(child, fresh["run_id"]).status, "completed")
                    self.assertEqual([entry.message.content for entry in history.read_after(child)],
                                     ["active", "resume explicitly", "fresh-child-result"])
