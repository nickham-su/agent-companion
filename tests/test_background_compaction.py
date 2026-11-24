import asyncio
from contextlib import closing
from dataclasses import replace
from pathlib import Path
import json
import sqlite3
import tempfile
from time import monotonic
import unittest
from unittest.mock import patch

import httpcore
import httpx

from agent_companion.app import prepare_private_feishu, running_agent
from agent_companion.config import FeishuSettings, RuntimeSettings, Settings
from agent_companion.models import _http_stream
from agent_companion.models.deepseek import DeepSeek
from agent_companion.protocol import (ChannelUser, InputUsage, Message, ProviderReplay, ReplayItem, ReplayScope,
                                      TextDelta, ToolCall, ToolReady)
from agent_companion.runtime.subtasks import demo_roles
from agent_companion.runtime.compaction import Compactions, Prefix
from agent_companion.runtime.context import _SUMMARIZE
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.tools import Tool, ToolContext, Tools
from agent_companion.storage.sqlite import History, ReplyOutcome, Summary
from test_history_context import Sink, SPEC


_SETTINGS = RuntimeSettings(context_budget=240, tail_budget=60)
_SUMMARY = Message("system", "此前会话摘要：\nbackground result")
_OLD = Message("user", "old history " + "a" * 500)


def _report_high_usage(history, session_id):
    history.finalize_reply(ReplyOutcome(f"prior-{session_id}", session_id, "seed", "complete", input_tokens=240), [])


async def _wait(event):
    await asyncio.wait_for(event.wait(), 2)


async def _idle(runtime):
    await asyncio.wait_for(runtime.wait_idle(), 2)


class _GatedModel:
    def __init__(self, reply=None, *, result_after_cancel=False):
        self.reply = reply
        self.result_after_cancel = result_after_cancel
        self.requests = []
        self.summary_requests = []
        self.generating = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def stream(self, messages, tools, *, thinking):
        if messages[-1] == _SUMMARIZE:
            self.summary_requests.append(tuple(messages))
            self.generating.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                if not self.result_after_cancel:
                    raise
            yield TextDelta("background result")
            return
        self.requests.append(tuple(messages))
        if self.reply is None:
            yield TextDelta("normal reply")
        else:
            async for event in self.reply(self.requests[-1]):
                yield event


class _NativeSummaryTransport:
    """真实 HTTPX/httpcore，底层连接和阻塞的关闭均在内存里。"""

    def __init__(self, *, complete=False):
        self.waiting, self.closing, self.release, self.closed, self.interrupted = (asyncio.Event() for _ in range(5))
        body = b'data: {"choices":[{"index":0,"delta":{"content":"background result"}}]}\n\n'
        if complete:
            body += b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        length = len(body) if complete else len(body) + 100
        packet = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n"
                  + f"Content-Length: {length}\r\n\r\n".encode() + body)
        fixture = self

        class Wire(httpcore.AsyncNetworkStream):
            sent = False

            async def read(self, max_bytes, timeout=None):
                if not self.sent:
                    self.sent = True
                    return packet
                fixture.waiting.set()
                await asyncio.Future()

            async def write(self, buffer, timeout=None):
                pass

            async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
                return self

            async def aclose(self):
                fixture.closing.set()
                try:
                    await fixture.release.wait()
                except asyncio.CancelledError:
                    fixture.interrupted.set()
                    raise
                fixture.closed.set()

        class Backend(httpcore.AsyncNetworkBackend):
            async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
                return Wire()

        pool = httpcore.AsyncConnectionPool
        with patch.object(httpcore, "AsyncConnectionPool", side_effect=lambda **kw: pool(network_backend=Backend(), **kw)):
            transport = httpx.AsyncHTTPTransport(trust_env=False)
        self.client = httpx.AsyncClient(transport=transport, trust_env=False)


class BackgroundCompactionTest(unittest.IsolatedAsyncioTestCase):
    async def _begin(self, runtime, model, text="first"):
        runtime.submit_batch([Message("user", text)])
        await _wait(model.generating)
        task = runtime._compactions._generating
        self.assertIsNotNone(task)
        return task

    async def _finish(self, model, task):
        model.release.set()
        await asyncio.wait_for(asyncio.shield(task), 2)

    async def test_first_and_tool_continuation_do_not_wait_and_apply_only_after_full_batch(self):
        calls = [ToolCall(f"call{i}", SPEC.name, json.dumps({"i": i})) for i in range(3)]
        reached, release = [asyncio.Event() for _ in calls], [asyncio.Event() for _ in calls]
        tools_run = []

        async def tool(args, context):
            i = args["i"]
            tools_run.append(context.call_id)
            reached[i].set()
            await release[i].wait()
            return f"result{i}"

        async def reply(messages):
            index = len(model.requests)
            if index == 1:
                yield ToolReady(calls[0])
            elif index == 2:
                yield ToolReady(calls[1])
                yield ToolReady(calls[2])
            else:
                yield TextDelta("ready to reply")
            yield InputUsage(240)

        model = _GatedModel(reply)
        with closing(History(":memory:")) as history:
            old_id, = history.append("main", [_OLD])
            _report_high_usage(history, "main")
            runtime = Runtime("main", history, model, Sink(), settings=_SETTINGS, tools=Tools([Tool(SPEC, tool)]))
            try:
                task = await self._begin(runtime, model)
                await _wait(reached[0])
                self.assertEqual(model.requests[0], (_OLD, Message("user", "first")))
                runtime.submit_batch([Message("user", "queued during first batch")])
                release[0].set()
                await _wait(reached[1])
                await _wait(reached[2])
                self.assertEqual(len(model.requests), 2)
                self.assertEqual(model.requests[1][-1], Message("user", "queued during first batch"))
                self.assertFalse(task.done())
                release[1].set()
                await self._finish(model, task)
                self.assertEqual(len(model.requests), 2)
                self.assertIsNone(history.summary("main"))
                self.assertEqual(model.summary_requests, [(_OLD, _SUMMARIZE)])
                runtime.submit_batch([Message("user", "queued during second batch")])
                release[2].set()
                await _idle(runtime)
                self.assertIsNone(runtime.last_failure)
                self.assertEqual(history.summary("main"), Summary(old_id, "background result"))
                self.assertEqual(model.requests[2], (
                    _SUMMARY, Message("user", "first"), Message("assistant", "", (calls[0],)),
                    Message("tool", "result0", tool_call_id=calls[0].id),
                    Message("user", "queued during first batch"), Message("assistant", "", tuple(calls[1:])),
                    Message("tool", "result1", tool_call_id=calls[1].id),
                    Message("tool", "result2", tool_call_id=calls[2].id),
                    Message("user", "queued during second batch"),
                ))
                self.assertCountEqual(tools_run, [call.id for call in calls])
                self.assertEqual(len(model.summary_requests), 1)  # 本次提交不使用旧 usage 再生成。
                self.assertEqual(history.read_after("main")[0].message, _OLD)
            finally:
                await runtime.close()

    async def test_child_candidate_survives_idle_worker_and_is_not_used_by_other_sessions(self):
        first_child_text = "child old input " + "x" * 500

        async def reply(messages):
            if messages[-1].content == first_child_text:
                yield InputUsage(240)
            yield TextDelta("finished")

        model = _GatedModel(reply)
        settings = Settings(runtime=_SETTINGS)
        with closing(History(":memory:")) as history:
            async with running_agent(settings, history, model, Sink(), session_id="main",
                                     roles=demo_roles()) as (main, tree):
                agent_id = main.agent.agent_id
                history.append_agent_inputs(agent_id, "test", [Message("user", "fixture parent input")], start=True)

                async def delegate(label, prompt, child=None):
                    args = {"prompt": prompt, **({"agent_id": child} if child else {"agent": "general"})}
                    context = ToolContext("main", "test", label, True, agent_id=agent_id)
                    accepted = json.loads(await tree._delegate(args, context))
                    self.assertEqual(accepted["status"], "accepted")
                    return accepted["agent_id"]

                child_agent = await delegate("start", first_child_text)
                child = history.agent(child_agent).session_id
                await _idle(tree)
                await _idle(main)
                await delegate("generate", "child while generating", child_agent)
                await _wait(model.generating)
                task = main._compactions._generating
                self.assertIsNotNone(task)
                await _idle(tree)
                await _idle(main)
                self.assertTrue(tree._bindings[child_agent].runtime.idle)
                self.assertIsNone(history.summary(child))
                self.assertFalse(task.done())

                history.append("main", [_OLD])
                _report_high_usage(history, "main")
                main.submit_batch([Message("user", "main while child is generating")])
                await _idle(main)
                self.assertIn(_OLD, model.requests[-1])
                self.assertEqual(len(model.summary_requests), 1)
                await self._finish(model, task)
                self.assertIsNone(history.summary(child))

                other_agent = await delegate("other", "other child")
                other = history.agent(other_agent).session_id
                await _idle(tree)
                await _idle(main)
                self.assertIsNone(history.summary(other))
                self.assertIsNone(history.summary("main"))
                self.assertTrue(all(_SUMMARY not in request for request in model.requests))
                await delegate("reuse", "reuse original child", child_agent)
                await _idle(tree)
                await _idle(main)
                applied = [request for request in model.requests if _SUMMARY in request]
                self.assertEqual(len(applied), 1)
                self.assertIn(Message("user", "child while generating"), applied[0])
                self.assertEqual(applied[0][-1], Message("user", "reuse original child"))
                self.assertIsNotNone(history.summary(child))
                self.assertEqual(len(model.summary_requests), 1)
                self.assertIn(Message("user", first_child_text), model.summary_requests[0])
                self.assertNotIn(Message("user", "child while generating"), model.summary_requests[0])

    async def test_private_application_shares_one_generation_slot_across_groups(self):
        async def reply(messages):
            yield TextDelta("normal reply")
            yield InputUsage(240)

        model = _GatedModel(reply)
        settings = Settings(runtime=_SETTINGS, feishu=FeishuSettings())

        class Discovery:
            async def account_user(self):
                return ChannelUser("ou_fixture")

        async def run_groups(_discovery, shared_history, _asr, *, open_agent, **kwargs):
            self.assertIs(shared_history, history)
            for session_id in ("main-a", "main-b"):
                history.append(session_id, [_OLD])
                _report_high_usage(history, session_id)

            async with open_agent(history.ensure_main_agent("main-b"), Sink(), ChannelUser("ou_b")) as (second, _):
                async with open_agent(history.ensure_main_agent("main-a"), Sink(), ChannelUser("ou_a")) as (first, _):
                    first_task = await self._begin(first, model, "first group input")
                    await _idle(first)
                    second.submit_batch([Message("user", "second group while occupied")])
                    await _idle(second)
                    self.assertIsNone(second.last_failure)
                    self.assertEqual(model.requests[-1][-1].content, "second group while occupied")
                    self.assertEqual(len(model.summary_requests), 1)
                    self.assertIs(first._compactions, second._compactions)
                self.assertFalse(first_task.done())
                self.assertFalse(model.cancelled.is_set())
                await self._finish(model, first_task)
                self.assertIsNone(history.summary("main-a"))
                model.generating.clear()
                model.release.clear()

                second_task = await self._begin(second, model, "second group gets released slot")
                await _idle(second)
                self.assertIsNone(second.last_failure)
                self.assertNotIn(_SUMMARY, model.requests[-1])
                self.assertEqual(len(model.summary_requests), 2)
            self.assertFalse(second_task.done())
            self.assertFalse(model.cancelled.is_set())

        with closing(History(":memory:")) as history:
            with patch("agent_companion.app.run_private_chats", side_effect=run_groups):
                async with prepare_private_feishu(settings, history, discovery=Discovery(), roles=demo_roles()) as (_, run_private):
                    await run_private(model, object(), None)
            self.assertTrue(model.cancelled.is_set())
            self.assertIsNone(history.summary("main-b"))

    async def test_closing_runtime_preserves_shared_compaction_and_rechecks_before_apply(self):
        requesting, allow_read, reading = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def reply(messages):
            if messages[-1].content == "call":
                requesting.set()
                await asyncio.Event().wait()
            yield TextDelta("normal reply")

        model, sink = _GatedModel(reply), Sink()
        with closing(History(":memory:")) as history:
            history.append("main", [_OLD])
            _report_high_usage(history, "main")
            compactions = Compactions(history, model)

            def new_runtime():
                return Runtime("main", history, model, sink, settings=_SETTINGS, compactions=compactions)

            first, revoked, next_runtime = new_runtime(), new_runtime(), new_runtime()
            try:
                task = await self._begin(first, model, "call")
                await _wait(requesting)
                await first.close()
                self.assertFalse(task.done())
                self.assertFalse(model.cancelled.is_set())
                await self._finish(model, task)
                self.assertIsNone(history.summary("main"))

                original = revoked._context._sqlite

                async def slow_read(*args, **kwargs):
                    reading.set()
                    await allow_read.wait()
                    return await original(*args, **kwargs)

                with patch.object(revoked._context, "_sqlite", side_effect=slow_read):
                    revoked.submit_batch([Message("user", "revoked during preparation")])
                    await _wait(reading)
                    revoked.stop_accepting()
                    allow_read.set()
                    await _idle(revoked)
                    self.assertIsNone(history.summary("main"))
                    self.assertEqual(len(model.requests), 1)
                next_runtime.submit_batch([Message("user", "text afterwards")])
                await _idle(next_runtime)
                self.assertIn(_SUMMARY, model.requests[-1])
                self.assertIn(Message("user", "revoked during preparation"), model.requests[-1])
            finally:
                await asyncio.gather(first.close(), revoked.close(), next_runtime.close())
                await compactions.close()

    async def test_application_shutdown_cancels_generation_and_discards_late_result(self):
        model = _GatedModel(result_after_cancel=True)
        with closing(History(":memory:")) as history:
            history.append("main", [_OLD])
            _report_high_usage(history, "main")
            async with running_agent(Settings(runtime=_SETTINGS), history, model, Sink(), session_id="main",
                                     roles=demo_roles()) as (main, _):
                task = await self._begin(main, model)
                await _idle(main)
                self.assertFalse(task.done())
            self.assertTrue(model.cancelled.is_set())
            self.assertTrue(task.done())
            self.assertIsNone(main._compactions.ready("main", 0))
            self.assertIsNone(history.summary("main"))

    async def test_close_falls_back_to_cancel_if_adapter_has_no_matching_request(self):
        model = _GatedModel(result_after_cancel=True)
        with (closing(History(":memory:")) as history,
              patch.object(model, "stop_stream", return_value=False, create=True) as stop_stream):
            compactions = Compactions(history, model)
            try:
                compactions.start("main", Prefix(0, 1, 1), [_SUMMARIZE], (), thinking=False, timeout=5)
                await _wait(model.generating)
                task = compactions._generating
                await asyncio.wait_for(compactions.close(), 0.5)
                stop_stream.assert_called_once_with(task)
                self.assertTrue(model.cancelled.is_set())
                self.assertIsNone(compactions.ready("main", 0))
            finally:
                await compactions.close()

    async def test_native_summary_close_starts_transport_budget(self):
        fixture = _NativeSummaryTransport()
        async with fixture.client as client:
            model = DeepSeek(client)
            with closing(History(":memory:")) as history:
                compactions = Compactions(history, model)
                prefix = Prefix(0, 1, 1)
                stopping = None
                try:
                    compactions.start("main", prefix, [_SUMMARIZE], (), thinking=False, timeout=5)
                    await _wait(fixture.waiting)
                    with (patch.object(_http_stream, "_RESPONSE_CLOSE_TIMEOUT", 0.02),
                          self.assertLogs(_http_stream.__name__, "WARNING") as logged):
                        stopping = asyncio.create_task(compactions.close())
                        await _wait(fixture.closing)
                        await asyncio.wait_for(asyncio.shield(stopping), 0.5)
                    self.assertTrue(fixture.interrupted.is_set())
                    self.assertFalse(fixture.closed.is_set())
                    self.assertIn("transport_close_timeout", "\n".join(logged.output))
                    self.assertIsNone(compactions.ready("main", 0))
                finally:
                    fixture.release.set()
                    if stopping is not None:
                        await stopping
                    await compactions.close()
                    await model.aclose()

    async def test_native_summary_result_is_only_registered_while_open(self):
        for stopped in (False, True):
            with self.subTest(stopped=stopped):
                fixture = _NativeSummaryTransport(complete=True)
                async with fixture.client as client:
                    model = DeepSeek(client)
                    with closing(History(":memory:")) as history:
                        compactions = Compactions(history, model)
                        prefix = Prefix(0, 1, 1)
                        try:
                            compactions.start("main", prefix, [_SUMMARIZE], (), thinking=False, timeout=5)
                            task = compactions._generating
                            await _wait(fixture.closing)
                            self.assertFalse(compactions.can_start("another", 0, 1))
                            if stopped:
                                compactions.stop()
                            fixture.release.set()
                            await asyncio.wait_for(asyncio.shield(task), 0.5)
                            result = None if stopped else (prefix, Summary(1, "background result"))
                            self.assertEqual(compactions.ready("main", 0), result)
                            self.assertEqual(compactions.can_start("another", 0, 1), not stopped)
                            self.assertTrue(fixture.closed.is_set())
                        finally:
                            fixture.release.set()
                            await compactions.close()
                            await model.aclose()

    async def test_native_generation_timeout_starts_transport_budget(self):
        for summary in (False, True):
            with self.subTest(summary=summary):
                fixture = _NativeSummaryTransport()
                async with fixture.client as client:
                    model = DeepSeek(client)
                    with closing(History(":memory:")) as history:
                        runtime = Runtime("main", history, model, Sink(), settings=replace(_SETTINGS, model_timeout=0.05))
                        compactions = runtime._compactions
                        try:
                            with (patch.object(_http_stream, "_RESPONSE_CLOSE_TIMEOUT", 0.02),
                                  self.assertLogs("agent_companion", "WARNING") as logged):
                                if summary:
                                    compactions.start("main", Prefix(0, 1, 1), [_SUMMARIZE], (), thinking=False,
                                                      timeout=0.05)
                                    settled = asyncio.shield(compactions._generating)
                                else:
                                    runtime.submit_batch([Message("user", "fixture input")])
                                    settled = runtime.wait_idle()
                                await asyncio.wait_for(settled, 0.5)
                            self.assertTrue(fixture.interrupted.is_set())
                            self.assertIn("transport_close_timeout", "\n".join(logged.output))
                            if summary:
                                self.assertIsNone(compactions.ready("main", 0))
                                self.assertFalse(compactions.can_start("main", 0, 1))
                            else:
                                self.assertEqual(runtime.last_failure, "model_timeout")
                        finally:
                            fixture.release.set()
                            await runtime.close()
                            await model.aclose()

    async def test_summary_timeout_cools_down_and_discards_late_result(self):
        async def reply(messages):
            yield TextDelta("normal reply")
            yield InputUsage(240)

        model = _GatedModel(reply, result_after_cancel=True)
        with closing(History(":memory:")) as history:
            history.append("main", [_OLD])
            _report_high_usage(history, "main")
            runtime = Runtime("main", history, model, Sink(), settings=replace(_SETTINGS, model_timeout=0.1))
            try:
                with self.assertLogs("agent_companion.runtime.compaction", level="WARNING") as logs:
                    task = await self._begin(runtime, model)
                    await _idle(runtime)
                    self.assertIsNone(runtime.last_failure)
                    await asyncio.wait_for(asyncio.shield(task), 2)
                self.assertTrue(model.cancelled.is_set())
                self.assertIn("timeout", logs.output[0])
                runtime.submit_batch([Message("user", "during timeout cooldown")])
                await _idle(runtime)
                self.assertIsNone(history.summary("main"))
                self.assertEqual(len(model.summary_requests), 1)
                self.assertNotIn(_SUMMARY, model.requests[-1])
            finally:
                await runtime.close()

    async def test_sqlite_write_lock_does_not_delay_or_apply_candidate_to_primary_request(self):
        model = _GatedModel()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compaction.sqlite"
            with closing(History(path)) as history, closing(sqlite3.connect(path)) as fault:
                history.append("main", [_OLD])
                _report_high_usage(history, "main")
                runtime = Runtime("main", history, model, Sink(), settings=_SETTINGS)
                try:
                    await self._finish(model, await self._begin(runtime, model))
                    await _idle(runtime)
                    self.assertIsNone(history.summary("main"))
                    attempt = history.try_save_summary
                    observations = []

                    original_timeout = history._db.execute("PRAGMA busy_timeout").fetchone()[0]

                    def locked(*args, **kwargs):
                        fault.execute("BEGIN IMMEDIATE")
                        started = monotonic()
                        try:
                            return attempt(*args, **kwargs)
                        finally:
                            observations.append((monotonic() - started, history._db.execute("PRAGMA busy_timeout").fetchone()[0]))
                            fault.rollback()

                    with (patch.object(history, "try_save_summary", side_effect=locked),
                          self.assertLogs("agent_companion.runtime.compaction", level="WARNING") as logs):
                        runtime.submit_batch([Message("user", "while write locked")])
                        await _idle(runtime)
                    elapsed, restored = observations[0]
                    self.assertLess(elapsed, 1)  # 普通连接等待上限为 5 秒。
                    self.assertEqual(restored, original_timeout)
                    self.assertEqual(len(observations), 1)
                    self.assertIn("summary_write_failed", logs.output[0])
                    self.assertIsNone(runtime.last_failure)
                    self.assertIsNone(history.summary("main"))
                    self.assertIn(_OLD, model.requests[-1])
                    self.assertNotIn(_SUMMARY, model.requests[-1])
                finally:
                    await runtime.close()

    async def test_sqlite_cas_conflict_refreshes_same_request_unless_locked_or_unsafe(self):
        for mode in ("fresh", "locked", "protected"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "compaction.sqlite"
                model = _GatedModel()
                model.replay_scope = ReplayScope("fixture", b"a" * 32, "default")
                compatible = ProviderReplay(model.replay_scope, (ReplayItem("reasoning", "fixture-reason"),))
                other_scope = ReplayScope("fixture", b"b" * 32, "default")
                incompatible = ProviderReplay(other_scope, (ReplayItem("reasoning", "different-reason"),))
                with closing(History(path)) as history, closing(History(path)) as other:
                    first_old = Message("user", "already summarized " + "b" * 500)
                    competing_id, target = history.append("main", [first_old, _OLD])
                    _report_high_usage(history, "main")
                    runtime = Runtime("main", history, model, Sink(), settings=_SETTINGS)
                    try:
                        await self._finish(model, await self._begin(runtime, model))
                        await _idle(runtime)
                        save = history.try_save_summary
                        read = history.try_context_snapshot
                        competing_text = "other connection summary"
                        concurrently_appended = Message("user", "new input from the other connection")
                        racing_input = Message("user", "racing summary")
                        matching_reply = Message("assistant", "matching provider replay", replay=compatible)
                        alternate_reply = Message("assistant", "other provider replay", replay=incompatible)
                        reads = []
                        original_timeout = history._db.execute("PRAGMA busy_timeout").fetchone()[0]

                        def commit_first(session_id, candidate, *, covered, protected_from):
                            self.assertEqual(candidate.covers_through, target)
                            through = protected_from if mode == "protected" else competing_id
                            other.save_summary(session_id, Summary(through, competing_text))
                            other.append(session_id, [concurrently_appended])
                            for index, message in enumerate((matching_reply, alternate_reply)):
                                other.finalize_reply(ReplyOutcome(f"concurrent-{index}", session_id, "test", "complete"),
                                                     [message])
                            other.append("child", [Message("user", "different session")])
                            saved = save(session_id, candidate, covered=covered, protected_from=protected_from)
                            self.assertFalse(saved)
                            if mode == "locked":
                                other._db.execute("BEGIN EXCLUSIVE")
                            return saved

                        def read_once(*args, **kwargs):
                            started = monotonic()
                            try:
                                return read(*args, **kwargs)
                            finally:
                                elapsed = monotonic() - started
                                reads.append((elapsed, history._db.execute("PRAGMA busy_timeout").fetchone()[0]))
                                if mode == "locked":
                                    other._db.rollback()

                        with (patch.object(history, "try_save_summary", side_effect=commit_first) as write,
                              patch.object(history, "try_context_snapshot", side_effect=read_once)):
                            runtime.submit_batch([racing_input])
                            await _idle(runtime)
                            write.assert_called_once()
                        self.assertEqual(len(reads), 1)
                        self.assertEqual(reads[0][1], original_timeout)
                        self.assertEqual(history.summary("main").text, competing_text)
                        if mode == "protected":
                            self.assertEqual(runtime.last_failure, "context_failed")
                            self.assertEqual(len(model.requests), 1)  # 不发送吞掉当前输入的投影。
                            continue
                        self.assertIsNone(runtime.last_failure)
                        latest = model.requests[1]  # 就在冲突这一轮，而非下一轮。
                        common = (_OLD, Message("user", "first"), Message("assistant", "normal reply"), racing_input)
                        if mode == "locked":
                            self.assertLess(reads[0][0], 1)
                            self.assertEqual(latest, (first_old, *common))
                        else:
                            self.assertEqual(latest, (Message("system", "此前会话摘要：\n" + competing_text),
                                                      *common, concurrently_appended, matching_reply,
                                                      replace(alternate_reply, replay=None)))
                        self.assertNotIn(_SUMMARY, latest)
                    finally:
                        await runtime.close()

    async def test_expired_candidate_is_not_committed(self):
        model = _GatedModel()
        with (closing(History(":memory:")) as history,
              patch("agent_companion.runtime.compaction.monotonic", return_value=100) as clock):
            history.append("main", [_OLD])
            _report_high_usage(history, "main")
            runtime = Runtime("main", history, model, Sink(), settings=_SETTINGS)
            try:
                await self._finish(model, await self._begin(runtime, model))
                await _idle(runtime)
                clock.return_value = 401
                with patch.object(history, "try_save_summary", wraps=history.try_save_summary) as write:
                    runtime.submit_batch([Message("user", "after expiration")])
                    await _idle(runtime)
                    write.assert_not_called()
                self.assertIsNone(history.summary("main"))
                self.assertIn(_OLD, model.requests[-1])
                self.assertNotIn(_SUMMARY, model.requests[-1])
            finally:
                await runtime.close()
