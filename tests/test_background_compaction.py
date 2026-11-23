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

from agent_companion.app import prepare_private_feishu, running_agent
from agent_companion.config import FeishuSettings, RuntimeSettings, Settings
from agent_companion.protocol import (InputUsage, Message, ProviderReplay, ReplayItem, ReplayScope,
                                      ReplyRoute, TextDelta, ToolCall, ToolReady)
from agent_companion.runtime.context import _SUMMARIZE
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.subtasks import DELEGATE_TOOL_NAME
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
            async with running_agent(settings, history, model, Sink(), session_id="main") as (main, subtasks):
                async def delegate(label, prompt, child=None):
                    args = {"prompt": prompt}
                    if child is not None:
                        args["child_session_id"] = child
                    call = ToolCall(label, DELEGATE_TOOL_NAME, json.dumps(args))
                    accepted = json.loads(await subtasks.tools.execute(call, ToolContext("main", "test", label, True)))
                    self.assertEqual(accepted["status"], "accepted")
                    return accepted["child_session_id"]

                child = await delegate("start", first_child_text)
                await _idle(subtasks)
                await _idle(main)
                await delegate("generate", "child while generating", child)
                await _wait(model.generating)
                task = main._compactions._generating
                self.assertIsNotNone(task)
                await _idle(subtasks)
                await _idle(main)
                self.assertNotIn(child, subtasks._workers)
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

                other = await delegate("other", "other child")
                await _idle(subtasks)
                await _idle(main)
                self.assertIsNone(history.summary(other))
                self.assertIsNone(history.summary("main"))
                self.assertTrue(all(_SUMMARY not in request for request in model.requests))
                await delegate("reuse", "reuse original child", child)
                await _idle(subtasks)
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
        feishu = FeishuSettings(own_open_id="ou_fixture")
        settings = Settings(runtime=_SETTINGS, feishu=feishu)

        class Discovery:
            async def account_open_id(self):
                return feishu.own_open_id

        async def run_groups(_discovery, shared_history, _asr, *, open_agent, **kwargs):
            self.assertIs(shared_history, history)
            for session_id in ("main-a", "main-b"):
                history.append(session_id, [_OLD])
                _report_high_usage(history, session_id)

            async with open_agent("main-b", Sink()) as (second, _):
                async with open_agent("main-a", Sink()) as (first, _):
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
                run_private = await prepare_private_feishu(settings, history, discovery=Discovery())
                await run_private(model, object(), None)
            self.assertTrue(model.cancelled.is_set())
            self.assertIsNone(history.summary("main-b"))

    async def test_hangup_only_cancels_foreground_and_route_is_rechecked_before_apply(self):
        requesting, allow_read, reading = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def reply(messages):
            if messages[-1].content == "call":
                requesting.set()
                await asyncio.Event().wait()
            yield TextDelta("normal reply")

        model = _GatedModel(reply)
        active = True
        sink = Sink()
        route = ReplyRoute(sink, lambda: active, 1)
        with closing(History(":memory:")) as history:
            history.append("main", [_OLD])
            _report_high_usage(history, "main")
            runtime = Runtime("main", history, model, sink, settings=_SETTINGS)
            try:
                runtime.submit_batch([Message("user", "call")], reply_route=route)
                await _wait(requesting)
                await _wait(model.generating)
                task = runtime._compactions._generating
                active = False
                runtime.stop_reply_route(route)
                await _idle(runtime)
                self.assertEqual(runtime.last_failure, "model_timeout")
                self.assertFalse(task.done())
                self.assertFalse(model.cancelled.is_set())
                await self._finish(model, task)
                self.assertIsNone(history.summary("main"))

                original = runtime._context._system_messages

                async def slow_read(*args):
                    reading.set()
                    await allow_read.wait()
                    return await original(*args)

                active = True
                next_route = ReplyRoute(sink, lambda: active, 1)
                with patch.object(runtime._context, "_system_messages", side_effect=slow_read):
                    runtime.submit_batch([Message("user", "route closes in preparation")], reply_route=next_route)
                    await _wait(reading)
                    active = False
                    allow_read.set()
                    await _idle(runtime)
                    self.assertIsNone(history.summary("main"))
                    self.assertEqual(len(model.requests), 1)
                runtime.submit_batch([Message("user", "text afterwards")])
                await _idle(runtime)
                self.assertIn(_SUMMARY, model.requests[-1])
                self.assertIn(Message("user", "route closes in preparation"), model.requests[-1])
            finally:
                await runtime.close()

    async def test_application_shutdown_cancels_generation_and_discards_late_result(self):
        model = _GatedModel(result_after_cancel=True)
        with closing(History(":memory:")) as history:
            history.append("main", [_OLD])
            _report_high_usage(history, "main")
            async with running_agent(Settings(runtime=_SETTINGS), history, model, Sink(), session_id="main") as (main, _):
                task = await self._begin(main, model)
                await _idle(main)
                self.assertFalse(task.done())
            self.assertTrue(model.cancelled.is_set())
            self.assertTrue(task.done())
            self.assertIsNone(main._compactions.ready("main", 0))
            self.assertIsNone(history.summary("main"))

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
