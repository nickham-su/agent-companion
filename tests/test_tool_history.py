import asyncio
from contextlib import aclosing, closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

from agent_companion.config import RuntimeSettings
from agent_companion.models.deepseek import DeepSeek
from agent_companion.protocol import (DeliveryStatus, Message, Receipt, Reasoning, Text, TextDelta, ToolCall,
                                      ToolReady, ToolSpec)
from agent_companion.runtime.compaction import Compactions
from agent_companion.runtime.context import Context
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.sqlite_access import SQLiteAccess
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.storage.sqlite import DatabaseFormatError, History, Summary


_SPEC = ToolSpec("observe", "read a test value", {"type": "object"})
_FIRST = ToolCall("first", _SPEC.name, "{}")
_SECOND = ToolCall("second", _SPEC.name, "{}")


class _Model:
    def __init__(self, *calls):
        self.calls = calls
        self.requests = []

    async def stream(self, messages, tools, *, thinking, instructions=None):
        self.requests.append(tuple(messages))
        if len(self.requests) == 1 and self.calls:
            for call in self.calls:
                yield ToolReady(call)
        else:
            yield TextDelta("model answer")


def _context(history, model=None, settings=None):
    compactions = Compactions(history)
    return (Context(history, settings or RuntimeSettings(), compactions, model=model or _Model(),
                    sqlite=SQLiteAccess(retry_locks=False, stopped=lambda: False)), compactions)


async def _prepare(context, *, protected_from=None, last_input_tokens=None):
    return await context.prepare("main", thinking=False, tools=(_SPEC,), protected_from=protected_from,
                                 can_send=lambda: True, last_input_tokens=last_input_tokens)


async def _wait_summary(compactions, covered):
    while (ready := compactions.ready("main", covered)) is None:
        await asyncio.sleep(0)
    return ready


# 两个真正的进程退出窗口。父进程先验证此时另一写入者仍活着，再让其不经 finally 退出。
_CRASH_PROGRAM = """
import asyncio
import os
from pathlib import Path
import sys
from agent_companion.protocol import (REPLAY_KEY_BYTES, AssistantPartsReady, Message, PrivateData, Reasoning,
                                      Text, TextDelta, ToolCall, ToolReady, ToolSpec)
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.storage.sqlite import History

path, marker, window = sys.argv[1:]
key = bytes(REPLAY_KEY_BYTES)


def crash():
    print('ready', flush=True)
    sys.stdin.buffer.read(1)
    os._exit(23)


class CrashHistory(History):
    def stage_tool_calls(self, *args):
        super().stage_tool_calls(*args)
        if window == 'before_handler':
            crash()

    def record_tool_result(self, *args):
        if window == 'after_side_effect':
            crash()
        super().record_tool_result(*args)


spec = ToolSpec('observe', 'read a test value', {'type': 'object'})
call = ToolCall('first', spec.name, '{}', private=PrivateData(key, b'native-call'))


class Model:
    async def stream(self, messages, tools, *, thinking, instructions=None):
        yield TextDelta('unconfirmed specimen')
        yield AssistantPartsReady((Text('unconfirmed specimen', PrivateData(key, b'native-text')),
                                   Reasoning(PrivateData(key, b'private-thought')), call))
        yield ToolReady(call)


async def tool(args, context):
    with Path(marker).open('a') as stream:
        stream.write('executed\\n')
    return 'not yet committed'


async def main():
    runtime = Runtime('main', CrashHistory(path), Model(), None, tools=Tools((Tool(spec, tool),)))
    runtime.submit_batch((Message('user', 'request'),))
    await runtime.wait_idle()
    raise RuntimeError('crash hook was not reached')


asyncio.run(main())
"""


_OBSERVE_FORK_PROGRAM = """
from contextlib import ExitStack
import os
import sqlite3
import sys
from agent_companion.storage._reply_writers import ReplyWriters


def assert_locked(writers):
    try:
        with writers.own('contender'):
            raise AssertionError('writer unexpectedly acquired the lock')
    except sqlite3.OperationalError as exc:
        assert str(exc) == 'database is locked'


parent_lock = ReplyWriters(sys.argv[1])
child_lock = ReplyWriters(sys.argv[1] + '.child')
ready_r, ready_w = os.pipe()
stop_r, stop_w = os.pipe()
with ExitStack() as child_locks:
    with parent_lock.observe():
        pid = os.fork()
        if pid == 0:
            # 子进程关闭自己的副本不能解开父进程的共享锁。
            assert_locked(parent_lock)
            # 先建立新锁，复用已关闭的 fd，再退出从父进程继承的 observe。
            child_locks.enter_context(child_lock.own('child'))
        else:
            os.close(ready_w)
            os.close(stop_r)
            assert os.read(ready_r, 1) == b'x'
    if pid == 0:
        os.close(ready_r)
        os.close(stop_w)
        os.write(ready_w, b'x')
        os.read(stop_r, 1)
    else:
        try:
            # 子进程仍存活，且其新锁没被旧 observe 误关；父进程可以重新写入。
            assert_locked(child_lock)
            with parent_lock.own('parent'):
                assert parent_lock.owns('parent')
        finally:
            os.close(stop_w)
            assert os.waitpid(pid, 0)[1] == 0
"""


class ReplyOwnershipTest(unittest.TestCase):
    def test_rejects_shared_memory_uris_before_creating_schema(self):
        with closing(sqlite3.connect(":memory:")) as probe:
            if not probe.execute("SELECT sqlite_compileoption_used('USE_URI')").fetchone()[0]:
                self.skipTest("SQLite 不会把默认连接中的文件名解释为 URI")
        for uri in ("file::memory:?cache=shared", "file:tool-history-test?mode=memory&cache=shared"):
            with self.subTest(uri=uri), closing(sqlite3.connect(uri)) as observer:
                self.assertEqual(observer.execute("PRAGMA database_list").fetchone()[2], "")
                with self.assertRaisesRegex(DatabaseFormatError, "不支持 URI 内存数据库"):
                    History(uri)
                self.assertEqual(observer.execute("PRAGMA user_version").fetchone()[0], 0)
                self.assertIsNone(observer.execute("SELECT 1 FROM sqlite_schema LIMIT 1").fetchone())

    def test_private_memory_and_regular_file_names_are_still_supported(self):
        with closing(History(":memory:")) as first, closing(History(Path(":memory:"))) as second:
            first.append("main", [Message("user", "private")])
            self.assertEqual(second.read_after("main"), [])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "file:regular?mode=memory&cache=shared"
            with closing(History(path)) as first, closing(History(path)) as second:
                first.append("main", [Message("user", "disk")])
                self.assertEqual(second.read_after("main")[0].message.text, "disk")

    def test_fork_does_not_keep_shared_observer_lock_in_child(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "history.db"
            result = subprocess.run([sys.executable, "-c", _OBSERVE_FORK_PROGRAM, str(path)],
                                    capture_output=True, timeout=5, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)


class ToolHistoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_crash_before_handler_and_between_side_effect_and_result(self):
        for window in ("before_handler", "after_side_effect"):
            with self.subTest(window=window), tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
                path = Path(folder) / "history.db"
                marker = Path(folder) / "effects"
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-c", _CRASH_PROGRAM, str(path), str(marker), window,
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                try:
                    ready = await asyncio.wait_for(process.stdout.readline(), 5)
                    self.assertEqual(ready, b"ready\n", await process.stderr.read() if not ready else None)
                    expected_effect = window == "after_side_effect"
                    self.assertEqual(marker.exists(), expected_effect)
                    with closing(History(path)) as other:
                        with self.assertRaisesRegex(sqlite3.OperationalError, "locked"):
                            with other.own_reply("independent-writer"):
                                self.fail("foreign writer was allowed")
                        _, entries = other.context_snapshot("main")
                        staged = entries[-1]
                        self.assertTrue(staged.active_reply)
                        self.assertEqual(staged.message.text, "")
                        self.assertIsInstance(staged.message.parts[0], Reasoning)
                        self.assertEqual(staged.message.tool_calls[0].private.payload, b"native-call")
                        self.assertIsNone(staged.message.tool_calls[0].result)
                        context, _ = _context(other)
                        _, projected = await _prepare(context)
                        self.assertEqual([message.kind for message in projected], ["user"])
                        later, = other.append("main", [Message("user", "later input")])
                        self.assertFalse(other.try_save_summary("main", Summary(staged.message_id, "must wait"),
                                                               covered=0, protected_from=later))
                    process.stdin.write(b"x")
                    await process.stdin.drain()
                    await asyncio.wait_for(process.wait(), 5)
                    self.assertEqual(process.returncode, 23)
                finally:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()

                reruns = []

                async def tool(args, context):
                    reruns.append(context.call_id)
                    return "unexpected"

                with closing(History(path)) as reopened:
                    _, entries = reopened.context_snapshot("main")
                    self.assertFalse(entries[1].active_reply)
                    self.assertEqual(entries[1].message_id, staged.message_id)
                    model = _Model()
                    runtime = Runtime("main", reopened, model, None, tools=Tools((Tool(_SPEC, tool),)))
                    try:
                        runtime.submit_batch((Message("user", "continue"),))
                        await asyncio.wait_for(runtime.wait_idle(), 2)
                        self.assertIsNone(runtime.last_failure)
                    finally:
                        await runtime.close()
                    self.assertEqual(reruns, [])
                    result = model.requests[0][1].tool_calls[0].result
                    self.assertEqual(json.loads(result)["status"], "unknown")
                    self.assertIn("未知", result)
                    self.assertIsNone(reopened.read_after("main")[1].message.tool_calls[0].result)
                    self.assertFalse(reopened.search("main", keyword="unknown"))
                    self.assertFalse(reopened.search("main", keyword="unconfirmed"))
                    self.assertEqual(marker.read_text() if expected_effect else None,
                                     "executed\n" if expected_effect else None)

    async def test_out_of_order_results_keep_real_empty_text_and_share_owner_across_connections(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "history.db"
            with closing(History(path)) as history, closing(History(path)) as other:
                first_started, second_committed = asyncio.Event(), asyncio.Event()

                async def tool(args, context):
                    if context.call_id == _FIRST.id:
                        first_started.set()
                        await asyncio.Future()
                    return ""

                record = history.record_tool_result

                def record_and_notify(session_id, reply_id, call_id, output):
                    record(session_id, reply_id, call_id, output)
                    if call_id == _SECOND.id:
                        second_committed.set()

                runtime = Runtime("main", history, _Model(_FIRST, _SECOND), None,
                                  tools=Tools((Tool(_SPEC, tool),)))
                try:
                    with patch.object(history, "record_tool_result", side_effect=record_and_notify):
                        runtime.submit_batch((Message("user", "request"),))
                        await asyncio.wait_for(asyncio.gather(first_started.wait(), second_committed.wait()), 2)
                        _, entries = other.context_snapshot("main")
                        self.assertTrue(entries[-1].active_reply)
                        self.assertEqual([(c.id, c.result) for c in entries[-1].message.tool_calls],
                                         [("first", None), ("second", "")])
                        context, _ = _context(other)
                        _, projected = await _prepare(context)
                        self.assertEqual([message.kind for message in projected], ["user"])
                finally:
                    await runtime.close()
                self.assertEqual(other.outcomes("main"), [])
                _, projected = await _prepare(context)
                first, second = projected[-1].tool_calls
                self.assertEqual(json.loads(first.result)["status"], "unknown")
                self.assertEqual(second.result, "")
                self.assertEqual([call.result for call in other.read_after("main")[-1].message.tool_calls], [None, ""])

    async def test_active_group_limits_summary_generation_and_final_commit(self):
        with closing(History(":memory:")) as history:
            old, = history.append("main", [Message("user", "old words " * 50)])
            summary_model = _Model()
            context, compactions = _context(history, summary_model, RuntimeSettings(context_budget=500, tail_budget=1))
            self.addAsyncCleanup(compactions.close)
            with history.own_reply("working"):
                history.stage_tool_calls("main", "working", Message("assistant", (_FIRST, _SECOND)))
                history.record_tool_result("main", "working", _SECOND.id, "true result")
                active = history.read_after("main")[-1].message_id
                later, latest = history.append("main", [Message("user", "later"), Message("user", "latest")])
                _, projected = await _prepare(context, protected_from=latest, last_input_tokens=1000)
                self.assertEqual([message.text for message in projected], ["old words " * 50, "later", "latest"])
                self.assertFalse(history.try_save_summary("main", Summary(later, "skip over active"),
                                                          covered=0, protected_from=latest))
                with self.assertRaisesRegex(ValueError, "active reply"):
                    history.save_summary("main", Summary(active, "skip active"))
                prefix, candidate = await asyncio.wait_for(_wait_summary(compactions, 0), 2)
                self.assertEqual((prefix.through, candidate.covers_through), (old, old))
                self.assertFalse(any(message.tool_calls for message in summary_model.requests[0]))
                await _prepare(context, protected_from=latest, last_input_tokens=1000)
                self.assertEqual(history.summary("main").covers_through, old)
            # 所有者退出后，同一组在普通续轮和摘要请求中使用相同的缺失占位与已有结果。
            _, projected = await _prepare(context, protected_from=latest, last_input_tokens=1000)
            assistant = projected[1]
            self.assertEqual(json.loads(assistant.tool_calls[0].result)["status"], "unknown")
            self.assertEqual(assistant.tool_calls[1].result, "true result")
            await asyncio.wait_for(_wait_summary(compactions, old), 2)
            self.assertEqual(summary_model.requests[-1][1], assistant)
            self.assertIsNone(history.read_after("main", old)[0].message.tool_calls[0].result)

    async def test_staging_write_failure_never_dispatches_even_if_commit_was_ambiguous(self):
        for committed in (False, True):
            with self.subTest(committed=committed), closing(History(":memory:")) as history:
                called = []

                async def tool(args, context):
                    called.append(context.call_id)
                    return "unexpected"

                stage = history.stage_tool_calls

                def fail(*args):
                    if committed:
                        stage(*args)
                    raise sqlite3.OperationalError("simulated write failure")

                with patch.object(history, "stage_tool_calls", side_effect=fail):
                    runtime = Runtime("main", history, _Model(_FIRST), None, tools=Tools((Tool(_SPEC, tool),)))
                    try:
                        runtime.submit_batch((Message("user", "request"),))
                        await asyncio.wait_for(runtime.wait_idle(), 2)
                        self.assertEqual(runtime.last_failure, "runtime_failed")
                    finally:
                        await runtime.close()
                self.assertEqual(called, [])
                self.assertEqual(history.outcomes("main"), [])
                raw = history.read_after("main")
                self.assertEqual(len(raw), 2 if committed else 1)
                if committed:
                    self.assertEqual(raw[-1].message.tool_calls, (_FIRST,))

    async def test_voice_interruption_during_busy_retry_prevents_staging_and_handler(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "history.db"
            with closing(History(path)) as history, closing(sqlite3.connect(path)) as competing:
                voice_ready, allow_tool, attempted, interrupted = (asyncio.Event() for _ in range(4))
                reader_closed = asyncio.Event()
                called = []

                class Voice:
                    async def deliver(self, reply):
                        reply.select_voice_delivery()
                        await anext(reply)
                        voice_ready.set()
                        await interrupted.wait()
                        return Receipt(reply.info.reply_id, DeliveryStatus.PARTIAL, "prefix")

                class Model:
                    async def stream(self, messages, tools, *, thinking, instructions=None):
                        try:
                            yield TextDelta("prefix then unconfirmed")
                            await allow_tool.wait()
                            yield ToolReady(_FIRST)
                        finally:
                            reader_closed.set()

                async def tool(args, context):
                    called.append(context.call_id)
                    return "unexpected"

                stage = history.stage_tool_calls

                def try_stage(*args):
                    attempted.set()
                    return stage(*args)

                runtime = Runtime("main", history, Model(), Voice(), retry_sqlite_locks=True,
                                  tools=Tools((Tool(_SPEC, tool),)))
                try:
                    with patch.object(history, "stage_tool_calls", side_effect=try_stage):
                        runtime.submit_batch((Message("user", "request"),))
                        await asyncio.wait_for(voice_ready.wait(), 2)
                        competing.execute("BEGIN IMMEDIATE")
                        allow_tool.set()
                        await asyncio.wait_for(attempted.wait(), 2)
                        interrupted.set()
                        # 保持锁，直到语音回执被接受，避免测试偶然先完成第二次写入。
                        await asyncio.wait_for(reader_closed.wait(), 2)
                        competing.rollback()
                        await asyncio.wait_for(runtime.wait_idle(), 2)
                    self.assertEqual(called, [])
                    rows = history.read_after("main")
                    self.assertEqual([row.message.text for row in rows], ["request", "prefix"])
                    self.assertEqual(rows[-1].message.tool_calls, ())
                finally:
                    competing.rollback()
                    await runtime.close()

    async def test_voice_interruption_keeps_verified_order_when_second_call_waits_for_sqlite(self):
        await self._interrupt_second_dispatch("voice")

    async def test_dispatch_deadline_keeps_verified_order_when_second_call_waits_for_sqlite(self):
        for cause in ("cooperative_deadline", "cancelled_await"):
            with self.subTest(cause=cause):
                await self._interrupt_second_dispatch(cause)

    async def _interrupt_second_dispatch(self, cause):
        by_voice = cause == "voice"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.db"
            with closing(History(path)) as history, closing(sqlite3.connect(path)) as competing:
                attempted, voice_cut, reader_closed, first_recorded = (asyncio.Event() for _ in range(4))
                called, busy = [], []
                requests = 0

                def delta(**fields):
                    return {"choices": [{"index": 0, "delta": fields, "finish_reason": None}]}

                def call(index, value):
                    return delta(tool_calls=[{"index": index, "id": value.id, "type": "function",
                                              "function": {"name": value.name, "arguments": value.arguments}}])

                frames = [delta(reasoning_content="synthetic thought"), call(0, _FIRST),
                          delta(content="prefix then unconfirmed"), call(1, _SECOND),
                          {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}]
                stream = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"

                class Model(DeepSeek):
                    async def stream(self, *args, **kwargs):
                        try:
                            async with aclosing(super().stream(*args, **kwargs)) as events:
                                async for event in events:
                                    if isinstance(event, ToolReady) and event.call.id == _SECOND.id:
                                        # 让 A 的真实空结果先提交，再由另一连接阻塞 B 的预写。
                                        await first_recorded.wait()
                                    yield event
                        finally:
                            reader_closed.set()

                class Voice:
                    async def deliver(self, reply):
                        reply.select_voice_delivery()
                        await anext(reply)
                        if by_voice:
                            await voice_cut.wait()
                        else:
                            # 让截止先到，不能由同时就绪的语音回执取消读流。
                            async for _ in reply:
                                pass
                        return Receipt(reply.info.reply_id, DeliveryStatus.PARTIAL, "prefix")

                async def tool(args, context):
                    called.append(context.call_id)
                    return ""

                stage = history.stage_tool_calls
                record = history.record_tool_result

                def lock_second(session_id, reply_id, message):
                    if len(message.tool_calls) != 2:
                        return stage(session_id, reply_id, message)
                    if not attempted.is_set():
                        competing.execute("BEGIN IMMEDIATE")
                        attempted.set()
                    try:
                        stage(session_id, reply_id, message)
                    except sqlite3.OperationalError as exc:
                        busy.append(exc.sqlite_errorcode)
                        raise

                def record_first(session_id, reply_id, call_id, output):
                    record(session_id, reply_id, call_id, output)
                    if call_id == _FIRST.id:
                        first_recorded.set()

                def endpoint(_):
                    nonlocal requests
                    requests += 1
                    return httpx.Response(200, text=stream)

                async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
                    model = Model(client)
                    if cause == "cancelled_await":
                        # 没有协作停止时，同一截止会直接取消 SQLite 锁重试中的 await。
                        model.stop_stream = None
                    settings = RuntimeSettings() if by_voice else RuntimeSettings(model_timeout=.3)
                    runtime = Runtime("main", history, model, Voice(), thinking=True, settings=settings,
                                      retry_sqlite_locks=True, tools=Tools((Tool(_SPEC, tool),)))
                    cancelled_during_write = False
                    write_input = runtime._sqlite.write_input

                    async def observe_write(operation):
                        nonlocal cancelled_during_write
                        try:
                            return await write_input(operation)
                        except asyncio.CancelledError:
                            cancelled_during_write = True
                            raise

                    try:
                        with (patch.object(history, "stage_tool_calls", side_effect=lock_second),
                              patch.object(history, "record_tool_result", side_effect=record_first),
                              patch.object(runtime._sqlite, "write_input", new=observe_write)):
                            runtime.submit_batch((Message("user", "request"),))
                            await asyncio.wait_for(attempted.wait(), 2)
                            if by_voice:
                                voice_cut.set()
                            await asyncio.wait_for(reader_closed.wait(), 2)
                            competing.rollback()
                            await asyncio.wait_for(runtime.wait_idle(), 2)
                    finally:
                        competing.rollback()
                        await runtime.close()
                self.assertTrue(busy)
                self.assertEqual(set(busy), {sqlite3.SQLITE_BUSY})
                if not by_voice:
                    self.assertEqual(cancelled_during_write, cause == "cancelled_await")
                self.assertEqual(requests, 1)
                self.assertEqual(called, [_FIRST.id])
                _, row = history.read_after("main")
                reason, call_part, text = row.message.parts
                self.assertIsInstance(reason, Reasoning)
                self.assertEqual(reason.private.payload, b"synthetic thought")
                self.assertEqual(call_part.private.payload, b"0")
                self.assertEqual((call_part.id, call_part.result), (_FIRST.id, ""))
                self.assertEqual(text, Text("prefix"))
                outcome, = history.outcomes("main")
                self.assertEqual(outcome.model_status, "model_interrupted" if by_voice else "model_timeout")
                self.assertEqual(outcome.delivery_status, DeliveryStatus.PARTIAL)
                self.assertFalse(history.search("main", keyword="unconfirmed"))

    async def test_result_write_failure_still_settles_delivery_without_retrying_tools(self):
        for delivery in (DeliveryStatus.FULL, DeliveryStatus.PARTIAL):
            for committed in (False, True):
                with self.subTest(delivery=delivery, committed=committed), closing(History(":memory:")) as history:
                    called, attempts = [], []
                    full = "confirmed prefix and suffix"
                    expected = full if delivery == DeliveryStatus.FULL else "confirmed prefix"

                    class Model(_Model):
                        async def stream(self, messages, tools, *, thinking, instructions=None):
                            yield TextDelta(full)
                            async for event in super().stream(messages, tools, thinking=thinking,
                                                              instructions=instructions):
                                yield event

                    class Channel:
                        async def deliver(self, reply):
                            self.full = "".join([text async for text in reply])
                            return Receipt(reply.info.reply_id, delivery, expected)

                    async def tool(args, context):
                        called.append(context.call_id)
                        return "unsettled result" if context.call_id == _FIRST.id else "other result"

                    record = history.record_tool_result

                    def fail_first(session_id, reply_id, call_id, output):
                        attempts.append(call_id)
                        if call_id == _FIRST.id:
                            if committed:
                                record(session_id, reply_id, call_id, output)
                            raise sqlite3.OperationalError("synthetic result failure")
                        record(session_id, reply_id, call_id, output)

                    model, channel = Model(_FIRST, _SECOND), Channel()
                    runtime = Runtime("main", history, model, channel, tools=Tools((Tool(_SPEC, tool),)))
                    try:
                        with patch.object(history, "record_tool_result", side_effect=fail_first), self.assertLogs(
                                "agent_companion.runtime.engine", level="ERROR") as logs:
                            runtime.submit_batch((Message("user", "request"),))
                            await asyncio.wait_for(runtime.wait_idle(), 2)
                            runtime.trigger_run()
                            await asyncio.wait_for(runtime.wait_idle(), 2)
                        self.assertEqual(runtime.last_failure, "runtime_failed")
                    finally:
                        await runtime.close()
                    self.assertIn("tool result write failed", " ".join(logs.output))
                    self.assertNotIn("synthetic result failure", " ".join(logs.output))
                    self.assertEqual(channel.full, full)
                    self.assertEqual(len(model.requests), 1)
                    self.assertCountEqual(called, [_FIRST.id, _SECOND.id])
                    self.assertCountEqual(attempts, called)
                    _, row = history.read_after("main")
                    self.assertEqual(row.message.text, expected)
                    first, second = row.message.tool_calls
                    self.assertEqual(first.result, "unsettled result" if committed else None)
                    self.assertEqual(second.result, "other result")
                    outcome, = history.outcomes("main")
                    self.assertEqual(outcome.assistant_message_id, row.message_id)
                    self.assertEqual(outcome.delivery_status, delivery)
                    self.assertEqual(outcome.model_status, "complete")
                    context, compactions = _context(history)
                    try:
                        _, projected = await _prepare(context)
                        result = projected[-1].tool_calls[0].result
                        if committed:
                            self.assertEqual(result, "unsettled result")
                        else:
                            self.assertEqual(json.loads(result)["status"], "unknown")
                            self.assertFalse(history.search("main", keyword="unsettled"))
                    finally:
                        await compactions.close()
