import asyncio
from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from agent_companion.protocol import (DeliveryStatus, InternalInput, Message, PersistedInput, Receipt,
                                      TextDelta, ToolCall, ToolReady, ToolSpec)
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.session_scheduler import SessionSchedulers
from agent_companion.storage.sqlite import History


class _Model:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    async def stream(self, messages, tools, *, thinking):
        self.requests.append(tuple(messages))
        for item in next(self.responses):
            yield item


class _Channel:
    async def deliver(self, reply):
        text = "".join([part async for part in reply])
        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)


class SessionSchedulerTest(unittest.IsolatedAsyncioTestCase):
    def make_runtime(self, model, **options):
        history = History(":memory:")
        self.addCleanup(history.close)
        runtime = Runtime("main", history, model, _Channel(), **options)
        self.addAsyncCleanup(runtime.close)
        return runtime, history

    def record_inbound(self, history, *texts):
        prior = history.pending_inbound("main", "test")
        history.record_inbound("main", "test", 0, [(text, Message("user", text)) for text in texts])
        (batch_id,) = set(history.pending_inbound("main", "test")) - set(prior)
        return PersistedInput(batch_id)

    async def test_first_use_and_rebind_keep_one_instance_but_clear_old_route_state(self):
        registry = SessionSchedulers()
        start = asyncio.Event()

        async def get():
            await start.wait()
            return registry.get_or_create("main")

        tasks = [asyncio.create_task(get()) for _ in range(8)]
        start.set()
        first, *rest = await asyncio.gather(*tasks)
        self.assertTrue(all(item is first for item in rest))
        self.assertIsNot(registry.get_or_create("another"), first)

        discarded, accepted = [], []
        batch = PersistedInput(1)
        owner = first.reserve_agent()
        revoke_old = first.bind(discarded.append, owner=owner)
        first.session_input((batch, Message("user", "old route")), input_active=True)
        revoke_old()
        self.assertEqual(discarded, [])
        with self.assertRaises(RuntimeError):
            first.session_input((Message("user", "late old route"),))
        with self.assertRaisesRegex(RuntimeError, "already bound or owned"):
            first.reserve_agent()
        first.release_agent(owner)

        self.assertIs(registry.get_or_create("main"), first)
        successor = first.reserve_agent()
        first.bind(accepted.append, owner=successor)
        revoke_old()
        first.release_agent(owner)  # 旧 Runtime 的延迟清理不能解除新绑定或释放新所有权。
        first.session_input((batch,))
        self.assertEqual(accepted, [(batch,)])
        first.release_agent(successor)

    async def test_idle_event_flushes_one_fifo_batch_and_busy_transaction_retries_without_duplicates(self):
        model = _Model((TextDelta("done"),))
        runtime, history = self.make_runtime(model, retry_sqlite_locks=True)
        scheduler = SessionSchedulers().get_or_create("main")
        handoffs = []

        def submit(items):
            handoffs.append(items)
            runtime.submit_inputs(items)

        scheduler.bind(submit)
        first = self.record_inbound(history, "feishu 1", "feishu 2")
        second = self.record_inbound(history, "feishu 3")
        internal = InternalInput(Message("user", "background result"))
        transcript = Message("user", "transcript")
        scheduler.session_input(input_active=True)
        scheduler.session_input((first,))
        scheduler.session_input((first, internal, second, first))
        self.assertEqual(handoffs, [])
        self.assertEqual(history.read_after("main"), [])
        self.assertEqual(model.requests, [])

        append = history._append
        rolled_back = asyncio.Event()

        def fail_after_final_insert(session_id, messages):
            messages = tuple(messages)
            result = append(session_id, messages)
            if messages == (transcript,) and not rolled_back.is_set():
                rolled_back.set()
                raise sqlite3.OperationalError("database is locked")
            return result

        with patch.object(history, "_append", side_effect=fail_after_final_insert):
            scheduler.session_input((transcript,), input_active=False)
            self.assertEqual(handoffs, [(first, internal, second, transcript)])
            await asyncio.wait_for(rolled_back.wait(), 2)
            self.assertFalse(history._db.in_transaction)
            self.assertEqual(history.read_after("main"), [])
            self.assertEqual(history.pending_inbound("main", "test"), [first.batch_id, second.batch_id])
            # 调度器已移交；Runtime 锁等待期间的轮询重投不创建第二条待执行输入。
            scheduler.session_input((first, second))
            await asyncio.wait_for(runtime.wait_idle(), 2)

        self.assertIsNone(runtime.last_failure)
        self.assertEqual(history.pending_inbound("main", "test"), [])
        expected = ["feishu 1", "feishu 2", "background result", "feishu 3", "transcript"]
        self.assertEqual([row.message.content for row in history.read_after("main")], [*expected, "done"])
        self.assertEqual([[m.content for m in request if m.role == "user"] for request in model.requests],
                         [expected])
        scheduler.session_input((first, second))
        await asyncio.wait_for(runtime.wait_idle(), 2)
        self.assertEqual(len(model.requests), 1)
        self.assertIsNone(runtime.last_failure)

    async def test_tool_continuation_authorizes_only_actually_recorded_user_inputs(self):
        for has_new_user in (False, True):
            with self.subTest(has_new_user=has_new_user):
                model = _Model((ToolReady(ToolCall("c1", "probe", "{}")),),
                               (ToolReady(ToolCall("c2", "probe", "{}")),), (TextDelta("done"),))
                authorizations, run_ids = [], []

                async def tool(args, context):
                    authorizations.append(context.has_user_input)
                    run_ids.append(context.run_id)
                    if context.call_id == "c1":
                        runtime.submit_inputs(inputs)
                    return "ok"

                spec = ToolSpec("probe", "check input permission", {"type": "object"})
                runtime, history = self.make_runtime(model, tools=Tools([Tool(spec, tool)]))
                stale = self.record_inbound(history, "already consumed")
                history.consume_inbound("main", stale.batch_id)
                with history._db:
                    row = history._db.execute("INSERT INTO inbound_batches (session_id, source, contents)"
                                              " VALUES ('main', 'test', '[]')")
                    empty = PersistedInput(row.lastrowid)
                inputs = [stale, stale, empty, InternalInput(Message("user", "internal continuation"))]
                if has_new_user:
                    inputs.append(self.record_inbound(history, "new persisted user input"))

                runtime.submit_inputs((InternalInput(Message("user", "internal start")),))
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertIsNone(runtime.last_failure)
                self.assertEqual(authorizations, [False, has_new_user])
                self.assertEqual(len(set(run_ids)), 1)
                self.assertEqual(history.pending_inbound("main", "test"), [])
                self.assertEqual([m.content for m in model.requests[0] if m.role == "user"],
                                 ["already consumed", "internal start"])
                expected = ["already consumed", "internal start", "internal continuation"]
                if has_new_user:
                    expected.append("new persisted user input")
                self.assertEqual([m.content for m in model.requests[1] if m.role == "user"], expected)

    async def test_invalid_event_cannot_change_fifo_or_input_state(self):
        runtime, history = self.make_runtime(_Model((TextDelta("done"),)))
        scheduler = SessionSchedulers().get_or_create("main")
        scheduler.bind(runtime.submit_inputs)
        first = self.record_inbound(history, "initial")
        second = self.record_inbound(history, "later")
        invalid = Message("assistant", "invalid input")
        scheduler.session_input((first,), input_active=True)
        with self.assertRaises(ValueError):
            scheduler.session_input((second, invalid), input_active=False)
        with self.assertRaises(ValueError):
            runtime.submit_inputs((second, InternalInput(invalid)))
        scheduler.session_input((Message("user", "valid transcript"), second))
        await asyncio.wait_for(runtime.wait_idle(), 2)
        self.assertEqual(history.read_after("main"), [])

        scheduler.session_input(input_active=False)
        await asyncio.wait_for(runtime.wait_idle(), 2)
        self.assertIsNone(runtime.last_failure)
        self.assertEqual([row.message.content for row in history.read_after("main")],
                         ["initial", "valid transcript", "later", "done"])

    async def test_statement_ioerr_preserves_entire_batch_and_fifo_on_initial_input_and_continuation(self):
        for continuation in (False, True):
            with self.subTest(continuation=continuation):
                async def tool(args, context):
                    return "ok"

                spec = ToolSpec("probe", "continue", {"type": "object"})
                responses = ((ToolReady(ToolCall("c1", "probe", "{}")),), (TextDelta("done"),)) if continuation else (
                    (TextDelta("A reply"),), (TextDelta("B reply"),))
                model = _Model(*responses)
                runtime, history = self.make_runtime(model, tools=Tools([Tool(spec, tool)]))
                ref = self.record_inbound(history, "A persistent")
                internal = InternalInput(Message("user", "A internal"))
                speech = Message("user", "A speech")
                later = Message("user", "B")
                attempts = 0
                interrupted = asyncio.Event()
                append = history._append

                def fail_after_statement(session_id, messages):
                    nonlocal attempts
                    messages = tuple(messages)
                    ids = append(session_id, messages)
                    if messages == (speech,):
                        attempts += 1
                        if attempts == 1:
                            interrupted.set()
                            error = sqlite3.OperationalError("simulated IOERR")
                            error.sqlite_errorcode = sqlite3.SQLITE_IOERR
                            raise error
                    return ids

                with patch.object(history, "_append", side_effect=fail_after_statement):
                    expected_prior = []
                    if continuation:
                        runtime.submit_batch((Message("user", "tool trigger"),))
                        expected_prior = ["tool trigger"]
                    runtime.submit_inputs((ref, internal, speech))
                    runtime.submit_batch((later,))
                    await asyncio.wait_for(interrupted.wait(), 2)
                    self.assertFalse(history._db.in_transaction)
                    self.assertEqual(history.pending_inbound("main", "test"), [ref.batch_id])
                    self.assertEqual([row.message.content for row in history.read_after("main")
                                      if row.message.role == "user"], expected_prior)
                    runtime.submit_persisted_batch(ref.batch_id)
                    await asyncio.wait_for(runtime.wait_idle(), 2)

                expected_a = ["A persistent", "A internal", "A speech"]
                self.assertEqual([row.message.content for row in history.read_after("main")
                                  if row.message.role == "user"], [*expected_prior, *expected_a, "B"])
                self.assertEqual([[m.content for m in request if m.role == "user"] for request in model.requests],
                                 [expected_prior if continuation else expected_a,
                                  [*expected_prior, *expected_a, "B"]])
                self.assertEqual(attempts, 2)
                self.assertIsNone(runtime.last_failure)
                self.assertEqual(history.pending_inbound("main", "test"), [])

    async def test_permanent_or_exhausted_input_failure_blocks_queue_without_restarting(self):
        for code, expected_attempts in ((sqlite3.SQLITE_CORRUPT, 1), (sqlite3.SQLITE_IOERR, 2)):
            with self.subTest(code=code):
                model = _Model()
                runtime, history = self.make_runtime(model, retry_sqlite_locks=True)
                ref = self.record_inbound(history, "A persistent")
                speech = Message("user", "A speech")
                error = sqlite3.OperationalError("simulated write error")
                error.sqlite_errorcode = code
                append = history._append
                attempts = 0

                def fail_statement(session_id, messages):
                    nonlocal attempts
                    messages = tuple(messages)
                    ids = append(session_id, messages)
                    if messages == (speech,):
                        attempts += 1
                        raise error
                    return ids

                with (patch.object(history, "_append", side_effect=fail_statement),
                      self.assertLogs("agent_companion.runtime.engine", "ERROR") as logs):
                    runtime.submit_inputs((ref, speech))
                    runtime.submit_batch((Message("user", "later B"),))
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertEqual(attempts, expected_attempts)
                self.assertEqual(runtime.last_failure, "input_write_failed")
                self.assertIn(f"sqlite_code={code}", logs.output[0])
                self.assertEqual(history.read_after("main"), [])
                self.assertEqual(history.pending_inbound("main", "test"), [ref.batch_id])
                failed_batch = runtime._pending[0]
                self.assertEqual(failed_batch.items, (ref, speech))

                # 故障消失后重投也不能越过队首，或自行重启 worker。
                runtime.submit_persisted_batch(ref.batch_id)
                runtime.submit_batch((Message("user", "new C"),))
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertIs(runtime._pending[0], failed_batch)
                self.assertEqual(len(runtime._pending), 3)
                self.assertIsNone(runtime._worker)
                self.assertEqual(runtime.last_failure, "input_write_failed")
                self.assertEqual(history.read_after("main"), [])
                self.assertEqual(model.requests, [])

    async def test_ioerr_from_commit_is_not_replayed_because_outcome_is_unknown(self):
        model = _Model()
        runtime, history = self.make_runtime(model)
        ref = self.record_inbound(history, "A persistent")
        connection = history._db

        def commit_with_unknown_outcome():
            connection.commit()
            error = sqlite3.OperationalError("simulated IOERR reporting commit outcome")
            error.sqlite_errorcode = sqlite3.SQLITE_IOERR
            raise error

        with (patch.object(history, "_connection", wraps=connection) as wrapped,
              self.assertLogs("agent_companion.runtime.engine", "ERROR")):
            wrapped.commit.side_effect = commit_with_unknown_outcome
            runtime.submit_inputs((ref, Message("user", "A speech")))
            runtime.submit_batch((Message("user", "later B"),))
            await asyncio.wait_for(runtime.wait_idle(), 2)
            self.assertEqual(wrapped.commit.call_count, 1)
        self.assertEqual(runtime.last_failure, "input_write_failed")
        self.assertEqual([row.message.content for row in history.read_after("main")], ["A persistent", "A speech"])
        self.assertEqual(history.pending_inbound("main", "test"), [])
        self.assertEqual(model.requests, [])

    async def test_rollback_failure_isolates_the_shared_connection_even_if_close_fails(self):
        for close_fails in (False, True):
            with self.subTest(close_fails=close_fails), TemporaryDirectory() as directory:
                path = Path(directory) / "shared.sqlite3"
                with closing(History(path)) as history:
                    ref = self.record_inbound(history, "A persistent")
                    history.append("other", (Message("user", "already committed B"),))
                    model = _Model()
                    first = Runtime("main", history, model, _Channel(), retry_sqlite_locks=True)
                    second = Runtime("other", history, model, _Channel(), retry_sqlite_locks=True)
                    connection = history._db
                    wrapped = Mock(wraps=connection)
                    history._connection = wrapped
                    wrapped.rollback.side_effect = sqlite3.OperationalError("synthetic rollback failure")
                    if close_fails:
                        wrapped.close.side_effect = sqlite3.OperationalError("synthetic close failure")
                    append = history._append
                    speech = Message("user", "A speech")
                    snapshots = []

                    def fail_after_write(session_id, messages):
                        messages = tuple(messages)
                        ids = append(session_id, messages)
                        if session_id == "main" and messages == (speech,):
                            snapshots.append((connection.in_transaction,
                                              [row.message.content for row in history.read_after("main")],
                                              history.pending_inbound("main", "test")))
                            error = sqlite3.OperationalError("synthetic statement failure")
                            error.sqlite_errorcode = sqlite3.SQLITE_IOERR
                            raise error
                        return ids

                    try:
                        with (patch.object(history, "_append", side_effect=fail_after_write),
                              self.assertLogs("agent_companion.runtime.engine", "ERROR") as logs):
                            first.submit_inputs((ref, speech))
                            await asyncio.wait_for(first.wait_idle(), 2)
                            second.submit_batch((Message("user", "new B"),))
                            await asyncio.wait_for(second.wait_idle(), 2)
                        self.assertEqual(snapshots, [(True, ["A persistent", "A speech"], [])])
                        self.assertEqual((first.last_failure, second.last_failure),
                                         ("input_write_failed", "input_write_failed"))
                        self.assertEqual(model.requests, [])
                        wrapped.rollback.assert_called_once()
                        wrapped.close.assert_called_once()
                        wrapped.commit.assert_not_called()
                        self.assertIn("InputRollbackFailed", logs.output[0])
                        self.assertNotIn("synthetic", "\n".join(logs.output))
                        with self.assertRaises(sqlite3.ProgrammingError):
                            history.read_after("main")
                        if close_fails:
                            self.assertTrue(connection.in_transaction)
                    finally:
                        await asyncio.gather(first.close(), second.close())
                        connection.close()

                with closing(History(path)) as reopened:
                    self.assertEqual(reopened.read_after("main"), [])
                    self.assertEqual(reopened.pending_inbound("main", "test"), [ref.batch_id])
                    self.assertEqual([row.message.content for row in reopened.read_after("other")],
                                     ["already committed B"])

    async def test_run_internal_keeps_legacy_lock_retry_and_does_not_replay_ioerr(self):
        for code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_IOERR):
            with self.subTest(code=code), closing(History(":memory:")) as history:
                model = _Model((TextDelta("done"),))
                child = Runtime("child", history, model, None, retry_sqlite_locks=True)
                error = sqlite3.OperationalError("synthetic subtask write failure")
                error.sqlite_errorcode = code
                append = history._append
                attempts = 0

                def fail_once(session_id, messages):
                    nonlocal attempts
                    messages = tuple(messages)
                    ids = append(session_id, messages)
                    if messages == (Message("user", "prompt"),):
                        attempts += 1
                        if attempts == 1:
                            raise error
                    return ids

                try:
                    with (patch.object(history, "_append", side_effect=fail_once),
                          patch("agent_companion.runtime.sqlite_access._INPUT_LOCK_RETRIES", 0)):
                        if code == sqlite3.SQLITE_BUSY:
                            self.assertIsNone(await asyncio.wait_for(child.run_internal("run", "prompt"), 2))
                            self.assertEqual(attempts, 2)
                            self.assertEqual([row.message.content for row in history.read_after("child")],
                                             ["prompt", "done"])
                        else:
                            with self.assertLogs("agent_companion.runtime.engine", "ERROR") as logs:
                                result = await asyncio.wait_for(child.run_internal("run", "prompt"), 2)
                            self.assertEqual(result, "runtime_failed")
                            self.assertEqual(attempts, 1)
                            self.assertEqual(history.read_after("child"), [])
                            self.assertEqual(model.requests, [])
                            self.assertIn("error_type=OperationalError", logs.output[0])
                            self.assertNotIn("synthetic", logs.output[0])
                finally:
                    await child.close()

    async def test_rollback_failure_does_not_mask_subtask_cancellation(self):
        with closing(History(":memory:")) as history:
            child = Runtime("child", history, _Model(), None, retry_sqlite_locks=True)
            connection = history._db
            wrapped = Mock(wraps=connection)
            history._connection = wrapped
            wrapped.rollback.side_effect = sqlite3.OperationalError("synthetic rollback failure")
            append = history._append

            def cancelled_after_write(session_id, messages):
                append(session_id, messages)
                raise asyncio.CancelledError

            try:
                with patch.object(history, "_append", side_effect=cancelled_after_write):
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(child.run_internal("run", "prompt"), 2)
                wrapped.close.assert_called_once()
                wrapped.commit.assert_not_called()
                with self.assertRaises(sqlite3.ProgrammingError):
                    history.read_after("child")
            finally:
                await child.close()
