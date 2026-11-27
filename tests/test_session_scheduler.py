import asyncio
from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from agent_companion.protocol import (AsyncToolResult, DelegatedInput, DeliveryStatus, Message, MessageDelivery, Receipt,
                                      TextDelta, ToolCall, ToolReady, ToolSpec)
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.session_scheduler import SessionSchedulers
from agent_companion.storage.sqlite import History


class _Model:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    async def stream(self, messages, tools, *, thinking, instructions=None):
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

    async def test_empty_call_signal_is_valid_only_for_its_current_settled_turn(self):
        registry = SessionSchedulers()
        scheduler = registry.get_or_create("main")
        triggers, accepted = [], []
        revoke = scheduler.bind(accepted.append, trigger=triggers.append)

        def stop():
            pass

        scheduler.call_started(stop)
        self.assertTrue(scheduler.call_input(stop, (), input_active=False))
        self.assertEqual(triggers, [])
        scheduler.call_input(stop, (), input_active=True)
        scheduler.call_input(stop, (), input_active=False, empty_transcript=True)
        self.assertTrue(triggers[-1]())
        scheduler.call_input(stop, (), input_active=True)
        scheduler.call_input(stop, (), input_active=False)
        self.assertFalse(triggers[-1]())
        scheduler.call_input(stop, (), input_active=True)
        scheduler.call_input(stop, (), input_active=False, empty_transcript=True)
        self.assertTrue(triggers[-1]())
        scheduler.call_ended(stop)
        self.assertFalse(triggers[-1]())
        self.assertFalse(scheduler.call_input(stop, (), input_active=False, empty_transcript=True))
        self.assertEqual(len(triggers), 2)
        self.assertEqual(accepted, [])
        revoke()

    async def test_call_end_records_follow_arrival_order_in_a_mixed_batch(self):
        late = MessageDelivery(Message("state_event", "通话已结束 - 2026-01-20 11:14:15+00:00"), "record")
        current_end = MessageDelivery(Message("state_event", "通话已结束 - 2026-01-20 11:18:19+00:00"), "record")
        text = MessageDelivery(Message("user", "通话中补充"), "trigger")
        model = _Model((TextDelta("收到了"),))
        runtime, history = self.make_runtime(model)
        scheduler = SessionSchedulers().get_or_create("main")
        accepted, wakes = [], []

        def submit(batch):
            accepted.append(batch)
            runtime.submit_messages(batch)

        def old_stop():
            pass

        def stop():
            pass

        scheduler.bind(submit, trigger=wakes.append)
        scheduler.call_started(stop)
        logger = "agent_companion.session_scheduler"
        with self.assertNoLogs(logger, level="DEBUG"):
            scheduler.call_input(stop, (), input_active=True)
            scheduler.call_ended(old_stop, (late,))
            scheduler.session_input((text,))
        self.assertEqual(accepted, [])
        self.assertEqual(history.read_after("main"), [])
        self.assertTrue(scheduler.has_pending_messages)

        with self.assertLogs(logger, level="DEBUG") as logs:
            scheduler.call_ended(stop, (current_end,))
        self.assertEqual(len(logs.output), 3)
        self.assertIn("position=1/3 kind=state_event intent=record", logs.output[0])
        self.assertIn("position=2/3 kind=user intent=trigger", logs.output[1])
        self.assertIn("position=3/3 kind=state_event intent=record", logs.output[2])
        self.assertNotIn(text.message.text, "\n".join(logs.output))
        await asyncio.wait_for(runtime.wait_idle(), 2)
        self.assertEqual(accepted, [(late, text, current_end)])
        self.assertEqual(wakes, [])
        messages = (late.message, text.message, current_end.message)
        self.assertEqual(model.requests, [messages])
        self.assertEqual(tuple(row.message for row in history.read_after("main")),
                         (*messages, Message("assistant", "收到了")))

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
        batch = MessageDelivery(Message("user", "same deliverable"), "trigger")
        owner = first.reserve_agent()
        revoke_old = first.bind(discarded.append, owner=owner)
        first.session_input((batch, MessageDelivery(Message("user", "old route"), "trigger")), input_active=True)
        revoke_old()
        self.assertEqual(discarded, [])
        with self.assertRaises(RuntimeError):
            first.session_input((MessageDelivery(Message("user", "late old route"), "trigger"),))
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

    async def test_idle_event_flushes_one_mixed_fifo_batch_before_model_and_retries_atomically(self):
        model = _Model((TextDelta("done"),))
        runtime, history = self.make_runtime(model, retry_sqlite_locks=True)
        scheduler = SessionSchedulers().get_or_create("main")
        handoffs = []

        def submit(batch):
            handoffs.append(batch)
            runtime.submit_messages(batch)

        scheduler.bind(submit)
        first = MessageDelivery(Message("user", "first record"), "record")
        second = MessageDelivery(Message("user", "feishu text"), "trigger")
        notice = MessageDelivery(Message("async_tool_result", async_result=AsyncToolResult("notice", {})), "trigger")
        transcript = MessageDelivery(Message("user", "transcript"), "trigger")
        last = MessageDelivery(Message("user", "last record"), "record")
        scheduler.session_input((first,), input_active=True)
        scheduler.session_input((second, notice))
        self.assertTrue(scheduler.has_pending_messages)
        self.assertFalse(runtime.has_pending_messages)
        self.assertEqual(handoffs, [])
        self.assertEqual(history.read_after("main"), [])
        self.assertEqual(model.requests, [])

        append = history._append
        rolled_back = asyncio.Event()
        batch = (first, second, notice, transcript, last)
        messages = tuple(item.message for item in batch)

        def fail_after_final_insert(session_id, items, **kwargs):
            items = tuple(items)
            result = append(session_id, items, **kwargs)
            if items == messages and not rolled_back.is_set():
                rolled_back.set()
                raise sqlite3.OperationalError("database is locked")
            return result

        with (patch.object(history, "_append", side_effect=fail_after_final_insert),
              patch.object(runtime, "_prepare_run", wraps=runtime._prepare_run) as prepare):
            scheduler.session_input((transcript, last), input_active=False)
            self.assertEqual(handoffs, [batch])
            self.assertFalse(scheduler.has_pending_messages)
            await asyncio.wait_for(rolled_back.wait(), 2)
            self.assertFalse(history._db.in_transaction)
            self.assertEqual(history.read_after("main"), [])
            self.assertTrue(runtime.has_pending_messages)
            await asyncio.wait_for(runtime.wait_idle(), 2)

        self.assertIsNone(runtime.last_failure)
        self.assertFalse(runtime.has_pending_messages)
        rows = history.read_after("main")
        self.assertEqual([row.message for row in rows], [*messages, Message("assistant", "done")])
        self.assertEqual(model.requests, [messages])
        self.assertEqual(prepare.call_args.args[1], rows[0].message_id)

    async def test_record_only_writes_history_without_starting_a_run_and_later_trigger_sees_it(self):
        history = History(":memory:")
        self.addCleanup(history.close)
        agent = history.ensure_main_agent("main")
        model = _Model((TextDelta("first reply"),), (TextDelta("second reply"),))
        runtime = Runtime("main", history, model, _Channel(), agent=agent)
        self.addAsyncCleanup(runtime.close)
        recorded = MessageDelivery(Message("user", "queued record"), "record")
        runtime.submit_messages((recorded,))
        await asyncio.wait_for(runtime.wait_idle(), 2)
        self.assertEqual([row.message for row in history.read_after("main")], [recorded.message])
        self.assertEqual(history._db.execute("SELECT count(*) FROM agent_runs").fetchone()[0], 0)
        self.assertEqual(model.requests, [])

        first = MessageDelivery(Message("user", "first trigger"), "trigger")
        between = MessageDelivery(Message("user", "between runs"), "record")
        second = MessageDelivery(Message("user", "second trigger"), "trigger")
        for batch in ((first,), (between,), (second,)):
            runtime.submit_messages(batch)
        await asyncio.wait_for(runtime.wait_idle(), 2)
        self.assertEqual(model.requests, [
            (recorded.message, first.message),
            (recorded.message, first.message, Message("assistant", "first reply"), between.message, second.message),
        ])
        first_reply, second_reply = history.outcomes("main")
        self.assertNotEqual(first_reply.run_id, second_reply.run_id)
        self.assertEqual(history._db.execute("SELECT count(*) FROM agent_runs").fetchone()[0], 2)

    async def test_tool_continuation_records_waiting_batch_without_restarting(self):
        for has_waiting in (False, True):
            with self.subTest(has_waiting=has_waiting):
                model = _Model((ToolReady(ToolCall("c1", "probe", "{}")),),
                               (ToolReady(ToolCall("c2", "probe", "{}")),), (TextDelta("done"),))
                run_ids = []
                initial = MessageDelivery(Message("user", "initial"), "trigger")
                waiting = (MessageDelivery(Message("user", "waiting record"), "record"),
                           MessageDelivery(Message("user", "waiting trigger"), "trigger")) if has_waiting else ()

                async def tool(args, context):
                    run_ids.append(context.run_id)
                    if context.call_id == "c1":
                        runtime.submit_messages(waiting)
                    return "ok"

                spec = ToolSpec("probe", "probe tool continuation", {"type": "object"})
                runtime, history = self.make_runtime(model, tools=Tools([Tool(spec, tool)]))
                runtime.submit_messages((initial,))
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertIsNone(runtime.last_failure)
                self.assertEqual(len(run_ids), 2)
                self.assertEqual(run_ids[0], run_ids[1])
                self.assertFalse(runtime.has_pending_messages)
                self.assertEqual(model.requests[0], (initial.message,))
                self.assertEqual([m for m in model.requests[1] if m.kind == "user"],
                                 [initial.message, *(item.message for item in waiting)])

    async def test_invalid_event_cannot_change_fifo_or_input_state(self):
        runtime, history = self.make_runtime(_Model((TextDelta("done"),)))
        scheduler = SessionSchedulers().get_or_create("main")
        scheduler.bind(runtime.submit_messages)
        first = MessageDelivery(Message("user", "initial"), "trigger")
        second = MessageDelivery(Message("user", "later"), "trigger")
        invalid = Message("assistant", "invalid input")
        scheduler.session_input((first,), input_active=True)
        with self.assertRaises(ValueError):
            scheduler.session_input((second, invalid), input_active=False)
        with self.assertRaises(ValueError):
            runtime.submit_messages((second, invalid))
        scheduler.session_input((MessageDelivery(Message("user", "valid transcript"), "trigger"), second))
        await asyncio.wait_for(runtime.wait_idle(), 2)
        self.assertEqual(history.read_after("main"), [])

        scheduler.session_input(input_active=False)
        await asyncio.wait_for(runtime.wait_idle(), 2)
        self.assertIsNone(runtime.last_failure)
        self.assertEqual([row.message.content for row in history.read_after("main")],
                         ["initial", "valid transcript", "later", "done"])

    async def test_external_handoff_and_internal_validation_errors_still_propagate(self):
        scheduler = SessionSchedulers().get_or_create("main")
        submit = Mock(side_effect=[RuntimeError("synthetic handoff failure"), None])
        scheduler.bind(submit)
        first = MessageDelivery(Message("user", "synthetic first"), "trigger")
        second = MessageDelivery(Message("user", "synthetic second"), "trigger")
        with self.assertRaises(RuntimeError):
            scheduler.session_input((first,))
        with self.assertRaises(ValueError):
            scheduler.submit_internal(Message("assistant", "invalid input"))
        submit.assert_called_once_with((first,))

        scheduler.session_input((second,))
        submit.assert_called_with((first, second))
        scheduler.session_input()
        self.assertEqual(submit.call_count, 2)

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
                ref = MessageDelivery(Message("user", "A record"), "record")
                internal = MessageDelivery(Message("user", "A internal"), "trigger")
                speech = Message("user", "A speech")
                later = Message("user", "B")
                attempts = 0
                interrupted = asyncio.Event()
                append = history._append

                def fail_after_statement(session_id, messages, **kwargs):
                    nonlocal attempts
                    messages = tuple(messages)
                    ids = append(session_id, messages, **kwargs)
                    if messages[-1:] == (speech,):
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
                        runtime.submit_messages((MessageDelivery(Message("user", "tool trigger"), "trigger"),))
                        expected_prior = ["tool trigger"]
                    runtime.submit_messages((ref, internal, MessageDelivery(speech, "trigger")))
                    runtime.submit_messages((MessageDelivery(later, "trigger"),))
                    await asyncio.wait_for(interrupted.wait(), 2)
                    self.assertFalse(history._db.in_transaction)
                    self.assertTrue(runtime.has_pending_messages)
                    self.assertEqual([row.message.content for row in history.read_after("main")
                                      if row.message.kind == "user"], expected_prior)
                    await asyncio.wait_for(runtime.wait_idle(), 2)

                expected_a = ["A record", "A internal", "A speech"]
                self.assertEqual([row.message.content for row in history.read_after("main")
                                  if row.message.kind == "user"], [*expected_prior, *expected_a, "B"])
                self.assertEqual([[m.content for m in request if m.kind == "user"] for request in model.requests],
                                 [expected_prior if continuation else expected_a,
                                  [*expected_prior, *expected_a, "B"]])
                self.assertEqual(attempts, 2)
                self.assertIsNone(runtime.last_failure)
                self.assertFalse(runtime.has_pending_messages)

    async def test_failed_input_waits_for_new_submission_and_only_retries_when_safe(self):
        cases = ((sqlite3.SQLITE_CORRUPT, 1, False), (sqlite3.SQLITE_IOERR, 2, True),
                 (sqlite3.SQLITE_IOERR, 2, False))
        for code, expected_attempts, recovers in cases:
            with self.subTest(code=code, recovers=recovers):
                model = _Model(*([(TextDelta("reply"),)] * 3))
                runtime, history = self.make_runtime(model, retry_sqlite_locks=True)
                ref = MessageDelivery(Message("user", "A record"), "record")
                speech = Message("user", "A speech")
                error = sqlite3.OperationalError("simulated write error")
                error.sqlite_errorcode = code
                append = history._append
                attempts = 0
                failed = True

                def fail_statement(session_id, messages, **kwargs):
                    nonlocal attempts
                    messages = tuple(messages)
                    ids = append(session_id, messages, **kwargs)
                    if messages[-1:] == (speech,):
                        attempts += 1
                        if failed:
                            raise error
                    return ids

                with (patch.object(history, "_append", side_effect=fail_statement),
                      self.assertLogs("agent_companion.runtime.engine", "ERROR") as logs):
                    runtime.submit_messages((ref, MessageDelivery(speech, "trigger")))
                    runtime.submit_messages((MessageDelivery(Message("user", "later B"), "trigger"),))
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertEqual(attempts, expected_attempts)
                self.assertEqual(runtime.last_failure, "input_write_failed")
                self.assertIn(f"sqlite_code={code}", logs.output[0])
                self.assertEqual(history.read_after("main"), [])
                self.assertTrue(runtime.has_pending_messages)
                failed_batch = runtime._pending[0]
                self.assertEqual(failed_batch, (ref, MessageDelivery(speech, "trigger")))

                # 没有新的投递时不会自行反复重试。
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertIsNone(runtime._worker)
                failed = not recovers
                with patch.object(history, "_append", side_effect=fail_statement):
                    if code == sqlite3.SQLITE_IOERR and not recovers:
                        with self.assertLogs("agent_companion.runtime.engine", "ERROR"):
                            runtime.submit_messages((MessageDelivery(Message("user", "new C"), "trigger"),))
                            await asyncio.wait_for(runtime.wait_idle(), 2)
                    else:
                        runtime.submit_messages((MessageDelivery(Message("user", "new C"), "trigger"),))
                        await asyncio.wait_for(runtime.wait_idle(), 2)
                if recovers:
                    self.assertEqual(attempts, expected_attempts + 1)
                    expected = ["A record", "A speech", "later B", "new C"]
                    self.assertEqual([[m.content for m in request if m.kind == "user"] for request in model.requests],
                                     [expected[:2], expected[:3], expected])
                    self.assertIsNone(runtime.last_failure)
                else:
                    self.assertEqual(attempts, expected_attempts * (2 if code == sqlite3.SQLITE_IOERR else 1))
                    self.assertIs(runtime._pending[0], failed_batch)
                    self.assertEqual(len(runtime._pending), 3)
                    self.assertIsNone(runtime._worker)
                    self.assertEqual(runtime.last_failure, "input_write_failed")
                    self.assertEqual(history.read_after("main"), [])
                    self.assertEqual(model.requests, [])

    async def test_invalid_delegations_do_not_trigger_a_run(self):
        with closing(History(":memory:")) as history:
            parent = history.ensure_main_agent("main")
            history.start_agent_run(parent.agent_id, "parent-run", can_start=lambda: True)
            child, consumed = history.accept_delegation(parent.agent_id, "parent-run", "used", role_id="sub")
            _, interrupted = history.accept_delegation(parent.agent_id, "parent-run", "closed", agent_id=child.agent_id)
            _, elsewhere = history.accept_delegation(parent.agent_id, "parent-run", "other", role_id="sub")

            def deliver(task):
                return MessageDelivery(Message("user", "once"), "trigger", delegation=DelegatedInput(task.delegation_id))

            history.append_agent_inputs(child.agent_id, "prior", (deliver(consumed),), start=True)
            history.finish_agent_run("prior", result="done")
            history.interrupt_delegation(interrupted.delegation_id, reason="closed")
            model = _Model()
            reply = Mock(return_value=False)
            runtime = Runtime(child.session_id, history, model, None, agent=child, reply_to_parent=reply)
            try:
                record = MessageDelivery(Message("user", "record"), "record")
                runtime.submit_messages((record, *(deliver(task) for task in (consumed, interrupted, elsewhere))))
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertEqual(model.requests, [])
                reply.assert_not_called()
                self.assertEqual([row.message.text for row in history.read_after(child.session_id)], ["once", "record"])
                self.assertEqual(history.delegation(elsewhere.delegation_id), elsewhere)
                self.assertEqual(history._db.execute("SELECT count(*) FROM agent_runs").fetchone()[0], 2)
                self.assertFalse(runtime.has_pending_messages)
            finally:
                await runtime.close()

    async def test_failed_commit_is_not_replayed_when_outcome_is_unknown(self):
        for code in (sqlite3.SQLITE_IOERR, sqlite3.SQLITE_BUSY):
            with self.subTest(code=code):
                model = _Model()
                runtime, history = self.make_runtime(model, retry_sqlite_locks=True)
                record = MessageDelivery(Message("user", "A record"), "record")
                speech = MessageDelivery(Message("user", "A speech"), "trigger")
                connection = history._db

                def commit_with_unknown_outcome():
                    connection.commit()
                    wrapped.in_transaction = connection.in_transaction
                    error = sqlite3.OperationalError("synthetic commit failure")
                    error.sqlite_errorcode = code
                    raise error

                with (patch.object(history, "_connection", wraps=connection) as wrapped,
                      self.assertLogs("agent_companion.runtime.engine", "ERROR")):
                    wrapped.commit.side_effect = commit_with_unknown_outcome
                    runtime.submit_messages((record, speech))
                    runtime.submit_messages((MessageDelivery(Message("user", "later B"), "trigger"),))
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                    self.assertEqual(wrapped.commit.call_count, 1)
                self.assertEqual(runtime.last_failure, "input_write_failed")
                self.assertTrue(runtime.input_retry_forbidden)
                self.assertEqual(runtime._pending[0], (record, speech))
                runtime.submit_messages((MessageDelivery(Message("user", "new C"), "trigger"),))
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertEqual(model.requests, [])
                self.assertEqual([row.message for row in history.read_after("main")], [record.message, speech.message])

    async def test_rollback_failure_isolates_the_shared_connection_even_if_close_fails(self):
        for close_fails in (False, True):
            with self.subTest(close_fails=close_fails), TemporaryDirectory() as directory:
                path = Path(directory) / "shared.sqlite3"
                with closing(History(path)) as history:
                    ref = MessageDelivery(Message("user", "A record"), "record")
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

                    def fail_after_write(session_id, messages, **kwargs):
                        messages = tuple(messages)
                        ids = append(session_id, messages, **kwargs)
                        if session_id == "main" and messages[-1:] == (speech,):
                            snapshots.append((connection.in_transaction,
                                              [row.message.content for row in history.read_after("main")]))
                            error = sqlite3.OperationalError("synthetic statement failure")
                            error.sqlite_errorcode = sqlite3.SQLITE_IOERR
                            raise error
                        return ids

                    try:
                        with (patch.object(history, "_append", side_effect=fail_after_write),
                              self.assertLogs("agent_companion.runtime.engine", "ERROR") as logs):
                            first.submit_messages((ref, MessageDelivery(speech, "trigger")))
                            await asyncio.wait_for(first.wait_idle(), 2)
                            second.submit_messages((MessageDelivery(Message("user", "new B"), "trigger"),))
                            await asyncio.wait_for(second.wait_idle(), 2)
                        self.assertEqual(snapshots, [(True, ["A record", "A speech"])])
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
                    self.assertEqual([row.message.content for row in reopened.read_after("other")],
                                     ["already committed B"])

    async def test_child_runtime_retries_and_records_inputs_atomically_in_shared_queue(self):
        for code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_IOERR):
            with self.subTest(code=code), closing(History(":memory:")) as history:
                model = _Model((TextDelta("done"),))
                child = Runtime("child", history, model, None, retry_sqlite_locks=True)
                error = sqlite3.OperationalError("synthetic child write failure")
                error.sqlite_errorcode = code
                append = history._append
                attempts = 0

                def fail_once(session_id, messages, **kwargs):
                    nonlocal attempts
                    messages = tuple(messages)
                    ids = append(session_id, messages, **kwargs)
                    if messages == (Message("user", "prompt"),):
                        attempts += 1
                        if attempts == 1:
                            raise error
                    return ids

                try:
                    with patch.object(history, "_append", side_effect=fail_once):
                        child.submit_messages((MessageDelivery(Message("user", "prompt"), "trigger"),))
                        await asyncio.wait_for(child.wait_idle(), 2)
                    self.assertIsNone(child.last_failure)
                    self.assertEqual(attempts, 2)
                    self.assertEqual([row.message.content for row in history.read_after("child")], ["prompt", "done"])
                    self.assertEqual(len(model.requests), 1)
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

            def cancelled_after_write(session_id, messages, **kwargs):
                append(session_id, messages, **kwargs)
                raise asyncio.CancelledError

            try:
                with patch.object(history, "_append", side_effect=cancelled_after_write):
                    child.submit_messages((MessageDelivery(Message("user", "prompt"), "trigger"),))
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(child._worker, 2)
                wrapped.close.assert_called_once()
                wrapped.commit.assert_not_called()
                with self.assertRaises(sqlite3.ProgrammingError):
                    history.read_after("child")
            finally:
                await child.close()
