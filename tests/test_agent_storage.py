from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import patch
from uuid import UUID

from agent_companion.protocol import AgentReply, DelegatedInput, Message, MessageDelivery
from agent_companion.storage.sqlite import (DatabaseFormatError, FeishuCandidate, History,
                                            is_sqlite_lock_error)


def _user(text: str) -> Message:
    return Message("user", text)


def _delivery(text: str, *, delegation_id: str | None = None, intent="trigger") -> MessageDelivery:
    delegation = DelegatedInput(delegation_id) if delegation_id is not None else None
    return MessageDelivery(_user(text), intent, delegation=delegation)


def _start(history: History, agent_id: str, run_id: str) -> None:
    history.append_agent_inputs(agent_id, run_id, (_delivery("start"),), start=True)


class AgentStorageTests(unittest.TestCase):
    def test_main_identity_is_unique_across_connections_and_restart(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.db"
            with closing(History(path)) as history:
                history.bind_feishu_private("owner", started_ms=1)
            ready = Barrier(2)

            def bind():
                with closing(History(path)) as history:
                    ready.wait(timeout=5)
                    return history.ensure_main_agent("session-a")

            with ThreadPoolExecutor(max_workers=2) as pool:
                first, second = pool.submit(bind), pool.submit(bind)
                agent = first.result(timeout=10)
                self.assertIsNotNone(agent)
                self.assertEqual(agent, second.result(timeout=10))

            with closing(History(path)) as history:
                self.assertEqual((agent.session_id, agent.role_id, agent.parent_agent_id), ("session-a", "main", None))
                self.assertEqual(history.ensure_main_agent("session-a"), agent)
                self.assertEqual(history._db.execute("SELECT count(*) FROM agents").fetchone()[0], 1)
                other = history.ensure_main_agent("session-b")
                self.assertNotEqual(other.agent_id, agent.agent_id)
                self.assertEqual(history.ensure_main_agent("session-a"), agent)

    def test_short_ids_retry_real_collisions_for_main_and_child(self) -> None:
        with closing(History(":memory:")) as history, patch(
                "agent_companion.storage.sqlite.token_hex",
                side_effect=("7c3e9b2f", "7c3e9b2f", "0a0b0c0d", "0a0b0c0d", "7c3e9b2f", "deadbeef")) as generate:
            first = history.ensure_main_agent("first")
            parent = history.ensure_main_agent("parent")
            self.assertEqual(history.ensure_main_agent("first"), first)
            _start(history, parent.agent_id, "parent-run")
            child, delegation = history.accept_delegation(parent.agent_id, "parent-run", "call", role_id="review")
            self.assertEqual((first.agent_id, parent.agent_id, child.agent_id),
                             ("a-7c3e9b2f", "a-0a0b0c0d", "a-deadbeef"))
            self.assertRegex(child.session_id, r"^child-[0-9a-f]{32}$")
            self.assertEqual(history.agent(child.agent_id), child)
            self.assertEqual(history.delegation(delegation.delegation_id).agent_id, child.agent_id)
            self.assertEqual([call.args for call in generate.call_args_list], [(4,)] * 6)

    def test_short_id_retry_is_bounded_and_session_conflicts_are_not_retried(self) -> None:
        with closing(History(":memory:")) as history:
            parent = history.ensure_main_agent("parent")
            other = history.ensure_main_agent("other")
            _start(history, parent.agent_id, "parent-run")
            with patch("agent_companion.storage.sqlite.token_hex",
                       return_value=other.agent_id.removeprefix("a-")) as generate:
                with self.assertRaisesRegex(sqlite3.IntegrityError, r"agents\.agent_id"):
                    history.accept_delegation(parent.agent_id, "parent-run", "collision", role_id="review")
                self.assertGreater(generate.call_count, 1)
                self.assertLessEqual(generate.call_count, 8)
            self.assertEqual(history._db.execute("SELECT count(*) FROM agents").fetchone()[0], 2)
            self.assertEqual(history._db.execute("SELECT count(*) FROM delegations").fetchone()[0], 0)

            child, _ = history.accept_delegation(parent.agent_id, "parent-run", "success", role_id="review")
            occupied_session = UUID(hex=child.session_id.removeprefix("child-"))
            available_id = next(value for value in ("11111111", "22222222", "33333333", "44444444")
                                if f"a-{value}" not in (parent.agent_id, other.agent_id, child.agent_id))
            with (patch("agent_companion.storage.sqlite.uuid4", return_value=occupied_session),
                  patch("agent_companion.storage.sqlite.token_hex", return_value=available_id) as generate):
                with self.assertRaisesRegex(sqlite3.IntegrityError, r"agents\.session_id"):
                    history.accept_delegation(parent.agent_id, "parent-run", "session-conflict", role_id="review")
                generate.assert_called_once_with(4)
            self.assertIsNone(history.agent(f"a-{available_id}"))
            self.assertEqual(history._db.execute("SELECT count(*) FROM delegations").fetchone()[0], 1)

    def test_new_database_uses_v5_without_reply_tracking_or_normalized_inbox(self) -> None:
        with closing(History(":memory:")) as history:
            self.assertEqual(history._db.execute("PRAGMA user_version").fetchone()[0], 5)
            self.assertEqual([row["name"] for row in history._db.execute("PRAGMA table_info(agent_runs)")],
                             ["run_id", "agent_id", "status", "started_at", "finished_at", "result", "error"])
            objects = {row["name"] for row in history._db.execute("SELECT name FROM sqlite_master")}
            self.assertFalse(objects & {"inbound_batches", "inbound_pending", "inbound_feishu_peers"})
            self.assertTrue({"inbound_windows", "inbound_seen", "feishu_raw_batches", "feishu_recovery",
                             "feishu_private_account"} <= objects)

    def test_old_database_is_not_changed(self) -> None:
        for version in (1, 2, 4):
            with self.subTest(version=version), TemporaryDirectory(dir=Path.cwd()) as directory:
                path = Path(directory) / "old.db"
                with closing(sqlite3.connect(path)) as db:
                    db.executescript("CREATE TABLE legacy (value TEXT); INSERT INTO legacy VALUES ('fixture');"
                                     f" PRAGMA user_version = {version};")
                before = path.read_bytes()
                with self.assertRaisesRegex(DatabaseFormatError, "保留原库"):
                    History(path)
                self.assertEqual(path.read_bytes(), before)

    def test_feishu_window_and_all_users_roll_back_together(self) -> None:
        with closing(History(":memory:")) as history:
            history.feishu_inbound_window(100)
            groups = {"ou_a": [("om_a", 101_000, "text", '"A"')], "ou_b": [("om_b", 101_000, "text", '"B"')]}
            retry = [FeishuCandidate("om_missing", "ou_a", "digest", "text", 101)]
            history._db.execute("CREATE TRIGGER reject_second BEFORE INSERT ON feishu_raw_batches"
                                " WHEN NEW.peer_open_id = 'ou_b' BEGIN SELECT RAISE(ABORT, 'synthetic'); END")
            with self.assertRaises(sqlite3.IntegrityError):
                history.record_feishu_window(110, now=110, verified=groups, discarded=("om_deleted",), retry=retry)
            self.assertEqual(history.feishu_inbound_window(0), (100, 100))
            self.assertFalse(history.feishu_inbound_seen("om_a"))
            self.assertFalse(history.feishu_inbound_seen("om_deleted"))
            self.assertIsNone(history.feishu_recovery_candidate("om_missing"))
            self.assertEqual(history.pending_feishu_users(), [])
            history._db.execute("DROP TRIGGER reject_second")
            history.record_feishu_window(110, now=110, verified=groups, discarded=("om_deleted",), retry=retry)
            self.assertEqual(history.feishu_inbound_window(0), (100, 110))
            self.assertEqual(history.pending_feishu_users(), ["ou_a", "ou_b"])
            self.assertTrue(history.feishu_inbound_seen("om_deleted"))
            self.assertFalse(history.feishu_inbound_seen("om_missing"))
            self.assertEqual(history.due_feishu_candidates(119, 1), [])
            self.assertEqual(history.due_feishu_candidates(120, 1), retry)
            history.record_feishu_window(120, now=120, verified=groups, discarded=(), retry=())
            (raw_a, _) = history.next_feishu_raw("ou_a")
            history._db.execute("CREATE TRIGGER reject_confirmation BEFORE DELETE ON feishu_raw_batches"
                                " BEGIN SELECT RAISE(ABORT, 'synthetic'); END")
            with self.assertRaises(sqlite3.IntegrityError):
                history.finish_feishu_raw("ou_a", raw_a)
            self.assertEqual(history.read_feishu_raw("ou_a", raw_a), groups["ou_a"])
            history._db.execute("DROP TRIGGER reject_confirmation")
            with self.assertRaises(ValueError):
                history.finish_feishu_raw("ou_b", raw_a)
            for peer in groups:
                raw_id, has_media = history.next_feishu_raw(peer)
                self.assertFalse(has_media)
                self.assertEqual(history.read_feishu_raw(peer, raw_id), groups[peer])
                history.finish_feishu_raw(peer, raw_id)
                self.assertIsNone(history.next_feishu_raw(peer))
                with self.assertRaises(ValueError):
                    history.finish_feishu_raw(peer, raw_id)
            self.assertEqual(history.pending_feishu_users(), [])

    def test_delegation_create_reuse_and_direct_ownership(self) -> None:
        with closing(History(":memory:")) as history:
            first = history.ensure_main_agent("first")
            second = history.ensure_main_agent("second")
            _start(history, first.agent_id, "first-run")
            _start(history, second.agent_id, "second-run")
            child, initial = history.accept_delegation(first.agent_id, "first-run", "call-a", role_id="review")
            sibling, _ = history.accept_delegation(first.agent_id, "first-run", "call-b", role_id="review")
            reused, followup = history.accept_delegation(first.agent_id, "first-run", "call-c", agent_id=child.agent_id)
            self.assertEqual(reused, child)
            self.assertNotEqual(child.agent_id, sibling.agent_id)
            self.assertNotEqual(child.session_id, sibling.session_id)
            self.assertEqual((child.role_id, child.parent_agent_id), ("review", first.agent_id))
            self.assertNotEqual(initial.delegation_id, followup.delegation_id)
            self.assertEqual(history.delegation(initial.delegation_id), initial)
            self.assertEqual((initial.parent_agent_id, initial.parent_run_id, initial.tool_call_id),
                             (first.agent_id, "first-run", "call-a"))
            self.assertEqual((initial.status, initial.run_id, followup.run_id), ("queued", None, None))
            self.assertEqual(history._db.execute("SELECT count(*) FROM agent_runs").fetchone()[0], 2)

            _start(history, child.agent_id, "child-run")
            for target in (child.agent_id, first.agent_id, sibling.agent_id, "missing"):
                self.assertIsNone(history.accept_delegation(child.agent_id, "child-run", "rejected", agent_id=target))
            self.assertIsNone(history.accept_delegation(second.agent_id, "second-run", "rejected",
                                                        agent_id=child.agent_id))
            with self.assertRaises(ValueError):
                history.accept_delegation(first.agent_id, "second-run", "rejected", role_id="review")
            self.assertEqual(history._db.execute("SELECT count(*) FROM agents").fetchone()[0], 4)
            self.assertEqual(history._db.execute("SELECT count(*) FROM delegations").fetchone()[0], 3)

    def test_child_creation_rolls_back_on_failed_acceptance(self) -> None:
        with closing(History(":memory:")) as history:
            parent = history.ensure_main_agent("parent")
            _start(history, parent.agent_id, "parent-run")
            history._db.execute("CREATE TRIGGER reject_delegation BEFORE INSERT ON delegations"
                                " BEGIN SELECT RAISE(ABORT, 'injected'); END")
            with self.assertRaises(sqlite3.IntegrityError):
                history.accept_delegation(parent.agent_id, "parent-run", "call", role_id="review")
            self.assertEqual(history._db.execute("SELECT count(*) FROM agents").fetchone()[0], 1)
            self.assertEqual(history._db.execute("SELECT count(*) FROM delegations").fetchone()[0], 0)

    def test_retry_after_sqlite_commit_lock_does_not_duplicate_acceptance(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.db"
            with closing(History(path)) as history, closing(sqlite3.connect(path, timeout=0)) as reader:
                history.bind_feishu_private("owner", started_ms=1)  # 运行时连接遇锁时不在 SQLite 内等待。
                parent = history.ensure_main_agent("parent")
                _start(history, parent.agent_id, "parent-run")
                reader.execute("BEGIN")
                reader.execute("SELECT * FROM agents").fetchall()
                with self.assertRaises(sqlite3.OperationalError) as raised:
                    history.accept_delegation(parent.agent_id, "parent-run", "call", role_id="review")
                self.assertTrue(is_sqlite_lock_error(raised.exception))
                self.assertFalse(history._db.in_transaction)
                reader.rollback()
                history.accept_delegation(parent.agent_id, "parent-run", "call", role_id="review")
                self.assertEqual(history._db.execute("SELECT count(*) FROM agents").fetchone()[0], 2)
                self.assertEqual(history._db.execute("SELECT count(*) FROM delegations").fetchone()[0], 1)

    def test_ordered_input_batch_attaches_multiple_delegations_to_one_run(self) -> None:
        with closing(History(":memory:")) as history:
            parent = history.ensure_main_agent("parent")
            _start(history, parent.agent_id, "parent-run")
            child, first = history.accept_delegation(parent.agent_id, "parent-run", "a", role_id="review")
            _, second = history.accept_delegation(parent.agent_id, "parent-run", "b", agent_id=child.agent_id)
            first_input = _delivery("first", delegation_id=first.delegation_id)
            second_input = _delivery("second", delegation_id=second.delegation_id)
            ids, started = history.append_agent_inputs(
                child.agent_id, "child-run", (_delivery("record", intent="record"), first_input), start=True)
            self.assertTrue(started)
            self.assertEqual(len(ids), 2)
            ids, started = history.append_agent_inputs(
                child.agent_id, "child-run", (_delivery("memory"), second_input, first_input), start=False)
            self.assertFalse(started)
            self.assertEqual(len(ids), 2)
            self.assertEqual([item.message.content for item in history.tail(child.session_id, 10)],
                             ["record", "first", "memory", "second"])
            self.assertEqual(history.run_delegation_ids("child-run"), (first.delegation_id, second.delegation_id))
            self.assertEqual(history.delegation(second.delegation_id).status, "consumed")
            self.assertEqual(history.delegation(second.delegation_id).run_id, "child-run")

    def test_async_reply_is_an_ordinary_message_in_the_input_transaction(self) -> None:
        with closing(History(":memory:")) as history:
            parent = history.ensure_main_agent("parent")
            notice = AgentReply("child-agent", "unknown-run", error="model_failed").to_history_message()
            history._db.execute("CREATE TRIGGER reject_reply BEFORE INSERT ON messages"
                                " BEGIN SELECT RAISE(ABORT, 'injected'); END")
            with self.assertRaises(sqlite3.IntegrityError):
                history.append_agent_inputs(parent.agent_id, "reply-run", (MessageDelivery(notice, "trigger"),), start=True)
            self.assertIsNone(history.agent_run("reply-run"))
            self.assertEqual(history.read_after(parent.session_id), [])
            history._db.execute("DROP TRIGGER reject_reply")

            (message_id,), started = history.append_agent_inputs(
                parent.agent_id, "reply-run", (MessageDelivery(notice, "trigger"),), start=True)
            self.assertTrue(started)
            (stored,) = history.read_after(parent.session_id)
            self.assertEqual((stored.message_id, stored.message), (message_id, notice))

    def test_child_requires_reply_payload_but_main_may_finish_silently(self) -> None:
        with closing(History(":memory:")) as history:
            parent = history.ensure_main_agent("parent")
            _start(history, parent.agent_id, "parent-run")
            child, _ = history.accept_delegation(parent.agent_id, "parent-run", "call", role_id="review")
            _start(history, child.agent_id, "child-run")
            history.finish_agent_run("parent-run")
            main = history.agent_run("parent-run")
            self.assertEqual((main.status, main.result, main.error), ("completed", None, None))

            running = history.agent_run("child-run")
            for payload in ({}, {"result": ""}, {"result": " \n\u3000"}, {"error": ""}, {"error": "\t"}):
                with self.subTest(payload=payload):
                    with self.assertRaises(ValueError):
                        history.finish_agent_run("child-run", **payload)
                    self.assertEqual(history.agent_run("child-run"), running)
            self.assertFalse(history._db.in_transaction)

            history.finish_agent_run("child-run", result="reply")
            completed = history.agent_run("child-run")
            self.assertEqual((completed.status, completed.result, completed.error), ("completed", "reply", None))

    def test_input_failure_keeps_queue_records_available_for_retry(self) -> None:
        with closing(History(":memory:")) as history:
            parent = history.ensure_main_agent("parent")
            _start(history, parent.agent_id, "parent-run")
            child, task = history.accept_delegation(parent.agent_id, "parent-run", "a", role_id="review")
            items = (_delivery("before", intent="record"), _delivery("delegate", delegation_id=task.delegation_id),
                     _delivery("reject", intent="record"))
            history._db.execute("CREATE TRIGGER reject_input BEFORE INSERT ON messages"
                                " WHEN json_extract(NEW.parts, '$[0].text') = 'reject' BEGIN SELECT RAISE(ABORT, 'injected'); END")
            with self.assertRaises(sqlite3.IntegrityError):
                history.append_agent_inputs(child.agent_id, "run", items, start=True)
            self.assertIsNone(history.agent_run("run"))
            self.assertEqual(history.delegation(task.delegation_id), task)
            self.assertEqual(history.tail(child.session_id, 10), [])
            history._db.execute("DROP TRIGGER reject_input")
            ids, started = history.append_agent_inputs(child.agent_id, "run", items, start=True)
            self.assertTrue(started)
            self.assertEqual(len(ids), 3)

    def test_shutdown_and_restart_only_interrupt_work_not_identities(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.db"
            with closing(History(path)) as history:
                parent = history.ensure_main_agent("parent")
                _start(history, parent.agent_id, "parent-run")
                child, queued = history.accept_delegation(parent.agent_id, "parent-run", "a", role_id="review")
                _, rejected = history.accept_delegation(parent.agent_id, "parent-run", "b", agent_id=child.agent_id)
                history.interrupt_delegation(rejected.delegation_id, reason="closed")
                _start(history, child.agent_id, "child-run")
                history.interrupt_agent_work(agent_id=parent.agent_id, reason="shutdown")
                self.assertEqual(history.agent_run("parent-run").error, "shutdown")
                self.assertEqual(history.agent_run("child-run").status, "running")

            with closing(History(path)) as history:
                history.interrupt_agent_work(reason="restart")
                self.assertEqual(history.ensure_main_agent("parent"), parent)
                self.assertEqual(history.agent(child.agent_id), child)
                self.assertEqual(history.delegation(queued.delegation_id).error, "restart")
                self.assertEqual(history.delegation(rejected.delegation_id).error, "closed")
                self.assertEqual(history.agent_run("child-run").status, "interrupted")
                with self.assertRaises(ValueError):
                    history.finish_agent_run("child-run", result="late")
                _start(history, parent.agent_id, "next-parent-run")
                reused, followup = history.accept_delegation(parent.agent_id, "next-parent-run", "retry",
                                                             agent_id=child.agent_id)
                self.assertEqual(reused, child)
                history.append_agent_inputs(reused.agent_id, "next-run",
                                            (_delivery("retry", delegation_id=followup.delegation_id),), start=True)
                history.finish_agent_run("next-run", result="complete")
                self.assertEqual(history.agent_run("next-run").result, "complete")
                self.assertEqual([item.message.content for item in history.tail(child.session_id, 10)], ["start", "retry"])
