import asyncio
from contextlib import closing
from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agent_companion import images
from agent_companion.config import RuntimeSettings, load_settings
from agent_companion.images import UNAVAILABLE_IMAGE, image_request_bytes, save_image
from agent_companion.models.deepseek import MODEL_NAME, DeepSeek, _replay_key, build_request
from agent_companion.protocol import (AssistantPartsReady, DeliveryStatus, Image, InputUsage, Message, MessageDelivery,
                                      ModelError, PrivateData, Reasoning, Receipt, Text, TextDelta,
                                      ToolCall, ToolReady, ToolSpec)
from agent_companion.runtime.context import _SUMMARIZE
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.storage.sqlite import DatabaseFormatError, FeishuBindingError, History, ReplyOutcome, Summary
from test_message_support import PNG_BYTES, PNG_URL, assistant_message, image_workspace, reasoning_parts


SPEC = ToolSpec("test", "test tool", {"type": "object"})
CALLS = (ToolCall("c1", "test", '{"a":1}'), ToolCall("c2", "test", '{"a":2}'))


def _replay(text):
    return PrivateData(_replay_key(MODEL_NAME), text.encode("utf-8"))


class HistoryTest(unittest.TestCase):
    def test_session_ids_search_time_and_reopen(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "test.sqlite"
            with closing(History(path)) as db:
                x, = db.append("main", [Message("user", (Text("two words"), Image(".agent/images/" + "a" * 32)))])
                y, = db.append("child", [Message("user", "two words child")])
                z, = db.append("main", [assistant_message("words ending", reasoning=_replay("INTERNAL"))])
                self.assertEqual([x, y, z], sorted({x, y, z}))
                with self.assertRaises(ValueError):
                    db.save_summary("main", Summary(y, "not from main"))
                db.save_summary("main", Summary(x, "covered first"))
                with self.assertRaises(ValueError):
                    db.save_summary("main", Summary(x, "non-advancing"))
                start = db.read_after("main")[0].created_at
                self.assertEqual([r.message_id for r in db.search("main", keyword="two words", since=start)], [x])
                self.assertEqual(db.search("main", keyword="two", until=start), [])
                self.assertEqual(db.search("main", keyword="INTERNAL"), [])
                self.assertEqual(db.search("main", keyword="TOKEN"), [])
            with closing(History(path)) as reopened:
                self.assertEqual(reopened.summary("main"), Summary(x, "covered first"))
                self.assertEqual([row.message_id for row in reopened.read_after("main", x)], [z])
                self.assertEqual([row.message_id for row in reopened.tail("main", 2)], [x, z])
                self.assertEqual(reasoning_parts(reopened.read_after("main", x)[0].message), (_replay("INTERNAL"),))
                self.assertEqual(reopened.summary("child"), None)

    def test_summary_cas_requires_a_target_in_this_session_before_protected_input(self):
        with closing(History(":memory:")) as db:
            target, protected = db.append("main", [Message("user", "covered"), Message("user", "protected")])
            foreign, = db.append("child", [Message("user", "different session")])
            self.assertFalse(db.try_save_summary("main", Summary(protected, "too far"), covered=0,
                                                 protected_from=protected))
            self.assertFalse(db.try_save_summary("main", Summary(foreign, "wrong session"), covered=0,
                                                 protected_from=foreign + 1))
            self.assertIsNone(db.summary("main"))
            self.assertTrue(db.try_save_summary("main", Summary(target, "valid"), covered=0, protected_from=protected))
            self.assertEqual(db.summary("main"), Summary(target, "valid"))

    def test_model_context_snapshot_cannot_mix_old_summary_and_new_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "snapshot.sqlite"
            with closing(History(path)) as db, closing(History(path)) as other:
                db._db.execute("PRAGMA journal_mode = WAL").fetchone()
                first, second = db.append("main", [Message("user", "to summarize"), Message("user", "old tail")])
                db.save_summary("main", Summary(first, "old summary"))
                other.append("child", [Message("user", "different session")])
                read_summary = db.summary

                def advance_after_read(session_id):
                    summary = read_summary(session_id)
                    other.save_summary(session_id, Summary(second, "new summary"))
                    other.append(session_id, [Message("user", "new tail")])
                    return summary

                with patch.object(db, "summary", side_effect=advance_after_read):
                    summary, tail = db.context_snapshot("main")
                self.assertEqual(summary, Summary(first, "old summary"))
                self.assertEqual([row.message for row in tail], [Message("user", "old tail")])
                summary, tail = db.context_snapshot("main")
                self.assertEqual(summary, Summary(second, "new summary"))
                self.assertEqual([row.message for row in tail], [Message("user", "new tail")])

    def test_assistant_tool_group_and_receipt_outcome_commit_atomically(self):
        with closing(History(":memory:")) as db:
            db.append("main", [Message("user", "request")])
            outcome = ReplyOutcome("same-id", "main", "run", "complete", DeliveryStatus.ZERO)
            db.finalize_reply(outcome, [assistant_message("", (replace(CALLS[0], result="done"),))])
            with self.assertRaises(sqlite3.IntegrityError):
                db.finalize_reply(outcome, [Message("assistant", "uncommitted")])
            self.assertEqual(len(db.read_after("main")), 2)
            self.assertEqual(db.search("main", keyword="uncommitted"), [])
            assistant_id = db.read_after("main")[1].message_id
            self.assertEqual(db.outcomes("main"), [replace(outcome, assistant_message_id=assistant_id)])

    def test_rejects_legacy_and_unsupported_databases_without_changes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            legacy = Path(folder) / "legacy.sqlite"
            future = Path(folder) / "future.sqlite"
            with closing(sqlite3.connect(legacy)) as db, db:
                db.execute("CREATE TABLE feishu_private_account (singleton INTEGER PRIMARY KEY,"
                           " owner_fingerprint TEXT NOT NULL, started_ms INTEGER NOT NULL)")
                db.execute("INSERT INTO feishu_private_account VALUES (1, 'owner', 0)")
                db.execute("CREATE TABLE messages (content TEXT NOT NULL)")
                db.execute("INSERT INTO messages VALUES ('old history')")
            with closing(sqlite3.connect(future)) as db, db:
                db.execute("PRAGMA user_version = 999")

            for path in (legacy, future):
                with self.subTest(path=path.name):
                    before = path.read_bytes()
                    with self.assertRaisesRegex(DatabaseFormatError, "请保留原库并配置新的数据库路径"):
                        History(path, feishu_private_owner="owner")
                    self.assertEqual(path.read_bytes(), before)

    def test_empty_database_initializes_and_keeps_reply_outcomes_on_reopen(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "current.sqlite"
            path.touch()
            with closing(History(path, feishu_private_owner="owner")) as db:
                self.assertIsNone(db.check_feishu_private("owner"))
                db.bind_feishu_private("owner", started_ms=1)
                db.finalize_reply(ReplyOutcome("old", "main", "run", "complete", DeliveryStatus.FULL), [])
                self.assertIsNone(db.last_input_tokens("main"))
                db.finalize_reply(ReplyOutcome("new", "main", "run", "complete", DeliveryStatus.FULL,
                                               input_tokens=42),
                                  [Message("assistant", "new body")])
                db.finalize_reply(ReplyOutcome("empty", "main", "run", "model_failed"), [])
                expected_id = db.tail("main", 1)[0].message_id
            before = path.read_bytes()
            with self.assertRaises(FeishuBindingError):
                History(path, feishu_private_owner="other owner")
            self.assertEqual(path.read_bytes(), before)
            with closing(History(path, feishu_private_owner="owner")) as db:
                self.assertEqual(db.check_feishu_private("owner"), 1)
                outcomes = db.outcomes("main")
                self.assertEqual(outcomes[0], ReplyOutcome("old", "main", "run", "complete", DeliveryStatus.FULL))
                self.assertEqual(outcomes[1].assistant_message_id, expected_id)
                self.assertIsNone(outcomes[2].assistant_message_id)
                self.assertEqual(db.last_input_tokens("main"), 42)
                self.assertIsNone(db.last_input_tokens("other"))
                db.finalize_reply(ReplyOutcome("without-usage", "main", "run", "complete"), [])
                self.assertIsNone(db.last_input_tokens("main"))

    def test_default_and_explicit_configuration(self):
        default = load_settings()
        self.assertEqual(default.models, {})
        self.assertEqual(default.runtime.model_timeout, 60)
        self.assertEqual(default.voice.idle_timeout, 30)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "test.toml"
            path.write_text('database = "project-local.sqlite"\n[voice]\nidle_timeout = 1.25\n',
                            encoding="utf-8")
            actual = load_settings(path)
            self.assertEqual(actual.database, path.parent / "project-local.sqlite")
            self.assertEqual(actual.voice.idle_timeout, 1.25)
            self.assertEqual(actual.models, {})
            path.write_text('[runtime]\ndelivery_timeout = nan\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runtime 配置字段无效"):
                load_settings(path)


class CapturingModel:
    def __init__(self, input_tokens=None):
        self.requests = []
        self.summaries = []
        self.input_tokens = input_tokens

    async def stream(self, messages, tools, *, thinking, instructions=None):
        if messages[-1] == _SUMMARIZE:
            self.summaries.append((list(messages), tools, thinking))
            yield TextDelta("summary-material")
        else:
            self.requests.append((list(messages), tools, thinking))
            if thinking:
                yield AssistantPartsReady((Reasoning(_replay("NEW_REASONING")),))
            yield TextDelta("ok")
            if self.input_tokens is not None:
                yield InputUsage(self.input_tokens)


class CapturingDeepSeek(CapturingModel):
    history_replay_bytes = staticmethod(DeepSeek(None).history_replay_bytes)


class Sink:
    async def deliver(self, reply):
        text = "".join([part async for part in reply])
        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)


class ProjectionTest(unittest.IsolatedAsyncioTestCase):
    async def run_turn(self, runtime, content):
        runtime.submit_messages([MessageDelivery(Message("user", content), "trigger")])
        await asyncio.wait_for(runtime.wait_idle(), 2)
        self.assertIsNone(runtime.last_failure)

    async def finish_summary(self, runtime):
        # 等待替身摘要结束，不改变 Runtime 自身的 idle 语义。
        task = runtime._compactions._generating
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), 2)

    async def turn(self, history, model, content, *, previous_input_tokens=None, followup=None, **options):
        if previous_input_tokens is not None:
            reply = f"prior-{len(history.outcomes('main'))}"
            history.finalize_reply(ReplyOutcome(reply, "main", "fixture", "complete",
                                                input_tokens=previous_input_tokens), [])
        runtime = Runtime("main", history, model, Sink(), **options)
        try:
            await self.run_turn(runtime, content)
            await self.finish_summary(runtime)
            if followup is not None:
                await self.run_turn(runtime, followup)
        finally:
            await runtime.close()

    async def test_retries_context_reads_after_inbound_and_tool_commit_without_repeating_work(self):
        class ToolModel(CapturingModel):
            async def stream(self, messages, tools, *, thinking, instructions=None):
                if messages[-1] != _SUMMARIZE and not self.requests:
                    self.requests.append((list(messages), tools, thinking))
                    yield ToolReady(CALLS[0])
                    yield InputUsage(240)
                else:
                    async for event in super().stream(messages, tools, thinking=thinking):
                        yield event

        class CapturingSink(Sink):
            def __init__(self):
                self.replies = []

            async def deliver(self, reply):
                receipt = await super().deliver(reply)
                if receipt.confirmed:
                    self.replies.append(receipt.confirmed)
                return receipt

        model, sink = ToolModel(), CapturingSink()
        tools_called, attempts = [], []
        read_names = ("last_input_tokens", "summary", "read_after")
        faults = dict.fromkeys(read_names, 2)

        async def tool(args, context):
            tools_called.append(context.call_id)
            return "tool result"

        with closing(History(":memory:")) as history:
            history.append("main", [Message("user", "old history" * 30), Message("assistant", "old reply")])
            history.finalize_reply(ReplyOutcome("prior", "main", "prior", "complete", input_tokens=240), [])
            batch = (MessageDelivery(Message("user", "new input"), "trigger"),)
            finalize = history.finalize_reply

            def commit(outcome, rows):
                result = finalize(outcome, rows)
                if rows and rows[0].tool_calls:
                    faults.update(dict.fromkeys(read_names, 2))
                return result

            def intercept(name):
                read = getattr(history, name)

                def attempt(*args, **kwargs):
                    attempts.append(name)
                    self.assertFalse(runtime.has_pending_messages)
                    if faults[name]:
                        faults[name] -= 1
                        raise sqlite3.OperationalError("database is locked")
                    return read(*args, **kwargs)

                return patch.object(history, name, side_effect=attempt)

            with (intercept("last_input_tokens"), intercept("summary"), intercept("read_after"),
                  patch.object(history, "finalize_reply", side_effect=commit)):
                runtime = Runtime("main", history, model, sink, retry_sqlite_locks=True,
                                  settings=RuntimeSettings(context_budget=240, tail_budget=20),
                                  tools=Tools([Tool(SPEC, tool)]))
                try:
                    runtime.submit_messages(batch)
                    await asyncio.wait_for(runtime.wait_idle(), 3)
                    await self.finish_summary(runtime)
                    self.assertIsNone(runtime.last_failure)
                finally:
                    await runtime.close()
            # 续尾读锁失败会回滚整份快照，再从摘要开始重读。
            one_round = ["last_input_tokens"] * 3 + ["summary"] * 2 + ["summary", "read_after"] * 3
            self.assertEqual(attempts, one_round * 2)
            self.assertEqual(tools_called, [CALLS[0].id])
            self.assertEqual(len(model.summaries), 1)
            self.assertEqual(model.requests[0][0][-1].text, "new input")
            self.assertEqual(model.requests[1][0][-1].tool_calls[0].result, "tool result")
            self.assertEqual(sink.replies, ["ok"])
            self.assertEqual([e.message.kind for e in history.read_after("main")],
                             ["user", "assistant", "user", "assistant", "assistant"])

    async def test_tool_continuation_keeps_tool_group_and_busy_inputs_under_compaction(self):
        reached, release = asyncio.Event(), asyncio.Event()

        class BusyModel(CapturingModel):
            async def stream(self, messages, tools, *, thinking, instructions=None):
                if not self.requests:
                    self.requests.append((list(messages), tools, thinking))
                    yield ToolReady(CALLS[0])
                    reached.set()
                    await release.wait()
                    yield InputUsage(240)
                    return
                async for event in super().stream(messages, tools, thinking=thinking):
                    yield event

        async def tool(args, context):
            return "tool result " + "b" * 72

        settings = RuntimeSettings(context_budget=240, tail_budget=65)
        model = BusyModel()
        with closing(History(":memory:")) as history:
            older = [Message("user", "old material " + "x" * 247) for _ in range(2)]
            history.append("main", older)
            runtime = Runtime("main", history, model, Sink(), settings=settings, tools=Tools([Tool(SPEC, tool)]))
            try:
                runtime.submit_messages([MessageDelivery(Message("user", "starting request"), "trigger")])
                await asyncio.wait_for(reached.wait(), 2)
                runtime.submit_messages((MessageDelivery(Message("user", "new intake one"), "trigger"),
                                         MessageDelivery(Message("user", "new intake two"), "trigger")))
                release.set()
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertIsNone(runtime.last_failure)
                await self.finish_summary(runtime)
                self.assertIsNone(history.summary("main"))
                await self.run_turn(runtime, "followup")
            finally:
                await runtime.close()
            self.assertEqual(model.requests[0][0][:2], older)
            self.assertTrue(model.summaries)
            starting_input = history.read_after("main")[2]
            self.assertEqual(model.summaries[0][0], [*older, starting_input.message, _SUMMARIZE])
            self.assertEqual(history.summary("main").covers_through, starting_input.message_id)
            self.assertEqual(model.requests[1][0][:2], older)
            self.assertEqual(model.requests[2][0][0], Message("compaction_summary", "summary-material"))
            self.assertEqual(model.requests[2][0][1:], [
                assistant_message("", (replace(CALLS[0], result="tool result " + "b" * 72),)),
                Message("user", "new intake one"), Message("user", "new intake two"),
                Message("assistant", "ok"), Message("user", "followup"),
            ])

    async def test_protected_tool_group_over_threshold_is_sent_whole(self):
        class BigToolModel(CapturingModel):
            async def stream(self, messages, tools, *, thinking, instructions=None):
                if not self.requests:
                    self.requests.append((list(messages), tools, thinking))
                    yield ToolReady(CALLS[0])
                    return
                async for event in super().stream(messages, tools, thinking=thinking):
                    yield event

        async def tool(args, context):
            return "b" * 600

        model = BigToolModel()
        settings = RuntimeSettings(context_budget=140, tail_budget=65)
        with closing(History(":memory:")) as history:
            runtime = Runtime("main", history, model, Sink(), settings=settings, tools=Tools([Tool(SPEC, tool)]))
            try:
                runtime.submit_messages([MessageDelivery(Message("user", "start"), "trigger")])
                runtime.submit_messages([MessageDelivery(Message("user", "queued"), "trigger")])
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertIsNone(runtime.last_failure)
                self.assertEqual(len(model.requests), 2)
                self.assertEqual(model.summaries, [])
                self.assertIsNone(history.summary("main"))
                raw = [row.message for row in history.read_after("main")]
                self.assertEqual(raw[1:2], [assistant_message("", (replace(CALLS[0], result="b" * 600),))])
                self.assertEqual(raw[-2], Message("user", "queued"))
                self.assertEqual(model.requests[-1][0][-2:], raw[1:3])
            finally:
                await runtime.close()

    async def test_summary_failure_falls_back_with_reasoning_compatibility_and_tool_continuation(self):
        class FailingSummary(CapturingDeepSeek):
            async def stream(self, messages, tools, *, thinking, instructions=None):
                if messages[-1] == _SUMMARIZE:
                    build_request(messages, tools, thinking=thinking)
                    self.summaries.append(tuple(messages))
                    raise ModelError("http_400")
                # Validate the actual DeepSeek contract independent of the simulated response.
                build_request(messages, tools, thinking=thinking)
                self.requests.append(tuple(messages))
                yield AssistantPartsReady((Reasoning(_replay("complete reasoning field")),))
                if len(self.requests) == 1:
                    yield ToolReady(CALLS[0])
                else:
                    yield TextDelta("正文仍能回答")
                yield InputUsage(330)

        async def tool(args, context):
            return "已完成测试工具"

        model = FailingSummary()
        settings = RuntimeSettings(context_budget=330, tail_budget=55)
        with closing(History(":memory:")) as history:
            covered, = history.append("main", [Message("user", "此前信息")])
            saved = Summary(covered, "已保存摘要")
            history.save_summary("main", saved)
            history.append("main", [Message("user", "待压缩旧内容" * 20), Message("assistant", "切换之前的旧回复")])
            runtime = Runtime("main", history, model, Sink(), settings=settings, thinking=True,
                              tools=Tools([Tool(SPEC, tool)]))
            try:
                with self.assertLogs("agent_companion.runtime.compaction", level="WARNING") as logs:
                    for new in ("第一个输入", "第二个输入", "第三个输入"):
                        runtime.submit_messages([MessageDelivery(Message("user", new), "trigger")])
                        await asyncio.wait_for(runtime.wait_idle(), 2)
                        self.assertIsNone(runtime.last_failure)
                self.assertEqual(len(logs.records), 1)
                self.assertIn("http_400", logs.output[0])
                self.assertEqual(history.summary("main"), saved)
                self.assertEqual(len(model.summaries), 1)
                self.assertEqual(len(model.requests), 4)
                self.assertIn("已保存摘要", model.requests[0][0].content)
                self.assertEqual(model.requests[1][-1].tool_calls,
                                 (replace(CALLS[0], result="已完成测试工具"),))
                self.assertTrue(any(row.message.content == "切换之前的旧回复" for row in history.read_after("main")))
            finally:
                await runtime.close()

    async def test_summary_write_failure_falls_back_and_retries_after_backoff(self):
        class ToolModel(CapturingDeepSeek):
            async def stream(self, messages, tools, *, thinking, instructions=None):
                if messages[-1] == _SUMMARIZE:
                    build_request(messages, tools, thinking=thinking)
                    async for delta in super().stream(messages, tools, thinking=thinking):
                        yield delta
                    return
                payload = build_request(messages, tools, thinking=thinking)
                self.requests.append(payload)
                yield AssistantPartsReady((Reasoning(_replay("complete reasoning field")),))
                if len(self.requests) == 1:
                    yield ToolReady(CALLS[0])
                else:
                    yield TextDelta("正文仍能回答")
                yield InputUsage(330)

        async def tool(args, context):
            return "已完成测试工具"

        model = ToolModel()
        settings = RuntimeSettings(context_budget=330, tail_budget=55)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "summary.sqlite"
            with (closing(History(path)) as history, closing(sqlite3.connect(path)) as fault,
                  patch("agent_companion.runtime.compaction.monotonic", return_value=100) as clock):
                covered, = history.append("main", [Message("user", "此前信息")])
                saved = Summary(covered, "已保存摘要")
                history.save_summary("main", saved)
                history.append("main", [Message("user", "待压缩旧内容" * 20),
                                        Message("assistant", "切换之前的旧回复")])
                with fault:
                    fault.execute("CREATE TRIGGER reject_summary BEFORE UPDATE ON summaries "
                                  "BEGIN SELECT RAISE(ABORT, 'synthetic detail'); END")
                runtime = Runtime("main", history, model, Sink(), settings=settings, thinking=True,
                                  tools=Tools([Tool(SPEC, tool)]))

                async def turn(text):
                    runtime.submit_messages([MessageDelivery(Message("user", text), "trigger")])
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                    self.assertIsNone(runtime.last_failure)

                try:
                    with self.assertLogs("agent_companion.runtime.compaction", level="WARNING") as logs:
                        await turn("第一个输入")
                        await self.finish_summary(runtime)
                        self.assertEqual(history.summary("main"), saved)
                        await turn("写入被拒绝")
                        with fault:
                            fault.execute("DROP TRIGGER reject_summary")
                        clock.return_value = 129
                        await turn("退避期间继续")
                    self.assertEqual(len(logs.records), 1)
                    self.assertIn("summary_write_failed", logs.output[0])
                    self.assertNotIn("synthetic detail", logs.output[0])
                    self.assertEqual(len(model.summaries), 1)
                    self.assertEqual(history.summary("main"), saved)
                    self.assertEqual(len(model.requests), 4)
                    self.assertIn(saved.text, model.requests[0]["messages"][0]["content"])
                    self.assertNotIn("summary-material", str(model.requests))
                    assistant, result = model.requests[1]["messages"][-2:]
                    self.assertEqual(assistant["reasoning_content"], "complete reasoning field")
                    self.assertEqual(assistant["tool_calls"][0]["id"], CALLS[0].id)
                    self.assertEqual((result["role"], result["tool_call_id"]), ("tool", CALLS[0].id))
                    stored = [row.message for row in history.read_after("main")]
                    self.assertIn(replace(CALLS[0], result="已完成测试工具"),
                                  [call for message in stored for call in message.tool_calls])
                    self.assertTrue(any(call.id == CALLS[0].id for m in stored for call in m.tool_calls))
                    self.assertIn("正文仍能回答", [m.content for m in stored])
                    self.assertIn("切换之前的旧回复", [m.content for m in stored])

                    clock.return_value = 131
                    await turn("退避结束后重试")
                    await self.finish_summary(runtime)
                    self.assertEqual(len(model.summaries), 2)
                    self.assertEqual(history.summary("main"), saved)
                    await turn("应用新摘要")
                    recovered = history.summary("main")
                    self.assertGreater(recovered.covers_through, saved.covers_through)
                    self.assertEqual(recovered.text, "summary-material")
                    self.assertIn(recovered.text, model.requests[-1]["messages"][0]["content"])
                finally:
                    await runtime.close()

    async def test_long_summary_sent_in_full_and_usage_from_summary_is_ignored(self):
        text = "长" * 4000

        class LongSummaryModel(CapturingModel):
            async def stream(self, messages, tools, *, thinking, instructions=None):
                if messages[-1] == _SUMMARIZE:
                    self.summaries.append(tuple(messages))
                    yield TextDelta(text)
                    yield InputUsage(7_000)
                else:
                    self.requests.append(tuple(messages))
                    yield TextDelta("正文")

        settings = RuntimeSettings(context_budget=100, tail_budget=70)
        old = Message("user", "旧" * 800)
        model = LongSummaryModel()
        with closing(History(":memory:")) as history:
            old_id, = history.append("main", [old])
            await self.turn(history, model, "新" * 800, settings=settings, previous_input_tokens=100,
                            followup="应用摘要")
            saved = Summary(old_id, text)
            self.assertEqual(history.summary("main"), saved)
            self.assertEqual(model.requests[0][0], old)
            self.assertEqual(model.requests[1][0], Message("compaction_summary", text))
            self.assertIsNone(history.last_input_tokens("main"))
            await self.turn(history, model, "第二轮", settings=settings)
            self.assertEqual(len(model.summaries), 1)
            self.assertEqual(model.requests[2][0], model.requests[1][0])
            self.assertEqual(history.summary("main"), saved)
            self.assertEqual(history.read_after("main")[0].message, old)

    async def test_normal_request_appends_path_list_in_order_and_loads_only_current_images(self):
        workspace = image_workspace(self)
        old_picture, current_picture, excess = (save_image(workspace, PNG_BYTES) for _ in range(3))
        missing = Image(".agent/images/" + "a" * 32)
        old = Message("user", (Text("旧图前"), old_picture, Text("旧图后")))
        current = (Text("本轮图前"), missing, current_picture, missing, Text("本轮图后"), excess)
        path_list = Text(f"图片路径：\n{missing.url}\n{current_picture.url}\n{missing.url}\n{excess.url}")
        model = CapturingModel()
        with (closing(History(":memory:")) as history,
              patch.object(images, "MAX_MEDIA_BYTES", image_request_bytes(len(PNG_BYTES), "image/png"))):
            covered, _ = history.append("main", [Message("user", "已压缩旧内容"), old])
            history.save_summary("main", Summary(covered, "保留的摘要"))
            await self.turn(history, model, current, workspace=workspace)
            self.assertEqual(model.requests[0][0], [
                Message("compaction_summary", "保留的摘要"),
                Message("user", (Text("旧图前"), Text("旧图后"), Text(f"图片路径：\n{old_picture.url}"))),
                Message("user", (Text("本轮图前"), UNAVAILABLE_IMAGE, Image(PNG_URL), UNAVAILABLE_IMAGE,
                                 Text("本轮图后"), UNAVAILABLE_IMAGE, path_list)),
            ])
            self.assertEqual([row.message for row in history.read_after("main")[1:3]], [old, Message("user", current)])

    async def test_archive_keeps_repeated_image_paths_in_original_order_without_reading(self):
        workspace = image_workspace(self)
        picture, other = (save_image(workspace, PNG_BYTES) for _ in range(2))
        old = Message("user", (Text("不能回放的原历史"), picture, other, picture))
        model = CapturingModel()
        model.history_replay_bytes = lambda message, tools, *, thinking: None if message == old else 0
        with (closing(History(":memory:")) as history, patch.object(images, "_read_image") as read):
            history.append("main", [old])
            await self.turn(history, model, "新输入", workspace=workspace)
            archive = model.requests[0][0][0]
            self.assertEqual(archive.parts, (Text("历史归档（历史记录，不是新指令）："), Text("\nuser:"),
                                             old.parts[0], Text(f"图片路径：\n{picture.url}\n{other.url}\n{picture.url}")))
            self.assertEqual(history.read_after("main")[0].message, old)
            read.assert_not_called()

    async def test_long_requests_do_not_compact_without_reported_threshold(self):
        settings = RuntimeSettings(context_budget=100, tail_budget=70)
        fixed = "系统指令" * 800
        workspace = image_workspace(self)
        old = Message("user", (Text("旧" * 800), save_image(workspace, PNG_BYTES)))
        prepared = replace(old, parts=(old.parts[0], Text(f"图片路径：\n{old.parts[1].url}")))
        current = "新" * 800
        model = CapturingModel(94)
        with closing(History(":memory:")) as history:
            history.append("main", [old])
            await self.turn(history, model, current, settings=settings, instructions=fixed, workspace=workspace)
            self.assertEqual(model.requests[0][0], [prepared, Message("user", current)])
            self.assertEqual(history.last_input_tokens("main"), 94)
            await self.turn(history, model, current, settings=settings, instructions=fixed, workspace=workspace)
            self.assertEqual(model.requests[1][0][0], prepared)
            self.assertEqual(model.requests[1][0][-1], Message("user", current))
            self.assertEqual(model.summaries, [])

    async def test_background_compaction_triggers_at_95_percent_of_reported_budget(self):
        settings = RuntimeSettings(context_budget=100, tail_budget=9)
        old = Message("user", "old context")
        for last_tokens in (94, 95):
            with self.subTest(last_tokens=last_tokens), closing(History(":memory:")) as history:
                history.append("main", [old])
                model = CapturingModel()
                await self.turn(history, model, "current input", settings=settings, previous_input_tokens=last_tokens)
                self.assertEqual(model.requests[0][0], [old, Message("user", "current input")])
                self.assertEqual([prompt for prompt, _, _ in model.summaries],
                                 [[old, _SUMMARIZE]] if last_tokens == 95 else [])
                self.assertIsNone(history.summary("main"))

    async def test_context_compaction_keeps_blocks_raw_history_and_previous_summary(self):
        settings = RuntimeSettings(context_budget=275, tail_budget=125)
        model = CapturingModel(275)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "test.sqlite"
            with closing(History(path)) as history:
                history.append("main", [Message("user", "old-start " + "a" * 125)])
                first = history.read_after("main")[0]
                history.append("main", [assistant_message("", (
                                            replace(CALLS[0], result="first-result"),
                                            replace(CALLS[1], result="second-result"),
                                        ), reasoning=_replay("INTERNAL_TEXT"))])
                tool_ids = [row.message_id for row in history.read_after("main")[1:]]
                history.append("child", [Message("user", "other-session")])
                history.append("main", [Message("user", f"old-{i} " + "z" * 130) for i in range(5)])
                await self.turn(history, model, "new tail", settings=settings, previous_input_tokens=275,
                                followup="after summary")
                summary = history.summary("main")
                self.assertIsNotNone(summary)
                self.assertGreaterEqual(summary.covers_through, tool_ids[0])
                self.assertEqual(history.read_after("main")[0], first)
                self.assertEqual([row.message for row in history.read_after("main")[1:2]], [
                    assistant_message("", (
                        replace(CALLS[0], result="first-result"),
                        replace(CALLS[1], result="second-result"),
                    ), reasoning=_replay("INTERNAL_TEXT")),
                ])
                shown = model.requests[-1][0]
                self.assertEqual(shown[0], Message("compaction_summary", "summary-material"))
                self.assertIn(Message("user", "new tail"), shown)
                self.assertEqual(shown[-1], Message("user", "after summary"))
                self.assertNotIn("other-session", repr(build_request(shown, (), thinking=False)))
                prompts = [repr(build_request(prompt, tools, thinking=thinking))
                           for prompt, tools, thinking in model.summaries]
                self.assertTrue(all("INTERNAL_TEXT" not in prompt for prompt in prompts))
                self.assertTrue(any("first-result" in prompt and "second-result" in prompt for prompt in prompts))
            with closing(History(path)) as history:
                self.assertEqual(history.summary("main"), summary)
                history.append("main", [Message("user", f"another-{i} " + "c" * 140) for i in range(4)])
                other = CapturingModel()
                await self.turn(history, other, "after restart", settings=settings, followup="apply after restart")
                self.assertGreater(history.summary("main").covers_through, summary.covers_through)
                prompt = build_request(other.summaries[0][0], (), thinking=False)
                self.assertIn("summary-material", repr(prompt))
                self.assertNotIn("old-start", repr(prompt))
                self.assertEqual(history.read_after("main")[0], first)

    async def test_compaction_retains_first_reply_that_exceeds_tail_budget(self):
        question = Message("user", "old question " + "q" * 220)
        answer = Message("assistant", "old answer")
        current = Message("user", "current input")
        model = CapturingModel()
        settings = RuntimeSettings(context_budget=200, tail_budget=20)  # 当前输入 13，旧回复再占 12。
        with closing(History(":memory:")) as history:
            question_id, _ = history.append("main", [question, answer])
            await self.turn(history, model, current.content, settings=settings, previous_input_tokens=200,
                            followup="apply summary")
            self.assertEqual(history.summary("main"), Summary(question_id, "summary-material"))
            (summary_prompt, _, _), = model.summaries
            self.assertEqual(summary_prompt, [question, _SUMMARIZE])
            self.assertEqual(model.requests[0][0], [question, answer, current])
            self.assertEqual(model.requests[1][0], [
                Message("compaction_summary", "summary-material"), answer, current,
                Message("assistant", "ok"), Message("user", "apply summary"),
            ])

    async def test_deepseek_prefix_matches_native_history_and_rejects_summary_tool_calls(self):
        calls = []

        async def forbidden_tool(args, context):
            calls.append(context.call_id)
            return "unexpected"

        class ToolDuringSummary(CapturingDeepSeek):
            async def stream(self, messages, tools, *, thinking, instructions=None):
                payload = build_request(messages, tools, thinking=thinking)
                if messages[-1] == _SUMMARIZE:
                    self.summaries.append(payload)
                    yield TextDelta("must not save partial summary")
                    yield ToolReady(CALLS[0])
                else:
                    self.requests.append(payload)
                    yield AssistantPartsReady((Reasoning(_replay("response reasoning")),))
                    yield TextDelta("normal reply")
                    yield InputUsage(420)

        model = ToolDuringSummary()
        second_spec = ToolSpec("earlier", "another tool", {"type": "object"})
        tools = Tools([Tool(second_spec, forbidden_tool), Tool(SPEC, forbidden_tool)])
        settings = RuntimeSettings(context_budget=420, tail_budget=120)
        with closing(History(":memory:")) as history:
            covered, = history.append("main", [Message("user", "previous seed")])
            previous = Summary(covered, "stable saved summary")
            history.save_summary("main", previous)
            native = [Message("user", "old " + "x" * 465),
                      assistant_message("old reply", (replace(CALLS[0], result="old result"),), reasoning=_replay("prior reasoning"))]
            history.append("main", native)
            options = dict(settings=settings, instructions="system instructions", tools=tools, thinking=True)
            tail = "tail " + "t" * 140
            await self.turn(history, model, tail, **options)
            current = "current " + "c" * 225
            with self.assertLogs("agent_companion.runtime.compaction", level="WARNING") as logs:
                await self.turn(history, model, current, **options)
            self.assertEqual(len(model.summaries), 1)
            self.assertIn("summary_tool_call", logs.output[0])
            warm = model.requests[0]
            request = model.summaries[0]
            self.assertEqual(request, {**warm, "messages": warm["messages"][:-1] + [
                {"role": "user", "content": _SUMMARIZE.content},
            ]})
            self.assertNotIn(tail, repr(request))
            self.assertNotIn(current, repr(request))
            self.assertIn(tail, repr(model.requests[-1]))
            self.assertEqual(request["messages"][-3]["reasoning_content"], "prior reasoning")
            self.assertEqual([tool["function"]["name"] for tool in request["tools"]], ["earlier", "test"])
            self.assertEqual(history.summary("main"), previous)
            self.assertNotIn(_SUMMARIZE, [entry.message for entry in history.read_after("main")])
            self.assertEqual(calls, [])

    async def test_large_old_reasoning_and_tool_group_use_native_summary_projection(self):
        async def unexpected_tool(args, context):
            self.fail("historical tool must not be dispatched")

        settings = RuntimeSettings(context_budget=290, tail_budget=9)
        with closing(History(":memory:")) as history:
            seed, = history.append("main", [Message("user", "summary seed")])
            saved = Summary(seed, "saved " * 300)
            history.save_summary("main", saved)
            private = "synthetic private thinking " * 70
            group = [assistant_message("historical decision", (
                         replace(CALLS[0], result="first historical outcome"),
                         replace(CALLS[1], result="second historical outcome"),
                     ), reasoning=_replay(private))]
            end, = history.append("main", group)
            model = CapturingDeepSeek()
            await self.turn(history, model, "next", settings=settings, previous_input_tokens=290, followup="apply summary",
                            tools=Tools([Tool(SPEC, unexpected_tool)]), thinking=True)
            self.assertEqual(len(model.summaries), 1)
            messages, tools, thinking = model.summaries[0]
            self.assertTrue(thinking)
            self.assertEqual(tools, (SPEC,))
            self.assertEqual(messages, [Message("compaction_summary", saved.text), *group, _SUMMARIZE])
            self.assertEqual(history.summary("main"), Summary(end, "summary-material"))
            self.assertEqual([entry.message for entry in history.read_after("main", seed)[:1]], group)

    async def test_deepseek_summary_keeps_useful_text_matching_original_reasoning(self):
        private = "synthetic prior decision that remains relevant to the task"

        class EchoModel(CapturingDeepSeek):
            async def stream(self, messages, tools, *, thinking, instructions=None):
                if messages[-1] == _SUMMARIZE:
                    self.summaries.append(build_request(messages, tools, thinking=thinking))
                    yield TextDelta("prefix " + private)
                    yield AssistantPartsReady((Reasoning(_replay("private output not part of summary")),))
                    return
                async for event in super().stream(messages, tools, thinking=thinking):
                    yield event

        async def unexpected_tool(args, context):
            self.fail("summary check must not dispatch tools")

        model = EchoModel()
        settings = RuntimeSettings(context_budget=320, tail_budget=80)
        with closing(History(":memory:")) as history:
            covered, = history.append("main", [Message("user", "prior seed")])
            saved = Summary(covered, "existing short summary")
            history.save_summary("main", saved)
            old_end, = history.append("main", [assistant_message("x" * 445, reasoning=_replay(private))])
            registry = Tools([Tool(SPEC, unexpected_tool)])
            await self.turn(history, model, "y" * 455, settings=settings, previous_input_tokens=320,
                            tools=registry, thinking=True, followup="apply summary")
            self.assertEqual(model.summaries[0]["messages"][-2]["reasoning_content"], private)
            self.assertEqual(history.summary("main"), Summary(old_end, "prefix " + private))


    async def test_thinking_switch_keeps_tool_groups_without_reasoning(self):
        model = CapturingDeepSeek()

        async def ignored_tool(args, context):
            return "unused"

        with closing(History(":memory:")) as history:
            history.append("main", [Message("user", "past question"),
                                    assistant_message("past reply", (
                                        replace(CALLS[0], result="first old result"),
                                        replace(CALLS[1], result="second old result"),
                                    )),
                                    assistant_message("newer reply", reasoning=_replay("KEEP_NOT_SUMMARIZE"))])
            before = history.read_after("main")
            registry = Tools([Tool(SPEC, ignored_tool)])
            await self.turn(history, model, "after switch", tools=registry, thinking=True)
            self.assertIsNone(history.summary("main"))
            self.assertEqual(model.summaries, [])
            self.assertEqual(history.read_after("main")[:len(before)], before)
            self.assertEqual(reasoning_parts(history.read_after("main")[-1].message), (_replay("NEW_REASONING"),))
            current = build_request(model.requests[-1][0], [SPEC], thinking=True)
            first, assistant, output1, output2, compatible, last = current["messages"]
            self.assertEqual(first["content"], "past question")
            self.assertEqual(assistant["role"], "assistant")
            self.assertEqual(assistant["content"], "past reply")
            self.assertNotIn("reasoning_content", assistant)
            self.assertEqual([item["content"] for item in (output1, output2)], ["first old result", "second old result"])
            self.assertEqual(compatible["reasoning_content"], "KEEP_NOT_SUMMARIZE")
            self.assertEqual(last, {"role": "user", "content": "after switch"})
