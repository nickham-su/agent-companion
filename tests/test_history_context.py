import asyncio
from contextlib import closing
from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agent_companion.config import RuntimeSettings, load_settings
from agent_companion.models.deepseek import build_request
from agent_companion.protocol import (DeliveryStatus, Image, InputUsage, Message, ModelError, ReasoningDelta, Receipt, Text, TextDelta,
                                      ToolCall, ToolReady, ToolSpec)
from agent_companion.runtime.context import _SUMMARIZE
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.storage.sqlite import History, ReplyOutcome, Summary


SPEC = ToolSpec("test", "test tool", {"type": "object"})
CALLS = (ToolCall("c1", "test", '{"a":1}'), ToolCall("c2", "test", '{"a":2}'))


class HistoryTest(unittest.TestCase):
    def test_session_ids_search_time_and_reopen(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "test.sqlite"
            with closing(History(path)) as db:
                x, = db.append("main", [Message("user", (Text("two words"), Image("data:image/png;base64,TOKEN")))])
                y, = db.append("child", [Message("user", "two words child")])
                z, = db.append("main", [Message("assistant", "words ending", reasoning="INTERNAL")])
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
                self.assertEqual(reopened.tail("main", 1)[0].message.reasoning, "INTERNAL")
                self.assertEqual(reopened.summary("child"), None)

    def test_assistant_tool_group_and_receipt_outcome_commit_atomically(self):
        with closing(History(":memory:")) as db:
            db.append("main", [Message("user", "request")])
            outcome = ReplyOutcome("same-id", "main", "run", "complete", DeliveryStatus.ZERO)
            db.finalize_reply(outcome, [Message("assistant", "", (CALLS[0],)),
                                        Message("tool", "done", tool_call_id=CALLS[0].id)])
            with self.assertRaises(sqlite3.IntegrityError):
                db.finalize_reply(outcome, [Message("assistant", "uncommitted")])
            self.assertEqual(len(db.read_after("main")), 3)
            self.assertEqual(db.search("main", keyword="uncommitted"), [])
            assistant_id = db.read_after("main")[1].message_id
            self.assertEqual(db.outcomes("main"), [replace(outcome, assistant_message_id=assistant_id)])

    def test_additive_migration_keeps_old_outcomes_and_links_new_replies(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "legacy.sqlite"
            with closing(sqlite3.connect(path)) as legacy, legacy:
                legacy.execute("CREATE TABLE reply_outcomes (reply_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,"
                               " run_id TEXT NOT NULL, model_status TEXT NOT NULL, delivery_status TEXT)")
                legacy.execute("INSERT INTO reply_outcomes VALUES ('old', 'main', 'run', 'complete', 'full')")
            with closing(History(path)) as db:
                self.assertIsNone(db.last_input_tokens("main"))
                db.finalize_reply(ReplyOutcome("new", "main", "run", "complete", DeliveryStatus.FULL,
                                               input_tokens=42),
                                  [Message("assistant", "new body")])
                db.finalize_reply(ReplyOutcome("empty", "main", "run", "model_failed"), [])
                expected_id = db.tail("main", 1)[0].message_id
            with closing(History(path)) as db:
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
        self.assertIsNone(default.llm)
        self.assertEqual(default.runtime.model_timeout, 60)
        self.assertEqual(default.voice.idle_timeout, 30)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "test.toml"
            path.write_text('database = "project-local.sqlite"\n[voice]\nidle_timeout = 1.25\n',
                            encoding="utf-8")
            actual = load_settings(path)
            self.assertEqual(actual.database, path.parent / "project-local.sqlite")
            self.assertEqual(actual.voice.idle_timeout, 1.25)
            self.assertIsNone(actual.llm)
            path.write_text('[runtime]\ndelivery_timeout = nan\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runtime 配置字段无效"):
                load_settings(path)


class CapturingModel:
    def __init__(self, input_tokens=None):
        self.requests = []
        self.summaries = []
        self.input_tokens = input_tokens

    async def stream(self, messages, tools, *, thinking):
        if messages[-1:] == [_SUMMARIZE]:
            self.summaries.append((list(messages), tools, thinking))
            yield TextDelta("summary-material")
        else:
            self.requests.append((list(messages), tools, thinking))
            if thinking:
                yield ReasoningDelta("NEW_REASONING")
            yield TextDelta("ok")
            if self.input_tokens is not None:
                yield InputUsage(self.input_tokens)


class Sink:
    async def deliver(self, reply):
        text = "".join([part async for part in reply])
        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)


class ProjectionTest(unittest.IsolatedAsyncioTestCase):
    async def turn(self, history, model, content, *, previous_input_tokens=None, **options):
        if previous_input_tokens is not None:
            reply = f"prior-{len(history.outcomes('main'))}"
            history.finalize_reply(ReplyOutcome(reply, "main", "fixture", "complete",
                                                input_tokens=previous_input_tokens), [])
        runtime = Runtime("main", history, model, Sink(), **options)
        try:
            runtime.submit_batch([Message("user", content)])
            await asyncio.wait_for(runtime.wait_idle(), 2)
            self.assertIsNone(runtime.last_failure)
        finally:
            await runtime.close()

    async def test_retries_context_reads_after_inbound_and_tool_commit_without_repeating_work(self):
        class ToolModel(CapturingModel):
            async def stream(self, messages, tools, *, thinking):
                if messages[-1:] != [_SUMMARIZE] and not self.requests:
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
            history.record_inbound("main", "test", 1, [("new", Message("user", "new input"))])
            batch, = history.pending_inbound("main", "test")
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
                    self.assertEqual(history.pending_inbound("main", "test"), [])
                    if faults[name]:
                        faults[name] -= 1
                        raise sqlite3.OperationalError("database is locked")
                    return read(*args, **kwargs)

                return patch.object(history, name, side_effect=attempt)

            with (intercept("last_input_tokens"), intercept("summary"), intercept("read_after"),
                  patch.object(history, "finalize_reply", side_effect=commit)):
                runtime = Runtime("main", history, model, sink, retry_sqlite_locks=True,
                                  settings=RuntimeSettings(context_budget=240, tail_budget=65),
                                  tools=Tools([Tool(SPEC, tool)]))
                try:
                    runtime.submit_persisted_batch(batch)
                    await asyncio.wait_for(runtime.wait_idle(), 3)
                    self.assertIsNone(runtime.last_failure)
                finally:
                    await runtime.close()
            self.assertEqual(attempts, [name for name in read_names for _ in range(3)] * 2)
            self.assertEqual(tools_called, [CALLS[0].id])
            self.assertEqual(len(model.summaries), 1)
            self.assertEqual([req[0][-1].content for req in model.requests], ["new input", "tool result"])
            self.assertEqual(sink.replies, ["ok"])
            self.assertEqual([e.message.role for e in history.read_after("main")],
                             ["user", "assistant", "user", "assistant", "tool", "assistant"])

    async def test_tool_continuation_keeps_tool_group_and_busy_inputs_under_compaction(self):
        reached, release = asyncio.Event(), asyncio.Event()

        class BusyModel(CapturingModel):
            async def stream(self, messages, tools, *, thinking):
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
            last_old_id = history.append("main", older)[-1]
            runtime = Runtime("main", history, model, Sink(), settings=settings, tools=Tools([Tool(SPEC, tool)]))
            try:
                runtime.submit_batch([Message("user", "starting request")])
                await asyncio.wait_for(reached.wait(), 2)
                runtime.submit_batch([Message("user", "new intake one"), Message("user", "new intake two")])
                release.set()
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertIsNone(runtime.last_failure)
            finally:
                await runtime.close()
            self.assertEqual(model.requests[0][0][:2], older)
            self.assertTrue(model.summaries)
            self.assertEqual(history.summary("main").covers_through, last_old_id)
            self.assertEqual(model.requests[1][0][1:], [
                Message("user", "starting request"), Message("assistant", "", (CALLS[0],)),
                Message("tool", "tool result " + "b" * 72, tool_call_id=CALLS[0].id),
                Message("user", "new intake one"), Message("user", "new intake two"),
            ])

    async def test_protected_tool_group_over_threshold_is_sent_whole(self):
        class BigToolModel(CapturingModel):
            async def stream(self, messages, tools, *, thinking):
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
                runtime.submit_batch([Message("user", "start")])
                runtime.submit_batch([Message("user", "queued")])
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertIsNone(runtime.last_failure)
                self.assertEqual(len(model.requests), 2)
                self.assertEqual(model.summaries, [])
                self.assertIsNone(history.summary("main"))
                raw = [row.message for row in history.read_after("main")]
                self.assertEqual(raw[1:3], [Message("assistant", "", (CALLS[0],)),
                                             Message("tool", "b" * 600, tool_call_id=CALLS[0].id)])
                self.assertEqual(raw[-2], Message("user", "queued"))
                self.assertEqual(model.requests[-1][0][-3:], raw[1:4])
            finally:
                await runtime.close()

    async def test_summary_failure_falls_back_with_reasoning_compatibility_and_tool_continuation(self):
        class FailingSummary(CapturingModel):
            async def stream(self, messages, tools, *, thinking):
                if messages[-1:] == [_SUMMARIZE]:
                    build_request(messages, tools, thinking=thinking)
                    self.summaries.append(tuple(messages))
                    raise ModelError("http_400")
                # Validate the actual DeepSeek contract independent of the simulated response.
                build_request(messages, tools, thinking=thinking)
                self.requests.append(tuple(messages))
                yield ReasoningDelta("complete reasoning field")
                if len(self.requests) == 1:
                    yield ToolReady(CALLS[0])
                else:
                    yield TextDelta("正文仍能回答")
                yield InputUsage(330)

        async def tool(args, context):
            return "已完成测试工具"

        model = FailingSummary()
        settings = RuntimeSettings(context_budget=330, tail_budget=150)
        with closing(History(":memory:")) as history:
            covered, = history.append("main", [Message("user", "此前信息")])
            saved = Summary(covered, "已保存摘要")
            history.save_summary("main", saved)
            history.append("main", [Message("user", "待压缩旧内容" * 20), Message("assistant", "切换之前的旧回复")])
            runtime = Runtime("main", history, model, Sink(), settings=settings, thinking=True,
                              tools=Tools([Tool(SPEC, tool)]))
            try:
                with self.assertLogs("agent_companion.runtime.context", level="WARNING") as logs:
                    for new in ("第一个输入", "第二个输入", "第三个输入"):
                        runtime.submit_batch([Message("user", new)])
                        await asyncio.wait_for(runtime.wait_idle(), 2)
                        self.assertIsNone(runtime.last_failure)
                self.assertEqual(len(logs.records), 1)
                self.assertIn("http_400", logs.output[0])
                self.assertEqual(history.summary("main"), saved)
                self.assertEqual(len(model.summaries), 1)
                self.assertEqual(len(model.requests), 4)
                self.assertIn("已保存摘要", model.requests[0][0].content)
                self.assertEqual([(m.role, m.tool_call_id) for m in model.requests[1][-2:]],
                                 [("assistant", None), ("tool", CALLS[0].id)])
                self.assertEqual(model.requests[1][-1].content, "已完成测试工具")
                self.assertTrue(any(row.message.content == "切换之前的旧回复" for row in history.read_after("main")))
            finally:
                await runtime.close()

    async def test_summary_write_failure_falls_back_and_retries_after_backoff(self):
        class ToolModel(CapturingModel):
            async def stream(self, messages, tools, *, thinking):
                if messages[-1:] == [_SUMMARIZE]:
                    build_request(messages, tools, thinking=thinking)
                    async for delta in super().stream(messages, tools, thinking=thinking):
                        yield delta
                    return
                payload = build_request(messages, tools, thinking=thinking)
                self.requests.append(payload)
                yield ReasoningDelta("complete reasoning field")
                if len(self.requests) == 1:
                    yield ToolReady(CALLS[0])
                else:
                    yield TextDelta("正文仍能回答")
                yield InputUsage(330)

        async def tool(args, context):
            return "已完成测试工具"

        model = ToolModel()
        settings = RuntimeSettings(context_budget=330, tail_budget=150)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "summary.sqlite"
            with (closing(History(path)) as history, closing(sqlite3.connect(path)) as fault,
                  patch("agent_companion.runtime.context.monotonic", return_value=100) as clock):
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
                    runtime.submit_batch([Message("user", text)])
                    await asyncio.wait_for(runtime.wait_idle(), 2)
                    self.assertIsNone(runtime.last_failure)

                try:
                    with self.assertLogs("agent_companion.runtime.context", level="WARNING") as logs:
                        await turn("第一个输入")
                        self.assertEqual(history.summary("main"), saved)
                        with fault:
                            fault.execute("DROP TRIGGER reject_summary")
                        clock.return_value = 129
                        await turn("退避期间继续")
                    self.assertEqual(len(logs.records), 1)
                    self.assertIn("summary_write_failed", logs.output[0])
                    self.assertNotIn("synthetic detail", logs.output[0])
                    self.assertEqual(len(model.summaries), 1)
                    self.assertEqual(history.summary("main"), saved)
                    self.assertEqual(len(model.requests), 3)
                    self.assertIn(saved.text, model.requests[0]["messages"][0]["content"])
                    self.assertNotIn("summary-material", str(model.requests))
                    assistant, result = model.requests[1]["messages"][-2:]
                    self.assertEqual(assistant["reasoning_content"], "complete reasoning field")
                    self.assertEqual(assistant["tool_calls"][0]["id"], CALLS[0].id)
                    self.assertEqual((result["role"], result["tool_call_id"]), ("tool", CALLS[0].id))
                    stored = [row.message for row in history.read_after("main")]
                    self.assertIn(Message("tool", "已完成测试工具", tool_call_id=CALLS[0].id), stored)
                    self.assertTrue(any(m.tool_calls == (CALLS[0],) for m in stored))
                    self.assertIn("正文仍能回答", [m.content for m in stored])
                    self.assertIn("切换之前的旧回复", [m.content for m in stored])

                    clock.return_value = 131
                    await turn("退避结束后重试")
                    self.assertGreater(len(model.summaries), 1)
                    recovered = history.summary("main")
                    self.assertGreater(recovered.covers_through, saved.covers_through)
                    self.assertEqual(recovered.text, "summary-material")
                    self.assertIn(recovered.text, model.requests[-1]["messages"][0]["content"])
                finally:
                    await runtime.close()

    async def test_long_summary_sent_in_full_and_usage_from_summary_is_ignored(self):
        text = "长" * 4000

        class LongSummaryModel(CapturingModel):
            async def stream(self, messages, tools, *, thinking):
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
            await self.turn(history, model, "新" * 800, settings=settings, previous_input_tokens=100)
            saved = Summary(old_id, text)
            self.assertEqual(history.summary("main"), saved)
            self.assertEqual(model.requests[0][0], Message("system", "此前会话摘要：\n" + text))
            self.assertIsNone(history.last_input_tokens("main"))
            await self.turn(history, model, "第二轮", settings=settings)
            self.assertEqual(len(model.summaries), 1)
            self.assertEqual(model.requests[1][0], model.requests[0][0])
            self.assertEqual(history.summary("main"), saved)
            self.assertEqual(history.read_after("main")[0].message, old)

    async def test_long_requests_do_not_compact_without_reported_threshold(self):
        settings = RuntimeSettings(context_budget=100, tail_budget=70)
        fixed = "系统指令" * 800
        old = Message("user", (Text("旧" * 800), Image("data:image/png;base64,AAAA")))
        current = "新" * 800
        model = CapturingModel(99)
        with closing(History(":memory:")) as history:
            history.append("main", [old])
            await self.turn(history, model, current, settings=settings, instructions=fixed)
            self.assertEqual(model.requests[0][0], [Message("system", fixed), old, Message("user", current)])
            self.assertEqual(history.last_input_tokens("main"), 99)
            await self.turn(history, model, current, settings=settings, instructions=fixed)
            self.assertEqual(model.requests[1][0][1], old)
            self.assertEqual(model.requests[1][0][-1], Message("user", current))
            self.assertEqual(model.summaries, [])

    async def test_context_compaction_keeps_blocks_raw_history_and_previous_summary(self):
        settings = RuntimeSettings(context_budget=275, tail_budget=125)
        model = CapturingModel(275)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "test.sqlite"
            with closing(History(path)) as history:
                history.append("main", [Message("user", "old-start " + "a" * 125)])
                first = history.read_after("main")[0]
                history.append("main", [Message("assistant", "", CALLS, reasoning="INTERNAL_TEXT"),
                                        Message("tool", "first-result", tool_call_id=CALLS[0].id),
                                        Message("tool", "second-result", tool_call_id=CALLS[1].id)])
                tool_ids = [row.message_id for row in history.read_after("main")[1:]]
                history.append("child", [Message("user", "other-session")])
                history.append("main", [Message("user", f"old-{i} " + "z" * 130) for i in range(5)])
                await self.turn(history, model, "new tail", settings=settings, previous_input_tokens=275)
                summary = history.summary("main")
                self.assertIsNotNone(summary)
                self.assertNotIn(summary.covers_through, tool_ids[:2])
                self.assertEqual(history.read_after("main")[0], first)
                self.assertEqual([row.message for row in history.read_after("main")[1:4]], [
                    Message("assistant", "", CALLS, reasoning="INTERNAL_TEXT"),
                    Message("tool", "first-result", tool_call_id=CALLS[0].id),
                    Message("tool", "second-result", tool_call_id=CALLS[1].id),
                ])
                shown = model.requests[-1][0]
                self.assertEqual(shown[0], Message("system", "此前会话摘要：\nsummary-material"))
                self.assertEqual(shown[-1], Message("user", "new tail"))
                self.assertNotIn("other-session", repr(build_request(shown, (), thinking=False)))
                prompts = [repr(build_request(prompt, tools, thinking=thinking))
                           for prompt, tools, thinking in model.summaries]
                self.assertTrue(all("INTERNAL_TEXT" not in prompt for prompt in prompts))
                self.assertTrue(any("first-result" in prompt and "second-result" in prompt for prompt in prompts))
            with closing(History(path)) as history:
                self.assertEqual(history.summary("main"), summary)
                history.append("main", [Message("user", f"another-{i} " + "c" * 140) for i in range(4)])
                other = CapturingModel()
                await self.turn(history, other, "after restart", settings=settings)
                self.assertGreater(history.summary("main").covers_through, summary.covers_through)
                prompt = build_request(other.summaries[0][0], (), thinking=False)
                self.assertIn("summary-material", repr(prompt))
                self.assertNotIn("old-start", repr(prompt))
                self.assertEqual(history.read_after("main")[0], first)

    async def test_deepseek_prefix_matches_native_history_and_rejects_summary_tool_calls(self):
        calls = []

        async def forbidden_tool(args, context):
            calls.append(context.call_id)
            return "unexpected"

        class ToolDuringSummary(CapturingModel):
            async def stream(self, messages, tools, *, thinking):
                payload = build_request(messages, tools, thinking=thinking)
                if messages[-1] == _SUMMARIZE:
                    self.summaries.append(payload)
                    yield TextDelta("must not save partial summary")
                    yield ToolReady(CALLS[0])
                else:
                    self.requests.append(payload)
                    yield ReasoningDelta("response reasoning")
                    yield TextDelta("normal reply")
                    yield InputUsage(420)

        model = ToolDuringSummary()
        second_spec = ToolSpec("earlier", "another tool", {"type": "object"})
        tools = Tools([Tool(second_spec, forbidden_tool), Tool(SPEC, forbidden_tool)])
        settings = RuntimeSettings(context_budget=420, tail_budget=165)
        with closing(History(":memory:")) as history:
            covered, = history.append("main", [Message("user", "previous seed")])
            previous = Summary(covered, "stable saved summary")
            history.save_summary("main", previous)
            native = [Message("user", "old " + "x" * 465),
                      Message("assistant", "old reply", (CALLS[0],), reasoning="prior reasoning"),
                      Message("tool", "old result", tool_call_id=CALLS[0].id)]
            history.append("main", native)
            options = dict(settings=settings, instructions="system instructions", tools=tools, thinking=True)
            tail = "tail " + "t" * 140
            await self.turn(history, model, tail, **options)
            current = "current " + "c" * 225
            with self.assertLogs("agent_companion.runtime.context", level="WARNING") as logs:
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

        settings = RuntimeSettings(context_budget=290, tail_budget=95)
        with closing(History(":memory:")) as history:
            seed, = history.append("main", [Message("user", "summary seed")])
            saved = Summary(seed, "saved " * 300)
            history.save_summary("main", saved)
            private = "synthetic private thinking " * 70
            group = [Message("assistant", "historical decision", CALLS, reasoning=private),
                     Message("tool", "first historical outcome", tool_call_id=CALLS[0].id),
                     Message("tool", "second historical outcome", tool_call_id=CALLS[1].id)]
            _, _, end = history.append("main", group)
            model = CapturingModel()
            await self.turn(history, model, "next", settings=settings, previous_input_tokens=290,
                            tools=Tools([Tool(SPEC, unexpected_tool)]), thinking=True)
            self.assertEqual(len(model.summaries), 1)
            messages, tools, thinking = model.summaries[0]
            self.assertTrue(thinking)
            self.assertEqual(tools, (SPEC,))
            self.assertEqual(messages, [Message("system", "此前会话摘要：\n" + saved.text), *group, _SUMMARIZE])
            self.assertEqual(history.summary("main"), Summary(end, "summary-material"))
            self.assertEqual([entry.message for entry in history.read_after("main", seed)[:3]], group)

    async def test_private_deepseek_reasoning_verbatim_echo_is_rejected(self):
        private = "synthetic private reasoning for echo detection"

        class EchoModel(CapturingModel):
            async def stream(self, messages, tools, *, thinking):
                if messages[-1] == _SUMMARIZE:
                    self.summaries.append(build_request(messages, tools, thinking=thinking))
                    yield TextDelta("prefix " + private)
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
            history.append("main", [Message("assistant", "x" * 445, reasoning=private)])
            registry = Tools([Tool(SPEC, unexpected_tool)])
            with self.assertLogs("agent_companion.runtime.context", level="WARNING") as logs:
                await self.turn(history, model, "y" * 455, settings=settings, previous_input_tokens=320,
                                tools=registry, thinking=True)
            self.assertEqual(model.summaries[0]["messages"][-2]["reasoning_content"], private)
            self.assertIn("summary_private_echo", logs.output[0])
            self.assertNotIn(private, logs.output[0])
            self.assertEqual(history.summary("main"), saved)

    async def test_thinking_switch_projects_incompatible_legacy_history_without_compacting(self):
        model = CapturingModel()

        async def ignored_tool(args, context):
            return "unused"

        with closing(History(":memory:")) as history:
            history.append("main", [Message("user", "past question"),
                                    Message("assistant", "past reply", CALLS),
                                    Message("tool", "first old result", tool_call_id=CALLS[0].id),
                                    Message("tool", "second old result", tool_call_id=CALLS[1].id),
                                    Message("assistant", "newer reply", reasoning="KEEP_NOT_SUMMARIZE")])
            before = history.read_after("main")
            registry = Tools([Tool(SPEC, ignored_tool)])
            await self.turn(history, model, "after switch", tools=registry, thinking=True)
            self.assertIsNone(history.summary("main"))
            self.assertEqual(model.summaries, [])
            self.assertEqual(history.read_after("main")[:len(before)], before)
            self.assertEqual(history.read_after("main")[-1].message.reasoning, "NEW_REASONING")
            current = build_request(model.requests[-1][0], [SPEC], thinking=True)
            first, archived, compatible, last = current["messages"]
            self.assertEqual(first["content"], "past question")
            self.assertEqual(archived["role"], "user")
            self.assertIn("past reply", repr(archived))
            self.assertIn("first old result", repr(archived))
            self.assertIn("second old result", repr(archived))
            self.assertNotIn("reasoning_content", archived)
            self.assertEqual(compatible["reasoning_content"], "KEEP_NOT_SUMMARIZE")
            self.assertEqual(last, {"role": "user", "content": "after switch"})
