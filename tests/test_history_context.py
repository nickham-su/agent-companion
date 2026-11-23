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
from agent_companion.protocol import (DeliveryStatus, Image, Message, ModelError, ReasoningDelta, Receipt, Text, TextDelta,
                                      ToolCall, ToolReady, ToolSpec)
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
                db.finalize_reply(ReplyOutcome("new", "main", "run", "complete", DeliveryStatus.FULL),
                                  [Message("assistant", "new body")])
                db.finalize_reply(ReplyOutcome("empty", "main", "run", "model_failed"), [])
                expected_id = db.tail("main", 1)[0].message_id
            with closing(History(path)) as db:
                outcomes = db.outcomes("main")
                self.assertEqual(outcomes[0], ReplyOutcome("old", "main", "run", "complete", DeliveryStatus.FULL))
                self.assertEqual(outcomes[1].assistant_message_id, expected_id)
                self.assertIsNone(outcomes[2].assistant_message_id)

    def test_default_and_explicit_configuration(self):
        default = load_settings()
        self.assertIsNone(default.llm)
        self.assertEqual(default.runtime.model_timeout, 60)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "test.toml"
            path.write_text('database = "project-local.sqlite"\n[runtime]\ndelivery_timeout = 1.25\n',
                            encoding="utf-8")
            actual = load_settings(path)
            self.assertEqual(actual.database, path.parent / "project-local.sqlite")
            self.assertEqual(actual.runtime.delivery_timeout, 1.25)
            self.assertIsNone(actual.llm)
            path.write_text('[runtime]\ndelivery_timeout = nan\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "delivery_timeout"):
                load_settings(path)


class CapturingModel:
    def __init__(self):
        self.requests = []
        self.summaries = []

    async def stream(self, messages, tools, *, thinking):
        if messages and messages[0].role == "system" and str(messages[0].content).startswith("把给出的历史"):
            self.summaries.append((list(messages), tools, thinking))
            yield TextDelta("summary-material")
        else:
            self.requests.append((list(messages), tools, thinking))
            if thinking:
                yield ReasoningDelta("NEW_REASONING")
            yield TextDelta("ok")


class Sink:
    async def deliver(self, reply):
        text = "".join([part async for part in reply])
        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)


class ProjectionTest(unittest.IsolatedAsyncioTestCase):
    async def turn(self, history, model, content, **options):
        runtime = Runtime("main", history, model, Sink(), **options)
        try:
            runtime.submit_batch([Message("user", content)])
            await asyncio.wait_for(runtime.wait_idle(), 2)
            self.assertIsNone(runtime.last_failure)
        finally:
            await runtime.close()

    async def test_tool_continuation_keeps_tool_group_and_busy_inputs_under_compaction(self):
        reached, release = asyncio.Event(), asyncio.Event()

        class BusyModel(CapturingModel):
            async def stream(self, messages, tools, *, thinking):
                if not self.requests:
                    self.requests.append((list(messages), tools, thinking))
                    yield ToolReady(CALLS[0])
                    reached.set()
                    await release.wait()
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

    async def test_protected_tool_group_over_budget_fails_without_incomplete_followup(self):
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
                self.assertEqual(runtime.last_failure, "protected_context_too_large")
                self.assertEqual(len(model.requests), 1)
                self.assertEqual(model.summaries, [])
                self.assertIsNone(history.summary("main"))
                raw = [row.message for row in history.read_after("main")]
                self.assertEqual(raw[1:3], [Message("assistant", "", (CALLS[0],)),
                                             Message("tool", "b" * 600, tool_call_id=CALLS[0].id)])
                self.assertEqual(raw[-1], Message("user", "queued"))
            finally:
                await runtime.close()

    async def test_summary_failure_falls_back_with_reasoning_compatibility_and_tool_continuation(self):
        class FailingSummary(CapturingModel):
            async def stream(self, messages, tools, *, thinking):
                if not tools:
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

        async def tool(args, context):
            return "已完成测试工具"

        model = FailingSummary()
        settings = RuntimeSettings(context_budget=240, tail_budget=150)
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
                if not tools:
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

        async def tool(args, context):
            return "已完成测试工具"

        model = ToolModel()
        settings = RuntimeSettings(context_budget=240, tail_budget=150)
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

    async def test_later_summary_chunk_failure_does_not_advance_coverage(self):
        class LaterFailure(CapturingModel):
            async def stream(self, messages, tools, *, thinking):
                if messages[0].role == "system" and str(messages[0].content).startswith("把给出的历史"):
                    self.summaries.append(tuple(messages))
                    if len(self.summaries) > 1:
                        raise TimeoutError
                    yield TextDelta("未提交的中间摘要")
                else:
                    self.requests.append(tuple(messages))
                    yield TextDelta("正文继续")

        model = LaterFailure()
        settings = RuntimeSettings(context_budget=230, tail_budget=70)
        with closing(History(":memory:")) as history:
            covered, = history.append("main", [Message("user", "此前信息")])
            saved = Summary(covered, "旧摘要")
            history.save_summary("main", saved)
            history.append("main", [Message("user", "old " + "x" * 258)] * 3)
            with self.assertLogs("agent_companion.runtime.context", level="WARNING") as logs:
                await self.turn(history, model, "新文本", settings=settings)
            self.assertEqual(len(model.summaries), 2)
            self.assertIn("timeout", logs.output[0])
            self.assertEqual(history.summary("main"), saved)
            self.assertIn("旧摘要", model.requests[0][0].content)
            self.assertEqual(model.requests[0][-1].content, "新文本")

    async def test_long_generated_summary_is_saved_in_full_and_does_not_mark_more_history_covered(self):
        chunks = ("\n摘要开头\n" + "长" * 4_001, "摘要结尾\n")

        class LongSummaryModel(CapturingModel):
            async def stream(self, messages, tools, *, thinking):
                async for event in super().stream(messages, tools, thinking=thinking):
                    if isinstance(event, TextDelta) and event.text == "summary-material":
                        for chunk in chunks:
                            yield TextDelta(chunk)
                    else:
                        yield event

        settings = RuntimeSettings(context_budget=500, tail_budget=200)
        old = Message("user", "旧消息 " + "x" * 800)
        current = "新消息 " + "y" * 800
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "test.sqlite"
            with closing(History(path)) as history:
                covered, = history.append("main", [old])
                model = LongSummaryModel()
                await self.turn(history, model, current, settings=settings)
                self.assertEqual(len(model.summaries), 1)
                self.assertEqual(model.requests[0][0], [Message("user", current)])
            with closing(History(path)) as history:
                saved = Summary(covered, "".join(chunks))
                self.assertEqual(history.summary("main"), saved)
                self.assertEqual(history.read_after("main")[0].message, old)

                model = CapturingModel()
                with self.assertLogs("agent_companion.runtime.context", level="WARNING") as logs:
                    await self.turn(history, model, current, settings=settings)
                self.assertIn("summary_context_too_large", logs.output[0])
                self.assertEqual(model.summaries, [])
                self.assertEqual(history.summary("main"), saved)
                self.assertEqual(history.read_after("main", covered)[0].message, Message("user", current))

    async def test_summary_that_does_not_fit_request_is_omitted_but_not_changed_in_storage(self):
        settings = RuntimeSettings(context_budget=180, tail_budget=70)
        model = CapturingModel()
        newest = "x" * 450
        with closing(History(":memory:")) as history:
            covered, = history.append("main", [Message("user", "prior text")])
            original_summary = Summary(covered, "previous summary " * 12)
            history.save_summary("main", original_summary)
            await self.turn(history, model, newest, settings=settings)
            self.assertEqual(model.requests[0][0], [Message("user", newest)])
            self.assertEqual(history.summary("main"), original_summary)

    async def test_no_room_for_summary_uses_existing_tail_without_marking_old_history_covered(self):
        settings = RuntimeSettings(context_budget=180, tail_budget=70)
        model = CapturingModel()
        newest = "x" * 510
        with closing(History(":memory:")) as history:
            covered, prior = history.append("main", [Message("user", "already summarized"),
                                                     Message("user", "old uncompressed " * 16)])
            original_summary = Summary(covered, "stored summary")
            history.save_summary("main", original_summary)
            await self.turn(history, model, newest, settings=settings)
            self.assertEqual(model.requests[0][0], [Message("user", newest)])
            self.assertEqual(model.summaries, [])
            self.assertEqual(history.summary("main"), original_summary)
            self.assertEqual(history.read_after("main", covered)[0].message_id, prior)

    async def test_one_oversized_input_does_not_prevent_later_turns(self):
        settings = RuntimeSettings(context_budget=180, tail_budget=90)
        model = CapturingModel()
        oversized = "q" * 2000
        with closing(History(":memory:")) as history:
            runtime = Runtime("main", history, model, Sink(), settings=settings)
            try:
                runtime.submit_batch([Message("user", oversized)])
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertEqual(runtime.last_failure, "protected_context_too_large")
                self.assertEqual(model.requests, [])

                runtime.submit_batch([Message("user", "fits")])
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertIsNone(runtime.last_failure)
                self.assertEqual(history.read_after("main")[0].message.content, oversized)
                self.assertIn("原文仍可查", history.summary("main").text)
                self.assertEqual(model.requests[0][0][-1], Message("user", "fits"))
                self.assertEqual(model.summaries, [])
            finally:
                await runtime.close()

    async def test_context_compaction_keeps_blocks_raw_history_and_previous_summary(self):
        settings = RuntimeSettings(context_budget=275, tail_budget=125)
        model = CapturingModel()
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
                await self.turn(history, model, "new tail", settings=settings)
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

    async def test_thinking_switch_after_legacy_history_summarizes_not_fabricates(self):
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
            self.assertEqual(history.summary("main").covers_through, before[3].message_id)
            self.assertEqual(history.read_after("main")[:len(before)], before)
            self.assertEqual(history.read_after("main")[-1].message.reasoning, "NEW_REASONING")
            first_summary = build_request(model.summaries[0][0], (), thinking=False)
            self.assertIn("first old result", repr(first_summary))
            self.assertIn("second old result", repr(first_summary))
            self.assertNotIn("reasoning_content", repr(first_summary))
            self.assertNotIn("KEEP_NOT_SUMMARIZE", repr(first_summary))
            current = build_request(model.requests[-1][0], [SPEC], thinking=True)
            self.assertEqual(current["messages"][1]["reasoning_content"], "KEEP_NOT_SUMMARIZE")
            self.assertEqual(current["messages"][-1], {"role": "user", "content": "after switch"})
            self.assertNotIn("past reply", repr(current))
