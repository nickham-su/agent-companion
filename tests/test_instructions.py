import asyncio
from contextlib import closing
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent_companion.config import RuntimeSettings
from agent_companion.protocol import DeliveryStatus, Message, Receipt, TextDelta, ToolCall, ToolReady, ToolSpec
from agent_companion.runtime.context import _SUMMARIZE
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.instructions import MAX_AGENTS_BYTES
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.storage.sqlite import History


class Model:
    def __init__(self, respond=None):
        self.requests = []
        self.respond = respond or (lambda index, messages: [TextDelta("answer")])

    async def stream(self, messages, tools, *, thinking):
        self.requests.append(tuple(messages))
        for event in self.respond(len(self.requests), messages):
            yield event


class Sink:
    async def deliver(self, reply):
        text = "".join([chunk async for chunk in reply])
        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)


class LocalInstructionsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        parent = Path(self.enterContext(TemporaryDirectory(dir=Path.cwd())))
        self.directory = parent / "working"
        self.directory.mkdir()
        self.path = self.directory / "AGENTS.md"
        (parent / "AGENTS.md").write_text("不应向父目录读取", encoding="utf-8")
        self.history = self.enterContext(closing(History(":memory:")))

    def runtime(self, model, **options):
        with patch("agent_companion.runtime.engine.Path") as cwd:
            cwd.cwd.return_value = self.directory
            runtime = Runtime("main", self.history, model, Sink(), instructions="固定指令", **options)
        self.addAsyncCleanup(runtime.close)
        return runtime

    async def turn(self, runtime, text="新消息"):
        runtime.submit_batch([Message("user", text)])
        await asyncio.wait_for(runtime.wait_idle(), 2)

    async def test_hot_update_in_fixed_directory_and_missing_or_blank_keep_original(self):
        model = Model()
        runtime = self.runtime(model)
        await self.turn(runtime)
        self.path.write_text(" \n\t", encoding="utf-8")
        await self.turn(runtime)
        self.path.write_text("本地指令", encoding="utf-8")
        await self.turn(runtime)
        self.path.unlink()
        await self.turn(runtime)
        self.assertEqual([request[0] for request in model.requests], [
            Message("system", "固定指令"), Message("system", "固定指令"),
            Message("system", "固定指令\n\n本地指令"), Message("system", "固定指令"),
        ])
        self.assertIsNone(runtime.last_failure)

    async def test_invalid_files_only_report_fixed_diagnostics_and_never_send(self):
        model = Model()
        runtime = self.runtime(model)
        cases = [
            ("agents_md_too_large", lambda: self.path.write_bytes(b"x" * (MAX_AGENTS_BYTES + 1)), self.path.unlink),
            ("agents_md_invalid_utf8", lambda: self.path.write_bytes(b"synthetic-detail\xff"), self.path.unlink),
            ("agents_md_invalid_file", lambda: self.path.symlink_to(self.directory.parent / "AGENTS.md"), self.path.unlink),
            ("agents_md_invalid_file", self.path.mkdir, self.path.rmdir),
            ("agents_md_invalid_file", lambda: os.mkfifo(self.path), self.path.unlink),
        ]
        for diagnosis, create, remove in cases:
            with self.subTest(diagnosis=diagnosis, create=create):
                create()
                with self.assertLogs("agent_companion.runtime.engine", level="WARNING") as logs:
                    await self.turn(runtime)
                self.assertEqual(runtime.last_failure, diagnosis)
                self.assertIn(diagnosis, logs.output[0])
                self.assertNotIn("synthetic-detail", logs.output[0])
                self.assertNotIn(str(self.directory), logs.output[0])
                remove()
        self.path.write_text("synthetic-detail", encoding="utf-8")
        with (patch("agent_companion.runtime.instructions.open", create=True,
                    side_effect=PermissionError("synthetic-detail")),
              self.assertLogs("agent_companion.runtime.engine", level="WARNING") as logs):
            await self.turn(runtime)
        self.assertEqual(runtime.last_failure, "agents_md_unreadable")
        self.assertNotIn("synthetic-detail", logs.output[0])
        self.assertEqual(model.requests, [])

    async def test_hot_update_during_tool_execution_affects_continuation(self):
        self.path.write_text("第一版", encoding="utf-8")
        spec = ToolSpec("test", "本地测试", {"type": "object"})
        call = ToolCall("first", spec.name, "{}")

        async def update(args, context):
            self.path.write_text("第二版", encoding="utf-8")
            return "工具结果"

        model = Model(lambda index, _: [ToolReady(call)] if index == 1 else [TextDelta("answer")])
        runtime = self.runtime(model, tools=Tools([Tool(spec, update)]))
        await self.turn(runtime)
        self.assertIsNone(runtime.last_failure)
        self.assertEqual([messages[0].content for messages in model.requests],
                         ["固定指令\n\n第一版", "固定指令\n\n第二版"])
        self.assertEqual(model.requests[-1][-1].content, "工具结果")

    async def test_each_summary_batch_and_later_normal_request_reads_a_new_snapshot(self):
        self.path.write_text("第一版", encoding="utf-8")
        self.history.append("main", [Message("user", f"old {i} " + "x" * 258) for i in range(3)])

        def respond(index, messages):
            if messages[-1] == _SUMMARIZE:
                self.path.write_text(f"更新 {index}", encoding="utf-8")
                return [TextDelta("摘要")]
            return [TextDelta("answer")]

        model = Model(respond)
        runtime = self.runtime(model, settings=RuntimeSettings(context_budget=230, tail_budget=70))
        await self.turn(runtime)
        self.assertIsNone(runtime.last_failure)
        self.assertEqual(len(model.requests), 4)
        self.assertEqual([request[0].content for request in model.requests],
                         [f"固定指令\n\n{value}" for value in ("第一版", "更新 1", "更新 2", "更新 3")])
        self.assertTrue(all(request[-1] == _SUMMARIZE for request in model.requests[:-1]))
        self.assertEqual(model.requests[-1][-1], Message("user", "新消息"))

    async def test_revalidates_refreshed_prompt_before_normal_request_without_recompacting(self):
        self.history.append("main", [Message("user", "old " + "x" * 360)])

        def respond(index, messages):
            self.path.write_text("y" * 700, encoding="utf-8")
            return [TextDelta("摘要")]

        model = Model(respond)
        runtime = self.runtime(model, settings=RuntimeSettings(context_budget=240, tail_budget=75))
        await self.turn(runtime, "新" * 120)
        self.assertEqual(runtime.last_failure, "protected_context_too_large")
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(model.requests[0][-1], _SUMMARIZE)
        self.assertEqual(self.history.summary("main").text, "摘要")
