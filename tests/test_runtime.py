import asyncio
from dataclasses import replace
import json
import sqlite3
import unittest
from unittest.mock import patch

from agent_companion.config import RuntimeSettings
from agent_companion.images import save_image
from agent_companion.models.deepseek import MODEL_NAME, DeepSeek, _replay_key, build_request
from agent_companion.protocol import (AssistantPartsReady, DeliveryStatus, Message, MessageDelivery, ModelError,
                                      PrivateData, Reasoning, Receipt,
                                      ReplyDelivery, Text, TextDelta, ToolCall,
                                      ToolReady, ToolSpec)
from agent_companion.runtime.context import PreparedContext
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.silent_end import SILENT_END_TOOL, SILENT_END_TOOL_NAME
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.storage.sqlite import History, Summary
from test_message_support import PNG_BYTES, assistant_message, image_workspace, reasoning_parts, tool_calls


SPEC = ToolSpec("test", "模拟工具", {"type": "object", "properties": {}})
CALL = ToolCall("call-1", "test", "{}")
SILENT_END_CALL = ToolCall("silent", SILENT_END_TOOL_NAME, "{}")


class ScriptModel:
    def __init__(self, *scripts):
        self.scripts = iter(scripts)
        self.requests = []

    async def stream(self, messages, tools, *, thinking, instructions=None):
        self.requests.append((list(messages), tools, thinking))
        async for event in next(self.scripts)():
            yield event


def events(*items):
    async def stream():
        for item in items:
            yield item
    return stream


class Channel:
    def __init__(self, receipt=None):
        self.receipt = receipt
        self.replies = []

    async def deliver(self, reply):
        text = "".join([part async for part in reply])
        self.replies.append((reply.info, text))
        if self.receipt is not None:
            return self.receipt(reply.info, text)
        return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)


class RuntimeTest(unittest.IsolatedAsyncioTestCase):
    def setup_runtime(self, model, channel, **kw):
        history = History(":memory:")
        self.addCleanup(history.close)
        runtime = Runtime("main", history, model, channel, **kw)
        self.addAsyncCleanup(runtime.close)
        return runtime, history

    async def idle(self, runtime):
        await asyncio.wait_for(runtime.wait_idle(), 2)
        self.assertIsNone(runtime.last_failure)

    async def test_silent_end_settles_batch_at_round_limit_and_leaves_input_for_new_run(self):
        started, release = asyncio.Event(), asyncio.Event()
        executed = []

        async def tool(args, context):
            executed.append((context.call_id, context.run_id))
            started.set()
            await release.wait()
            return '{"error":"not_active"}'

        again = replace(SILENT_END_CALL, id="silent-again")
        model = ScriptModel(events(ToolReady(SILENT_END_CALL), ToolReady(CALL), ToolReady(again)),
                            events(TextDelta("下一轮正文")))
        channel = Channel()
        runtime, history = self.setup_runtime(model, channel, tools=Tools([SILENT_END_TOOL, Tool(SPEC, tool)]),
                                              settings=RuntimeSettings(max_tool_rounds=1))
        runtime.submit_messages([MessageDelivery(Message("user", "首轮输入"), "trigger")])
        await asyncio.wait_for(started.wait(), 2)
        self.assertEqual(channel.replies, [])
        runtime.submit_messages([MessageDelivery(Message("user", "排队输入"), "trigger")])
        release.set()
        await self.idle(runtime)

        first, second = history.outcomes("main")
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(executed, [(CALL.id, first.run_id)])
        self.assertEqual((first.model_status, first.delivery_status), ("complete", None))
        self.assertEqual([(info.run_id, text) for info, text in channel.replies], [(second.run_id, "下一轮正文")])
        first_input, settled, queued_input = model.requests[1][0]
        self.assertEqual([first_input.content, queued_input.content], ["首轮输入", "排队输入"])
        self.assertEqual([(call.id, json.loads(call.result)) for call in settled.tool_calls],
                         [(SILENT_END_CALL.id, {"status": "accepted"}), (CALL.id, {"error": "not_active"}),
                          (again.id, {"status": "accepted"})])

    async def test_silent_end_with_only_whitespace_never_selects_delivery(self):
        class VoicePossible(Channel):
            selections = 0

            def select_delivery(self, reply):
                self.selections += 1
                reply.select_voice_delivery()
                return ReplyDelivery(self.deliver)

        model = ScriptModel(events(TextDelta(" \t"), ToolReady(SILENT_END_CALL), TextDelta("\n\u3000")))
        channel = VoicePossible()
        runtime, history = self.setup_runtime(model, channel, tools=Tools([SILENT_END_TOOL]))
        runtime.submit_messages([MessageDelivery(Message("user", "输入"), "trigger")])
        await self.idle(runtime)

        self.assertEqual((len(model.requests), channel.selections, channel.replies), (1, 0, []))
        outcome, = history.outcomes("main")
        self.assertEqual((outcome.model_status, outcome.delivery_status), ("complete", None))
        _, settled = history.read_after("main")
        self.assertEqual(settled.message.content, "")
        self.assertEqual(json.loads(settled.message.tool_calls[0].result), {"status": "accepted"})

    async def test_whitespace_only_delivery_depends_on_silent_end_permission(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                model = ScriptModel(events(TextDelta(" \n")))
                channel = Channel()
                tools = Tools([SILENT_END_TOOL] if enabled else [])
                runtime, history = self.setup_runtime(model, channel, tools=tools)
                runtime.submit_messages([MessageDelivery(Message("user", "输入"), "trigger")])
                await self.idle(runtime)
                self.assertEqual([text for _, text in channel.replies], [] if enabled else [" \n"])
                outcome, = history.outcomes("main")
                self.assertEqual(outcome.delivery_status, None if enabled else DeliveryStatus.FULL)

    async def test_whitespace_prefix_streams_with_first_visible_text_and_keeps_voice_progress(self):
        handoffs = []
        first_chunk = "\n\t流式正文"
        full_text = first_chunk + "\n后续"

        class Voice(Channel):
            def select_delivery(self, reply):
                reply.select_voice_delivery()
                return ReplyDelivery(self.deliver)

            async def deliver(self, reply):
                first = await anext(reply)
                reply.report_progress(reply.info.reply_id, 2, first)
                handoffs.append((first, reply.progress))
                rest = "".join([chunk async for chunk in reply])
                self.replies.append((reply.info, first + rest))
                return Receipt(reply.info.reply_id, DeliveryStatus.FULL, first + rest)

        async def stream():
            yield ToolReady(SILENT_END_CALL)
            yield TextDelta("\n")
            yield TextDelta("\t")
            self.assertEqual(handoffs, [])
            yield TextDelta("流式正文")
            # 首段必须已进入语音通道，且进度验证接收的是原始模型文本前缀。
            self.assertEqual(handoffs, [(first_chunk, (2, first_chunk))])
            yield TextDelta("\n后续")

        model = ScriptModel(stream)
        channel = Voice()
        runtime, history = self.setup_runtime(model, channel, tools=Tools([SILENT_END_TOOL]))
        runtime.submit_messages([MessageDelivery(Message("user", "输入"), "trigger")])
        await self.idle(runtime)

        self.assertEqual([text for _, text in channel.replies], [full_text])
        _, settled = history.read_after("main")
        self.assertEqual(settled.message.content, full_text)
        self.assertEqual(json.loads(settled.message.tool_calls[0].result),
                         {"status": "ignored", "reason": "text_present"})

    async def test_text_wins_in_either_frame_order_without_buffering_or_follow_up(self):
        for silent_first in (True, False):
            with self.subTest(silent_first=silent_first):
                streamed = asyncio.Event()

                class ObservingChannel:
                    async def deliver(self, reply):
                        self.text = ""
                        async for chunk in reply:
                            self.text += chunk
                            streamed.set()
                        return Receipt(reply.info.reply_id, DeliveryStatus.ZERO)

                async def stream():
                    if silent_first:
                        yield ToolReady(SILENT_END_CALL)
                    yield TextDelta("流式正文")
                    await asyncio.wait_for(streamed.wait(), 1)
                    if not silent_first:
                        yield ToolReady(SILENT_END_CALL)

                channel = ObservingChannel()
                model = ScriptModel(stream)
                runtime, history = self.setup_runtime(model, channel, tools=Tools([SILENT_END_TOOL]))
                runtime.submit_messages([MessageDelivery(Message("user", "输入"), "trigger")])
                await self.idle(runtime)
                self.assertEqual(channel.text, "流式正文")
                self.assertEqual(len(model.requests), 1)
                self.assertEqual(history.outcomes("main")[0].delivery_status, DeliveryStatus.ZERO)
                settled, = tool_calls(row.message for row in history.read_after("main"))
                self.assertEqual(json.loads(settled.result), {"status": "ignored", "reason": "text_present"})

    async def test_ignored_silent_end_preserves_other_tool_follow_up(self):
        async def tool(args, context):
            return "普通工具结果"

        model = ScriptModel(events(ToolReady(SILENT_END_CALL), TextDelta("实时正文"), ToolReady(CALL)),
                            events(TextDelta("续轮正文")))
        channel = Channel()
        runtime, history = self.setup_runtime(model, channel, tools=Tools([SILENT_END_TOOL, Tool(SPEC, tool)]))
        runtime.submit_messages([MessageDelivery(Message("user", "输入"), "trigger")])
        await self.idle(runtime)
        self.assertEqual([text for _, text in channel.replies], ["实时正文", "续轮正文"])
        self.assertEqual(len({outcome.run_id for outcome in history.outcomes("main")}), 1)
        ignored, ordinary = model.requests[1][0][-1].tool_calls
        self.assertEqual(json.loads(ignored.result), {"status": "ignored", "reason": "text_present"})
        self.assertEqual(ordinary.result, "普通工具结果")

    async def test_invalid_or_unregistered_silent_end_does_not_end_run(self):
        for tools, arguments, error in (([], "{}", "unknown_tool"), ([SILENT_END_TOOL], "[]", "invalid_arguments"),
                                        ([SILENT_END_TOOL], '{"extra":null}', "invalid_arguments")):
            with self.subTest(arguments=arguments, error=error):
                call = replace(SILENT_END_CALL, arguments=arguments)
                model = ScriptModel(events(ToolReady(call)), events(TextDelta("正常后续")))
                channel = Channel()
                runtime, _ = self.setup_runtime(model, channel, tools=Tools(tools))
                runtime.submit_messages([MessageDelivery(Message("user", "输入"), "trigger")])
                await self.idle(runtime)
                self.assertEqual([text for _, text in channel.replies], ["正常后续"])
                settled, = model.requests[1][0][-1].tool_calls
                self.assertEqual(json.loads(settled.result), {"error": error})

    async def test_silent_end_does_not_hide_model_failure(self):
        async def broken_stream():
            yield ToolReady(SILENT_END_CALL)
            raise ModelError("truncated_stream")

        model = ScriptModel(broken_stream)
        runtime, history = self.setup_runtime(model, Channel(), tools=Tools([SILENT_END_TOOL]))
        runtime.submit_messages([MessageDelivery(Message("user", "输入"), "trigger")])
        await asyncio.wait_for(runtime.wait_idle(), 2)
        self.assertEqual(runtime.last_failure, "model_failed")
        self.assertEqual(history.outcomes("main")[0].model_status, "model_failed")
        settled, = tool_calls(row.message for row in history.read_after("main"))
        self.assertEqual(json.loads(settled.result), {"status": "ignored", "reason": "model_incomplete"})

    async def test_silent_end_does_not_hide_a_batch_result_write_failure(self):
        async def tool(args, context):
            return "写入失败的工具结果"

        model = ScriptModel(events(ToolReady(SILENT_END_CALL), ToolReady(CALL)))
        channel = Channel()
        runtime, history = self.setup_runtime(model, channel, tools=Tools([SILENT_END_TOOL, Tool(SPEC, tool)]))
        persist = history.record_tool_result

        def fail_one_result(session_id, reply_id, call_id, result):
            if call_id == CALL.id:
                raise sqlite3.OperationalError("synthetic write failure")
            persist(session_id, reply_id, call_id, result)

        with patch.object(history, "record_tool_result", side_effect=fail_one_result):
            runtime.submit_messages([MessageDelivery(Message("user", "输入"), "trigger")])
            await asyncio.wait_for(runtime.wait_idle(), 2)
        self.assertEqual(runtime.last_failure, "runtime_failed")
        self.assertEqual((len(model.requests), channel.replies), (1, []))

    async def test_trigger_run_can_use_configured_tool_and_protects_summary_only_context(self):
        usage = []

        async def tool(args, context):
            usage.append(context.call_id)
            return "工具结果"

        model = ScriptModel(events(ToolReady(CALL)), events(TextDelta("新回复")))
        runtime, history = self.setup_runtime(model, Channel(), tools=Tools([Tool(SPEC, tool)]))
        runtime.trigger_run()
        await self.idle(runtime)
        self.assertEqual(model.requests, [])
        (old_id,) = history.append("main", [Message("user", "已压缩的输入")])
        history.save_summary("main", Summary(old_id, "合成摘要"))
        runtime.trigger_run()
        await self.idle(runtime)
        self.assertEqual(usage, [CALL.id])
        (first, _, _), (second, _, _) = model.requests
        self.assertEqual(first, [Message("compaction_summary", "合成摘要")])
        self.assertEqual([message.kind for message in second], ["compaction_summary", "assistant"])
        self.assertEqual([row.message.kind for row in history.tail("main", 10)],
                         ["user", "assistant", "assistant"])

    async def test_real_pending_input_replaces_or_prevents_trigger(self):
        for before_trigger in (False, True):
            with self.subTest(before_trigger=before_trigger):
                model = ScriptModel(events(TextDelta("仅真实输入的回复")))
                runtime, history = self.setup_runtime(model, Channel())
                message = Message("user", "真正输入")
                history.append("main", [Message("assistant", "已有历史")])
                if before_trigger:
                    runtime.submit_messages((MessageDelivery(message, "trigger"),))
                runtime.trigger_run()
                if not before_trigger:
                    runtime.submit_messages((MessageDelivery(message, "trigger"),))
                await self.idle(runtime)
                self.assertEqual(len(model.requests), 1)
                self.assertEqual(model.requests[0][0][-1], message)

    async def test_real_input_wins_during_trigger_run_record_lock_retry(self):
        model, channel = ScriptModel(events(TextDelta("真实输入的回复"))), Channel()
        self.addCleanup((history := History(":memory:")).close)
        agent = history.ensure_main_agent("main")
        runtime = Runtime("main", history, model, channel, agent=agent, retry_sqlite_locks=True)
        self.addAsyncCleanup(runtime.close)
        history.append("main", [Message("assistant", "已有历史")])
        attempted = asyncio.Event()
        failed_run_ids = []

        def locked(agent_id, run_id, *, can_start):
            failed_run_ids.append(run_id)
            attempted.set()
            error = sqlite3.OperationalError("synthetic lock")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise error

        with patch.object(history, "start_agent_run", side_effect=locked):
            runtime.trigger_run()
            await asyncio.wait_for(attempted.wait(), 2)
            real = Message("user", "在重试期间到达")
            runtime.submit_messages((MessageDelivery(real, "trigger"),))
            await self.idle(runtime)
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(model.requests[0][0][-1], real)
        (skipped,) = failed_run_ids
        self.assertIsNone(history.agent_run(skipped))
        ((info, _),) = channel.replies
        self.assertEqual(history.agent_run(info.run_id).status, "completed")

    async def test_empty_wake_rechecks_binding_and_real_input_after_context_assembly(self):
        for incoming in (False, True):
            with self.subTest(incoming=incoming):
                workspace = image_workspace(self)
                picture = save_image(workspace, PNG_BYTES)
                model = ScriptModel(events(TextDelta("真实输入回复")))
                history = History(":memory:")
                self.addCleanup(history.close)
                agent = history.ensure_main_agent("main")
                history.append("main", [Message("user", (picture,))])
                runtime = Runtime("main", history, model, Channel(), agent=agent, workspace=workspace)
                self.addAsyncCleanup(runtime.close)
                assembling, release = asyncio.Event(), asyncio.Event()
                call_binding = original_call = object()
                assemble = PreparedContext.assemble

                async def pause_assembly(context, workspace):
                    if not assembling.is_set():
                        assembling.set()
                        await release.wait()
                    return await assemble(context, workspace)

                with (patch.object(PreparedContext, "assemble", pause_assembly),
                      patch("agent_companion.images._read_image") as read,
                      self.assertLogs("agent_companion.runtime.engine", level="DEBUG") as logs):
                    runtime.trigger_run(lambda: call_binding is original_call)
                    try:
                        await asyncio.wait_for(assembling.wait(), 2)
                        pending_run, = history._db.execute("SELECT run_id FROM agent_runs").fetchone()
                        real = Message("user", "真正输入")
                        if incoming:
                            runtime.submit_messages((MessageDelivery(real, "trigger"),))
                        else:
                            call_binding = object()
                    finally:
                        release.set()
                    await self.idle(runtime)
                self.assertEqual(history.agent_run(pending_run).status, "completed")
                self.assertEqual(len(history.outcomes("main")), int(incoming))
                self.assertEqual(len(model.requests), int(incoming))
                self.assertEqual(sum("model request started" in line for line in logs.output), int(incoming))
                read.assert_not_called()
                if incoming:
                    self.assertEqual(model.requests[0][0][-1], real)
                    self.assertEqual(model.requests[0][0][0].parts, (Text(f"图片路径：\n{picture.url}"),))

    async def test_reentrant_trigger_guard_does_not_steal_queued_input(self):
        model = ScriptModel(events(TextDelta("处理真实输入")))
        runtime, history = self.setup_runtime(model, Channel())
        history.append("main", [Message("assistant", "已有历史")])
        real = Message("user", "检查通话时同步交接")

        def handoff():
            runtime.submit_messages((MessageDelivery(real, "trigger"),))
            return True

        runtime.trigger_run(handoff)
        await self.idle(runtime)
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(model.requests[0][0][-1], real)

    async def test_failed_reply_commit_discards_queued_and_late_wake_without_blocking_real_input(self):
        model = ScriptModel(events(TextDelta("没有可靠结算")), events(TextDelta("真实输入的回复")))
        runtime, history = self.setup_runtime(model, Channel())
        original = Message("user", "最初输入")

        def fail_commit(outcome, rows, **kwargs):
            runtime.trigger_run()
            self.assertIsNotNone(runtime._trigger)
            raise RuntimeError("synthetic reply commit failure")

        with patch.object(history, "finalize_reply", side_effect=fail_commit):
            runtime.submit_messages((MessageDelivery(original, "trigger"),))
            await asyncio.wait_for(runtime.wait_idle(), 2)
        self.assertEqual(runtime.last_failure, "runtime_failed")
        self.assertEqual(history.outcomes("main"), [])
        self.assertEqual(len(model.requests), 1)
        runtime.trigger_run()  # ASR 也可能晚于失败的回执写入。
        await asyncio.wait_for(runtime.wait_idle(), 2)
        self.assertEqual(len(model.requests), 1)
        real = Message("user", "下一个实际输入")
        runtime.submit_messages((MessageDelivery(real, "trigger"),))
        await self.idle(runtime)
        self.assertEqual([request[0] for request in model.requests], [[original], [original, real]])
        self.assertEqual(len(history.outcomes("main")), 1)

    async def test_close_cancels_active_and_queued_tools_before_waiting_for_child_model(self):
        model_closing, model_release, model_stopped = (asyncio.Event() for _ in range(3))
        tool_active, tool_closing, tool_release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        queued_started, queued_closing = asyncio.Event(), asyncio.Event()
        lock = asyncio.Lock()

        class Model:
            async def stream(self, messages, tools, *, thinking, instructions=None):
                try:
                    yield ToolReady(CALL)
                    yield ToolReady(ToolCall("queued", SPEC.name, "{}"))
                    await asyncio.Future()
                finally:
                    model_closing.set()
                    await model_release.wait()
                    model_stopped.set()

        async def tool(args, context):
            if context.call_id == "queued":
                queued_started.set()
                try:
                    async with lock:
                        self.fail("queued tool executed after shutdown")
                finally:
                    queued_closing.set()
            async with lock:
                tool_active.set()
                try:
                    await asyncio.Future()
                finally:
                    tool_closing.set()
                    await tool_release.wait()

        runtime, _ = self.setup_runtime(Model(), None, tools=Tools([Tool(SPEC, tool)]))
        runtime.submit_messages((MessageDelivery(Message("user", "synthetic"), "trigger"),))
        await asyncio.wait_for(tool_active.wait(), 2)
        await asyncio.wait_for(queued_started.wait(), 2)
        first = asyncio.create_task(runtime.close())
        second = None
        try:
            await asyncio.wait_for(model_closing.wait(), 2)
            await asyncio.wait_for(tool_closing.wait(), 2)
            await asyncio.wait_for(queued_closing.wait(), 2)
            second = asyncio.create_task(runtime.close())
            first.cancel()
            await asyncio.sleep(0)
            first.cancel()
            self.assertFalse(runtime.idle)
            model_release.set()
            await asyncio.wait_for(model_stopped.wait(), 2)
            await asyncio.sleep(0)
            self.assertFalse(runtime.idle)
            self.assertFalse(first.done())
            self.assertFalse(second.done())
            tool_release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(first, 2)
            await asyncio.wait_for(second, 2)
            self.assertTrue(runtime.idle)
        finally:
            tool_release.set()
            model_release.set()
            await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)

    async def test_mixed_receipt_and_tool_complete_in_both_orders(self):
        for first in ("tool", "receipt"):
            with self.subTest(first=first):
                tool_started, tool_release, tool_sent = (asyncio.Event() for _ in range(3))
                receipt_started, receipt_release, receipt_sent = (asyncio.Event() for _ in range(3))
                model_release = asyncio.Event()

                async def mixed():
                    yield TextDelta("first")
                    yield ToolReady(CALL)
                    await model_release.wait()
                    yield TextDelta(" more")

                async def tool(args, context):
                    self.assertEqual((args, context.session_id, context.call_id), ({}, "main", CALL.id))
                    tool_started.set()
                    await tool_release.wait()
                    tool_sent.set()
                    return "tool says done"

                class GatedChannel(Channel):
                    async def deliver(self, reply):
                        if len(self.replies):
                            return await super().deliver(reply)
                        prefix = await anext(reply)
                        self.replies.append((reply.info, prefix))
                        receipt_started.set()
                        await receipt_release.wait()
                        receipt_sent.set()
                        return Receipt(reply.info.reply_id, DeliveryStatus.PARTIAL, prefix)

                model = ScriptModel(mixed, events(TextDelta("second")))
                channel = GatedChannel()
                runtime, history = self.setup_runtime(model, channel, tools=Tools([Tool(SPEC, tool)]))
                runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
                await asyncio.wait_for(asyncio.gather(tool_started.wait(), receipt_started.wait()), 2)
                runtime.submit_messages((MessageDelivery(Message("user", "two"), "trigger"),
                                         MessageDelivery(Message("user", "three"), "trigger")))
                if first == "tool":
                    tool_release.set()
                    await tool_sent.wait()
                    self.assertFalse(receipt_sent.is_set())
                    model_release.set()
                else:
                    receipt_release.set()
                    await receipt_sent.wait()
                    self.assertFalse(tool_sent.is_set())
                    model_release.set()
                self.assertEqual(len(model.requests), 1)
                _, snapshot = history.context_snapshot("main")
                self.assertEqual([row.message.content for row in snapshot], ["one", ""])
                self.assertTrue(snapshot[-1].active_reply)
                staged_id = snapshot[-1].message_id
                self.assertEqual(snapshot[-1].message.tool_calls[0].result,
                                 "tool says done" if first == "tool" else None)
                if first == "tool":
                    receipt_release.set()
                else:
                    tool_release.set()
                await self.idle(runtime)
                self.assertEqual(len(model.requests), 2)
                self.assertEqual(history.read_after("main")[1].message_id, staged_id)
                self.assertEqual([item.content for item in model.requests[1][0]],
                                 ["one", "first", "two", "three"])
                self.assertEqual(channel.replies[0][0].run_id, channel.replies[1][0].run_id)
                self.assertNotEqual(channel.replies[0][0].reply_id, channel.replies[1][0].reply_id)

    async def test_full_voice_receipt_does_not_stop_active_model(self):
        stop, receipt_done, tool_done = (asyncio.Event() for _ in range(3))

        async def mixed():
            yield TextDelta("already confirmed")
            yield ToolReady(CALL)
            await stop.wait()

        async def tool(args, context):
            tool_done.set()
            return "done"

        class EarlyChannel(Channel):
            async def deliver(self, reply):
                reply.select_voice_delivery()
                text = await anext(reply)
                receipt_done.set()
                return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)

        model = ScriptModel(mixed, events())
        runtime, history = self.setup_runtime(model, EarlyChannel(), tools=Tools([Tool(SPEC, tool)]))
        runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
        await asyncio.wait_for(asyncio.gather(tool_done.wait(), receipt_done.wait()), 2)
        runtime.submit_messages([MessageDelivery(Message("user", "two"), "trigger")])
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(history.outcomes("main"), [])
        stop.set()
        await self.idle(runtime)
        self.assertEqual(len(model.requests), 2)

    async def test_voice_receipt_stops_blocked_model_but_settles_started_tool_and_next_run(self):
        for status, confirmed in ((DeliveryStatus.PARTIAL, "first"), (DeliveryStatus.ZERO, "")):
            with self.subTest(status=status):
                receipt_ready, model_cancelled, model_closed, release_model = (asyncio.Event() for _ in range(4))
                tool_started, release_tool = asyncio.Event(), asyncio.Event()
                requests, ran_calls = [], []
                late_call = ToolCall("late", SPEC.name, "{}")

                class Model:
                    async def stream(self, messages, tools, *, thinking, instructions=None):
                        requests.append(list(messages))
                        if len(requests) > 1:
                            yield TextDelta("next")
                            return
                        try:
                            yield TextDelta("first unconfirmed")
                            yield ToolReady(CALL)
                            try:
                                await asyncio.Future()
                            except asyncio.CancelledError:
                                model_cancelled.set()
                                await release_model.wait()
                                yield ToolReady(late_call)
                        finally:
                            model_closed.set()

                class Voice(Channel):
                    async def deliver(self, reply):
                        reply.select_voice_delivery()
                        if self.replies:
                            return await super().deliver(reply)
                        self.replies.append((reply.info, await anext(reply)))
                        await receipt_ready.wait()
                        return Receipt(reply.info.reply_id, status, confirmed)

                async def tool(args, context):
                    ran_calls.append(context.call_id)
                    tool_started.set()
                    await release_tool.wait()
                    return "tool result"

                voice = Voice()
                runtime, history = self.setup_runtime(Model(), voice, tools=Tools([Tool(SPEC, tool)]))
                try:
                    runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
                    await asyncio.wait_for(tool_started.wait(), 2)
                    runtime.submit_messages([MessageDelivery(Message("user", "two"), "trigger")])
                    receipt_ready.set()
                    await asyncio.wait_for(model_cancelled.wait(), 2)
                    self.assertEqual(history.outcomes("main"), [])
                    self.assertEqual(len(requests), 1)
                    release_tool.set()
                    await self.idle(runtime)
                    self.assertFalse(model_closed.is_set())
                    self.assertEqual(requests[1], [Message("user", "one"), assistant_message(confirmed, (replace(CALL, result="tool result"),)),
                                                  Message("user", "two")])
                    first, second = history.outcomes("main")
                    self.assertEqual((first.model_status, first.delivery_status), ("model_interrupted", status))
                    self.assertNotEqual(first.run_id, second.run_id)
                    release_model.set()
                    await asyncio.wait_for(model_closed.wait(), 2)
                    self.assertEqual(ran_calls, [CALL.id])
                finally:
                    release_model.set()

    async def test_cancelled_model_child_leaves_voice_settlement_and_queue_running(self):
        for order in ("simultaneous", "model first", "ordinary full"):
            with self.subTest(order=order):
                release_model, model_exiting, release_receipt = (asyncio.Event() for _ in range(3))
                tool_started, release_tool = asyncio.Event(), asyncio.Event()
                recorded = []

                async def script():
                    yield TextDelta("confirmed tail")
                    yield ToolReady(CALL)
                    await release_model.wait()
                    model_exiting.set()
                    raise asyncio.CancelledError

                class ChannelForTest(Channel):
                    async def deliver(self, reply):
                        if self.replies:
                            return await super().deliver(reply)
                        if order != "ordinary full":
                            reply.select_voice_delivery()
                        text = await anext(reply)
                        self.replies.append((reply.info, text))
                        await release_receipt.wait()
                        if order == "simultaneous":
                            release_model.set()  # 子任务先于等待者醒来，与有效回执一同就绪。
                        if order == "ordinary full":
                            return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)
                        return Receipt(reply.info.reply_id, DeliveryStatus.PARTIAL, "confirmed")

                async def tool(args, context):
                    tool_started.set()
                    await release_tool.wait()
                    recorded.append(context.call_id)
                    return "result"

                model = ScriptModel(script, events(TextDelta("next")))
                runtime, history = self.setup_runtime(model, ChannelForTest(), tools=Tools([Tool(SPEC, tool)]))
                runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
                await asyncio.wait_for(tool_started.wait(), 2)
                runtime.submit_messages([MessageDelivery(Message("user", "two"), "trigger")])
                if order != "simultaneous":
                    release_model.set()
                    await asyncio.wait_for(model_exiting.wait(), 2)
                release_receipt.set()
                await asyncio.wait_for(model_exiting.wait(), 2)
                self.assertEqual(history.outcomes("main"), [])
                release_tool.set()
                await self.idle(runtime)
                first, second = history.outcomes("main")
                self.assertNotEqual(first.run_id, second.run_id)
                self.assertEqual(recorded, [CALL.id])
                if order == "ordinary full":
                    self.assertEqual((first.model_status, first.delivery_status),
                                     ("model_cancelled", DeliveryStatus.FULL))
                    prefix = "confirmed tail"
                else:
                    self.assertEqual((first.model_status, first.delivery_status),
                                     ("model_interrupted", DeliveryStatus.PARTIAL))
                    prefix = "confirmed"
                self.assertEqual(model.requests[1][0], [Message("user", "one"), assistant_message(prefix, (replace(CALL, result="result"),)),
                                                      Message("user", "two")])

    async def test_stuck_model_cleanup_pauses_at_limit_and_shutdown_does_not_cancel_again(self):
        releases = [asyncio.Event(), asyncio.Event()]
        cleaned = [asyncio.Event(), asyncio.Event()]
        cancellations = [0, 0]

        def stuck(index):
            async def script():
                yield TextDelta("interrupted")
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    cancellations[index] += 1
                    try:
                        await releases[index].wait()
                    except asyncio.CancelledError:
                        cancellations[index] += 1
                        raise
                finally:
                    cleaned[index].set()
            return script

        class Voice(Channel):
            async def deliver(self, reply):
                reply.select_voice_delivery()
                if len(self.replies) == 2:
                    return await super().deliver(reply)
                self.replies.append((reply.info, await anext(reply)))
                return Receipt(reply.info.reply_id, DeliveryStatus.ZERO)

        model = ScriptModel(stuck(0), stuck(1), events(TextDelta("last response")))
        runtime, history = self.setup_runtime(model, Voice())
        try:
            with self.assertLogs("agent_companion.runtime.engine", level="WARNING") as reached:
                for text in ("one", "two"):
                    runtime.submit_messages([MessageDelivery(Message("user", text), "trigger")])
                runtime.submit_messages((MessageDelivery(Message("user", "deferred record"), "record"),))
                runtime.submit_messages((MessageDelivery(Message("user", "three"), "trigger"),))
                await asyncio.wait_for(runtime.wait_idle(), 2)
            self.assertEqual(len(model.requests), 2)
            self.assertEqual(runtime.last_failure, "model_cleanup_blocked")
            self.assertTrue(any("capacity reached; pending=2" in row for row in reached.output))
            self.assertEqual([r.message.content for r in history.read_after("main")], ["one", "two"])
            self.assertTrue(runtime.has_pending_messages)

            releases[0].set()
            async with asyncio.timeout(2):
                while len(history.outcomes("main")) != 3:
                    await asyncio.sleep(0)
            await self.idle(runtime)
            self.assertEqual([r.message.content for r in history.read_after("main")],
                             ["one", "two", "deferred record", "three", "last response"])
            self.assertIn(Message("user", "deferred record"), model.requests[-1][0])
            with patch("agent_companion.runtime.engine._MODEL_CLOSE_GRACE", .01):
                with self.assertLogs("agent_companion.runtime.engine", level="WARNING") as closing:
                    await asyncio.wait_for(runtime.close(), 2)
            self.assertTrue(any("still running after shutdown grace; pending=1" in row for row in closing.output))
            self.assertFalse(cleaned[1].is_set())
            self.assertEqual(cancellations, [1, 1])
        finally:
            for release in releases:
                release.set()
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in cleaned)), 2)

    async def test_tool_continuation_waits_for_capacity_without_restarting_the_run(self):
        for shutdown in (False, True):
            with self.subTest(shutdown=shutdown):
                await self._check_tool_continuation_at_capacity(shutdown=shutdown)

    async def _check_tool_continuation_at_capacity(self, *, shutdown):
        cleanup_started, release_cleanup, cleanup_finished = (asyncio.Event() for _ in range(3))
        tool_started, release_tool = asyncio.Event(), asyncio.Event()
        cancellations = []
        calls = []

        async def interrupted():
            yield TextDelta("old")
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancellations.append("reader")
                cleanup_started.set()
                try:
                    await release_cleanup.wait()
                except asyncio.CancelledError:
                    cancellations.append("cleanup")
                    raise
                finally:
                    cleanup_finished.set()

        async def tool(args, context):
            calls.append(context.call_id)
            tool_started.set()
            await release_tool.wait()
            return "result"

        class Voice(Channel):
            async def deliver(self, reply):
                reply.select_voice_delivery()
                if self.replies:
                    return await super().deliver(reply)
                self.replies.append((reply.info, await anext(reply)))
                return Receipt(reply.info.reply_id, DeliveryStatus.ZERO)

        model = ScriptModel(interrupted, events(TextDelta("tool round"), ToolReady(CALL)),
                            events(TextDelta("final answer")))
        voice = Voice()
        runtime, history = self.setup_runtime(model, voice, tools=Tools([Tool(SPEC, tool)]))
        try:
            runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
            runtime.submit_messages([MessageDelivery(Message("user", "two"), "trigger")])
            await asyncio.wait_for(asyncio.gather(tool_started.wait(), cleanup_started.wait()), 2)
            # 模拟执行中的工具链遇到容量紧缩；正常完成的应答本身不占退役名额。
            with patch("agent_companion.runtime.engine._MAX_RETIRED_MODELS", 1):
                with self.assertLogs("agent_companion.runtime.engine", level="WARNING") as blocked:
                    release_tool.set()
                    async with asyncio.timeout(2):
                        while runtime.last_failure != "model_cleanup_blocked":
                            await asyncio.sleep(0)
                self.assertTrue(any("capacity reached; pending=1" in row for row in blocked.output))
                self.assertEqual(len(model.requests), 2)
                rows = [row.message for row in history.read_after("main")]
                self.assertEqual(rows[-1].tool_calls, (replace(CALL, result="result"),))
                if shutdown:
                    worker = runtime._worker
                    with (patch("agent_companion.runtime.engine._MODEL_CLOSE_GRACE", .01),
                          self.assertLogs("agent_companion.runtime.engine", level="WARNING") as closing):
                        await asyncio.wait_for(runtime.close(), 2)
                    self.assertTrue(worker.cancelled())
                    self.assertTrue(any("still running after shutdown grace" in row for row in closing.output))
                    self.assertFalse(cleanup_finished.is_set())
                release_cleanup.set()
                await asyncio.wait_for(cleanup_finished.wait(), 2)
                if not shutdown:
                    await self.idle(runtime)
            self.assertEqual(cancellations, ["reader"])
            self.assertEqual(calls, [CALL.id])
            outcomes = history.outcomes("main")
            if shutdown:
                self.assertEqual(len(model.requests), 2)
                self.assertEqual(len(outcomes), 2)
                self.assertEqual([row.message for row in history.read_after("main")], rows)
            else:
                old, tool_round, final = outcomes
                self.assertNotEqual(old.run_id, tool_round.run_id)
                self.assertEqual(tool_round.run_id, final.run_id)
                self.assertEqual(voice.replies[-1][1], "final answer")
        finally:
            release_tool.set()
            release_cleanup.set()

    async def test_voice_receipt_wins_over_simultaneous_tool_frame(self):
        ready = asyncio.Event()

        async def script():
            yield TextDelta("not played")
            await ready.wait()
            yield ToolReady(CALL)

        class Voice:
            async def deliver(self, reply):
                reply.select_voice_delivery()
                await anext(reply)
                # 模型帧会先于等待投递任务的 Runtime 唤醒，但此时回执已返回。
                ready.set()
                return Receipt(reply.info.reply_id, DeliveryStatus.ZERO)

        ran_tool = asyncio.Event()

        async def tool(args, context):
            ran_tool.set()
            return "unexpected"

        model = ScriptModel(script)
        runtime, history = self.setup_runtime(model, Voice(), tools=Tools([Tool(SPEC, tool)]))
        runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
        await self.idle(runtime)
        self.assertFalse(ran_tool.is_set())
        self.assertEqual([entry.message for entry in history.read_after("main")], [Message("user", "one")])
        self.assertEqual(len(model.requests), 1)
        outcome, = history.outcomes("main")
        self.assertEqual((outcome.model_status, outcome.delivery_status, outcome.assistant_message_id),
                         ("model_interrupted", DeliveryStatus.ZERO, None))

    async def test_voice_receipt_after_model_completion_still_blocks_tool_continuation(self):
        model_finished, release_receipt = asyncio.Event(), asyncio.Event()
        replay = PrivateData(bytes(16), b"opaque tool replay")

        async def script():
            yield TextDelta("first unconfirmed")
            yield ToolReady(CALL)
            yield AssistantPartsReady((Reasoning(replay),))
            model_finished.set()

        class Voice(Channel):
            async def deliver(self, reply):
                reply.select_voice_delivery()
                text = "".join([part async for part in reply])
                self.replies.append((reply.info, text))
                if len(self.replies) == 1:
                    await release_receipt.wait()
                    return Receipt(reply.info.reply_id, DeliveryStatus.PARTIAL, "first")
                return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)

        async def tool(args, context):
            return "tool result"

        model = ScriptModel(script, events(TextDelta("next")))
        runtime, history = self.setup_runtime(model, Voice(), tools=Tools([Tool(SPEC, tool)]))
        runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
        await asyncio.wait_for(model_finished.wait(), 2)
        runtime.submit_messages([MessageDelivery(Message("user", "two"), "trigger")])
        release_receipt.set()
        await self.idle(runtime)
        first, second = history.outcomes("main")
        self.assertEqual((first.model_status, first.delivery_status), ("complete", DeliveryStatus.PARTIAL))
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(model.requests[1][0], [Message("user", "one"), assistant_message("first", (replace(CALL, result="tool result"),), reasoning=replay),
                                                Message("user", "two")])

    async def test_voice_interruption_blocks_queued_serial_tool(self):
        for model_complete in (False, True):
            with self.subTest(model_complete=model_complete):
                started, release = asyncio.Event(), asyncio.Event()
                queued = ToolCall("queued", SPEC.name, "{}")
                invocations = []

                async def script():
                    yield TextDelta("first unconfirmed")
                    yield ToolReady(CALL)
                    yield ToolReady(queued)
                    if not model_complete:
                        await asyncio.Future()

                async def tool(args, context):
                    invocations.append(context.call_id)
                    if context.call_id == CALL.id:
                        started.set()
                        await release.wait()
                    return "tool result"

                class Voice:
                    async def deliver(self, reply):
                        reply.select_voice_delivery()
                        await anext(reply)
                        await started.wait()
                        # 锁释放与回执汇合竞争：等待者不能依赖 worker 已先结算回执。
                        release.set()
                        return Receipt(reply.info.reply_id, DeliveryStatus.PARTIAL, "first")

                model = ScriptModel(script)
                runtime, history = self.setup_runtime(model, Voice(), tools=Tools([Tool(SPEC, tool, serial_per_request=True)]))
                runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
                await self.idle(runtime)
                self.assertEqual(invocations, [CALL.id])
                self.assertEqual(len(model.requests), 1)
                self.assertEqual(history.read_after("main")[-1].message.tool_calls, (
                    replace(CALL, result="tool result"), replace(queued, result='{"error":"tool_cancelled"}'),
                ))
                outcome, = history.outcomes("main")
                self.assertEqual(outcome.model_status, "complete" if model_complete else "model_interrupted")

    async def test_runtime_shutdown_blocks_queued_serial_tool_at_lock_handoff(self):
        for close_now in (False, True):
            with self.subTest(close_now=close_now):
                started, release = asyncio.Event(), asyncio.Event()
                queued = ToolCall("queued", SPEC.name, "{}")
                queued_result = asyncio.get_running_loop().create_future()
                invocations = []

                async def tool(args, context):
                    invocations.append(context.call_id)
                    if context.call_id == CALL.id:
                        started.set()
                        await release.wait()
                    return "tool result"

                tools = Tools([Tool(SPEC, tool, serial_per_request=True)])
                execute = tools.execute

                async def record(call, context, **options):
                    result = await execute(call, context, **options)
                    if call.id == queued.id:
                        queued_result.set_result(result)
                    return result

                model = ScriptModel(events(ToolReady(CALL), ToolReady(queued)))
                runtime, _ = self.setup_runtime(model, None, tools=tools)
                with patch.object(tools, "execute", new=record):
                    runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
                    await asyncio.wait_for(started.wait(), 2)
                    release.set()
                    if close_now:
                        # 第一条释放锁后，close 先同步关停，再在等待者之后唤醒 worker 处理取消。
                        closing = asyncio.create_task(runtime.close())
                        self.addAsyncCleanup(asyncio.wait_for, closing, 2)
                    else:
                        runtime.stop_accepting()
                    self.assertEqual(await asyncio.wait_for(queued_result, 2), '{"error":"tool_cancelled"}')
                    await self.idle(runtime)
                self.assertEqual(invocations, [CALL.id])
                self.assertEqual(len(model.requests), 1)

    async def test_pure_tool_does_not_call_channel_and_records_failure(self):
        async def tool(args, context):
            raise RuntimeError("internal details")

        model = ScriptModel(events(ToolReady(CALL)), events(TextDelta("done")))
        channel = Channel()
        runtime, history = self.setup_runtime(model, channel, tools=Tools([Tool(SPEC, tool)]))
        runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
        await self.idle(runtime)
        self.assertEqual(len(channel.replies), 1)
        self.assertEqual([outcome.delivery_status for outcome in history.outcomes("main")], [None, DeliveryStatus.FULL])
        self.assertEqual(model.requests[1][0][-1:], [assistant_message("", (replace(CALL, result='{"error":"tool_failed"}'),))])

    async def test_delivery_statuses_persist_only_prefix_and_allow_next_turn(self):
        cases = [(DeliveryStatus.FULL, "abcdef"), (DeliveryStatus.PARTIAL, "abc"),
                 (DeliveryStatus.ZERO, ""), (DeliveryStatus.FAILED, ""),
                 (DeliveryStatus.UNKNOWN, "ab"), (DeliveryStatus.FULL, "bad prefix")]
        for status, confirmed in cases:
            with self.subTest(status=status, confirmed=confirmed):
                model = ScriptModel(events(TextDelta("abcdef")), events())
                channel = Channel(lambda info, text: Receipt(info.reply_id, status, confirmed))
                runtime, history = self.setup_runtime(model, channel)
                runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
                await self.idle(runtime)
                runtime.submit_messages([MessageDelivery(Message("user", "two"), "trigger")])
                await self.idle(runtime)
                prefix = "" if confirmed == "bad prefix" else confirmed
                saved = [row.message for row in history.read_after("main") if row.message.kind == "assistant"]
                self.assertEqual(saved, [] if status == DeliveryStatus.ZERO else [Message("assistant", prefix)])
                self.assertEqual(history.outcomes("main")[0].delivery_status,
                                 DeliveryStatus.UNKNOWN if confirmed == "bad prefix" else status)
                self.assertEqual([m.content for m in model.requests[1][0]],
                                 ["one", *([prefix] if prefix else []), "two"])
                self.assertEqual(len(channel.replies), 1)

    async def test_mixed_zero_delivery_keeps_tools(self):
        async def tool(args, context):
            return "ok"

        model = ScriptModel(events(TextDelta("not sent"), ToolReady(CALL)), events())
        channel = Channel(lambda info, text: Receipt(info.reply_id, DeliveryStatus.ZERO))
        runtime, _ = self.setup_runtime(model, channel, tools=Tools([Tool(SPEC, tool)]))
        runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
        await self.idle(runtime)
        self.assertEqual(model.requests[1][0][-1:], [assistant_message("", (replace(CALL, result="ok"),))])

    async def test_delivery_timeout_does_not_retry_and_does_not_hold_the_queue(self):
        release = asyncio.Event()

        class UnresponsiveChannel:
            async def deliver(self, reply):
                await release.wait()

        model = ScriptModel(events(TextDelta("unknown")), events())
        runtime, history = self.setup_runtime(model, UnresponsiveChannel())
        with patch("agent_companion.runtime.engine._DELIVERY_TIMEOUT", .01):
            runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
            runtime.submit_messages([MessageDelivery(Message("user", "two"), "trigger")])
            await self.idle(runtime)
        self.assertEqual(history.outcomes("main")[0].delivery_status, DeliveryStatus.UNKNOWN)
        self.assertEqual([m.content for m in model.requests[1][0]], ["one", "two"])

    async def test_terminal_uses_completion_time_and_rejects_late_progress(self):
        loop = asyncio.get_running_loop()
        for idle in (None, 2):
            for on_time in (False, True):
                with self.subTest(idle=idle, on_time=on_time):
                    clock = [loop.time()]
                    entered, release = asyncio.Event(), asyncio.Event()

                    def jump():
                        clock[0] += 3

                    class TimedChannel:
                        async def deliver(self, reply):
                            full = "".join([part async for part in reply])
                            reply.report_progress(reply.info.reply_id, 2, "early")
                            entered.set()
                            await release.wait()
                            if on_time:
                                # 任务结束后、等待方恢复前让时钟跨过截止。
                                loop.call_soon(jump)
                            else:
                                jump()
                                reply.report_progress(reply.info.reply_id, 4, full)
                            return Receipt(reply.info.reply_id, DeliveryStatus.FULL, full)

                    model = ScriptModel(events(TextDelta("early late")))
                    runtime, history = self.setup_runtime(model, TimedChannel(), delivery_idle_timeout=idle)
                    with (patch.object(loop, "time", new=lambda: clock[0]),
                          patch.object(loop, "slow_callback_duration", float("inf")),
                          patch("agent_companion.runtime.engine._DELIVERY_TIMEOUT", 2)):
                        runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
                        await entered.wait()
                        release.set()
                        await runtime.wait_idle()
                    expected = (DeliveryStatus.FULL, "early late") if on_time else (
                        DeliveryStatus.UNKNOWN, "early" if idle is not None else "")
                    self.assertEqual((history.outcomes("main")[0].delivery_status,
                                      history.tail("main", 1)[0].message.content), expected)

    async def test_downstream_cancellation_does_not_cancel_the_queue_worker(self):
        async def tool(args, context):
            raise asyncio.CancelledError

        class CancellingChannel:
            async def deliver(self, reply):
                raise asyncio.CancelledError

        model = ScriptModel(events(TextDelta("not confirmed"), ToolReady(CALL)), events())
        runtime, history = self.setup_runtime(model, CancellingChannel(), tools=Tools([Tool(SPEC, tool)]))
        runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
        runtime.submit_messages([MessageDelivery(Message("user", "two"), "trigger")])
        await self.idle(runtime)
        self.assertEqual(history.outcomes("main")[0].delivery_status, DeliveryStatus.UNKNOWN)
        self.assertEqual(model.requests[-1][0][1:], [assistant_message("", (replace(CALL, result='{"error":"tool_cancelled"}'),)),
                                                    Message("user", "two")])

    async def test_watchdog_ignores_late_receipts_and_cleanup_errors(self):
        for cleanup_error in (False, True):
            with self.subTest(cleanup_error=cleanup_error):
                cancelled, release, cleaned = (asyncio.Event() for _ in range(3))
                abandoned, remaining = [], []

                class SlowCleanupChannel(Channel):
                    async def deliver(self, reply):
                        prefix = await anext(reply)
                        if prefix != "first":
                            reply.report_progress("wrong reply", 2, "next")
                            if reply.progress != (0, ""):
                                raise AssertionError("accepted another reply's progress")
                            text = prefix + "".join([part async for part in reply])
                            return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)
                        reply.report_progress(reply.info.reply_id, 10, "first")
                        reply.report_progress(reply.info.reply_id, 20, "first invalid")
                        reply.report_progress(reply.info.reply_id, 5, "first stale")
                        try:
                            while True:
                                reply.report_progress(reply.info.reply_id, 10, "first")
                                await asyncio.sleep(.002)
                        except asyncio.CancelledError:
                            abandoned.append(reply.abandoned)
                            remaining.extend([part async for part in reply])
                            cancelled.set()
                            await release.wait()
                            reply.report_progress(reply.info.reply_id, 30, "first stale buffer")
                            self_snapshot.append(reply.progress)
                            cleaned.set()
                            if cleanup_error:
                                raise RuntimeError("late cleanup")
                            return Receipt(reply.info.reply_id, DeliveryStatus.FULL, "first stale buffer")

                self_snapshot = []
                model = ScriptModel(events(TextDelta("first"), TextDelta(" stale buffer")), events(TextDelta("next")))
                runtime, history = self.setup_runtime(model, SlowCleanupChannel(),
                                                      delivery_idle_timeout=.02)
                runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
                runtime.submit_messages([MessageDelivery(Message("user", "two"), "trigger")])
                try:
                    await asyncio.wait_for(cancelled.wait(), 2)
                    await self.idle(runtime)
                    self.assertFalse(cleaned.is_set())
                    self.assertEqual(abandoned, [True])
                    self.assertEqual(remaining, [])
                    outcomes = history.outcomes("main")
                    self.assertEqual([item.delivery_status for item in outcomes],
                                     [DeliveryStatus.UNKNOWN, DeliveryStatus.FULL])
                finally:
                    release.set()
                await asyncio.wait_for(cleaned.wait(), 2)
                self.assertEqual(history.outcomes("main"), outcomes)
                self.assertEqual(self_snapshot, [(10, "first")])
                self.assertEqual([m.content for m in model.requests[1][0]], ["one", "first", "two"])
                self.assertEqual([row.message.content for row in history.read_after("main")], ["one", "first", "two", "next"])

    async def test_background_reply_is_queued_only_after_atomic_history_and_tool_completion(self):
        tool_started, tool_released = asyncio.Event(), asyncio.Event()

        async def tool(args, context):
            tool_started.set()
            await tool_released.wait()
            return "ok"

        class Background(Channel):
            def select_delivery(self, reply):
                return ReplyDelivery(self.deliver, after_commit=self.enqueue, require_complete_model=True)

            async def deliver(self, reply):
                text = "".join([part async for part in reply])
                return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)

            def enqueue(self, info, text):
                self.replies.append((info, text, history.read_after("main")))

        async def failed():
            yield TextDelta("unfinished")
            raise ModelError("model_transport")

        model = ScriptModel(events(TextDelta("start"), ToolReady(CALL)), events(TextDelta("end")), failed(),
                            events(TextDelta("not committed")))
        channel = Background()
        runtime, history = self.setup_runtime(model, channel, tools=Tools([Tool(SPEC, tool)]))
        runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
        await asyncio.wait_for(tool_started.wait(), 1)
        self.assertEqual(channel.replies, [])
        tool_released.set()
        await self.idle(runtime)
        self.assertEqual([item[1] for item in channel.replies], ["start", "end"])
        self.assertEqual([row.message.kind for row in channel.replies[0][2]], ["user", "assistant"])
        self.assertEqual([item.delivery_status for item in history.outcomes("main")],
                         [DeliveryStatus.FULL, DeliveryStatus.FULL])

        runtime.submit_messages([MessageDelivery(Message("user", "two"), "trigger")])
        await asyncio.wait_for(runtime.wait_idle(), 2)
        self.assertEqual(runtime.last_failure, "model_failed")
        self.assertNotIn("unfinished", [row.message.content for row in history.read_after("main")])
        with patch.object(history, "finalize_reply", side_effect=ValueError("simulated write failure")):
            runtime.submit_messages([MessageDelivery(Message("user", "three"), "trigger")])
            await asyncio.wait_for(runtime.wait_idle(), 2)
            self.assertEqual(runtime.last_failure, "runtime_failed")
        self.assertEqual([item[1] for item in channel.replies], ["start", "end"])

    async def test_runtime_close_still_cancels_parent_instead_of_committing_child_failures(self):
        delivery_started, delivery_cancelled, tool_started, tool_cancelled = (asyncio.Event() for _ in range(4))

        class HangingChannel:
            async def deliver(self, reply):
                reply.select_voice_delivery()
                delivery_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    delivery_cancelled.set()

        async def tool(args, context):
            tool_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                tool_cancelled.set()

        model_closed = asyncio.Event()
        late_calls = []

        class Model:
            async def stream(self, messages, tools, *, thinking, instructions=None):
                yield TextDelta("waiting")
                yield ToolReady(CALL)
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    yield ToolReady(ToolCall("late", SPEC.name, "{}"))
                finally:
                    model_closed.set()

        async def check_tool(args, context):
            if context.call_id == CALL.id:
                return await tool(args, context)
            late_calls.append(context.call_id)
            return "unexpected"

        runtime, history = self.setup_runtime(Model(), HangingChannel(), tools=Tools([Tool(SPEC, check_tool)]))
        runtime.submit_messages([MessageDelivery(Message("user", "started"), "trigger")])
        await asyncio.wait_for(asyncio.gather(delivery_started.wait(), tool_started.wait()), 2)
        runtime.submit_messages([MessageDelivery(Message("user", "discarded on close"), "trigger")])
        await asyncio.wait_for(runtime.close(), 2)
        await asyncio.wait_for(asyncio.gather(delivery_cancelled.wait(), tool_cancelled.wait(), model_closed.wait()), 2)
        self.assertEqual(late_calls, [])
        self.assertEqual(history.outcomes("main"), [])
        _, snapshot = history.context_snapshot("main")
        self.assertEqual([row.message.content for row in snapshot], ["started", ""])
        self.assertFalse(snapshot[-1].active_reply)
        self.assertEqual(snapshot[-1].message.tool_calls, (CALL,))

    async def test_reply_identity_precedes_first_output(self):
        loop = asyncio.get_running_loop()
        model_started, release_model = asyncio.Event(), asyncio.Event()
        model_times, seen = [], []

        async def delayed():
            model_times.append(loop.time())
            model_started.set()
            await release_model.wait()
            yield TextDelta("discarded")

        class DiscardingChannel:
            async def deliver(self, reply):
                seen.append(reply.info)
                return Receipt(reply.info.reply_id, DeliveryStatus.ZERO)

        model = ScriptModel(delayed)
        runtime, history = self.setup_runtime(model, DiscardingChannel())
        runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
        await asyncio.wait_for(model_started.wait(), 2)
        newer_input_started = loop.time()
        self.assertEqual(seen, [])
        release_model.set()
        await self.idle(runtime)
        self.assertLessEqual(seen[0].started_at, model_times[0])
        self.assertLess(seen[0].started_at, newer_input_started)
        self.assertEqual(history.outcomes("main")[0].delivery_status, DeliveryStatus.ZERO)
        self.assertEqual([row.message for row in history.read_after("main")], [Message("user", "one")])

    async def test_timeout_and_model_failure_set_terminal_states_and_keep_confirmed(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout):
                async def incomplete():
                    yield TextDelta("already sent")
                    # 即使替身提前交付了包，也不能在流失败或超时后保存。
                    yield AssistantPartsReady((Reasoning(PrivateData(bytes(16), b"discard on failure")),))
                    if timeout:
                        await asyncio.Event().wait()
                    raise ModelError("truncated_stream")

                model = ScriptModel(incomplete, events())
                runtime, history = self.setup_runtime(model, Channel(), settings=RuntimeSettings(model_timeout=.01))
                runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
                runtime.submit_messages([MessageDelivery(Message("user", "two"), "trigger")])
                await self.idle(runtime)
                outcomes = history.outcomes("main")
                self.assertEqual(outcomes[0].model_status, "model_timeout" if timeout else "model_failed")
                self.assertNotEqual(outcomes[0].run_id, outcomes[1].run_id)
                self.assertEqual([m.content for m in model.requests[1][0]], ["one", "already sent", "two"])
                self.assertFalse(reasoning_parts(history.read_after("main")[1].message))

    async def test_model_failure_still_finishes_started_tool_in_same_group(self):
        async def failed():
            yield ToolReady(CALL)
            raise ModelError("truncated_stream")

        async def tool(args, context):
            return "done despite model failure"

        model = ScriptModel(failed, events())
        runtime, history = self.setup_runtime(model, Channel(), tools=Tools([Tool(SPEC, tool)]))
        runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
        runtime.submit_messages([MessageDelivery(Message("user", "two"), "trigger")])
        await self.idle(runtime)
        self.assertEqual(history.outcomes("main")[0].model_status, "model_failed")
        self.assertEqual(model.requests[1][0][1:2], [assistant_message("", (replace(CALL, result="done despite model failure"),))])

    async def test_failed_reasoning_is_never_reused_as_a_complete_protocol_field(self):
        unfinished = PrivateData(_replay_key(MODEL_NAME), b"unfinished reasoning")
        complete = PrivateData(_replay_key(MODEL_NAME), b"complete reasoning")

        async def failed():
            yield AssistantPartsReady((Reasoning(unfinished),))
            yield TextDelta("confirmed prefix")
            yield ToolReady(CALL)
            raise ModelError("truncated_stream")

        async def tool(args, context):
            return "completed tool result"

        model = ScriptModel(failed, events(TextDelta("new answer"), AssistantPartsReady((Reasoning(complete),))))
        model.history_replay_bytes = DeepSeek(None).history_replay_bytes
        runtime, history = self.setup_runtime(model, Channel(), tools=Tools([Tool(SPEC, tool)]), thinking=True)
        runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
        runtime.submit_messages([MessageDelivery(Message("user", "two"), "trigger")])
        await self.idle(runtime)
        raw = [item.message for item in history.read_after("main")]
        self.assertEqual(raw[1], assistant_message("confirmed prefix", (replace(CALL, result="completed tool result"),)))
        self.assertEqual(reasoning_parts(raw[-1]), (complete,))
        self.assertIsNone(history.summary("main"))
        self.assertEqual(len(model.requests), 2)
        projected, tools, thinking = model.requests[1]
        self.assertTrue(thinking)
        self.assertEqual(tools, (SPEC,))
        request = build_request(projected, tools, thinking=thinking)
        self.assertEqual(request["messages"][1]["role"], "assistant")
        prompt = repr(request)
        self.assertIn("confirmed prefix", prompt)
        self.assertIn("completed tool result", prompt)
        self.assertNotIn("unfinished reasoning", prompt)

    async def test_invalid_tool_arguments_never_invoke_the_handler(self):
        invoked = []

        async def tool(args, context):
            invoked.append(args)
            return "wrong"

        broken = ToolCall("broken", "test", '{"a":')
        model = ScriptModel(events(ToolReady(broken)), events())
        runtime, _ = self.setup_runtime(model, Channel(), tools=Tools([Tool(SPEC, tool)]))
        runtime.submit_messages([MessageDelivery(Message("user", "one"), "trigger")])
        await self.idle(runtime)
        self.assertEqual(invoked, [])
        self.assertEqual(model.requests[1][0][-1].tool_calls[0].result, '{"error":"invalid_arguments"}')
