import asyncio
import unittest
from unittest.mock import patch

from agent_companion.config import RuntimeSettings
from agent_companion.models.deepseek import MODEL_NAME, DeepSeek, _replay_key, build_request
from agent_companion.protocol import (DeliveryStatus, Message, ModelError, Receipt, Replay, ReplayReady,
                                      ReplyDelivery, TextDelta, ToolCall,
                                      ToolReady, ToolSpec)
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.storage.sqlite import History


SPEC = ToolSpec("test", "模拟工具", {"type": "object", "properties": {}})
CALL = ToolCall("call-1", "test", "{}")


class ScriptModel:
    def __init__(self, *scripts):
        self.scripts = iter(scripts)
        self.requests = []

    async def stream(self, messages, tools, *, thinking):
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

    async def test_close_cancels_active_and_queued_tools_before_waiting_for_child_model(self):
        model_closing, model_release, model_stopped = (asyncio.Event() for _ in range(3))
        tool_active, tool_closing, tool_release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        queued_started, queued_closing = asyncio.Event(), asyncio.Event()
        lock = asyncio.Lock()

        class Model:
            async def stream(self, messages, tools, *, thinking):
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
        runtime.submit_batch((Message("user", "synthetic"),))
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
                runtime.submit_batch([Message("user", "one")])
                await asyncio.wait_for(asyncio.gather(tool_started.wait(), receipt_started.wait()), 2)
                runtime.submit_batch([Message("user", "two"), Message("user", "three")])
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
                self.assertEqual([row.message.content for row in history.read_after("main")], ["one"])
                if first == "tool":
                    receipt_release.set()
                else:
                    tool_release.set()
                await self.idle(runtime)
                self.assertEqual(len(model.requests), 2)
                self.assertEqual([item.content for item in model.requests[1][0]],
                                 ["one", "first", "tool says done", "two", "three"])
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
        runtime.submit_batch([Message("user", "one")])
        await asyncio.wait_for(asyncio.gather(tool_done.wait(), receipt_done.wait()), 2)
        runtime.submit_batch([Message("user", "two")])
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
                    async def stream(self, messages, tools, *, thinking):
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
                    runtime.submit_batch([Message("user", "one")])
                    await asyncio.wait_for(tool_started.wait(), 2)
                    runtime.submit_batch([Message("user", "two")])
                    receipt_ready.set()
                    await asyncio.wait_for(model_cancelled.wait(), 2)
                    self.assertEqual(history.outcomes("main"), [])
                    self.assertEqual(len(requests), 1)
                    release_tool.set()
                    await self.idle(runtime)
                    self.assertFalse(model_closed.is_set())
                    self.assertEqual(requests[1], [Message("user", "one"), Message("assistant", confirmed, (CALL,)),
                                                  Message("tool", "tool result", tool_call_id=CALL.id),
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
                runtime.submit_batch([Message("user", "one")])
                await asyncio.wait_for(tool_started.wait(), 2)
                runtime.submit_batch([Message("user", "two")])
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
                self.assertEqual(model.requests[1][0], [Message("user", "one"), Message("assistant", prefix, (CALL,)),
                                                      Message("tool", "result", tool_call_id=CALL.id),
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
                for text in ("one", "two", "three"):
                    runtime.submit_batch([Message("user", text)])
                await asyncio.wait_for(runtime.wait_idle(), 2)
            self.assertEqual(len(model.requests), 2)
            self.assertEqual(runtime.last_failure, "model_cleanup_blocked")
            self.assertTrue(any("capacity reached; pending=2" in row for row in reached.output))
            self.assertEqual([r.message.content for r in history.read_after("main")], ["one", "two"])

            releases[0].set()
            async with asyncio.timeout(2):
                while len(history.outcomes("main")) != 3:
                    await asyncio.sleep(0)
            await self.idle(runtime)
            self.assertEqual([r.message.content for r in history.read_after("main")],
                             ["one", "two", "three", "last response"])
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
            runtime.submit_batch([Message("user", "one")])
            runtime.submit_batch([Message("user", "two")])
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
                self.assertEqual(rows[-1], Message("tool", "result", tool_call_id=CALL.id))
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
        runtime.submit_batch([Message("user", "one")])
        await self.idle(runtime)
        self.assertFalse(ran_tool.is_set())
        self.assertEqual([entry.message for entry in history.read_after("main")], [Message("user", "one")])
        self.assertEqual(len(model.requests), 1)
        outcome, = history.outcomes("main")
        self.assertEqual((outcome.model_status, outcome.delivery_status, outcome.assistant_message_id),
                         ("model_interrupted", DeliveryStatus.ZERO, None))

    async def test_voice_receipt_after_model_completion_still_blocks_tool_continuation(self):
        model_finished, release_receipt = asyncio.Event(), asyncio.Event()
        replay = Replay(bytes(16), b"opaque tool replay")

        async def script():
            yield TextDelta("first unconfirmed")
            yield ToolReady(CALL)
            yield ReplayReady(replay)
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
        runtime.submit_batch([Message("user", "one")])
        await asyncio.wait_for(model_finished.wait(), 2)
        runtime.submit_batch([Message("user", "two")])
        release_receipt.set()
        await self.idle(runtime)
        first, second = history.outcomes("main")
        self.assertEqual((first.model_status, first.delivery_status), ("complete", DeliveryStatus.PARTIAL))
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(model.requests[1][0], [Message("user", "one"), Message("assistant", "first", (CALL,), replay=replay),
                                                Message("tool", "tool result", tool_call_id=CALL.id),
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
                runtime.submit_batch([Message("user", "one")])
                await self.idle(runtime)
                self.assertEqual(invocations, [CALL.id])
                self.assertEqual(len(model.requests), 1)
                self.assertEqual([row.message for row in history.read_after("main")][-2:], [
                    Message("tool", "tool result", tool_call_id=CALL.id),
                    Message("tool", '{"error":"tool_cancelled"}', tool_call_id=queued.id),
                ])
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
                    runtime.submit_batch([Message("user", "one")])
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
        runtime.submit_batch([Message("user", "one")])
        await self.idle(runtime)
        self.assertEqual(len(channel.replies), 1)
        self.assertEqual([outcome.delivery_status for outcome in history.outcomes("main")], [None, DeliveryStatus.FULL])
        self.assertEqual(model.requests[1][0][-2:], [Message("assistant", "", (CALL,)),
                                                    Message("tool", '{"error":"tool_failed"}', tool_call_id=CALL.id)])

    async def test_delivery_statuses_persist_only_prefix_and_allow_next_turn(self):
        cases = [(DeliveryStatus.FULL, "abcdef"), (DeliveryStatus.PARTIAL, "abc"),
                 (DeliveryStatus.ZERO, ""), (DeliveryStatus.FAILED, ""),
                 (DeliveryStatus.UNKNOWN, "ab"), (DeliveryStatus.FULL, "bad prefix")]
        for status, confirmed in cases:
            with self.subTest(status=status, confirmed=confirmed):
                model = ScriptModel(events(TextDelta("abcdef")), events())
                channel = Channel(lambda info, text: Receipt(info.reply_id, status, confirmed))
                runtime, history = self.setup_runtime(model, channel)
                runtime.submit_batch([Message("user", "one")])
                await self.idle(runtime)
                runtime.submit_batch([Message("user", "two")])
                await self.idle(runtime)
                prefix = "" if confirmed == "bad prefix" else confirmed
                saved = [row.message for row in history.read_after("main") if row.message.role == "assistant"]
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
        runtime.submit_batch([Message("user", "one")])
        await self.idle(runtime)
        self.assertEqual(model.requests[1][0][-2:], [Message("assistant", "", (CALL,)),
                                                    Message("tool", "ok", tool_call_id=CALL.id)])

    async def test_delivery_timeout_does_not_retry_and_does_not_hold_the_queue(self):
        release = asyncio.Event()

        class UnresponsiveChannel:
            async def deliver(self, reply):
                await release.wait()

        model = ScriptModel(events(TextDelta("unknown")), events())
        runtime, history = self.setup_runtime(model, UnresponsiveChannel())
        with patch("agent_companion.runtime.engine._DELIVERY_TIMEOUT", .01):
            runtime.submit_batch([Message("user", "one")])
            runtime.submit_batch([Message("user", "two")])
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
                        runtime.submit_batch([Message("user", "one")])
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
        runtime.submit_batch([Message("user", "one")])
        runtime.submit_batch([Message("user", "two")])
        await self.idle(runtime)
        self.assertEqual(history.outcomes("main")[0].delivery_status, DeliveryStatus.UNKNOWN)
        self.assertEqual(model.requests[-1][0][1:], [Message("assistant", "", (CALL,)),
                                                    Message("tool", '{"error":"tool_cancelled"}', tool_call_id=CALL.id),
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
                runtime.submit_batch([Message("user", "one")])
                runtime.submit_batch([Message("user", "two")])
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
        runtime.submit_batch([Message("user", "one")])
        await asyncio.wait_for(tool_started.wait(), 1)
        self.assertEqual(channel.replies, [])
        tool_released.set()
        await self.idle(runtime)
        self.assertEqual([item[1] for item in channel.replies], ["start", "end"])
        self.assertEqual([row.message.role for row in channel.replies[0][2]], ["user", "assistant", "tool"])
        self.assertEqual([item.delivery_status for item in history.outcomes("main")],
                         [DeliveryStatus.FULL, DeliveryStatus.FULL])

        runtime.submit_batch([Message("user", "two")])
        await asyncio.wait_for(runtime.wait_idle(), 2)
        self.assertEqual(runtime.last_failure, "model_failed")
        self.assertNotIn("unfinished", [row.message.content for row in history.read_after("main")])
        with patch.object(history, "finalize_reply", side_effect=ValueError("simulated write failure")):
            runtime.submit_batch([Message("user", "three")])
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
            async def stream(self, messages, tools, *, thinking):
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
        runtime.submit_batch([Message("user", "started")])
        await asyncio.wait_for(asyncio.gather(delivery_started.wait(), tool_started.wait()), 2)
        runtime.submit_batch([Message("user", "discarded on close")])
        await asyncio.wait_for(runtime.close(), 2)
        await asyncio.wait_for(asyncio.gather(delivery_cancelled.wait(), tool_cancelled.wait(), model_closed.wait()), 2)
        self.assertEqual(late_calls, [])
        self.assertEqual(history.outcomes("main"), [])
        self.assertEqual([row.message.content for row in history.read_after("main")], ["started"])

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
        runtime.submit_batch([Message("user", "one")])
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
                    yield ReplayReady(Replay(bytes(16), b"discard on failure"))
                    if timeout:
                        await asyncio.Event().wait()
                    raise ModelError("truncated_stream")

                model = ScriptModel(incomplete, events())
                runtime, history = self.setup_runtime(model, Channel(), settings=RuntimeSettings(model_timeout=.01))
                runtime.submit_batch([Message("user", "one")])
                runtime.submit_batch([Message("user", "two")])
                await self.idle(runtime)
                outcomes = history.outcomes("main")
                self.assertEqual(outcomes[0].model_status, "model_timeout" if timeout else "model_failed")
                self.assertNotEqual(outcomes[0].run_id, outcomes[1].run_id)
                self.assertEqual([m.content for m in model.requests[1][0]], ["one", "already sent", "two"])
                self.assertIsNone(history.read_after("main")[1].message.replay)

    async def test_model_failure_still_finishes_started_tool_in_same_group(self):
        async def failed():
            yield ToolReady(CALL)
            raise ModelError("truncated_stream")

        async def tool(args, context):
            return "done despite model failure"

        model = ScriptModel(failed, events())
        runtime, history = self.setup_runtime(model, Channel(), tools=Tools([Tool(SPEC, tool)]))
        runtime.submit_batch([Message("user", "one")])
        runtime.submit_batch([Message("user", "two")])
        await self.idle(runtime)
        self.assertEqual(history.outcomes("main")[0].model_status, "model_failed")
        self.assertEqual(model.requests[1][0][1:3], [Message("assistant", "", (CALL,)),
                                                     Message("tool", "done despite model failure", tool_call_id=CALL.id)])

    async def test_failed_reasoning_is_never_reused_as_a_complete_protocol_field(self):
        unfinished = Replay(_replay_key(MODEL_NAME), b"unfinished reasoning")
        complete = Replay(_replay_key(MODEL_NAME), b"complete reasoning")

        async def failed():
            yield ReplayReady(unfinished)
            yield TextDelta("confirmed prefix")
            yield ToolReady(CALL)
            raise ModelError("truncated_stream")

        async def tool(args, context):
            return "completed tool result"

        model = ScriptModel(failed, events(TextDelta("new answer"), ReplayReady(complete)))
        model.history_replay_bytes = DeepSeek(None).history_replay_bytes
        runtime, history = self.setup_runtime(model, Channel(), tools=Tools([Tool(SPEC, tool)]), thinking=True)
        runtime.submit_batch([Message("user", "one")])
        runtime.submit_batch([Message("user", "two")])
        await self.idle(runtime)
        raw = [item.message for item in history.read_after("main")]
        self.assertEqual(raw[1], Message("assistant", "confirmed prefix", (CALL,)))
        self.assertEqual(raw[2], Message("tool", "completed tool result", tool_call_id=CALL.id))
        self.assertEqual(raw[-1].replay, complete)
        self.assertIsNone(history.summary("main"))
        self.assertEqual(len(model.requests), 2)
        projected, tools, thinking = model.requests[1]
        self.assertTrue(thinking)
        self.assertEqual(tools, (SPEC,))
        request = build_request(projected, tools, thinking=thinking)
        self.assertEqual(request["messages"][1]["role"], "user")
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
        runtime.submit_batch([Message("user", "one")])
        await self.idle(runtime)
        self.assertEqual(invoked, [])
        self.assertEqual(model.requests[1][0][-1].content, '{"error":"invalid_arguments"}')
