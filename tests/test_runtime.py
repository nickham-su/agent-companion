import asyncio
import unittest

from agent_companion.config import RuntimeSettings
from agent_companion.models.deepseek import build_request
from agent_companion.protocol import (DeliveryStatus, Message, ModelError, ReasoningDelta, Receipt, TextDelta, ToolCall,
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

    async def test_model_stream_completion_is_a_separate_gate(self):
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
                self.assertEqual(history.read_after("main")[1].message, Message("assistant", prefix))
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
        runtime, history = self.setup_runtime(model, UnresponsiveChannel(), settings=RuntimeSettings(delivery_timeout=.01))
        runtime.submit_batch([Message("user", "one")])
        runtime.submit_batch([Message("user", "two")])
        await self.idle(runtime)
        self.assertEqual(history.outcomes("main")[0].delivery_status, DeliveryStatus.UNKNOWN)
        self.assertEqual([m.content for m in model.requests[1][0]], ["one", "two"])

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
                            text = prefix + "".join([part async for part in reply])
                            return Receipt(reply.info.reply_id, DeliveryStatus.FULL, text)
                        try:
                            await asyncio.Event().wait()
                        except asyncio.CancelledError:
                            abandoned.append(reply.abandoned)
                            remaining.extend([part async for part in reply])
                            cancelled.set()
                            await release.wait()
                            cleaned.set()
                            if cleanup_error:
                                raise RuntimeError("late cleanup")
                            return Receipt(reply.info.reply_id, DeliveryStatus.FULL, "first stale buffer")

                model = ScriptModel(events(TextDelta("first"), TextDelta(" stale buffer")), events(TextDelta("next")))
                runtime, history = self.setup_runtime(model, SlowCleanupChannel(),
                                                      settings=RuntimeSettings(delivery_timeout=.01))
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
                self.assertEqual([row.message.content for row in history.read_after("main")], ["one", "", "two", "next"])

    async def test_runtime_close_still_cancels_parent_instead_of_committing_child_failures(self):
        delivery_started, delivery_cancelled, tool_started, tool_cancelled = (asyncio.Event() for _ in range(4))

        class HangingChannel:
            async def deliver(self, reply):
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

        model = ScriptModel(events(TextDelta("waiting"), ToolReady(CALL)))
        runtime, history = self.setup_runtime(model, HangingChannel(), tools=Tools([Tool(SPEC, tool)]))
        runtime.submit_batch([Message("user", "started")])
        await asyncio.wait_for(asyncio.gather(delivery_started.wait(), tool_started.wait()), 2)
        runtime.submit_batch([Message("user", "discarded on close")])
        await asyncio.wait_for(runtime.close(), 2)
        await asyncio.wait_for(asyncio.gather(delivery_cancelled.wait(), tool_cancelled.wait()), 2)
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
        self.assertEqual(history.read_after("main")[1].message, Message("assistant", ""))

    async def test_timeout_and_model_failure_set_terminal_states_and_keep_confirmed(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout):
                async def incomplete():
                    yield TextDelta("already sent")
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
        async def failed():
            yield ReasoningDelta("unfinished reasoning")
            yield TextDelta("confirmed prefix")
            yield ToolReady(CALL)
            raise ModelError("truncated_stream")

        async def tool(args, context):
            return "completed tool result"

        model = ScriptModel(failed, events(TextDelta("old response summary")),
                            events(ReasoningDelta("complete reasoning"), TextDelta("new answer")))
        runtime, history = self.setup_runtime(model, Channel(), tools=Tools([Tool(SPEC, tool)]), thinking=True)
        runtime.submit_batch([Message("user", "one")])
        runtime.submit_batch([Message("user", "two")])
        await self.idle(runtime)
        raw = [item.message for item in history.read_after("main")]
        self.assertEqual(raw[1], Message("assistant", "confirmed prefix", (CALL,)))
        self.assertEqual(raw[2], Message("tool", "completed tool result", tool_call_id=CALL.id))
        self.assertEqual(raw[-1].reasoning, "complete reasoning")
        summary_messages, tools, thinking = model.requests[1]
        self.assertFalse(thinking)
        self.assertEqual(tools, ())
        prompt = repr(build_request(summary_messages, tools, thinking=thinking))
        self.assertIn("confirmed prefix", prompt)
        self.assertIn("completed tool result", prompt)
        self.assertNotIn("unfinished reasoning", prompt)
        request = repr(build_request(model.requests[2][0], (SPEC,), thinking=True))
        self.assertIn("old response summary", request)
        self.assertNotIn("unfinished reasoning", request)

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
