import asyncio
from contextlib import closing
from dataclasses import replace
import unittest
from unittest.mock import AsyncMock, patch

import httpcore
import httpx

from agent_companion.config import RuntimeSettings
from agent_companion.models import _http_stream
from agent_companion.models.deepseek import DeepSeek
from agent_companion.protocol import (AssistantPartsReady, InputUsage, Message, MessageDelivery, ModelClosing,
                                      ModelError, PrivateData, Reasoning, TextDelta, ToolReady)
from agent_companion.runtime import engine
from agent_companion.runtime.context import Context, PreparedContext
from agent_companion.runtime.model_deadline import model_deadline_at
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.storage.sqlite import History
from test_message_support import reasoning_parts
from test_runtime import CALL, SPEC, Channel, ScriptModel, events


class Clock:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.real_time = self.loop.time
        self.offset = 0

    def time(self):
        return self.real_time() + self.offset

    def advance(self, seconds):
        self.offset += seconds

    async def sleep(self, seconds):
        self.advance(seconds)
        await asyncio.sleep(0)


class ResponseBody(httpx.AsyncByteStream):
    def __init__(self, data=b"", *, entered=None, closing=None, release=None):
        self.data = data
        self.entered = entered
        self.closing = closing
        self.release = release

    async def __aiter__(self):
        if self.entered is not None:
            self.entered.set()
            await asyncio.Future()
        yield self.data

    async def aclose(self):
        if self.closing is not None:
            self.closing.set()
            await self.release.wait()


DONE = b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
FRAGMENT = (b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"partial",'
            b'"function":{"name":"test","arguments":"{"}}]}}]}\n\n')


def input_message(text="input"):
    return [MessageDelivery(Message("user", text), "trigger")]


class ModelRetryTests(unittest.IsolatedAsyncioTestCase):
    def setup_runtime(self, model, channel=None, **kwargs):
        history = self.enterContext(closing(History(":memory:")))
        agent = history.ensure_main_agent("main")
        runtime = engine.Runtime("main", history, model, channel, agent=agent, **kwargs)
        self.addAsyncCleanup(runtime.close)
        return runtime, history

    def clock(self):
        clock = Clock()
        self.enterContext(patch.object(clock.loop, "time", clock.time))
        self.enterContext(patch.object(clock.loop, "slow_callback_duration", 1200))
        return clock

    def fast_retries(self, side_effect=None):
        sleep = AsyncMock(side_effect=side_effect)
        return self.enterContext(patch.object(engine, "sleep", sleep))

    async def outcome(self, runtime, history):
        # 留出虚拟时钟跳跃的空间；没有跳时钟的卡死仍由测试命令的上限保护。
        await asyncio.wait_for(runtime.wait_idle(), 1200)
        result, = history.outcomes(runtime.session_id)
        return result

    async def test_network_http_stream_error_and_incomplete_tool_then_success(self):
        wait = self.fast_retries()
        closed, release = asyncio.Event(), asyncio.Event()
        calls = []
        body = ResponseBody(closing=closed, release=release)

        async def vendor(request):
            calls.append(request.content)
            match len(calls):
                case 1:
                    raise httpx.ConnectError("synthetic-only", request=request)
                case 2:
                    return httpx.Response(401, stream=body)
                case 3:
                    return httpx.Response(503)
                case 4:
                    return httpx.Response(200, stream=ResponseBody(b'data: {"error":{"code":"test"}}\n\n'))
                case 5:
                    return httpx.Response(200, stream=ResponseBody(FRAGMENT))
                case 6:
                    return httpx.Response(200, stream=ResponseBody(DONE))
                case _:
                    self.fail("unexpected model request")

        async with httpx.AsyncClient(transport=httpx.MockTransport(vendor), base_url="https://example.invalid") as client:
            model = DeepSeek(client)
            channel = Channel()
            runtime, history = self.setup_runtime(model, channel)
            try:
                with self.assertLogs(engine._LOG, "DEBUG") as logs:
                    runtime.submit_messages(input_message())
                    await asyncio.wait_for(closed.wait(), 1)
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(wait.await_count, 1)
                    release.set()
                    outcome = await self.outcome(runtime, history)
            finally:
                release.set()
                await runtime.close()
                await model.aclose()
        self.assertEqual((outcome.model_status, len(calls), len(set(calls))), ("complete", 6, 1))
        self.assertEqual([call.args[0] for call in wait.await_args_list], [1, 2, 4, 8, 16])
        self.assertEqual([(info.reply_id, text) for info, text in channel.replies], [(outcome.reply_id, "ok")])
        self.assertFalse(history.read_after("main")[-1].message.tool_calls)
        for reason in ("transport_error", "http_401", "http_503", "invalid_sse_event", "truncated_stream"):
            self.assertTrue(any(f"reason={reason}" in line for line in logs.output), reason)
        self.assertTrue(any(f"reply={outcome.reply_id} attempts=6 result=complete" in line for line in logs.output))

    async def test_non_output_does_not_renew_idle_and_failed_metadata_does_not_leak(self):
        clock = self.clock()
        wait = self.fast_retries()
        marker = PrivateData(bytes(16), b"discard")
        attempts = []

        async def stalled():
            attempts.append("first")
            clock.advance(65)
            yield InputUsage(999)
            clock.advance(65)
            yield AssistantPartsReady((Reasoning(marker),))
            yield TextDelta("")
            yield ModelClosing(complete=False)
            clock.advance(51)
            await asyncio.sleep(0)
            await asyncio.Future()

        async def recovered():
            attempts.append("second")
            yield TextDelta("ok")

        model = ScriptModel(stalled, recovered)
        channel = Channel()
        runtime, history = self.setup_runtime(model, channel)
        with self.assertLogs(engine._LOG, "DEBUG") as logs:
            runtime.submit_messages(input_message())
            outcome = await self.outcome(runtime, history)
        self.assertEqual((outcome.model_status, attempts), ("complete", ["first", "second"]))
        wait.assert_awaited_once_with(1)
        self.assertTrue(any("reason=model_idle_timeout" in line for line in logs.output))
        self.assertEqual(model.requests[0], model.requests[1])
        self.assertIsNone(history.last_input_tokens("main"))
        self.assertFalse(reasoning_parts(history.read_after("main")[-1].message))

    async def test_native_idle_timeout_stops_old_stream_then_retries(self):
        clock = self.clock()
        wait = self.fast_retries()
        entered, closing_body, release = (asyncio.Event() for _ in range(3))
        calls = 0

        async def vendor(_):
            nonlocal calls
            calls += 1
            body = ResponseBody(entered=entered, closing=closing_body, release=release) if calls == 1 else ResponseBody(DONE)
            return httpx.Response(200, stream=body)

        async with httpx.AsyncClient(transport=httpx.MockTransport(vendor), base_url="https://example.invalid") as client:
            model = DeepSeek(client)
            runtime, history = self.setup_runtime(model)
            try:
                runtime.submit_messages(input_message())
                await asyncio.wait_for(entered.wait(), 1)
                clock.advance(181)
                await asyncio.wait_for(closing_body.wait(), 1)
                self.assertEqual(calls, 1)
                wait.assert_not_awaited()
                release.set()
                outcome = await self.outcome(runtime, history)
            finally:
                release.set()
                await runtime.close()
                await model.aclose()
        self.assertEqual((outcome.model_status, calls), ("complete", 2))
        wait.assert_awaited_once_with(1)

    async def test_no_retry_when_internal_close_is_forced_while_waiting_for_headers(self):
        await self.check_internal_read_close(at_headers=True, force=True)

    async def test_no_retry_when_internal_close_is_forced_while_waiting_for_body(self):
        await self.check_internal_read_close(at_headers=False, force=True)

    async def test_internal_close_completing_in_budget_allows_retry(self):
        await self.check_internal_read_close(at_headers=False, force=False)

    async def check_internal_read_close(self, *, at_headers, force):
        clock = self.clock()
        wait = self.fast_retries()
        reading, closing_wire, release, closed, interrupted = (asyncio.Event() for _ in range(5))
        request_count = connections = 0
        read_task = close_task = None
        heartbeat = b": keep-alive\n\n"

        def packet(body, length):
            return (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n"
                    + f"Content-Length: {length}\r\n\r\n".encode() + body)

        class Wire(httpcore.AsyncNetworkStream):
            def __init__(self, first):
                self.first = first
                self.sent = False

            async def write(self, buffer, timeout=None):
                nonlocal request_count
                if buffer.startswith(b"POST "):
                    request_count += 1

            async def start_tls(self, **kwargs):
                return self

            async def read(self, max_bytes, timeout=None):
                nonlocal read_task
                if not self.sent and (not self.first or not at_headers):
                    self.sent = True
                    if self.first:
                        return packet(heartbeat, len(heartbeat) + 100)
                    return packet(DONE, len(DONE))
                read_task = asyncio.current_task()
                reading.set()
                await asyncio.Future()

            async def aclose(self):
                nonlocal close_task
                if self.first:
                    close_task = asyncio.current_task()
                    closing_wire.set()
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        interrupted.set()
                        raise
                    closed.set()

        class Backend(httpcore.AsyncNetworkBackend):
            async def connect_tcp(self, **kwargs):
                nonlocal connections
                connections += 1
                return Wire(connections == 1)

        pool = httpcore.AsyncConnectionPool
        with patch.object(httpcore, "AsyncConnectionPool",
                          side_effect=lambda **kw: pool(network_backend=Backend(), **kw)):
            transport = httpx.AsyncHTTPTransport(trust_env=False)
        self.enterContext(patch("agent_companion.models.deepseek.CHAT_COMPLETIONS", "https://example.invalid/chat"))
        async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
            model = DeepSeek(client)
            runtime, history = self.setup_runtime(model)
            try:
                with self.assertLogs("agent_companion", "DEBUG") as logs:
                    runtime.submit_messages(input_message())
                    await asyncio.wait_for(reading.wait(), 1)
                    clock.advance(181)
                    await asyncio.wait_for(closing_wire.wait(), 1)
                    # httpcore 在读任务内部关闭连接；显式 _OwnedStream 关闭在独立任务中。
                    self.assertIs(close_task, read_task)
                    self.assertEqual(request_count, 1)
                    wait.assert_not_awaited()
                    if force:
                        clock.advance(_http_stream._RESPONSE_CLOSE_TIMEOUT + 1)
                        await asyncio.wait_for(interrupted.wait(), 1)
                    else:
                        release.set()
                    await asyncio.wait_for(runtime.wait_idle(), 1)
                    outcome, = history.outcomes("main")
            finally:
                release.set()
                await runtime.close()
                await model.aclose()
        if force:
            self.assertEqual((outcome.model_status, request_count), ("model_timeout", 1))
            self.assertFalse(closed.is_set())
            wait.assert_not_awaited()
            self.assertTrue(any("reason=model_timeout " in line and "close_failure=close_unconfirmed" in line
                                for line in logs.output))
        else:
            self.assertEqual((outcome.model_status, request_count), ("complete", 2))
            self.assertTrue(closed.is_set())
            self.assertFalse(interrupted.is_set())
            wait.assert_awaited_once_with(1)

    async def test_http_error_does_not_retry_when_response_close_is_unconfirmed(self):
        await self.check_unconfirmed_close(idle=False)

    async def test_idle_timeout_does_not_retry_when_response_close_is_unconfirmed(self):
        await self.check_unconfirmed_close(idle=True)

    async def check_unconfirmed_close(self, *, idle):
        clock = self.clock()
        wait = self.fast_retries()
        entered, close_cancelled, release, finished = (asyncio.Event() for _ in range(4))
        calls = 0

        class StuckBody(ResponseBody):
            async def aclose(self):
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    close_cancelled.set()
                    await release.wait()
                finally:
                    finished.set()

        async def vendor(_):
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(200 if idle else 503, stream=StuckBody(entered=entered if idle else None))
            return httpx.Response(200, stream=ResponseBody(DONE))

        async with httpx.AsyncClient(transport=httpx.MockTransport(vendor), base_url="https://example.invalid") as client:
            model = DeepSeek(client)
            runtime, history = self.setup_runtime(model)
            try:
                with (patch.object(_http_stream, "_RESPONSE_CLOSE_TIMEOUT", .02),
                      self.assertLogs("agent_companion", "DEBUG") as logs):
                    runtime.submit_messages(input_message())
                    if idle:
                        await asyncio.wait_for(entered.wait(), 1)
                        clock.advance(181)
                    await asyncio.wait_for(close_cancelled.wait(), 1)
                    await asyncio.wait_for(runtime.wait_idle(), 1)
                    outcome, = history.outcomes("main")
                    self.assertFalse(finished.is_set())
                    self.assertEqual(calls, 1)
            finally:
                release.set()
                await runtime.close()
                await model.aclose()
        self.assertTrue(finished.is_set())
        wait.assert_not_awaited()
        self.assertEqual((outcome.model_status, outcome.model_error),
                         ("model_timeout", None) if idle else ("model_failed", "http_503"))
        expected_reason = "model_timeout" if idle else "http_503"
        self.assertTrue(any(f"reason={expected_reason} " in line and "close_failure=close_timeout" in line
                            for line in logs.output))

    async def test_continuous_text_renews_idle_but_not_total_budget(self):
        clock = self.clock()
        wait = self.fast_retries()

        async def long_stream():
            for number in range(1, 7):
                clock.advance(100)
                yield TextDelta(str(number))
                await asyncio.sleep(0)

        model = ScriptModel(long_stream)
        channel = Channel()
        runtime, history = self.setup_runtime(model, channel)
        runtime.submit_messages(input_message())
        outcome = await self.outcome(runtime, history)
        self.assertEqual((outcome.model_status, len(model.requests)), ("model_timeout", 1))
        self.assertEqual([text for _, text in channel.replies], ["12345"])
        wait.assert_not_awaited()

    async def test_complete_tool_renews_idle_and_only_retry_next_round(self):
        clock = self.clock()
        wait = self.fast_retries()
        executed = []

        async def tool(args, context):
            executed.append(context.reply_id)
            return "one result"

        async def tool_stream():
            clock.advance(100)
            yield ToolReady(CALL)
            clock.advance(100)
            yield ModelClosing(complete=True)

        async def incomplete():
            yield ModelClosing(complete=False)
            raise ModelError("transport_error")

        model = ScriptModel(tool_stream, incomplete, events(TextDelta("answer")))
        runtime, history = self.setup_runtime(model, Channel(), tools=Tools([Tool(SPEC, tool)]))
        runtime.submit_messages(input_message())
        await asyncio.wait_for(runtime.wait_idle(), 1200)
        first, second = history.outcomes("main")
        self.assertEqual((first.model_status, second.model_status), ("complete", "complete"))
        self.assertEqual(executed, [first.reply_id])
        self.assertEqual(len(model.requests), 3)
        self.assertEqual(model.requests[1], model.requests[2])
        self.assertEqual(model.requests[1][0][-1].tool_calls[0].result, "one result")
        wait.assert_awaited_once_with(1)

    async def test_confirmed_model_completion_disarms_generation_budget_before_close(self):
        clock = self.clock()
        wait = self.fast_retries()

        async def completed():
            try:
                yield TextDelta("result")
                yield ModelClosing(complete=True)
            finally:
                clock.advance(601)
                await asyncio.sleep(0)
                raise ModelError("transport_close_timeout")

        class Model:
            calls = 0

            def stream(self, *_args, **_kwargs):
                self.calls += 1
                return completed()

        model = Model()
        channel = Channel()
        runtime, history = self.setup_runtime(model, channel)
        runtime.submit_messages(input_message())
        outcome = await self.outcome(runtime, history)
        self.assertEqual((outcome.model_status, model.calls), ("complete", 1))
        self.assertEqual([text for _, text in channel.replies], ["result"])
        wait.assert_not_awaited()

    async def test_backoff_doubles_caps_and_stops_at_total_budget(self):
        clock = self.clock()
        wait = self.fast_retries(clock.sleep)
        self.enterContext(patch.object(engine, "_MODEL_RETRY_BUDGET", 100))

        async def error():
            raise ModelError("vendor_error")
            yield

        model = ScriptModel(*([error] * 9))
        runtime, history = self.setup_runtime(model)
        runtime.submit_messages(input_message())
        outcome = await self.outcome(runtime, history)
        waits = [call.args[0] for call in wait.await_args_list]
        self.assertEqual(waits[:-1], [1, 2, 4, 8, 16, 30, 30])
        self.assertAlmostEqual(waits[-1], 9, delta=.5)
        self.assertEqual((outcome.model_status, len(model.requests)), ("model_timeout", 8))

    async def test_preparation_does_not_consume_budget_and_failed_close_does(self):
        clock = self.clock()
        wait = self.fast_retries()
        self.enterContext(patch.object(engine, "_MODEL_RETRY_BUDGET", 100))
        assemble = PreparedContext.assemble
        preparations = []

        async def prepare_once(prepared, workspace):
            preparations.append(prepared)
            clock.advance(110)
            return await assemble(prepared, workspace)

        async def error():
            try:
                raise ModelError("transport_error")
                yield
            finally:
                clock.advance(101)
                await asyncio.sleep(0)

        model = ScriptModel(error)
        runtime, history = self.setup_runtime(model, settings=RuntimeSettings(model_timeout=200))
        with patch.object(PreparedContext, "assemble", prepare_once):
            runtime.submit_messages(input_message())
            outcome = await self.outcome(runtime, history)
        self.assertEqual((outcome.model_status, len(model.requests), len(preparations)), ("model_timeout", 1, 1))
        wait.assert_not_awaited()

    async def test_preparation_error_is_not_a_model_retry_candidate(self):
        wait = self.fast_retries()
        model = ScriptModel(events(TextDelta("unexpected")))
        runtime, history = self.setup_runtime(model)
        with patch.object(PreparedContext, "assemble", side_effect=ModelError("transport_error")) as assemble:
            runtime.submit_messages(input_message())
            outcome = await self.outcome(runtime, history)
        self.assertEqual((outcome.model_status, len(model.requests)), ("model_failed", 0))
        assemble.assert_awaited_once()
        wait.assert_not_awaited()

    async def test_retry_does_not_assemble_or_schedule_summary_again(self):
        wait = self.fast_retries()
        prepare, assemble = Context.prepare, PreparedContext.assemble
        assemblies = []

        async def with_summary(context, *args, **kwargs):
            return replace(await prepare(context, *args, **kwargs), compactable_groups=1)

        async def assemble_once(prepared, workspace):
            assemblies.append(prepared)
            return await assemble(prepared, workspace)

        async def error():
            await asyncio.sleep(0)
            raise ModelError("transport_error")
            yield

        model = ScriptModel(error, events(TextDelta("ok")))
        runtime, history = self.setup_runtime(model)
        with (patch.object(Context, "prepare", with_summary),
              patch.object(PreparedContext, "assemble", assemble_once),
              patch.object(Context, "summary_request", return_value=None) as summary):
            runtime.submit_messages(input_message())
            outcome = await self.outcome(runtime, history)
        self.assertEqual((outcome.model_status, len(assemblies), len(model.requests)), ("complete", 1, 2))
        summary.assert_called_once()
        wait.assert_awaited_once_with(1)

    async def test_child_keeps_fixed_deadline_even_with_text(self):
        clock = self.clock()
        wait = self.fast_retries()
        history = self.enterContext(closing(History(":memory:")))
        parent = history.ensure_main_agent("main")
        self.assertTrue(history.start_agent_run(parent.agent_id, "parent-run", can_start=lambda: True))
        child, _ = history.accept_delegation(parent.agent_id, "parent-run", "call", role_id="review")

        async def long_stream():
            clock.advance(30)
            yield TextDelta("prefix")
            clock.advance(31)
            yield TextDelta("not accepted")

        model = ScriptModel(long_stream)
        runtime = engine.Runtime(child.session_id, history, model, None, agent=child, reply_to_parent=lambda _: True)
        self.addAsyncCleanup(runtime.close)
        runtime.submit_messages(input_message())
        outcome = await self.outcome(runtime, history)
        self.assertEqual((outcome.model_status, len(model.requests)), ("model_timeout", 1))
        wait.assert_not_awaited()

    async def test_shutdown_cancels_backoff_without_resending(self):
        sleeping, cancelled = asyncio.Event(), asyncio.Event()

        async def sleep(_):
            sleeping.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        wait = self.fast_retries(sleep)

        async def error():
            raise ModelError("transport_error")
            yield

        model = ScriptModel(error)
        runtime, _ = self.setup_runtime(model)
        runtime.submit_messages(input_message())
        await asyncio.wait_for(sleeping.wait(), 1)
        await runtime.close()
        self.assertTrue(cancelled.is_set())
        wait.assert_awaited_once_with(1)
        self.assertEqual(len(model.requests), 1)

    async def test_queued_input_preempts_unprompted_retry(self):
        failed = asyncio.Event()
        self.enterContext(patch.object(engine, "_MODEL_RETRY_INITIAL_DELAY", 30))

        async def unprompted():
            failed.set()
            raise ModelError("transport_error")
            yield

        model = ScriptModel(unprompted, events(TextDelta("real reply")))
        channel = Channel()
        runtime, history = self.setup_runtime(model, channel)
        history.append("main", [Message("user", "history")])
        runtime.trigger_run()
        await asyncio.wait_for(failed.wait(), 1)
        self.assertIsNotNone(runtime._trigger_retry_wake)
        runtime.submit_messages(input_message("real input"))
        await asyncio.wait_for(runtime.wait_idle(), 1)
        outcome, = history.outcomes("main")
        self.assertEqual(outcome.model_status, "complete")
        self.assertEqual([text for _, text in channel.replies], ["real reply"])
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(model.requests[0][0][-1].content, "history")
        self.assertEqual(model.requests[1][0][-1].content, "real input")

    async def test_queued_input_does_not_preempt_ordinary_retry(self):
        sleeping, release = asyncio.Event(), asyncio.Event()

        async def sleep(_):
            sleeping.set()
            await release.wait()

        wait = self.fast_retries(sleep)

        async def failure():
            raise ModelError("transport_error")
            yield

        model = ScriptModel(failure, events(TextDelta("first reply")), events(TextDelta("second reply")))
        runtime, _ = self.setup_runtime(model)
        runtime.submit_messages(input_message("first input"))
        await asyncio.wait_for(sleeping.wait(), 1)
        runtime.submit_messages(input_message("second input"))
        await asyncio.sleep(0)
        self.assertEqual(len(model.requests), 1)
        release.set()
        await asyncio.wait_for(runtime.wait_idle(), 1)
        self.assertEqual([request[0][-1].content for request in model.requests],
                         ["first input", "first input", "second input"])
        wait.assert_awaited_once_with(1)

    async def test_tool_write_keeps_private_metadata_only_for_local_deadline_cancellation(self):
        clock = self.clock()
        wait = self.fast_retries()
        marker = PrivateData(bytes(16), b"test-private")
        first, second = replace(CALL, private=marker), replace(CALL, id="second")
        for cause in ("deadline", "external_before_timer", "external_with_timer"):
            with self.subTest(cause=cause):
                calls = []

                async def tool(_args, context):
                    calls.append(context.call_id)
                    return "result"

                model = ScriptModel(events(AssistantPartsReady((Reasoning(marker), first, second)),
                                           ToolReady(first), ToolReady(second)))
                runtime, history = self.setup_runtime(model, tools=Tools([Tool(SPEC, tool)]))
                original_write = runtime._sqlite.write_input
                writes = 0

                async def write(operation):
                    nonlocal writes
                    if operation.__name__ == "persist_call":
                        writes += 1
                        if writes == 2:
                            reader = asyncio.current_task()
                            if cause == "external_before_timer":
                                reader.cancel()
                            elif cause == "external_with_timer":
                                # 同一轮事件循环：截止先发取消，外部也取消，之后读任务才恢复。
                                clock.loop.call_later(180.5, reader.cancel)
                            clock.advance(181)
                            await asyncio.Future()
                    return await original_write(operation)

                with patch.object(runtime._sqlite, "write_input", new=write):
                    runtime.submit_messages(input_message())
                    outcome = await self.outcome(runtime, history)
                message = history.read_after("main")[-1].message
                local = cause == "deadline"
                self.assertEqual(outcome.model_status, "model_timeout" if local else "model_cancelled")
                self.assertEqual(reasoning_parts(message), (marker,) if local else ())
                self.assertEqual(message.tool_calls, (replace(first, result="result", private=marker if local else None),))
                self.assertEqual(calls, [first.id])
                self.assertEqual(len(model.requests), 1)
        wait.assert_not_awaited()

    async def test_external_cancellation_racing_native_deadline_stays_cancellation(self):
        loop = asyncio.get_running_loop()

        def stop(reader):
            reader.cancel()  # 模拟截止回调运行时与外部发出的取消竞争。
            return False

        async def reader():
            async with model_deadline_at(loop.time(), stop):
                await asyncio.Future()

        with self.assertRaises(asyncio.CancelledError):
            await asyncio.create_task(reader())
