import asyncio
from collections import Counter
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import sqlite3
import tempfile
from threading import Event
from time import monotonic
import unittest
from unittest.mock import Mock, patch

import httpcore
import httpx

from agent_companion import images
from agent_companion.app import prepare_private_feishu, running_agent
from agent_companion.config import FeishuSettings, RuntimeSettings, Settings
from agent_companion.images import image_filename, save_image
from agent_companion.models import _http_stream
from agent_companion.models.deepseek import DeepSeek
from agent_companion.protocol import (ChannelUser, Image, InputUsage, Message, MessageDelivery, PrivateData,
                                      Text, TextDelta, ToolCall, ToolReady)
from agent_companion.runtime.subtasks import demo_roles
from agent_companion.runtime.compaction import Compactions, Prefix
from agent_companion.runtime.context import _SUMMARIZE
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.tools import Tool, ToolContext, Tools
from agent_companion.speech.asr import ASRCapabilities, FileASR
from agent_companion.storage.sqlite import History, ReplyOutcome, Summary
from test_history_context import Sink, SPEC
from test_message_support import PNG_BYTES, PNG_URL, assistant_message, image_workspace


# 这些调度测试让任意非空最新消息越过尾窗，避免因输入长度改变待摘要前缀。
_SETTINGS = RuntimeSettings(context_budget=240, tail_budget=8)
_SUMMARY = Message("compaction_summary", "background result")
_OLD = Message("user", "old history " + "a" * 500)


def _report_high_usage(history, session_id):
    history.finalize_reply(ReplyOutcome(f"prior-{session_id}", session_id, "seed", "complete", input_tokens=240), [])


async def _wait(event):
    await asyncio.wait_for(event.wait(), 2)


async def _idle(runtime):
    await asyncio.wait_for(runtime.wait_idle(), 2)


class _GatedModel:
    def __init__(self, reply=None, *, result_after_cancel=False):
        self.reply = reply
        self.result_after_cancel = result_after_cancel
        self.requests = []
        self.summary_requests = []
        self.generating = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def stream(self, messages, tools, *, thinking, instructions=None):
        if messages[-1] == _SUMMARIZE:
            self.summary_requests.append(tuple(messages))
            self.generating.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                if not self.result_after_cancel:
                    raise
            yield TextDelta("background result")
            return
        self.requests.append(tuple(messages))
        if self.reply is None:
            yield TextDelta("normal reply")
        else:
            async for event in self.reply(self.requests[-1]):
                yield event


class _NativeSummaryTransport:
    """真实 HTTPX/httpcore，底层连接和阻塞的关闭均在内存里。"""

    def __init__(self, *, complete=False):
        self.waiting, self.closing, self.release, self.closed, self.interrupted = (asyncio.Event() for _ in range(5))
        body = b'data: {"choices":[{"index":0,"delta":{"content":"background result"}}]}\n\n'
        if complete:
            body += b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        length = len(body) if complete else len(body) + 100
        packet = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n"
                  + f"Content-Length: {length}\r\n\r\n".encode() + body)
        fixture = self

        class Wire(httpcore.AsyncNetworkStream):
            sent = False

            async def read(self, max_bytes, timeout=None):
                if not self.sent:
                    self.sent = True
                    return packet
                fixture.waiting.set()
                await asyncio.Future()

            async def write(self, buffer, timeout=None):
                pass

            async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
                return self

            async def aclose(self):
                fixture.closing.set()
                try:
                    await fixture.release.wait()
                except asyncio.CancelledError:
                    fixture.interrupted.set()
                    raise
                fixture.closed.set()

        class Backend(httpcore.AsyncNetworkBackend):
            async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
                return Wire()

        pool = httpcore.AsyncConnectionPool
        with patch.object(httpcore, "AsyncConnectionPool", side_effect=lambda **kw: pool(network_backend=Backend(), **kw)):
            transport = httpx.AsyncHTTPTransport(trust_env=False)
        self.client = httpx.AsyncClient(transport=transport, trust_env=False)


class BackgroundCompactionTest(unittest.IsolatedAsyncioTestCase):
    async def _begin(self, runtime, model, text="first"):
        runtime.submit_messages([MessageDelivery(Message("user", text), "trigger")])
        await _wait(model.generating)
        task = runtime._compactions._generating
        self.assertIsNotNone(task)
        return task

    async def _finish(self, model, task):
        model.release.set()
        await asyncio.wait_for(asyncio.shield(task), 2)

    async def test_call_state_messages_are_visible_to_background_compaction(self):
        connected = Message("state_event", "通话已接通 - 2026-01-20 11:12:13+00:00")
        ended = Message("state_event", "通话已结束 - 2026-01-20 11:14:15+00:00")
        model = _GatedModel()
        with closing(History(":memory:")) as history:
            history.append("main", [connected, ended, _OLD])
            _report_high_usage(history, "main")
            runtime = Runtime("main", history, model, Sink(), settings=_SETTINGS)
            try:
                task = await self._begin(runtime, model, "当前消息")
                await _idle(runtime)
                plain, = model.requests
                compressed, = model.summary_requests
                self.assertEqual(plain[:3], (connected, ended, _OLD))
                self.assertEqual(compressed, (connected, ended, _OLD, _SUMMARIZE))
                await self._finish(model, task)
            finally:
                model.release.set()
                await runtime.close()

    async def test_normal_stream_enters_before_summary_tail_and_tool_schema_preparation(self):
        order = []
        release_normal = asyncio.Event()
        copied_tools = []

        async def reply(_messages):
            order.append("normal")
            await release_normal.wait()
            yield TextDelta("normal response")

        async def tool(_args, _ctx):
            return "not called"

        def freeze_tools(tools):
            order.append("schema")
            frozen = deepcopy(tools)
            copied_tools.extend(frozen)
            return frozen

        model = _GatedModel(reply)
        with closing(History(":memory:")) as history:
            history.append("main", [_OLD])
            _report_high_usage(history, "main")
            runtime = Runtime("main", history, model, Sink(), settings=_SETTINGS, tools=Tools([Tool(SPEC, tool)]))
            tail_cut = runtime._context._tail_cut

            def cut(*args):
                order.append("tail")
                return tail_cut(*args)

            try:
                with (patch.object(runtime._context, "_tail_cut", side_effect=cut),
                      patch("agent_companion.runtime.compaction.deepcopy", side_effect=freeze_tools)):
                    task = await self._begin(runtime, model)
                    self.assertEqual(order, ["normal", "tail", "schema"])
                    (copied,) = copied_tools
                    self.assertEqual(copied, SPEC)
                    self.assertIsNot(copied.parameters, SPEC.parameters)
                    release_normal.set()
                    await _idle(runtime)
                    self.assertFalse(task.done())
                    await self._finish(model, task)
            finally:
                release_normal.set()
                model.release.set()
                await runtime.close()

    async def test_invalid_normal_stream_does_not_send_summary_before_or_after_freezing(self):
        loop = asyncio.get_running_loop()
        clock = loop.time
        for failure in ("cancel_before_prepare", "expire_before_prepare", "expire_during_freeze"):
            with self.subTest(failure=failure), closing(History(":memory:")) as history:
                offset = 0

                async def reply(_messages):
                    nonlocal offset
                    if failure == "cancel_before_prepare":
                        asyncio.current_task().cancel()
                    elif failure == "expire_before_prepare":
                        offset = 2
                    await asyncio.Future()
                    yield TextDelta("unreachable")

                def freeze_tools(tools):
                    nonlocal offset
                    offset = 2
                    return deepcopy(tools)

                model = _GatedModel(reply)
                history.append("main", [_OLD])
                _report_high_usage(history, "main")
                runtime = Runtime("main", history, model, Sink(), settings=replace(_SETTINGS, model_timeout=1))
                try:
                    with (patch.object(loop, "time", side_effect=lambda: clock() + offset),
                          patch.object(runtime._context, "summary_request", wraps=runtime._context.summary_request) as plan,
                          patch("agent_companion.runtime.compaction.deepcopy", side_effect=freeze_tools) as freeze):
                        runtime.submit_messages([MessageDelivery(Message("user", "current input"), "trigger")])
                        await asyncio.wait_for(runtime.wait_idle(), 5)
                        if failure == "expire_during_freeze":
                            plan.assert_called_once()
                            freeze.assert_called_once()
                        else:
                            plan.assert_not_called()
                            freeze.assert_not_called()
                        self.assertEqual(len(model.requests), 1)
                        self.assertEqual(model.summary_requests, [])
                        self.assertIsNone(runtime._compactions._generating)
                        expected = "model_cancelled" if failure == "cancel_before_prepare" else "model_timeout"
                        self.assertEqual(runtime.last_failure, expected)
                finally:
                    await runtime.close()

    async def test_summary_schema_freeze_failure_does_not_fail_reply_and_starts_cooldown(self):
        async def reply(_messages):
            yield TextDelta("normal reply")
            yield InputUsage(240)

        model = _GatedModel(reply)
        with closing(History(":memory:")) as history:
            history.append("main", [_OLD])
            _report_high_usage(history, "main")
            runtime = Runtime("main", history, model, Sink(), settings=_SETTINGS)
            try:
                with (patch("agent_companion.runtime.compaction.deepcopy", side_effect=ValueError) as freeze,
                      self.assertLogs("agent_companion.runtime.compaction", "WARNING") as logs):
                    for text in ("first request", "second request"):
                        runtime.submit_messages([MessageDelivery(Message("user", text), "trigger")])
                        await _idle(runtime)
                        self.assertIsNone(runtime.last_failure)
                    freeze.assert_called_once()
                self.assertIn("summary_freeze_failed", logs.output[0])
                self.assertEqual(len(model.requests), 2)
                self.assertEqual(model.summary_requests, [])
            finally:
                await runtime.close()

    async def test_summary_uses_single_loaded_projection_and_does_not_delay_normal_reply(self):
        workspace = image_workspace(self)
        covered, old, recent = (save_image(workspace, PNG_BYTES) for _ in range(3))
        missing = Image(".agent/images/" + "a" * 32)
        old_message = Message("user", (Text("前"), old, missing, Text("后")))
        recent_message = Message("user", (recent,))
        model = _GatedModel()
        with (closing(History(":memory:")) as history,
              patch.object(images, "_read_image", wraps=images._read_image) as read):
            covered_id, old_id = history.append("main", [Message("user", (covered,)), old_message])
            history.save_summary("main", Summary(covered_id, "已压缩"))
            _report_high_usage(history, "main")
            runtime = Runtime("main", history, model, Sink(), settings=_SETTINGS, workspace=workspace)
            try:
                runtime.submit_messages([MessageDelivery(recent_message, "trigger")])
                await _wait(model.generating)
                await _idle(runtime)
                summary_task = runtime._compactions._generating
                self.assertIsNotNone(summary_task)
                self.assertFalse(summary_task.done())
                plain, = model.requests
                compressed, = model.summary_requests
                sent_old = (Text("前"), Text("后"), Text(f"图片路径：\n{old.url}\n{missing.url}"))
                self.assertEqual(plain[1].parts, sent_old)
                self.assertEqual(plain[2].parts, (Image(PNG_URL), Text(f"图片路径：\n{recent.url}")))
                self.assertEqual(compressed[1].parts, sent_old)
                self.assertEqual(compressed[2], _SUMMARIZE)
                self.assertIs(compressed[1], plain[1])
                # 本次 Run 的图在尾部，只随普通请求加载一次；旧的可用或缺失图片均不读盘。
                self.assertEqual([call.args[1] for call in read.call_args_list], [image_filename(recent.url)])
                await self._finish(model, summary_task)
                read.reset_mock()
                runtime.submit_messages([MessageDelivery(Message("user", "第二轮"), "trigger")])
                await _idle(runtime)
                self.assertEqual(history.summary("main").covers_through, old_id)
                read.assert_not_called()
                self.assertIn(Message("user", (Text(f"图片路径：\n{recent.url}"),)), model.requests[1])
                self.assertFalse(any(isinstance(part, Image) for message in model.requests[1] for part in message.parts))
                self.assertEqual([row.message for row in history.read_after("main")[:3]],
                                 [Message("user", (covered,)), old_message, recent_message])
            finally:
                model.release.set()
                await runtime.close()

    async def test_old_image_paths_count_as_text_in_summary_tail(self):
        workspace = image_workspace(self)
        old_picture = save_image(workspace, PNG_BYTES)
        crossing = Message("user", "older crossing group")
        next_input = Message("user", "next")
        model = _GatedModel()
        with (closing(History(":memory:")) as history,
              patch.object(images, "_read_image", wraps=images._read_image) as read):
            history.append("main", [_OLD, crossing, Message("user", (old_picture,))])
            _report_high_usage(history, "main")
            runtime = Runtime("main", history, model, Sink(), settings=replace(_SETTINGS, tail_budget=42),
                              workspace=workspace)
            try:
                runtime.submit_messages([MessageDelivery(next_input, "trigger")])
                await _wait(model.generating)
                await _idle(runtime)
                self.assertEqual(model.summary_requests, [(_OLD, _SUMMARIZE)])
                self.assertEqual(model.requests[0], (_OLD, crossing,
                                                     Message("user", f"图片路径：\n{old_picture.url}"), next_input))
                read.assert_not_called()
                await self._finish(model, runtime._compactions._generating)
            finally:
                await runtime.close()

    async def test_no_summary_prefix_does_not_start_task_or_read_images_twice(self):
        workspace = image_workspace(self)
        picture = save_image(workspace, PNG_BYTES)
        settings = RuntimeSettings(context_budget=3000, tail_budget=2200)
        model = _GatedModel()
        with (closing(History(":memory:")) as history,
              patch.object(images, "_read_image", wraps=images._read_image) as read):
            history.append("main", [Message("user", "kept text")])
            history.finalize_reply(ReplyOutcome("previous", "main", "seed", "complete", input_tokens=3000), [])
            runtime = Runtime("main", history, model, Sink(), settings=settings, workspace=workspace)
            try:
                with patch.object(runtime._compactions, "start", wraps=runtime._compactions.start) as start:
                    runtime.submit_messages([MessageDelivery(Message("user", (picture,)), "trigger")])
                    await _idle(runtime)
                    start.assert_not_called()
                self.assertEqual([call.args[1] for call in read.call_args_list], [image_filename(picture.url)])
                self.assertEqual(model.requests[0][-1].parts, (Image(PNG_URL), Text(f"图片路径：\n{picture.url}")))
                self.assertEqual(model.summary_requests, [])
                self.assertIsNone(runtime._compactions._generating)
            finally:
                await runtime.close()

    async def test_slow_image_wait_can_be_interrupted_without_starting_either_model_request(self):
        workspace = image_workspace(self)
        picture = Message("user", (save_image(workspace, PNG_BYTES),))
        loop = asyncio.get_running_loop()
        actual_read = images._read_image
        for cancel in (False, True):
            with self.subTest(cancel=cancel), closing(History(":memory:")) as history:
                history.append("main", [_OLD])
                _report_high_usage(history, "main")
                started, finished = asyncio.Event(), asyncio.Event()
                release = Event()

                def slow_read(*args):
                    loop.call_soon_threadsafe(started.set)
                    try:
                        if not release.wait(3):
                            raise AssertionError("image read blocked the event loop")
                        return actual_read(*args)
                    finally:
                        loop.call_soon_threadsafe(finished.set)

                model = Mock(spec=["stream", "stop_stream"])  # 图片准备时还没有可关闭的请求。
                model.stop_stream.return_value = False
                other_model = _GatedModel()
                settings = replace(_SETTINGS, model_timeout=5 if cancel else 0.15)
                runtime = Runtime("main", history, model, Sink(), settings=settings, workspace=workspace)
                other = Runtime("other", history, other_model, Sink())
                try:
                    with (patch.object(images, "_read_image", side_effect=slow_read) as read,
                          patch.object(runtime._compactions, "start", wraps=runtime._compactions.start) as start):
                        runtime.submit_messages([MessageDelivery(picture, "trigger")])
                        await _wait(started)
                        other.submit_messages([MessageDelivery(Message("user", "另一个用户"), "trigger")])
                        await _idle(other)
                        self.assertEqual(len(other_model.requests), 1)
                        if cancel:
                            await asyncio.wait_for(runtime.close(), 1)
                        else:
                            await _idle(runtime)
                            self.assertEqual(runtime.last_failure, "model_timeout")
                        self.assertFalse(finished.is_set())  # 取消等待不等于中止同步读取。
                        read.assert_called_once()
                        start.assert_not_called()
                        model.stream.assert_not_called()
                finally:
                    release.set()
                    await runtime.close()
                    await other.close()
                    if started.is_set():
                        await _wait(finished)
                model.stream.assert_not_called()

    async def test_already_expired_picture_request_does_not_start_either_model_request(self):
        workspace = image_workspace(self)
        picture = Message("user", (save_image(workspace, PNG_BYTES),))
        loop = asyncio.get_running_loop()
        clock = loop.time
        offset = 0

        async def prepare(messages, *_args, **_kwargs):
            nonlocal offset
            # 返回与检查间无让步，让单调时间先越过截止，而定时器还未运行。
            offset = 2
            return messages

        with closing(History(":memory:")) as history:
            history.append("main", [_OLD])
            _report_high_usage(history, "main")
            model = Mock(spec=["stream", "stop_stream"])
            model.stop_stream.return_value = False
            runtime = Runtime("main", history, model, Sink(), settings=replace(_SETTINGS, model_timeout=1))
            try:
                with (patch("agent_companion.runtime.context.prepare_request_images", side_effect=prepare),
                      patch.object(loop, "time", side_effect=lambda: clock() + offset),
                      patch.object(runtime._compactions, "start", wraps=runtime._compactions.start) as start):
                    runtime.submit_messages([MessageDelivery(picture, "trigger")])
                    await runtime.wait_idle()
                    self.assertEqual(runtime.last_failure, "model_timeout")
                    start.assert_not_called()
                model.stream.assert_not_called()
                model.stop_stream.assert_not_called()
            finally:
                await runtime.close()

    async def test_first_and_tool_continuation_do_not_wait_and_apply_only_after_full_batch(self):
        calls = [ToolCall(f"call{i}", SPEC.name, json.dumps({"i": i})) for i in range(3)]
        reached, release = [asyncio.Event() for _ in calls], [asyncio.Event() for _ in calls]
        tools_run = []

        async def tool(args, context):
            i = args["i"]
            tools_run.append(context.call_id)
            reached[i].set()
            await release[i].wait()
            return f"result{i}"

        async def reply(messages):
            index = len(model.requests)
            if index == 1:
                yield ToolReady(calls[0])
            elif index == 2:
                yield ToolReady(calls[1])
                yield ToolReady(calls[2])
            else:
                yield TextDelta("ready to reply")
            yield InputUsage(240)

        model = _GatedModel(reply)
        with closing(History(":memory:")) as history:
            old_id, = history.append("main", [_OLD])
            _report_high_usage(history, "main")
            runtime = Runtime("main", history, model, Sink(), settings=_SETTINGS, tools=Tools([Tool(SPEC, tool)]))
            try:
                task = await self._begin(runtime, model)
                await _wait(reached[0])
                self.assertEqual(model.requests[0], (_OLD, Message("user", "first")))
                runtime.submit_messages([MessageDelivery(Message("user", "queued during first batch"), "trigger")])
                release[0].set()
                await _wait(reached[1])
                await _wait(reached[2])
                self.assertEqual(len(model.requests), 2)
                self.assertEqual(model.requests[1][-1], Message("user", "queued during first batch"))
                self.assertFalse(task.done())
                release[1].set()
                await self._finish(model, task)
                self.assertEqual(len(model.requests), 2)
                self.assertIsNone(history.summary("main"))
                self.assertEqual(model.summary_requests, [(_OLD, _SUMMARIZE)])
                runtime.submit_messages([MessageDelivery(Message("user", "queued during second batch"), "trigger")])
                release[2].set()
                await _idle(runtime)
                self.assertIsNone(runtime.last_failure)
                self.assertEqual(history.summary("main"), Summary(old_id, "background result"))
                self.assertEqual(model.requests[2], (
                    _SUMMARY, Message("user", "first"), assistant_message("", (replace(calls[0], result="result0"),)),
                    Message("user", "queued during first batch"),
                    assistant_message(calls=(replace(calls[1], result="result1"),
                                             replace(calls[2], result="result2"))),
                    Message("user", "queued during second batch"),
                ))
                self.assertCountEqual(tools_run, [call.id for call in calls])
                self.assertEqual(len(model.summary_requests), 1)  # 本次提交不使用旧 usage 再生成。
                self.assertEqual(history.read_after("main")[0].message, _OLD)
            finally:
                await runtime.close()

    async def test_long_run_compacts_earlier_tool_group_but_not_latest_group_before_queued_input(self):
        workspace = image_workspace(self)
        initial_image, tail_image = (save_image(workspace, PNG_BYTES) for _ in range(2))
        calls = [ToolCall(f"long{i}", SPEC.name, json.dumps({"i": i})) for i in range(6)]
        batches = (calls[:2], calls[2:4], calls[4:5], calls[5:])
        reached, release = [asyncio.Event() for _ in calls], [asyncio.Event() for _ in calls]
        for i in (0, 1, 2, 5):
            release[i].set()
        executed = []

        async def tool(args, context):
            i = args["i"]
            executed.append((context.run_id, context.call_id))
            reached[i].set()
            await release[i].wait()
            return f"result{i}"

        async def reply(messages):
            round_no = len(model.requests)
            if round_no <= len(batches):
                for call in batches[round_no - 1]:
                    yield ToolReady(call)
                # 第一轮不触发；第二轮在 95% 线上，摘要前缀应包含同 Run 的第一批工具。
                yield InputUsage(228 if round_no == 2 else 227)
            else:
                yield TextDelta("finished")

        model = _GatedModel(reply)
        first = Message("user", (Text("start long run"), initial_image))
        queued = Message("user", (Text("queued after latest tool group " + "q" * 90), tail_image))
        sent_first = replace(first, parts=(first.parts[0], Image(PNG_URL), Text(f"图片路径：\n{initial_image.url}")))
        sent_queued = replace(queued, parts=(queued.parts[0], Image(PNG_URL), Text(f"图片路径：\n{tail_image.url}")))
        initial_name, tail_name = (image_filename(picture.url) for picture in (initial_image, tail_image))
        result_calls = tuple(replace(call, result=f"result{i}") for i, call in enumerate(calls))
        first_group = assistant_message(calls=result_calls[:2])
        second_group = assistant_message(calls=result_calls[2:4])
        third_group = assistant_message(calls=result_calls[4:5])
        fourth_group = assistant_message(calls=result_calls[5:])
        with (closing(History(":memory:")) as history,
              patch.object(images, "_read_image", wraps=images._read_image) as read):
            settings = replace(_SETTINGS, tail_budget=25)
            runtime = Runtime("main", history, model, Sink(), settings=settings, tools=Tools([Tool(SPEC, tool)]),
                              workspace=workspace)
            try:
                runtime.submit_messages([MessageDelivery(first, "trigger")])
                await _wait(reached[3])
                runtime.submit_messages([MessageDelivery(queued, "trigger")])
                release[3].set()
                await _wait(reached[4])
                await _wait(model.generating)
                self.assertEqual(model.summary_requests, [(sent_first, first_group, _SUMMARIZE)])
                self.assertEqual(model.requests[2], (sent_first, first_group, second_group, sent_queued))
                self.assertIs(model.summary_requests[0][0], model.requests[2][0])
                self.assertEqual(Counter(call.args[1] for call in read.call_args_list), {initial_name: 3, tail_name: 1})
                await self._finish(model, runtime._compactions._generating)
                self.assertIsNone(history.summary("main"))
                release[4].set()
                await _idle(runtime)
                self.assertIsNone(runtime.last_failure)
                self.assertEqual(model.requests[3], (_SUMMARY, second_group, sent_queued, third_group))
                self.assertEqual(model.requests[4], (_SUMMARY, second_group, sent_queued, third_group, fourth_group))
                self.assertEqual(Counter(call.args[1] for call in read.call_args_list), {initial_name: 3, tail_name: 3})
                entries = history.read_after("main")
                self.assertEqual(history.summary("main"), Summary(entries[1].message_id, "background result"))
                self.assertEqual([row.message for row in entries[:6]],
                                 [first, first_group, second_group, queued, third_group, fourth_group])
                self.assertEqual(len({run_id for run_id, _ in executed}), 1)
                self.assertCountEqual([call_id for _, call_id in executed], [call.id for call in calls])
            finally:
                await runtime.close()

    async def test_empty_wake_uses_separate_run_start_and_skips_first_summary_trigger(self):
        workspace = image_workspace(self)
        picture = save_image(workspace, PNG_BYTES)
        call = ToolCall("wake", SPEC.name, "{}")

        async def tool(args, context):
            return "wake result"

        async def reply(messages):
            yield ToolReady(call) if len(model.requests) == 1 else TextDelta("done")

        model = _GatedModel(reply)
        latest = Message("user", (Text("latest old input"), picture))
        sent_latest = replace(latest, parts=(latest.parts[0], Text(f"图片路径：\n{picture.url}")))
        with (closing(History(":memory:")) as history,
              patch.object(images, "_read_image", wraps=images._read_image) as read):
            _, latest_id, blank_id = history.append("main", [_OLD, latest, Message("assistant", ())])
            _report_high_usage(history, "main")
            runtime = Runtime("main", history, model, Sink(), settings=_SETTINGS, tools=Tools([Tool(SPEC, tool)]),
                              workspace=workspace)
            prepare = runtime._context.prepare
            contexts = []

            async def capture(*args, **kwargs):
                prepared = await prepare(*args, **kwargs)
                contexts.append(prepared)
                return prepared

            try:
                with patch.object(runtime._context, "prepare", side_effect=capture):
                    runtime.trigger_run()
                    await _idle(runtime)
                self.assertIsNone(runtime.last_failure)
                self.assertEqual(model.summary_requests, [])
                self.assertEqual(model.requests[0], (_OLD, sent_latest))
                self.assertEqual(model.requests[1][1], sent_latest)
                read.assert_not_called()
                first, followup = contexts
                self.assertEqual(first.protected_from, latest_id)
                self.assertEqual(first.run_start.message_id, blank_id + 1)
                self.assertEqual(followup.run_start, first.run_start)
                self.assertEqual(followup.protected_from, history.read_after("main")[-2].message_id)
            finally:
                await runtime.close()

    async def test_child_candidate_survives_idle_worker_and_is_not_used_by_other_sessions(self):
        first_child_text = "child old input " + "x" * 500

        async def reply(messages):
            if messages[-1].content == first_child_text:
                yield InputUsage(240)
            yield TextDelta("finished")

        model = _GatedModel(reply)
        settings = Settings(runtime=_SETTINGS)
        with closing(History(":memory:")) as history:
            async with running_agent(settings, history, model, Sink(), session_id="main",
                                     roles=demo_roles()) as (main, tree):
                agent_id = main.agent.agent_id
                fixture = MessageDelivery(Message("user", "fixture parent input"), "trigger")
                history.append_agent_inputs(agent_id, "test", (fixture,), start=True)

                async def delegate(label, prompt, child=None):
                    args = {"prompt": prompt, **({"agent_id": child} if child else {"agent": "general"})}
                    context = ToolContext("main", "test", label, agent_id=agent_id)
                    accepted = json.loads(await tree._delegate(args, context))
                    self.assertEqual(accepted["status"], "accepted")
                    return accepted["agent_id"]

                child_agent = await delegate("start", first_child_text)
                child = history.agent(child_agent).session_id
                await _idle(tree)
                await _idle(main)
                await delegate("generate", "child while generating", child_agent)
                await _wait(model.generating)
                task = main._compactions._generating
                self.assertIsNotNone(task)
                await _idle(tree)
                await _idle(main)
                self.assertTrue(tree._bindings[child_agent].runtime.idle)
                self.assertIsNone(history.summary(child))
                self.assertFalse(task.done())

                history.append("main", [_OLD])
                _report_high_usage(history, "main")
                main.submit_messages([MessageDelivery(Message("user", "main while child is generating"), "trigger")])
                await _idle(main)
                self.assertIn(_OLD, model.requests[-1])
                self.assertEqual(len(model.summary_requests), 1)
                await self._finish(model, task)
                self.assertIsNone(history.summary(child))

                other_agent = await delegate("other", "other child")
                other = history.agent(other_agent).session_id
                await _idle(tree)
                await _idle(main)
                self.assertIsNone(history.summary(other))
                self.assertIsNone(history.summary("main"))
                self.assertTrue(all(_SUMMARY not in request for request in model.requests))
                await delegate("reuse", "reuse original child", child_agent)
                await _idle(tree)
                await _idle(main)
                applied = [request for request in model.requests if _SUMMARY in request]
                self.assertEqual(len(applied), 1)
                self.assertIn("child while generating", [msg.text for msg in applied[0]])
                self.assertEqual(applied[0][-1].text, "reuse original child")
                self.assertIsNotNone(history.summary(child))
                self.assertEqual(len(model.summary_requests), 1)
                self.assertIn(first_child_text, [msg.text for msg in model.summary_requests[0]])
                self.assertNotIn(Message("user", "child while generating"), model.summary_requests[0])

    async def test_private_application_shares_one_generation_slot_across_groups(self):
        async def reply(messages):
            yield TextDelta("normal reply")
            yield InputUsage(240)

        model = _GatedModel(reply)
        settings = Settings(runtime=_SETTINGS, feishu=FeishuSettings())

        class Discovery:
            async def account_user(self):
                return ChannelUser("ou_fixture")

        class PrivateSink(Sink):
            async def send_plain_text(self, text):
                raise AssertionError("unexpected explicit text send")

        async def run_groups(_discovery, shared_history, _asr, *, open_agent, **kwargs):
            self.assertIs(shared_history, history)
            for session_id in ("main-a", "main-b"):
                history.append(session_id, [_OLD])
                _report_high_usage(history, session_id)

            async with open_agent(history.ensure_main_agent("main-b"), PrivateSink(), ChannelUser("ou_b")) as (second, _):
                async with open_agent(history.ensure_main_agent("main-a"), PrivateSink(), ChannelUser("ou_a")) as (first, _):
                    first_task = await self._begin(first, model, "first group input")
                    await _idle(first)
                    second.submit_messages([MessageDelivery(Message("user", "second group while occupied"), "trigger")])
                    await _idle(second)
                    self.assertIsNone(second.last_failure)
                    self.assertEqual(model.requests[-1][-1].content, "second group while occupied")
                    self.assertEqual(len(model.summary_requests), 1)
                    self.assertIs(first._compactions, second._compactions)
                self.assertFalse(first_task.done())
                self.assertFalse(model.cancelled.is_set())
                await self._finish(model, first_task)
                self.assertIsNone(history.summary("main-a"))
                model.generating.clear()
                model.release.clear()

                second_task = await self._begin(second, model, "second group gets released slot")
                await _idle(second)
                self.assertIsNone(second.last_failure)
                self.assertNotIn(_SUMMARY, model.requests[-1])
                self.assertEqual(len(model.summary_requests), 2)
            self.assertFalse(second_task.done())
            self.assertFalse(model.cancelled.is_set())

        with closing(History(":memory:")) as history:
            with patch("agent_companion.app.run_private_chats", side_effect=run_groups):
                async with prepare_private_feishu(settings, history, discovery=Discovery(), roles=demo_roles()) as (_, run_private):
                    await run_private(model, ASRCapabilities(file=Mock(spec=FileASR)), None)
            self.assertTrue(model.cancelled.is_set())
            self.assertIsNone(history.summary("main-b"))

    async def test_closing_runtime_preserves_shared_compaction_and_rechecks_before_apply(self):
        requesting, allow_read, reading = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def reply(messages):
            if messages[-1].content == "call":
                requesting.set()
                await asyncio.Event().wait()
            yield TextDelta("normal reply")

        model, sink = _GatedModel(reply), Sink()
        with closing(History(":memory:")) as history:
            history.append("main", [_OLD])
            _report_high_usage(history, "main")
            compactions = Compactions(history)

            def new_runtime():
                return Runtime("main", history, model, sink, settings=_SETTINGS, compactions=compactions)

            first, revoked, next_runtime = new_runtime(), new_runtime(), new_runtime()
            try:
                task = await self._begin(first, model, "call")
                await _wait(requesting)
                await first.close()
                self.assertFalse(task.done())
                self.assertFalse(model.cancelled.is_set())
                await self._finish(model, task)
                self.assertIsNone(history.summary("main"))

                original = revoked._context._sqlite

                async def slow_read(*args, **kwargs):
                    reading.set()
                    await allow_read.wait()
                    return await original(*args, **kwargs)

                with patch.object(revoked._context, "_sqlite", side_effect=slow_read):
                    revoked.submit_messages([MessageDelivery(Message("user", "revoked during preparation"), "trigger")])
                    await _wait(reading)
                    revoked.stop_accepting()
                    allow_read.set()
                    await _idle(revoked)
                    self.assertIsNone(history.summary("main"))
                    self.assertEqual(len(model.requests), 1)
                next_runtime.submit_messages([MessageDelivery(Message("user", "text afterwards"), "trigger")])
                await _idle(next_runtime)
                self.assertIn(_SUMMARY, model.requests[-1])
                self.assertIn(Message("user", "revoked during preparation"), model.requests[-1])
            finally:
                await asyncio.gather(first.close(), revoked.close(), next_runtime.close())
                await compactions.close()

    async def test_application_shutdown_cancels_generation_and_discards_late_result(self):
        model = _GatedModel(result_after_cancel=True)
        with closing(History(":memory:")) as history:
            history.append("main", [_OLD])
            _report_high_usage(history, "main")
            async with running_agent(Settings(runtime=_SETTINGS), history, model, Sink(), session_id="main",
                                     roles=demo_roles()) as (main, _):
                task = await self._begin(main, model)
                await _idle(main)
                self.assertFalse(task.done())
            self.assertTrue(model.cancelled.is_set())
            self.assertTrue(task.done())
            self.assertIsNone(main._compactions.ready("main", 0))
            self.assertIsNone(history.summary("main"))

    async def test_close_falls_back_to_cancel_if_adapter_has_no_matching_request(self):
        model = _GatedModel(result_after_cancel=True)
        with (closing(History(":memory:")) as history,
              patch.object(model, "stop_stream", return_value=False, create=True) as stop_stream):
            compactions = Compactions(history)
            try:
                compactions.start("main", Prefix(0, 1, 1), [_SUMMARIZE], (), model=model, thinking=False, timeout=5)
                await _wait(model.generating)
                task = compactions._generating
                await asyncio.wait_for(compactions.close(), 0.5)
                stop_stream.assert_called_once_with(task)
                self.assertTrue(model.cancelled.is_set())
                self.assertIsNone(compactions.ready("main", 0))
            finally:
                await compactions.close()

    async def test_native_summary_close_waiters_do_not_cancel_transport_cleanup(self):
        fixture = _NativeSummaryTransport(complete=True)
        async with fixture.client as client:
            with closing(History(":memory:")) as history:
                model = DeepSeek(client)
                compactions = Compactions(history)
                shutdown = []

                async def close_app():
                    try:
                        await compactions.close()
                    finally:
                        shutdown.append(("model", fixture.closed.is_set()))
                        await model.aclose()
                        shutdown.append(("client", fixture.closed.is_set()))
                        await client.aclose()

                waiters = []
                with patch.object(model, "stop_stream", wraps=model.stop_stream) as stop_stream:
                    try:
                        compactions.start("main", Prefix(0, 1, 1), [_SUMMARIZE], (),
                                          model=model, thinking=False, timeout=5)
                        reader = compactions._generating
                        await _wait(fixture.closing)
                        app = asyncio.create_task(close_app())
                        cancelled = asyncio.create_task(compactions.close())
                        observer = asyncio.create_task(compactions.close())
                        waiters.extend((app, cancelled, observer))
                        await asyncio.sleep(0)  # 三个等待方都已进入 close。
                        stop_stream.assert_called_once_with(reader)

                        for waiter in (app, cancelled, app):
                            self.assertTrue(waiter.cancel())
                            await asyncio.sleep(0)  # 每次取消分别投递，而非合并。
                        self.assertTrue(all(not waiter.done() for waiter in waiters))
                        self.assertFalse(fixture.interrupted.is_set())
                        self.assertEqual(shutdown, [])
                        self.assertFalse(client.is_closed)
                    finally:
                        fixture.release.set()
                        outcomes = await asyncio.wait_for(asyncio.gather(*waiters, return_exceptions=True), 2)
                        await compactions.close()
                        await model.aclose()

                self.assertTrue(fixture.closed.is_set())
                self.assertFalse(fixture.interrupted.is_set())
                self.assertIsInstance(outcomes[0], asyncio.CancelledError)
                self.assertIsInstance(outcomes[1], asyncio.CancelledError)
                self.assertIsNone(outcomes[2])
                self.assertEqual(shutdown, [("model", True), ("client", True)])

    async def test_native_summary_close_starts_transport_budget(self):
        fixture = _NativeSummaryTransport()
        async with fixture.client as client:
            model = DeepSeek(client)
            with closing(History(":memory:")) as history:
                compactions = Compactions(history)
                prefix = Prefix(0, 1, 1)
                stopping = None
                try:
                    compactions.start("main", prefix, [_SUMMARIZE], (), model=model, thinking=False, timeout=5)
                    await _wait(fixture.waiting)
                    with (patch.object(_http_stream, "_RESPONSE_CLOSE_TIMEOUT", 0.02),
                          self.assertLogs(_http_stream.__name__, "WARNING") as logged):
                        stopping = asyncio.create_task(compactions.close())
                        await _wait(fixture.closing)
                        await asyncio.wait_for(asyncio.shield(stopping), 0.5)
                    self.assertTrue(fixture.interrupted.is_set())
                    self.assertFalse(fixture.closed.is_set())
                    self.assertIn("transport_close_timeout", "\n".join(logged.output))
                    self.assertIsNone(compactions.ready("main", 0))
                finally:
                    fixture.release.set()
                    if stopping is not None:
                        await stopping
                    await compactions.close()
                    await model.aclose()

    async def test_native_summary_result_is_only_registered_while_open(self):
        for stopped in (False, True):
            with self.subTest(stopped=stopped):
                fixture = _NativeSummaryTransport(complete=True)
                async with fixture.client as client:
                    model = DeepSeek(client)
                    with closing(History(":memory:")) as history:
                        compactions = Compactions(history)
                        prefix = Prefix(0, 1, 1)
                        try:
                            compactions.start("main", prefix, [_SUMMARIZE], (), model=model, thinking=False, timeout=5)
                            task = compactions._generating
                            await _wait(fixture.closing)
                            self.assertFalse(compactions.can_start("another", 0, 1))
                            if stopped:
                                compactions.stop()
                            fixture.release.set()
                            await asyncio.wait_for(asyncio.shield(task), 0.5)
                            result = None if stopped else (prefix, Summary(1, "background result"))
                            self.assertEqual(compactions.ready("main", 0), result)
                            self.assertEqual(compactions.can_start("another", 0, 1), not stopped)
                            self.assertTrue(fixture.closed.is_set())
                        finally:
                            fixture.release.set()
                            await compactions.close()
                            await model.aclose()

    async def test_native_generation_timeout_starts_transport_budget(self):
        for summary in (False, True):
            with self.subTest(summary=summary):
                fixture = _NativeSummaryTransport()
                async with fixture.client as client:
                    model = DeepSeek(client)
                    with closing(History(":memory:")) as history:
                        runtime = Runtime("main", history, model, Sink(), settings=replace(_SETTINGS, model_timeout=0.05))
                        compactions = runtime._compactions
                        try:
                            with (patch.object(_http_stream, "_RESPONSE_CLOSE_TIMEOUT", 0.02),
                                  self.assertLogs("agent_companion", "WARNING") as logged):
                                if summary:
                                    compactions.start("main", Prefix(0, 1, 1), [_SUMMARIZE], (), model=model, thinking=False,
                                                      timeout=0.05)
                                    settled = asyncio.shield(compactions._generating)
                                else:
                                    runtime.submit_messages([MessageDelivery(Message("user", "fixture input"), "trigger")])
                                    settled = runtime.wait_idle()
                                await asyncio.wait_for(settled, 0.5)
                            self.assertTrue(fixture.interrupted.is_set())
                            self.assertIn("transport_close_timeout", "\n".join(logged.output))
                            if summary:
                                self.assertIsNone(compactions.ready("main", 0))
                                self.assertFalse(compactions.can_start("main", 0, 1))
                            else:
                                self.assertEqual(runtime.last_failure, "model_timeout")
                        finally:
                            fixture.release.set()
                            await runtime.close()
                            await model.aclose()

    async def test_summary_timeout_cools_down_and_discards_late_result(self):
        async def reply(messages):
            yield TextDelta("normal reply")
            yield InputUsage(240)

        model = _GatedModel(reply, result_after_cancel=True)
        with closing(History(":memory:")) as history:
            history.append("main", [_OLD])
            _report_high_usage(history, "main")
            runtime = Runtime("main", history, model, Sink(), settings=replace(_SETTINGS, model_timeout=0.1))
            try:
                with self.assertLogs("agent_companion.runtime.compaction", level="WARNING") as logs:
                    task = await self._begin(runtime, model)
                    await _idle(runtime)
                    self.assertIsNone(runtime.last_failure)
                    await asyncio.wait_for(asyncio.shield(task), 2)
                self.assertTrue(model.cancelled.is_set())
                self.assertIn("timeout", logs.output[0])
                runtime.submit_messages([MessageDelivery(Message("user", "during timeout cooldown"), "trigger")])
                await _idle(runtime)
                self.assertIsNone(history.summary("main"))
                self.assertEqual(len(model.summary_requests), 1)
                self.assertNotIn(_SUMMARY, model.requests[-1])
            finally:
                await runtime.close()

    async def test_sqlite_write_lock_does_not_delay_or_apply_candidate_to_primary_request(self):
        model = _GatedModel()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compaction.sqlite"
            with closing(History(path)) as history, closing(sqlite3.connect(path)) as fault:
                history.append("main", [_OLD])
                _report_high_usage(history, "main")
                runtime = Runtime("main", history, model, Sink(), settings=_SETTINGS)
                try:
                    await self._finish(model, await self._begin(runtime, model))
                    await _idle(runtime)
                    self.assertIsNone(history.summary("main"))
                    attempt = history.try_save_summary
                    observations = []

                    original_timeout = history._db.execute("PRAGMA busy_timeout").fetchone()[0]

                    def locked(*args, **kwargs):
                        fault.execute("BEGIN IMMEDIATE")
                        started = monotonic()
                        try:
                            return attempt(*args, **kwargs)
                        finally:
                            observations.append((monotonic() - started, history._db.execute("PRAGMA busy_timeout").fetchone()[0]))
                            fault.rollback()

                    with (patch.object(history, "try_save_summary", side_effect=locked),
                          self.assertLogs("agent_companion.runtime.compaction", level="WARNING") as logs):
                        runtime.submit_messages([MessageDelivery(Message("user", "while write locked"), "trigger")])
                        await _idle(runtime)
                    elapsed, restored = observations[0]
                    self.assertLess(elapsed, 1)  # 普通连接等待上限为 5 秒。
                    self.assertEqual(restored, original_timeout)
                    self.assertEqual(len(observations), 1)
                    self.assertIn("summary_write_failed", logs.output[0])
                    self.assertIsNone(runtime.last_failure)
                    self.assertIsNone(history.summary("main"))
                    self.assertIn(_OLD, model.requests[-1])
                    self.assertNotIn(_SUMMARY, model.requests[-1])
                finally:
                    await runtime.close()

    async def test_sqlite_cas_conflict_refreshes_same_request_unless_locked_or_unsafe(self):
        for mode in ("fresh", "locked", "protected"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "compaction.sqlite"
                model = _GatedModel()
                compatible = PrivateData(b"a" * 16, b"fixture-reason")
                incompatible = PrivateData(b"b" * 16, b"different-reason")
                with closing(History(path)) as history, closing(History(path)) as other:
                    first_old = Message("user", "already summarized " + "b" * 500)
                    competing_id, target = history.append("main", [first_old, _OLD])
                    _report_high_usage(history, "main")
                    runtime = Runtime("main", history, model, Sink(), settings=_SETTINGS)
                    try:
                        await self._finish(model, await self._begin(runtime, model))
                        await _idle(runtime)
                        save = history.try_save_summary
                        read = history.try_context_snapshot
                        competing_text = "other connection summary"
                        concurrently_appended = Message("user", "new input from the other connection")
                        racing_input = Message("user", "racing summary")
                        matching_reply = assistant_message("matching provider replay", reasoning=compatible)
                        alternate_reply = assistant_message("other provider replay", reasoning=incompatible)
                        reads = []
                        original_timeout = history._db.execute("PRAGMA busy_timeout").fetchone()[0]

                        def commit_first(session_id, candidate, *, covered, protected_from):
                            self.assertEqual(candidate.covers_through, target)
                            through = protected_from if mode == "protected" else competing_id
                            other.save_summary(session_id, Summary(through, competing_text))
                            other.append(session_id, [concurrently_appended])
                            for index, message in enumerate((matching_reply, alternate_reply)):
                                other.finalize_reply(ReplyOutcome(f"concurrent-{index}", session_id, "test", "complete"),
                                                     [message])
                            other.append("child", [Message("user", "different session")])
                            saved = save(session_id, candidate, covered=covered, protected_from=protected_from)
                            self.assertFalse(saved)
                            if mode == "locked":
                                other._db.execute("BEGIN EXCLUSIVE")
                            return saved

                        def read_once(*args, **kwargs):
                            started = monotonic()
                            try:
                                return read(*args, **kwargs)
                            finally:
                                elapsed = monotonic() - started
                                reads.append((elapsed, history._db.execute("PRAGMA busy_timeout").fetchone()[0]))
                                if mode == "locked":
                                    other._db.rollback()

                        with (patch.object(history, "try_save_summary", side_effect=commit_first) as write,
                              patch.object(history, "try_context_snapshot", side_effect=read_once)):
                            runtime.submit_messages([MessageDelivery(racing_input, "trigger")])
                            await _idle(runtime)
                            write.assert_called_once()
                        self.assertEqual(len(reads), 1)
                        self.assertEqual(reads[0][1], original_timeout)
                        self.assertEqual(history.summary("main").text, competing_text)
                        if mode == "protected":
                            self.assertEqual(runtime.last_failure, "context_failed")
                            self.assertEqual(len(model.requests), 1)  # 不发送吞掉当前输入的投影。
                            continue
                        self.assertIsNone(runtime.last_failure)
                        latest = model.requests[1]  # 就在冲突这一轮，而非下一轮。
                        common = (_OLD, Message("user", "first"), Message("assistant", "normal reply"), racing_input)
                        if mode == "locked":
                            self.assertLess(reads[0][0], 1)
                            self.assertEqual(latest, (first_old, *common))
                        else:
                            self.assertEqual(latest, (Message("compaction_summary", competing_text),
                                                      *common, concurrently_appended, matching_reply,
                                                      alternate_reply))
                        self.assertNotIn(_SUMMARY, latest)
                    finally:
                        await runtime.close()

    async def test_expired_candidate_is_not_committed(self):
        model = _GatedModel()
        with (closing(History(":memory:")) as history,
              patch("agent_companion.runtime.compaction.monotonic", return_value=100) as clock):
            history.append("main", [_OLD])
            _report_high_usage(history, "main")
            runtime = Runtime("main", history, model, Sink(), settings=_SETTINGS)
            try:
                await self._finish(model, await self._begin(runtime, model))
                await _idle(runtime)
                clock.return_value = 401
                with patch.object(history, "try_save_summary", wraps=history.try_save_summary) as write:
                    runtime.submit_messages([MessageDelivery(Message("user", "after expiration"), "trigger")])
                    await _idle(runtime)
                    write.assert_not_called()
                self.assertIsNone(history.summary("main"))
                self.assertIn(_OLD, model.requests[-1])
                self.assertNotIn(_SUMMARY, model.requests[-1])
            finally:
                await runtime.close()
