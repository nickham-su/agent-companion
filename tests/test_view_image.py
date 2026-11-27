import asyncio
from base64 import b64decode, b64encode
from contextlib import closing
from dataclasses import replace
import json
import os
import unittest
from unittest.mock import patch

import httpx

from agent_companion import images
from agent_companion.app import running_agent
from agent_companion.config import RuntimeSettings, Settings
from agent_companion.images import MAX_IMAGE_BYTES, VIEW_IMAGE_OK, VIEW_IMAGE_TOOL_NAME, image_request_bytes, save_image
from agent_companion.models.bailian import Bailian
from agent_companion.models.deepseek import build_request
from agent_companion.models.minimax import MiniMax
from agent_companion.models.openai_responses import OpenAIResponses
from agent_companion.protocol import (ChannelUser, Image, InputUsage, Message, MessageDelivery, Text, TextDelta,
                                      ToolCall, ToolReady, ToolSpec)
from agent_companion.runtime.context import _IMAGE_COST, _SUMMARIZE, _cost
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.instructions import load_roles
from agent_companion.runtime.subtasks import DELEGATE_TOOL_NAME
from agent_companion.runtime.tools import Tool, ToolContext, Tools
from agent_companion.runtime.view_image import VIEW_IMAGE_TOOL
from agent_companion.storage.sqlite import History
from agent_companion.workspace import Workspace
from test_history_context import Sink
from test_message_support import PNG_BYTES, PNG_URL, image_workspace, tool_calls


_GIF_BYTES = b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
_GIF_URL = "data:image/gif;base64," + b64encode(_GIF_BYTES).decode("ascii")
_UNAVAILABLE = '{"error":"image_unavailable"}'
_OTHER_SPEC = ToolSpec("other", "Synthetic text tool", {"type": "object"})


def _view_call(call_id, path):
    return ToolCall(call_id, VIEW_IMAGE_TOOL_NAME, json.dumps({"path": path}))


def _write_image(workspace, path, data):
    file = workspace.directory / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(data)
    return file


async def _other(_args, _context):
    return VIEW_IMAGE_OK  # 仅字符串相同，工具名并不是 view_image。


class _Script:
    def __init__(self, events):
        self.events = events
        self.requests = []

    async def stream(self, messages, tools, *, thinking, instructions=None):
        self.requests.append((tuple(messages), tuple(tools), instructions))
        for event in self.events(self.requests):
            yield event


class ViewImageTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.workspace = image_workspace(self)
        self.tools = Tools([VIEW_IMAGE_TOOL])

    async def execute(self, call, *, workspace=None):
        return await self.tools.execute(call, ToolContext("main", "fixture-run", call.id, workspace=workspace))

    async def view(self, path):
        return await self.execute(_view_call("image", path), workspace=self.workspace)

    async def test_accepts_an_upload_reference_and_ordinary_relative_path(self):
        upload = save_image(self.workspace, PNG_BYTES)
        _write_image(self.workspace, "screenshots/frame.bin", _GIF_BYTES)  # 内容而非扩展名决定格式。
        with patch.object(images, "b64encode", side_effect=AssertionError("execution must not encode")):
            self.assertEqual(await self.view(upload.url), VIEW_IMAGE_OK)
            self.assertEqual(await self.view("./screenshots/frame.bin"), VIEW_IMAGE_OK)
        self.assertEqual(await self.execute(_view_call("unbound", upload.url)), _UNAVAILABLE)
        for arguments in ('{}', '{"path":1}', '{"path":""}', '{"path":"frame","extra":true}', '[]'):
            with self.subTest(arguments=arguments):
                call = ToolCall("invalid", VIEW_IMAGE_TOOL_NAME, arguments)
                self.assertEqual(await self.execute(call, workspace=self.workspace), '{"error":"invalid_arguments"}')

    async def test_accepts_backtracking_through_real_directories(self):
        _write_image(self.workspace, "screen.png", PNG_BYTES)
        _write_image(self.workspace, "screenshots/inner/frame.png", PNG_BYTES)
        for path in ("screenshots/../screen.png", "./screenshots//inner/../../screen.png",
                     "screenshots/inner/../inner/frame.png"):
            with self.subTest(path=path):
                self.assertEqual(await self.view(path), VIEW_IMAGE_OK)
                projected, = await images.prepare_request_images(
                    [Message("assistant", (replace(_view_call("image", path), result=VIEW_IMAGE_OK),))], self.workspace)
                self.assertEqual(projected.tool_calls[0].result_image, Image(PNG_URL))

    async def test_rejects_unsafe_paths_special_files_and_invalid_image_data(self):
        good = _write_image(self.workspace, "screenshots/frame.png", PNG_BYTES)
        sibling = self.workspace.directory.with_name("sibling")
        outside = _write_image(Workspace(sibling), "frame.png", PNG_BYTES)
        (self.workspace.directory / "in-file-link").symlink_to(good)
        (self.workspace.directory / "out-file-link").symlink_to(outside)
        (self.workspace.directory / "in-dir-link").symlink_to(good.parent, target_is_directory=True)
        (self.workspace.directory / "out-dir-link").symlink_to(sibling, target_is_directory=True)
        os.link(_write_image(self.workspace, "twin.png", PNG_BYTES), self.workspace.directory / "hard-link")
        os.mkfifo(self.workspace.directory / "pipe")
        _write_image(self.workspace, "invalid.png", b"not a picture")
        oversized = self.workspace.directory / "oversized.png"
        with oversized.open("wb") as output:
            output.write(PNG_BYTES)
            output.truncate(MAX_IMAGE_BYTES + 1)
        actual_open = os.open

        def nonblocking_open(file, flags, *args, **kwargs):
            # FIFO 应被打开前的文件类型检查挡住；若有所回归，不让测试自身阻塞。
            self.assertNotEqual(file, "pipe")
            return actual_open(file, flags, *args, **kwargs)

        paths = [str(good), "../sibling/frame.png", "screenshots/../../sibling/frame.png",
                 "screenshots/../../user/screenshots/frame.png", "absent/../screenshots/frame.png",
                 "in-dir-link/../screenshots/frame.png", "out-dir-link/../screenshots/frame.png",
                 "screenshots/frame.png/../../screenshots/frame.png",
                 "in-file-link", "out-file-link", "in-dir-link/frame.png", "out-dir-link/frame.png", "hard-link",
                 "screenshots", "screenshots/frame.png/", "screenshots/frame.png/.", "pipe", "absent", "invalid.png",
                 "oversized.png", "screenshots/frame.png\0", "\ud800"]
        with patch.object(images.os, "open", side_effect=nonblocking_open):
            for path in paths:
                with self.subTest(path=path):
                    self.assertEqual(await self.view(path), _UNAVAILABLE)

    async def test_rejects_fifo_swapped_between_stat_and_open(self):
        file = _write_image(self.workspace, "swap.png", PNG_BYTES)
        actual_open = os.open
        swapped = False

        def swap_then_open(name, flags, *args, **kwargs):
            nonlocal swapped
            if name == "swap.png" and not swapped:
                swapped = True
                self.assertTrue(flags & os.O_NONBLOCK)
                self.assertTrue(flags & os.O_NOFOLLOW)
                file.unlink()
                os.mkfifo(file)
            return actual_open(name, flags, *args, **kwargs)

        with patch.object(images.os, "open", side_effect=swap_then_open):
            self.assertEqual(await self.view("swap.png"), _UNAVAILABLE)
        self.assertTrue(swapped)

    async def test_directory_descriptors_close_when_traversal_or_final_open_fails(self):
        _write_image(self.workspace, "screenshots/inner/frame.png", PNG_BYTES)
        real_open = images.open_child_directory
        descriptors = []

        def open_child(*args, **kwargs):
            fd = real_open(*args, **kwargs)
            descriptors.append(fd)
            return fd

        # 同步验证，避免事件循环中其他任务重用被关闭的 FD。
        for path in ("screenshots/inner/missing/file.png", "screenshots/inner/../missing.png"):
            with self.subTest(path=path), patch.object(images, "open_child_directory", side_effect=open_child):
                descriptors.clear()
                with self.assertRaises(FileNotFoundError):
                    images._read_workspace_image(self.workspace, path, images.MAX_MEDIA_BYTES)
                self.assertEqual(len(descriptors), 2)
                for fd in descriptors:
                    with self.assertRaises(OSError):
                        os.fstat(fd)

    async def test_projection_shares_media_budget_and_holds_loaded_bytes(self):
        upload = save_image(self.workspace, PNG_BYTES)
        file = _write_image(self.workspace, "screen.png", PNG_BYTES)
        call = _view_call("screen", "screen.png")
        self.assertEqual(await self.execute(call, workspace=self.workspace), VIEW_IMAGE_OK)
        recorded = replace(call, result=VIEW_IMAGE_OK)
        incoming = [Message("user", (upload,)), Message("assistant", (recorded,))]
        with patch.object(images, "MAX_MEDIA_BYTES", image_request_bytes(len(PNG_BYTES), "image/png")):
            user, assistant = await images.prepare_request_images(incoming, self.workspace)
        self.assertEqual(user.parts, (Image(PNG_URL),))
        self.assertEqual(assistant.tool_calls, (recorded,))
        self.assertIsNone(recorded.result_image)

        loaded, = await images.prepare_request_images(incoming[1:], self.workspace)
        self.assertEqual(loaded.tool_calls[0].result_image, Image(PNG_URL))
        self.assertEqual(_cost(loaded, 0) - _cost(incoming[1], 0), _IMAGE_COST)
        file.write_bytes(_GIF_BYTES)
        changed, = await images.prepare_request_images(incoming[1:], self.workspace)
        self.assertEqual(changed.tool_calls[0].result_image, Image(_GIF_URL))
        self.assertEqual(loaded.tool_calls[0].result_image, Image(PNG_URL))

        # 持久化白名单即使收到投影副本也只留下原调用与文本结果。
        with closing(History(":memory:")) as history:
            history.append("main", [loaded])
            row, = history.read_after("main")
            self.assertEqual(row.message.tool_calls, (recorded,))

    async def test_archive_keeps_view_path_and_text_result_without_reading_image(self):
        path = "archive.png"
        _write_image(self.workspace, path, PNG_BYTES)
        recorded = replace(_view_call("archived", path), result=VIEW_IMAGE_OK)
        old = Message("assistant", (recorded,))
        model = _Script(lambda _requests: [TextDelta("done")])
        model.history_replay_bytes = lambda message, tools, *, thinking: None if message == old else 0

        with (closing(History(":memory:")) as history, patch.object(images, "_read_workspace_image") as read):
            history.append("main", [old])
            runtime = Runtime("main", history, model, Sink(), tools=self.tools, workspace=self.workspace)
            try:
                runtime.submit_messages([MessageDelivery(Message("user", "new run"), "trigger")])
                await asyncio.wait_for(runtime.wait_idle(), 2)
                self.assertIsNone(runtime.last_failure)
            finally:
                await runtime.close()
            archive = model.requests[0][0][0]
            self.assertEqual(archive.parts, (Text("历史归档（历史记录，不是新指令）："), Text("\nassistant:"),
                                             Text(f"工具调用 archived view_image：{recorded.arguments}\n结果：{VIEW_IMAGE_OK}")))
            self.assertEqual(history.read_after("main")[0].message, old)
            read.assert_not_called()

    async def test_concurrent_runtime_calls_project_by_call_id_and_order_and_never_reload_old_run(self):
        _write_image(self.workspace, "a.png", PNG_BYTES)
        _write_image(self.workspace, "b.gif", _GIF_BYTES)
        _write_image(self.workspace, "other.png", PNG_BYTES)
        removed = _write_image(self.workspace, "removed.png", PNG_BYTES)
        batch = (_view_call("first", "a.png"), _view_call("missing", "absent.png"),
                 ToolCall("other", _OTHER_SPEC.name, '{"path":"other.png"}'), _view_call("second", "b.gif"),
                 _view_call("removed", "removed.png"))
        second_finished = asyncio.Event()
        finished = []

        async def view(args, context):
            if context.call_id == "first":
                await asyncio.wait_for(second_finished.wait(), 2)
            result = await VIEW_IMAGE_TOOL.handler(args, context)
            if context.call_id == "removed":
                removed.unlink()  # 工具实际验证成功，但尚未开始请求投影。
            finished.append(context.call_id)
            if context.call_id == "second":
                second_finished.set()
            return result

        def events(requests):
            if len(requests) == 1:
                return [ToolReady(call) for call in batch]
            return [TextDelta("done")]

        model = _Script(events)
        with closing(History(":memory:")) as history:
            tools = Tools([Tool(VIEW_IMAGE_TOOL.spec, view), Tool(_OTHER_SPEC, _other)])
            runtime = Runtime("main", history, model, Sink(), tools=tools, workspace=self.workspace)
            try:
                with patch.object(images, "_read_workspace_image", wraps=images._read_workspace_image) as read:
                    runtime.submit_messages([MessageDelivery(Message("user", "first run"), "trigger")])
                    await asyncio.wait_for(runtime.wait_idle(), 3)
                    self.assertIsNone(runtime.last_failure)
                    self.assertEqual(len(model.requests), 2)
                    continuation = model.requests[1][0]
                    self.assertEqual([message.kind for message in continuation], ["user", "assistant"])
                    self.assertLess(finished.index("second"), finished.index("first"))
                    calls = tool_calls(continuation)
                    self.assertEqual([call.id for call in calls], [call.id for call in batch])
                    self.assertEqual([call.result_image for call in calls],
                                     [Image(PNG_URL), None, None, Image(_GIF_URL), None])
                    self.assertEqual([call.result for call in calls],
                                     [VIEW_IMAGE_OK, _UNAVAILABLE, VIEW_IMAGE_OK, VIEW_IMAGE_OK, VIEW_IMAGE_OK])
                    paths = [entry.args[1] for entry in read.call_args_list]
                    self.assertEqual(paths.count("absent.png"), 1)
                    self.assertNotIn("other.png", paths)

                    read.reset_mock()
                    runtime.submit_messages([MessageDelivery(Message("user", "next run"), "trigger")])
                    await asyncio.wait_for(runtime.wait_idle(), 3)
                    self.assertIsNone(runtime.last_failure)
                    read.assert_not_called()
            finally:
                await runtime.close()
            next_request = model.requests[-1][0]
            expected_records = [replace(call, result_image=None) for call in calls]
            self.assertEqual(tool_calls(next_request), expected_records)
            self.assertEqual(tool_calls(row.message for row in history.read_after("main")), expected_records)
            self.assertEqual(dict(history._db.execute("SELECT call_id, output FROM tool_results")),
                             {call.id: call.result for call in expected_records})
            for (parts,) in history._db.execute("SELECT parts FROM messages"):
                self.assertNotIn("base64", parts)
                self.assertNotIn("result_image", parts)

        await self.assert_adapter_requests(continuation, next_request)

    async def assert_adapter_requests(self, continuation, next_request):
        calls = tool_calls(continuation)
        async with httpx.AsyncClient() as client:
            responses = OpenAIResponses(client, model="fixture-vision", api_key="fixture", reasoning_effort="low")
            items = responses._payload(continuation, (VIEW_IMAGE_TOOL.spec,))["input"]
            outputs = [item for item in items if item.get("type") == "function_call_output"]
            self.assertEqual([item["call_id"] for item in outputs], [call.id for call in calls])
            for item, call in zip(outputs, calls):
                if call.result_image is None:
                    self.assertEqual(item["output"], call.result)
                else:
                    self.assertEqual(item["output"], [{"type": "input_text", "text": call.result},
                                                     {"type": "input_image", "image_url": call.result_image.url,
                                                      "detail": "auto"}])
            old_items = responses._payload(next_request, (VIEW_IMAGE_TOOL.spec,))["input"]
            self.assertEqual([item["output"] for item in old_items if item.get("type") == "function_call_output"],
                             [call.result for call in calls])

            minimax = MiniMax(client, model="fixture-vision", api_key="fixture")
            bailian = Bailian(client, model="fixture-vision", api_key="fixture",
                              base_url="https://example.invalid/compatible-mode/v1")
            builders = (lambda messages: build_request(messages, (), thinking=False),
                        lambda messages: minimax._payload(messages, (), False),
                        lambda messages: bailian._payload(messages, (), False))
            expected_tools = [{"role": "tool", "tool_call_id": call.id, "content": call.result} for call in calls]
            expected_images = {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": PNG_URL}},
                {"type": "image_url", "image_url": {"url": _GIF_URL}},
            ]}
            for build in builders:
                projected = build(continuation)["messages"]
                self.assertEqual([item["role"] for item in projected[:2]], ["user", "assistant"])
                self.assertEqual(projected[2:], expected_tools + [expected_images])
                old_projected = build(next_request)["messages"]
                self.assertEqual(old_projected[2:], expected_tools + [
                    {"role": "assistant", "content": "done"}, {"role": "user", "content": "next run"}])

    async def test_same_run_tool_image_is_loaded_into_summary_prefix_only_once_per_request(self):
        _write_image(self.workspace, "screen.png", PNG_BYTES)
        call = _view_call("view", "screen.png")
        summary_started = asyncio.Event()
        normal, summarized = [], []

        def events(requests):
            messages, _, _ = requests[-1]
            if messages[-1] == _SUMMARIZE:
                summarized.append(messages)
                summary_started.set()
                return [TextDelta("summary")]
            normal.append(messages)
            if len(normal) == 1:
                return [ToolReady(call), InputUsage(94)]
            if len(normal) == 2:
                return [ToolReady(ToolCall("tail", _OTHER_SPEC.name, "{}")), InputUsage(95)]
            return [TextDelta("done")]

        model = _Script(events)
        tools = Tools([VIEW_IMAGE_TOOL, Tool(_OTHER_SPEC, _other)])
        with closing(History(":memory:")) as history:
            runtime = Runtime("main", history, model, Sink(), tools=tools, workspace=self.workspace,
                              settings=RuntimeSettings(context_budget=100, tail_budget=8))
            try:
                with patch.object(images, "_read_workspace_image", wraps=images._read_workspace_image) as read:
                    runtime.submit_messages([MessageDelivery(Message("user", "start"), "trigger")])
                    await asyncio.wait_for(runtime.wait_idle(), 3)
                    await asyncio.wait_for(summary_started.wait(), 3)
                    self.assertIsNone(runtime.last_failure)
                    self.assertEqual(read.call_count, 3)  # 验证一次、两个普通续轮各一次、摘要不再读。
            finally:
                await runtime.close()
        self.assertEqual(len(normal), 3)
        summary, = summarized
        viewed, = tool_calls(summary)
        self.assertEqual(viewed, replace(call, result=VIEW_IMAGE_OK, result_image=Image(PNG_URL)))
        self.assertIs(viewed, tool_calls(normal[-1])[0])
        self.assertEqual([entry.id for entry in tool_calls(normal[-1])], [call.id, "tail"])

    async def test_main_and_child_with_role_opt_in_use_the_same_uploaded_path(self):
        root = self.workspace.directory
        agents = root / "agents"
        agents.mkdir(parents=True)
        for role, enabled in (("main", (DELEGATE_TOOL_NAME, VIEW_IMAGE_TOOL_NAME)), ("review", (VIEW_IMAGE_TOOL_NAME,))):
            (agents / f"{role}.md").write_text(
                f'+++\nmodel_id = "chat"\ndescription = "fixture"\ntools = {json.dumps(enabled)}\n+++\n{role}',
                encoding="utf-8")
        settings = Settings(workspace_root=root / "workspaces")
        user = ChannelUser("fixture-user")
        upload = save_image(Workspace.for_user(settings.workspace_root, user.key), PNG_BYTES)
        delegate = ToolCall("delegate", DELEGATE_TOOL_NAME,
                            json.dumps({"agent": "review", "prompt": upload.url}))
        first_seen = set()

        def events(requests):
            messages, tools, instructions = requests[-1]
            role = instructions.strip()
            if role in first_seen:
                return [TextDelta("done")]
            first_seen.add(role)
            specifies_image = [spec for spec in tools if spec.name == VIEW_IMAGE_TOOL_NAME]
            self.assertEqual(specifies_image, [VIEW_IMAGE_TOOL.spec])
            if role == "main":
                self.assertEqual(messages[-1].parts[-1], Text(f"图片路径：\n{upload.url}"))
                return [ToolReady(_view_call("main-view", upload.url)), ToolReady(delegate)]
            self.assertEqual(messages[-1].text, upload.url)
            self.assertEqual(messages[-1].source.kind, "delegation")
            return [ToolReady(_view_call("child-view", messages[-1].text))]

        model = _Script(events)
        with closing(History(":memory:")) as history:
            async with running_agent(settings, history, model, Sink(), session_id="main", roles=load_roles(root),
                                     user=user) as (runtime, tree):
                runtime.submit_messages([MessageDelivery(Message("user", (Text("start"), upload)), "trigger")])
                await asyncio.wait_for(tree.wait_idle(), 3)
                self.assertIsNone(runtime.last_failure)
        projected = {(instructions.strip(), call.id, call.result_image, call.result)
                     for messages, _, instructions in model.requests for call in tool_calls(messages)
                     if call.result_image is not None}
        self.assertEqual(projected, {("main", "main-view", Image(PNG_URL), VIEW_IMAGE_OK),
                                     ("review", "child-view", Image(PNG_URL), VIEW_IMAGE_OK)})
