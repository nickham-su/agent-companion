"""工作区图片的保存和请求投影；通道和 Runtime 共用格式与体积规则。"""

import asyncio
from base64 import b64encode
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import CancelledError as WorkCancelled, ThreadPoolExecutor
from contextlib import contextmanager, suppress
from contextvars import copy_context
from dataclasses import replace
import json
import os
import re
import stat
from threading import BoundedSemaphore, Event, Lock
from typing import Any, TypeVar
from uuid import uuid4

from .protocol import Image, Message, Part, Text, ToolCall
from .workspace import Workspace, open_child_directory


MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_MEDIA_BYTES = 32 * 1024 * 1024  # 给模型工具声明和协议封装留余量。
UNAVAILABLE_IMAGE = Text("图片不可用")
VIEW_IMAGE_TOOL_NAME = "view_image"
VIEW_IMAGE_OK = '{"status":"ok"}'
_IMAGE_PREFIX = ".agent/images/"
_IMAGE_NAME = re.compile(r"[a-f0-9]{32}")
_IMAGE_WORKERS = 4
# 进程内共享、按需起线程；不占用语音等任务的默认池。
_IMAGE_EXECUTOR = ThreadPoolExecutor(max_workers=_IMAGE_WORKERS, thread_name_prefix="agent-images")
_IMAGE_SLOTS = BoundedSemaphore(_IMAGE_WORKERS)
_IMAGE_GATE_LOCK = Lock()
_IMAGE_WAITERS: set[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = set()
_Result = TypeVar("_Result")
ToolCallOrder = Callable[[tuple[ToolCall, ...]], Sequence[str]]  # 同一批调用的出站 ID 顺序。


async def _acquire_image_slot() -> None:
    loop = asyncio.get_running_loop()
    while True:
        with _IMAGE_GATE_LOCK:
            if _IMAGE_SLOTS.acquire(blocking=False):
                return
            ready = asyncio.Event()
            waiter = (loop, ready)
            _IMAGE_WAITERS.add(waiter)
        # 在事件循环侧等额度；取消的等待者不会给线程池增加排队项。
        try:
            await ready.wait()
        finally:
            with _IMAGE_GATE_LOCK:
                _IMAGE_WAITERS.discard(waiter)


def _release_image_slot() -> None:
    with _IMAGE_GATE_LOCK:
        _IMAGE_SLOTS.release()
        waiters = tuple(_IMAGE_WAITERS)
        _IMAGE_WAITERS.clear()
    for loop, ready in waiters:
        # 读任务收尾时，等待它的事件循环可能已关闭。
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(ready.set)


def _discard_image_result(worker: asyncio.Future) -> None:
    if not worker.cancelled():
        worker.exception()


async def run_image_io(operation: Callable[[], _Result], *, settle_write: bool = False) -> _Result:
    """阻塞图片读写的唯一线程入口；取消等待不提前释放在途工作额度。"""
    abandoned = Event()
    context = copy_context()

    def work() -> _Result:
        if abandoned.is_set():
            raise WorkCancelled()
        return context.run(operation)

    await _acquire_image_slot()
    try:
        job = _IMAGE_EXECUTOR.submit(work)
    except BaseException:
        _release_image_slot()
        raise
    job.add_done_callback(lambda _: _release_image_slot())
    worker = asyncio.wrap_future(job)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        abandoned.set()
        while settle_write and not worker.done():
            # 飞书写入真的结束前，外层继续持有自己的媒体槽，不确认或交接批次。
            with suppress(asyncio.CancelledError):
                await asyncio.wait((worker,))
        if worker.done():
            _discard_image_result(worker)
        else:
            worker.add_done_callback(_discard_image_result)
        raise


def image_filename(reference: str) -> str | None:
    """历史只接受本地生成的引用，不接受 URL 或其它形式的路径。"""
    if not isinstance(reference, str) or not reference.startswith(_IMAGE_PREFIX):
        return None
    name = reference[len(_IMAGE_PREFIX):]
    return name if _IMAGE_NAME.fullmatch(name) else None


def image_media_type(data: bytes) -> str | None:
    size = len(data)
    if size > MAX_IMAGE_BYTES:
        return None
    if (size >= 45 and data.startswith(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
            and data.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82")):
        return "image/png"
    if size >= 12 and data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9"):
        return "image/jpeg"
    if size >= 14 and data.startswith((b"GIF87a", b"GIF89a")) and data.endswith(b";"):
        return "image/gif"
    if size >= 20 and data.startswith(b"RIFF") and data[8:16] in (
        b"WEBPVP8 ", b"WEBPVP8L", b"WEBPVP8X"
    ) and int.from_bytes(data[4:8], "little") + 8 == size:
        return "image/webp"
    return None


def image_request_bytes(size: int, media_type: str) -> int:
    # 与飞书原有预算一致：计算 data URL 全长和图片块的 JSON 包装。
    return len(f"data:{media_type};base64,") + 4 * ((size + 2) // 3) + 64


@contextmanager
def _image_directory(workspace: Workspace, *, create: bool) -> Iterator[int]:
    with workspace.open_directory(create=create) as (_, user_fd):
        agent_fd = open_child_directory(user_fd, ".agent", create=create)
        try:
            images_fd = open_child_directory(agent_fd, "images", create=create)
            try:
                yield images_fd
            finally:
                os.close(images_fd)
        finally:
            os.close(agent_fd)


def save_image(workspace: Workspace, data: bytes) -> Image:
    """调用方已校验媒体；只有完整写入后才发布由应用生成的相对引用。"""
    name = uuid4().hex
    with _image_directory(workspace, create=True) as directory:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        fd = os.open(name, flags, 0o600, dir_fd=directory)
        try:
            with os.fdopen(fd, "wb", closefd=False) as output:
                output.write(data)
        except OSError:
            with suppress(OSError):
                os.unlink(name, dir_fd=directory)
            raise
        finally:
            os.close(fd)
    return Image(_IMAGE_PREFIX + name)


def _read_regular_image(directory: int, name: str, remaining: int) -> bytes | None:
    # 先筛除特殊文件；打开后再查 FD，且即使换成 FIFO 也不会阻塞打开。
    if not stat.S_ISREG(os.stat(name, dir_fd=directory, follow_symlinks=False).st_mode):
        return None
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not 0 < info.st_size <= MAX_IMAGE_BYTES:
            return None
        # 最短 MIME 前缀也放不下就不读；文件在打开后增长也不能突破读取上限。
        if image_request_bytes(info.st_size, "image/png") > remaining:
            return None
        with os.fdopen(fd, "rb", closefd=False) as source:
            data = source.read(info.st_size + 1)
        return data if len(data) == info.st_size else None
    finally:
        os.close(fd)


def _read_image(workspace: Workspace, name: str, remaining: int) -> bytes | None:
    with _image_directory(workspace, create=False) as directory:
        return _read_regular_image(directory, name, remaining)


def view_image_path(args: dict[str, Any]) -> str | None:
    path = args.get("path")
    return path if len(args) == 1 and isinstance(path, str) and path else None


def _read_workspace_image(workspace: Workspace, path: str, remaining: int) -> bytes | None:
    # 先实际打开每一级再处理 `..`，不能让文本归一化掩盖软链接或不存在的目录。
    components = path.split("/")
    if path.startswith("/") or "\0" in path or components[-1] in ("", ".", ".."):
        return None
    with workspace.open_directory(create=False, search_only=True) as (_, root):
        directories = [root]
        try:
            for component in components[:-1]:
                if component in ("", "."):
                    continue
                if component == "..":
                    if len(directories) == 1:
                        return None
                    os.close(directories.pop())
                else:
                    directories.append(open_child_directory(directories[-1], component, create=False, search_only=True))
            return _read_regular_image(directories[-1], components[-1], remaining)
        finally:
            for directory in reversed(directories[1:]):
                os.close(directory)


def _workspace_image_data(workspace: Workspace, path: str, remaining: int) -> tuple[bytes, str] | None:
    try:
        data = _read_workspace_image(workspace, path, remaining)
    except (OSError, UnicodeError):
        return None
    if data is not None and (media_type := image_media_type(data)) is not None:
        return data, media_type
    return None


async def validate_workspace_image(workspace: Workspace, path: str) -> bool:
    return await run_image_io(lambda: _workspace_image_data(workspace, path, MAX_MEDIA_BYTES) is not None)


def _request_image(data: bytes, media_type: str, remaining: int) -> tuple[Image, int] | None:
    cost = image_request_bytes(len(data), media_type)
    if cost <= remaining:
        return Image(f"data:{media_type};base64,{b64encode(data).decode('ascii')}"), cost
    return None


def _load_image(part: Image, workspace: Workspace | None, remaining: int) -> tuple[Image | Text, int]:
    name = image_filename(part.url)
    if workspace is not None and name is not None:
        try:
            data = _read_image(workspace, name, remaining)
        except OSError:
            pass
        else:
            if data is not None and (media_type := image_media_type(data)) is not None:
                if loaded := _request_image(data, media_type, remaining):
                    return loaded
    return UNAVAILABLE_IMAGE, 0


def _successful_view_image(part: Part) -> bool:
    return isinstance(part, ToolCall) and part.name == VIEW_IMAGE_TOOL_NAME and part.result == VIEW_IMAGE_OK


def _load_view_image(call: ToolCall, workspace: Workspace, remaining: int) -> tuple[Image, int] | None:
    try:
        args = json.loads(call.arguments)
    except (ValueError, RecursionError):
        return None
    if isinstance(args, dict) and (path := view_image_path(args)) is not None:
        if image := _workspace_image_data(workspace, path, remaining):
            return _request_image(*image, remaining)
    return None


def _image_part_order(message: Message, tool_call_order: ToolCallOrder | None) -> Iterator[int]:
    calls = message.tool_calls if tool_call_order is not None else ()
    if tool_call_order is None or len(calls) < 2:
        yield from range(len(message.parts))
        return
    positions = {part.id: i for i, part in enumerate(message.parts) if isinstance(part, ToolCall)}
    ordered_ids = iter(tool_call_order(calls))
    # 只调整读取工具图片的顺序；通过调用 ID 找回原位置，不改正文、推理或工具顺序。
    for i, part in enumerate(message.parts):
        yield positions[next(ordered_ids)] if isinstance(part, ToolCall) else i


def prepare_images(messages: Sequence[Message], workspace: Workspace | None, *,
                   current_start: int = 0, tool_call_order: ToolCallOrder | None = None) -> Sequence[Message]:
    """仅加载本 Run 图片；上传与工具查看按消息顺序共用媒体预算。"""
    remaining = MAX_MEDIA_BYTES
    prepared = list(messages)
    for index in range(current_start, len(messages)):
        message = messages[index]
        if not any(isinstance(part, Image) or _successful_view_image(part) for part in message.parts):
            continue
        parts = list(message.parts)
        for position in _image_part_order(message, tool_call_order):
            part = message.parts[position]
            if isinstance(part, Image):
                part, cost = _load_image(part, workspace, remaining)
                remaining -= cost
            elif isinstance(part, ToolCall) and workspace is not None and _successful_view_image(part):
                if loaded := _load_view_image(part, workspace, remaining):
                    image, cost = loaded
                    part = replace(part, result_image=image)
                    remaining -= cost
            parts[position] = part
        prepared[index] = replace(message, parts=tuple(parts))
    return prepared


async def prepare_request_images(messages: Sequence[Message], workspace: Workspace | None, *,
                                 current_start: int = 0, tool_call_order: ToolCallOrder | None = None) -> Sequence[Message]:
    if not any(isinstance(part, Image) or _successful_view_image(part)
               for message in messages[current_start:] for part in message.parts):
        return messages
    return await run_image_io(lambda: prepare_images(messages, workspace, current_start=current_start,
                                                      tool_call_order=tool_call_order))
