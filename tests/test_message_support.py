"""构造带有有序内容片段的测试消息。"""

from base64 import b64decode, b64encode
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_companion.protocol import Message, Reasoning, Text
from agent_companion.workspace import Workspace


# 合成 1x1 PNG，不使用真实用户媒体。
PNG_BYTES = b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGMw7XoHAAKlAa6/EWr0AAAAAElFTkSuQmCC")
PNG_URL = "data:image/png;base64," + b64encode(PNG_BYTES).decode("ascii")


def image_workspace(test):
    directory = TemporaryDirectory(dir=Path.cwd())
    test.addCleanup(directory.cleanup)
    return Workspace(Path(directory.name) / "user")


def assistant_message(text="", calls=(), *, reasoning=None):
    parts = ((Reasoning(reasoning),) if reasoning is not None else ()) + ((Text(text),) if text else ())
    return Message("assistant", parts + tuple(calls))


def reasoning_parts(message):
    return tuple(part.private for part in message.parts if isinstance(part, Reasoning))


def tool_calls(messages):
    return [call for message in messages for call in message.tool_calls]
