"""百炼语音 WebSocket：固定业务空间域名；连接与任务一一对应。"""

import asyncio
from dataclasses import dataclass, field
import json
import logging
import re
from types import TracebackType
from typing import Any
from uuid import uuid4

from websockets.asyncio.client import ClientConnection, connect


ASR_MODEL = "qwen-audio-3.1-asr-flash-message"
TTS_MODELS = frozenset({"qwen-audio-3.1-tts-flash", "cosyvoice-v3-flash"})
MAX_MESSAGE_BYTES = 512 * 1024
MAX_EVENTS = 32_768
_REGIONS = frozenset({"cn-beijing", "ap-southeast-1"})
# websockets 的 DEBUG 会打印鉴权头、音频和文本。不要继承应用的日志级别。
_WIRE_LOG = logging.Logger(__name__ + ".wire")
_WIRE_LOG.disabled = True


class _DirectConnect(connect):
    def process_redirect(self, exc: Exception) -> Exception:
        # websockets 15 默认跟随重定向并复用额外请求头。Bearer 只能发到选中的域名。
        return exc

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None,
                        traceback: TracebackType | None) -> None:
        # 每个连接只有一个任务；任务完成以协议事件为准。快速释放本地传输，不让
        # WebSocket 关闭握手占用业务时限，也不为每次关闭创建后台任务。
        self.connection.transport.abort()


@dataclass(frozen=True, slots=True)
class BailianConnection:
    api_key: str = field(repr=False)
    workspace_id: str
    region: str = "cn-beijing"

    def __post_init__(self) -> None:
        if not isinstance(self.region, str) or self.region not in _REGIONS:
            raise ValueError("invalid Bailian region")
        if not isinstance(self.workspace_id, str) or not re.fullmatch(
            r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", self.workspace_id
        ):
            raise ValueError("invalid Bailian workspace_id")
        if not isinstance(self.api_key, str) or not self.api_key or any(
            not 33 <= ord(char) <= 126 for char in self.api_key
        ):
            raise ValueError("invalid Bailian API key")

    def _open(self, timeout: float) -> _DirectConnect:
        return _DirectConnect(
            f"wss://{self.workspace_id}.{self.region}.maas.aliyuncs.com/api-ws/v1/inference",
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            proxy=None, compression=None, logger=_WIRE_LOG,
            open_timeout=timeout, close_timeout=min(timeout, 2.0),
            max_size=MAX_MESSAGE_BYTES, max_queue=4,
        )


class TaskFailed(Exception):
    """百炼明确报告任务失败；不保留原始错误消息。"""


class InvalidEvent(Exception):
    """收到无效或不符合当前任务状态的事件。"""


class SpeechTask:
    def __init__(self, socket: ClientConnection) -> None:
        self.socket = socket
        self._id = str(uuid4())

    async def send(self, action: str, payload: dict[str, Any]) -> None:
        await self.socket.send(json.dumps({
            "header": {"action": action, "task_id": self._id, "streaming": "duplex"},
            "payload": payload,
        }))

    def parse(self, raw: str | bytes) -> tuple[str, dict[str, Any]]:
        if not isinstance(raw, str):
            raise InvalidEvent
        try:
            message = json.loads(raw)
        except (ValueError, RecursionError):
            raise InvalidEvent from None
        if not isinstance(message, dict):
            raise InvalidEvent
        header = message.get("header")
        if not isinstance(header, dict) or header.get("task_id") != self._id:
            raise InvalidEvent
        event = header.get("event")
        if event == "task-failed":
            raise TaskFailed
        payload = message.get("payload")
        if not isinstance(event, str) or not isinstance(payload, dict):
            raise InvalidEvent
        return event, payload

    async def start(self, task: str, function: str, model: str, parameters: dict[str, Any], timeout: float) -> None:
        async with asyncio.timeout(timeout):
            await self.send("run-task", {
                "task_group": "audio", "task": task, "function": function, "model": model,
                "parameters": parameters, "input": {},
            })
            event, _ = self.parse(await self.socket.recv())
            if event != "task-started":
                raise InvalidEvent
