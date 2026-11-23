"""通道、运行时与模型共享的值和最薄接口。"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Text:
    text: str


@dataclass(frozen=True, slots=True)
class Image:
    url: str = field(repr=False)


Part = Text | Image
OMITTED_INPUT_IMAGE = Text("[未见图，请告知]")


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class ReplayScope:
    protocol: str
    fingerprint: bytes = field(repr=False)
    reasoning_effort: str


MAX_REPLAY_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ReplayItem:
    kind: Literal["reasoning", "message", "function_call"]
    item_id: str
    call_id: str | None = None
    phase: str | None = None
    # 文本内容只存于确认正文；原输出位置以类型和字符数表示。
    parts: tuple[tuple[str, int], ...] = ()
    encrypted_content: str | None = field(default=None, repr=False)
    summary: tuple[str, ...] = field(default=(), repr=False)


@dataclass(frozen=True, slots=True)
class ProviderReplay:
    scope: ReplayScope
    items: tuple[ReplayItem, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class ReplayReady:
    replay: ProviderReplay = field(repr=False)


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | tuple[Part, ...] = field(repr=False)
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    reasoning: str | None = field(default=None, repr=False)
    replay: ProviderReplay | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class ToolReady:
    """适配器已确认该调用的标识、名称及参数均已收齐，不再追加。"""

    call: ToolCall


ModelEvent = TextDelta | ReasoningDelta | ToolReady | ReplayReady


class ModelError(Exception):
    """模型协议或传输失败；仅允许已知诊断类别进入日志和存储。"""

    @property
    def diagnostic(self) -> str:
        code = self.args[0] if len(self.args) == 1 and type(self.args[0]) is str else ""
        if code in _MODEL_DIAGNOSTICS or re.fullmatch(r"http_[1-5][0-9]{2}", code):
            return code
        return "model_failed"


_MODEL_DIAGNOSTICS = frozenset({
    "abnormal_stream_finish", "duplicate_tool_id", "empty_summary", "incomplete_tool_call",
    "internal_tool_call", "invalid_sse_event", "invalid_stream_finish", "invalid_tool_arguments",
    "invalid_tool_delta", "invalid_tool_id", "missing_reasoning", "transport_error", "truncated_stream",
    "unsupported_tool_type", "request_body_too_large", "responses_output_limit", "responses_event_limit",
    "responses_unsupported_encoding", "responses_unsupported_output", "responses_unexpected_response",
    "summary_context_too_large", "summary_tool_call", "summary_private_echo", "agents_md_invalid_file",
    "agents_md_unreadable", "agents_md_too_large", "agents_md_invalid_utf8",
})


class RequestTooLarge(ModelError):
    """请求超出模型提供方的 HTTP 正文字节预算。"""


class Model(Protocol):
    def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, thinking: bool
    ) -> AsyncGenerator[ModelEvent, None]: ...


@runtime_checkable
class RequestBudget(Protocol):
    """可选：模型适配器的请求正文上限，独立于 Runtime 的 token 预算。"""

    def request_fits(self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, thinking: bool) -> bool: ...

    def remaining_request_bytes(self, messages: Sequence[Message], tools: Sequence[ToolSpec], *,
                                thinking: bool) -> int | None:
        """准确剩余字节数；None 表示当前请求已超限。"""
        ...

    def image_replacement_bytes(self, image: Image, placeholder: Text) -> int:
        """同一内容数组中用图片替换文字占位带来的正文字节增量。"""
        ...


class DeliveryStatus(StrEnum):
    FULL = "full"
    PARTIAL = "partial"
    ZERO = "zero"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Receipt:
    reply_id: str
    status: DeliveryStatus
    confirmed: str = field(default="", repr=False)


@dataclass(frozen=True, slots=True)
class ReplyInfo:
    reply_id: str
    session_id: str
    run_id: str
    # asyncio 事件循环单调时钟；早于模型请求和首个文本片段。
    started_at: float


class ReplyStream(AsyncIterator[str]):
    """单消费者文本流；首个片段发布时通道才收到流，纯工具响应不经过通道。"""

    def __init__(self, info: ReplyInfo) -> None:
        self.info = info
        self._segments: asyncio.Queue[str | None] = asyncio.Queue()
        self._closed = False
        self._exhausted = False
        self._abandoned = False
        self._published_chars = 0
        self._text_limit: int | None = None
        self._overflowed = False

    @property
    def abandoned(self) -> bool:
        return self._abandoned

    @property
    def overflowed(self) -> bool:
        return self._overflowed

    def limit_text(self, chars: int) -> None:
        """可选通道限额：不阻塞模型流，限制该回复还会排入多少文本。"""
        if type(chars) is not int or chars <= 0:
            raise ValueError("invalid reply text limit")
        self._text_limit = chars if self._text_limit is None else min(chars, self._text_limit)
        if self._published_chars > self._text_limit:
            self._overflowed = True
            self.abandon()

    def publish(self, text: str) -> None:
        if self._closed or not text:
            return
        if self._text_limit is not None and self._published_chars + len(text) > self._text_limit:
            self._overflowed = True
            self.close()  # 仍可消费已排队的受限前缀
            return
        self._published_chars += len(text)
        self._segments.put_nowait(text)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._segments.put_nowait(None)

    def abandon(self) -> None:
        if self._abandoned:
            return
        self._abandoned = True
        self._closed = True
        while not self._segments.empty():
            self._segments.get_nowait()
        self._segments.put_nowait(None)

    async def __anext__(self) -> str:
        if not self._exhausted:
            segment = await self._segments.get()
            if segment is not None:
                return segment
            self._exhausted = True
        raise StopAsyncIteration


class Channel(Protocol):
    async def deliver(self, reply: ReplyStream) -> Receipt:
        """消费一个回复并给出唯一终态，允许在模型流结束前返回。

        Runtime 从模型流关闭起用看门狗等待回执；通道应自行设定更短的时限，
        及时返回终态及已确认前缀，包括部分投递、已知失败和结果未知。
        Runtime 不推测通道进度，也不接纳超时后的回执。通道取消后只做清理，
        不再开始外部发送；若捕获取消，可用 ``reply.abandoned`` 判断该回复
        是否已放弃。
        """
        ...
