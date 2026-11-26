"""通道、运行时与模型共享的值和最薄接口。"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import json
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


REPLAY_KEY_BYTES = 16


@dataclass(frozen=True, slots=True)
class Replay:
    """完整响应的适配器私有数据。只有适配器可以比较 key 并解释 payload。"""

    replay_key: bytes = field(repr=False)
    payload: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.replay_key) is not bytes or len(self.replay_key) != REPLAY_KEY_BYTES:
            raise ValueError("invalid replay key")
        if type(self.payload) is not bytes:
            raise ValueError("invalid replay payload")


@dataclass(frozen=True, slots=True)
class ReplayReady:
    replay: Replay = field(repr=False)


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | tuple[Part, ...] = field(repr=False)
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    replay: Replay | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class PersistedInput:
    """已持久化的用户输入，交给 Runtime 在执行边界消费。"""

    batch_id: int


AGENT_REPLY_PREFIX = "[内部子任务结果]\n"


@dataclass(frozen=True, slots=True)
class DelegationTask:
    """已受理的父级委派；正文角色与可信来源分别处理。"""

    delegation_id: str
    prompt: str = field(repr=False)

    def to_model_message(self) -> Message:
        return Message("user", self.prompt)


@dataclass(frozen=True, slots=True)
class AgentReply:
    """直属子 Agent 的整轮回复，不是原委派调用的第二条工具消息。"""

    agent_id: str
    run_id: str
    delegation_ids: tuple[str, ...] = ()
    result: str | None = field(default=None, repr=False)
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.agent_id or not self.run_id or (self.result is None) == (self.error is None):
            raise ValueError("agent reply requires source IDs and exactly one result or error")

    def to_model_message(self) -> Message:
        payload = {"agent_id": self.agent_id, "run_id": self.run_id, "delegation_ids": self.delegation_ids}
        if self.error is not None:
            payload.update(status="failed", error=self.error)
        else:
            payload.update(status="completed", result=self.result)
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8", "backslashreplace").decode("utf-8")
        return Message("user", AGENT_REPLY_PREFIX + content)


TIMER_EVENT_PREFIX = "[内部定时器事件]\n"


@dataclass(frozen=True, slots=True)
class TimerFired:
    timer_id: str
    fire_count: int
    max_fires: int
    repeat: bool
    message: str | None = field(default=None, repr=False)

    def to_model_message(self) -> Message:
        completed = self.fire_count == self.max_fires
        if not self.repeat:
            description = "单次定时器已到期并完成，不再触发。"
        else:
            progress = f"定时器已触发（第 {self.fire_count}/{self.max_fires} 次），"
            description = progress + ("已达到最大次数，不再触发。" if completed else "还会继续触发。")
        payload = {
            "event": "timer_fired",
            "timer_id": self.timer_id,
            "fire_count": self.fire_count,
            "max_fires": self.max_fires,
            "status": "completed" if completed else "active",
            "description": description,
        }
        if self.message is not None:
            payload["message"] = self.message
        # JSON 可包含孤立代理项；让它保持转义，避免传输或 SQLite 拒绝整条输入。
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8", "backslashreplace").decode("utf-8")
        return Message("user", TIMER_EVENT_PREFIX + content)


@dataclass(frozen=True, slots=True)
class InternalInput:
    """可进入模型上下文，但不授予用户输入权限。"""

    content: Message | AgentReply | DelegationTask | TimerFired = field(repr=False)

    def to_model_message(self) -> Message:
        return self.content if isinstance(self.content, Message) else self.content.to_model_message()


# 未包装的 Message 表示真实用户输入。
InputItem = Message | PersistedInput | InternalInput


def validate_input_items(items: Sequence[InputItem]) -> None:
    """在调度器或 Runtime 修改队列前校验同一批输入。"""
    for item in items:
        if isinstance(item, PersistedInput):
            if type(item.batch_id) is not int:
                raise ValueError("persisted input requires an integer batch ID")
            continue
        message = item.content if isinstance(item, InternalInput) else item
        if isinstance(item, InternalInput) and isinstance(message, (AgentReply, DelegationTask, TimerFired)):
            continue
        if (not isinstance(message, Message) or message.role != "user" or message.tool_calls
                or message.tool_call_id is not None or message.replay is not None):
            raise ValueError("runtime inputs must be plain user messages")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolReady:
    """适配器已确认该调用的标识、名称及参数均已收齐，不再追加。"""

    call: ToolCall


@dataclass(frozen=True, slots=True)
class InputUsage:
    """服务端报告的整次请求实际输入 token 数。"""

    input_tokens: int


@dataclass(frozen=True, slots=True)
class ModelClosing:
    """适配器接下来收尾 HTTP 响应，不再解析模型帧。"""

    # False 表示关闭成功后仍有尚未交付的终态事件；True 表示已全部验证并交付。
    complete: bool


ModelEvent = TextDelta | ToolReady | ReplayReady | InputUsage | ModelClosing


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
    "invalid_tool_delta", "invalid_tool_id", "missing_reasoning", "transport_error", "transport_close_timeout",
    "truncated_stream", "vendor_error",
    "unsupported_tool_type", "responses_output_limit", "responses_event_limit",
    "responses_unsupported_encoding", "responses_unsupported_output", "responses_unexpected_response",
    "summary_tool_call", "incompatible_protected_history", "protected_history_already_compacted",
    "agents_md_invalid_file",
    "agents_md_unreadable", "agents_md_too_large", "agents_md_invalid_utf8",
})


class Model(Protocol):
    def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], *, thinking: bool
    ) -> AsyncGenerator[ModelEvent, None]: ...


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

    def __init__(self, info: ReplyInfo, *, validate_progress: Callable[[str], bool] | None = None) -> None:
        self.info = info
        self._segments: asyncio.Queue[str | None] = asyncio.Queue()
        self._progress: tuple[int, str] = (0, "")
        self._validate_progress = validate_progress
        self._delivery_deadline = 0.0
        self._idle_timeout: float | None = None
        self._arm_idle: Callable[[float, float | None], float] | None = None
        self._read_playback: Callable[[], tuple[float, float, str, float | None]] | None = None
        # Runtime 收到首个非空模型文本片段的单调时钟时间。
        self.first_text_at: float | None = None
        self.delivery_finished_at: float | None = None
        self._closed = False
        self._exhausted = False
        self._abandoned = False
        self._published_chars = 0
        self._text_limit: int | None = None
        self._overflowed = False
        self._voice_delivery = False

    @property
    def abandoned(self) -> bool:
        return self._abandoned

    @property
    def overflowed(self) -> bool:
        return self._overflowed

    @property
    def voice_delivery(self) -> bool:
        return self._voice_delivery

    def select_voice_delivery(self) -> None:
        """选定语音投递时（包括交给语音层前拦截），须在返回回执前调用。"""
        self._voice_delivery = True

    @property
    def progress(self) -> tuple[int, str]:
        """本条回复最新的设备消费字节数及已确认正文前缀。"""
        return self._progress

    def bind_playback(self, arm_idle: Callable[[float, float | None], float],
                      snapshot: Callable[[], tuple[float, float, str, float | None]]) -> None:
        """快照含观察时点、空闲截止、截止时的前缀和已发生的播放终态时点。"""
        self._arm_idle, self._read_playback = arm_idle, snapshot
        if self._idle_timeout is not None:
            arm_idle(self._idle_timeout, self._delivery_deadline)

    def start_delivery_wait(self, idle_timeout: float | None, fallback: float) -> None:
        self._idle_timeout = idle_timeout
        if idle_timeout is not None and self._arm_idle is not None:
            self._delivery_deadline = self._arm_idle(idle_timeout, None)
        else:
            timeout = fallback if idle_timeout is None else idle_timeout
            self._delivery_deadline = asyncio.get_running_loop().time() + timeout

    def delivery_snapshot(self) -> tuple[float, float, str, float | None]:
        if self._idle_timeout is not None and self._read_playback is not None:
            return self._read_playback()
        confirmed = self._progress[1] if self._idle_timeout is not None else ""
        return asyncio.get_running_loop().time(), self._delivery_deadline, confirmed, None

    def report_progress(self, reply_id: str, played_bytes: int, confirmed: str) -> None:
        """仅在事件循环线程上报；迟到、回退或重复的进展不能覆盖快照。"""
        previous_bytes, previous_text = self._progress
        if (self._abandoned or reply_id != self.info.reply_id or type(played_bytes) is not int
                or played_bytes <= 0 or played_bytes < previous_bytes or not isinstance(confirmed, str)
                or not confirmed.startswith(previous_text) or (played_bytes, confirmed) == self._progress):
            return
        if confirmed != previous_text and self._validate_progress is not None and not self._validate_progress(confirmed):
            return
        if self._idle_timeout is not None:
            now = asyncio.get_running_loop().time()
            if now >= self._delivery_deadline:
                return
            if played_bytes > previous_bytes:
                self._delivery_deadline = now + self._idle_timeout
        self._progress = (played_bytes, confirmed)

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
        self._validate_progress = None
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


@dataclass(frozen=True, slots=True)
class ChannelUser:
    """已由通道适配器核验的用户，以该适配器提供的稳定键标识。"""

    key: str = field(repr=False)


class Channel(Protocol):
    async def deliver(self, reply: ReplyStream) -> Receipt:
        """消费一个回复并给出唯一终态，允许在模型流结束前返回。

        Runtime 并行观察回执；只有选定语音投递的非完整回执可截断模型流。
        回执等待的空闲计时从模型流关闭开始。
        通道取消后只做清理，不再开始外部发送；若捕获取消，可查看
        ``reply.abandoned``。通道可另设不可续期的总时限。
        """
        ...


@dataclass(frozen=True, slots=True)
class ReplyDelivery:
    """一条回复在首次交接时选定的投递方式，之后不再查询连接。"""

    deliver: Callable[[ReplyStream], Awaitable[Receipt]]
    idle_timeout: float | None = None
    after_commit: Callable[[ReplyInfo, str], None] | None = None
    # 全文交接仅确认完整模型文字；实际发送由 after_commit 调度。
    require_complete_model: bool = False


@runtime_checkable
class SelectableChannel(Protocol):
    def select_delivery(self, reply: ReplyStream) -> ReplyDelivery:
        """同步为首段正文锁定交接方式，不等待 I/O。"""
        ...
