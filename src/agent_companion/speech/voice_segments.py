"""可选分段转写的话轮缓冲，只保留尚未识别的音频。"""

from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass(slots=True, eq=False)
class SegmentedTurn:
    pcm: bytearray = field(default_factory=bytearray)
    asr_owner: UUID = field(default_factory=uuid4, init=False, repr=False)
    endpoint_at: float | None = None  # 长静默确认时间，仅用于日志计时。
    recorded_bytes: int = 0  # 累计收录的样本；成功识别、分段和补交都不重置。
    speech_end: int = 0  # 当前缓冲里最后一帧人声的末尾；用以排除纯静音尾段。
    parts: list[str] = field(default_factory=list)
    retry: bool = False
    queued: bool = False
    closed: bool = False
    discarded: bool = False

    @property
    def needs_transcription(self) -> bool:
        return bool(self.speech_end or self.retry)

    def record(self, pcm: bytes, *, hot: bool, limit: int) -> bool:
        size = min(len(pcm), limit - self.recorded_bytes)
        self.pcm.extend(pcm[:size])
        self.recorded_bytes += size
        if hot and size:
            self.speech_end = len(self.pcm)
        return self.recorded_bytes == limit

    def snapshot(self, end: int) -> bytes:
        return bytes(memoryview(self.pcm)[:end])

    def accept(self, end: int, text: str) -> None:
        self.parts.append(text)
        # 请求期间仍在录音。只移除该请求成功覆盖的前缀，保留其后的每个样本。
        del self.pcm[:end]
        self.speech_end = max(0, self.speech_end - end)
        self.retry = False

    def discard(self) -> None:
        self.discarded = True
        self.pcm.clear()
        self.parts.clear()
        self.speech_end = 0
        self.retry = False

    def transcript(self) -> str | None:
        return "\n".join(self.parts) or None


@dataclass(frozen=True, slots=True)
class SegmentJob:
    turn: SegmentedTurn
    early_bytes: int = 0  # 入队时的前缀边界，不能随在途期间的新录音移动。
