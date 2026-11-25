"""文件和实时话轮转写契约。"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import io
import re
import struct
from typing import Literal, Protocol
import wave


# 本地单文件字节保护；非 WAV 的时长仍由服务端校验。
MAX_AUDIO_BYTES = 50_000_000
AsrPhase = Literal["connect", "start", "send", "receive", "finish"]
_ASR_PHASES = frozenset({"connect", "start", "send", "receive", "finish"})
_ASR_DIAGNOSTICS = frozenset({
    "asr_adapter_cancelled", "asr_business_error", "asr_busy", "asr_empty_transcript", "asr_input_failed",
    "asr_invalid_response", "asr_premature_result", "asr_response_limit", "asr_timeout", "asr_transport",
    "asr_unsupported_audio_format", "invalid_audio_chunk", "invalid_audio_duration", "invalid_audio_size",
    "invalid_wav", "unsupported_audio_format",
})


class AsrError(Exception):
    """已脱敏的文件格式、识别协议或传输错误。"""

    def __init__(self, *args: object, phase: AsrPhase | None = None) -> None:
        super().__init__(*args)
        self._phase = phase

    @property
    def diagnostic(self) -> str:
        code = self.args[0] if len(self.args) == 1 and type(self.args[0]) is str else ""
        if code in _ASR_DIAGNOSTICS or re.fullmatch(r"asr_http_[1-5][0-9]{2}", code):
            return code
        return "asr_failed"

    @property
    def phase(self) -> AsrPhase | None:
        return self._phase if type(self._phase) is str and self._phase in _ASR_PHASES else None


@dataclass(frozen=True, slots=True)
class AudioFile:
    # 不信任扩展名或远端 Content-Type，提交接口前由文件头确定容器。
    data: bytes = field(repr=False)
    extension: str = field(init=False)
    content_type: str = field(init=False)
    duration_seconds: float | None = field(init=False)

    def __post_init__(self) -> None:
        data = self.data
        if not 0 < len(data) <= MAX_AUDIO_BYTES:
            raise AsrError("invalid_audio_size")
        extension, content_type, duration = _audio_format(data)
        object.__setattr__(self, "extension", extension)
        object.__setattr__(self, "content_type", content_type)
        object.__setattr__(self, "duration_seconds", duration)


def _audio_format(data: bytes) -> tuple[str, str, float | None]:
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(data), "rb") as file:
                frames = file.getnframes()
                sample_rate = file.getframerate()
                if frames <= 0 or sample_rate <= 0:
                    raise AsrError("invalid_audio_duration")
                # seek 到最后一帧，避免只认头部或为文件内容分配第二个完整副本。
                file.setpos(frames - 1)
                if len(file.readframes(1)) != file.getnchannels() * file.getsampwidth():
                    raise AsrError("invalid_wav")
                duration = frames / sample_rate
        except (wave.Error, EOFError, ValueError, RuntimeError, OverflowError, struct.error):
            # 标准库 wave 读取损坏的 chunk 时也可能直接抛 EOFError 或 RuntimeError。
            raise AsrError("invalid_wav") from None
        return "wav", "audio/wav", duration
    if len(data) >= 36 and data.startswith(b"OggS\x00") and data[26]:
        # 实际容器是 Ogg；飞书的 Opus 语音通常封装在此容器中。
        packet_start = 27 + data[26]
        if data.startswith((b"OpusHead", b"\x01vorbis", b"\x7fFLAC"), packet_start):
            return "ogg", "audio/ogg", None
    if len(data) >= 42 and data.startswith(b"fLaC"):
        return "flac", "audio/flac", None
    if len(data) >= 16 and data.startswith(b"FORM") and data[8:12] in (b"AIFF", b"AIFC"):
        return "aiff", "audio/aiff", None
    if len(data) >= 16 and data[4:8] == b"ftyp" and data[8:12] in (b"M4A ", b"M4B "):
        return "m4a", "audio/mp4", None
    if len(data) >= 7 and data[0] == 0xFF and data[1] & 0xF6 == 0xF0:
        return "aac", "audio/aac", None
    if len(data) >= 10 and (data.startswith(b"ID3") or (
        data[0] == 0xFF and data[1] & 0xE0 == 0xE0 and data[1] & 0x06 and data[2] >> 4 not in (0, 15)
    )):
        return "mp3", "audio/mpeg", None
    raise AsrError("unsupported_audio_format")


class FileASR(Protocol):
    async def transcribe(self, file: AudioFile) -> str: ...


class StreamingASR(Protocol):
    async def transcribe_stream(self, frames: AsyncIterator[bytes]) -> str:
        """持续读取非空、对齐的 16 kHz 单声道 s16le PCM，输入结束后只返回最终文本。

        输入由调用方拥有；`__anext__` 必须可取消并及时退出，不能吞掉取消后继续生产。
        只有完整话轮才应正常结束迭代，缺帧或挂断应取消调用。若取消后输入源未退出，
        适配器可拒绝新的实时话轮；调用方应保证该输入源最终退出。
        """
        ...


@dataclass(frozen=True, slots=True)
class ASRCapabilities:
    file: FileASR | None = None
    streaming: StreamingASR | None = None


class ASRAdapter(Protocol):
    def capabilities(self) -> ASRCapabilities:
        """返回当前配置可用的调用接口，不连接网络。"""
        ...
