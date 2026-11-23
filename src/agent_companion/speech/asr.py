"""完整音频文件转写契约；同时供语音条和本地话轮使用。"""

from dataclasses import dataclass, field
import io
import struct
from typing import Protocol
import wave


# MiniMax 文件接口的上限。WAV 可在本地校验时长，其它容器由服务端校验。
MAX_AUDIO_BYTES = 50_000_000
MAX_AUDIO_SECONDS = 500


class AsrError(Exception):
    """已脱敏的文件格式、识别协议或传输错误。"""


@dataclass(frozen=True, slots=True)
class AudioFile:
    # 不信任扩展名或远端 Content-Type，提交接口前由文件头确定容器。
    data: bytes = field(repr=False)
    extension: str = field(init=False)
    content_type: str = field(init=False)

    def __post_init__(self) -> None:
        data = self.data
        if not 0 < len(data) <= MAX_AUDIO_BYTES:
            raise AsrError("invalid_audio_size")
        extension, content_type = _audio_format(data)
        object.__setattr__(self, "extension", extension)
        object.__setattr__(self, "content_type", content_type)


def _audio_format(data: bytes) -> tuple[str, str]:
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(data), "rb") as file:
                frames = file.getnframes()
                if frames <= 0 or frames > MAX_AUDIO_SECONDS * file.getframerate():
                    raise AsrError("invalid_audio_duration")
                # seek 到最后一帧，避免只认头部或为文件内容分配第二个完整副本。
                file.setpos(frames - 1)
                if len(file.readframes(1)) != file.getnchannels() * file.getsampwidth():
                    raise AsrError("invalid_wav")
        except (wave.Error, EOFError, ValueError, RuntimeError, OverflowError, struct.error):
            # 标准库 wave 读取损坏的 chunk 时也可能直接抛 EOFError 或 RuntimeError。
            raise AsrError("invalid_wav") from None
        return "wav", "audio/wav"
    if len(data) >= 36 and data.startswith(b"OggS\x00") and data[26]:
        # 实际容器是 Ogg；飞书的 Opus 语音通常封装在此容器中。
        packet_start = 27 + data[26]
        if data.startswith((b"OpusHead", b"\x01vorbis", b"\x7fFLAC"), packet_start):
            return "ogg", "audio/ogg"
    if len(data) >= 42 and data.startswith(b"fLaC"):
        return "flac", "audio/flac"
    if len(data) >= 16 and data.startswith(b"FORM") and data[8:12] in (b"AIFF", b"AIFC"):
        return "aiff", "audio/aiff"
    if len(data) >= 16 and data[4:8] == b"ftyp" and data[8:12] in (b"M4A ", b"M4B "):
        return "m4a", "audio/mp4"
    if len(data) >= 7 and data[0] == 0xFF and data[1] & 0xF6 == 0xF0:
        return "aac", "audio/aac"
    if len(data) >= 10 and (data.startswith(b"ID3") or (
        data[0] == 0xFF and data[1] & 0xE0 == 0xE0 and data[1] & 0x06 and data[2] >> 4 not in (0, 15)
    )):
        return "mp3", "audio/mpeg"
    raise AsrError("unsupported_audio_format")


class FileASR(Protocol):
    async def transcribe(self, file: AudioFile) -> str: ...
