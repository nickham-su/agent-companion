"""流式语音合成契约：单段文字对应一条有序的 PCM 流。"""

from collections.abc import AsyncGenerator
import re
from typing import Protocol


class TtsError(Exception):
    """语音合成错误；向上只保留允许的诊断类别。"""

    @property
    def diagnostic(self) -> str:
        code = self.args[0] if len(self.args) == 1 and type(self.args[0]) is str else ""
        if code in _TTS_DIAGNOSTICS or re.fullmatch(r"tts_http_[1-5][0-9]{2}", code):
            return code
        return "tts_failed"


_TTS_DIAGNOSTICS = frozenset({
    "tts_business_error", "tts_empty_audio", "tts_event_limit", "tts_incomplete_stream", "tts_invalid_audio",
    "tts_invalid_event", "tts_invalid_json_response", "tts_invalid_pcm", "tts_invalid_pcm_chunk", "tts_invalid_status",
    "tts_json_response_limit", "tts_timeout", "tts_transport", "tts_unexpected_json_response",
    "tts_unsupported_content_encoding", "voice_reply_text_limit",
})


class StreamingTTS(Protocol):
    def synthesize(self, text: str, *, sample_rate: int) -> AsyncGenerator[bytes, None]:
        """PCM 为单声道、有符号 16-bit、小端；所有块都在样本边界对齐。"""
        ...
