"""MiniMax ASR：一次 multipart 完整文件请求，不重试。"""

import asyncio
import json

import httpx

from ..config import MINIMAX_ASR_MODEL, require_asr_audio_seconds, require_seconds
from .asr import AsrError, AudioFile


_ENDPOINTS = {
    "cn": "https://api.minimax.cn/v1/speech_to_text",
    "global": "https://api.minimax.io/v1/speech_to_text",
}
_RESPONSE_LIMIT = 1024 * 1024


class MiniMaxASR:
    def __init__(self, client: httpx.AsyncClient, *, region: str, timeout: float, max_audio_seconds: float,
                 model: str = MINIMAX_ASR_MODEL) -> None:
        if region not in _ENDPOINTS:
            raise ValueError("invalid MiniMax region")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("ASR model must be a non-empty string")
        require_seconds(timeout, "ASR timeout")
        require_asr_audio_seconds(max_audio_seconds)
        self._client = client
        self._model = model
        self._endpoint = _ENDPOINTS[region]
        self._timeout = timeout
        self._max_audio_seconds = max_audio_seconds

    async def transcribe(self, file: AudioFile) -> str:
        if file.duration_seconds is not None and file.duration_seconds > self._max_audio_seconds:
            raise AsrError("invalid_audio_duration")
        try:
            async with asyncio.timeout(self._timeout):
                async with self._client.stream(
                    "POST", self._endpoint,
                    data={"model": self._model, "response_format": "json", "stream": "false"},
                    files={"file": (f"audio.{file.extension}", file.data, file.content_type)},
                    headers={"Accept": "application/json"}, follow_redirects=False,
                ) as response:
                    if not response.is_success:
                        raise AsrError(f"asr_http_{response.status_code}")
                    body = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=32 * 1024):
                        if len(body) + len(chunk) > _RESPONSE_LIMIT:
                            raise AsrError("asr_response_limit")
                        body.extend(chunk)
                    result = json.loads(body)
        except (TimeoutError, httpx.TimeoutException):
            raise AsrError("asr_timeout") from None
        except httpx.RequestError:
            raise AsrError("asr_transport") from None
        except (ValueError, RecursionError):
            raise AsrError("asr_invalid_response") from None
        text = result.get("text") if isinstance(result, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise AsrError("asr_empty_transcript")
        return text.strip()
