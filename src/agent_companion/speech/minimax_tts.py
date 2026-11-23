"""MiniMax HTTP /v1/t2a_v2：每段提交一次，逐个消费 SSE 音频块。"""

import asyncio
from collections.abc import AsyncGenerator
import json
from typing import NoReturn

import httpx

from .tts import TtsError

_ENDPOINTS = {
    "cn": "https://api.minimax.cn/v1/t2a_v2",
    "global": "https://api.minimax.io/v1/t2a_v2",
}
_MAX_LINE_BYTES = 1024 * 1024
_MAX_JSON_BYTES = 64 * 1024


async def _raise_json_failure(response: httpx.Response, timeout: float) -> NoReturn:
    # MiniMax 在流式请求的 HTTP 200 中也可能直接返回业务错误 JSON。
    body = bytearray()
    async with asyncio.timeout(timeout):
        async for chunk in response.aiter_raw():
            if len(body) + len(chunk) > _MAX_JSON_BYTES:
                raise TtsError("tts_json_response_limit")
            body.extend(chunk)
    try:
        value = json.loads(body)
    except (ValueError, RecursionError):
        raise TtsError("tts_invalid_json_response") from None
    if not isinstance(value, dict):
        raise TtsError("tts_invalid_json_response")
    meta = value.get("base_resp")
    if isinstance(meta, dict):
        code = meta.get("status_code")
        if type(code) is int and code != 0:
            raise TtsError("tts_business_error")
    raise TtsError("tts_unexpected_json_response")


async def _lines(response: httpx.Response, timeout: float) -> AsyncGenerator[bytes, None]:
    # 不要求 httpx 攒满固定大小的块：收到第一条 SSE 数据就可开始播放。
    chunks = response.aiter_raw()
    line = bytearray()
    while True:
        async with asyncio.timeout(timeout):
            chunk = await anext(chunks, None)
        if chunk is None:
            break
        start = 0
        while start < len(chunk):
            end = chunk.find(b"\n", start)
            stop = len(chunk) if end == -1 else end
            if len(line) + stop - start > _MAX_LINE_BYTES:
                raise TtsError("tts_event_limit")
            line.extend(chunk[start:stop])
            if end == -1:
                break
            yield bytes(line).rstrip(b"\r")
            line.clear()
            start = end + 1
    if line:
        yield bytes(line).rstrip(b"\r")


class MiniMaxTTS:
    def __init__(self, client: httpx.AsyncClient, *, region: str, model: str,
                 voice_id: str, timeout: float) -> None:
        self._client = client
        self._endpoint = _ENDPOINTS[region]
        self._model = model
        self._voice_id = voice_id
        self._timeout = timeout

    async def synthesize(self, text: str, *, sample_rate: int) -> AsyncGenerator[bytes, None]:
        payload = {
            "model": self._model, "text": text, "stream": True,
            "stream_options": {"exclude_aggregated_audio": True}, "output_format": "hex",
            "voice_setting": {"voice_id": self._voice_id},
            "audio_setting": {"sample_rate": sample_rate, "format": "pcm", "channel": 1},
        }
        try:
            async with self._client.stream(
                "POST", self._endpoint, json=payload,
                headers={"Accept": "text/event-stream", "Accept-Encoding": "identity"},
                follow_redirects=False, timeout=self._timeout,
            ) as response:
                if not response.is_success:
                    raise TtsError(f"tts_http_{response.status_code}")
                # 在 JSON 与 SSE 分流前拒绝压缩，避免解压先于正文限额。
                encoding = response.headers.get("content-encoding", "").strip().lower()
                if encoding not in ("", "identity"):
                    raise TtsError("tts_unsupported_content_encoding")
                media_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
                if media_type == "application/json":
                    await _raise_json_failure(response, self._timeout)
                received = False
                tail = b""
                async for line in _lines(response, self._timeout):
                    if not line.startswith(b"data:"):
                        continue
                    value = json.loads(line[5:])
                    if not isinstance(value, dict):
                        raise TtsError("tts_invalid_event")
                    meta = value.get("base_resp", {})
                    if not isinstance(meta, dict) or meta.get("status_code", 0) != 0:
                        raise TtsError("tts_business_error")
                    data = value.get("data")
                    if not isinstance(data, dict):
                        continue
                    status = data.get("status")
                    if type(status) is not int or status not in (1, 2):
                        raise TtsError("tts_invalid_status")
                    done = status == 2
                    audio = data.get("audio")
                    # status=2 中的 audio 是可选的汇总。即使供应商未忽略它，
                    # 也不把已发过的流式音频重新播放；只兼容单一终态块。
                    if audio and (not done or not received):
                        if not isinstance(audio, str):
                            raise TtsError("tts_invalid_audio")
                        decoded = bytes.fromhex(audio)
                        if not decoded:
                            raise TtsError("tts_invalid_pcm")
                        received = True
                        pcm = tail + decoded
                        boundary = len(pcm) - len(pcm) % 2
                        tail = pcm[boundary:]
                        if boundary:
                            yield pcm[:boundary]
                    if done:
                        if tail:
                            raise TtsError("tts_invalid_pcm")
                        if not received:
                            raise TtsError("tts_empty_audio")
                        return
                if tail:
                    raise TtsError("tts_invalid_pcm")
                raise TtsError("tts_incomplete_stream")
        except (httpx.TimeoutException, TimeoutError):
            raise TtsError("tts_timeout") from None
        except httpx.RequestError:
            raise TtsError("tts_transport") from None
        except (ValueError, RecursionError):
            raise TtsError("tts_invalid_event") from None
