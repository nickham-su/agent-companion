"""百炼 TTS：单段文字对应一个任务，边收 PCM 边交给播放层。"""

import asyncio
from collections.abc import AsyncGenerator

from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

from ..config import require_seconds
from .bailian_ws import MAX_EVENTS, TTS_MODELS, BailianConnection, InvalidEvent, SpeechTask, TaskFailed
from .tts import TtsError


_SAMPLE_RATES = frozenset({8_000, 16_000, 22_050, 24_000, 44_100, 48_000})
_MAX_TEXT_CHARS = 20_000


class BailianTTS:
    def __init__(self, connection: BailianConnection, *, model: str, voice_id: str, timeout: float) -> None:
        if not isinstance(model, str) or model not in TTS_MODELS:
            raise ValueError("unsupported Bailian TTS model")
        if not (isinstance(voice_id, str) and 0 < len(voice_id) <= 128
                and voice_id.strip() == voice_id and voice_id.isprintable()):
            raise ValueError("invalid Bailian TTS voice_id")
        require_seconds(timeout, "TTS timeout")
        self._connection = connection
        self._model = model
        self._voice_id = voice_id
        self._timeout = timeout

    async def synthesize(self, text: str, *, sample_rate: int) -> AsyncGenerator[bytes, None]:
        if type(sample_rate) is not int or sample_rate not in _SAMPLE_RATES:
            raise TtsError("tts_invalid_audio")
        if not text.strip():
            raise TtsError("tts_empty_audio")
        if len(text) > _MAX_TEXT_CHARS:
            raise TtsError("voice_reply_text_limit")
        try:
            async with self._connection._open(self._timeout) as socket:
                task = SpeechTask(socket)
                await task.start("tts", "SpeechSynthesizer", self._model, {
                    "text_type": "PlainText", "voice": self._voice_id, "format": "pcm", "sample_rate": sample_rate,
                }, self._timeout)
                async with asyncio.timeout(self._timeout):
                    await task.send("continue-task", {"input": {"text": text}})
                    await task.send("finish-task", {"input": {}})
                tail = b""
                received_pcm = False
                loop = asyncio.get_running_loop()
                deadline = loop.time() + self._timeout
                for _ in range(MAX_EVENTS):
                    async with asyncio.timeout_at(deadline):
                        raw = await socket.recv()
                    if loop.time() >= deadline:
                        raise TimeoutError
                    if isinstance(raw, bytes):
                        if not raw:
                            continue
                        # WS 消息边界未必落在采样边界。保留至多 1 字节跨块补齐。
                        data = tail + raw
                        end = len(data) & ~1
                        tail = data[end:]
                        if end:
                            received_pcm = True
                            yield data[:end]
                            # 播放层背压不消耗网络无进展时限。
                            deadline = loop.time() + self._timeout
                        continue
                    event, _ = task.parse(raw)
                    if event == "task-finished":
                        if tail:
                            raise TtsError("tts_invalid_pcm")
                        if not received_pcm:
                            raise TtsError("tts_empty_audio")
                        return
                    if event != "result-generated":
                        raise InvalidEvent
                raise TtsError("tts_event_limit")
        except TimeoutError:
            raise TtsError("tts_timeout") from None
        except InvalidStatus as exc:
            raise TtsError(f"tts_http_{exc.response.status_code}") from None
        except ConnectionClosed:
            raise TtsError("tts_incomplete_stream") from None
        except (OSError, WebSocketException):
            raise TtsError("tts_transport") from None
        except InvalidEvent:
            raise TtsError("tts_invalid_event") from None
        except TaskFailed:
            raise TtsError("tts_business_error") from None
