import asyncio
import io
import unittest
from unittest.mock import patch
import wave

import httpx

from agent_companion.speech.asr import AsrError, AudioFile
from agent_companion.speech.minimax import MiniMaxASR


_OGG_OPUS = b"OggS\x00" + bytes(21) + b"\x01\x13OpusHead" + bytes(11)


def _wave(rate=16000, frames=160):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        w.writeframes(b"\0\0" * frames)
    return buf.getvalue()


class AudioFileTests(unittest.TestCase):
    def test_formats_and_limits(self):
        data = _wave()
        wav = AudioFile(data)
        self.assertEqual((wav.extension, wav.content_type, wav.duration_seconds), ("wav", "audio/wav", 0.01))
        ogg = AudioFile(_OGG_OPUS)
        self.assertEqual((ogg.extension, ogg.duration_seconds), ("ogg", None))
        corrupt_chunk = b"RIFF\x0c\x00\x00\x00WAVEJUNK\xff\xff\xff\xff"
        for invalid in (b"", b"\x00\x00" * 64, b'{"error":"not audio"}', data[:-1], corrupt_chunk):
            with self.subTest(invalid_size=len(invalid)), self.assertRaises(AsrError):
                AudioFile(invalid)
        with patch("agent_companion.speech.asr.MAX_AUDIO_BYTES", len(data) - 1):
            with self.assertRaisesRegex(AsrError, "invalid_audio_size"):
                AudioFile(data)
        for invalid in (_wave(frames=0), data[:24] + bytes(4) + data[28:]):
            with self.assertRaisesRegex(AsrError, "invalid_audio_duration"):
                AudioFile(invalid)
        self.assertEqual(AudioFile(_wave(rate=8000, frames=501 * 8000)).duration_seconds, 501)


class MiniMaxASRTests(unittest.IsolatedAsyncioTestCase):
    async def test_official_region_and_complete_multipart(self):
        for region, host in (("cn", "api.minimax.cn"), ("global", "api.minimax.io")):
            calls = []
            data = _wave()

            async def respond(request):
                calls.append(request)
                body = await request.aread()
                self.assertEqual((request.method, request.url.host, request.url.path), ("POST", host, "/v1/speech_to_text"))
                self.assertTrue(request.headers["Content-Type"].startswith("multipart/form-data; boundary="))
                self.assertEqual(request.headers["Accept"], "application/json")
                for value in (b'name="model"\r\n\r\nasr-1.0', b'name="response_format"\r\n\r\njson',
                              b'name="stream"\r\n\r\nfalse', b'name="file"; filename="audio.wav"',
                              b'Content-Type: audio/wav', data):
                    self.assertIn(value, body)
                return httpx.Response(200, json={"text": " 模拟识别结果 ", "duration": 0.01})

            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                asr = MiniMaxASR(client, region=region, timeout=1, max_audio_seconds=1)
                actual = await asr.transcribe(AudioFile(data))
                self.assertEqual((actual, len(calls)), ("模拟识别结果", 1))

    async def test_configured_wav_limit_is_inclusive_and_other_formats_reach_server(self):
        calls = []

        async def respond(request):
            calls.append(await request.aread())
            return httpx.Response(200, json={"text": "模拟识别结果"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            asr = MiniMaxASR(client, region="cn", timeout=1, max_audio_seconds=0.01)
            with self.assertRaisesRegex(AsrError, "invalid_audio_duration"):
                await asr.transcribe(AudioFile(_wave(frames=161)))
            self.assertEqual(calls, [])
            wav = _wave(frames=160)
            await asr.transcribe(AudioFile(wav))
            await asr.transcribe(AudioFile(_OGG_OPUS))
            self.assertEqual(len(calls), 2)
            self.assertIn(wav, calls[0])
            self.assertIn(_OGG_OPUS, calls[1])

    async def test_timeout_and_http_error_never_retry(self):
        for delayed in (False, True):
            calls = 0

            async def respond(request):
                nonlocal calls
                calls += 1
                if delayed:
                    await asyncio.sleep(20)
                return httpx.Response(503, text="synthetic private error body")

            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                with self.assertRaisesRegex(AsrError, "asr_timeout" if delayed else "asr_http_503"):
                    asr = MiniMaxASR(client, region="cn", timeout=0.05, max_audio_seconds=1)
                    await asr.transcribe(AudioFile(_wave()))
                self.assertEqual(calls, 1)
