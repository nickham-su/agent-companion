import asyncio
import gzip
import json
import unittest

import httpx

from agent_companion.speech.minimax_tts import MiniMaxTTS
from agent_companion.speech.tts import TtsError


class _Stream(httpx.AsyncByteStream):
    def __init__(self, chunks, *, release=None):
        self.chunks, self.release = chunks, release
        self.closed = False
        self.read_attempts = 0

    async def __aiter__(self):
        self.read_attempts += 1
        for index, chunk in enumerate(self.chunks):
            if index == 2 and self.release is not None:
                await self.release.wait()
            yield chunk

    async def aclose(self):
        self.closed = True


def _event(status, audio):
    return f'data: {json.dumps({"data": {"status": status, "audio": audio}})}\r\n\r\n'.encode()


class MiniMaxTtsTest(unittest.IsolatedAsyncioTestCase):
    def _tts(self, client, timeout=1):
        return MiniMaxTTS(client, region="cn", model="speech-2.8-turbo", voice_id="synthetic_voice", timeout=timeout)

    async def test_stream_early_first_pcm_and_skip_final_aggregate(self):
        requests = []
        first = _event(1, "0100")
        release = asyncio.Event()
        stream = _Stream([first[:6], first[6:], _event(1, "0200") + _event(2, "01000200")], release=release)

        async def handle(request):
            requests.append(request)
            return httpx.Response(200, stream=stream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            chunks = self._tts(client).synthesize("合成文字。", sample_rate=44100)
            self.assertEqual(await asyncio.wait_for(anext(chunks), 0.5), b"\1\0")
            self.assertFalse(release.is_set())
            release.set()
            self.assertEqual([chunk async for chunk in chunks], [b"\2\0"])
        self.assertTrue(stream.closed)
        self.assertEqual(len(requests), 1)
        self.assertEqual(str(requests[0].url), "https://api.minimax.cn/v1/t2a_v2")
        request = json.loads(requests[0].content)
        self.assertEqual(request["audio_setting"], {"format": "pcm", "channel": 1, "sample_rate": 44100})
        self.assertEqual(request["stream_options"], {"exclude_aggregated_audio": True})
        self.assertTrue(request["stream"])
        self.assertEqual(request["text"], "合成文字。")

    async def test_pcm_sample_split_across_provider_packets(self):
        body = _event(1, "010002") + _event(1, "00") + _event(2, "01000200")
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=_Stream([body]))
        )) as client:
            result = [chunk async for chunk in self._tts(client).synthesize("test", sample_rate=16000)]
        self.assertEqual(result, [b"\1\0", b"\2\0"])

    async def test_business_failure_invalid_pcm_and_truncated_stream(self):
        for body, reason in ((b'data: {"base_resp":{"status_code":1000}}\n\n', "tts_business_error"),
                             (_event(1, "01"), "tts_invalid_pcm"),
                             (_event(1, "0100"), "tts_incomplete_stream")):
            with self.subTest(reason=reason):
                async with httpx.AsyncClient(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, stream=_Stream([body]))
                )) as client:
                    with self.assertRaisesRegex(TtsError, reason):
                        _ = [chunk async for chunk in self._tts(client).synthesize("test", sample_rate=16000)]

    async def test_http_200_json_business_failure_is_classified_without_raw_response(self):
        body = b'{"base_resp":{"status_code":1004,"status_msg":"synthetic-private-diagnostic"}}'
        stream = _Stream([body[:13], body[13:]])
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, stream=stream, headers={"Content-Type": "Application/JSON; charset=utf-8",
                                         "Content-Encoding": "Identity"},
        ))) as client:
            with self.assertRaises(TtsError) as error:
                _ = [chunk async for chunk in self._tts(client).synthesize("test", sample_rate=16000)]
        self.assertEqual(str(error.exception), "tts_business_error")
        self.assertTrue(stream.closed)

    async def test_compressed_responses_are_rejected_before_reading(self):
        for media_type, body in (("application/json", b'{"base_resp":{"status_code":1004}}'),
                                 ("text/event-stream", _event(1, "0100") + _event(2, ""))):
            with self.subTest(media_type=media_type):
                stream = _Stream([gzip.compress(body)])

                async def handle(request):
                    self.assertEqual(request.headers["Accept-Encoding"], "identity")
                    return httpx.Response(200, stream=stream,
                                          headers={"Content-Type": media_type, "Content-Encoding": "gzip"})

                async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                    with self.assertRaisesRegex(TtsError, "^tts_unsupported_content_encoding$"):
                        _ = [chunk async for chunk in self._tts(client).synthesize("test", sample_rate=16000)]
                self.assertEqual(stream.read_attempts, 0)
                self.assertTrue(stream.closed)

    async def test_json_response_is_bounded_and_not_confused_with_sse(self):
        for body, reason in ((b"x" * (64 * 1024 + 1), "tts_json_response_limit"),
                             (b"{broken payload", "tts_invalid_json_response"),
                             (b'{"base_resp":{"status_code":0}}', "tts_unexpected_json_response")):
            with self.subTest(reason=reason):
                stream = _Stream([body[:32_000], body[32_000:]])
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
                    200, stream=stream, headers={"Content-Type": "application/json"},
                ))) as client:
                    with self.assertRaisesRegex(TtsError, f"^{reason}$"):
                        _ = [chunk async for chunk in self._tts(client).synthesize("test", sample_rate=16000)]
                self.assertTrue(stream.closed)

    async def test_json_response_total_timeout(self):
        stream = _Stream([b"{", b"\n", b"}"], release=asyncio.Event())
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, stream=stream, headers={"Content-Type": "application/json"},
        ))) as client:
            with self.assertRaisesRegex(TtsError, "^tts_timeout$"):
                _ = [chunk async for chunk in self._tts(client, 0.02).synthesize("test", sample_rate=16000)]
        self.assertTrue(stream.closed)

    async def test_stream_read_timeout_closes_request(self):
        stream = _Stream([_event(1, "0100"), b"\n", _event(2, "")], release=asyncio.Event())
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=stream)
        )) as client:
            with self.assertRaisesRegex(TtsError, "tts_timeout"):
                _ = [chunk async for chunk in self._tts(client, 0.02).synthesize("test", sample_rate=16000)]
        self.assertTrue(stream.closed)
