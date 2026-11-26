import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from agent_companion.__main__ import _serve
from agent_companion.channels.feishu.messages import BatchMediaBudget, IncomingMessage, MessageNormalizer
from agent_companion.config import ConfigError, MiniMaxConnection, load_settings
from agent_companion.speech.audio import AudioFormat
from agent_companion.speech.bailian_asr import BailianASR
from agent_companion.speech.bailian_ws import ASR_MODEL, BailianConnection
from agent_companion.speech.minimax import MiniMaxASR
from agent_companion.speech.vad import INPUT_RATE

from feishu_test_support import configured_roles, private_discovery
from test_bailian_speech import _Socket, _opus
from test_voice import _Device, _Vad, _until
from test_voice_segments import _Frames


_EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.toml"
_VOICE = '''
[feishu]
mode = "voice"
[voice]
input_device = "input_fixture"
output_device = "output_fixture"
vad_model = "fixture.onnx"
speech_frames = 1
silence_seconds = 0.064
segment_pause_seconds = 0.032
'''


def _example(asr=False, tts=False, *, voice=True):
    config = _EXAMPLE.read_text(encoding="utf-8")
    for ability, bailian in (("asr", asr), ("tts", tts)):
        if bailian:
            config = config.replace(f'{ability}_model = "minimax_{ability}"', f'{ability}_model = "bailian_{ability}"')
    return config + (_VOICE if voice else '\n[feishu]\nmode = "text"\n')


@asynccontextmanager
async def _settings(config):
    with TemporaryDirectory(dir=_EXAMPLE.parent) as folder:
        path = Path(folder) / "fixture.toml"
        path.write_text(config, encoding="utf-8")
        yield load_settings(path, for_run=True)


class BailianConfigIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_example_keeps_minimax_defaults_and_can_select_each_bailian_capability(self):
        for asr, tts in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(bailian_asr=asr, bailian_tts=tts):
                async with _settings(_example(asr, tts)) as settings:
                    self.assertIsInstance(settings.asr.connection, BailianConnection if asr else MiniMaxConnection)
                    self.assertIsInstance(settings.tts.connection, BailianConnection if tts else MiniMaxConnection)
                    if asr:
                        self.assertEqual(settings.asr.model, ASR_MODEL)
                        self.assertEqual(settings.asr.connection.region, "cn-beijing")
                    if tts:
                        self.assertEqual(settings.tts.model, "qwen-audio-3.1-tts-flash")
                        self.assertEqual(settings.tts.voice_id, "longanhuan_v3.1")
                    self.assertEqual(settings.voice.output_sample_rate, 44_100)
                    self.assertEqual(settings.asr_timeout, 30)
                    self.assertEqual(settings.tts_timeout, 25)

    async def test_rejects_unsupported_selected_bailian_settings(self):
        valid = _example(True, True)
        for old, new, expected in (
            ('workspace_id = "your-workspace-id"', '', "workspace_id"),
            ('workspace_id = "your-workspace-id"', 'workspace_id = "wrong.example"', "workspace_id"),
            ('region = "cn-beijing"', 'region = "cn"', "region"),
            (f'model = "{ASR_MODEL}"', 'model = "file-only"', "asr.model"),
            ('"qwen-audio-3.1-tts-flash"', '"qwen3-tts-flash-realtime"', "tts.model"),
            ('"longanhuan_v3.1"', '""', "tts.voice_id"),
            ('"longanhuan_v3.1"', '"longanyang"', "不匹配"),
            ('\n[voice]\n', '\n[voice]\noutput_sample_rate = 32000\n', "output_sample_rate"),
        ):
            with self.subTest(expected=expected, change=old):
                with self.assertRaisesRegex(ConfigError, expected):
                    async with _settings(valid.replace(old, new)):
                        self.fail("invalid config accepted")

    async def test_cosyvoice_pair_and_singapore_region(self):
        config = _example(True, True).replace('region = "cn-beijing"', 'region = "ap-southeast-1"')
        config = config.replace('"qwen-audio-3.1-tts-flash"', '"cosyvoice-v3-flash"').replace(
            '"longanhuan_v3.1"', '"longanyang"')
        async with _settings(config) as settings:
            self.assertEqual(settings.tts.model, "cosyvoice-v3-flash")
            self.assertEqual(settings.tts.voice_id, "longanyang")
            self.assertIs(settings.asr.connection, settings.tts.connection)
            self.assertEqual(settings.tts.connection.region, "ap-southeast-1")


class BailianStartupIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_mode_only_uses_bailian_file_even_if_tts_is_selected(self):
        config = _example(True, True, voice=False).replace('"qwen-audio-3.1-tts-flash"', '"unsupported"')

        async def run_chats(_discovery, _db, file, *, incoming, **_):
            self.assertIsNone(incoming)
            self.assertIsInstance(file, BailianASR)

        async with _settings(config) as settings:
            self.assertIsNone(settings.tts)
            with (patch("agent_companion.app.run_private_chats", side_effect=run_chats) as run,
                  patch("agent_companion.speech.bailian_tts.BailianTTS", side_effect=AssertionError("no TTS")),
                  patch("agent_companion.speech.bailian_ws.BailianConnection._open",
                        side_effect=AssertionError("startup must not dial"))):
                await _serve(settings, discovery=self._discovery(), roles=configured_roles())
            run.assert_awaited_once()

    async def test_actual_startup_routes_audio_file_pcm_stream_and_tts_with_isolated_auth(self):
        for asr, tts in ((True, True), (True, False), (False, True)):
            with self.subTest(bailian_asr=asr, bailian_tts=tts):
                await self._exercise_mixed_startup(asr, tts)

    @staticmethod
    def _discovery():
        return private_discovery("oc_fixture", "ou_peer_fixture", "ou_self_fixture")

    async def _exercise_mixed_startup(self, use_bailian_asr, use_bailian_tts):
        http_clients, requests, sockets = [], [], []
        httpx_client = httpx.AsyncClient
        opus = _opus()
        pcm = b"\x01\x00"

        async def respond(request):
            requests.append(request)
            if request.url.path == "/v1/speech_to_text":
                return httpx.Response(200, json={"text": "合成转写"})
            if request.url.path.endswith("/t2a_v2"):
                return httpx.Response(200, stream=httpx.ByteStream(
                    b'data: {"data":{"status":1,"audio":"0100"}}\n\ndata: {"data":{"status":2}}\n\n'))
            self.fail("unexpected offline HTTP request")

        def client(**options):
            self.assertFalse(options["follow_redirects"])
            self.assertFalse(options["trust_env"])
            result = httpx_client(transport=httpx.MockTransport(respond), **options)
            http_clients.append(result)
            return result

        async def ws_reply(socket, raw):
            if not isinstance(raw, str) or json.loads(raw)["header"]["action"] != "finish-task":
                return
            params = socket.messages[0]["payload"]
            if params["task"] == "asr":
                socket.sentence(1, "合成转写")
            else:
                socket.incoming.put_nowait(pcm)
            socket.event("task-finished")

        def open_ws(connection, _timeout):
            self.assertIsInstance(connection, BailianConnection)
            self.assertTrue(connection is settings.asr.connection or connection is settings.tts.connection)
            socket = _Socket(on_send=ws_reply)
            sockets.append(socket)
            return socket

        async def run_chats(_discovery, _db, file, *, incoming, **_):
            self.assertIsInstance(file, BailianASR if use_bailian_asr else MiniMaxASR)
            cli = SimpleNamespace(download=AsyncMock(return_value=opus))
            message = IncomingMessage("om_fixture", 1, "audio", '{"file_key":"file_fixture"}')
            actual = await MessageNormalizer(cli, file).normalize(message, BatchMediaBudget(1024))
            self.assertEqual(actual.content, "合成转写")
            voice, device = await incoming._voice_factory()
            events = []
            self.assertEqual(voice._segment_frames is None, use_bailian_asr)
            async with voice.connected(device, lambda *event: events.append(event), capture_enabled=False):
                frames = _Frames(voice)
                hot = frames.push(True)
                if use_bailian_asr:
                    await _until(lambda: len(sockets) == 2 and sockets[-1].audio.is_set())
                    self.assertEqual([x for x in sockets[-1].messages if isinstance(x, bytes)], [hot])
                    self.assertNotIn("finish-task", sockets[-1].actions())
                frames.push(False, voice._end_frames)
                await _until(lambda: any(text for text, _ in events))
            self.assertEqual(events, [(None, True), ("合成转写", False)])
            audio = [chunk async for chunk in voice._output._tts.synthesize("合成段", sample_rate=44_100)]
            self.assertEqual(audio, [pcm])

        device = _Device()
        device.output_format = AudioFormat(44_100)
        self.assertEqual(device.input_format, AudioFormat(INPUT_RATE))
        async with _settings(_example(use_bailian_asr, use_bailian_tts)) as settings:
            async with asyncio.timeout(3):
                with (patch("agent_companion.__main__.httpx.AsyncClient", side_effect=client),
                      patch.object(BailianConnection, "_open", autospec=True, side_effect=open_ws),
                      patch("agent_companion.channels.feishu.macos_calls.MacCallDesktop"),
                      patch("agent_companion.channels.audio.macos.MacAudioDevice", return_value=device),
                      patch("agent_companion.speech.vad.SileroVAD", return_value=_Vad()),
                      patch("agent_companion.app.run_private_chats", side_effect=run_chats) as run):
                    await _serve(settings, discovery=self._discovery(), roles=configured_roles())
                run.assert_awaited_once()
            self.assertTrue(all(client.is_closed for client in http_clients))
            self.assertEqual(len(http_clients), 1 if use_bailian_asr and use_bailian_tts else 2)
            self.assertTrue(all(socket.closed for socket in sockets))
            for request in requests:
                # 混搭只将 MiniMax 对应占位凭据发送给 MiniMax，不给百炼使用 HTTP 客户端。
                selected = settings.asr if request.url.path == "/v1/speech_to_text" else settings.tts
                self.assertIsInstance(selected.connection, MiniMaxConnection)
                self.assertTrue(request.headers["authorization"] == "Bearer " + selected.connection.api_key)
            payloads = [socket.messages[0]["payload"] for socket in sockets]
            expected = (["opus", "pcm"] if use_bailian_asr else []) + (["pcm"] if use_bailian_tts else [])
            self.assertEqual([payload["parameters"]["format"] for payload in payloads], expected)
            if use_bailian_asr:
                self.assertEqual(b"".join(x for x in sockets[0].messages if isinstance(x, bytes)), opus)
            if use_bailian_tts:
                self.assertEqual(payloads[-1]["parameters"]["sample_rate"], 44_100)
                self.assertEqual(payloads[-1]["parameters"]["voice"], "longanhuan_v3.1")
