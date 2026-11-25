"""Run both real app compositions against fake CLI/model/audio, sharing one disk database."""

import asyncio
from contextlib import asynccontextmanager, closing, suppress
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

from agent_companion.app import run_feishu
from agent_companion.channels.feishu.cli import CommandResult, FeishuDiscovery
from agent_companion.channels.feishu.macos_calls import RingingWindow, WindowIdentity
from agent_companion.config import (ASRSettings, FeishuSettings, MiniMaxConnection, RuntimeSettings, Settings,
                                    VoiceSettings)
from agent_companion.protocol import AGENT_REPLY_PREFIX, DeliveryStatus, TextDelta, ToolCall, ToolReady
from agent_companion.runtime.subtasks import DELEGATE_TOOL_NAME, demo_roles
from agent_companion.runtime.timers import CANCEL_TIMER_TOOL_NAME, SET_TIMER_TOOL_NAME
from agent_companion.speech.audio import AudioFormat
from agent_companion.speech.vad import INPUT_FRAMES, INPUT_RATE
from agent_companion.speech.voice import VoiceChannel
from agent_companion.storage.sqlite import History


_CHAT, _PEER, _OWN = "oc_mvp", "ou_synthetic", "ou_companion"
_USER = "请委派一项合成测试。"
_PROMPT = "子会话单独计算一加一。"
_RESULT = "合成结果：一加一等于二。"
_ACCEPTED = "已受理，稍后回投。"
_RETURNED = "已收到子会话结果。"
_FAILED = "模拟明确拒绝。"
_UNKNOWN = "模拟无法确认。"
_OLD_INPUT = "播报长句测试。"
_TEXT_IN_CALL = "在语音模式发文字。"
_TEXT_REPLY = "仍然回复文字。"
_OLD_VOICE_PREFIX = "已完整交给设备。"
_OLD_VOICE_TAIL = "甲乙丙丁戊己庚辛。"
_OLD_VOICE = _OLD_VOICE_PREFIX + _OLD_VOICE_TAIL
_ASR_TEXT = "合成语音转写。"
_NEW_VOICE = "新一轮语音回复。"
_RESTARTED = "已恢复主会话。"
_HOT = b"\x01\x00" * INPUT_FRAMES
_PCM = b"\x02\x00"


def _success(data):
    return CommandResult(0, json.dumps({"ok": True, "identity": "user", "data": data}).encode(), b"")


class _CLITransport:
    """Fake CLI responses beneath the real FeishuCLI serialization and reply parser."""

    def __init__(self):
        self.now = 1_700_000_000
        self.first_tick = asyncio.Event()
        self.inbox = []
        self.sends = []
        self.posts = []
        self.meeting_lookups = []

    def clock(self):
        self.first_tick.set()
        return self.now

    def push(self, text):
        self.inbox.append({
            "message_id": f"om_input_{len(self.inbox)}", "chat_id": _CHAT, "msg_type": "text",
            "create_time": str(self.now * 1000 + 1), "deleted": False,
            "sender": {"id": _PEER, "id_type": "open_id", "sender_type": "user"},
            "body": {"content": json.dumps({"text": text})},
        })
        self.now += 1

    async def __call__(self, argv, data, timeout):
        assert argv[0] == "mock-lark-cli"
        if argv[1:4] == ("api", "GET", "/open-apis/authen/v1/user_info"):
            return _success({"open_id": _OWN})
        if argv[1:3] == ("vc", "+meeting-list-active"):
            self.meeting_lookups.append("list")
            return _success({"meetings": [{"meeting_id": "1"}]})
        if argv[1:4] == ("vc", "meeting", "get"):
            self.meeting_lookups.append("detail")
            return _success({"meeting": {"id": "1", "status": 1, "host_user": {"id": _PEER},
                                         "participants": [{"id": _PEER}]}})
        assert argv[-4:] == ("--as", "user", "--format", "json")
        params = json.loads(argv[argv.index("--params") + 1])
        if argv[1:4] == ("api", "POST", "/open-apis/im/v1/messages/search"):
            query = json.loads(data)
            assert query["query"] == "" and query["filter"]["chat_type"] == "p2p"
            bounds = query["filter"]["time_range"]
            start, end = (1000 * datetime.fromisoformat(bounds[key]).timestamp()
                          for key in ("start_time", "end_time"))
            return _success({"items": [{"meta_data": {
                "message_id": item["message_id"], "chat_id": item["chat_id"], "type": item["msg_type"],
                "from_id": item["sender"]["id"], "is_p2p_chat": True,
                "create_time": datetime.fromtimestamp(int(item["create_time"]) / 1000, timezone.utc).isoformat(),
            }} for item in self.inbox if start <= int(item["create_time"]) < end], "has_more": False})
        if argv[1:4] == ("api", "GET", "/open-apis/im/v1/messages/mget"):
            return _success({"items": [item for item in self.inbox if item["message_id"] in params["message_ids"]]})
        assert argv[1:4] == ("api", "POST", "/open-apis/im/v1/messages")
        assert params == {"receive_id_type": "open_id"}
        payload = json.loads(data)
        assert payload["receive_id"] == _PEER
        content = json.loads(payload["content"])
        if payload["msg_type"] == "post":
            self.posts.append(content)
            return _success({"message_id": f"om_post_{len(self.posts)}", "chat_id": _CHAT})
        assert payload["msg_type"] == "text"
        text = content["text"]
        self.sends.append(text)
        if text == _FAILED:
            return CommandResult(1, b"", b'{"ok":false,"error":{"type":"authorization"}}')
        if text == _UNKNOWN:
            return CommandResult(1, b"", b'{"ok":false,"error":{"type":"api"}}')
        return _success({"message_id": f"om_reply_{len(self.sends)}", "chat_id": _CHAT})


class _ScriptedModel:
    def __init__(self):
        self.child_started = asyncio.Event()
        self.finish_child = asyncio.Event()
        self.restart_request = None

    async def stream(self, messages, tools, *, thinking):
        last = messages[-1]
        if not tools:
            assert last.role == "user" and last.content == _PROMPT
            self.child_started.set()
            await self.finish_child.wait()
            yield TextDelta(_RESULT)
        elif last.role == "tool":
            yield TextDelta(_ACCEPTED)
        elif last.content == _USER:
            assert {tool.name for tool in tools} == {DELEGATE_TOOL_NAME, SET_TIMER_TOOL_NAME, CANCEL_TIMER_TOOL_NAME}
            yield ToolReady(ToolCall("synthetic-call", DELEGATE_TOOL_NAME, json.dumps({"agent": "general", "prompt": _PROMPT})))
        elif last.content.startswith(AGENT_REPLY_PREFIX):
            yield TextDelta(_RETURNED)
        elif last.content == "明确失败测试":
            yield TextDelta(_FAILED)
        elif last.content == "未知投递测试":
            yield TextDelta(_UNKNOWN)
        elif last.content == _OLD_INPUT:
            yield TextDelta(_OLD_VOICE)
        elif last.content == _TEXT_IN_CALL:
            yield TextDelta(_TEXT_REPLY)
        elif last.content == _ASR_TEXT:
            yield TextDelta(_NEW_VOICE)
        elif last.content == "重启后的新输入":
            self.restart_request = tuple(messages)
            yield TextDelta(_RESTARTED)
        else:
            raise AssertionError(f"unexpected synthetic input: {last.content!r}")


class _Desktop:
    popup = RingingWindow(WindowIdentity(7, frozenset({10})), (100, 20, 368, 86))
    main = WindowIdentity(7, frozenset({11}))

    def __init__(self):
        self.ringing_now = False
        self.connected = False
        self.clicks = 0

    def ringing(self):
        return (self.popup,) if self.ringing_now else ()

    def main_windows(self):
        return (self.main,) if self.connected else ()

    def main_window_diagnostics(self, pid):
        return {"desktop_records": int(self.connected)}

    def window_exists(self, identity):
        return self.connected and identity == self.main

    def answer(self, ring, *, before_deadline):
        assert self.ringing_now and ring == self.popup and before_deadline()
        self.clicks += 1
        self.ringing_now = False
        self.connected = True
        return True


class _AudioDevice:
    input_format = AudioFormat(INPUT_RATE)
    output_format = AudioFormat(INPUT_RATE)

    def __init__(self):
        self.closed = False

    def start(self, capture, render):
        self.capture, self.render = capture, render

    def check_active(self):
        assert not self.closed

    def close(self):
        self.closed = True

    def frames(self, frame, count):
        for _ in range(count):
            self.capture(memoryview(frame), False)

    def pull(self, frames):
        buffer = bytearray(frames * 2)
        self.render(memoryview(buffer))
        return bytes(buffer)


class _VAD:
    def probability(self, pcm):
        return float(pcm == _HOT)

    def reset(self):
        pass


class _ASR:
    def __init__(self):
        self.files = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def transcribe(self, file):
        self.files.append(file)
        if len(self.files) == 1:
            return _OLD_INPUT
        self.started.set()
        await self.release.wait()
        return _ASR_TEXT


class _TTS:
    def __init__(self):
        self.old_buffered = asyncio.Event()
        self.old_closed = asyncio.Event()
        self.new_buffered = asyncio.Event()
        self.calls = []

    async def synthesize(self, text, *, sample_rate):
        assert sample_rate == INPUT_RATE
        self.calls.append(text)
        if text == _OLD_VOICE_PREFIX:
            yield _PCM * 4
        elif text == _OLD_VOICE_TAIL:
            try:
                yield _PCM * INPUT_RATE
                self.old_buffered.set()
                await asyncio.Future()  # Actual voice layer must stop this stream.
            finally:
                self.old_closed.set()
        else:
            assert text in (_NEW_VOICE, _TEXT_REPLY)
            yield _PCM * 4
            self.new_buffered.set()


@asynccontextmanager
async def _service(settings, history, model, transport, asr, *, desktop=None, tts=None, device=None):
    discovery = FeishuDiscovery("mock-lark-cli", timeout=settings.feishu.cli_timeout, runner=transport)
    transport.first_tick.clear()

    def make_desktop():
        assert desktop is not None, "text must not initialize the desktop"
        return desktop

    def make_voice(settings, asr, tts, *, streaming=None):
        assert device is not None, "text must not prepare audio"
        return VoiceChannel(asr, tts, _VAD(), settings.voice, asr_timeout=settings.asr_timeout,
                            max_audio_seconds=settings.asr.max_audio_seconds, streaming=streaming), device

    with (patch("agent_companion.app.time", side_effect=lambda: transport.now),
          patch("agent_companion.feishu_private.time", side_effect=transport.clock),
          patch("agent_companion.channels.feishu.channel._POLL_SECONDS", 0.01),
          patch("agent_companion.feishu_calls._POLL_SECONDS", 0.01),
          patch("agent_companion.channels.feishu.macos_calls.MacCallDesktop", side_effect=make_desktop),
          patch("agent_companion.app._make_voice", side_effect=make_voice)):
        task = asyncio.create_task(run_feishu(settings, history, model, asr, discovery=discovery, tts=tts,
                                              roles=demo_roles()))
        try:
            await transport.first_tick.wait()
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


async def _outcome(history, session, count):
    while len(outcomes := history.outcomes(session)) < count:
        await asyncio.sleep(0.002)
    last = outcomes[-1]
    message = next((row.message for row in history.read_after(session)
                    if row.message_id == last.assistant_message_id), None)
    return last.delivery_status, message.content if message is not None else None


class MVPIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_feishu_subtask_voice_interrupt_failure_and_restart_share_history(self):
        with TemporaryDirectory(dir=Path.cwd(), prefix=".mvp-test-") as directory:
            async with asyncio.timeout(10):
                await self._scenario(Path(directory) / "session.sqlite3")

    async def _scenario(self, database):
        settings = Settings(
            database=database,
            feishu=FeishuSettings(cli_timeout=1),
            runtime=RuntimeSettings(model_timeout=3),
            asr=ASRSettings(MiniMaxConnection("synthetic"), max_audio_seconds=500),
            voice=VoiceSettings(input_device="mock-in", output_device="mock-out", vad_model=Path("synthetic.onnx"),
                                output_sample_rate=INPUT_RATE, silence_seconds=0.06, delivery_timeout=2),
        )
        session = "feishu-private:" + sha256(_PEER.encode("ascii")).hexdigest()
        transport, model, asr = _CLITransport(), _ScriptedModel(), _ASR()

        with closing(History(settings.database)) as history:
            async with _service(settings, history, model, transport, asr):
                transport.push(_USER)
                await model.child_started.wait()
                self.assertEqual(await _outcome(history, session, 2), (DeliveryStatus.FULL, _ACCEPTED))
                tool_result = next(row.message for row in history.read_after(session) if row.message.role == "tool")
                accepted = json.loads(tool_result.content)
                self.assertEqual(tool_result.tool_call_id, "synthetic-call")
                child, delegation_id = accepted["agent_id"], accepted["delegation_id"]
                self.assertEqual(set(accepted), {"status", "agent_id", "delegation_id"})
                delegation = history.delegation(delegation_id)
                run = delegation.run_id
                self.assertEqual((accepted["status"], history.agent_run(run).status), ("accepted", "running"))

                # The main session has settled. Completion alone must wake it; no new CLI message.
                model.finish_child.set()
                self.assertEqual(await _outcome(history, session, 3), (DeliveryStatus.FULL, _RETURNED))
                returned = next(row.message.content for row in history.read_after(session)
                                if isinstance(row.message.content, str) and row.message.content.startswith(AGENT_REPLY_PREFIX))
                self.assertEqual(json.loads(returned.removeprefix(AGENT_REPLY_PREFIX)), {
                    "agent_id": child, "run_id": run, "status": "completed", "result": _RESULT,
                    "delegation_ids": [delegation_id],
                })
                self.assertIsNotNone(history.agent_run(run).reply_enqueued_at)
                self.assertEqual([row.message.content for row in history.read_after(history.agent(child).session_id)], [_PROMPT, _RESULT])
                self.assertNotIn(_PROMPT, [row.message.content for row in history.read_after(session)])

                transport.push("明确失败测试")
                self.assertEqual(await _outcome(history, session, 4), (DeliveryStatus.FULL, _FAILED))
                transport.push("未知投递测试")
                self.assertEqual(await _outcome(history, session, 5), (DeliveryStatus.FULL, _UNKNOWN))
                while len(transport.sends) < 4:
                    await asyncio.sleep(0.002)
                self.assertEqual(transport.sends, [_ACCEPTED, _RETURNED, _FAILED, _UNKNOWN])
                self.assertFalse(transport.meeting_lookups)

        voice_settings = replace(settings, feishu=replace(settings.feishu, mode="voice"))
        desktop, device, tts = _Desktop(), _AudioDevice(), _TTS()
        with closing(History(voice_settings.database)) as history:
            async with _service(voice_settings, history, model, transport, asr, desktop=desktop, tts=tts, device=device):
                desktop.ringing_now = True
                while not desktop.connected:
                    await asyncio.sleep(0.002)
                self.assertEqual(desktop.clicks, 1)
                self.assertEqual(transport.meeting_lookups, ["list", "detail"])
                device.frames(_HOT, 3)
                device.frames(bytes(len(_HOT)), 2)
                await tts.old_buffered.wait()
                self.assertEqual(device.pull(4 + INPUT_RATE // 2), _PCM * (4 + INPUT_RATE // 2))
                device.frames(_HOT, 3)
                device.frames(bytes(len(_HOT)), 2)
                await asr.started.wait()
                status, prefix = await _outcome(history, session, 6)
                self.assertEqual((status, prefix), (DeliveryStatus.PARTIAL, _OLD_VOICE_PREFIX))
                await tts.old_closed.wait()
                self.assertEqual(device.pull(4), bytes(8))
                self.assertEqual(len(asr.files), 2)
                self.assertEqual(asr.files[-1].content_type, "audio/wav")
                with wave.open(io.BytesIO(asr.files[-1].data), "rb") as recording:
                    self.assertEqual((recording.getnchannels(), recording.getsampwidth(), recording.getframerate()),
                                     (1, 2, INPUT_RATE))
                    self.assertIn(_HOT, recording.readframes(recording.getnframes()))

                asr.release.set()
                await tts.new_buffered.wait()
                self.assertEqual(device.pull(4), _PCM * 4)
                self.assertEqual(await _outcome(history, session, 7), (DeliveryStatus.FULL, _NEW_VOICE))
                self.assertEqual(tts.calls, [_OLD_VOICE_PREFIX, _OLD_VOICE_TAIL, _NEW_VOICE])
                while len(transport.sends) < 6:
                    await asyncio.sleep(0.002)
                self.assertEqual(transport.sends, [_ACCEPTED, _RETURNED, _FAILED, _UNKNOWN, prefix, _NEW_VOICE])
                self.assertEqual(transport.posts, [
                    {"zh_cn": {"content": [[{"tag": "text", "text": text, "style": ["bold"]}]]}}
                    for text in (_OLD_INPUT, _ASR_TEXT)
                ])
                tts.new_buffered.clear()
                transport.push(_TEXT_IN_CALL)
                await tts.new_buffered.wait()
                self.assertEqual(device.pull(4), _PCM * 4)
                self.assertEqual(await _outcome(history, session, 8), (DeliveryStatus.FULL, _TEXT_REPLY))
                self.assertEqual(tts.calls, [_OLD_VOICE_PREFIX, _OLD_VOICE_TAIL, _NEW_VOICE, _TEXT_REPLY])
                while len(transport.sends) < 7:
                    await asyncio.sleep(0.002)
                self.assertEqual(transport.sends[-1], _TEXT_REPLY)
                desktop.connected = False
                while not device.closed:
                    await asyncio.sleep(0.002)
            self.assertTrue(device.closed)
            before_restart = history.read_after(session)

        # Reopen the same database with a fresh model and real text-mode composition.
        restart_model = _ScriptedModel()
        with closing(History(settings.database)) as history:
            self.assertEqual(history.read_after(session), before_restart)
            async with _service(settings, history, restart_model, transport, asr):
                transport.push("重启后的新输入")
                self.assertEqual(await _outcome(history, session, 9), (DeliveryStatus.FULL, _RESTARTED))
                while len(transport.sends) < 8:
                    await asyncio.sleep(0.002)
                self.assertEqual(transport.sends,
                                 [_ACCEPTED, _RETURNED, _FAILED, _UNKNOWN, prefix, _NEW_VOICE, _TEXT_REPLY, _RESTARTED])
                messages = [row.message.content for row in history.read_after(session)]
                self.assertEqual(messages.count(_ASR_TEXT), 1)
                self.assertEqual(messages.count(_USER), 1)
                self.assertEqual(transport.meeting_lookups, ["list", "detail"])
                self.assertEqual(history.agent_run(run).status, "completed")
                model_history = [message.content for message in restart_model.restart_request]
                for recorded in (_ACCEPTED, _RETURNED, _FAILED, _UNKNOWN, prefix, _ASR_TEXT, _NEW_VOICE):
                    self.assertIn(recorded, model_history)
                self.assertNotIn(_OLD_VOICE, model_history)
