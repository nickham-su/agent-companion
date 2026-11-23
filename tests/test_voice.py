import asyncio
from contextlib import closing
from dataclasses import replace
import io
import unittest
from unittest.mock import patch
import wave

from agent_companion.app import run_feishu
from agent_companion.channels.feishu.channel import FeishuPoller
from agent_companion.config import FeishuSettings, RuntimeSettings, Settings, VoiceSettings
from agent_companion.protocol import (DeliveryStatus, Message, ReplyInfo, ReplyStream, TextDelta,
                                      ToolCall, ToolReady, ToolSpec)
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.speech.audio import AudioFormat
from agent_companion.speech.asr import AsrError
from agent_companion.speech.playback import VoiceOutput
from agent_companion.speech.tts import TtsError
from agent_companion.speech.vad import INPUT_FRAMES, VoiceSetupError
from agent_companion.speech.voice import VoiceChannel
from agent_companion.storage.sqlite import History


_RATE = 16000
_SETTINGS = VoiceSettings(playback_rate=_RATE, speech_frames=3, silence_seconds=0.06, delivery_timeout=0.7)
_HOT = b"\1\0" * INPUT_FRAMES
_QUIET = bytes(len(_HOT))
_A = b"\2\0"
_B = b"\3\0"


class _Device:
    input_format = AudioFormat(_RATE)
    output_format = AudioFormat(_RATE)

    def __init__(self):
        self.started = asyncio.Event()
        self.closed = False
        self.fault = False

    def start(self, capture, render):
        self.capture = capture
        self.render = render
        self.started.set()

    def check_active(self):
        assert not self.closed and not self.fault

    def close(self):
        self.closed = True

    def frame(self, hot, count=1, *, overflow=False):
        for _ in range(count):
            self.capture(memoryview(_HOT if hot else _QUIET), overflow)

    def pull(self, frames):
        out = bytearray(frames * 2)
        self.render(memoryview(out))
        return bytes(out)


class _Vad:
    def __init__(self):
        self.count = 0
        self.resets = 0

    def probability(self, pcm):
        self.count += 1
        return float(pcm == _HOT)

    def reset(self):
        self.resets += 1


class _ASR:
    def __init__(self, text="转写结果", *, block=False):
        self.files = []
        self.text = text
        self.called = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        if not block:
            self.release.set()

    async def transcribe(self, file):
        self.files.append(file)
        self.called.set()
        try:
            await self.release.wait()
            if isinstance(self.text, Exception):
                raise self.text
            return self.text
        finally:
            self.finished.set()


class _TTS:
    def __init__(self, size=4):
        self.calls = []
        self.size = size

    async def synthesize(self, text, *, sample_rate):
        self.calls.append((text, sample_rate))
        yield _A * self.size


def _reply(id_, *text, close=True):
    stream = ReplyStream(ReplyInfo(id_, "main", "run", asyncio.get_running_loop().time()))
    for part in text:
        stream.publish(part)
    if close:
        stream.close()
    return stream


async def _until(condition, *, timeout=1):
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.002)


async def _processed(vad, count):
    await _until(lambda: vad.count >= count)
    # 让执行线程的决策回到事件循环。
    await asyncio.sleep(0.012)


class VoiceOutputTest(unittest.IsolatedAsyncioTestCase):
    async def test_first_block_plays_before_stream_end_and_next_segment_gated_by_buffer(self):
        release = asyncio.Event()
        calls = []

        class TTS:
            async def synthesize(self, text, *, sample_rate):
                calls.append(text)
                if len(calls) == 1:
                    yield _A * 4
                    await release.wait()
                    yield _A * 12
                else:
                    yield _B * 2

        # 下一段只能在待播低于 8 个样本的阈值时提交。
        output = VoiceOutput(TTS(), replace(_SETTINGS, prefetch_seconds=8 / _RATE))
        device = _Device()
        device.render = output.render
        task = asyncio.create_task(output.deliver(_reply("seg", "第一", "句。第二句。")))
        try:
            await _until(lambda: calls == ["第一句。"])
            self.assertEqual(device.pull(4), _A * 4)
            await asyncio.sleep(0.015)
            self.assertEqual(len(calls), 1)  # 未结束，即便缓冲为空也不请求下一段
            release.set()
            await asyncio.sleep(0.02)
            self.assertEqual(device.pull(4), _A * 4)
            await asyncio.sleep(0.02)
            self.assertEqual(len(calls), 1)  # 缓冲恰好等于阈值
            self.assertEqual(device.pull(1), _A)
            await _until(lambda: len(calls) == 2)
            self.assertEqual(device.pull(9), _A * 7 + _B * 2)
            receipt = await task
            self.assertEqual((receipt.status, receipt.confirmed), (DeliveryStatus.FULL, "第一句。第二句。"))
        finally:
            await output.close()

    async def test_text_limit_closes_consumer_without_blocking_model_publisher(self):
        tts = _TTS()
        output = VoiceOutput(tts, _SETTINGS)
        device = _Device()
        device.render = output.render
        reply = _reply("limit", "已确认。", close=False)
        task = asyncio.create_task(output.deliver(reply))
        try:
            await _until(lambda: bool(tts.calls))
            reply.publish("未发送" * 10_000)
            reply.publish("也不发送。")
            self.assertTrue(reply.overflowed)
            self.assertEqual(device.pull(4), _A * 4)
            receipt = await task
            self.assertEqual((receipt.status, receipt.confirmed), (DeliveryStatus.FAILED, "已确认。"))
            self.assertEqual(len(tts.calls), 1)
        finally:
            await output.close()

    async def test_tts_diagnostics_preserve_known_category_but_not_raw_details(self):
        class FailingTTS:
            def __init__(self, failure):
                self.failure = failure

            async def synthesize(self, text, *, sample_rate):
                if False:
                    yield b""
                raise self.failure

        for detail, expected in [("tts_business_error", "tts_business_error"),
                                 ("synthetic arbitrary upstream text", "tts_failed")]:
            with self.subTest(expected=expected):
                output = VoiceOutput(FailingTTS(TtsError(detail)), _SETTINGS)
                try:
                    with self.assertLogs("agent_companion.speech.playback", level="WARNING") as logs:
                        receipt = await output.deliver(_reply("synthetic-reply", "正文。"))
                    self.assertEqual(receipt.status, DeliveryStatus.FAILED)
                    self.assertIn(f"({expected}); reply=synthetic-reply", logs.output[0])
                    self.assertNotIn("arbitrary upstream", logs.output[0])
                    self.assertNotIn("正文", logs.output[0])
                finally:
                    await output.close()

    async def test_tts_failure_or_channel_timeout_keeps_confirmed_prefix(self):
        for failure, expected in ((True, DeliveryStatus.FAILED), (False, DeliveryStatus.UNKNOWN)):
            with self.subTest(failure=failure):
                first = asyncio.Event()
                release = asyncio.Event()
                closed = asyncio.Event()

                class TTS:
                    async def synthesize(self, text, *, sample_rate):
                        try:
                            yield _A * _RATE
                            first.set()
                            await release.wait()
                            if failure:
                                raise TtsError("test_failure")
                        finally:
                            closed.set()

                output = VoiceOutput(TTS(), replace(_SETTINGS, delivery_timeout=0.15))
                device = _Device()
                device.render = output.render
                task = asyncio.create_task(output.deliver(_reply("fail", "甲乙丙丁戊己庚辛。")))
                try:
                    await first.wait()
                    self.assertEqual(device.pull(_RATE // 2), _A * (_RATE // 2))
                    if failure:
                        release.set()
                    receipt = await task
                    self.assertEqual((receipt.status, receipt.confirmed), (expected, "甲乙"))
                    await _until(closed.is_set)
                    self.assertEqual(device.pull(1), b"\0\0")
                finally:
                    await output.close()


class VoiceInputTest(unittest.IsolatedAsyncioTestCase):
    async def test_declared_formats_must_match_before_opening_audio(self):
        for name, declared in [("input", AudioFormat(48_000)), ("input", AudioFormat(_RATE, channels=2)),
                               ("output", AudioFormat(_RATE, sample_format="float32"))]:
            with self.subTest(name=name, declared=declared):
                device = _Device()
                setattr(device, f"{name}_format", declared)
                voice = VoiceChannel(_ASR(), _TTS(), _Vad(), _SETTINGS, asr_timeout=0.1)
                with self.assertRaisesRegex(VoiceSetupError, f"voice_{name}_device_format_mismatch"):
                    async with voice.connected(device, lambda _: None):
                        self.fail("invalid format must not start")
                self.assertFalse(device.started.is_set())

    async def test_debounce_wav_once_and_short_pause_inside_turn(self):
        asr, tts, vad, device = _ASR(), _TTS(), _Vad(), _Device()
        voice = VoiceChannel(asr, tts, vad, _SETTINGS, asr_timeout=0.5)
        received = []
        async with voice.connected(device, received.append):
            device.frame(True, 2)  # 小于连续三帧门槛
            device.frame(False, 2)
            await _processed(vad, 4)
            self.assertEqual(asr.files, [])
            device.frame(True, 3)
            device.frame(False)  # 32 ms 不结束
            device.frame(True)
            await _processed(vad, 9)
            self.assertFalse(asr.called.is_set())
            device.frame(False, 2)
            await _until(lambda: received == ["转写结果"])
            device.frame(False, 4)
            await _processed(vad, 15)
            self.assertEqual(len(asr.files), 1)
            self.assertEqual(asr.files[0].content_type, "audio/wav")
            with wave.open(io.BytesIO(asr.files[0].data), "rb") as file:
                self.assertEqual((file.getnchannels(), file.getsampwidth(), file.getframerate()), (1, 2, 16000))
                self.assertIn(_HOT + _QUIET + _HOT, file.readframes(file.getnframes()))
        self.assertTrue(device.closed)

    async def test_no_old_audio_during_speech_pending_asr_or_after_asr_result(self):
        for result in ("新输入", "", AsrError("synthetic")):
            with self.subTest(result=type(result).__name__):
                asr, tts, vad, device = _ASR(result, block=True), _TTS(), _Vad(), _Device()
                voice = VoiceChannel(asr, tts, vad, _SETTINGS, asr_timeout=0.5)
                received = []
                async with voice.connected(device, received.append):
                    speaking = _reply("speaking", "旧回复。")
                    pending = _reply("pending", "旧回复。")
                    idle_late = _reply("idle_late", "旧回复。")
                    device.frame(True, 3)
                    await _processed(vad, 3)
                    self.assertEqual((await voice.deliver(speaking)).status, DeliveryStatus.ZERO)
                    device.frame(False, 2)
                    await asr.called.wait()
                    self.assertEqual((await voice.deliver(pending)).status, DeliveryStatus.ZERO)
                    started_in_asr = _reply("created_in_asr", "还是旧回复。")
                    asr.release.set()
                    await asr.finished.wait()
                    await asyncio.sleep(0.012)
                    for old in (idle_late, started_in_asr):
                        self.assertEqual((await voice.deliver(old)).status, DeliveryStatus.ZERO)
                    self.assertEqual(tts.calls, [])
                    self.assertEqual(received, [result] if isinstance(result, str) and result else [])
                    new = asyncio.create_task(voice.deliver(_reply("new", "新回复。")))
                    await _until(lambda: bool(tts.calls))
                    self.assertEqual(device.pull(4), _A * 4)
                    self.assertEqual((await new).status, DeliveryStatus.FULL)

    async def test_partial_interrupt_does_not_wait_for_tts_cancellation_and_drops_late_block(self):
        asr, vad, device = _ASR(block=True), _Vad(), _Device()
        first, cancelled, late, finalize = (asyncio.Event() for _ in range(4))

        class SlowTTS:
            async def synthesize(self, text, *, sample_rate):
                if text == "新回复。":
                    yield _B * 4
                    return
                try:
                    yield _A * _RATE
                    first.set()
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await late.wait()
                    yield _A * 8  # 故意模拟吞取消的适配器
                finally:
                    finalize.set()

        voice = VoiceChannel(asr, SlowTTS(), vad, _SETTINGS, asr_timeout=0.5)
        async with voice.connected(device, lambda _: None):
            old = asyncio.create_task(voice.deliver(_reply("old", "甲乙丙丁戊己庚辛。")))
            try:
                await first.wait()
                self.assertEqual(device.pull(_RATE // 2), _A * (_RATE // 2))
                device.frame(True, 3)
                receipt = await asyncio.wait_for(old, 0.3)
                self.assertEqual((receipt.status, receipt.confirmed), (DeliveryStatus.PARTIAL, "甲乙"))
                await cancelled.wait()
                self.assertFalse(finalize.is_set())
                self.assertEqual(device.pull(4), bytes(8))
                device.frame(False, 2)
                await asr.called.wait()
                asr.release.set()
                await asr.finished.wait()
                await asyncio.sleep(0.012)
                new = asyncio.create_task(voice.deliver(_reply("new", "新回复。")))
                late.set()
                await finalize.wait()
                await asyncio.sleep(0.012)
                self.assertEqual(device.pull(8), _B * 4 + bytes(8))
                self.assertEqual((await new).confirmed, "新回复。")
            finally:
                late.set()

    async def test_device_fault_returns_prefix_and_never_falls_back_to_text(self):
        asr, vad, device = _ASR(), _Vad(), _Device()
        first = asyncio.Event()

        class TTS:
            async def synthesize(self, text, *, sample_rate):
                yield _A * _RATE
                first.set()
                await asyncio.Event().wait()

        voice = VoiceChannel(asr, TTS(), vad, _SETTINGS, asr_timeout=0.5)
        with patch("agent_companion.speech.voice._DEVICE_CHECK_SECONDS", 0.02):
            async with voice.connected(device, lambda _: None):
                old = asyncio.create_task(voice.deliver(_reply("fault", "甲乙丙丁戊己庚辛。")))
                await _until(first.is_set)
                self.assertEqual(device.pull(_RATE // 2), _A * (_RATE // 2))
                device.fault = True
                receipt = await asyncio.wait_for(old, 0.3)
                self.assertEqual((receipt.status, receipt.confirmed), (DeliveryStatus.FAILED, "甲乙"))
                self.assertEqual(device.pull(1), bytes(2))
                self.assertEqual((await voice.deliver(_reply("new", "不回退。"))).status, DeliveryStatus.FAILED)
        self.assertTrue(device.closed)

    async def test_device_fault_drops_waiting_recording_and_ignores_late_asr(self):
        vad, device = _Vad(), _Device()
        started, cancelled, release, finished = (asyncio.Event() for _ in range(4))
        requests = []

        class SlowASR:
            async def transcribe(self, file):
                requests.append(file)
                if len(requests) != 1:
                    return "不应上传的第二轮"
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                    return "故障后迟到的转写"
                finally:
                    finished.set()

        voice = VoiceChannel(SlowASR(), _TTS(), vad, _SETTINGS, asr_timeout=1)
        submitted = []
        with patch("agent_companion.speech.voice._DEVICE_CHECK_SECONDS", 0.02):
            async with voice.connected(device, submitted.append):
                try:
                    device.frame(True, 3)
                    device.frame(False, 2)
                    await _until(started.is_set)
                    device.frame(True, 3)
                    device.frame(False, 2)
                    await _until(lambda: voice._turns.qsize() == 1)
                    self.assertEqual(len(requests), 1)

                    device.fault = True
                    await _until(lambda: device.closed and cancelled.is_set())
                    self.assertFalse(finished.is_set())
                    self.assertTrue(voice._turns.empty())
                    self.assertEqual((await voice.deliver(_reply("disabled", "停用。"))).status,
                                     DeliveryStatus.FAILED)

                    device.frame(True, 3)  # 模拟停止前已经排定的设备回调
                    device.frame(False, 2)
                    release.set()
                    await _until(finished.is_set)
                    await _until(lambda: voice._asr_worker is not None and voice._asr_worker.done())
                    self.assertEqual(voice._pending, 0)
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(submitted, [])
                finally:
                    release.set()

    async def test_overlong_gap_and_asr_timeout_drop_only_affected_turns(self):
        asr, vad, device = _ASR(block=True), _Vad(), _Device()
        settings = replace(_SETTINGS, max_turn_seconds=0.2)
        voice = VoiceChannel(asr, _TTS(), vad, settings, asr_timeout=0.03)
        received = []
        async with voice.connected(device, received.append):
            device.frame(True, 9)
            device.frame(False, 2)
            await _processed(vad, 11)
            self.assertFalse(asr.called.is_set())
            device.frame(True, 4, overflow=True)
            device.frame(False, 3)
            await _processed(vad, 18)
            self.assertFalse(asr.called.is_set())
            self.assertGreater(vad.resets, 0)
            device.frame(True, 3)
            device.frame(False, 2)
            await asr.called.wait()
            await _until(asr.finished.is_set)
            self.assertEqual(received, [])
            asr.release.set()
            device.frame(True, 3)
            device.frame(False, 2)
            await _until(lambda: received == ["转写结果"])
            self.assertEqual(len(asr.files), 2)

    async def test_runtime_tool_continues_and_asr_enters_same_session(self):
        asr, tts, vad, device = _ASR(block=True), _TTS(), _Vad(), _Device()
        tool_started, release_tool = asyncio.Event(), asyncio.Event()
        tool_done = asyncio.Event()
        calls = []

        class Model:
            async def stream(self, messages, tools, *, thinking):
                calls.append(messages)
                if len(calls) == 1:
                    yield ToolReady(ToolCall("a", "wait", "{}"))
                else:
                    yield TextDelta("新回复。")

        async def tool(args, context):
            tool_started.set()
            await release_tool.wait()
            tool_done.set()
            return "工具结果"

        tools = Tools((Tool(ToolSpec("wait", "等待", {"type": "object"}), tool),))
        voice = VoiceChannel(asr, tts, vad, _SETTINGS, asr_timeout=0.5)
        with closing(History(":memory:")) as history:
            runtime = Runtime("main", history, Model(), voice, tools=tools)
            try:
                async with voice.connected(device, lambda text: runtime.submit_batch([Message("user", text)])):
                    runtime.submit_batch([Message("user", "运行工具")])
                    await tool_started.wait()
                    device.frame(True, 3)
                    device.frame(False, 2)
                    await asr.called.wait()
                    self.assertFalse(tool_done.is_set())
                    self.assertEqual(tts.calls, [])
                    asr.release.set()
                    await asr.finished.wait()
                    await asyncio.sleep(0.012)
                    release_tool.set()
                    await _until(lambda: len(tts.calls) == 1)
                    self.assertEqual(device.pull(4), _A * 4)
                    await runtime.wait_idle()
                    self.assertTrue(tool_done.is_set())
                    self.assertEqual([msg.content for msg in calls[-1] if msg.role == "tool"], ["工具结果"])
                    self.assertEqual(calls[-1][-1].content, "转写结果")
            finally:
                await runtime.close()

    async def test_interrupted_mixed_response_keeps_model_and_tool_alive(self):
        asr, device, vad = _ASR(block=True), _Device(), _Vad()
        release_model, ran_tool, spoke = (asyncio.Event() for _ in range(3))
        prompts = []

        class TTS:
            async def synthesize(self, text, *, sample_rate):
                yield _A * _RATE
                spoke.set()
                await asyncio.Event().wait()

        class Model:
            async def stream(self, messages, tools, *, thinking):
                prompts.append(messages)
                if len(prompts) == 1:
                    yield TextDelta("甲乙丙丁戊己庚辛。")
                    await release_model.wait()
                    yield TextDelta("未播的正文。")
                    yield ToolReady(ToolCall("a", "wait", "{}"))
                else:
                    yield TextDelta("仍在说话，不能播报。")

        async def tool(args, context):
            ran_tool.set()
            return "工具结果"

        tools = Tools((Tool(ToolSpec("wait", "测试工具", {"type": "object"}), tool),))
        voice = VoiceChannel(asr, TTS(), vad, _SETTINGS, asr_timeout=0.5)
        with closing(History(":memory:")) as history:
            runtime = Runtime("main", history, Model(), voice, tools=tools)
            try:
                async with voice.connected(device, lambda _: None):
                    runtime.submit_batch([Message("user", "测试请求")])
                    await _until(spoke.is_set)
                    self.assertEqual(device.pull(_RATE // 2), _A * (_RATE // 2))
                    device.frame(True, 3)
                    await _processed(vad, 3)
                    self.assertEqual(device.pull(4), bytes(8))
                    self.assertFalse(ran_tool.is_set())
                    release_model.set()
                    await asyncio.wait_for(runtime.wait_idle(), 1)
                    self.assertTrue(ran_tool.is_set())
                    self.assertEqual([msg.message.content for msg in history.read_after("main")],
                                     ["测试请求", "甲乙", "工具结果", ""])
                    self.assertEqual([r.delivery_status for r in history.outcomes("main")],
                                     [DeliveryStatus.PARTIAL, DeliveryStatus.ZERO])
            finally:
                await runtime.close()

    async def test_app_voice_keeps_feishu_input_and_voice_playback_then_cleans_device(self):
        asr, tts, vad, device = _ASR(), _TTS(), _Vad(), _Device()
        settings = Settings(feishu=FeishuSettings("oc_test", "ou_peer", "ou_self", mode="voice"), voice=_SETTINGS,
                            runtime=RuntimeSettings())
        called = asyncio.Event()
        messages = []

        class Model:
            async def stream(self, prompt, tools, *, thinking):
                messages.extend(prompt)
                called.set()
                yield TextDelta("飞书回复。")

        # 用真实 poll_once／持久化移交流程，不等待真实 10 秒轮询栅格。
        original = FeishuPoller.poll_once

        async def run(poller):
            ticks = iter((100, 102))
            poller._clock = lambda: next(ticks)
            await original(poller)
            await original(poller)
            await asyncio.Event().wait()

        class FakeCli:
            async def list_page(self, start, end, token, *, page_size):
                msg = {"message_id": "om_text", "create_time": "101000", "msg_type": "text",
                       "chat_id": "oc_test", "sender": {"id": "ou_peer", "id_type": "open_id", "sender_type": "user"},
                       "body": {"content": '{"text":"飞书输入"}'}}
                return [msg], None

        with closing(History(":memory:")) as history:
            with patch.object(FeishuPoller, "run", run):
                task = asyncio.create_task(run_feishu(settings, history, Model(), asr, tts=tts, vad=vad,
                                                     audio=device, cli=FakeCli()))
                await asyncio.wait_for(device.started.wait(), 1)
                try:
                    await asyncio.wait_for(called.wait(), 1)
                    await _until(lambda: bool(tts.calls))
                    self.assertEqual(messages[-1].content, "飞书输入")
                    self.assertEqual(device.pull(4), _A * 4)
                    await _until(lambda: len(history.outcomes("main")) == 1)
                    self.assertEqual(history.tail("main", 1)[0].message.content, "飞书回复。")
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(device.closed)
