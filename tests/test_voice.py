import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing, closing
from dataclasses import replace
import io
from threading import Event
import unittest
from unittest.mock import patch
import wave
from weakref import ref

from agent_companion import images
from agent_companion.config import VoiceSettings
from agent_companion.protocol import (DeliveryStatus, Message, MessageDelivery, ReplyInfo, ReplyStream, TextDelta,
                                      ToolCall, ToolReady, ToolSpec)
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.tools import Tool, Tools
from agent_companion.speech.audio import AudioFormat
from agent_companion.speech.asr import AsrError
from agent_companion.speech.playback import VoiceOutput, _segments
from agent_companion.speech.tts import TtsError
from agent_companion.speech.vad import INPUT_FRAMES, VoiceSetupError
from agent_companion.speech.voice import VoiceChannel
from agent_companion.storage.sqlite import History
from test_message_support import PNG_BYTES, image_workspace, tool_calls


_RATE = 16000
_SETTINGS = VoiceSettings(output_sample_rate=_RATE, speech_frames=3, silence_seconds=0.06, delivery_timeout=0.7)
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


def _transcripts(submit):
    def on_input(text, _active):
        if text:
            submit(text)
    return on_input


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


class SpeechSegmentationTest(unittest.IsolatedAsyncioTestCase):
    async def test_short_first_sentence_is_emitted_while_reply_is_still_open(self):
        for end in "。！？.!?\n":
            with self.subTest(end=end):
                first = "好，行;稍等；我看看,马上"
                reply = _reply("segments", "。\n好，行;", "稍等；我看看,马上", end, close=False)
                async with aclosing(_segments(reply)) as segments:
                    self.assertEqual(await asyncio.wait_for(anext(segments), 1), "。")
                    self.assertEqual(await asyncio.wait_for(anext(segments), 1), "\n")
                    self.assertEqual(await asyncio.wait_for(anext(segments), 1), first + end)

    async def test_followups_wait_for_a_sentence_end_after_at_least_thirty_body_chars(self):
        short = "甲" * 13 + "， " + "乙" * 16 + "。"  # 29 个正文字符
        sentence = "丙丁？"  # 攒到「丙」恰好 30 个，仍需等到句尾。
        exact = "字" * 30 + "!"
        reply = _reply("segments", "好。", short, sentence, exact, "末尾")
        self.assertEqual([part async for part in _segments(reply)], ["好。", short + sentence, exact, "末尾"])

    async def test_hard_cap_counts_all_characters_and_applies_to_every_segment(self):
        first = "甲" * 198 + "， "
        second = "乙" * 200
        tail = "剩下。下一句！"
        reply = _reply("segments", first + second + tail)
        self.assertEqual([part async for part in _segments(reply)], [first, second, tail])


class VoiceOutputTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _render(output, frames=4):
        data = bytearray(frames * 2)
        output.render(memoryview(data))
        return bytes(data)

    async def test_first_audio_latency_links_model_text_to_device_output(self):
        finish_sentence = asyncio.Event()
        tts = _TTS()
        output = VoiceOutput(tts, _SETTINGS)

        class Model:
            async def stream(self, messages, tools, *, thinking, instructions=None):
                yield TextDelta("合成测试正文")
                await finish_sentence.wait()
                yield TextDelta("。")

        # 启动后尚无可朗读句子，设备先收到本机补的静音。
        with closing(History(":memory:")) as history:
            runtime = Runtime("main", history, Model(), output)
            try:
                with self.assertLogs("agent_companion", level="DEBUG") as logs:
                    runtime.submit_messages([MessageDelivery(Message("user", "合成测试输入"), "trigger")])
                    await _until(lambda: any("voice playback started;" in row for row in logs.output))
                    self.assertEqual(self._render(output), bytes(8))
                    self.assertEqual(tts.calls, [])
                    finish_sentence.set()
                    await _until(lambda: bool(tts.calls))
                    self.assertEqual(await asyncio.to_thread(self._render, output), _A * 4)
                    await runtime.wait_idle()
            finally:
                await runtime.close()

        first = next(row for row in logs.output if "model first text;" in row)
        rows = [row for row in logs.output if "voice first audio latency;" in row]
        self.assertEqual(len(rows), 1)
        fields = dict(item.split("=", 1) for item in rows[0].split("; ", 1)[1].split())
        self.assertIn(f"reply={fields['reply']} ", first)
        parts = [int(fields[key]) for key in ("first_text_to_tts_ms", "tts_to_first_pcm_ms", "first_pcm_to_output_ms")]
        self.assertTrue(all(value >= 0 for value in parts))
        self.assertAlmostEqual(int(fields["first_text_to_output_ms"]), sum(parts), delta=2)
        self.assertNotIn("合成测试正文", rows[0])

    async def test_first_audio_latency_leaves_device_timing_missing_if_interrupted_before_output(self):
        tts = _TTS()
        output = VoiceOutput(tts, _SETTINGS)
        reply = _reply("interrupted", "合成测试正文。")
        reply.first_text_at = reply.info.started_at
        with self.assertLogs("agent_companion.speech.playback", level="DEBUG") as logs:
            task = asyncio.create_task(output.deliver(reply))
            await _until(lambda: bool(tts.calls))
            output.interrupt()  # 有 TTS 首块，但尚无设备回调取走它。
            self.assertEqual(self._render(output), bytes(8))
            self.assertEqual((await task).status, DeliveryStatus.ZERO)
        latency = next(row for row in logs.output if "voice first audio latency;" in row)
        self.assertRegex(latency, r"first_text_to_tts_ms=\d+ tts_to_first_pcm_ms=\d+ "
                                  r"first_pcm_to_output_ms=- first_text_to_output_ms=-$")

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
            with self.assertLogs("agent_companion.speech.playback", level="DEBUG") as logs:
                await _until(lambda: bool(tts.calls))
                reply.publish("未发送" * 10_000)
                reply.publish("也不发送。")
                self.assertTrue(reply.overflowed)
                self.assertEqual(device.pull(4), _A * 4)
                receipt = await task
            self.assertEqual((receipt.status, receipt.confirmed), (DeliveryStatus.FAILED, "已确认。"))
            debug = "\n".join(logs.output)
            self.assertIn("voice playback finished; reply=limit status=failed confirmed_chars=4", debug)
            self.assertNotIn(receipt.confirmed, debug)
            self.assertNotIn("未发送", debug)
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
                        if text == "已交付。":
                            yield _B * 4
                            return
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
                task = asyncio.create_task(output.deliver(_reply("fail", "已交付。甲乙丙丁戊己庚辛。")))
                try:
                    await first.wait()
                    self.assertEqual(device.pull(4 + _RATE // 2), _B * 4 + _A * (_RATE // 2))
                    if failure:
                        release.set()
                    receipt = await task
                    self.assertEqual((receipt.status, receipt.confirmed), (expected, "已交付。"))
                    await _until(closed.is_set)
                    self.assertEqual(device.pull(1), b"\0\0")
                finally:
                    await output.close()

    async def test_interrupt_only_confirms_complete_segments_taken_by_device(self):
        for first_segment_delivered in (False, True):
            with self.subTest(first_segment_delivered=first_segment_delivered):
                buffered = asyncio.Event()

                class TTS:
                    async def synthesize(self, text, *, sample_rate):
                        if text == "已交付。":
                            yield _A * 4
                        else:
                            yield bytes(2) + _B * 3  # 未交完的合成段只取到一个静音采样。
                            asyncio.get_running_loop().call_soon(buffered.set)

                output = VoiceOutput(TTS(), _SETTINGS)
                self.addAsyncCleanup(output.close)
                device = _Device()
                device.render = output.render
                prefix = "已交付。" if first_segment_delivered else ""
                task = asyncio.create_task(output.deliver(_reply("stop", prefix + "尚未交付。")))
                await asyncio.wait_for(buffered.wait(), 1)
                self.assertEqual(device.pull(5 if first_segment_delivered else 1),
                                 (_A * 4 if first_segment_delivered else b"") + bytes(2))
                output.interrupt()
                receipt = await asyncio.wait_for(task, 1)
                self.assertEqual((receipt.status, receipt.confirmed),
                                 (DeliveryStatus.PARTIAL if first_segment_delivered else DeliveryStatus.ZERO, prefix))


    async def test_hard_deadline_decides_by_worker_completion_not_observation(self):
        loop = asyncio.get_running_loop()
        for on_time in (False, True):
            with self.subTest(on_time=on_time):
                buffered, drained, release = (asyncio.Event() for _ in range(3))
                clock = [loop.time()]

                def jump():
                    clock[0] += 3

                class TTS:
                    async def synthesize(self, text, *, sample_rate):
                        yield _A * 4
                        buffered.set()

                class Output(VoiceOutput):
                    async def _produce(self, reply, state):
                        await super()._produce(reply, state)
                        drained.set()
                        await release.wait()
                        if on_time:
                            loop.call_soon(jump)
                        else:
                            jump()

                output = Output(TTS(), replace(_SETTINGS, delivery_timeout=2))
                self.addAsyncCleanup(output.close)
                device = _Device()
                device.render = output.render
                with (patch.object(loop, "time", new=lambda: clock[0]),
                      patch.object(loop, "slow_callback_duration", float("inf"))):
                    task = asyncio.create_task(output.deliver(_reply("hard", "已播正文。")))
                    await buffered.wait()
                    self.assertEqual(device.pull(4), _A * 4)
                    clock[0] += .02  # 唤醒生产任务的设备排空轮询。
                    await drained.wait()
                    release.set()
                    receipt = await task
                self.assertEqual((receipt.status, receipt.confirmed), (
                    DeliveryStatus.FULL if on_time else DeliveryStatus.UNKNOWN, "已播正文。"))

    async def test_hard_deadline_excludes_segment_only_completed_after_cutoff(self):
        buffered, finish = asyncio.Event(), asyncio.Event()

        class TTS:
            async def synthesize(self, text, *, sample_rate):
                yield _A * 4
                buffered.set()
                await finish.wait()

        loop = asyncio.get_running_loop()
        clock = [loop.time()]
        output = VoiceOutput(TTS(), replace(_SETTINGS, delivery_timeout=2))
        self.addAsyncCleanup(output.close)
        device = _Device()
        device.render = output.render
        with (patch.object(loop, "time", new=lambda: clock[0]),
              patch.object(loop, "slow_callback_duration", float("inf"))):
            task = asyncio.create_task(output.deliver(_reply("hard", "迟到的合成终态。")))
            await buffered.wait()
            self.assertEqual(device.pull(4), _A * 4)
            clock[0] += 3
            finish.set()
            receipt = await task
        self.assertEqual((receipt.status, receipt.confirmed), (DeliveryStatus.UNKNOWN, ""))

    async def test_hangup_before_last_pcm_confirms_only_the_prefix(self):
        for at_cutoff in (False, True):
            with self.subTest(at_cutoff=at_cutoff):
                loop = asyncio.get_running_loop()
                clock = [loop.time()]
                buffered = asyncio.Event()

                class TTS(_TTS):
                    async def synthesize(self, text, *, sample_rate):
                        async for pcm in super().synthesize(text, sample_rate=sample_rate):
                            yield pcm
                        if len(self.calls) == 2:
                            buffered.set()

                output = VoiceOutput(TTS(), replace(_SETTINGS, delivery_timeout=2))
                self.addAsyncCleanup(output.close)
                device = _Device()
                device.render = output.render
                with (patch.object(loop, "time", new=lambda: clock[0]),
                      patch.object(loop, "slow_callback_duration", float("inf"))):
                    task = asyncio.create_task(output.deliver(_reply("stop", "已播句。", "未播句。")))
                    await buffered.wait()
                    self.assertEqual(device.pull(4), _A * 4)
                    clock[0] += 2 if at_cutoff else 0.02
                    output.interrupt()
                    self.assertEqual(device.pull(4), bytes(8))
                    receipt = await task
                self.assertEqual(receipt.confirmed, "已播句。")
                self.assertEqual(receipt.status, DeliveryStatus.UNKNOWN if at_cutoff else DeliveryStatus.PARTIAL)

    async def test_abandoned_stream_cannot_complete_playback(self):
        tts = _TTS()
        output = VoiceOutput(tts, _SETTINGS)
        self.addAsyncCleanup(output.close)
        device = _Device()
        device.render = output.render
        reply = _reply("abandoned", "已播句。", close=False)
        task = asyncio.create_task(output.deliver(reply))
        await _until(lambda: bool(tts.calls))
        self.assertEqual(device.pull(4), _A * 4)
        reply.abandon()
        receipt = await task
        self.assertEqual((receipt.status, receipt.confirmed), (DeliveryStatus.FAILED, "已播句。"))


class VoiceDeliveryWatchdogTest(unittest.IsolatedAsyncioTestCase):
    def setup_runtime(self, output, model, settings, **kwargs):
        history = History(":memory:")
        self.addCleanup(history.close)
        runtime = Runtime("main", history, model, output, delivery_idle_timeout=settings.idle_timeout, **kwargs)
        self.addAsyncCleanup(output.close)
        self.addAsyncCleanup(runtime.close)
        device = _Device()
        device.render = output.render
        return runtime, history, device

    async def test_completed_playback_before_hangup_and_idle_deadline_keeps_tool_continuation(self):
        buffered = asyncio.Event()
        requests = []
        loop = asyncio.get_running_loop()
        clock = [loop.time()]
        ordering = []

        class Model:
            async def stream(self, messages, tools, *, thinking, instructions=None):
                requests.append(messages)
                if len(requests) == 1:
                    yield ToolReady(ToolCall("one", "lookup", "{}"))
                    yield TextDelta("已播正文。")

        class TTS:
            async def synthesize(self, text, *, sample_rate):
                yield _A * 4
                buffered.set()

        class Output(VoiceOutput):
            async def _produce(self, reply, state):
                await super()._produce(reply, state)
                ordering.append("production_complete")

                def hangup():
                    clock[0] += .02  # 确保停播发生在 worker 完成之后、两个截止之前。
                    self.interrupt()
                    ordering.append("hangup")
                    clock[0] += 3  # 投递协程恢复前，同时越过滚动空闲与硬截止。

                loop.call_soon(hangup)  # 先于 worker.done() 唤醒投递协程的回调。

            async def deliver(self, reply):
                receipt = await super().deliver(reply)
                ordering.append("receipt")
                return receipt

        async def lookup(args, context):
            return "工具结果"

        settings = replace(_SETTINGS, idle_timeout=1, delivery_timeout=2)
        tools = Tools((Tool(ToolSpec("lookup", "查询", {"type": "object"}), lookup),))
        runtime, history, device = self.setup_runtime(Output(TTS(), settings), Model(), settings, tools=tools)
        with (patch.object(loop, "time", new=lambda: clock[0]),
              patch.object(loop, "slow_callback_duration", float("inf"))):
            runtime.submit_messages([MessageDelivery(Message("user", "运行工具"), "trigger")])
            await buffered.wait()
            self.assertEqual(device.pull(4), _A * 4)
            clock[0] += .02
            await runtime.wait_idle()
        self.assertEqual(ordering, ["production_complete", "hangup", "receipt"])
        self.assertEqual(history.outcomes("main")[0].delivery_status, DeliveryStatus.FULL)
        self.assertEqual([call.result for call in tool_calls(requests[1])], ["工具结果"])

    async def test_no_pcm_starts_idle_timer_only_after_model_closes(self):
        model_close, tts_started = asyncio.Event(), asyncio.Event()

        class Model:
            async def stream(self, messages, tools, *, thinking, instructions=None):
                yield TextDelta("未开始。")
                await model_close.wait()

        class TTS:
            async def synthesize(self, text, *, sample_rate):
                tts_started.set()
                await asyncio.Event().wait()
                yield _A

        settings = replace(_SETTINGS, idle_timeout=.05)
        runtime, history, device = self.setup_runtime(VoiceOutput(TTS(), settings), Model(), settings)
        runtime.submit_messages([MessageDelivery(Message("user", "合成测试"), "trigger")])
        await asyncio.wait_for(tts_started.wait(), 1)
        await asyncio.sleep(.08)
        self.assertEqual(history.outcomes("main"), [])
        model_close.set()
        async with asyncio.timeout(.4):
            while not history.outcomes("main"):
                self.assertEqual(device.pull(4), bytes(8))
                await asyncio.sleep(.01)
        self.assertEqual(history.outcomes("main")[0].delivery_status, DeliveryStatus.UNKNOWN)
        self.assertEqual(history.tail("main", 1)[0].message, Message("assistant", ""))

    async def test_playback_stalls_during_second_segment_next_request_only_sees_first(self):
        requests = []

        class Model:
            async def stream(self, messages, tools, *, thinking, instructions=None):
                requests.append(list(messages))
                if len(requests) == 1:
                    yield TextDelta("已交给设备。只播了一点。")

        tts = _TTS()
        settings = replace(_SETTINGS, idle_timeout=.08)
        runtime, history, device = self.setup_runtime(VoiceOutput(tts, settings), Model(), settings)
        runtime.submit_messages([MessageDelivery(Message("user", "先问"), "trigger")])
        await _until(lambda: len(tts.calls) == 2)
        self.assertEqual(device.pull(5), _A * 5)
        runtime.submit_messages([MessageDelivery(Message("user", "再问"), "trigger")])
        await asyncio.wait_for(runtime.wait_idle(), .5)
        self.assertEqual(history.outcomes("main")[0].delivery_status, DeliveryStatus.UNKNOWN)
        self.assertEqual(requests[1], [Message("user", "先问"), Message("assistant", "已交给设备。"),
                                       Message("user", "再问")])
        self.assertEqual(device.pull(4), bytes(8))

    async def test_device_keeps_pre_cutoff_chain_without_event_loop_sampling(self):
        buffered = asyncio.Event()
        second = "甲乙丙丁戊己庚辛壬癸" * 3 + "。"

        class Model:
            async def stream(self, messages, tools, *, thinking, instructions=None):
                yield TextDelta(f"第一句。{second}迟到句。")

        class TTS:
            async def synthesize(self, text, *, sample_rate):
                yield _A * 4
                if text == "迟到句。":
                    buffered.set()

        settings = replace(_SETTINGS, idle_timeout=2, delivery_timeout=50)
        runtime, history, device = self.setup_runtime(VoiceOutput(TTS(), settings), Model(), settings)
        loop = asyncio.get_running_loop()
        clock = [loop.time()]
        with (patch.object(loop, "time", new=lambda: clock[0]),
              patch.object(loop, "slow_callback_duration", float("inf"))):
            runtime.submit_messages([MessageDelivery(Message("user", "测试"), "trigger")])
            await buffered.wait()
            # 两次消费都发生在各自的滚动截止前；第二次已过原始截止。
            clock[0] += 1.75
            self.assertEqual(device.pull(4), _A * 4)
            clock[0] += 1.75
            self.assertEqual(device.pull(4), _A * 4)
            # 已过第二次消费延长后的截止：第三次不能掩盖证据或扩大确认前缀。
            clock[0] += 2.25
            self.assertEqual(device.pull(4), bytes(8))
            await runtime.wait_idle()
        self.assertEqual(history.outcomes("main")[0].delivery_status, DeliveryStatus.UNKNOWN)
        self.assertEqual(history.tail("main", 1)[0].message.content, "第一句。" + second)

    async def test_real_progress_renews_idle_but_never_the_hard_deadline(self):
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                produced = asyncio.Event()

                class Model:
                    async def stream(self, messages, tools, *, thinking, instructions=None):
                        yield TextDelta("持续播放。不会播完。" if continuous else "持续播放。")

                class TTS:
                    async def synthesize(self, text, *, sample_rate):
                        if text == "持续播放。":
                            yield _A * 120
                            produced.set()
                        else:
                            while True:
                                yield _B * 64
                                await asyncio.sleep(.005)

                settings = replace(_SETTINGS, idle_timeout=.10, delivery_timeout=.34 if continuous else .9)
                runtime, history, device = self.setup_runtime(VoiceOutput(TTS(), settings), Model(), settings)
                runtime.submit_messages([MessageDelivery(Message("user", "播放器合成测试"), "trigger")])
                await asyncio.wait_for(produced.wait(), 1)
                a, b = 0, 0
                async with asyncio.timeout(.7):
                    while not history.outcomes("main"):
                        pcm = device.pull(8)
                        a += pcm.count(_A)
                        b += pcm.count(_B)
                        await asyncio.sleep(.014)
                self.assertEqual(a, 120)  # 15 次带真实音频的回调，跨越两个空闲窗口。
                if continuous:
                    self.assertGreater(b, 0)
                self.assertEqual(history.outcomes("main")[0].delivery_status,
                                 DeliveryStatus.UNKNOWN if continuous else DeliveryStatus.FULL)
                self.assertEqual(history.tail("main", 1)[0].message, Message("assistant", "持续播放。"))


class VoiceInputTest(unittest.IsolatedAsyncioTestCase):
    async def test_declared_formats_must_match_before_opening_audio(self):
        for name, declared in [("input", AudioFormat(48_000)), ("input", AudioFormat(_RATE, channels=2)),
                               ("output", AudioFormat(_RATE, sample_format="float32"))]:
            with self.subTest(name=name, declared=declared):
                device = _Device()
                setattr(device, f"{name}_format", declared)
                voice = VoiceChannel(_ASR(), _TTS(), _Vad(), _SETTINGS, asr_timeout=0.1, max_audio_seconds=500)
                with self.assertRaisesRegex(VoiceSetupError, f"voice_{name}_device_format_mismatch"):
                    async with voice.connected(device, lambda *_: None):
                        self.fail("invalid format must not start")
                self.assertFalse(device.started.is_set())

    async def test_debounce_wav_once_and_short_pause_inside_turn(self):
        asr, tts, vad, device = _ASR(), _TTS(), _Vad(), _Device()
        voice = VoiceChannel(asr, tts, vad, _SETTINGS, asr_timeout=0.5, max_audio_seconds=500)
        received = []
        async with voice.connected(device, _transcripts(received.append)):
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

    async def test_omitted_segment_pause_keeps_one_asr_request_for_long_turn(self):
        asr, vad, device = _ASR(), _Vad(), _Device()
        settings = replace(_SETTINGS, silence_seconds=1)
        voice = VoiceChannel(asr, _TTS(), vad, settings, asr_timeout=0.5, max_audio_seconds=500)
        received = []
        async with voice.connected(device, _transcripts(received.append)):
            # 分批输入 10.24 秒人声，避免溢出设备采集队列。
            for batch in range(1, 21):
                device.frame(True, 16)
                await _processed(vad, batch * 16)
            device.frame(False, 16)  # 0.512 秒，位于短停顿示例和话轮结束阈值之间。
            await _processed(vad, 336)
            self.assertFalse(asr.called.is_set())
            device.frame(True, 4)
            await _processed(vad, 340)
            device.frame(False, 32)
            await _until(lambda: received == ["转写结果"])
            self.assertEqual(len(asr.files), 1)
            with wave.open(io.BytesIO(asr.files[0].data), "rb") as file:
                self.assertEqual(file.readframes(file.getnframes()), _HOT * 320 + _QUIET * 16 + _HOT * 4 + _QUIET * 31)

    async def test_no_old_audio_during_speech_pending_asr_or_after_asr_result(self):
        for result in ("新输入", "", AsrError("synthetic")):
            with self.subTest(result=type(result).__name__):
                asr, tts, vad, device = _ASR(result, block=True), _TTS(), _Vad(), _Device()
                voice = VoiceChannel(asr, tts, vad, _SETTINGS, asr_timeout=0.5, max_audio_seconds=500)
                received = []
                async with voice.connected(device, _transcripts(received.append)):
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
                    await _until(lambda: voice._pending == 0)
                    self.assertIsNone(voice._asr_task)
                    for old in (idle_late, started_in_asr):
                        self.assertEqual((await voice.deliver(old)).status, DeliveryStatus.ZERO)
                    self.assertEqual(tts.calls, [])
                    self.assertEqual(received, [result] if isinstance(result, str) and result else [])
                    new = asyncio.create_task(voice.deliver(_reply("new", "新回复。")))
                    await _until(lambda: bool(tts.calls))
                    self.assertEqual(device.pull(4), _A * 4)
                    self.assertEqual((await new).status, DeliveryStatus.FULL)

    async def test_idle_transcription_worker_releases_pcm(self):
        class ObservablePCM(bytearray):
            pass

        asr, device = _ASR(), _Device()
        voice = VoiceChannel(asr, _TTS(), _Vad(), _SETTINGS, asr_timeout=0.5, max_audio_seconds=500)
        received = []
        async with voice.connected(device, _transcripts(received.append)):
            # 注入可弱引用的 PCM 来检查空闲 worker 的局部变量是否仍持有它。
            recording = ObservablePCM(_HOT)
            recording_ref = ref(recording)
            voice._enqueue_turn(recording)
            del recording
            await _until(lambda: received == ["转写结果"])
            self.assertIsNone(recording_ref())

    async def test_normal_close_releases_current_recording_and_preroll(self):
        for hot in (False, True):
            with self.subTest(hot=hot):
                asr, vad, device = _ASR(), _Vad(), _Device()
                voice = VoiceChannel(asr, _TTS(), vad, _SETTINGS, asr_timeout=0.5, max_audio_seconds=500)
                async with voice.connected(device, lambda *_: None):
                    device.frame(hot, 3)
                    await _processed(vad, 3)
                    self.assertTrue(voice._record if hot else voice._preroll)
                self.assertIsNone(voice._record)
                self.assertFalse(voice._preroll)
                self.assertEqual(asr.files, [])

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

        voice = VoiceChannel(asr, SlowTTS(), vad, _SETTINGS, asr_timeout=0.5, max_audio_seconds=500)
        async with voice.connected(device, lambda *_: None):
            old = asyncio.create_task(voice.deliver(_reply("old", "甲乙丙丁戊己庚辛。")))
            try:
                await first.wait()
                self.assertEqual(device.pull(_RATE // 2), _A * (_RATE // 2))
                device.frame(True, 3)
                receipt = await asyncio.wait_for(old, 0.3)
                self.assertEqual((receipt.status, receipt.confirmed), (DeliveryStatus.ZERO, ""))
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

    async def test_capture_stall_counts_from_enable_and_ignores_invalid_frames(self):
        asr, vad, device = _ASR(), _Vad(), _Device()
        voice = VoiceChannel(asr, _TTS(), vad, _SETTINGS, asr_timeout=1, max_audio_seconds=500)
        clock = [0.0]
        events = []
        with (patch("agent_companion.speech.voice.monotonic", new=lambda: clock[0]),
              patch("agent_companion.speech.voice._CAPTURE_STALL_SECONDS", .25),
              self.assertLogs("agent_companion.speech.voice", level="ERROR") as log):
            async with voice.connected(device, lambda *event: events.append(event), capture_enabled=False):
                clock[0] = 10.0  # 等待接通的时间不计入首帧宽限。
                device.frame(True, 3)
                await asyncio.sleep(.025)
                self.assertFalse(voice.failed)
                self.assertEqual(vad.count, 0)
                voice.enable_capture()
                clock[0] += .125
                device.frame(False, overflow=True)
                device.capture(memoryview(b""), False)
                await _processed(vad, 1)
                self.assertFalse(voice.failed)
                clock[0] += .125
                device.frame(True, 3)  # 截止后到达，不能抢在监听任务前重置宽限。
                await _until(lambda: device.closed)
        self.assertEqual(events, [(None, False)])
        self.assertFalse(asr.called.is_set())
        self.assertIn("capture PCM stalled", "\n".join(log.output))

    async def test_continuous_real_pcm_renews_capture_deadline_even_in_silence(self):
        vad, device = _Vad(), _Device()
        voice = VoiceChannel(_ASR(), _TTS(), vad, _SETTINGS, asr_timeout=1, max_audio_seconds=500)
        clock = [0.0]
        with (patch("agent_companion.speech.voice.monotonic", new=lambda: clock[0]),
              patch("agent_companion.speech.voice._CAPTURE_STALL_SECONDS", .25)):
            async with voice.connected(device, lambda *_: None):
                for count in range(1, 7):
                    clock[0] += .125
                    device.frame(False)
                    await _processed(vad, count)
                device.check_active()

    async def test_slow_vad_does_not_stall_capture_when_valid_pcm_keeps_arriving(self):
        loop = asyncio.get_running_loop()
        entered, release = asyncio.Event(), Event()

        class SlowVAD(_Vad):
            def probability(self, pcm):
                if not self.count:
                    loop.call_soon_threadsafe(entered.set)
                    release.wait()
                return super().probability(pcm)

        vad, device = SlowVAD(), _Device()
        voice = VoiceChannel(_ASR(), _TTS(), vad, _SETTINGS, asr_timeout=1, max_audio_seconds=500)
        clock = [0.0]
        with (patch("agent_companion.speech.voice.monotonic", new=lambda: clock[0]),
              patch("agent_companion.speech.voice._CAPTURE_STALL_SECONDS", .0625)):
            async with voice.connected(device, lambda *_: None):
                try:
                    device.frame(False)
                    await asyncio.wait_for(entered.wait(), 1)
                    for count in range(6):
                        clock[0] += .03125
                        if count == 0:
                            device.frame(False, 40)  # 先填满队列，覆盖旧帧。
                        else:
                            with voice._capture._lock:
                                device.frame(False)  # 实际到达，但消费者侧无法入队。
                        await asyncio.sleep(.04)
                        device.check_active()
                    self.assertEqual(vad.count, 0)
                finally:
                    release.set()
                await _until(lambda: vad.count > 1)

    async def test_cancelled_image_reads_at_capacity_do_not_delay_actual_vad(self):
        loop = asyncio.get_running_loop()
        workers = images._IMAGE_WORKERS
        # 若图片误送默认池，这些未结束的读操作会占用 VAD 使用的所有线程。
        loop.set_default_executor(ThreadPoolExecutor(max_workers=workers))
        workspace = image_workspace(self)
        pictures = [images.save_image(workspace, PNG_BYTES) for _ in range(workers + 2)]
        blocked_names = {images.image_filename(picture.url) for picture in pictures[:workers]}
        skipped = images.image_filename(pictures[-2].url)
        started, submitted = set(), []
        unexpected_read, unexpected_write = Event(), Event()
        release = Event()
        actual_read = images._read_image
        actual_submit = images._IMAGE_EXECUTOR.submit

        def slow_read(workspace, name, remaining):
            if name == skipped:
                unexpected_read.set()
            if name not in blocked_names:
                return actual_read(workspace, name, remaining)
            loop.call_soon_threadsafe(started.add, name)
            if not release.wait(5):
                raise AssertionError("image worker was not released")
            return actual_read(workspace, name, remaining)

        def track_submit(operation):
            job = actual_submit(operation)
            submitted.append(job)
            return job

        def request(picture):
            return asyncio.create_task(images.prepare_request_images([Message("user", (picture,))], workspace))

        vad, device = _Vad(), _Device()
        voice = VoiceChannel(_ASR(), _TTS(), vad, _SETTINGS, asr_timeout=1, max_audio_seconds=500)
        tasks = []
        with (patch.object(images, "_read_image", side_effect=slow_read),
              patch.object(images._IMAGE_EXECUTOR, "submit", side_effect=track_submit)):
            try:
                tasks.extend(request(picture) for picture in pictures[:workers])
                await _until(lambda: started == blocked_names, timeout=2)
                for task in tasks:
                    task.cancel()
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 1)

                queued = [request(pictures[-2]), asyncio.create_task(
                    images.run_image_io(unexpected_write.set, settle_write=True))]
                tasks.extend(queued)
                await asyncio.sleep(0)  # 两种任务都已开始等图片额度。
                for task in queued:
                    task.cancel()
                await asyncio.wait_for(asyncio.gather(*queued, return_exceptions=True), 1)
                live = request(pictures[-1])
                tasks.append(live)
                async with voice.connected(device, lambda *_: None):
                    device.frame(False)
                    await _processed(vad, 1)  # 真实 _listen → 默认池 → VAD，在图片释放前应完成。
                    self.assertFalse(any(job.done() for job in submitted))
                    self.assertEqual(len(submitted), workers)
                    self.assertFalse(live.done())
                release.set()
                await asyncio.wait_for(live, 1)
                self.assertFalse(unexpected_read.is_set())
                self.assertFalse(unexpected_write.is_set())
                self.assertEqual(len(submitted), workers + 1)
            finally:
                release.set()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.gather(*(asyncio.wrap_future(job) for job in submitted), return_exceptions=True)

    async def test_device_fault_returns_prefix_and_never_falls_back_to_text(self):
        asr, vad, device = _ASR(), _Vad(), _Device()
        first = asyncio.Event()

        class TTS:
            async def synthesize(self, text, *, sample_rate):
                yield _A * _RATE
                first.set()
                await asyncio.Event().wait()

        voice = VoiceChannel(asr, TTS(), vad, _SETTINGS, asr_timeout=0.5, max_audio_seconds=500)
        with patch("agent_companion.speech.voice._DEVICE_CHECK_SECONDS", 0.02):
            async with voice.connected(device, lambda *_: None):
                old = asyncio.create_task(voice.deliver(_reply("fault", "甲乙丙丁戊己庚辛。")))
                await _until(first.is_set)
                self.assertEqual(device.pull(_RATE // 2), _A * (_RATE // 2))
                device.fault = True
                receipt = await asyncio.wait_for(old, 0.3)
                self.assertEqual((receipt.status, receipt.confirmed), (DeliveryStatus.FAILED, ""))
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

        voice = VoiceChannel(SlowASR(), _TTS(), vad, _SETTINGS, asr_timeout=1, max_audio_seconds=500)
        submitted = []
        with patch("agent_companion.speech.voice._DEVICE_CHECK_SECONDS", 0.02):
            async with voice.connected(device, _transcripts(submitted.append)):
                try:
                    device.frame(True, 3)
                    device.frame(False, 2)
                    await _until(started.is_set)
                    device.frame(True, 3)
                    device.frame(False, 2)
                    await _until(lambda: len(voice._turns) == 1)
                    self.assertEqual(len(requests), 1)
                    in_flight = voice._asr_task
                    self.assertIsNotNone(in_flight)

                    device.fault = True
                    await _until(lambda: device.closed and cancelled.is_set())
                    self.assertFalse(finished.is_set())
                    self.assertIs(voice._asr_task, in_flight)
                    self.assertFalse(voice._turns)
                    self.assertEqual((await voice.deliver(_reply("disabled", "停用。"))).status,
                                     DeliveryStatus.FAILED)

                    device.frame(True, 3)  # 模拟停止前已经排定的设备回调
                    device.frame(False, 2)
                    release.set()
                    await _until(finished.is_set)
                    await _until(lambda: voice._asr_worker is not None and voice._asr_worker.done())
                    await _until(lambda: voice._asr_task is None)
                    self.assertEqual(voice._pending, 0)
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(submitted, [])
                finally:
                    release.set()

    async def test_audio_limit_reports_transcript_with_idle_only_after_trusted_silence(self):
        asr, tts, vad, device = _ASR(), _TTS(size=10), _Vad(), _Device()
        samples = 6 * INPUT_FRAMES + 37
        voice = VoiceChannel(asr, tts, vad, _SETTINGS, asr_timeout=0.5, max_audio_seconds=samples / _RATE)
        events, prompts = [], []

        class Model:
            async def stream(self, messages, tools, *, thinking, instructions=None):
                prompts.append(messages[-1])
                yield TextDelta("正在回复。")

        with closing(History(":memory:")) as history:
            runtime = Runtime("main", history, Model(), voice)

            def on_input(text, active):
                events.append((text, active))
                if text:
                    runtime.submit_messages([MessageDelivery(Message("user", text), "trigger")])

            try:
                async with voice.connected(device, on_input):
                    device.frame(False, 2)  # 预录计入上限
                    device.frame(True, 3)
                    device.frame(False)  # 话轮内短停顿也计入
                    await _processed(vad, 6)
                    self.assertEqual(asr.files, [])
                    device.frame(True)
                    await asr.finished.wait()
                    await _until(lambda: voice._pending == 0)
                    self.assertEqual(events, [(None, True)])
                    self.assertEqual(prompts, [])
                    with wave.open(io.BytesIO(asr.files[0].data), "rb") as wav:
                        self.assertEqual(wav.getnframes(), samples)
                        self.assertEqual(wav.readframes(samples), _QUIET * 2 + _HOT * 3 + _QUIET + _HOT[:74])
                    # 即使绕过调度器，语音通道也不能在仍有尾音时播报。
                    receipt = await voice.deliver(_reply("tail", "不应播放。"))
                    self.assertEqual(receipt.status, DeliveryStatus.ZERO)

                    device.frame(True, 4)
                    device.frame(False)
                    device.frame(True, 3)
                    await _processed(vad, 15)
                    device.frame(False)
                    device.frame(False, overflow=True)  # 缺口两侧静默不能相加
                    device.frame(True, 3)
                    await _processed(vad, 20)
                    device.frame(False, overflow=True)  # 溢出的静默帧本身也不算
                    device.frame(False)
                    device.frame(True, 3)
                    await _processed(vad, 25)
                    self.assertGreater(vad.resets, 0)
                    self.assertEqual(len(asr.files), 1)
                    self.assertEqual(events, [(None, True)])
                    self.assertEqual(tts.calls, [])
                    self.assertEqual(device.pull(2), bytes(4))

                    device.frame(False, 2)
                    await _until(lambda: len(tts.calls) == 1)
                    self.assertEqual(events, [(None, True), ("转写结果", False)])
                    self.assertEqual(prompts, [Message("user", "转写结果")])
                    self.assertEqual(device.pull(10), _A * 10)
                    await asyncio.wait_for(runtime.wait_idle(), 1)
                    device.frame(True, 3)
                    device.frame(False, 2)
                    await _until(lambda: len(tts.calls) == 2)
                    self.assertEqual(events, [(None, True), ("转写结果", False)] * 2)
                    self.assertEqual(len(asr.files), 2)
                    with wave.open(io.BytesIO(asr.files[1].data), "rb") as wav:
                        self.assertEqual(wav.readframes(wav.getnframes()), _HOT * 3 + _QUIET)
                    self.assertEqual(device.pull(10), _A * 10)
                    await asyncio.wait_for(runtime.wait_idle(), 1)
            finally:
                await runtime.close()

    async def test_capped_turn_dropped_at_asr_limit_does_not_hold_earlier_transcripts(self):
        class ASR(_ASR):
            async def transcribe(self, file):
                index = len(self.files) + 1
                return f"{await super().transcribe(file)}{index}"

        asr, vad, device = ASR(block=True), _Vad(), _Device()
        voice = VoiceChannel(asr, _TTS(), vad, _SETTINGS, asr_timeout=0.5,
                             max_audio_seconds=5 * INPUT_FRAMES / _RATE)
        events = []
        async with voice.connected(device, lambda *event: events.append(event)):
            device.frame(True, 3)
            device.frame(False, 2)
            await asyncio.wait_for(asr.called.wait(), 1)
            device.frame(True, 3)
            device.frame(False, 2)
            await _processed(vad, 10)
            self.assertEqual(voice._pending, 2)
            device.frame(True, 5)
            await _processed(vad, 15)
            asr.release.set()
            await _until(lambda: voice._pending == 0)
            self.assertEqual(events, [(None, True)] * 3 + [("转写结果1", None), ("转写结果2", None)])
            self.assertEqual(len(asr.files), 2)
            device.frame(False, 2)
            await _until(lambda: events[-1] == (None, False))
            self.assertEqual(len(events), 6)

    async def test_quiet_frames_before_cut_count_towards_rearming(self):
        asr, vad, device = _ASR(), _Vad(), _Device()
        seconds = (3 * INPUT_FRAMES + 11) / _RATE
        voice = VoiceChannel(asr, _TTS(), vad, _SETTINGS, asr_timeout=0.5, max_audio_seconds=seconds)
        async with voice.connected(device, lambda *_: None):
            for count in (1, 2):
                device.frame(True, 3)
                device.frame(False)  # 上限在第一帧静默期间到达
                await _until(lambda: len(asr.files) == count)
                device.frame(False)  # 只需再等一帧静默即可接纳下一轮

    async def test_tiny_limit_preserves_speech_and_large_limit_respects_wav_bytes(self):
        for seconds, expected_samples in ((1 / _RATE, 1), (10**400, 11), (1e308, 11)):
            with (self.subTest(expected_samples=expected_samples, seconds_type=type(seconds)),
                  patch("agent_companion.speech.voice.MAX_AUDIO_BYTES", 44 + 2 * 11 + 1),
                  patch("agent_companion.speech.asr.MAX_AUDIO_BYTES", 44 + 2 * 11 + 1)):
                asr, vad, device = _ASR(), _Vad(), _Device()
                voice = VoiceChannel(asr, _TTS(), vad, _SETTINGS, asr_timeout=0.5, max_audio_seconds=seconds)
                async with voice.connected(device, lambda *_: None):
                    device.frame(False, 3)
                    device.frame(True, 3)
                    await _until(lambda: len(asr.files) == 1)
                    data = asr.files[0].data
                    self.assertEqual(len(data), 44 + 2 * expected_samples)
                    with wave.open(io.BytesIO(data), "rb") as wav:
                        self.assertEqual(wav.readframes(wav.getnframes()), _HOT[:2 * expected_samples])
        with self.assertRaisesRegex(VoiceSetupError, "voice_asr_max_audio_seconds_below_one_sample"):
            VoiceChannel(_ASR(), _TTS(), _Vad(), _SETTINGS, asr_timeout=0.5, max_audio_seconds=0.5 / _RATE)

    async def test_capture_gap_debug_counts_loss_sources_even_when_idle(self):
        asr, vad, device = _ASR(), _Vad(), _Device()
        with (patch("agent_companion.speech.voice._MAX_INPUT_FRAMES", 4),
              self.assertLogs("agent_companion.speech.voice", level="DEBUG") as log):
            voice = VoiceChannel(asr, _TTS(), vad, _SETTINGS, asr_timeout=0.5, max_audio_seconds=500)
            async with voice.connected(device, lambda *_: None, capture_enabled=False):
                device.frame(False, overflow=True)  # 接通前不计入本通累计值。
                voice.enable_capture()
                device.frame(False, overflow=True)
                device.capture(memoryview(b""), False)
                with voice._capture._lock:
                    device.frame(False)
                device.frame(False, 4)  # 队列覆盖最早入队的帧。
                await _processed(vad, 4)
        details = [record.getMessage() for record in log.records if record.levelname == "DEBUG"]
        self.assertEqual(details, ["voice capture gap details; dropped_turn=False driver_overflow_events_total=1"
                                   " queue_overwrites_total=1 lock_misses_total=1 invalid_frames_total=1"])
        self.assertFalse(asr.called.is_set())

    async def test_capture_gap_debug_marks_discarded_turn_without_repeating_for_consecutive_gaps(self):
        asr, vad, device = _ASR(), _Vad(), _Device()
        voice = VoiceChannel(asr, _TTS(), vad, _SETTINGS, asr_timeout=0.5, max_audio_seconds=500)
        events = []
        with self.assertLogs("agent_companion.speech.voice", level="DEBUG") as log:
            async with voice.connected(device, lambda *event: events.append(event)):
                device.frame(True, 3)
                await _until(lambda: events == [(None, True)])
                device.frame(True, 3, overflow=True)
                await _processed(vad, 6)
        details = [record.getMessage() for record in log.records
                   if record.getMessage().startswith("voice capture gap details;")]
        self.assertEqual(len(details), 1)
        self.assertIn("dropped_turn=True", details[0])
        self.assertIn("driver_overflow_events_total=3", details[0])
        self.assertFalse(asr.called.is_set())

    async def test_gap_and_asr_timeout_drop_only_affected_turns(self):
        asr, vad, device = _ASR(block=True), _Vad(), _Device()
        voice = VoiceChannel(asr, _TTS(), vad, _SETTINGS, asr_timeout=0.03, max_audio_seconds=500)
        received = []
        async with voice.connected(device, _transcripts(received.append)):
            device.frame(True, 3)
            device.frame(True, 4, overflow=True)
            device.frame(False, 3)
            await _processed(vad, 10)
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

    async def test_gap_frame_does_not_end_active_turn_as_silence(self):
        asr, vad, device = _ASR(), _Vad(), _Device()
        settings = replace(_SETTINGS, speech_frames=1, silence_seconds=0.02)
        voice = VoiceChannel(asr, _TTS(), vad, settings, asr_timeout=0.5, max_audio_seconds=500)
        events = []
        async with voice.connected(device, lambda *event: events.append(event)):
            device.frame(True)
            await _processed(vad, 1)
            device.frame(False, overflow=True)
            await _processed(vad, 2)
            device.frame(True)
            await _processed(vad, 3)
            self.assertEqual(events, [(None, True)])
            self.assertFalse(asr.called.is_set())

            device.frame(False)
            await _until(lambda: events == [(None, True), (None, False)])
            self.assertFalse(asr.called.is_set())  # 不拼接缺口两边的人声。

            device.frame(True)
            device.frame(False)
            await _until(lambda: len(events) == 4)
            self.assertEqual(events[2:], [(None, True), ("转写结果", False)])
            self.assertEqual(len(asr.files), 1)

    async def test_capped_file_empty_waits_for_quiet_and_gap_revokes_wake(self):
        for outcome in ("empty", "gap", "timeout", "wrong_phase"):
            with self.subTest(outcome=outcome):
                phase = "receive" if outcome == "wrong_phase" else None
                asr = _ASR(AsrError("asr_empty_transcript", phase=phase), block=outcome == "timeout")
                device = _Device()
                voice = VoiceChannel(asr, _TTS(), _Vad(), _SETTINGS, asr_timeout=0.05, max_audio_seconds=0.1)
                events = []
                async with voice.connected(device, lambda *event: events.append(event), capture_enabled=False):
                    playback = asyncio.create_task(voice.deliver(_reply("old", "正在播报。")))
                    await _until(lambda: voice._output._active is not None and voice._output._active.draining)
                    self.assertEqual(device.pull(1), _A)
                    for _ in range(4):
                        voice._process(False, True, _HOT)
                    await playback
                    await _until(lambda: len(asr.files) == 1 and voice._pending == 0)
                    self.assertEqual(events, [(None, True)])
                    if outcome == "gap":
                        voice._process(True, False, _QUIET)
                    for _ in range(voice._end_frames):
                        voice._process(False, False, _QUIET)
                    self.assertEqual(events, [(None, True), ("" if outcome == "empty" else None, False)])

    async def test_empty_failed_and_timed_out_asr_report_idle(self):
        for text, blocked in (("  ", False), (RuntimeError("synthetic"), False), ("迟到结果", True)):
            with self.subTest(text=text, timeout=blocked):
                asr, vad, device = _ASR(text, block=blocked), _Vad(), _Device()
                voice = VoiceChannel(asr, _TTS(), vad, _SETTINGS, asr_timeout=0.03, max_audio_seconds=500)
                events = []
                async with voice.connected(device, lambda *event: events.append(event)):
                    device.frame(True, 3)
                    await _processed(vad, 3)
                    self.assertEqual(events, [(None, True)])
                    device.frame(False, 2)
                    await asr.called.wait()
                    await _until(lambda: len(events) == 2)
                    self.assertEqual(events, [(None, True), (None, False)])

    async def test_runtime_tool_continues_and_asr_enters_same_session(self):
        asr, tts, vad, device = _ASR(block=True), _TTS(), _Vad(), _Device()
        tool_started, release_tool = asyncio.Event(), asyncio.Event()
        tool_done = asyncio.Event()
        calls = []

        class Model:
            async def stream(self, messages, tools, *, thinking, instructions=None):
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
        voice = VoiceChannel(asr, tts, vad, _SETTINGS, asr_timeout=0.5, max_audio_seconds=500)
        with closing(History(":memory:")) as history:
            runtime = Runtime("main", history, Model(), voice, tools=tools)
            try:
                def submit(text):
                    runtime.submit_messages((MessageDelivery(Message("user", text), "trigger"),))

                async with voice.connected(device, _transcripts(submit)):
                    runtime.submit_messages([MessageDelivery(Message("user", "运行工具"), "trigger")])
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
                    self.assertEqual([call.result for call in tool_calls(calls[-1])], ["工具结果"])
                    self.assertEqual(calls[-1][-1].content, "转写结果")
            finally:
                await runtime.close()

    async def test_interrupted_playback_cancels_model_before_it_starts_new_tool(self):
        asr, device, vad = _ASR(block=True), _Device(), _Vad()
        model_cancelled, ran_tool, spoke = (asyncio.Event() for _ in range(3))
        prompts = []

        class TTS:
            async def synthesize(self, text, *, sample_rate):
                if text == "已交付。":
                    yield _B * 4
                    return
                yield _A * _RATE
                spoke.set()
                await asyncio.Event().wait()

        class Model:
            async def stream(self, messages, tools, *, thinking, instructions=None):
                prompts.append(messages)
                if len(prompts) == 1:
                    yield TextDelta("已交付。" + "甲乙丙丁戊己庚辛壬癸" * 3 + "。")
                    try:
                        await asyncio.Future()
                    except asyncio.CancelledError:
                        model_cancelled.set()
                    # 即使适配器尝试在取消后继续输出，Runtime 也不会执行这条调用。
                    yield ToolReady(ToolCall("a", "wait", "{}"))
                else:
                    yield TextDelta("仍在说话，不能播报。")

        async def tool(args, context):
            ran_tool.set()
            return "工具结果"

        tools = Tools((Tool(ToolSpec("wait", "测试工具", {"type": "object"}), tool),))
        voice = VoiceChannel(asr, TTS(), vad, _SETTINGS, asr_timeout=0.5, max_audio_seconds=500)
        with closing(History(":memory:")) as history:
            runtime = Runtime("main", history, Model(), voice, tools=tools)
            try:
                async with voice.connected(device, lambda *_: None):
                    runtime.submit_messages([MessageDelivery(Message("user", "测试请求"), "trigger")])
                    await _until(spoke.is_set)
                    self.assertEqual(device.pull(4 + _RATE // 2), _B * 4 + _A * (_RATE // 2))
                    device.frame(True, 3)
                    await _processed(vad, 3)
                    self.assertEqual(device.pull(4), bytes(8))
                    await asyncio.wait_for(runtime.wait_idle(), 1)
                    await asyncio.wait_for(model_cancelled.wait(), 1)
                    self.assertFalse(ran_tool.is_set())
                    self.assertEqual(len(prompts), 1)
                    self.assertEqual([msg.message.content for msg in history.read_after("main")],
                                     ["测试请求", "已交付。"])
                    self.assertEqual([r.delivery_status for r in history.outcomes("main")],
                                     [DeliveryStatus.PARTIAL])
            finally:
                await runtime.close()
