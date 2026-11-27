import asyncio
from contextlib import asynccontextmanager, contextmanager
from dataclasses import replace
import io
import unittest
from unittest.mock import patch
import wave

from agent_companion.protocol import DeliveryStatus, Message, MessageDelivery
from agent_companion.session_scheduler import SessionScheduler
from agent_companion.speech.asr import AsrError
from agent_companion.speech.vad import INPUT_FRAMES, INPUT_RATE, SAMPLE_BYTES, silence_frames
from agent_companion.speech.voice import VoiceChannel

from test_voice import _Device, _SETTINGS, _TTS, _Vad, _reply, _until


_SETTINGS = replace(_SETTINGS, segment_pause_seconds=0.4, silence_seconds=1)
_SHORT = silence_frames(0.4)
_END = silence_frames(1)
_LONG = 320  # 10.24 秒，不含后续停顿。


class _ASR:
    def __init__(self, responses):
        self.responses = responses
        self.pcm = []

    async def transcribe(self, audio):
        with wave.open(io.BytesIO(audio.data), "rb") as wav:
            self.pcm.append(wav.readframes(wav.getnframes()))
        response = self.responses[len(self.pcm) - 1]
        if isinstance(response, Exception):
            raise response
        return await response() if callable(response) else response


class _Frames:
    def __init__(self, voice):
        self.voice = voice
        self.index = 0

    def push(self, hot, count=1, *, gap=False):
        """直接喂入 VAD 判定；用不同的样本值标记每帧，逐字节检查切口。"""
        data = []
        for _ in range(count):
            self.index += 1
            pcm = self.index.to_bytes(SAMPLE_BYTES, "little") * INPUT_FRAMES
            self.voice._process(gap, hot, pcm)
            data.append(pcm)
        return b"".join(data)

    def short_pause(self):
        return self.push(False, _SHORT)

    def end_turn(self):
        # 末帧只用于确认静默，与生产代码旧录音边界一致，不落到 WAV 中。
        pcm = self.push(False, _END - 1)
        self.push(False)
        return pcm


@asynccontextmanager
async def _channel(*responses, timeout=0.5, max_seconds=500, settings=_SETTINGS):
    asr, device = _ASR(responses), _Device()
    voice = VoiceChannel(asr, _TTS(), _Vad(), settings, asr_timeout=timeout, max_audio_seconds=max_seconds)
    events = []
    async with voice.connected(device, lambda *event: events.append(event)):
        yield voice, _Frames(voice), asr, events


@contextmanager
def _loop_time_jump():
    loop = asyncio.get_running_loop()
    original = loop.time
    offset = 0.0

    def jump():
        nonlocal offset
        offset += 0.3

    # 用单调时钟前跳模拟循环阻塞，不依赖真实 sleep 或竞速。
    # 模拟的时钟跳变不是真慢回调，避免 asyncio 打印任务结果。
    with (patch.object(loop, "time", new=lambda: original() + offset),
          patch.object(loop, "slow_callback_duration", new=float("inf"))):
        yield jump


class SegmentedVoiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_latency_logs_only_endpoint_and_full_transcript_if_early_asr_already_ready(self):
        with (self.assertLogs("agent_companion.speech.voice", "DEBUG") as log,
              patch("agent_companion.speech.voice.monotonic", return_value=1000) as clock):
            async with _channel("合成文本") as (voice, frames, asr, events):
                frames.push(True, _LONG)
                frames.short_pause()
                await _until(lambda: len(asr.pcm) == 1 and voice._pending == 0)
                self.assertEqual(log.records, [])
                clock.return_value = 1001
                frames.push(False, _END - _SHORT)
                await _until(lambda: not voice._input_active)
                self.assertEqual(events[-1], ("合成文本", False))
        endpoint, ready = [record.getMessage() for record in log.records]
        event, fields = endpoint.split("; ", 1)
        self.assertEqual(event, "voice turn endpoint detected")
        self.assertIn(" at_mono_ms=1001000", fields)
        self.assertEqual(ready, f"voice turn transcript ready; {fields} endpoint_to_transcript_ms=0")

    async def test_latency_includes_wait_for_early_failure_and_final_retry(self):
        release = asyncio.Event()

        async def fail_late():
            await release.wait()
            raise AsrError("合成错误")

        with (self.assertLogs("agent_companion.speech.voice", "DEBUG") as log,
              patch("agent_companion.speech.voice.monotonic", return_value=1000) as clock):
            async with _channel(fail_late, "补交成功") as (voice, frames, asr, events):
                frames.push(True, _LONG)
                frames.short_pause()
                await _until(lambda: len(asr.pcm) == 1)
                clock.return_value = 1001
                frames.push(False, _END - _SHORT)
                self.assertEqual(len(log.records), 1)
                clock.return_value = 1001.25
                release.set()
                await _until(lambda: not voice._input_active)
                self.assertEqual(events[-1], ("补交成功", False))
        endpoint, warning, ready = log.records
        turn = endpoint.getMessage().split("turn=", 1)[1].split()[0]
        self.assertEqual(warning.levelname, "WARNING")
        self.assertEqual(warning.getMessage(), f"voice segment transcription failed; phase=early turn={turn}")
        self.assertEqual(ready.getMessage(), f"voice turn transcript ready; turn={turn} "
                                             "at_mono_ms=1001250 endpoint_to_transcript_ms=250")

    async def test_short_turn_is_sent_once_at_final_silence(self):
        async with _channel("短句") as (voice, frames, asr, events):
            recorded = frames.push(False, 3) + frames.push(True, 3) + frames.short_pause()
            await asyncio.sleep(0)
            self.assertEqual(asr.pcm, [])
            recorded += frames.push(True, 2) + frames.end_turn()
            await _until(lambda: not voice._input_active)
            self.assertEqual(asr.pcm, [recorded])
            self.assertEqual(events, [(None, True), ("短句", False)])

    async def test_pause_and_end_use_strict_frame_boundaries_only_when_segmenting(self):
        cases = (
            (0.288, 0.28800000000000003, 9, 10),
            (0.28800000000000003, 0.352, 10, 11),
            (None, 0.28800000000000003, None, 9),
        )
        for pause, silence, short_frames, end_frames in cases:
            with self.subTest(pause=pause, silence=silence):
                settings = replace(_SETTINGS, segment_pause_seconds=pause, silence_seconds=silence)
                async with _channel("结果", settings=settings) as (voice, frames, asr, events):
                    recorded = frames.push(True, _LONG)
                    first = short_frames if short_frames is not None else end_frames
                    recorded += frames.push(False, first - 1)
                    self.assertEqual(voice._pending, 0)
                    last = frames.push(False)
                    if short_frames is not None:
                        recorded += last
                        await _until(lambda: bool(asr.pcm) and voice._pending == 0)
                        self.assertEqual(events, [(None, True)])
                        frames.push(False, end_frames - short_frames - 1)
                        self.assertTrue(voice._input_active)
                        frames.push(False)
                    await _until(lambda: not voice._input_active)
                    self.assertEqual(asr.pcm, [recorded])
                    self.assertEqual(events, [(None, True), ("结果", False)])

    async def test_passes_ten_seconds_during_same_pause_but_only_checks_once(self):
        async with _channel("前段") as (voice, frames, asr, events):
            # 299 + 13 帧 = 9.984 秒；随后多录一帧会过 10 秒，但不是新的切点。
            recorded = frames.push(True, 299) + frames.short_pause() + frames.push(False)
            self.assertEqual(voice._pending, 0)
            recorded += frames.push(True) + frames.push(False, _SHORT - 1)
            self.assertEqual(voice._pending, 0)
            recorded += frames.push(False)
            await _until(lambda: bool(asr.pcm) and voice._pending == 0)
            self.assertEqual(asr.pcm, [recorded])
            self.assertEqual(events, [(None, True)])
            frames.push(False, _END - _SHORT)
            await _until(lambda: not voice._input_active)
            self.assertEqual(asr.pcm, [recorded])
            self.assertEqual(events, [(None, True), ("前段", False)])

    async def test_cut_continue_and_cut_again_preserves_pcm_and_only_hands_off_once(self):
        async with _channel("第一段", "second part", "最后") as (voice, frames, asr, events):
            prefix = frames.push(True, _LONG) + frames.short_pause()
            await _until(lambda: len(asr.pcm) == 1 and voice._pending == 0)
            middle = frames.push(False, 3) + frames.push(True, _LONG) + frames.short_pause()
            await _until(lambda: len(asr.pcm) == 2 and voice._pending == 0)
            self.assertEqual(events, [(None, True)])
            self.assertEqual((await voice.deliver(_reply("waiting", "不该播放"))).status, DeliveryStatus.ZERO)
            suffix = frames.push(False, 2) + frames.push(True, 3) + frames.end_turn()
            await _until(lambda: not voice._input_active)
            self.assertEqual(asr.pcm, [prefix, middle, suffix])
            self.assertEqual(b"".join(asr.pcm), prefix + middle + suffix)
            self.assertEqual(events, [(None, True), ("第一段\nsecond part\n最后", False)])

    async def test_busy_at_next_pause_keeps_recording_even_if_asr_frees_in_same_pause(self):
        release = asyncio.Event()

        async def early():
            await release.wait()
            return "开头"

        async with _channel(early, "尾部") as (voice, frames, asr, events):
            prefix = frames.push(True, _LONG) + frames.short_pause()
            await _until(lambda: len(asr.pcm) == 1)
            tail = frames.push(True, _LONG) + frames.short_pause()  # 忙；不另起片段。
            release.set()
            await _until(lambda: voice._pending == 0)
            tail += frames.push(False, _END - _SHORT - 1)
            self.assertEqual(voice._pending, 0)
            self.assertEqual(events, [(None, True)])
            frames.push(False)
            await _until(lambda: not voice._input_active)
            self.assertEqual(asr.pcm, [prefix, tail])
            self.assertEqual(events[-1], ("开头\n尾部", False))

    async def test_failed_middle_attempt_retries_at_next_short_pause_without_resending_success(self):
        async with _channel("第一段", AsrError("synthetic"), "恢复", "尾段") as (voice, frames, asr, events):
            prefix = frames.push(True, _LONG) + frames.short_pause()
            await _until(lambda: len(asr.pcm) == 1 and voice._pending == 0)
            failed = frames.push(True, _LONG) + frames.short_pause()
            await _until(lambda: len(asr.pcm) == 2 and voice._pending == 0)
            rest = frames.push(False, 5)
            self.assertEqual(voice._pending, 0)
            rest += frames.push(True, 3) + frames.short_pause()
            await _until(lambda: len(asr.pcm) == 3 and voice._pending == 0)
            self.assertEqual(events, [(None, True)])
            tail = frames.push(True, 3) + frames.end_turn()
            await _until(lambda: not voice._input_active)
            self.assertEqual(asr.pcm, [prefix, failed, failed + rest, tail])
            self.assertEqual(events[-1], ("第一段\n恢复\n尾段", False))

    async def test_late_early_failure_retried_after_final_silence_without_new_voice(self):
        release = asyncio.Event()

        async def early():
            await release.wait()
            raise AsrError("synthetic")

        async with _channel(early, "完整转写") as (voice, frames, asr, events):
            prefix = frames.push(True, _LONG) + frames.short_pause()
            await _until(lambda: len(asr.pcm) == 1)
            extra = frames.push(False, _END - _SHORT - 1)
            frames.push(False)
            self.assertEqual(voice._pending, 1)
            self.assertEqual(events, [(None, True)])
            self.assertEqual((await voice.deliver(_reply("blocked", "不要抢答"))).status, DeliveryStatus.ZERO)
            release.set()
            await _until(lambda: not voice._input_active)
            self.assertEqual(asr.pcm, [prefix, prefix + extra])
            self.assertEqual(events[-1], ("完整转写", False))

    async def test_final_empty_wakes_only_if_no_segment_already_has_text(self):
        empty = AsrError("asr_empty_transcript")
        for earlier in (None, "已识别的前段"):
            with self.subTest(earlier=earlier):
                responses = (empty,) if earlier is None else (earlier, empty)
                async with _channel(*responses) as (voice, frames, asr, events):
                    playback = asyncio.create_task(voice.deliver(_reply("old", "仍在播报。")))
                    await _until(lambda: voice._output._active is not None and voice._output._active.draining)
                    pcm = bytearray(2)
                    voice._output.render(memoryview(pcm))
                    self.assertTrue(any(pcm))
                    if earlier is not None:
                        frames.push(True, _LONG)
                        frames.short_pause()
                        await _until(lambda: len(asr.pcm) == 1 and voice._pending == 0)
                    frames.push(True, 3)
                    frames.end_turn()
                    await playback
                    await _until(lambda: not voice._input_active)
                    self.assertEqual(events, [(None, True), (earlier or "", False)])

    async def test_final_failure_supresses_partial_text_and_next_turn_is_unaffected(self):
        async with _channel("不要交付", AsrError("synthetic"), "下一轮") as (voice, frames, asr, events):
            prefix = frames.push(True, _LONG) + frames.short_pause()
            await _until(lambda: len(asr.pcm) == 1 and voice._pending == 0)
            tail = frames.push(True, 3) + frames.end_turn()
            await _until(lambda: not voice._input_active)
            self.assertEqual(events, [(None, True), (None, False)])
            following = frames.push(True, 3) + frames.end_turn()
            await _until(lambda: not voice._input_active)
            self.assertEqual(asr.pcm, [prefix, tail, following])
            self.assertEqual(events[-2:], [(None, True), ("下一轮", False)])

    async def test_pending_early_and_its_tail_use_one_slot_and_keep_following_turn_order(self):
        release = asyncio.Event()

        async def early():
            await release.wait()
            return "首段"

        async with _channel(early, "尾段", "下一轮") as (voice, frames, asr, events):
            prefix = frames.push(True, _LONG) + frames.short_pause()
            await _until(lambda: len(asr.pcm) == 1)
            tail = frames.push(True, 3) + frames.end_turn()
            self.assertEqual(voice._pending, 1)
            following = frames.push(True, _LONG) + frames.short_pause() + frames.push(True, 3) + frames.end_turn()
            self.assertEqual(voice._pending, 2)
            # 真的第三轮积压仍按旧策略丢弃，不会挤掉第一轮的尾段。
            with self.assertLogs("agent_companion.speech.voice", "WARNING") as log:
                frames.push(True, 3)
                frames.end_turn()
            self.assertIn("asr backlog", log.output[0])
            self.assertEqual(voice._pending, 2)
            release.set()
            await _until(lambda: not voice._input_active)
            self.assertEqual(asr.pcm, [prefix, tail, following])
            self.assertEqual(events, [(None, True), (None, True), (None, True),
                                      ("首段\n尾段", None), ("下一轮", False)])

    async def test_cap_below_ten_seconds_is_only_transcribed_at_long_silence(self):
        samples = 6 * INPUT_FRAMES + 37
        async with _channel("前缀", max_seconds=samples / INPUT_RATE) as (voice, frames, asr, events):
            captured = frames.push(True, 9)[:samples * SAMPLE_BYTES]
            self.assertTrue(voice._ignoring_tail)
            self.assertEqual(voice._pending, 0)
            frames.short_pause()
            self.assertEqual(voice._pending, 0)
            self.assertEqual(asr.pcm, [])
            frames.push(False, _END - _SHORT)
            await _until(lambda: not voice._input_active)
            self.assertEqual(asr.pcm, [captured])
            self.assertEqual(events, [(None, True), ("前缀", False)])

    async def test_cap_over_ten_seconds_waits_for_short_silence_and_resets_at_gap(self):
        samples = _LONG * INPUT_FRAMES + 37
        async with _channel("前缀", max_seconds=samples / INPUT_RATE) as (voice, frames, asr, events):
            captured = frames.push(True, _LONG + 4)[:samples * SAMPLE_BYTES]
            self.assertTrue(voice._ignoring_tail)
            self.assertEqual(voice._pending, 0)
            frames.push(False, _SHORT - 1)
            frames.push(False, gap=True)
            frames.push(False, _SHORT - 1)
            self.assertEqual(voice._pending, 0)
            self.assertEqual(asr.pcm, [])
            frames.push(False)
            await _until(lambda: bool(asr.pcm) and voice._pending == 0)
            self.assertEqual(asr.pcm, [captured])
            self.assertEqual(events, [(None, True)])
            frames.push(False, _END - _SHORT)
            await _until(lambda: not voice._input_active)
            self.assertEqual(asr.pcm, [captured])
            self.assertEqual(events[-1], ("前缀", False))

    async def test_legacy_cap_still_starts_asr_without_silence(self):
        settings = replace(_SETTINGS, segment_pause_seconds=None)
        samples = 6 * INPUT_FRAMES + 37
        async with _channel("整段", settings=settings, max_seconds=samples / INPUT_RATE) as (voice, frames, asr, events):
            captured = frames.push(True, 9)[:samples * SAMPLE_BYTES]
            await _until(lambda: bool(asr.pcm) and voice._pending == 0)
            self.assertEqual(asr.pcm, [captured])
            self.assertEqual(events, [(None, True)])
            frames.end_turn()
            self.assertEqual(events[-1], ("整段", False))

    async def test_cap_includes_successful_segments_and_keeps_short_tail_until_long_silence(self):
        cap_samples = (_LONG + _SHORT + 3) * INPUT_FRAMES + 37
        async with _channel("上半句", "封顶尾段",
                            max_seconds=cap_samples / INPUT_RATE) as (voice, frames, asr, events):
            prefix = frames.push(True, _LONG) + frames.short_pause()
            await _until(lambda: len(asr.pcm) == 1 and voice._pending == 0)
            before_cap = frames.push(True, 4)
            tail = before_cap[:3 * INPUT_FRAMES * SAMPLE_BYTES + 37 * SAMPLE_BYTES]
            frames.push(True, 4)  # 封顶后说话不会进入 WAV。
            frames.push(False, _END - 1)
            frames.push(False, gap=True)
            frames.push(False, _END - 1)
            self.assertEqual(voice._pending, 0)
            self.assertEqual(asr.pcm, [prefix])
            self.assertEqual(events, [(None, True)])
            frames.push(False)
            await _until(lambda: not voice._input_active)
            self.assertEqual(asr.pcm, [prefix, tail])
            self.assertEqual(len(prefix + tail), cap_samples * SAMPLE_BYTES)
            self.assertEqual(events[-1], ("上半句\n封顶尾段", False))

    async def test_cap_retries_failed_early_audio_after_next_short_or_long_silence(self):
        for next_short in (False, True):
            with self.subTest(next_short=next_short):
                cap_samples = _LONG * INPUT_FRAMES + 37
                async with _channel(AsrError("synthetic"), "补交恢复",
                                    max_seconds=cap_samples / INPUT_RATE) as (voice, frames, asr, events):
                    captured = frames.push(True, _LONG + 3)[:cap_samples * SAMPLE_BYTES]
                    frames.short_pause()
                    await _until(lambda: len(asr.pcm) == 1 and voice._pending == 0)
                    frames.push(False, 3)
                    self.assertEqual(voice._pending, 0)  # 同一次静默不重试。
                    if next_short:
                        frames.push(True)
                        frames.short_pause()
                        await _until(lambda: len(asr.pcm) == 2 and voice._pending == 0)
                        self.assertEqual(events, [(None, True)])
                    frames.push(False, _END - _SHORT - (0 if next_short else 3))
                    await _until(lambda: not voice._input_active)
                    self.assertEqual(asr.pcm, [captured, captured])
                    self.assertEqual(events, [(None, True), ("补交恢复", False)])

    async def test_gap_drops_successful_prefix_and_incomplete_tail(self):
        async with _channel("丢弃", "下一轮") as (voice, frames, asr, events):
            prefix = frames.push(True, _LONG) + frames.short_pause()
            await _until(lambda: len(asr.pcm) == 1 and voice._pending == 0)
            frames.push(True, 3)
            frames.push(True, gap=True)
            frames.push(True, 3)
            frames.end_turn()
            self.assertEqual(events, [(None, True), (None, False)])
            following = frames.push(True, 3) + frames.end_turn()
            await _until(lambda: not voice._input_active)
            self.assertEqual(asr.pcm, [prefix, following])
            self.assertEqual(events[-1], ("下一轮", False))

    async def test_gap_cancels_running_segment_and_late_result_is_not_handed_off(self):
        cancelled, release = asyncio.Event(), asyncio.Event()

        async def early():
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                return "不可交付"

        try:
            async with _channel(early, "另一轮") as (voice, frames, asr, events):
                prefix = frames.push(True, _LONG) + frames.short_pause()
                await _until(lambda: len(asr.pcm) == 1)
                frames.push(True, gap=True)
                frames.end_turn()
                following = frames.push(True, 3) + frames.end_turn()
                await asyncio.wait_for(cancelled.wait(), 1)
                self.assertEqual(events, [(None, True), (None, True)])
                release.set()
                await _until(lambda: not voice._input_active)
                self.assertEqual(asr.pcm, [prefix, following])
                self.assertEqual(events[-1], ("另一轮", False))
        finally:
            release.set()

    async def test_gap_does_not_cancel_another_turns_request_even_around_its_exit(self):
        for status in ("running", "timed_out", "done_before_reap"):
            with self.subTest(status=status):
                release, cancelled, gap_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
                observations = []

                def gap():
                    before = old_task.cancelling()
                    frames.push(True, gap=True)
                    observations.append((voice._asr_task is old_task, old_task.done(), before, old_task.cancelling()))
                    gap_done.set()

                async def first():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        cancelled.set()
                        await release.wait()
                    if status == "done_before_reap":
                        asyncio.get_running_loop().call_soon(gap)
                    return "上一轮"

                try:
                    async with _channel(first, timeout=0.15) as (voice, frames, asr, events):
                        recorded = frames.push(True, 3) + frames.end_turn()
                        await _until(lambda: len(asr.pcm) == 1)
                        old_task = voice._asr_task
                        frames.push(True, 3)
                        if status == "timed_out":
                            await asyncio.wait_for(cancelled.wait(), 1)
                        if status == "done_before_reap":
                            release.set()
                        else:
                            gap()
                        await asyncio.wait_for(gap_done.wait(), 1)
                        expected_cancels = int(status == "timed_out")
                        self.assertEqual(observations, [(True, status == "done_before_reap",
                                                         expected_cancels, expected_cancels)])
                        frames.end_turn()
                        release.set()
                        await _until(lambda: voice._asr_task is None and not voice._input_active)
                        self.assertEqual(asr.pcm, [recorded])
                        self.assertEqual([text for text, _ in events if text],
                                         [] if status == "timed_out" else ["上一轮"])
                finally:
                    release.set()

    async def test_gap_releases_scheduler_after_inflight_exit_but_not_during_another_turn(self):
        for another_turn in (False, True):
            with self.subTest(another_turn=another_turn):
                release, cancelled = asyncio.Event(), asyncio.Event()

                async def early():
                    try:
                        await asyncio.Future()
                    except asyncio.CancelledError:
                        cancelled.set()
                        await release.wait()
                        return "缺帧后的迟到结果"

                asr, device = _ASR(["之前成功的片段", early, "下一轮"]), _Device()
                voice = VoiceChannel(asr, _TTS(), _Vad(), _SETTINGS, asr_timeout=1, max_audio_seconds=500)
                scheduler = SessionScheduler()
                batches = []
                scheduler.bind(batches.append)
                scheduler.call_started(voice.stop_io)
                pending = MessageDelivery(Message("user", "暂存的会话输入"), "trigger")

                def on_input(text, active):
                    deliveries = (MessageDelivery(Message("user", text), "trigger"),) if text else ()
                    scheduler.call_input(voice.stop_io, deliveries, input_active=active)

                try:
                    async with voice.connected(device, on_input):
                        frames = _Frames(voice)
                        frames.push(True, _LONG)
                        frames.short_pause()
                        await _until(lambda: len(asr.pcm) == 1 and voice._pending == 0)
                        frames.push(True, _LONG)
                        frames.short_pause()
                        await _until(lambda: len(asr.pcm) == 2)
                        scheduler.session_input((pending,))
                        frames.push(True, gap=True)
                        frames.end_turn()
                        await asyncio.wait_for(cancelled.wait(), 1)
                        self.assertEqual(batches, [])
                        if another_turn:
                            frames.push(True, 3)
                        release.set()
                        await _until(lambda: voice._pending == 0)
                        if another_turn:
                            self.assertEqual(batches, [])
                            frames.end_turn()
                            await _until(lambda: not voice._input_active)
                        next_turn = MessageDelivery(Message("user", "下一轮"), "trigger")
                        expected = (pending, next_turn) if another_turn else (pending,)
                        self.assertEqual(batches, [expected])
                finally:
                    release.set()

    async def test_after_early_timeout_ignores_late_text_and_waits_to_actually_resend(self):
        late_cancelled, release = asyncio.Event(), asyncio.Event()

        async def early():
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                late_cancelled.set()
                await release.wait()
                return "迟到不能回传"

        try:
            async with _channel(early, "最终成功", timeout=0.1) as (voice, frames, asr, events):
                prefix = frames.push(True, _LONG) + frames.short_pause()
                await _until(lambda: len(asr.pcm) == 1)
                extra = frames.push(False, _END - _SHORT - 1)
                frames.push(False)
                await asyncio.wait_for(late_cancelled.wait(), 1)
                self.assertEqual(asr.pcm, [prefix])
                self.assertEqual(events, [(None, True)])
                release.set()
                await _until(lambda: not voice._input_active)
                self.assertEqual(asr.pcm, [prefix, prefix + extra])
                self.assertEqual(events[-1], ("最终成功", False))
        finally:
            release.set()

    async def test_deadline_rejects_late_early_success_and_resends_full_audio(self):
        with _loop_time_jump() as jump:
            async def late():
                jump()
                return "已超限的提前结果"

            async with _channel(late, "最终补交", timeout=0.2) as (voice, frames, asr, events):
                with self.assertLogs("agent_companion.speech.voice", "WARNING") as log:
                    prefix = frames.push(True, _LONG) + frames.short_pause()
                    await _until(lambda: len(asr.pcm) == 1 and voice._pending == 0)
                self.assertIn("deadline reached; phase=early", log.output[0])
                self.assertEqual(events, [(None, True)])
                remainder = frames.push(False, _END - _SHORT - 1)
                frames.push(False)
                await _until(lambda: not voice._input_active)
                self.assertEqual(asr.pcm, [prefix, prefix + remainder])
                self.assertEqual(events[-1], ("最终补交", False))

    async def test_deadline_rejects_late_final_success_without_releasing_partial_text(self):
        with _loop_time_jump() as jump:
            async def late():
                jump()
                return "已超限的最终结果"

            async with _channel("提前成功", late, timeout=0.2) as (voice, frames, asr, events):
                prefix = frames.push(True, _LONG) + frames.short_pause()
                await _until(lambda: len(asr.pcm) == 1 and voice._pending == 0)
                with self.assertLogs("agent_companion.speech.voice", "WARNING") as log:
                    tail = frames.push(True, 3) + frames.end_turn()
                    await _until(lambda: not voice._input_active)
                self.assertIn("deadline reached; phase=final", log.output[0])
                self.assertEqual(asr.pcm, [prefix, tail])
                self.assertEqual(events, [(None, True), (None, False)])

    async def test_deadline_accepts_actual_timely_result_even_if_observed_late(self):
        with _loop_time_jump() as jump:
            async def timely():
                # 请求自身在时钟前跳前返回，观察请求的协程只能在前跳后运行。
                asyncio.get_running_loop().call_soon(jump)
                return "按时完成"

            async with _channel(timely, timeout=0.2) as (voice, frames, asr, events):
                recorded = frames.push(True, 3) + frames.end_turn()
                await _until(lambda: not voice._input_active)
                self.assertEqual(asr.pcm, [recorded])
                self.assertEqual(events, [(None, True), ("按时完成", False)])

    async def test_omitting_segment_option_keeps_existing_deadline_observation(self):
        settings = replace(_SETTINGS, segment_pause_seconds=None)
        with _loop_time_jump() as jump:
            async def late():
                jump()
                return "整段转写原有行为"

            async with _channel(late, timeout=0.2, settings=settings) as (voice, frames, asr, events):
                recorded = frames.push(True, 3) + frames.end_turn()
                await _until(lambda: not voice._input_active)
                self.assertEqual(asr.pcm, [recorded])
                self.assertEqual(events, [(None, True), ("整段转写原有行为", False)])

    async def test_late_file_empty_does_not_wake_even_if_asr_finishes_before_observation(self):
        settings = replace(_SETTINGS, segment_pause_seconds=None)
        with _loop_time_jump() as jump:
            async def late():
                jump()
                raise AsrError("asr_empty_transcript")

            async with _channel(late, timeout=0.2, settings=settings) as (voice, frames, _, events):
                playback = asyncio.create_task(voice.deliver(_reply("old", "仍在播报。")))
                await _until(lambda: voice._output._active is not None and voice._output._active.draining)
                pcm = bytearray(2)
                voice._output.render(memoryview(pcm))
                self.assertTrue(any(pcm))
                frames.push(True, 3)
                frames.end_turn()
                await playback
                await _until(lambda: not voice._input_active)
                self.assertEqual(events, [(None, True), (None, False)])

    async def test_new_short_turn_does_not_wait_for_another_turns_stuck_request(self):
        for previous_was_early in (False, True):
            with self.subTest(previous_was_early=previous_was_early):
                release, cancelled = asyncio.Event(), asyncio.Event()

                async def stuck():
                    try:
                        await asyncio.Future()
                    except asyncio.CancelledError:
                        cancelled.set()
                        await release.wait()
                        return "上一轮迟到"

                try:
                    async with _channel(stuck, "恢复", timeout=0.05) as (voice, frames, asr, events):
                        if previous_was_early:
                            recorded = frames.push(True, _LONG) + frames.short_pause()
                            await _until(lambda: len(asr.pcm) == 1)
                            frames.push(False, _END - _SHORT)
                        else:
                            recorded = frames.push(True, 3) + frames.end_turn()
                        await asyncio.wait_for(cancelled.wait(), 1)
                        await _until(lambda: not voice._input_active)
                        self.assertIsNotNone(voice._asr_task)
                        self.assertFalse(voice._asr_task.done())
                        with (patch.object(asyncio, "wait", wraps=asyncio.wait) as wait,
                              self.assertLogs("agent_companion.speech.voice", "WARNING") as log):
                            frames.push(True, 3)
                            frames.end_turn()
                            await _until(lambda: not voice._input_active)
                            wait.assert_not_awaited()
                        self.assertIn("skipped (ASR busy); phase=final", log.output[0])
                        self.assertEqual(asr.pcm, [recorded])
                        self.assertEqual(events, [(None, True), (None, False)] * 2)
                        release.set()
                        await _until(lambda: voice._asr_task is None)
                        following = frames.push(True, 3) + frames.end_turn()
                        await _until(lambda: not voice._input_active)
                        self.assertEqual(asr.pcm, [recorded, following])
                        self.assertEqual(events[-1], ("恢复", False))
                finally:
                    release.set()

    async def test_short_turn_queues_while_ordinary_final_request_is_receiving_cancellation(self):
        observations = []

        async def first():
            nonlocal first_task
            first_task = asyncio.current_task()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                # 在适配器收到取消而尚未返回时结束 B，保证命中原来会被提前丢弃的窗口。
                state = (voice._asr_task is first_task, first_task.cancelling(), first_task.done())
                frames.push(False)
                observations.append((state, voice._pending))
                return "超时的第一轮结果"

        async def second():
            self.assertTrue(first_task.done())
            return "完整的第二轮结果"

        first_task = None
        async with _channel(first, second, timeout=0.05) as (voice, frames, asr, events):
            first_pcm = frames.push(True, 3) + frames.end_turn()
            second_pcm = frames.push(True, 4) + frames.push(False, _END - 1)
            await _until(lambda: not voice._input_active)
            self.assertEqual(observations, [((True, 1, False), 1)])
            self.assertEqual(asr.pcm, [first_pcm, second_pcm])
            self.assertEqual(events, [(None, True), (None, True), ("完整的第二轮结果", False)])

    async def test_short_turn_during_another_turns_cleanup_does_not_take_a_slot(self):
        release, cancelled = asyncio.Event(), asyncio.Event()
        just_released = []

        def close_third_turn():
            # 适配器已返回，但其 done 回调尚未运行；不可把该请求当作仍占槽。
            state = (voice._asr_task is old_task, old_task.done(), voice._pending)
            pcm = frames.end_turn()
            just_released.append((state, voice._pending, pcm))

        async def early():
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                asyncio.get_running_loop().call_soon(close_third_turn)
                return "过期结果"

        try:
            async with _channel(early, "第一轮补交", "第三轮", timeout=0.2) as (voice, frames, asr, events):
                first_pcm = frames.push(True, _LONG) + frames.short_pause()
                await _until(lambda: len(asr.pcm) == 1)
                old_task = voice._asr_task
                old_request = voice._asr_request
                first_extra = frames.push(False, _END - _SHORT - 1)
                frames.push(False)
                await asyncio.wait_for(cancelled.wait(), 1)
                self.assertTrue(old_request.waiting_for_retry_cleanup)
                with self.assertLogs("agent_companion.speech.voice", "WARNING") as log:
                    frames.push(True, 3)
                    frames.end_turn()
                self.assertIn("skipped (ASR busy); phase=final", log.output[0])
                self.assertFalse(old_task.done())
                self.assertEqual(voice._pending, 1)
                self.assertFalse(voice._turns)
                self.assertEqual(events, [(None, True), (None, True)])
                third_pcm = frames.push(True, 3)
                release.set()
                await _until(lambda: not voice._input_active)
                [(request_state, pending, third_tail)] = just_released
                self.assertEqual((request_state, pending), ((True, True, 1), 2))
                self.assertFalse(old_request.waiting_for_retry_cleanup)
                self.assertEqual(asr.pcm, [first_pcm, first_pcm + first_extra, third_pcm + third_tail])
                self.assertEqual(events[-2:], [("第一轮补交", None), ("第三轮", False)])
        finally:
            release.set()

    async def test_final_cannot_start_if_cancelled_adapter_keeps_its_slot(self):
        release = asyncio.Event()

        async def early():
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                await release.wait()
                return "过期不能回传"

        try:
            async with _channel(early, "新一轮", timeout=0.04) as (voice, frames, asr, events):
                prefix = frames.push(True, _LONG) + frames.short_pause()
                await _until(lambda: len(asr.pcm) == 1)
                with self.assertLogs("agent_companion.speech.voice", "WARNING") as log:
                    frames.push(False, _END - _SHORT)
                    await _until(lambda: not voice._input_active)
                self.assertTrue(any("could not start" in line for line in log.output))
                self.assertEqual(asr.pcm, [prefix])
                self.assertEqual(events, [(None, True), (None, False)])
                release.set()
                await _until(lambda: voice._asr_task is None)
                following = frames.push(True, 3) + frames.end_turn()
                await _until(lambda: not voice._input_active)
                self.assertEqual(asr.pcm, [prefix, following])
                self.assertEqual(events[-1], ("新一轮", False))
        finally:
            release.set()

    async def test_final_attempt_timeout_drops_the_turn_and_does_not_publish_early_text(self):
        async def late():
            await asyncio.Future()

        async with _channel("不要交付前段", late, "新话轮", timeout=0.08) as (voice, frames, asr, events):
            prefix = frames.push(True, _LONG) + frames.short_pause()
            await _until(lambda: len(asr.pcm) == 1 and voice._pending == 0)
            tail = frames.push(True, 3) + frames.end_turn()
            await _until(lambda: not voice._input_active)
            self.assertEqual(events, [(None, True), (None, False)])
            following = frames.push(True, 3) + frames.end_turn()
            await _until(lambda: not voice._input_active)
            self.assertEqual(asr.pcm, [prefix, tail, following])
            self.assertEqual(events[-1], ("新话轮", False))

    async def test_disconnect_clears_retry_cleanup_wait_before_adapter_exits(self):
        cancelled, release = asyncio.Event(), asyncio.Event()

        async def early():
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                return "挂断后的迟到结果"

        asr, device = _ASR([early]), _Device()
        voice = VoiceChannel(asr, _TTS(), _Vad(), _SETTINGS, asr_timeout=0.2, max_audio_seconds=500)
        events = []
        try:
            async with voice.connected(device, lambda *event: events.append(event), wait_for_cleanup=False):
                frames = _Frames(voice)
                frames.push(True, _LONG)
                frames.short_pause()
                await _until(lambda: bool(asr.pcm))
                frames.push(False, _END - _SHORT)
                await asyncio.wait_for(cancelled.wait(), 1)
                request = voice._asr_request
                self.assertTrue(request.waiting_for_retry_cleanup)
            self.assertFalse(request.waiting_for_retry_cleanup)
            self.assertFalse(request.task.done())
            self.assertEqual(events, [(None, True)])
        finally:
            release.set()
        await _until(lambda: voice._asr_task is None)

    async def test_device_failure_does_not_publish_completed_early_text(self):
        asr, device = _ASR(["已提前转写"]), _Device()
        voice = VoiceChannel(asr, _TTS(), _Vad(), _SETTINGS, asr_timeout=1, max_audio_seconds=500)
        events = []
        async with voice.connected(device, lambda *event: events.append(event)):
            frames = _Frames(voice)
            frames.push(True, _LONG)
            frames.short_pause()
            await _until(lambda: len(asr.pcm) == 1 and voice._pending == 0)
            voice._disable_device(device, "device stream stopped")
            self.assertTrue(device.closed)
            self.assertIsNone(voice._segment_turn)
            self.assertEqual(events, [(None, True), (None, False)])

    async def test_disconnect_drops_incomplete_transcript_without_waiting_for_adapter(self):
        cancelled, release = asyncio.Event(), asyncio.Event()

        async def early():
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                return "迟到不能回传"

        asr, device = _ASR(["已识别", early]), _Device()
        voice = VoiceChannel(asr, _TTS(), _Vad(), _SETTINGS, asr_timeout=1, max_audio_seconds=500)
        frames = _Frames(voice)
        events = []
        try:
            async with asyncio.timeout(0.5):
                async with voice.connected(device, lambda *event: events.append(event), wait_for_cleanup=False):
                    frames.push(True, _LONG)
                    frames.short_pause()
                    await _until(lambda: len(asr.pcm) == 1 and voice._pending == 0)
                    frames.push(True, _LONG)
                    frames.short_pause()
                    await _until(lambda: len(asr.pcm) == 2)
            await asyncio.wait_for(cancelled.wait(), 1)
            self.assertTrue(device.closed)
            self.assertIsNone(voice._segment_turn)
            self.assertEqual(voice._pending, 0)
            release.set()
            await _until(lambda: voice._asr_task is None)
            self.assertEqual(events, [(None, True)])
        finally:
            release.set()
