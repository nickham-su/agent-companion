import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from math import inf, nextafter
import re
from time import monotonic
import unittest
from unittest.mock import patch

from agent_companion.protocol import DeliveryStatus
from agent_companion.speech.asr import AsrError
from agent_companion.speech.vad import INPUT_BYTES, INPUT_RATE, SAMPLE_BYTES
from agent_companion.speech.voice import VoiceChannel

from test_voice import _ASR, _Device, _HOT, _QUIET, _SETTINGS, _TTS, _Vad, _reply, _until
from test_voice_segments import _Frames


_SETTINGS = replace(_SETTINGS, segment_pause_seconds=0.128, silence_seconds=0.32)


async def _read(frames, chunks):
    async for chunk in frames:
        chunks.append(chunk)


class _StreamASR:
    def __init__(self, *handlers):
        self.handlers = handlers
        self.calls = []

    async def transcribe_stream(self, frames):
        chunks = []
        i = len(self.calls)
        self.calls.append(chunks)
        if i < len(self.handlers) and self.handlers[i] is not None:
            return await self.handlers[i](frames, chunks)
        await _read(frames, chunks)
        return f"合成话轮 {i + 1}"


@asynccontextmanager
async def _channel(*handlers, timeout=0.5, max_seconds=120):
    live, file, device = _StreamASR(*handlers), _ASR(), _Device()
    voice = VoiceChannel(file, _TTS(), _Vad(), _SETTINGS, streaming=live,
                         asr_timeout=timeout, max_audio_seconds=max_seconds)
    events = []
    async with voice.connected(device, lambda *event: events.append(event), capture_enabled=False):
        yield voice, _Frames(voice), live, file, events


def _finish(voice, frames):
    tail = frames.push(False, voice._end_frames - 1)
    frames.push(False)
    return tail


def _texts(events):
    return [text for text, _ in events if text]


class StreamingVoiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_frame_count_matches_file_mode_without_changing_legacy_rounding(self):
        # 刚超过 9 帧的浮点值，旧整轮路径向上取整前的除法仍得到 9。
        above = nextafter(0.288, inf)
        for pause, seconds, expected in ((0.128, 0.288, 9), (0.128, above, 10), (None, above, 9)):
            settings = replace(_SETTINGS, segment_pause_seconds=pause, silence_seconds=seconds)
            for streaming in (None, _StreamASR()):
                with self.subTest(pause=pause, seconds=seconds, streaming=streaming is not None):
                    voice = VoiceChannel(_ASR(), _TTS(), _Vad(), settings, streaming=streaming,
                                         asr_timeout=0.5, max_audio_seconds=120)
                    async with voice.connected(_Device(), lambda *_: None, capture_enabled=False):
                        frames = _Frames(voice)
                        frames.push(True, settings.speech_frames)
                        frames.push(False, expected - 1)
                        self.assertTrue(voice._speaking)
                        frames.push(False)
                        self.assertFalse(voice._speaking)
                        if streaming is not None:
                            self.assertIsNone(voice._segment_frames)

    async def test_device_capture_is_consumed_while_user_is_still_speaking(self):
        live, file, vad, device = _StreamASR(), _ASR(), _Vad(), _Device()
        voice = VoiceChannel(file, _TTS(), vad, _SETTINGS, asr_timeout=0.5, max_audio_seconds=120, streaming=live)
        events = []
        async with voice.connected(device, lambda *event: events.append(event)):
            device.frame(False, 2)
            device.frame(True, _SETTINGS.speech_frames)
            await _until(lambda: bool(live.calls and live.calls[0]))
            self.assertEqual(events, [(None, True)])
            self.assertEqual(live.calls[0], [_QUIET * 2 + _HOT * _SETTINGS.speech_frames])
            device.frame(True, 4)
            await _until(lambda: len(live.calls[0]) == 5)
            self.assertEqual(_texts(events), [])
            device.frame(False, voice._end_frames)
            await _until(lambda: bool(_texts(events)))
            self.assertEqual(b"".join(live.calls[0]),
                             _QUIET * 2 + _HOT * (_SETTINGS.speech_frames + 4) + _QUIET * (voice._end_frames - 1))
            self.assertEqual(events[-1], ("合成话轮 1", False))
            self.assertEqual(file.files, [])

    async def test_short_turn_sends_preroll_and_live_frames_before_endpoint(self):
        async with _channel() as (voice, frames, live, file, events):
            with self.assertLogs("agent_companion.speech.voice", "DEBUG") as captured:
                lead = frames.push(False, 2) + frames.push(True, _SETTINGS.speech_frames)
                await _until(lambda: bool(live.calls and live.calls[0]))
                self.assertEqual(live.calls, [[lead]])
                self.assertEqual(events, [(None, True)])
                self.assertEqual((await voice.deliver(_reply("obsolete", "合成播报"))).status,
                                 DeliveryStatus.ZERO)
                middle = frames.push(True, 3)
                await _until(lambda: len(live.calls[0]) == 4)
                self.assertEqual(b"".join(live.calls[0]), lead + middle)
                self.assertEqual(_texts(events), [])
                tail = _finish(voice, frames)
                await _until(lambda: bool(_texts(events)))
            self.assertEqual(events, [(None, True), ("合成话轮 1", False)])
            self.assertEqual(b"".join(live.calls[0]), lead + middle + tail)
            self.assertEqual(file.files, [])
            start, end = captured.output
            match = re.search(r"endpoint detected; turn=(\S+) at_mono_ms=(\d+)", start)
            ready = re.search(r"transcript ready; turn=(\S+) at_mono_ms=(\d+) endpoint_to_transcript_ms=(\d+)", end)
            self.assertIsNotNone(match)
            self.assertIsNotNone(ready)
            self.assertEqual(match[1], ready[1])
            self.assertLessEqual(abs(int(ready[2]) - int(match[2]) - int(ready[3])), 1)
            self.assertNotIn("合成话轮", end)

    async def test_long_turn_does_not_segment_or_time_out_while_speaking(self):
        skew = 0.0
        with patch("agent_companion.speech.voice.monotonic", side_effect=lambda: monotonic() + skew):
            async with _channel() as (voice, frames, live, file, events):
                sent = frames.push(True, 3)
                await _until(lambda: bool(live.calls and live.calls[0]))
                skew = 60  # 已超过文件时限，但还没停口。
                for _ in range(21):
                    sent += frames.push(True, 16)
                    target = len(sent)
                    await _until(lambda: sum(map(len, live.calls[0])) == target)
                sent += frames.push(False, 5) + frames.push(True, 10)  # 已超过文件分段条件。
                self.assertFalse(_texts(events))
                sent += _finish(voice, frames)
                await _until(lambda: bool(_texts(events)))
                self.assertEqual([b"".join(call) for call in live.calls], [sent])
                self.assertEqual(file.files, [])

    async def test_cap_freezes_pcm_but_does_not_finish_before_local_silence(self):
        cap = 5 * INPUT_BYTES + 2 * SAMPLE_BYTES
        async with _channel(max_seconds=cap / (INPUT_RATE * SAMPLE_BYTES)) as (voice, frames, live, _, events):
            kept = frames.push(True, 10)[:cap]
            await _until(lambda: bool(live.calls and sum(map(len, live.calls[0])) == cap))
            self.assertTrue(voice._input_active)
            self.assertFalse(_texts(events))
            _finish(voice, frames)
            await _until(lambda: bool(_texts(events)))
            self.assertEqual(b"".join(live.calls[0]), kept)
            self.assertEqual(events[-1], ("合成话轮 1", False))

    async def test_queues_second_turn_drops_third_and_preserves_order(self):
        eof = asyncio.Event()
        release = asyncio.Event()

        async def first(frames, chunks):
            await _read(frames, chunks)
            eof.set()
            await release.wait()
            return "首个合成话轮"

        async with _channel(first) as (voice, frames, live, file, events):
            pcm1 = frames.push(True, 4) + _finish(voice, frames)
            await _until(eof.is_set)
            pcm2 = frames.push(True, 5) + _finish(voice, frames)
            with self.assertLogs("agent_companion.speech.voice", "WARNING") as captured:
                frames.push(True, 7)
                _finish(voice, frames)
            self.assertIn("asr backlog", captured.output[0])
            self.assertEqual(len(live.calls), 1)
            release.set()
            await _until(lambda: len(_texts(events)) == 2)
            self.assertEqual([b"".join(call) for call in live.calls], [pcm1, pcm2])
            self.assertEqual(events[-2:], [("首个合成话轮", None), ("合成话轮 2", False)])
            self.assertEqual(file.files, [])
            self.assertEqual(voice._pending, 0)
            self.assertFalse(voice._input_active)

    async def test_queued_turn_abort_frees_its_slot_before_worker_finishes_previous_turn(self):
        for reason in ("capture gap", "buffer full"):
            with self.subTest(reason=reason):
                eof, release = asyncio.Event(), asyncio.Event()

                async def first(frames, chunks):
                    await _read(frames, chunks)
                    eof.set()
                    await release.wait()
                    return "首个合成话轮"

                async with _channel(first) as (voice, frames, live, file, events):
                    pcm1 = frames.push(True, 4) + _finish(voice, frames)
                    await _until(eof.is_set)
                    first_turn = voice._stream_in_flight
                    frames.push(True, 3)
                    self.assertEqual(voice._pending, 2)
                    with self.assertLogs("agent_companion.speech.voice", "WARNING") as logged:
                        if reason == "capture gap":
                            frames.push(True, gap=True)
                        else:
                            frames.push(True, 65)
                    self.assertIn(reason, logged.output[0])
                    self.assertEqual(voice._pending, 1)
                    self.assertIs(voice._stream_in_flight, first_turn)
                    _finish(voice, frames)
                    self.assertTrue(voice._input_active)
                    pcm3 = frames.push(True, 5) + _finish(voice, frames)
                    self.assertEqual(voice._pending, 2)
                    with self.assertLogs("agent_companion.speech.voice", "WARNING") as logged:
                        frames.push(True, 7)
                        _finish(voice, frames)
                    self.assertIn("asr backlog", logged.output[0])
                    self.assertEqual(events, [(None, True)] * 4)
                    self.assertEqual(len(live.calls), 1)
                    release.set()
                    await _until(lambda: not voice._input_active)
                    self.assertEqual([b"".join(call) for call in live.calls], [pcm1, pcm3])
                    self.assertEqual(events[4:], [("首个合成话轮", None), ("合成话轮 2", False)])
                    self.assertEqual(voice._pending, 0)
                    self.assertEqual(file.files, [])

    async def test_pcm_backpressure_drops_whole_turn_and_recovers(self):
        blocked = asyncio.Event()
        cancelled = asyncio.Event()

        async def slow(frames, chunks):
            chunks.append(await anext(frames))
            blocked.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        async with _channel(slow) as (voice, frames, live, file, events):
            frames.push(True, 3)
            await _until(blocked.is_set)
            with self.assertLogs("agent_companion.speech.voice", "WARNING") as captured:
                # 不等待网络；超过单轮待发的 2 秒后丢弃本轮。
                frames.push(True, 65)
            self.assertIn("buffer full", captured.output[0])
            self.assertEqual(voice._pending, 1)  # 在途槽仍归 worker，由它清理。
            self.assertFalse(cancelled.is_set())
            await _until(cancelled.is_set)
            _finish(voice, frames)
            self.assertEqual(_texts(events), [])
            pcm = frames.push(True, 3) + _finish(voice, frames)
            await _until(lambda: bool(_texts(events)))
            self.assertEqual(_texts(events), ["合成话轮 2"])
            self.assertEqual(b"".join(live.calls[1]), pcm)
            self.assertEqual(file.files, [])

    async def test_capture_gap_never_signals_normal_eof_and_next_turn_recovers(self):
        consumed = asyncio.Event()
        cancelled = asyncio.Event()

        async def read(frames, chunks):
            chunks.append(await anext(frames))
            consumed.set()
            try:
                await _read(frames, chunks)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            self.fail("a capture gap cannot become normal EOF")

        async with _channel(read) as (voice, frames, live, _, events):
            frames.push(True, 3)
            await _until(consumed.is_set)
            frames.push(True, gap=True)
            _finish(voice, frames)
            await _until(cancelled.is_set)
            self.assertEqual(_texts(events), [])
            self.assertEqual(events[-1], (None, False))
            pcm = frames.push(True, 3) + _finish(voice, frames)
            await _until(lambda: bool(_texts(events)))
            self.assertEqual(b"".join(live.calls[1]), pcm)

    async def test_expired_queued_turn_does_not_start_and_later_turn_recovers(self):
        release = asyncio.Event()
        closed = asyncio.Event()
        skew = 0.0

        async def first(frames, chunks):
            await _read(frames, chunks)
            closed.set()
            await release.wait()
            return "不应接收"

        with patch("agent_companion.speech.voice.monotonic", side_effect=lambda: monotonic() + skew):
            async with _channel(first) as (voice, frames, live, _, events):
                frames.push(True, 3)
                _finish(voice, frames)
                await _until(closed.is_set)
                frames.push(True, 3)
                _finish(voice, frames)
                skew = 2
                release.set()
                await _until(lambda: not voice._input_active)
                self.assertEqual(_texts(events), [])
                self.assertEqual(len(live.calls), 1)
                frames.push(True, 3)
                _finish(voice, frames)
                await _until(lambda: bool(_texts(events)))
                self.assertEqual(_texts(events), ["合成话轮 2"])

    async def test_eof_reached_by_background_reader_after_adapter_return_cannot_make_it_valid(self):
        for already_ended in (False, True):
            with self.subTest(already_ended=already_ended):
                reader = None

                async def early(pcm, chunks):
                    nonlocal reader
                    chunks.append(await anext(pcm))
                    if already_ended:
                        _finish(voice, frames)

                    async def drain():
                        if not already_ended:
                            _finish(voice, frames)
                        await _read(pcm, chunks)

                    # 此回调先于 worker 被唤醒；旧检查可在这里看到补上的 EOF。
                    reader = asyncio.create_task(drain())
                    return "未等 EOF 的合成文本"

                async with _channel(early) as (voice, frames, live, file, events):
                    with self.assertLogs("agent_companion.speech.voice", "WARNING"):
                        frames.push(True, 3)
                        await _until(lambda: reader is not None)
                        await reader  # 已实际读到正常 EOF，但发生在适配器返回之后。
                        await _until(lambda: not voice._input_active)
                        self.assertEqual(_texts(events), [])
                    frames.push(True, 3)
                    _finish(voice, frames)
                    await _until(lambda: not voice._input_active)
                    self.assertEqual(_texts(events), ["合成话轮 2"])
                    self.assertEqual(file.files, [])

    async def test_final_result_uses_adapter_return_time_even_if_worker_runs_after_deadline(self):
        for duration in (0.25, 0.5):
            with self.subTest(duration=duration):
                now = 100.0

                def delay_worker():
                    nonlocal now
                    now = 102.0

                async def finish_after_eof(frames, chunks):
                    nonlocal now
                    await _read(frames, chunks)
                    now += duration
                    asyncio.get_running_loop().call_soon(delay_worker)
                    return "及时且完整的合成文本"

                with (patch("agent_companion.speech.voice.monotonic", new=lambda: now),
                      self.assertLogs("agent_companion.speech.voice", "DEBUG") as logged):
                    async with _channel(finish_after_eof) as (voice, frames, live, file, events):
                        pcm = frames.push(True, 3) + _finish(voice, frames)
                        await _until(lambda: not voice._input_active)
                self.assertEqual(now, 102.0)
                self.assertEqual([b"".join(call) for call in live.calls], [pcm])
                self.assertEqual(file.files, [])
                if duration < 0.5:
                    self.assertEqual(events, [(None, True), ("及时且完整的合成文本", False)])
                    self.assertIn("at_mono_ms=100250 endpoint_to_transcript_ms=250", logged.output[-1])
                else:
                    self.assertEqual(events, [(None, True), (None, False)])
                    self.assertIn("deadline reached", logged.output[-1])

    async def test_slow_start_is_cancelled_before_endpoint_and_never_retried(self):
        cancelled = asyncio.Event()

        async def slow(_frames, _chunks):
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        async with _channel(slow, timeout=0.08) as (voice, frames, live, file, events):
            frames.push(True, 3)
            await _until(cancelled.is_set)
            self.assertTrue(voice._input_active)
            _finish(voice, frames)
            self.assertFalse(_texts(events))
            self.assertEqual(len(live.calls), 1)
            self.assertEqual(file.files, [])

    async def test_late_cancellation_cleanup_does_not_deliver_or_consume_next_slot(self):
        eof = asyncio.Event()
        cancelled = asyncio.Event()
        release = asyncio.Event()

        async def late(frames, chunks):
            try:
                await _read(frames, chunks)
                eof.set()
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                return "迟到的合成文本"

        async with _channel(late, timeout=0.12) as (voice, frames, live, file, events):
            frames.push(True, 3)
            _finish(voice, frames)
            await _until(eof.is_set)
            await _until(cancelled.is_set)
            self.assertFalse(_texts(events))
            first = frames.push(True, 3)
            await asyncio.sleep(0)
            self.assertEqual(len(live.calls), 1)
            release.set()
            await _until(lambda: len(live.calls) == 2 and live.calls[1])
            tail = _finish(voice, frames)
            await _until(lambda: bool(_texts(events)))
            self.assertEqual(_texts(events), ["合成话轮 2"])
            self.assertEqual(b"".join(live.calls[1]), first + tail)
            self.assertEqual(file.files, [])

    async def test_hangup_cancels_closed_turn_and_flushes_queued_turn(self):
        eof = asyncio.Event()
        cancelled = asyncio.Event()

        async def pending(frames, chunks):
            try:
                await _read(frames, chunks)
                eof.set()
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                return "关闭后的合成文本"

        async with _channel(pending) as (voice, frames, live, _, events):
            frames.push(True, 3)
            _finish(voice, frames)
            await _until(eof.is_set)
            frames.push(True, 3)
            _finish(voice, frames)
            voice.stop_io()
            await _until(cancelled.is_set)
        self.assertEqual(_texts(events), [])
        self.assertEqual(len(live.calls), 1)
        self.assertEqual(voice._pending, 0)

    async def test_failed_transcript_logs_safe_reason_phase_and_endpoint_turn(self):
        async def fail_after_eof(frames, chunks):
            await _read(frames, chunks)
            raise AsrError("asr_empty_transcript", phase="finish")

        async with _channel(fail_after_eof) as (voice, frames, _, _, events):
            with self.assertLogs("agent_companion.speech.voice", "DEBUG") as captured:
                frames.push(True, 3)
                _finish(voice, frames)
                await _until(lambda: not voice._input_active)
            endpoint, failed = captured.records
            self.assertEqual((endpoint.levelname, failed.levelname), ("DEBUG", "WARNING"))
            turn, _ = endpoint.args
            self.assertEqual(failed.getMessage(), f"voice streaming transcription failed; turn={turn}"
                                                 " reason=asr_empty_transcript phase=finish")
            self.assertEqual(_texts(events), [])

    async def test_adapter_failure_or_premature_success_drops_only_its_turn(self):
        async def fail(frames, chunks):
            chunks.append(await anext(frames))
            raise ValueError("不应记录的响应内容")

        async def unknown(frames, chunks):
            chunks.append(await anext(frames))
            raise AsrError("不应记录的错误码", phase="不应记录的阶段")

        async def early(frames, chunks):
            chunks.append(await anext(frames))
            return "不完整合成文本"

        for handler, reason in ((fail, "asr_failed"), (unknown, "asr_failed"), (early, "asr_premature_result")):
            async with _channel(handler) as (voice, frames, live, _, events):
                with self.assertLogs("agent_companion.speech.voice", "WARNING") as captured:
                    frames.push(True, 3)
                    await _until(lambda: len(live.calls) == 1 and voice._pending == 0)
                    _finish(voice, frames)
                warning, = captured.output
                self.assertTrue(warning.endswith(f" reason={reason} phase=-"))
                self.assertNotIn("不应记录", warning)
                self.assertEqual(_texts(events), [])
                frames.push(True, 3)
                _finish(voice, frames)
                await _until(lambda: bool(_texts(events)))
                self.assertEqual(_texts(events), ["合成话轮 2"])
