"""设备无关的实时语音通道：本地打断、话轮转写与语音投递。"""

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
import io
import logging
from math import ceil
from threading import Lock
from time import monotonic
import wave

from ..config import VoiceSettings, require_asr_audio_seconds
from ..protocol import DeliveryStatus, Receipt, ReplyStream
from .asr import MAX_AUDIO_BYTES, AudioFile, FileASR
from .audio import AudioDevice, AudioFormat
from .playback import VoiceOutput
from .tts import StreamingTTS
from .vad import INPUT_BYTES, INPUT_FRAMES, INPUT_RATE, SAMPLE_BYTES, VoiceDetector, VoiceSetupError

_LOG = logging.getLogger(__name__)
_SAMPLE_SECONDS = INPUT_FRAMES / INPUT_RATE
_WAV_HEADER_BYTES = 44  # wave 为本通道的单声道 16-bit PCM 写出的固定头长
_MAX_INPUT_FRAMES = 32  # 约 1 秒设备缓冲，溢出时丢弃不完整话轮
_MAX_ASR_TURNS = 2  # 含在途请求；不无限累积录音
_VAD_BATCH = 8
_DEVICE_CHECK_SECONDS = 1
_CAPTURE_STALL_SECONDS = 2.0  # 接通后首帧也用此宽限，远大于 32 ms 的正常采集周期。

# 状态为 None 表示只交付一条转写，不结束仍在进行的其他话轮。
VoiceInput = Callable[[str | None, bool | None], None]


class _CaptureQueue:
    def __init__(self) -> None:
        self._lock = Lock()
        self._frames: deque[tuple[int, bytes]] = deque(maxlen=_MAX_INPUT_FRAMES)
        self._sequence = 0
        self._last_sample_at: float | None = None  # 和帧序一样，只由设备输入线程写入。
        self._enabled_at: float | None = None  # 由事件循环启用或封闸，每个队列仅用于一通。
        self._closed = False

    def enable(self) -> None:
        with self._lock:
            if not self._closed and self._enabled_at is None:
                self._enabled_at = monotonic()

    def _deadline(self, enabled_at: float) -> float:
        last = self._last_sample_at
        return (enabled_at if last is None else last) + _CAPTURE_STALL_SECONDS

    def stall_remaining(self) -> float | None:
        enabled_at = self._enabled_at
        if enabled_at is None:
            return None
        now = monotonic()
        return self._deadline(enabled_at) - now

    def push(self, pcm: memoryview, overflow: bool = False) -> None:
        enabled_at = self._enabled_at
        if enabled_at is None:
            return  # 接通前不计入帧序，第一帧不能被当作采集缺口。
        self._sequence += 2 if overflow else 1
        if pcm.nbytes != INPUT_BYTES:
            return
        now = monotonic()
        if now >= self._deadline(enabled_at):
            return  # 迟到帧不能在监测任务醒来前重置已过的截止。
        if not overflow:
            # 原始数据仍在到达；处理侧覆盖旧帧或抢锁丢帧是采集缺口，不是采集停滞。
            self._last_sample_at = now
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self._enabled_at is not None:
                self._frames.append((self._sequence, bytes(pcm)))
        finally:
            self._lock.release()

    def drain(self) -> list[tuple[int, bytes]]:
        with self._lock:
            return [self._frames.popleft() for _ in range(min(_VAD_BATCH, len(self._frames)))]

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._enabled_at = None
            self._frames.clear()


def _wav(pcm: bytes) -> AudioFile:
    file = io.BytesIO()
    with wave.open(file, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_BYTES)
        wav.setframerate(INPUT_RATE)
        wav.writeframes(pcm)
    return AudioFile(file.getvalue())


class VoiceChannel:
    """只对当前语音生命周期投递；不因为打断取消主 Runtime。"""

    def __init__(self, asr: FileASR, tts: StreamingTTS, vad: VoiceDetector, settings: VoiceSettings, *,
                 asr_timeout: float, max_audio_seconds: float) -> None:
        require_asr_audio_seconds(max_audio_seconds)
        file_samples = (MAX_AUDIO_BYTES - _WAV_HEADER_BYTES) // SAMPLE_BYTES
        if max_audio_seconds >= file_samples / INPUT_RATE:
            max_samples = file_samples
        else:
            # 用与 WAV 校验相同的时长比较，找到不超过上限的最大整样本数。
            max_samples = round(max_audio_seconds * INPUT_RATE)
            if max_samples / INPUT_RATE > max_audio_seconds:
                max_samples -= 1
        if max_samples < 1:
            raise VoiceSetupError("voice_asr_max_audio_seconds_below_one_sample")
        self._max_recording_bytes = max_samples * SAMPLE_BYTES
        self._asr, self._vad = asr, vad
        self._settings = settings
        self._asr_timeout = asr_timeout
        self._output = VoiceOutput(tts, settings)
        self._capture = _CaptureQueue()
        self._turns: asyncio.Queue[bytes] = asyncio.Queue(maxsize=_MAX_ASR_TURNS)
        self._pending = 0
        self._asr_task: asyncio.Task[str] | None = None
        self._asr_worker: asyncio.Task[None] | None = None
        self._on_input: VoiceInput | None = None
        self._speaking = False
        self._ignoring_tail = False
        self._tail_recording: bytes | None = None
        self._tail_text: str | None = None
        self._obsolete_until = -float("inf")
        self._record: bytearray | None = None
        self._hot_frames = self._quiet_frames = 0
        self._discard_gap = False
        self._preroll: deque[bytes] = deque(maxlen=settings.speech_frames + 3)
        self._end_frames = ceil(settings.silence_seconds / _SAMPLE_SECONDS)
        self._last_sequence = 0
        self._opened = self._closed = False
        self._failed = self._io_stopped = False

    async def deliver(self, reply: ReplyStream) -> Receipt:
        reply.select_voice_delivery()
        if self._io_stopped:
            reply.abandon()
            return Receipt(reply.info.reply_id, DeliveryStatus.ZERO)
        if self._failed:
            reply.abandon()
            return Receipt(reply.info.reply_id, DeliveryStatus.FAILED)
        if self._input_active or reply.info.started_at <= self._obsolete_until:
            reply.abandon()
            return Receipt(reply.info.reply_id, DeliveryStatus.ZERO)
        return await self._output.deliver(reply)

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def _input_active(self) -> bool:
        return self._speaking or self._ignoring_tail or self._pending > 0

    def enable_capture(self) -> None:
        self._capture.enable()

    def stop_io(self) -> None:
        """同步关闭输入和清除待播；设备和网络请求由 connected 收尾。"""
        self._io_stopped = True
        self._capture.close()
        self._speaking = self._ignoring_tail = False
        self._tail_recording = None
        self._tail_text = None  # 尚未结束的话轮没有交接；挂断只放行已入调度器的内容。
        self._record = None
        self._preroll.clear()
        self._output.interrupt()
        self._stop_transcription()

    def _begin_turn(self) -> None:
        self._speaking = True
        self._quiet_frames = 0
        self._obsolete_until = asyncio.get_running_loop().time()
        _LOG.debug("voice turn started")
        self._output.interrupt()
        if self._on_input is not None:
            self._on_input(None, True)
        if self._io_stopped:  # 开口事件本身可能同步发现通话已经解绑。
            return
        if not self._discard_gap:
            preroll = b"".join(self._preroll)
            speech_bytes = self._hot_frames * INPUT_BYTES
            # 上限太短时先放入已确认的人声，不让开口前的静默占满。
            first_speech = len(preroll) - speech_bytes
            lead_in = min(first_speech, max(0, self._max_recording_bytes - speech_bytes))
            self._record = bytearray()
            self._record_audio(preroll[first_speech - lead_in:])
        self._preroll.clear()

    def _record_audio(self, pcm: bytes) -> None:
        if self._record is None:
            return
        remaining = self._max_recording_bytes - len(self._record)
        self._record.extend(pcm[:remaining])
        if len(self._record) == self._max_recording_bytes:
            self._finish_turn(ignore_tail=True)

    def _report_input(self, text: str | None = None) -> None:
        if self._failed or self._io_stopped:
            return
        idle = not self._input_active
        if idle:
            # 在 ASR 等待窗口内开始的请求也不能因识别结束而恢复旧回复。
            self._obsolete_until = asyncio.get_running_loop().time()
        if self._on_input is not None and (text or idle):
            self._on_input(text, False if idle else None)

    def _finish_turn(self, *, ignore_tail: bool = False) -> None:
        self._speaking = False
        self._ignoring_tail = ignore_tail
        self._discard_gap = False
        self._hot_frames = 0
        if not ignore_tail:
            self._quiet_frames = 0
        recording, self._record = self._record, None
        if recording and self._pending < _MAX_ASR_TURNS:
            pcm = bytes(recording)
            if ignore_tail:
                self._tail_recording = pcm
            self._pending += 1
            self._turns.put_nowait(pcm)
            _LOG.debug("voice turn queued for transcription; pcm_bytes=%d duration_limited=%s pending=%d",
                       len(recording), ignore_tail, self._pending)
        elif recording:
            _LOG.warning("voice turn dropped (asr backlog)")
        self._report_input()

    def _detect(self, frames: list[tuple[int, bytes]]) -> list[tuple[bool, bool, bytes]]:
        result = []
        for sequence, pcm in frames:
            missed = sequence != self._last_sequence + 1
            self._last_sequence = sequence
            if missed:
                self._vad.reset()
            result.append((missed, self._vad.probability(pcm) >= self._settings.vad_threshold, pcm))
        return result

    def _process(self, missed: bool, hot: bool, pcm: bytes) -> None:
        if self._ignoring_tail:
            # 不再记录尾音，但它仍属于原话轮；缺帧不视为静默。
            if missed:
                # 溢出丢样可能发生在本帧内部，这一帧不算可信静默。
                self._quiet_frames = 0
                return
            self._quiet_frames = 0 if hot else self._quiet_frames + 1
            if self._quiet_frames >= self._end_frames:
                self._ignoring_tail = False
                self._quiet_frames = 0
                self._tail_recording = None
                text, self._tail_text = self._tail_text, None
                self._report_input(text)
            return
        if missed:
            # 采集溢出不能拼接成假的完整话轮；静默后再接纳下一轮。
            if not self._discard_gap:
                _LOG.warning("voice capture gap; dropping incomplete turn")
            self._discard_gap = True
            self._record = None
            self._preroll.clear()
            self._hot_frames = self._quiet_frames = 0
            return  # 缺口可能在本帧内部；不能用它结束话轮或拼入后续录音。
        if not self._speaking:
            if self._discard_gap:
                self._quiet_frames = 0 if hot else self._quiet_frames + 1
                if self._quiet_frames >= self._end_frames:
                    self._discard_gap = False
            if not self._discard_gap:
                self._preroll.append(pcm)
            self._hot_frames = self._hot_frames + 1 if hot else 0
            if self._hot_frames >= self._settings.speech_frames:
                self._begin_turn()
            return
        self._quiet_frames = 0 if hot else self._quiet_frames + 1
        if self._quiet_frames >= self._end_frames:
            self._finish_turn()
            return
        self._record_audio(pcm)

    async def _watch_device(self, device: AudioDevice) -> None:
        clock = asyncio.get_running_loop().time
        check_at = clock()
        while not self._failed and not self._io_stopped:
            now = clock()
            if now >= check_at:
                try:
                    device.check_active()
                except Exception:
                    self._disable_device(device, "device stream stopped")
                    return
                check_at = now + _DEVICE_CHECK_SECONDS
            remaining = self._capture.stall_remaining()
            if remaining is not None and remaining <= 0:
                self._disable_device(device, "capture PCM stalled")
                return
            # 一个会话计时器兼顾设备状态和采集截止；未启用时也不睡过首帧宽限。
            await asyncio.sleep(min(check_at - clock(), _CAPTURE_STALL_SECONDS if remaining is None else remaining))

    async def _listen(self, device: AudioDevice) -> None:
        while not self._failed and not self._io_stopped:
            frames = self._capture.drain()
            if not frames:
                await asyncio.sleep(0.008)
                continue
            # 独立于端口回调与事件循环，且同一 VAD 实例仅串行执行一个批次。
            try:
                decisions = await asyncio.to_thread(self._detect, frames)
            except Exception:
                self._disable_device(device, "VAD inference failed")
                return
            for decision in decisions:
                if self._failed or self._io_stopped:
                    break
                self._process(*decision)

    def _disable_device(self, device: AudioDevice, reason: str) -> None:
        if self._failed or self._io_stopped:
            return
        # 不改走文字，不取消模型或工具；当前回复按已发进度失败收尾。
        self._failed = True
        self._speaking = self._ignoring_tail = False
        self._tail_recording = None
        self._tail_text = None
        self._record = None
        self._preroll.clear()
        self._capture.close()
        self._stop_transcription()
        self._output.fail()
        if not self._io_stopped and self._on_input is not None:
            self._on_input(None, False)
        _LOG.error("voice device disabled (%s)", reason)
        try:
            device.close()
        except Exception:
            _LOG.warning("voice device could not be closed")

    def _stop_transcription(self) -> None:
        if self._asr_worker is not None and not self._asr_worker.done() and not self._asr_worker.cancelling():
            self._asr_worker.cancel()
        while not self._turns.empty():
            self._turns.get_nowait()
            self._pending -= 1  # 已开始处理的话轮由转写协程的 finally 扣除
        if self._asr_task is not None and not self._asr_task.done() and not self._asr_task.cancelling():
            self._asr_task.cancel()  # 不等待在途请求的取消清理

    def _reap(self, task: asyncio.Task[str]) -> None:
        if not task.cancelled():
            task.exception()
        # 只在任务真正结束后释放（异常回溯可能持有 WAV），不清掉后继任务。
        if self._asr_task is task:
            self._asr_task = None

    async def _recognize(self, recording: bytes) -> str:
        if self._asr_task is not None and not self._asr_task.done():
            # 已超时但尚未退出的异常适配器不能阻塞或无限累积外部请求。
            return ""
        task = asyncio.create_task(self._asr.transcribe(_wav(recording)))
        self._asr_task = task
        task.add_done_callback(self._reap)
        done, _ = await asyncio.wait((task,), timeout=self._asr_timeout)
        if not done:
            task.cancel()
            _LOG.warning("voice transcription deadline reached")
            return ""
        return "" if task.cancelled() else task.result().strip()

    async def _transcribe(self) -> None:
        clock = asyncio.get_running_loop().time
        while not self._failed and not self._io_stopped:
            recording = await self._turns.get()
            started = clock()
            try:
                text = await self._recognize(recording)
            except Exception:
                _LOG.warning("voice transcription failed")
                text = ""
            finally:
                waiting_for_tail = recording is self._tail_recording
                if waiting_for_tail:
                    self._tail_recording = None
                # 等待下一轮入队期间不保留本轮 PCM。
                del recording
                self._pending -= 1
            if text and not self._failed and not self._io_stopped:
                _LOG.debug("voice transcription ready; chars=%d elapsed_ms=%.0f", len(text), (clock() - started) * 1000)
            if waiting_for_tail:
                # 只有这个封顶话轮的转写等静默；较早话轮仍按序交接。
                self._tail_text = text or None
            else:
                self._report_input(text or None)

    @asynccontextmanager
    async def connected(self, device: AudioDevice, on_input: VoiceInput, *,
                        capture_enabled: bool = True, wait_for_cleanup: bool = True) -> AsyncIterator[None]:
        if self._opened or self._closed:
            raise VoiceSetupError("voice_already_started")
        if device.input_format != AudioFormat(INPUT_RATE):
            raise VoiceSetupError("voice_input_device_format_mismatch")
        if device.output_format != AudioFormat(self._settings.output_sample_rate):
            raise VoiceSetupError("voice_output_device_format_mismatch")
        self._opened = True
        self._on_input = on_input
        try:
            device.start(self._capture.push, self._output.render)
            if capture_enabled:
                self.enable_capture()
            async with asyncio.TaskGroup() as group:
                watchdog = group.create_task(self._watch_device(device))
                listener = group.create_task(self._listen(device))
                self._asr_worker = group.create_task(self._transcribe())
                try:
                    yield
                finally:
                    self.stop_io()
                    watchdog.cancel()
                    listener.cancel()
        finally:
            self._on_input = None
            self._closed = True
            self._record = None
            self._preroll.clear()
            self._capture.close()
            # 关闭音频 I/O，再收尾未交付的输出与仍在清理的请求。
            try:
                device.close()
            finally:
                # 动态来电不等待旧网络适配器退出；任务保留自己的结果回收回调。
                await self._output.close(wait_for_cleanup=wait_for_cleanup)
                if wait_for_cleanup and self._asr_task is not None and not self._asr_task.done():
                    await asyncio.wait((self._asr_task,), timeout=1)
