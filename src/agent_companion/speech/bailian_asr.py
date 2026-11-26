"""百炼 ASR：文件和实时 PCM 共用一个 WebSocket 任务协议。"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import io
import struct
import wave

from websockets.exceptions import InvalidStatus, WebSocketException

from ..config import require_asr_audio_seconds, require_seconds
from .asr import MAX_AUDIO_BYTES, ASRCapabilities, AsrError, AsrPhase, AudioFile
from .bailian_ws import ASR_MODEL, MAX_EVENTS, BailianConnection, InvalidEvent, SpeechTask, TaskFailed


_SAMPLE_RATE = 16_000
_PACKET_BYTES = 3_200  # 16 kHz s16le 单声道的 100 ms；也是文件发送的单包上限。
_MAX_TRANSCRIPT_CHARS = 65_536
_OGG_SCAN_BUDGET = 2_048
_OGG_SCAN_CPU_SLICE_SECONDS = 0.004
_OGG_SCAN_CPU_PAUSE_SECONDS = 0.001


@dataclass(slots=True)
class _OggScanBudget:
    deadline: float
    work: int = 0
    _next_cpu_pause_at: float = field(init=False)

    def __post_init__(self) -> None:
        self._next_cpu_pause_at = asyncio.get_running_loop().time() + _OGG_SCAN_CPU_SLICE_SECONDS

    async def charge(self, units: int) -> None:
        self.work += units
        if self.work >= _OGG_SCAN_BUDGET:
            self.work = 0
            clock = asyncio.get_running_loop().time
            now = clock()
            if now >= self.deadline:
                raise TimeoutError
            # sleep(0) 只让出事件循环，扫描连续就绪时输入线程仍可能拿不到 GIL。
            # 到达时间片才用非零暂停，给采集线程运行机会，不给短文件增加固定等待。
            cpu_pause = now >= self._next_cpu_pause_at
            await asyncio.sleep(_OGG_SCAN_CPU_PAUSE_SECONDS if cpu_pause else 0)
            now = clock()
            if now >= self.deadline:
                raise TimeoutError
            if cpu_pause:
                self._next_cpu_pause_at = now + _OGG_SCAN_CPU_SLICE_SECONDS


@dataclass(slots=True)
class _OpusTags:
    # 签名由页扫描器检查。跳过 vendor 和评论，只保留跨页的长度字段（最多 4 字节）。
    _skip: int = 8
    _vendor_length_read: bool = False
    _comments_left: int | None = None
    _partial_word: bytearray = field(default_factory=bytearray)

    @property
    def complete(self) -> bool:
        return self._comments_left == 0 and self._skip == 0

    async def consume(self, data: bytes, start: int, end: int, budget: _OggScanBudget) -> None:
        # 必需字段读完后停止；其余是允许的可选数据或填充。
        while start < end and not self.complete:
            if self._skip:
                size = min(self._skip, end - start)
                start += size
                self._skip -= size
                if start == end or self.complete:
                    return
            await budget.charge(1)  # 评论可以很短，不能仅按页数限制工作量。
            if not self._partial_word and end - start >= 4:
                value = struct.unpack_from("<I", data, start)[0]
                start += 4
            else:
                size = min(4 - len(self._partial_word), end - start)
                self._partial_word.extend(data[start:start + size])
                start += size
                if len(self._partial_word) < 4:
                    return
                value = int.from_bytes(self._partial_word, "little")
                self._partial_word.clear()
            if not self._vendor_length_read:
                self._vendor_length_read = True
                self._skip = value
            elif self._comments_left is None:
                self._comments_left = value
            else:
                self._comments_left -= 1
                self._skip = value


async def _file_format(file: AudioFile, deadline: float) -> str:
    if file.extension == "wav":
        # AudioFile 已校验容器与帧，这里再检查模型要求的采样格式。
        with wave.open(io.BytesIO(file.data), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (_SAMPLE_RATE, 1, 2):
                return "wav"
        raise AsrError("asr_unsupported_audio_format")
    if file.extension == "ogg" and await _is_mono_ogg_opus(file.data, deadline):
        return "opus"
    raise AsrError("asr_unsupported_audio_format")


async def _is_mono_ogg_opus(data: bytes, deadline: float) -> bool:
    # AudioFile 已限制总大小。只扫页头、分段表和头包必需字段，不校验 CRC 或解码音频。
    end = len(data)
    if end < 27:
        return False
    stream_id, expected_sequence = struct.unpack_from("<II", data, 14)
    page = 0
    budget = _OggScanBudget(deadline)
    tags = _OpusTags()
    continued = tags_complete = has_audio = False
    while page + 27 <= end:
        if not data.startswith(b"OggS\x00", page):
            return False
        flags = data[page + 5]
        if (flags & ~7 or struct.unpack_from("<II", data, page + 14) != (stream_id, expected_sequence)):
            return False
        lacing_start = page + 27
        body = lacing_start + data[page + 26]
        if body > end:
            return False
        lacing = data[lacing_start:body]
        # 固定成本覆盖大量空页，分段数覆盖每页最多 255 个分段的处理。
        await budget.charge(16 + len(lacing))
        body_size = sum(lacing)
        next_page = body + body_size
        if next_page > end:
            return False

        if page == 0:
            # BOS 页必须独占且完成 OpusHead。原始采样率字段只是提示。
            if (flags != 2 or not lacing or lacing[-1] == 255 or any(size != 255 for size in lacing[:-1])
                    or body_size < 19 or not data.startswith(b"OpusHead", body)
                    or not 0 < data[body + 8] <= 15 or data[body + 9] != 1 or data[body + 18] != 0):
                return False
        else:
            # 255 表示包还未结束；即使中间遇到空页，下一有数据页也要延续该包。
            if flags & 2 or (lacing and bool(flags & 1) != continued):
                return False
            if lacing and not tags_complete:
                # 评论头可跨页，但必须在其最后一页末尾完成。
                if any(size != 255 for size in lacing[:-1]):
                    return False
                if not continued and (lacing[0] < 16 or not data.startswith(b"OpusTags", body)):
                    return False
                await tags.consume(data, body, next_page, budget)
                continued = lacing[-1] == 255
                if not continued and not tags.complete:
                    return False
                tags_complete = not continued
            else:
                for size in lacing:
                    if size < 255 and (size or continued):
                        has_audio = True
                    continued = size == 255
            if flags & 4:
                return has_audio and not continued and next_page == end

        page = next_page
        expected_sequence = (expected_sequence + 1) & 0xFFFFFFFF
    return False


async def _file_chunks(data: bytes, rate: int) -> AsyncIterator[bytes]:
    """有界上传速率；只在超过预算时等待，不照实时录音时钟回放文件。"""
    loop = asyncio.get_running_loop()
    next_send = loop.time()
    for offset in range(0, len(data), _PACKET_BYTES):
        delay = next_send - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        chunk = data[offset:offset + _PACKET_BYTES]
        next_send = loop.time() + len(chunk) / rate
        yield chunk


@dataclass(slots=True)
class _Upload:
    caller: asyncio.Task
    initial_cancellations: int
    finish_by: float | None = None
    stopped: bool = False
    sender: asyncio.Task[None] | None = None
    detached: bool = False  # 调用已退出，槽位仅为正在退出的发送者保留。

    @classmethod
    def new(cls) -> "_Upload":
        caller = asyncio.current_task()
        assert caller is not None
        return cls(caller, caller.cancelling())

    def can_send(self) -> bool:
        # 调用者已请求取消时，即使发送协程先被调度，也不能再发。
        return not self.stopped and self.caller.cancelling() <= self.initial_cancellations


def _reap(worker: asyncio.Task) -> None:
    if not worker.cancelled():
        worker.exception()


@contextmanager
def _asr_errors(phase: AsrPhase) -> Iterator[None]:
    """将网络或协议异常转成固定诊断类别，不保留异常原文。"""
    try:
        yield
    except TimeoutError:
        raise AsrError("asr_timeout", phase=phase) from None
    except InvalidStatus as exc:
        raise AsrError(f"asr_http_{exc.response.status_code}", phase=phase) from None
    except (OSError, WebSocketException):
        raise AsrError("asr_transport", phase=phase) from None
    except InvalidEvent:
        raise AsrError("asr_invalid_response", phase=phase) from None
    except TaskFailed:
        raise AsrError("asr_business_error", phase=phase) from None


class BailianASR:
    def __init__(self, connection: BailianConnection, *, timeout: float, max_audio_seconds: float,
                 model: str = ASR_MODEL, file_upload_bytes_per_second: int = 128_000) -> None:
        if model != ASR_MODEL:
            raise ValueError("unsupported Bailian ASR model")
        require_seconds(timeout, "ASR timeout")
        require_asr_audio_seconds(max_audio_seconds)
        if type(file_upload_bytes_per_second) is not int or not 1 <= file_upload_bytes_per_second <= MAX_AUDIO_BYTES:
            raise ValueError("invalid Bailian ASR file upload rate")
        self._connection = connection
        self._timeout = timeout
        self._max_audio_seconds = max_audio_seconds
        self._model = model
        self._file_rate = file_upload_bytes_per_second
        # 多个私聊共用实例；不要让并行的 Ogg 扫描填满彼此给采集线程留下的暂停。
        self._ogg_preflight_lock = asyncio.Lock()
        self._live_upload: _Upload | None = None

    def capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(file=self, streaming=self)

    async def transcribe(self, file: AudioFile) -> str:
        clock = asyncio.get_running_loop().time
        deadline = clock() + self._timeout
        try:
            async with asyncio.timeout_at(deadline):
                if file.duration_seconds is not None and file.duration_seconds > self._max_audio_seconds:
                    raise AsrError("invalid_audio_duration")
                if file.extension == "ogg":
                    async with self._ogg_preflight_lock:
                        caller = asyncio.current_task()
                        if caller is not None and caller.cancelling():
                            # 锁空闲时 acquire 不挂起；有待投递的取消要在扫描前处理。
                            await asyncio.sleep(0)
                        if clock() >= deadline:
                            raise TimeoutError
                        audio_format = await _file_format(file, deadline)
                else:
                    audio_format = await _file_format(file, deadline)
                # 预检锁不跨越网络阶段，其他 Ogg 可在本次上传期间预检。
                # 绝对截止已到时，即使 timeout 的取消回调还没执行，也不拨号。
                if clock() >= deadline:
                    raise TimeoutError
                text = await self._transcribe(_file_chunks(file.data, self._file_rate), audio_format, _Upload.new())
                if clock() >= deadline:
                    raise TimeoutError
                return self._required_text(text)
        except TimeoutError:
            raise AsrError("asr_timeout") from None

    @staticmethod
    def _required_text(text: str) -> str:
        # 到这里已确认本地 finish-task 写入成功；空结果不能绕过发送收尾。
        if not text:
            raise AsrError("asr_empty_transcript", phase="finish")
        return text

    async def transcribe_stream(self, frames: AsyncIterator[bytes]) -> str:
        previous = self._live_upload
        # 已退出的旧发送者不再占槽，即使它的完成回调还排在事件循环中。
        if previous is not None and not (
            previous.detached and previous.sender is not None and previous.sender.done()
        ):
            raise AsrError("asr_busy")
        upload = self._live_upload = _Upload.new()
        try:
            return self._required_text(await self._transcribe(frames, "pcm", upload))
        finally:
            if upload.sender is not None and not upload.sender.done():
                # 连接已经释放；输入若吞掉取消，仍占用唯一的实时槽，不能累积孤儿。
                upload.detached = True
                upload.sender.add_done_callback(self._release_live_upload)
            else:
                self._live_upload = None

    def _release_live_upload(self, sender: asyncio.Task[None]) -> None:
        current = self._live_upload
        if current is not None and current.sender is sender:
            self._live_upload = None

    async def _transcribe(self, frames: AsyncIterator[bytes], audio_format: str, upload: _Upload) -> str:
        with _asr_errors("connect"):
            async with self._connection._open(self._timeout) as socket:
                task = SpeechTask(socket)
                with _asr_errors("start"):
                    await task.start("asr", "recognition", self._model, {
                        "format": audio_format, "sample_rate": _SAMPLE_RATE,
                        "intermediate_result_enabled": False,
                    }, self._timeout)
                return await self._exchange(task, frames, audio_format, upload)

    async def _exchange(self, task: SpeechTask, frames: AsyncIterator[bytes], audio_format: str,
                        upload: _Upload) -> str:
        sender = upload.sender = asyncio.create_task(self._send_audio(task, frames, audio_format, upload))
        receiver = asyncio.create_task(self._receive_text(task, upload))
        for worker in (sender, receiver):
            worker.add_done_callback(_reap)
        try:
            done, _ = await asyncio.wait((sender, receiver), return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                # 服务端可先失败；成功响应也可能早于 finish-task 的本地 send 返回。
                text = receiver.result()
                await asyncio.wait((sender,))
                sender.result()
                return text
            sender.result()
            assert upload.finish_by is not None
            with _asr_errors("finish"):
                async with asyncio.timeout_at(upload.finish_by):
                    return await asyncio.shield(receiver)
        finally:
            upload.stopped = True
            for worker in (sender, receiver):
                if not worker.done():
                    worker.cancel()
            # 不等待输入源响应取消：让 _transcribe 立即退出连接上下文。

    async def _send_audio(self, task: SpeechTask, frames: AsyncIterator[bytes], audio_format: str,
                          upload: _Upload) -> None:
        length = 0
        while upload.can_send():
            try:
                chunk = await anext(frames)
            except StopAsyncIteration:
                break
            except Exception:
                raise AsrError("asr_input_failed", phase="send") from None
            if not isinstance(chunk, bytes) or not chunk or (audio_format == "pcm" and len(chunk) % 2):
                raise AsrError("invalid_audio_chunk", phase="send")
            length += len(chunk)
            if length > MAX_AUDIO_BYTES:
                raise AsrError("invalid_audio_size", phase="send")
            for offset in range(0, len(chunk), _PACKET_BYTES):
                if not upload.can_send():
                    return
                with _asr_errors("send"):
                    async with asyncio.timeout(self._timeout):
                        await task.socket.send(chunk[offset:offset + _PACKET_BYTES])
                await asyncio.sleep(0)  # send 可能同步完成，让接收协程有机会处理服务端失败。
        if not upload.can_send():
            return
        if not length:
            raise AsrError("invalid_audio_size", phase="send")
        # EOF 和入站消息可以同时就绪。先把执行权交给读端：websockets.recv 会连续
        # 取出已经可以读取的完整消息，不能把缓冲中的提前终止归到还未发送的 finish。
        await asyncio.sleep(0)
        if not upload.can_send():
            return
        clock = asyncio.get_running_loop().time
        upload.finish_by = clock() + self._timeout
        with _asr_errors("finish"):
            async with asyncio.timeout_at(upload.finish_by):
                # 单个字符串帧写入传输后才等待排空，所以读端可先收到正常 task-finished；
                # 但 _exchange 仍要确认本地 send 成功，才会交付文本。
                await task.send("finish-task", {"input": {}})
            if clock() >= upload.finish_by:
                raise TimeoutError

    async def _receive_text(self, task: SpeechTask, upload: _Upload) -> str:
        try:
            with _asr_errors("receive"):
                return await self._collect_text(task, upload)
        finally:
            # 在读取协程结束前关闭发送闸门，不依赖协调协程先得到调度。
            upload.stopped = True

    async def _collect_text(self, task: SpeechTask, upload: _Upload) -> str:
        sentences: dict[int, str] = {}
        unfinished: set[int] = set()
        text_chars = 0
        clock = asyncio.get_running_loop().time
        for _ in range(MAX_EVENTS):
            event, payload = task.parse(await task.socket.recv())
            if event == "task-finished":
                if upload.finish_by is None:
                    raise InvalidEvent
                if clock() >= upload.finish_by:
                    raise AsrError("asr_timeout", phase="finish")
                if unfinished or (sentences and max(sentences) != len(sentences)):
                    raise InvalidEvent
                return "\n".join(value for _, value in sorted(sentences.items()) if value)
            if event != "result-generated":
                raise InvalidEvent
            output = payload.get("output")
            sentence = output.get("sentence") if isinstance(output, dict) else None
            if not isinstance(sentence, dict):
                raise InvalidEvent
            if sentence.get("heartbeat") is True:
                continue
            index, text = sentence.get("sentence_id"), sentence.get("text")
            final = sentence.get("sentence_end")
            if type(final) is not bool or type(index) is not int or not 1 <= index <= MAX_EVENTS:
                raise InvalidEvent
            if text is None and not final and sentence.get("sentence_begin") is True:
                continue  # 仅有起句标记，不代表已有实质识别内容。
            if not isinstance(text, str):
                raise InvalidEvent
            text = text.strip()
            if not final:
                if text:
                    previous = sentences.get(index)
                    if previous is None:
                        unfinished.add(index)
                    elif not previous:
                        raise InvalidEvent  # 已空结句的序号不能再出现实质内容。
                continue
            if index in unfinished:
                if not text:
                    raise InvalidEvent  # 空最终结果不能掩盖已见的实质中间结果。
                unfinished.remove(index)
            if index in sentences:
                if sentences[index] != text:
                    raise InvalidEvent
                continue
            text_chars += len(text) + 1
            if text_chars > _MAX_TRANSCRIPT_CHARS:
                raise AsrError("asr_response_limit", phase="receive")
            sentences[index] = text
        raise AsrError("asr_response_limit", phase="receive")
