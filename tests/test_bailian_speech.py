import asyncio
from contextlib import aclosing, asynccontextmanager
import io
import json
import struct
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID
import wave

import httpx
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from websockets.http11 import Response

from agent_companion.speech import bailian_asr
from agent_companion.speech.asr import AsrError, AudioFile
from agent_companion.speech.bailian_asr import BailianASR
from agent_companion.speech.bailian_tts import BailianTTS
from agent_companion.speech.bailian_ws import ASR_MODEL, BailianConnection, _DirectConnect
from agent_companion.speech.minimax import MiniMaxASR
from agent_companion.speech.tts import TtsError


def _wav(*, rate=16_000, channels=1, width=2, frames=160):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        output.writeframes(bytes(frames * channels * width))
    return buffer.getvalue()


def _ogg_page(packet, *, flag=2, lacing=None, sequence=0, granule=0, stream_id=1):
    lacing = bytes([len(packet)]) if lacing is None else bytes(lacing)
    header = struct.pack("<4sBBqIIIB", b"OggS", 0, flag, granule, stream_id, sequence, 0, len(lacing))
    page = header + lacing + packet
    crc = 0
    for octet in page:
        crc ^= octet << 24
        for _ in range(8):
            crc = ((crc << 1) ^ (0x04C11DB7 if crc & 0x80000000 else 0)) & 0xFFFFFFFF
    return page[:22] + struct.pack("<I", crc) + page[26:]


def _opus_pages(channels=1):
    # 合成流：有正确页校验、OpusHead、OpusTags 和一帧 20 ms 的静音 Opus 包。
    header = struct.pack("<8sBBHIhB", b"OpusHead", 1, channels, 312, 48_000, 0, 0)
    tags = b"OpusTags" + struct.pack("<I4sI", 4, b"test", 0)
    silence = b"\xf8\xff\xfe" if channels == 1 else b"\xfc\xff\xfe"
    return (_ogg_page(header), _ogg_page(tags, flag=0, sequence=1),
            _ogg_page(silence, flag=4, sequence=2, granule=960))


def _opus(channels=1):
    return b"".join(_opus_pages(channels))


def _opus_tag_pages(tags, *, page_size=255):
    for sequence, start in enumerate(range(0, len(tags) + 1, page_size), start=1):
        chunk = tags[start:start + page_size]
        full, tail = divmod(len(chunk), 255)
        continued = len(chunk) == page_size
        lacing = [255] * full + ([] if continued else [tail])
        yield _ogg_page(chunk, flag=int(sequence > 1), lacing=lacing, sequence=sequence,
                        granule=-1 if continued else 0)


def _opus_with_continued_packets(vendor_size):
    head, _, _ = _opus_pages()
    vendor = b"v" * vendor_size
    comment = b"TEST=" + b"x" * 266
    tags = (b"OpusTags" + struct.pack("<I", len(vendor)) + vendor
            + struct.pack("<II", 1, len(comment)) + comment + b"\x00padding")
    tag_pages = list(_opus_tag_pages(tags))
    # Opus code 3：一个 20 ms 静音帧，添加 251 个填充字节使音频包跨页。
    silence = b"\xfb\x41\xfb\xff\xfe" + bytes(251)
    sequence = len(tag_pages) + 1
    return (head + b"".join(tag_pages)
            + _ogg_page(silence[:255], flag=0, sequence=sequence, granule=-1)
            + _ogg_page(silence[255:], flag=5, sequence=sequence + 1, granule=960))


def _opus_with_empty_pages(count):
    head, tags, _ = _opus_pages()
    pages = (_ogg_page(b"", flag=0, lacing=[], sequence=idx + 2, granule=-1) for idx in range(count))
    return head + tags + b"".join(pages) + _ogg_page(b"\xf8\xff\xfe", flag=4, sequence=count + 2, granule=960)


def _opus_with_dense_audio_pages(count):
    head, tags, _ = _opus_pages()
    packets = b"\xf8\xff\xfe" * 255
    pages = (_ogg_page(packets, flag=4 if idx + 1 == count else 0, lacing=[3] * 255,
                       sequence=idx + 2, granule=(idx + 1) * 255 * 960) for idx in range(count))
    return head + tags + b"".join(pages)


async def _frames(*pcm):
    for chunk in pcm:
        yield chunk


@asynccontextmanager
async def _using_socket(socket, *, timeout=2):
    with patch.object(BailianConnection, "_open", return_value=socket):
        async with asyncio.timeout(timeout):
            yield


class _Socket:
    def __init__(self, *, started=True, on_send=None):
        self.task_id = None
        self.messages = []
        self.incoming = asyncio.Queue()
        self.received = asyncio.Queue()
        self.opened = self.closed = False
        self.started = started
        self.finish = asyncio.Event()
        self.audio = asyncio.Event()
        self.on_send = on_send

    async def __aenter__(self):
        self.opened = True
        return self

    async def __aexit__(self, *_):
        self.closed = True

    def event(self, name, payload=None, *, task_id=None):
        self.incoming.put_nowait(json.dumps({
            "header": {"task_id": task_id or self.task_id, "event": name}, "payload": payload or {},
        }))

    def sentence(self, index, text, *, final=True, heartbeat=False):
        self.event("result-generated", {"output": {"sentence": {
            "sentence_id": index, "text": text, "sentence_end": final, "heartbeat": heartbeat,
        }}})

    async def send(self, raw):
        if isinstance(raw, bytes):
            self.messages.append(raw)
            self.audio.set()
        else:
            msg = json.loads(raw)
            self.messages.append(msg)
            header = msg["header"]
            if header["action"] == "run-task":
                self.task_id = header["task_id"]
                if self.started:
                    self.event("task-started")
            elif header["action"] == "finish-task":
                self.finish.set()
        if self.on_send is not None:
            await self.on_send(self, raw)

    async def recv(self):
        raw = await self.incoming.get()
        self.received.put_nowait(raw)
        if isinstance(raw, Exception):
            raise raw
        return raw

    def actions(self):
        return [msg["header"]["action"] for msg in self.messages if isinstance(msg, dict)]


class BailianConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_host_and_private_handshake_does_not_follow_redirects(self):
        connection = BailianConnection("test-only", "llm-example", "ap-southeast-1")
        connect = connection._open(1.0)
        self.assertEqual(connect.uri, "wss://llm-example.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/inference")
        self.assertIsNone(connect.proxy)
        self.assertTrue(connect.logger.disabled)
        self.assertNotIn("test-only", repr(connection))
        for target in ("wss://other.example/upload", "/redirect-within-host"):
            with self.subTest(target=target):
                redirect = InvalidStatus(Response(302, "Found", Headers({"Location": target})))
                socket = Mock(handshake=AsyncMock(side_effect=redirect))
                with patch.object(_DirectConnect, "create_connection", new=AsyncMock(return_value=socket)) as create:
                    with self.assertRaises(InvalidStatus):
                        await connection._open(1.0)
                create.assert_awaited_once_with()
                socket.transport.abort.assert_called_once_with()
                socket.handshake.assert_awaited_once()
                self.assertEqual(socket.handshake.await_args.args[0], {"Authorization": "Bearer test-only"})

    async def test_bad_connection_settings_and_capability_queries_do_not_dial(self):
        for workspace in ("bad.example", "bad:443", "bad@host", "a/b", "a\n", "-bad", "a" * 64):
            with self.subTest(workspace=workspace), self.assertRaisesRegex(ValueError, "invalid Bailian workspace_id"):
                BailianConnection("test-only", workspace)
        with self.assertRaisesRegex(ValueError, "invalid Bailian region"):
            BailianConnection("test-only", "workspace", "other.example")
        with self.assertRaisesRegex(ValueError, "invalid Bailian API key"):
            BailianConnection("example\r\nheader", "workspace")
        connection = BailianConnection("test-only", "workspace")
        with patch.object(BailianConnection, "_open", side_effect=AssertionError("unexpected dial")):
            async with httpx.AsyncClient() as client:
                minimax = MiniMaxASR(client, region="cn", timeout=1, max_audio_seconds=10)
                self.assertIs(minimax.capabilities().file, minimax)
                self.assertIsNone(minimax.capabilities().streaming)
            bailian = BailianASR(connection, timeout=1, max_audio_seconds=10)
            self.assertIs(bailian.capabilities().file, bailian)
            self.assertIs(bailian.capabilities().streaming, bailian)
            with self.assertRaisesRegex(ValueError, "unsupported Bailian ASR model"):
                BailianASR(connection, timeout=1, max_audio_seconds=10, model="unsupported")


class BailianASRTests(unittest.IsolatedAsyncioTestCase):
    def asr(self, *, timeout=1, max_audio_seconds=10, **options):
        return BailianASR(BailianConnection("test-only", "workspace"), timeout=timeout,
                          max_audio_seconds=max_audio_seconds, **options)

    async def test_file_wav_and_ogg_opus_use_same_task_contract(self):
        async def respond(socket, raw):
            if not isinstance(raw, bytes) and socket.finish.is_set():
                socket.sentence(2, " 测试第二句 ")
                socket.sentence(1, "不要拼接", final=False)
                socket.sentence(0, "心跳", heartbeat=True)
                socket.sentence(1, "测试第一句")
                socket.sentence(1, "测试第一句")
                socket.event("task-finished")

        cases = [("wav", _wav(frames=1_801), "wav"), ("short_opus", _opus(), "opus")]
        # 分别跨页：评论长度字段、评论数量字段、vendor 内容；评论正文和音频包也跨页。
        cases.extend((f"continued_{size}", _opus_with_continued_packets(size), "opus")
                     for size in (237, 240, 245))
        for case, data, expected in cases:
            with self.subTest(case=case):
                socket = _Socket(on_send=respond)
                with patch.object(BailianConnection, "_open", return_value=socket):
                    self.assertEqual(await self.asr().transcribe(AudioFile(data)), "测试第一句\n测试第二句")
                self.assertTrue(socket.closed)
                run = socket.messages[0]["payload"]
                self.assertEqual((run["task"], run["function"], run["model"]), ("asr", "recognition", ASR_MODEL))
                self.assertEqual(run["parameters"], {
                    "format": expected, "sample_rate": 16_000, "intermediate_result_enabled": False,
                })
                chunks = socket.messages[1:-1]
                self.assertTrue(all(0 < len(chunk) <= 3_200 for chunk in chunks))
                self.assertEqual(b"".join(chunks), data)
                self.assertEqual(socket.actions(), ["run-task", "finish-task"])
                self.assertEqual(socket.messages[-1]["header"]["task_id"], socket.task_id)

    async def test_incomplete_ogg_opus_rejected_before_dialing(self):
        head, tags, last = _opus_pages()
        silence = b"\xf8\xff\xfe"
        # 评论长度跨页但只提供 3 字节；不能读到后续音频页凑齐长度。
        cut_comment_length = b"OpusTags" + struct.pack("<I", 237) + b"v" * 237 + struct.pack("<I", 1) + bytes(3)
        cut_tag_pages = list(_opus_tag_pages(cut_comment_length))
        cases = {
            "only_opus_head": head,
            "truncated_page_header": head + tags + last[:26],
            "truncated_lacing": head + tags + last[:27],
            "truncated_audio": head + tags + last[:-1],
            "no_audio": head + tags + _ogg_page(b"", flag=4, lacing=[], sequence=2),
            "missing_eos": head + tags + _ogg_page(silence, flag=0, sequence=2, granule=960),
            "unfinished_audio_at_eos": head + tags + _ogg_page(bytes(255), flag=4, sequence=2),
            "missing_tags": head + _ogg_page(silence, flag=4, sequence=1, granule=960),
            "truncated_tags": head + _ogg_page(b"OpusTags" + struct.pack("<II", 999, 0), flag=0, sequence=1) + last,
            "missing_comment_length": (head
                                       + _ogg_page(b"OpusTags" + struct.pack("<II", 0, 1), flag=0, sequence=1) + last),
            "truncated_comment": (head
                                  + _ogg_page(b"OpusTags" + struct.pack("<III", 0, 1, 9) + b"A=B", flag=0, sequence=1) + last),
            "truncated_cross_page_comment_length": (head + b"".join(cut_tag_pages)
                                                    + _ogg_page(silence, flag=4, sequence=3, granule=960)),
            "different_stream": head + tags + _ogg_page(silence, flag=4, sequence=2, granule=960, stream_id=2),
            "missing_page": head + tags + _ogg_page(silence, flag=4, sequence=3, granule=960),
            "missing_continuation": (head + tags + _ogg_page(bytes(255), flag=0, sequence=2, granule=-1)
                                     + _ogg_page(silence, flag=4, sequence=3, granule=960)),
            "data_after_eos": head + tags + last + head,
        }
        for case, data in cases.items():
            with self.subTest(case=case), patch.object(BailianConnection, "_open") as dial:
                with self.assertRaisesRegex(AsrError, "asr_unsupported_audio_format"):
                    await self.asr().transcribe(AudioFile(data))
                dial.assert_not_called()

    async def test_ogg_preflight_yields_for_empty_pages_dense_audio_and_comment_fields(self):
        async def respond(socket, raw):
            if not isinstance(raw, bytes) and socket.finish.is_set():
                socket.sentence(1, "识别成功")
                socket.event("task-finished")

        head, _, last = _opus_pages()
        comment = struct.pack("<I", 3) + b"A=B"
        tags = b"OpusTags" + struct.pack("<II", 0, 4_096) + comment * 4_096
        dense_tags = head + b"".join(_opus_tag_pages(tags, page_size=65_025)) + last
        cases = {"empty_pages": _opus_with_empty_pages(1024), "dense_audio": _opus_with_dense_audio_pages(16),
                 "dense_tags": dense_tags}
        for case, blob in cases.items():
            socket = _Socket(on_send=respond)
            with self.subTest(case=case), patch.object(BailianConnection, "_open", return_value=socket) as dial:
                asr = self.asr(timeout=3, file_upload_bytes_per_second=50_000_000)
                task = asyncio.create_task(asr.transcribe(AudioFile(blob)))
                try:
                    # 旁路任务至少在预检还没结束、尚未拨号时已经被运行。
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                    dial.assert_not_called()
                    self.assertEqual(await task, "识别成功")
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_ogg_preflight_periodically_pauses_and_cancel_does_not_dial(self):
        audio = AudioFile(_opus_with_empty_pages(1024))
        loop = asyncio.get_running_loop()
        clock, sleep = loop.time, asyncio.sleep
        offset = 0.0
        pauses = []

        async def observe_pause(delay):
            nonlocal offset
            if delay > 0:
                pauses.append(delay)
                if len(pauses) == 2:
                    # 让真实的非零 sleep 先挂起，再由事件循环取消它。
                    loop.call_soon(task.cancel)
            else:
                # 不依赖这批页恰好耗时多久；让下一检查点超过时间片。
                offset += 0.02
            await sleep(delay)

        with (patch.object(loop, "time", new=lambda: clock() + offset),
              patch.object(loop, "slow_callback_duration", new=float("inf")),
              patch.object(bailian_asr.asyncio, "sleep", new=observe_pause),
              patch.object(BailianConnection, "_open", side_effect=AssertionError("unexpected dial")) as dial):
            task = asyncio.create_task(self.asr(timeout=1).transcribe(audio))
            with self.assertRaises(asyncio.CancelledError):
                await task
            dial.assert_not_called()
        self.assertEqual(pauses, [0.001, 0.001])

    async def test_small_ogg_preflight_does_not_sleep(self):
        deadline = asyncio.get_running_loop().time() + 1
        with patch.object(bailian_asr.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            self.assertEqual(await bailian_asr._file_format(AudioFile(_opus()), deadline), "opus")
        sleep.assert_not_awaited()

    async def test_ogg_preflight_cancel_or_deadline_does_not_dial(self):
        audio = AudioFile(_opus_with_empty_pages(1024))
        loop = asyncio.get_running_loop()
        clock = loop.time
        for interrupt in ("cancel", "deadline"):
            offset = 0.0
            # 虚拟时钟前跳不代表慢回调。
            with (self.subTest(interrupt=interrupt), patch.object(loop, "time", new=lambda: clock() + offset),
                  patch.object(loop, "slow_callback_duration", new=float("inf")),
                  patch.object(BailianConnection, "_open") as dial):
                task = asyncio.create_task(self.asr(timeout=1).transcribe(audio))
                try:
                    await asyncio.sleep(0)
                    dial.assert_not_called()
                    if interrupt == "cancel":
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        offset = 2.0
                        with self.assertRaisesRegex(AsrError, "^asr_timeout$"):
                            await task
                    dial.assert_not_called()
                finally:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_queued_ogg_cancel_or_deadline_does_not_scan_or_dial(self):
        loop = asyncio.get_running_loop()
        clock = loop.time
        audio = AudioFile(_opus())
        for mode in ("cancel", "both_time_out", "expired_when_acquired"):
            offset = 0.0
            entered, release = asyncio.Event(), asyncio.Event()
            asr = self.asr()

            async def hold_scan(data, deadline):
                nonlocal offset
                if entered.is_set():
                    self.fail("queued Ogg started scanning")
                entered.set()
                await release.wait()
                if mode == "expired_when_acquired":
                    # 释放锁后，等待者先就绪，超时回调还未执行。
                    offset = 2.0
                return False

            with (self.subTest(mode=mode), patch.object(loop, "time", new=lambda: clock() + offset),
                  patch.object(loop, "slow_callback_duration", new=float("inf")),
                  patch.object(bailian_asr, "_is_mono_ogg_opus", side_effect=hold_scan) as scan,
                  patch.object(BailianConnection, "_open", side_effect=AssertionError("unexpected dial")) as dial):
                holder = asyncio.create_task(asr.transcribe(audio))
                tasks = [holder]
                try:
                    await asyncio.wait_for(entered.wait(), 1)
                    waiter = asyncio.create_task(asr.transcribe(audio))
                    tasks.append(waiter)
                    await asyncio.sleep(0)
                    self.assertFalse(waiter.done())
                    self.assertEqual(scan.await_count, 1)
                    if mode == "cancel":
                        waiter.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await waiter
                        self.assertFalse(holder.done())
                        release.set()
                    else:
                        if mode == "both_time_out":
                            offset = 2.0
                        else:
                            release.set()
                        with self.assertRaisesRegex(AsrError, "^asr_timeout$"):
                            await waiter
                    holder_error = "asr_timeout" if mode == "both_time_out" else "asr_unsupported_audio_format"
                    with self.assertRaisesRegex(AsrError, f"^{holder_error}$"):
                        await holder
                    self.assertEqual(scan.await_count, 1)
                    dial.assert_not_called()
                    self.assertFalse(asr._ogg_preflight_lock.locked())
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

    async def test_ogg_lock_acquisition_delivers_pending_cancel_before_scanning(self):
        asr = self.asr()
        audio = AudioFile(_opus())

        async def already_cancelled():
            asyncio.current_task().cancel()
            return await asr.transcribe(audio)

        with (patch.object(bailian_asr, "_is_mono_ogg_opus", return_value=True) as scan,
              patch.object(BailianConnection, "_open", side_effect=AssertionError("unexpected dial")) as dial):
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.create_task(already_cancelled())
            scan.assert_not_awaited()
            dial.assert_not_called()
        self.assertFalse(asr._ogg_preflight_lock.locked())

    async def test_ogg_preflight_does_not_lock_network_wav_or_live_stream(self):
        asr = self.asr(timeout=3)
        first_audio = AudioFile(_opus())
        second_audio = AudioFile(_opus_with_continued_packets(237))
        entered, release = asyncio.Event(), asyncio.Event()
        original_scan = bailian_asr._is_mono_ogg_opus

        async def hold_first_scan(data, deadline):
            if data is first_audio.data:
                entered.set()
                await release.wait()
            return await original_scan(data, deadline)

        async def respond(socket, raw):
            if not isinstance(raw, bytes) and socket.finish.is_set():
                socket.sentence(1, "旁路完成")
                socket.event("task-finished")

        allow_first_upload = asyncio.Event()

        async def hold_first_upload(socket, raw):
            if isinstance(raw, bytes):
                await allow_first_upload.wait()
            elif socket.finish.is_set():
                socket.sentence(1, "第一条完成")
                socket.event("task-finished")

        wav_socket, live_socket, second_socket = (_Socket(on_send=respond) for _ in range(3))
        first_socket = _Socket(on_send=hold_first_upload)
        sockets = [wav_socket, live_socket, first_socket, second_socket]
        with (patch.object(bailian_asr, "_is_mono_ogg_opus", side_effect=hold_first_scan) as scan,
              patch.object(BailianConnection, "_open", side_effect=sockets) as dial):
            first = asyncio.create_task(asr.transcribe(first_audio))
            try:
                async with asyncio.timeout(2):
                    await entered.wait()
                    self.assertEqual(await asr.transcribe(AudioFile(_wav())), "旁路完成")
                    self.assertEqual(await asr.transcribe_stream(_frames(b"\x00\x00")), "旁路完成")
                    self.assertEqual(scan.await_count, 1)
                    release.set()
                    await first_socket.audio.wait()
                    self.assertFalse(first_socket.finish.is_set())
                    self.assertEqual(await asr.transcribe(second_audio), "旁路完成")
                    self.assertEqual(scan.await_count, 2)
                    self.assertTrue(second_socket.finish.is_set())
                    self.assertFalse(first.done())
                    allow_first_upload.set()
                    self.assertEqual(await first, "第一条完成")
                self.assertEqual(dial.call_count, 4)
            finally:
                if not first.done():
                    first.cancel()
                await asyncio.gather(first, return_exceptions=True)

    async def test_file_deadline_reached_before_timeout_callback_does_not_dial(self):
        loop = asyncio.get_running_loop()
        clock = loop.time
        file_format = bailian_asr._file_format
        offset = 0.0

        async def finish_late(file, deadline):
            nonlocal offset
            result = await file_format(file, deadline)
            offset = 2.0
            return result

        with (patch.object(loop, "time", new=lambda: clock() + offset),
              patch.object(bailian_asr, "_file_format", side_effect=finish_late),
              patch.object(BailianConnection, "_open") as dial):
            with self.assertRaisesRegex(AsrError, "^asr_timeout$"):
                await self.asr(timeout=1).transcribe(AudioFile(_opus()))
            dial.assert_not_called()

    async def test_unsupported_formats_and_wav_size_rejected_before_dialing(self):
        vorbis = _ogg_page(b"\x01vorbis" + bytes(12))
        broken_opus = _ogg_page(b"OpusHead\x01\x01", lacing=[19])
        for blob in (_wav(rate=8_000), _wav(channels=2), _wav(width=1), _opus(2), vorbis, broken_opus):
            with self.subTest(size=len(blob)), patch.object(BailianConnection, "_open") as dial:
                with self.assertRaisesRegex(AsrError, "asr_unsupported_audio_format"):
                    await self.asr().transcribe(AudioFile(blob))
                dial.assert_not_called()
        with patch.object(BailianConnection, "_open") as dial:
            with self.assertRaisesRegex(AsrError, "invalid_audio_duration"):
                await self.asr(max_audio_seconds=0.001).transcribe(AudioFile(_wav()))
            dial.assert_not_called()

    async def test_stream_sends_and_receives_before_end_of_capture(self):
        frame = b"\x01\x02" * 512
        local_endpoint = asyncio.Event()
        socket = _Socket()

        async def capture():
            yield frame
            await local_endpoint.wait()
            yield frame

        async with _using_socket(socket):
            call = asyncio.create_task(self.asr(timeout=0.1).transcribe_stream(capture()))
            try:
                await socket.audio.wait()
                socket.sentence(1, "前半句")
                await socket.received.get()  # task-started
                self.assertIn("result-generated", await socket.received.get())
                self.assertFalse(socket.finish.is_set())
                self.assertFalse(call.done())
                # 时间超过文件模式的总时限，实时输入仍然有效。
                await asyncio.sleep(0.12)
                local_endpoint.set()
                await socket.finish.wait()
                socket.sentence(2, "后半句")
                socket.event("task-finished")
                self.assertEqual(await call, "前半句\n后半句")
            finally:
                if not call.done():
                    call.cancel()
                    await asyncio.gather(call, return_exceptions=True)
        self.assertEqual(socket.messages[0]["payload"]["parameters"]["format"], "pcm")
        self.assertEqual(socket.messages[1:-1], [frame, frame])
        self.assertTrue(socket.closed)

    async def test_early_terminal_cancels_waiting_capture_and_never_delivers(self):
        for terminal, error in (("task-finished", "asr_invalid_response"), ("task-failed", "asr_business_error")):
            with self.subTest(terminal=terminal):
                source_started = asyncio.Event()
                source_ended = asyncio.Event()

                async def capture():
                    yield b"\x00\x00"
                    try:
                        source_started.set()
                        await asyncio.Future()
                    finally:
                        source_ended.set()

                socket = _Socket()
                async with _using_socket(socket):
                    call = asyncio.create_task(self.asr().transcribe_stream(capture()))
                    await source_started.wait()
                    socket.sentence(1, "不得交付的残文")
                    socket.event(terminal)
                    with self.assertRaisesRegex(AsrError, error):
                        await call
                self.assertTrue(source_ended.is_set())
                self.assertTrue(socket.closed)
                self.assertFalse(socket.finish.is_set())

    async def test_finish_cannot_succeed_if_local_send_fails(self):
        async def fail_finish(socket, _):
            if socket.finish.is_set():
                socket.sentence(1, "不得交付的残文")
                socket.event("task-finished")
                await asyncio.sleep(0)
                raise OSError("不应暴露的内容")

        socket = _Socket(on_send=fail_finish)
        with patch.object(BailianConnection, "_open", return_value=socket):
            with self.assertRaisesRegex(AsrError, "^asr_transport$"):
                await self.asr().transcribe_stream(_frames(b"\x00\x00"))
        self.assertTrue(socket.closed)

    async def test_timeout_during_start_send_finish_and_file_total(self):
        async def block_audio(_, raw):
            if isinstance(raw, bytes):
                await asyncio.Future()

        async def block_finish(socket, _):
            if socket.finish.is_set():
                await asyncio.Future()

        for socket in (_Socket(started=False), _Socket(on_send=block_audio), _Socket(on_send=block_finish), _Socket()):
            async with _using_socket(socket, timeout=1):
                with self.assertRaisesRegex(AsrError, "^asr_timeout$"):
                    await self.asr(timeout=0.025).transcribe_stream(_frames(b"\x00\x00"))
            self.assertTrue(socket.closed)
        socket = _Socket()
        with patch.object(BailianConnection, "_open", return_value=socket):
            with self.assertRaisesRegex(AsrError, "^asr_timeout$"):
                await self.asr(timeout=0.01).transcribe(AudioFile(_wav(frames=6_000)))
        self.assertTrue(socket.closed)
        self.assertFalse(socket.finish.is_set())

    async def test_stream_cancel_cleans_source_without_sending_finish(self):
        blocked = asyncio.Event()
        ended = asyncio.Event()

        async def capture():
            yield b"\x00\x00"
            try:
                blocked.set()
                await asyncio.Future()
            finally:
                ended.set()

        socket = _Socket()
        async with _using_socket(socket):
            task = asyncio.create_task(self.asr().transcribe_stream(capture()))
            await blocked.wait()
            socket.sentence(1, "不得交付的残文")
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(ended.is_set())
        self.assertTrue(socket.closed)
        self.assertFalse(socket.finish.is_set())

    async def test_close_missing_conflicting_or_foreign_result_does_not_deliver_text(self):
        for variant, error in (("closed", "asr_transport"), ("missing", "asr_invalid_response"),
                               ("conflicting", "asr_invalid_response"), ("foreign", "asr_invalid_response"),
                               ("empty", "asr_empty_transcript")):
            async def respond(socket, _):
                if not socket.finish.is_set():
                    return
                if variant == "closed":
                    socket.sentence(1, "不可提交")
                    socket.incoming.put_nowait(ConnectionClosedError(None, None))
                elif variant == "missing":
                    socket.sentence(2, "缺第一句")
                elif variant == "conflicting":
                    socket.sentence(1, "甲")
                    socket.sentence(1, "乙")
                else:
                    socket.event("task-finished", task_id="foreign" if variant == "foreign" else None)
                    return
                socket.event("task-finished")

            socket = _Socket(on_send=respond)
            with self.subTest(variant=variant), patch.object(BailianConnection, "_open", return_value=socket):
                with self.assertRaisesRegex(AsrError, f"^{error}$"):
                    await self.asr().transcribe_stream(_frames(b"\x00\x00"))
            self.assertTrue(socket.closed)

    async def test_input_failure_and_bad_pcm_are_sanitized(self):
        async def broken():
            yield b"\x00\x00"
            raise RuntimeError("不应暴露的内容")

        for stream, error in ((_frames(), "invalid_audio_size"), (_frames(b"\x00"), "invalid_audio_chunk"),
                              (broken(), "asr_input_failed")):
            socket = _Socket()
            with patch.object(BailianConnection, "_open", return_value=socket):
                with self.assertRaisesRegex(AsrError, f"^{error}$"):
                    await self.asr().transcribe_stream(stream)
            self.assertFalse(socket.finish.is_set())
            self.assertTrue(socket.closed)


class BailianTTSTests(unittest.IsolatedAsyncioTestCase):
    def tts(self, *, model="qwen-audio-3.1-tts-flash", voice_id="longanhuan_v3.1", timeout=1):
        return BailianTTS(BailianConnection("test-only", "workspace"), model=model, voice_id=voice_id, timeout=timeout)

    async def test_both_models_send_one_segment_and_yield_before_task_finishes(self):
        async def respond(socket, _):
            if socket.finish.is_set():
                socket.event("result-generated", {"output": {"type": "sentence-begin"}})
                socket.incoming.put_nowait(b"\x01")
                socket.incoming.put_nowait(b"\x02\x03")
                socket.incoming.put_nowait(b"\x04")

        for model, voice in (("qwen-audio-3.1-tts-flash", "longanhuan_v3.1"), ("cosyvoice-v3-flash", "longanyang")):
            socket = _Socket(on_send=respond)
            with self.subTest(model=model), patch.object(BailianConnection, "_open", return_value=socket):
                async with aclosing(self.tts(model=model, voice_id=voice).synthesize("测试。", sample_rate=44_100)) as pcm:
                    self.assertEqual(await anext(pcm), b"\x01\x02")
                    self.assertEqual(await anext(pcm), b"\x03\x04")
                    self.assertFalse(socket.closed)
                    socket.event("task-finished")
                    with self.assertRaises(StopAsyncIteration):
                        await anext(pcm)
            self.assertTrue(socket.closed)
            self.assertEqual(socket.actions(), ["run-task", "continue-task", "finish-task"])
            self.assertIsInstance(UUID(socket.task_id), UUID)
            self.assertEqual({msg["header"]["task_id"] for msg in socket.messages}, {socket.task_id})
            run = socket.messages[0]["payload"]
            self.assertEqual((run["task"], run["function"], run["model"]), ("tts", "SpeechSynthesizer", model))
            self.assertEqual(run["parameters"], {
                "text_type": "PlainText", "voice": voice, "format": "pcm", "sample_rate": 44_100,
            })
            self.assertEqual(socket.messages[1]["payload"], {"input": {"text": "测试。"}})

    async def test_early_close_releases_socket_and_keeps_next_task_separate(self):
        async def first(socket, _):
            if socket.finish.is_set():
                socket.incoming.put_nowait(b"\x01\x02")

        async def second(socket, _):
            if socket.finish.is_set():
                socket.incoming.put_nowait(b"\x03\x04")
                socket.event("task-finished")

        previous = _Socket(on_send=first)
        current = _Socket(on_send=second)
        adapter = self.tts()
        with patch.object(BailianConnection, "_open", side_effect=[previous, current]) as dial:
            async with aclosing(adapter.synthesize("上一段", sample_rate=44_100)) as stream:
                self.assertEqual(await anext(stream), b"\x01\x02")
            self.assertTrue(previous.closed)
            chunks = [chunk async for chunk in adapter.synthesize("下一段", sample_rate=44_100)]
        self.assertEqual(dial.call_count, 2)
        self.assertEqual(chunks, [b"\x03\x04"])
        self.assertTrue(current.closed)
        self.assertNotEqual(previous.task_id, current.task_id)

    async def test_partial_audio_requires_successful_terminal_and_sample_alignment(self):
        for variant, error in (("odd", "tts_invalid_pcm"), ("disconnect", "tts_incomplete_stream"),
                               ("error", "tts_business_error"), ("foreign", "tts_invalid_event"),
                               ("timeout", "tts_timeout"), ("empty", "tts_empty_audio")):
            async def respond(socket, _):
                if not socket.finish.is_set():
                    return
                if variant != "empty":
                    socket.incoming.put_nowait(b"\x01\x02")
                if variant == "odd":
                    socket.incoming.put_nowait(b"\x03")
                    socket.event("task-finished")
                elif variant == "disconnect":
                    socket.incoming.put_nowait(ConnectionClosedError(None, None))
                elif variant == "error":
                    socket.event("task-failed", {"error_message": "不应暴露的内容"})
                elif variant == "foreign":
                    socket.event("task-finished", task_id="foreign")
                elif variant == "empty":
                    socket.event("task-finished")

            socket = _Socket(on_send=respond)
            with self.subTest(variant=variant), patch.object(BailianConnection, "_open", return_value=socket):
                async with aclosing(self.tts(timeout=0.05).synthesize("测试。", sample_rate=44_100)) as pcm:
                    if variant != "empty":
                        self.assertEqual(await anext(pcm), b"\x01\x02")
                    with self.assertRaisesRegex(TtsError, f"^{error}$") as exc:
                        await anext(pcm)
                    self.assertEqual(exc.exception.diagnostic, error)
            self.assertTrue(socket.closed)

    async def test_cancel_pending_receive_releases_connection(self):
        socket = _Socket()
        async with _using_socket(socket):
            async with aclosing(self.tts().synthesize("测试。", sample_rate=44_100)) as pcm:
                call = asyncio.create_task(anext(pcm))
                await socket.finish.wait()
                call.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await call
        self.assertTrue(socket.closed)

    async def test_reject_other_protocol_and_invalid_sample_rate(self):
        with self.assertRaisesRegex(ValueError, "unsupported Bailian TTS model"):
            self.tts(model="qwen3-tts-flash-realtime")
        with patch.object(BailianConnection, "_open") as dial:
            with self.assertRaisesRegex(TtsError, "^tts_invalid_audio$"):
                await anext(self.tts().synthesize("测试。", sample_rate=12_345))
            dial.assert_not_called()
