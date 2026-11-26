import asyncio
from contextlib import aclosing
import unittest
from unittest.mock import AsyncMock, Mock, patch

from websockets.asyncio.client import ClientConnection
from websockets.client import ClientProtocol
from websockets.datastructures import Headers
from websockets.frames import Frame, Opcode
from websockets.http11 import Response
from websockets.protocol import State
from websockets.uri import parse_uri

from agent_companion.speech.asr import AsrError, AudioFile
from agent_companion.speech.bailian_asr import BailianASR
from agent_companion.speech.bailian_tts import BailianTTS
from agent_companion.speech.bailian_ws import BailianConnection, _DirectConnect
from test_bailian_speech import _Socket, _frames, _using_socket, _wav


_FRAME = b"\x01\x02"
_LATE_FRAME = b"\x03\x04"


def _connection():
    return BailianConnection("test-only", "workspace")


def _asr(timeout=1):
    return BailianASR(_connection(), timeout=timeout, max_audio_seconds=10)


def _reply_on_finish(*events):
    async def respond(socket, _):
        if socket.finish.is_set():
            for event in events:
                event(socket)
            socket.event("task-finished")
    return _Socket(on_send=respond)


class _BufferedWebSocket(_Socket):
    """使用真实 websockets 帧解析、接收缓冲和 finish 写入背压，不建立网络连接。"""

    def __init__(self):
        super().__init__()
        protocol = ClientProtocol(parse_uri("wss://workspace.example/"), state=State.OPEN)
        self.client = ClientConnection(protocol, ping_interval=None)
        self.transport = Mock(spec=asyncio.Transport)
        self.transport.is_closing.return_value = False
        self.client.connection_made(self.transport)
        self.client.process_event(Response(101, "Switching Protocols", Headers()))

    def event(self, *args, **kwargs):
        super().event(*args, **kwargs)
        raw = self.incoming.get_nowait()
        self.client.data_received(Frame(Opcode.TEXT, raw.encode()).serialize(mask=False))

    async def recv(self):
        raw = await self.client.recv()
        self.received.put_nowait(raw)
        return raw

    async def send(self, raw):
        await super().send(raw)
        if self.finish.is_set():
            await self.client.send(raw)

    async def __aexit__(self, *args):
        if self.client.state is not State.CLOSED:
            self.client.connection_lost(None)
        await super().__aexit__(*args)


class BailianASRCompletenessTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_final_after_substantive_tail_is_not_success(self):
        for blank_final in (False, True):
            steps = [lambda ws: ws.sentence(1, "已确认句"), lambda ws: ws.sentence(2, "未确认尾句", final=False)]
            if blank_final:
                steps.append(lambda ws: ws.sentence(2, "  \n"))
            with self.subTest(blank_final=blank_final), self.assertRaisesRegex(AsrError, "^asr_invalid_response$"):
                async with _using_socket(_reply_on_finish(*steps)):
                    await _asr().transcribe_stream(_frames(_FRAME))

    async def test_blank_begin_blank_final_and_heartbeat_do_not_create_text(self):
        def begin(ws, index, **fields):
            ws.event("result-generated", {"output": {"sentence": {
                "sentence_id": index, "sentence_end": False, "sentence_begin": True, **fields,
            }}})

        socket = _reply_on_finish(
            lambda ws: ws.sentence(3, "结尾"), lambda ws: ws.sentence(1, "开头"),
            lambda ws: begin(ws, 2, text=""), lambda ws: ws.sentence(2, " "),
            lambda ws: begin(ws, 4), lambda ws: ws.sentence(99, "心跳不是识别", heartbeat=True),
        )
        async with _using_socket(socket):
            self.assertEqual(await _asr().transcribe_stream(_frames(_FRAME)), "开头\n结尾")

        socket = _reply_on_finish(lambda ws: begin(ws, 1, text=""), lambda ws: ws.sentence(1, ""))
        async with _using_socket(socket):
            with self.assertRaisesRegex(AsrError, "^asr_empty_transcript$"):
                await _asr().transcribe_stream(_frames(_FRAME))

    async def test_substantive_partial_is_resolved_only_by_its_own_final(self):
        socket = _reply_on_finish(
            lambda ws: ws.sentence(2, "有内容", final=False), lambda ws: ws.sentence(1, "第一句"),
            lambda ws: ws.sentence(2, "第二句最终文本"),
        )
        async with _using_socket(socket):
            self.assertEqual(await _asr().transcribe_stream(_frames(_FRAME)), "第一句\n第二句最终文本")


class BailianASRTerminationTests(unittest.IsolatedAsyncioTestCase):
    async def test_uncooperative_input_never_keeps_socket_or_creates_more_writers(self):
        for terminal in ("remote-failure", "caller-cancel"):
            for late_eof in (False, True):
                with self.subTest(terminal=terminal, late_eof=late_eof):
                    await self._verify_uncooperative_input(terminal, late_eof)

    async def _verify_uncooperative_input(self, terminal, late_eof):
        waiting = asyncio.Event()
        cancelled = asyncio.Event()
        release = asyncio.Event()
        adapter = _asr()
        socket = _Socket()
        success = _reply_on_finish(lambda ws: ws.sentence(1, "新任务"))

        class Input:
            sender = None

            async def __anext__(self):
                if self.sender is None:
                    self.sender = asyncio.current_task()
                    return _FRAME
                waiting.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()  # 吞掉第一次取消，直到外部允许才生产一帧或结束。
                    if late_eof:
                        raise StopAsyncIteration from None
                    return _LATE_FRAME

        source = Input()
        with patch.object(BailianConnection, "_open", side_effect=[socket, success]) as dial:
            call = asyncio.create_task(adapter.transcribe_stream(source))
            try:
                await asyncio.wait_for(waiting.wait(), 1)
                if terminal == "remote-failure":
                    socket.event("task-failed", {"error_message": "不应暴露"})
                else:
                    call.cancel()
                # 用 wait 而不是 wait_for：旧实现会等待吞取消的输入，测试仍可执行 finally 解卡。
                done, _ = await asyncio.wait((call,), timeout=0.5)
                self.assertIn(call, done)
                if terminal == "remote-failure":
                    with self.assertRaisesRegex(AsrError, "^asr_business_error$"):
                        call.result()
                else:
                    self.assertTrue(call.cancelled())
                await asyncio.wait_for(cancelled.wait(), 1)
                self.assertTrue(socket.closed)
                self.assertFalse(source.sender.done())
                for _ in range(3):
                    with self.assertRaisesRegex(AsrError, "^asr_busy$"):
                        await adapter.transcribe_stream(_frames(_FRAME))
                self.assertEqual(dial.call_count, 1)
            finally:
                release.set()
                if not call.done():
                    call.cancel()
                await asyncio.gather(call, return_exceptions=True)
                if source.sender is not None:
                    await asyncio.wait_for(asyncio.gather(source.sender, return_exceptions=True), 1)
            self.assertEqual(socket.messages[1:], [_FRAME])
            self.assertEqual(socket.actions(), ["run-task"])
            self.assertEqual(await adapter.transcribe_stream(_frames(_FRAME)), "新任务")
            self.assertEqual(dial.call_count, 2)

    async def test_ready_frame_or_eof_does_not_win_receiver_failure_or_caller_cancellation(self):
        for terminal in ("remote-failure", "caller-cancel"):
            for late_eof in (False, True):
                with self.subTest(terminal=terminal, late_eof=late_eof):
                    waiting = asyncio.Event()
                    release = asyncio.Event()
                    socket = _Socket()

                    async def source():
                        yield _FRAME
                        waiting.set()
                        await release.wait()
                        if not late_eof:
                            yield _LATE_FRAME

                    async with _using_socket(socket), aclosing(source()) as frames:
                        call = asyncio.create_task(_asr().transcribe_stream(frames))
                        await waiting.wait()
                        if terminal == "remote-failure":
                            socket.event("task-failed")
                            release.set()  # 读取器先醒，发送器可早于协调者被调度。
                            with self.assertRaisesRegex(AsrError, "^asr_business_error$"):
                                await call
                        else:
                            release.set()  # 发送器先醒，协调者还没处理 CancelledError。
                            call.cancel()
                            with self.assertRaises(asyncio.CancelledError):
                                await call
                    self.assertEqual(socket.messages[1:], [_FRAME])

    async def test_terminal_in_receive_buffer_before_finish_is_rejected(self):
        source_waiting = asyncio.Event()
        eof = asyncio.Event()
        socket = _BufferedWebSocket()

        async def source():
            yield _FRAME
            source_waiting.set()
            await eof.wait()

        async with _using_socket(socket), aclosing(source()) as frames:
            call = asyncio.create_task(_asr().transcribe_stream(frames))
            await source_waiting.wait()
            # 先唤醒发送者但不让它运行；终止帧随后全部进入本地接收缓冲。
            eof.set()
            socket.sentence(1, "不能交付的提前结果")
            socket.event("task-finished")
            self.assertFalse(socket.finish.is_set())
            with self.assertRaisesRegex(AsrError, "^asr_invalid_response$"):
                await call
        self.assertEqual(socket.actions(), ["run-task"])
        socket.transport.write.assert_not_called()

    async def test_reader_can_finish_while_finish_send_is_still_draining(self):
        for text in ("完整结果", ""):
            for drain in ("resume", "failed", "timeout"):
                socket = _BufferedWebSocket()
                socket.client.pause_writing()
                with self.subTest(text=text, drain=drain):
                    async with _using_socket(socket):
                        asr = _asr(timeout=0.2 if drain == "timeout" else 1)
                        call = asyncio.create_task(asr.transcribe_stream(_frames(_FRAME)))
                        try:
                            await asyncio.wait_for(socket.finish.wait(), 2)
                            socket.transport.write.assert_called_once()  # 已写入，send 仍卡在排空。
                            if text:
                                socket.sentence(1, text)
                            socket.event("task-finished")
                            for _ in range(3 if text else 2):
                                raw = await asyncio.wait_for(socket.received.get(), 2)
                            self.assertIn("task-finished", raw)
                            self.assertFalse(call.done())
                            if drain == "failed":
                                socket.client.connection_lost(OSError("不应暴露"))
                            elif drain == "resume":
                                socket.client.resume_writing()
                            diagnostic = {"failed": "asr_transport", "timeout": "asr_timeout"}.get(drain)
                            if diagnostic is None and text:
                                self.assertEqual(await asyncio.wait_for(call, 2), text)
                            else:
                                with self.assertRaises(AsrError) as captured:
                                    await asyncio.wait_for(call, 2)
                                self.assertEqual(captured.exception.diagnostic, diagnostic or "asr_empty_transcript")
                                self.assertEqual(captured.exception.phase, "finish")
                        finally:
                            if not call.done():
                                call.cancel()
                                await asyncio.gather(call, return_exceptions=True)
                    self.assertTrue(socket.closed)

    async def test_done_writer_slot_is_reclaimed_before_old_cleanup_callback(self):
        waiting = asyncio.Event()
        cancelled = asyncio.Event()
        release = asyncio.Event()
        adapter = _asr()
        socket = _Socket()
        success = _reply_on_finish(lambda ws: ws.sentence(1, "新任务"))

        async def stalled():
            yield _FRAME
            waiting.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()

        saw_new_slot = []

        async def next_frames():
            slot = adapter._live_upload
            saw_new_slot.append(slot is not None and slot.sender is asyncio.current_task())
            yield _FRAME

        async with asyncio.timeout(2), aclosing(stalled()) as frames:
            with patch.object(BailianConnection, "_open", side_effect=[socket, success]) as dial:
                old_call = asyncio.create_task(adapter.transcribe_stream(frames))
                await waiting.wait()
                try:
                    old_call.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await old_call
                    await cancelled.wait()
                    old_slot = adapter._live_upload
                    release.set()
                    await asyncio.sleep(0)  # 发送者已退出，完成回调排在当前协程之后。
                    self.assertTrue(old_slot.sender.done())
                    self.assertIs(adapter._live_upload, old_slot)
                    self.assertEqual(await adapter.transcribe_stream(next_frames()), "新任务")
                    self.assertEqual(saw_new_slot, [True])
                    self.assertEqual(dial.call_count, 2)
                finally:
                    release.set()


class BailianCloseHandshakeTests(unittest.IsolatedAsyncioTestCase):
    def patch_slow_close(self, socket):
        async def close():
            await asyncio.Future()

        # 不替换 _open：通过真实库的连接上下文，避免只测到测试替身的 __aexit__。
        socket.handshake = AsyncMock()
        socket.start_keepalive = Mock()
        socket.close = AsyncMock(side_effect=close)
        socket.transport = Mock()
        return patch.object(_DirectConnect, "create_connection", new=AsyncMock(return_value=socket))

    async def test_file_success_is_not_a_timeout_when_ws_close_would_block(self):
        socket = _reply_on_finish(lambda ws: ws.sentence(1, "文件结果"))
        with self.patch_slow_close(socket):
            self.assertEqual(await _asr(timeout=0.2).transcribe(AudioFile(_wav())), "文件结果")
        socket.transport.abort.assert_called_once_with()
        socket.close.assert_not_awaited()

    async def test_tts_iterator_finishes_without_waiting_for_ws_close(self):
        socket = _reply_on_finish(lambda ws: ws.incoming.put_nowait(_FRAME))
        tts = BailianTTS(_connection(), model="qwen-audio-3.1-tts-flash", voice_id="longanhuan_v3.1", timeout=1)
        with self.patch_slow_close(socket):
            async with aclosing(tts.synthesize("测试", sample_rate=44_100)) as stream:
                self.assertEqual(await anext(stream), _FRAME)
                async with asyncio.timeout(0.5):
                    with self.assertRaises(StopAsyncIteration):
                        await anext(stream)
        socket.transport.abort.assert_called_once_with()
        socket.close.assert_not_awaited()
