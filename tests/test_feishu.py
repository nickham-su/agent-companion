import asyncio
from base64 import b64decode
from contextlib import closing
from functools import partial
import io
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

from feishu_test_support import private_discovery

from agent_companion.app import run_feishu, running_agent
from agent_companion.channels.feishu.channel import FeishuChannel, FeishuPoller
from agent_companion.channels.feishu.cli import CliError, CliStdoutLimit, CommandResult, FeishuCLI, run_command
from agent_companion.config import FeishuSettings, Settings, load_settings
from agent_companion.protocol import (OMITTED_INPUT_IMAGE, DeliveryStatus, Image, Message, Receipt, ReplyDelivery, ReplyInfo,
                                      Text, TextDelta)
from agent_companion.runtime.subtasks import demo_roles
from agent_companion.speech.asr import AsrError
from agent_companion.storage.sqlite import History


_CHAT, _PEER, _SELF = "oc_synthetic", "ou_peer", "ou_self"
_NOTICE = "语音接收失败，请重新发送。"
# Synthetic 1x1 PNG.
_PNG = b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGMw7XoHAAKlAa6/EWr0AAAAAElFTkSuQmCC")


def _wave() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        w.writeframes(b"\0\0" * 160)
    return buf.getvalue()


def _success(data, identity="user"):
    return CommandResult(0, json.dumps({"ok": True, "identity": identity, "data": data}).encode(), b"")


def _failure(category, code=None):
    return CommandResult(1, b"", json.dumps({"ok": False, "error": {"type": category, "code": code,
                                                                     "message": "synthetic private diagnostic"}}).encode())


def _raw(message_id, stamp, kind, content, *, sender=_PEER, chat=_CHAT, deleted=False,
         id_type="open_id", sender_type="user"):
    return {"message_id": message_id, "create_time": str(stamp), "msg_type": kind, "chat_id": chat,
            "sender": {"id": sender, "id_type": id_type, "sender_type": sender_type}, "deleted": deleted,
            "body": {"content": json.dumps(content) if not isinstance(content, str) else content}}


class _Runner:
    def __init__(self):
        self.pages = {None: {"items": [], "has_more": False}}
        self.calls = []
        self.sent = []
        self.posts = []
        self.transcript_sent = asyncio.Event()
        self.files = {"img_a": _PNG, "file_a": _wave()}
        self.notice_entered = asyncio.Event()
        self.release_notice = asyncio.Event()
        self.block_notice = False
        self.send_result = _success({"message_id": "om_reply"})

    async def __call__(self, argv, data, timeout):
        self.calls.append((argv, timeout))
        if len(argv) > 4 and argv[3].endswith("_file_limit.py"):
            argv = argv[5:]
        if argv[1:3] == ("api", "GET"):
            page = self.pages[json.loads(argv[argv.index("--params") + 1]).get("page_token")]
            return _success(page) if isinstance(page, dict) else page
        if argv[1:3] == ("im", "+messages-resources-download"):
            target = Path(argv[argv.index("--output") + 1])
            if blob := self.files.get(argv[argv.index("--file-key") + 1]):
                target.write_bytes(blob)
                return _success({"saved_path": "ignored", "size_bytes": len(blob)})
            return _failure("api", 230026)
        payload = json.loads(data)
        content = json.loads(payload["content"])
        if payload["msg_type"] == "post":
            self.posts.append(content)
            self.transcript_sent.set()
            return self.send_result
        text = content["text"]
        self.sent.append(text)
        if text == _NOTICE:
            self.notice_entered.set()
            if self.block_notice:
                await self.release_notice.wait()
        return self.send_result


class _ASR:
    def __init__(self):
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.error = None

    async def transcribe(self, file):
        self.calls += 1
        assert file.extension == "wav"
        self.entered.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error
        return "模拟转写"


class _Model:
    def __init__(self):
        self.calls = []

    async def stream(self, messages, tools, *, thinking):
        self.calls.append(tuple(messages))
        yield TextDelta("模拟回复")


def _poller(history, main, cli, asr, clock):
    return FeishuPoller(cli, asr, FeishuChannel(cli), chat_id=_CHAT, sender_id=_PEER, own_id=_SELF,
                        session_id="main", history=history, submit=main, private_started_ms=clock[0] * 1000,
                        clock=lambda: clock[0])


class FeishuCLITests(unittest.IsolatedAsyncioTestCase):
    async def test_raw_requests_and_stdin_not_argv(self):
        calls = []

        async def runner(argv, stdin, timeout):
            calls.append((argv, stdin, timeout))
            return _success({"message_id": "om_example", "items": [], "has_more": False})

        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        self.assertEqual(await cli.list_page(10, 20, 'opaque+/?="'), ([], None))
        text = '多行 "hello"\n--as bot; not an option'
        self.assertEqual(await cli.send_text(text), DeliveryStatus.FULL)
        first, stdin, _ = calls[0]
        self.assertIsNone(stdin)
        self.assertEqual(first[:5], ("lark-cli", "api", "GET", "/open-apis/im/v1/messages", "--params"))
        self.assertEqual(json.loads(first[5]), {"container_id_type": "chat", "container_id": _CHAT,
                         "start_time": "10", "end_time": "20", "sort_type": "ByCreateTimeAsc",
                         "page_size": 50, "page_token": 'opaque+/?="'})
        second, stdin, _ = calls[1]
        self.assertNotIn(text, " ".join(second))
        self.assertEqual(second[6:8], ("--data", "-"))
        self.assertEqual(second[-4:], ("--as", "user", "--format", "json"))
        body = json.loads(stdin)
        self.assertEqual((body["receive_id"], body["msg_type"], json.loads(body["content"])),
                          (_CHAT, "text", {"text": text}))

    async def test_transcript_uses_literal_bold_post_in_stdin(self):
        calls = []

        async def runner(argv, stdin, timeout):
            calls.append((argv, stdin))
            return _success({"message_id": "om_post", "chat_id": _CHAT})

        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        text = '  多行 "hello" &amp; **注意**\r\n<at>\\路径\n🙂\n'
        payload = cli.prepare_transcript(text)
        assert payload is not None
        self.assertEqual(await cli.send_prepared_transcript(payload), DeliveryStatus.FULL)
        (argv, stdin), = calls
        self.assertIs(stdin, payload)
        self.assertEqual(argv, ("lark-cli", "api", "POST", "/open-apis/im/v1/messages", "--params",
                                '{"receive_id_type":"chat_id"}', "--data", "-", "--as", "user", "--format", "json"))
        body = json.loads(stdin)
        self.assertEqual((body["receive_id"], body["msg_type"], json.loads(body["content"])),
                         (_CHAT, "post", {"zh_cn": {"content": [[{"tag": "text", "text": text, "style": ["bold"]}]]}}))

    def test_transcript_preparation_rejects_invalid_encoding_or_oversized_post(self):
        runner = _Runner()
        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        for text in ("\ud800", "字" * 10_000, '"' * 10_000, "a" * 30_001):
            with self.subTest(length=len(text)):
                self.assertIsNone(cli.prepare_transcript(text))
        self.assertEqual(runner.calls, [])

    async def test_list_page_size_bounds(self):
        runner = _Runner()
        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        for size in (1, 50):
            self.assertEqual(await cli.list_page(10, 20, None, page_size=size), ([], None))
        for size in (-1, 0, 51, 1.5, True, "25"):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "invalid Feishu page size"):
                await cli.list_page(10, 20, None, page_size=size)
        self.assertEqual([json.loads(argv[5])["page_size"] for argv, _ in runner.calls], [1, 50])

    async def test_text_receipts_are_terminal_and_not_retried(self):
        for raw, expected in (
            (_success({"message_id": "om_sent"}), DeliveryStatus.FULL),
            (_failure("authorization"), DeliveryStatus.FAILED),
            (_failure("validation", 230001), DeliveryStatus.FAILED),
            (_failure("api", 230001), DeliveryStatus.UNKNOWN),
            (_failure("api", 99999999), DeliveryStatus.UNKNOWN),
            (_failure("api", 400), DeliveryStatus.UNKNOWN),
            (_failure("network"), DeliveryStatus.UNKNOWN),
            (_failure("policy"), DeliveryStatus.UNKNOWN),
            (_failure("api", 503), DeliveryStatus.UNKNOWN),
            (_success({}, "user"), DeliveryStatus.UNKNOWN),
            (_success({"message_id": "om_sent"}, "bot"), DeliveryStatus.UNKNOWN),
            (CommandResult(1, b"", b"unparseable"), DeliveryStatus.UNKNOWN),
            (CliError("cli_timeout"), DeliveryStatus.UNKNOWN),
        ):
            with self.subTest(expected=expected):
                num = 0

                async def runner(argv, stdin, timeout):
                    nonlocal num
                    num += 1
                    if isinstance(raw, CliError):
                        raise raw
                    return raw

                cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
                self.assertEqual((await cli.send_text("第一第二"), num), (expected, 1))

    async def test_bad_resources_are_not_read_or_sent(self):
        runner = _Runner()
        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        with self.assertRaisesRegex(CliError, "invalid_resource_reference"):
            await cli.download("om_one", "img_../bad", "image", limit=100)
        self.assertEqual(runner.calls, [])
        with self.assertRaisesRegex(CliError, "invalid_resource_size"):
            await cli.download("om_one", "img_a", "image", limit=len(_PNG) - 1)
        self.assertEqual(await cli.send_text("a" * 150_000), DeliveryStatus.FAILED)
        self.assertEqual(runner.sent, [])
        target = Path(runner.calls[0][0][runner.calls[0][0].index("--output") + 1])
        self.assertFalse(target.parent.exists())

    @unittest.skipUnless(sys.platform in ("linux", "darwin"), "POSIX file limit")
    async def test_download_hard_limits_real_child_writes_and_cleans_up(self):
        import resource

        parent_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
        with TemporaryDirectory(dir=Path.cwd()) as root:
            worker = Path(root) / "synthetic-cli"
            audit = Path(root) / "audit.json"
            worker.write_text(f"#!{sys.executable}\n" + '''
import json
import os
from pathlib import Path
import resource
import sys

target = Path(sys.argv[sys.argv.index("--output") + 1])
key = sys.argv[sys.argv.index("--file-key") + 1]
attempts = 10 if key == "img_large" else 2
peak = 0
failed = False
with target.open("wb", buffering=0) as stream:
    for i in range(attempts):
        try:
            stream.write(b"x" * 64)
        except OSError:
            failed = True
        peak = max(peak, os.fstat(stream.fileno()).st_size)
        if failed:
            break
(Path(__file__).parent / "audit.json").write_text(json.dumps(
    [peak, target.parent.name, resource.getrlimit(resource.RLIMIT_FSIZE)]))
if failed:
    raise SystemExit(1)
print(json.dumps({"ok": True, "identity": "user", "data": {"size_bytes": peak}}))
''', encoding="utf-8")
            worker.chmod(0o700)
            cli = FeishuCLI(str(worker), _CHAT, timeout=3)
            with self.assertRaisesRegex(CliError, "^cli_result_unknown$"):
                await cli.download("om_one", "img_large", "image", limit=128)
            peak, directory, child_limit = json.loads(audit.read_text())
            self.assertEqual((peak, child_limit), (128, [128, 128]))
            self.assertFalse((Path.cwd() / directory).exists())
            self.assertEqual(await cli.download("om_one", "img_exact", "image", limit=128), b"x" * 128)
            _, directory, _ = json.loads(audit.read_text())
            self.assertFalse((Path.cwd() / directory).exists())
            self.assertEqual(resource.getrlimit(resource.RLIMIT_FSIZE), parent_limit)

    async def test_subprocess_stdin_timeout_and_bounded_capture(self):
        command = (sys.executable, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())")
        data = b"x" * 180_000
        result = await run_command(command, data, 3)
        self.assertEqual(result.stdout, data)
        with self.assertRaisesRegex(CliError, "cli_output_limit") as err:
            await run_command((sys.executable, "-c", "import sys; sys.stderr.write('x'*150000)"), None, 3)
        self.assertNotIsInstance(err.exception, CliStdoutLimit)
        with self.assertRaisesRegex(CliError, "cli_timeout"):
            await run_command((sys.executable, "-c", "import time; time.sleep(30)"), None, 0.05)


class FeishuBackgroundDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejected_transcripts_do_not_take_queue_slots(self):
        for reason, invalid in (("多字节超限", "字" * 10_000), ("转义超限", '"' * 10_000),
                                ("无效编码", "\ud800")):
            with self.subTest(reason=reason):
                runner = _Runner()
                channel = FeishuChannel(FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner))
                self.addAsyncCleanup(channel.close)
                with self.assertLogs("agent_companion.channels.feishu.channel", level="WARNING") as logs:
                    for _ in range(8):
                        channel.enqueue_transcript(invalid)
                    channel.enqueue_transcript("有效短转写")
                    self.assertEqual(runner.calls, [])  # 没有让出事件循环，发送只能在后台发生。
                    await asyncio.sleep(0)
                self.assertEqual(runner.posts, [{"zh_cn": {"content": [[
                    {"tag": "text", "text": "有效短转写", "style": ["bold"]}
                ]]}}])
                self.assertEqual(len(runner.calls), 1)
                self.assertEqual([record.getMessage() for record in logs.records],
                                 ["Feishu transcript dropped (invalid_payload)"] * 8)

    async def test_transcript_backlog_does_not_block_model_reply_and_sends_fifo_without_retry(self):
        first_started, release_first, reply_sent, last_started = (asyncio.Event() for _ in range(4))
        attempts = []

        async def runner(argv, data, timeout):
            payload = json.loads(data)
            content = json.loads(payload["content"])
            if payload["msg_type"] == "text":
                self.assertEqual(content, {"text": "模拟回复"})
                reply_sent.set()
            else:
                text = content["zh_cn"]["content"][0][0]["text"]
                attempts.append(text)
                if text == "转写 0":
                    first_started.set()
                    await release_first.wait()
                    raise RuntimeError("synthetic raw diagnostic")
                if text == "转写 1":
                    raise CliError("cli_timeout")
                if text == "转写 8":
                    last_started.set()
            return _success({"message_id": "om_reply"})

        model = _Model()
        channel = FeishuChannel(FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner))
        self.addAsyncCleanup(channel.close)
        with closing(History(":memory:")) as history:
            async with running_agent(Settings(), history, model, channel, session_id="main",
                                     roles=demo_roles()) as (main, _):
                channel.enqueue_transcript("转写 0")
                await asyncio.wait_for(first_started.wait(), 1)
                with self.assertLogs("agent_companion.channels.feishu.channel", level="WARNING") as logs:
                    for i in range(1, 10):
                        channel.enqueue_transcript(f"转写 {i}")
                    main.submit_batch([Message("user", "文字纠正")])
                    await asyncio.wait_for(reply_sent.wait(), 1)
                    self.assertEqual(attempts, ["转写 0"])
                    self.assertEqual(model.calls[0][-1], Message("user", "文字纠正"))
                    self.assertEqual([(row.message.role, row.message.content) for row in history.read_after("main")],
                                     [("user", "文字纠正"), ("assistant", "模拟回复")])
                    release_first.set()
                    await asyncio.wait_for(last_started.wait(), 1)
                self.assertEqual(attempts, [f"转写 {i}" for i in range(9)])
                self.assertEqual(len(logs.output), 3)
                self.assertIn("queue_full", logs.output[0])
                self.assertIn("status=unknown", logs.output[2])
                self.assertNotIn("转写", "".join(logs.output))
                self.assertNotIn("synthetic raw diagnostic", "".join(logs.output))

    async def test_close_cancels_inflight_transcript_and_discards_waiting_transcripts(self):
        started, cancelled = asyncio.Event(), asyncio.Event()
        calls = []

        async def runner(argv, data, timeout):
            calls.append(json.loads(json.loads(data)["content"]))
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        channel = FeishuChannel(FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner))
        self.addAsyncCleanup(channel.close)
        channel.enqueue_transcript("发送中")
        await asyncio.wait_for(started.wait(), 1)
        channel.enqueue_transcript("待发")
        await asyncio.wait_for(channel.close(), 1)
        channel.enqueue_transcript("已经关闭")
        self.assertTrue(cancelled.is_set())
        self.assertEqual(len(calls), 1)

    async def test_slow_cli_does_not_block_runtime_fifo_or_later_sends_after_failures(self):
        first_started, release_first, finished = (asyncio.Event() for _ in range(3))
        sent, committed_before_send = [], []

        async def runner(argv, data, timeout):
            text = json.loads(json.loads(data)["content"])["text"]
            sent.append(text)
            committed_before_send.append(text in [row.message.content for row in history.read_after("main")])
            if text == "回复 0":
                first_started.set()
                await release_first.wait()
            if text == "回复 1":
                raise CliError("cli_timeout")
            if text == "回复 2":
                raise RuntimeError("synthetic raw diagnostic")
            if text == "回复 8":
                finished.set()
            return _success({"message_id": "om_reply"})

        class Echo:
            async def stream(self, messages, tools, *, thinking):
                yield TextDelta(f"回复 {messages[-1].content}")

        channel = FeishuChannel(FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner))
        self.addAsyncCleanup(channel.close)
        with closing(History(":memory:")) as history:
            async with running_agent(Settings(), history, Echo(), channel, session_id="main",
                                     roles=demo_roles()) as (main, _):
                main.submit_batch([Message("user", "0")])
                await asyncio.wait_for(first_started.wait(), 1)
                with self.assertLogs("agent_companion.channels.feishu.channel", level="WARNING") as logs:
                    for i in range(1, 10):
                        main.submit_batch([Message("user", str(i))])
                    await asyncio.wait_for(main.wait_idle(), 1)
                    self.assertEqual(sent, ["回复 0"])
                    self.assertEqual([row.message.content for row in history.read_after("main") if
                                      row.message.role == "assistant"], [f"回复 {i}" for i in range(10)])
                    self.assertEqual([outcome.delivery_status for outcome in history.outcomes("main")],
                                     [DeliveryStatus.FULL] * 10)
                    release_first.set()
                    await asyncio.wait_for(finished.wait(), 1)
                self.assertEqual(sent, [f"回复 {i}" for i in range(9)])
                self.assertEqual(committed_before_send, [True] * 9)
                self.assertEqual(len(logs.output), 3)
                self.assertIn("queue_full", logs.output[0])
                self.assertNotIn("回复", "".join(logs.output))
                self.assertNotIn("synthetic raw diagnostic", "".join(logs.output))

    async def test_shutdown_discards_queue_and_waits_for_active_cli_cancellation_cleanup(self):
        entered, cancelling, release, cleaned = (asyncio.Event() for _ in range(4))
        sent = []

        class CLI:
            async def send_text(self, text):
                sent.append(text)
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelling.set()
                    await release.wait()
                    cleaned.set()

        channel = FeishuChannel(CLI())
        info = ReplyInfo("simulated", "main", "run", 0)
        channel.enqueue(info, "当前")
        await asyncio.wait_for(entered.wait(), 1)
        channel.enqueue(info, "待发")
        with self.assertLogs("agent_companion.channels.feishu.channel", level="WARNING") as logs:
            shutdown = asyncio.create_task(channel.close())
            try:
                await asyncio.wait_for(cancelling.wait(), 1)
                self.assertFalse(shutdown.done())
                channel.enqueue(info, "关闭后")
            finally:
                release.set()
                await asyncio.wait_for(shutdown, 1)
        self.assertTrue(cleaned.is_set())
        self.assertEqual(sent, ["当前"])
        self.assertIn("count=1", logs.output[0])
        self.assertIn("closed", logs.output[1])

    async def test_voice_copies_only_the_confirmed_text_after_history_commit(self):
        full, prefix = "  模拟回复。\n 后续", "  模拟回复。\n "
        cases = (
            ("full", DeliveryStatus.FULL, full, full),
            ("partial", DeliveryStatus.PARTIAL, prefix, prefix),
            ("failed", DeliveryStatus.FAILED, prefix, prefix),
            ("idle timeout", DeliveryStatus.UNKNOWN, prefix, prefix),
            ("zero", DeliveryStatus.ZERO, "", ""),
            ("invalid receipt", DeliveryStatus.FULL, "错误回执", ""),
            ("disabled", DeliveryStatus.FULL, full, full),
            ("commit failed", DeliveryStatus.FULL, full, full),
        )
        for case, status, confirmed, saved in cases:
            with self.subTest(case=case), closing(History(":memory:")) as history:
                sent = asyncio.Event()
                outbound = []

                class Model:
                    async def stream(self, messages, tools, *, thinking):
                        yield TextDelta(full)

                class Voice:
                    async def deliver(self, reply):
                        reply.select_voice_delivery()
                        async for _ in reply:
                            pass
                        if case == "idle timeout":
                            reply.report_progress(reply.info.reply_id, 42, confirmed)
                            await asyncio.Future()
                        return Receipt(reply.info.reply_id, status, confirmed)

                class Runner(_Runner):
                    async def __call__(self, argv, data, timeout):
                        result = await super().__call__(argv, data, timeout)
                        if argv[1:3] == ("api", "POST"):
                            body = json.loads(data)
                            outbound.append((body["receive_id"], json.loads(body["content"])["text"],
                                             history.tail("main", 1)[0].message))
                            sent.set()
                        return result

                runner = Runner()
                cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
                finalize_reply = history.finalize_reply

                def finalize(outcome, messages):
                    if case == "commit failed":
                        raise sqlite3.OperationalError("synthetic write failure")
                    return finalize_reply(outcome, messages)

                expected_to_send = bool(saved) and case not in ("disabled", "commit failed")

                voice = Voice()

                def select_voice(reply):
                    reply.select_voice_delivery()
                    return ReplyDelivery(voice.deliver, 0.05, channel.enqueue if case != "disabled" else None)

                channel = FeishuChannel(cli, select_voice=select_voice)
                try:
                    with patch.object(history, "finalize_reply", finalize):
                        async with running_agent(Settings(), history, Model(), channel,
                                                 session_id="main", roles=demo_roles()) as (main, _):
                            main.submit_batch([Message("user", "测试输入")])
                            await asyncio.wait_for(main.wait_idle(), 1)
                            if expected_to_send:
                                await asyncio.wait_for(sent.wait(), 1)
                            else:
                                await asyncio.sleep(0)  # 若错误入队，给发送器执行机会。
                finally:
                    await channel.close()
                self.assertEqual(outbound, [(_CHAT, saved, Message("assistant", saved))] if expected_to_send else [])
                assistant = [entry.message for entry in history.read_after("main") if entry.message.role == "assistant"]
                outcomes = history.outcomes("main")
                self.assertEqual(assistant, [] if case in ("commit failed", "zero") else [Message("assistant", saved)])
                expected_status = DeliveryStatus.UNKNOWN if case == "invalid receipt" else status
                self.assertEqual([outcome.delivery_status for outcome in outcomes],
                                 [] if case == "commit failed" else [expected_status])

    async def test_application_stops_ingress_and_sends_before_async_poll_cleanup(self):
        active, poll_ready, poll_cleaning, release_poll = (asyncio.Event() for _ in range(4))
        release_send, send_cancelled, send_cleaned = (asyncio.Event() for _ in range(3))
        sent = []
        submit = None

        class CLI(FeishuCLI):
            async def send_text(self, text):
                sent.append(text)
                active.set()
                try:
                    await release_send.wait()
                except asyncio.CancelledError:
                    send_cancelled.set()
                    await release_send.wait()
                finally:
                    send_cleaned.set()
                return DeliveryStatus.FULL

        async def poll(poller):
            nonlocal submit
            submit = poller._submit
            info = ReplyInfo("simulated", "main", "run", 0)
            poller._channel.enqueue(info, "当前")
            await active.wait()
            poller._channel.enqueue(info, "不能开始")
            poll_ready.set()
            try:
                await asyncio.Future()
            finally:
                poll_cleaning.set()
                await release_poll.wait()

        settings = Settings(feishu=FeishuSettings(own_open_id=_SELF))
        discovery = private_discovery(_CHAT, _PEER, _SELF, cli=CLI("lark-cli", _CHAT, timeout=2))
        with (closing(History(":memory:")) as history, patch.object(FeishuPoller, "run", poll),
              self.assertLogs("agent_companion.channels.feishu.channel", level="WARNING")):
            app = asyncio.create_task(run_feishu(settings, history, _Model(), _ASR(), discovery=discovery,
                                                 roles=demo_roles()))
            try:
                await asyncio.wait_for(poll_ready.wait(), 1)
                app.cancel()
                await asyncio.wait_for(poll_cleaning.wait(), 1)
                self.assertTrue(send_cancelled.is_set())
                self.assertIsNone(await submit(-1))
                release_send.set()
                await asyncio.wait_for(send_cleaned.wait(), 1)
                self.assertFalse(app.done())
                self.assertEqual(sent, ["当前"])
            finally:
                release_send.set()
                release_poll.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(app, 1)

    async def test_poll_failure_stops_replies_before_audio_notice_cleanup(self):
        active, release_send, send_cleaned = (asyncio.Event() for _ in range(3))
        send_cancelled, notice_cleaning = asyncio.Event(), asyncio.Event()
        sent = []
        clock = [100]
        submit = None

        class Runner(_Runner):
            async def __call__(self, argv, data, timeout):
                if argv[1:3] != ("api", "POST"):
                    return await super().__call__(argv, data, timeout)
                text = json.loads(json.loads(data)["content"])["text"]
                if text == _NOTICE:
                    self.notice_entered.set()
                    try:
                        await asyncio.Future()
                    finally:
                        notice_cleaning.set()
                        await self.release_notice.wait()
                else:
                    sent.append(text)
                    if text == "当前":
                        active.set()
                        try:
                            await release_send.wait()
                        except asyncio.CancelledError:
                            send_cancelled.set()
                            await release_send.wait()
                        finally:
                            send_cleaned.set()
                return self.send_result

        runner, asr = Runner(), _ASR()
        runner.pages[None] = {"items": [_raw("om_bad_audio", 101_000, "audio", {"file_key": "file_a"})],
                              "has_more": False}
        asr.error = AsrError("asr_transport")
        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        real_poll_once = FeishuPoller.poll_once

        async def poll_once(poller):
            nonlocal submit
            submit = poller._submit
            await real_poll_once(poller)
            clock[0] = 110
            await real_poll_once(poller)  # 走真实 ASR 失败路径，启动固定通知。
            info = ReplyInfo("simulated", "main", "run", 0)
            poller._channel.enqueue(info, "当前")
            poller._channel.enqueue(info, "不能开始")
            await active.wait()
            await runner.notice_entered.wait()
            raise RuntimeError("synthetic poll failure")

        settings = Settings(feishu=FeishuSettings(own_open_id=_SELF))
        discovery = private_discovery(_CHAT, _PEER, _SELF, cli=cli)
        with (closing(History(":memory:")) as history, patch.object(FeishuPoller, "poll_once", poll_once),
              patch("agent_companion.app.time", return_value=clock[0]),
              patch("agent_companion.feishu_private.FeishuPoller", partial(FeishuPoller, clock=lambda: clock[0]))):
            app = asyncio.create_task(run_feishu(settings, history, _Model(), asr, discovery=discovery,
                                                 roles=demo_roles()))
            try:
                await asyncio.wait_for(notice_cleaning.wait(), 1)
                release_send.set()
                await asyncio.wait_for(send_cleaned.wait(), 1)
                self.assertFalse(app.done())
                self.assertEqual(sent, ["当前"])
                self.assertTrue(send_cancelled.is_set())
                self.assertIsNone(await submit(-1))
            finally:
                release_send.set()
                runner.release_notice.set()
                with self.assertRaises(ExceptionGroup) as raised:
                    await asyncio.wait_for(app, 1)
                self.assertEqual([str(exc) for exc in raised.exception.exceptions], ["synthetic poll failure"])


class FeishuPollingTests(unittest.IsolatedAsyncioTestCase):
    async def test_across_pages_order_filter_post_and_delayed_voice(self):
        clock = [100]
        runner, asr, model = _Runner(), _ASR(), _Model()
        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        audio = _raw("om_voice", 102_000, "audio", {"file_key": "file_a"})
        rich = _raw("om_rich", 103_000, "post", {"zh_cn": {"title": "标题", "content": [[
            {"tag": "text", "text": "前"}, {"tag": "img", "image_key": "img_a"},
            {"tag": "a", "text": "后", "href": "https://example.invalid/link"}
        ], [{"tag": "text", "text": "次行"}]]}})
        runner.pages = {None: {"items": [audio, _raw("om_text", 101_000, "text", {"text": "先到"}),
                                        _raw("om_own", 103_100, "text", {"text": "自身"}, sender=_SELF),
                                        _raw("om_other", 102_100, "text", {"text": "其他人"}, sender="ou_other"),
                                        _raw("om_wrong", 102_100, "text", {"text": "其他频道"}, chat="oc_other"),
                                        _raw("om_bot", 102_100, "text", {"text": "应用"}, sender_type="app"),
                                        _raw("om_deleted", 102_100, "text", {"text": "撤回"}, deleted=True)],
                               "has_more": True, "page_token": "next"},
                        "next": {"items": [audio, rich, _raw("om_pic", 103_000, "image", {"image_key": "img_a"})],
                                 "has_more": False}}
        asr.release.clear()
        with closing(History(":memory:")) as history:
            channel = FeishuChannel(cli)
            self.addAsyncCleanup(channel.close)
            async with running_agent(Settings(), history, model, channel, session_id="main",
                                     roles=demo_roles()) as (main, _):
                submit = main.submit_persisted_batch
                poller = _poller(history, submit, cli, asr, clock)
                await poller.poll_once()
                clock[0] = 110
                with patch.object(history, "record_inbound", side_effect=sqlite3.OperationalError("database is locked")):
                    polling = asyncio.create_task(poller.poll_once())
                    await asyncio.wait_for(asr.entered.wait(), 1)
                    self.assertEqual(model.calls, [])
                    self.assertEqual(history.tail("main", 20), [])
                    self.assertEqual(runner.posts, [])
                    asr.release.set()
                    with self.assertRaisesRegex(sqlite3.OperationalError, "database is locked"):
                        await polling
                self.assertEqual(runner.posts, [])
                await poller.poll_once()
                await asyncio.wait_for(runner.transcript_sent.wait(), 1)
                await main.wait_idle()
                self.assertEqual(runner.posts, [{"zh_cn": {"content": [[
                    {"tag": "text", "text": "模拟转写", "style": ["bold"]}
                ]]}}])
                self.assertEqual(asr.calls, 1)
                self.assertEqual(len(model.calls), 1)
                text, voice, post, image = model.calls[0][-4:]
                self.assertEqual((text.content, voice.content), ("先到", "模拟转写"))
                self.assertEqual(post.content, (Text("标题"), Text("\n"), Text("前"), image.content[0],
                                                Text("后 (https://example.invalid/link)"), Text("\n"), Text("次行")))
                self.assertIsInstance(image.content[0], Image)
                self.assertEqual(b64decode(image.content[0].url.split(",", 1)[1]), _PNG)
                # 模拟下一轮列表读到自己刚发送的回显，既不能喂给模型也不能再发。
                runner.pages["next"]["items"].append(_raw("om_echo", 108_000, "post", runner.posts[0], sender=_SELF))
                clock[0] = 120
                await poller.poll_once()
                await main.wait_idle()
                self.assertEqual((len(model.calls), asr.calls), (1, 1))
                self.assertEqual(len(runner.posts), 1)
                self.assertFalse(history.inbound_seen("main", "feishu-private", "om_echo"))
                self.assertEqual([row.message.content for row in history.tail("main", 20)].count("模拟转写"), 1)
                self.assertEqual(runner.sent, ["模拟回复"])
                await poller.close()

    async def test_failed_audio_never_retried_or_added(self):
        for error in (AsrError("asr_transport"), TimeoutError()):
            with self.subTest(error=type(error).__name__):
                clock = [100]
                runner, asr = _Runner(), _ASR()
                runner.pages[None] = {"items": [_raw("om_before", 100_001, "text", {"text": "前"}),
                                                _raw("om_voice", 101_000, "audio", {"file_key": "file_a"}),
                                                _raw("om_after", 101_001, "text", {"text": "后"})], "has_more": False}
                runner.block_notice = True
                asr.error = error
                cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
                with closing(History(":memory:")) as history:
                    submitted = asyncio.Event()
                    accepted = []

                    def submit(batch):
                        history.consume_inbound("main", batch)
                        accepted.append(batch)
                        submitted.set()

                    poller = _poller(history, submit, cli, asr, clock)
                    await poller.poll_once()
                    clock[0] = 110
                    await poller.poll_once()
                    await asyncio.wait_for(runner.notice_entered.wait(), 1)
                    self.assertTrue(submitted.is_set())
                    self.assertEqual([row.message.content for row in history.tail("main", 20)], ["前", "后"])
                    self.assertEqual((runner.sent, asr.calls, len(accepted)), ([_NOTICE], 1, 1))
                    clock[0] = 120
                    await poller.poll_once()
                    await asyncio.sleep(0)
                    self.assertEqual((runner.sent, asr.calls, len(accepted)), ([_NOTICE], 1, 1))
                    self.assertEqual(runner.posts, [])
                    await poller.close()

    async def test_failed_page_does_not_advance_window_or_publish(self):
        clock = [100]
        runner, asr = _Runner(), _ASR()
        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        runner.pages = {None: {"items": [_raw("om_one", 101_000, "text", {"text": "正常"})],
                               "has_more": True, "page_token": "next"}, "next": _failure("network")}
        with closing(History(":memory:")) as history:
            accepted = []
            poller = _poller(history, accepted.append, cli, asr, clock)
            await poller.poll_once()
            clock[0] = 110
            with self.assertRaisesRegex(CliError, "cli_result_unknown"):
                await poller.poll_once()
            self.assertEqual(accepted, [])
            clock[0] = 150
            runner.pages["next"] = {"items": [], "has_more": False}
            await poller.poll_once()
            self.assertEqual(len(accepted), 1)
            self.assertEqual(json.loads(runner.calls[-1][0][5])["start_time"], "100")
            await poller.close()

    async def test_page_budget_resumes_fixed_window_and_commits_only_once_complete(self):
        clock = [100]
        runner = _Runner()
        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        one = _raw("om_first", 101_000, "text", {"text": "最先"})
        two = _raw("om_second", 101_000, "text", {"text": "同一毫秒"})
        three = _raw("om_third", 104_000, "text", {"text": "最后"})
        runner.pages = {None: {"items": [one], "has_more": True, "page_token": "next"},
                        "next": {"items": [one, two], "has_more": True, "page_token": "last"},
                        "last": {"items": [three], "has_more": False}}
        source = "feishu-private"
        with closing(History(":memory:")) as history, patch("agent_companion.channels.feishu.channel._MAX_PAGES", 1):
            accepted = []
            poller = _poller(history, accepted.append, cli, _ASR(), clock)
            await poller.poll_once()
            for current_time in (110, 120, 130):
                clock[0] = current_time
                await poller.poll_once()
                if current_time < 130:
                    self.assertEqual(history.inbound_window("main", source, current_time), (100, 100))
                    self.assertEqual(accepted, [])
            pages = [json.loads(argv[5]) for argv, _ in runner.calls]
            self.assertEqual([(p["start_time"], p["end_time"], p.get("page_token")) for p in pages],
                             [("100", "110", None), ("100", "110", "next"), ("100", "110", "last")])
            self.assertEqual(history.inbound_window("main", source, 130), (100, 110))
            self.assertEqual(len(accepted), 1)
            history.consume_inbound("main", accepted[0])
            self.assertEqual([row.message.content for row in history.tail("main", 5)], ["最先", "同一毫秒", "最后"])
            await poller.close()

    @unittest.skipUnless(sys.platform in ("linux", "darwin"), "POSIX executable CLI substitute")
    async def test_large_list_stdout_restarts_fixed_window_and_commits_one_ordered_batch(self):
        clock = [100]
        source = "feishu-private"
        with TemporaryDirectory(dir=Path.cwd()) as root, closing(History(":memory:")) as history:
            worker = Path(root) / "synthetic-cli"
            audit = Path(root) / "queries.jsonl"
            worker.write_text(f"#!{sys.executable}\n" + '''
import json
from pathlib import Path
import sys

params = json.loads(sys.argv[sys.argv.index("--params") + 1])
start, end, size = params["start_time"], params["end_time"], params["page_size"]
token = params.get("page_token")
with Path(__file__).with_name("queries.jsonl").open("a", encoding="utf-8") as audit:
    audit.write(json.dumps([start, end, size, token]) + "\\n")
prefix = f"{start}:{end}:{size}:"
assert token is None or token.startswith(prefix)
offset = 0 if token is None else int(token.removeprefix(prefix))
items = []
for position in range(offset, min(offset + size, 100)):
    i = 74 if position == 75 else position  # 两个小页之间重复一条消息。
    text = str(i) + ":" + ("x" * 90_000 if i >= 50 else "")
    if size == 50 and i == 0:
        text += "stale"  # 重读必须丢弃旧页，而不是用 ID 去重留下旧正文。
    items.append({"message_id": f"om_{i}", "create_time": str(101_000 + i // 2),
                  "msg_type": "text", "chat_id": params["container_id"],
                  "sender": {"id": "ou_peer", "id_type": "open_id", "sender_type": "user"},
                  "body": {"content": json.dumps({"text": text})}})
next_offset = offset + len(items)
page = {"items": items, "has_more": next_offset < 100}
if page["has_more"]:
    page["page_token"] = prefix + str(next_offset)
print(json.dumps({"ok": True, "identity": "user", "data": page}))
''', encoding="utf-8")
            worker.chmod(0o700)
            cli = FeishuCLI(str(worker), _CHAT, timeout=3)
            accepted = []
            poller = _poller(history, accepted.append, cli, _ASR(), clock)
            await poller.poll_once()
            with patch("agent_companion.channels.feishu.channel._MAX_PAGES", 2):
                for current_time in (110, 120):
                    clock[0] = current_time
                    await poller.poll_once()
                    self.assertEqual(history.inbound_window("main", source, current_time), (100, 100))
                    self.assertEqual(accepted, [])
                    self.assertEqual(history.pending_inbound("main", source), [])
                    self.assertFalse(history.inbound_seen("main", source, "om_0"))
                clock[0] = 130
                await poller.poll_once()

            queries = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(queries, [["100", "110", size, token] for size, token in (
                (50, None), (50, "100:110:50:50"), (25, None), (25, "100:110:25:25"),
                (25, "100:110:25:50"), (25, "100:110:25:75"),
            )])
            self.assertEqual(history.inbound_window("main", source, 130), (100, 110))
            self.assertEqual(len(accepted), 1)
            self.assertEqual(history.pending_inbound("main", source), accepted)
            history.consume_inbound("main", accepted[0])
            contents = [row.message.content for row in history.tail("main", 101)]
            self.assertEqual([(text.split(":", 1)[0], len(text)) for text in contents], [
                (str(i), len(str(i)) + 1 + (90_000 if i >= 50 else 0)) for i in range(100) if i != 75
            ])
            await poller.close()

    async def test_single_result_stdout_limit_does_not_skip_or_commit(self):
        clock = [100]
        source = "feishu-private"
        queries = []
        first = _raw("om_first", 101_000, "text", {"text": "可读取"})

        async def runner(argv, data, timeout):
            params = json.loads(argv[5])
            size, token = params["page_size"], params.get("page_token")
            queries.append((params["start_time"], params["end_time"], size, token))
            if size == 1 and token is None:
                return _success({"items": [first], "has_more": True, "page_token": "blocked"})
            raise CliStdoutLimit("cli_output_limit")

        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        with closing(History(":memory:")) as history:
            poller = _poller(history, self.fail, cli, _ASR(), clock)
            await poller.poll_once()
            for current_time in (110, 120, 130, 140, 150):
                clock[0] = current_time
                await poller.poll_once()
            for current_time in (160, 170):
                clock[0] = current_time
                with self.assertRaisesRegex(CliError, "^cli_single_result_output_limit$"):
                    await poller.poll_once()
                self.assertEqual(history.inbound_window("main", source, current_time), (100, 100))
                self.assertEqual(history.pending_inbound("main", source), [])
                self.assertFalse(history.inbound_seen("main", source, "om_first"))
            self.assertEqual(queries, [("100", "110", size, token) for size, token in (
                (50, None), (25, None), (12, None), (6, None), (3, None), (1, None), (1, "blocked"), (1, "blocked"),
            )])
            await poller.close()

    async def test_partial_paginated_scan_restarts_at_durable_watermark(self):
        clock = [100]
        runner = _Runner()
        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        runner.pages = {None: {"items": [], "has_more": True, "page_token": "next"},
                        "next": {"items": [], "has_more": False}}
        with TemporaryDirectory(dir=Path.cwd()) as root, patch("agent_companion.channels.feishu.channel._MAX_PAGES", 1):
            db = Path(root) / "test.sqlite3"
            with closing(History(db)) as history:
                poller = _poller(history, self.fail, cli, _ASR(), clock)
                await poller.poll_once()
                clock[0] = 110
                await poller.poll_once()
            clock[0] = 130
            with closing(History(db)) as history:
                await _poller(history, self.fail, cli, _ASR(), clock).poll_once()
            params = json.loads(runner.calls[-1][0][5])
            self.assertEqual((params["start_time"], params["end_time"], params.get("page_token")), ("100", "130", None))

    async def test_total_window_state_limit_can_narrow_without_skipping_history(self):
        clock = [100]
        source = "feishu-private"
        early = _raw("om_early", 105_000, "text", {"text": "第一段"})
        late = _raw("om_late", 130_000, "text", {"text": "第二段"})
        queries = []

        async def runner(argv, data, timeout):
            params = json.loads(argv[5])
            start, end = int(params["start_time"]), int(params["end_time"])
            token = params.get("page_token")
            queries.append((start, end, token))
            if end == 140:
                return _success({"items": [] if token else [early], "has_more": True,
                                 "page_token": "overflow" if token else "next"})
            return _success({"items": [early, late], "has_more": False})

        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        with closing(History(":memory:")) as history, patch("agent_companion.channels.feishu.channel._MAX_WINDOW_PAGES", 2):
            collected = []

            def submit(batch_id):
                collected.append(history.consume_inbound("main", batch_id))

            poller = _poller(history, submit, cli, _ASR(), clock)
            await poller.poll_once()
            clock[0] = 140
            with self.assertRaisesRegex(CliError, "^cli_window_limit$"):
                await poller.poll_once()
            self.assertEqual(history.inbound_window("main", source, 140), (100, 100))
            self.assertEqual(collected, [])
            clock[0] = 150
            await poller.poll_once()
            self.assertEqual(history.inbound_window("main", source, 150), (100, 120))
            clock[0] = 160
            await poller.poll_once()
            self.assertEqual([row.message.content for row in history.tail("main", 5)], ["第一段", "第二段"])
            self.assertEqual(queries, [(100, 140, None), (100, 140, "next"), (100, 120, None), (110, 160, None)])

    async def test_persisted_batch_is_not_visible_before_busy_session_boundary(self):
        started, release = asyncio.Event(), asyncio.Event()

        class BlockedModel(_Model):
            async def stream(self, messages, tools, *, thinking):
                self.calls.append(tuple(messages))
                if len(self.calls) == 1:
                    started.set()
                    await release.wait()
                yield TextDelta("模拟回复")

        clock = [100]
        runner, asr, model = _Runner(), _ASR(), BlockedModel()
        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        runner.pages[None] = {"items": [_raw("om_next", 101_000, "text", {"text": "稍后处理"})],
                              "has_more": False}
        with closing(History(":memory:")) as history:
            channel = FeishuChannel(cli)
            self.addAsyncCleanup(channel.close)
            async with running_agent(Settings(), history, model, channel, session_id="main",
                                     roles=demo_roles()) as (main, _):
                submit = main.submit_persisted_batch
                poller = _poller(history, submit, cli, asr, clock)
                await poller.poll_once()
                main.submit_batch([Message("user", "先处理")])
                await asyncio.wait_for(started.wait(), 1)
                clock[0] = 110
                await poller.poll_once()
                self.assertEqual([row.message.content for row in history.tail("main", 5)], ["先处理"])
                clock[0] = 120
                await poller.poll_once()  # Recovery scan must not queue the same pending ID twice.
                release.set()
                await main.wait_idle()
                self.assertEqual(len(model.calls), 2)
                self.assertEqual([row.message.content for row in history.tail("main", 5)],
                                 ["先处理", "模拟回复", "稍后处理", "模拟回复"])

    async def test_restart_recovers_pending_then_deduplicates_consumed(self):
        clock = [100]
        runner, asr, model = _Runner(), _ASR(), _Model()
        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        runner.pages[None] = {"items": [_raw("om_once", 103_000, "text", {"text": "只消费一次"})], "has_more": False}
        with TemporaryDirectory(dir=Path.cwd()) as root:
            path = Path(root) / "test.sqlite3"
            with closing(History(path)) as history:
                seen = []
                poller = _poller(history, seen.append, cli, asr, clock)
                await poller.poll_once()
                clock[0] = 110
                await poller.poll_once()
                self.assertEqual(len(seen), 1)  # durable acceptance; simulate exit before consumption
                self.assertEqual(history.tail("main", 5), [])
            for expected_calls in (1, 1):
                clock[0] += 1
                with closing(History(path)) as history:
                    channel = FeishuChannel(cli)
                    self.addAsyncCleanup(channel.close)
                    async with running_agent(Settings(), history, model, channel, session_id="main",
                                             roles=demo_roles()) as (main, _):
                        submit = main.submit_persisted_batch
                        await _poller(history, submit, cli, asr, clock).poll_once()
                        await main.wait_idle()
                        self.assertEqual(len(model.calls), expected_calls)
                        self.assertEqual([row.message.content for row in history.tail("main", 20)],
                                         ["只消费一次", "模拟回复"])

    async def test_media_budget_keeps_small_image_and_text_in_one_ordered_batch(self):
        clock = [100]
        runner, asr, model = _Runner(), _ASR(), _Model()
        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        post = _raw("om_rich_budget", 102_000, "post", {"content": [[
            {"tag": "text", "text": "前"}, {"tag": "img", "image_key": "img_a"},
            {"tag": "img", "image_key": "img_a"}, {"tag": "text", "text": "后"},
        ]]})
        with closing(History(":memory:")) as history:
            channel = FeishuChannel(cli)
            self.addAsyncCleanup(channel.close)
            async with running_agent(Settings(), history, model, channel, session_id="main",
                                     roles=demo_roles()) as (main, _):
                submit = main.submit_persisted_batch
                poller = FeishuPoller(cli, asr, channel, chat_id=_CHAT, sender_id=_PEER, own_id=_SELF,
                                      session_id="main", history=history, submit=submit,
                                      private_started_ms=100_000, clock=lambda: clock[0], batch_budget=470)
                await poller.poll_once()
                runner.pages[None]["items"] = [post, _raw("om_text_budget", 101_000, "text", {"text": "同批文字"})]
                clock[0] = 110
                await poller.poll_once()
                await main.wait_idle()
            self.assertEqual(len(model.calls), 1)
            self.assertEqual(model.calls[0][0].content, "同批文字")
            before, image, omitted, after = model.calls[0][1].content
            self.assertEqual((before, omitted, after), (Text("前"), OMITTED_INPUT_IMAGE, Text("后")))
            self.assertIsInstance(image, Image)

    def test_sample_config_offline_and_id_validation(self):
        settings = load_settings(Path("config.example.toml"))
        self.assertIsNone(settings.feishu)
        self.assertIsNone(settings.llm)
        self.assertIsNone(settings.asr)
        self.assertIsNone(settings.tts)
        with self.assertRaises(ValueError):
            FeishuSettings(own_open_id="invalid")

if __name__ == "__main__":
    unittest.main()
