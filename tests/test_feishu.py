import asyncio
from collections import deque
from base64 import b64decode
from contextlib import closing
from datetime import datetime, timezone
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
from agent_companion.channels.feishu.channel import FeishuChannel, FeishuInput, FeishuPoller, private_session_id
from agent_companion.channels.feishu.cli import CliError, CliStdoutLimit, CommandResult, FeishuCLI, FeishuDiscovery, run_command
from agent_companion.config import FeishuSettings, Settings, load_settings
from agent_companion.protocol import (OMITTED_INPUT_IMAGE, ChannelUser, DeliveryStatus, Image, Message, Receipt,
                                      ReplyDelivery, ReplyInfo, Text, TextDelta)
from agent_companion.runtime.subtasks import demo_roles
from agent_companion.speech.asr import AsrError
from agent_companion.storage.sqlite import History


_CHAT, _PEER, _SELF = "oc_synthetic", "ou_peer", "ou_self"
_SESSION = private_session_id(ChannelUser(_PEER))
_SEARCH = "/open-apis/im/v1/messages/search"
_MGET = "/open-apis/im/v1/messages/mget"
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


def _hit(row, *, p2p=True):
    return {"meta_data": {"message_id": row["message_id"], "type": row["msg_type"],
                          "chat_id": row["chat_id"], "from_id": row["sender"]["id"], "is_p2p_chat": p2p,
                          "create_time": datetime.fromtimestamp(int(row["create_time"]) / 1000,
                                                                timezone.utc).isoformat()}}


def _search_bounds(stdin):
    times = json.loads(stdin)["filter"]["time_range"]
    return tuple(int(datetime.fromisoformat(times[k]).timestamp()) for k in ("start_time", "end_time"))


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
        if argv[3] == _SEARCH:
            page = self.pages[json.loads(argv[5]).get("page_token")]
            if isinstance(page, dict):
                return _success({**page, "items": [row if "meta_data" in row else _hit(
                    row, p2p=row["chat_id"] != "oc_group") for row in page.get("items", [])
                    if "meta_data" in row or row["sender"]["sender_type"] == "user"]})
            return page
        if argv[3] == _MGET:
            requested = json.loads(argv[5])["message_ids"]
            found = {row["message_id"]: row for page in self.pages.values() if isinstance(page, dict)
                     for row in page.get("items", []) if "message_id" in row}
            return _success({"items": [found[mid] for mid in requested if mid in found]})
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


def _poller(history, main, cli, asr, clock, *, batch_budget=None, initial_ms=100_000):
    discovery = FeishuDiscovery(cli._executable, timeout=cli._timeout, runner=cli._runner)
    channel = FeishuChannel(cli)
    submitted = set()

    def notify(user):
        assert user == ChannelUser(_PEER)
        for batch in history.pending_inbound(_SESSION, "feishu-private"):
            if batch not in submitted:
                main(batch)
                submitted.add(batch)

    inputs = FeishuInput(discovery, asr, history, **({} if batch_budget is None else {"batch_budget": batch_budget}))

    class DrainedPoller(FeishuPoller):
        async def poll_once(self):
            await super().poll_once()
            user = ChannelUser(_PEER)
            while await inputs.process_next(user, channel, lambda: notify(user)):
                pass

        async def close(self):
            await channel.close()

    return DrainedPoller(discovery, account=ChannelUser(_SELF), history=history, notify_user=notify,
                         private_started_ms=initial_ms, clock=lambda: clock[0])


class FeishuCLITests(unittest.IsolatedAsyncioTestCase):
    async def test_raw_requests_and_stdin_not_argv(self):
        calls = []

        async def runner(argv, stdin, timeout):
            calls.append((argv, stdin, timeout))
            return _success({"message_id": "om_example", "items": [], "has_more": False})

        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        reader = FeishuDiscovery("lark-cli", timeout=2, runner=runner)
        self.assertEqual(await reader.search_page(10, 20, 'opaque+/?="'), ([], None))
        self.assertEqual(await reader.messages(("om_one", "om_two")), [])
        text = '多行 "hello"\n--as bot; not an option'
        self.assertEqual(await cli.send_text(text), DeliveryStatus.FULL)
        first, stdin, _ = calls[0]
        self.assertEqual(first[:5], ("lark-cli", "api", "POST", _SEARCH, "--params"))
        self.assertEqual(json.loads(first[5]), {"user_id_type": "open_id", "page_size": 20,
                                               "page_token": 'opaque+/?="'})
        self.assertEqual(json.loads(stdin), {"query": "", "filter": {"chat_type": "p2p", "from_types": ["user"],
            "time_range": {"start_time": "1970-01-01T00:00:10+00:00", "end_time": "1970-01-01T00:00:20+00:00"}}})
        read, stdin, _ = calls[1]
        self.assertIsNone(stdin)
        self.assertEqual(read[1:5], ("api", "GET", _MGET, "--params"))
        self.assertEqual(json.loads(read[5]), {"message_ids": ["om_one", "om_two"], "user_id_type": "open_id"})
        second, stdin, _ = calls[2]
        self.assertNotIn(text, " ".join(second))
        self.assertEqual(second[6:8], ("--data", "-"))
        self.assertEqual(second[-4:], ("--as", "user", "--format", "json"))
        body = json.loads(stdin)
        self.assertEqual((body["receive_id"], body["msg_type"], json.loads(body["content"])),
                          (_PEER, "text", {"text": text}))

    async def test_transcript_uses_literal_bold_post_in_stdin(self):
        calls = []

        async def runner(argv, stdin, timeout):
            calls.append((argv, stdin))
            return _success({"message_id": "om_post", "chat_id": _CHAT})

        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        text = '  多行 "hello" &amp; **注意**\r\n<at>\\路径\n🙂\n'
        payload = cli.prepare_transcript(text)
        assert payload is not None
        self.assertEqual(await cli.send_prepared_transcript(payload), DeliveryStatus.FULL)
        (argv, stdin), = calls
        self.assertIs(stdin, payload)
        self.assertEqual(argv, ("lark-cli", "api", "POST", "/open-apis/im/v1/messages", "--params",
                                '{"receive_id_type":"open_id"}', "--data", "-", "--as", "user", "--format", "json"))
        body = json.loads(stdin)
        self.assertEqual((body["receive_id"], body["msg_type"], json.loads(body["content"])),
                         (_PEER, "post", {"zh_cn": {"content": [[{"tag": "text", "text": text, "style": ["bold"]}]]}}))

    def test_transcript_preparation_rejects_invalid_encoding_or_oversized_post(self):
        runner = _Runner()
        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        for text in ("\ud800", "字" * 10_000, '"' * 10_000, "a" * 30_001):
            with self.subTest(length=len(text)):
                self.assertIsNone(cli.prepare_transcript(text))
        self.assertEqual(runner.calls, [])

    async def test_search_page_size_bounds(self):
        runner = _Runner()
        cli = FeishuDiscovery("lark-cli", timeout=2, runner=runner)
        for size in (1, 50):
            self.assertEqual(await cli.search_page(10, 20, None, page_size=size), ([], None))
        for size in (-1, 0, 51, 1.5, True, "25"):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "invalid Feishu search page size"):
                await cli.search_page(10, 20, None, page_size=size)
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

                cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
                self.assertEqual((await cli.send_text("第一第二"), num), (expected, 1))

    async def test_bad_resources_are_not_read_or_sent(self):
        runner = _Runner()
        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
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
            cli = FeishuCLI(str(worker), ChannelUser(_PEER), timeout=3)
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
                channel = FeishuChannel(FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner))
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
        channel = FeishuChannel(FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner))
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

        channel = FeishuChannel(FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner))
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

        channel = FeishuChannel(FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner))
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
                cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
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
                self.assertEqual(outbound, [(_PEER, saved, Message("assistant", saved))] if expected_to_send else [])
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
        channel = None

        def capture_channel(*args, **kwargs):
            nonlocal channel
            channel = FeishuChannel(*args, **kwargs)
            return channel

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
            poller._notify_user(ChannelUser(_PEER))
            info = ReplyInfo("simulated", "main", "run", 0)
            channel.enqueue(info, "当前")
            await active.wait()
            channel.enqueue(info, "不能开始")
            poll_ready.set()
            try:
                await asyncio.Future()
            finally:
                poller._before_close()
                poll_cleaning.set()
                await release_poll.wait()

        settings = Settings(feishu=FeishuSettings())
        discovery = private_discovery(_CHAT, _PEER, _SELF, cli=CLI("lark-cli", ChannelUser(_PEER), timeout=2))
        with (closing(History(":memory:")) as history,
              patch("agent_companion.feishu_private.FeishuChannel", side_effect=capture_channel),
              patch.object(FeishuPoller, "run", poll),
              self.assertLogs("agent_companion.channels.feishu.channel", level="WARNING")):
            app = asyncio.create_task(run_feishu(settings, history, _Model(), _ASR(), discovery=discovery,
                                                 roles=demo_roles()))
            try:
                await asyncio.wait_for(poll_ready.wait(), 1)
                app.cancel()
                await asyncio.wait_for(poll_cleaning.wait(), 1)
                self.assertTrue(send_cancelled.is_set())
                self.assertTrue(channel._closed)
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
        channel = None

        def capture_channel(*args, **kwargs):
            nonlocal channel
            channel = FeishuChannel(*args, **kwargs)
            return channel

        class Runner(_Runner):
            async def __call__(self, argv, data, timeout):
                if argv[1:4] != ("api", "POST", "/open-apis/im/v1/messages"):
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
        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        real_poll_once = FeishuPoller.poll_once

        async def poll_once(poller):
            await real_poll_once(poller)
            clock[0] = 110
            await real_poll_once(poller)
            await runner.notice_entered.wait()  # 实际按用户任务归一化失败并启动通知。
            info = ReplyInfo("simulated", "main", "run", 0)
            channel.enqueue(info, "当前")
            channel.enqueue(info, "不能开始")
            await active.wait()
            raise RuntimeError("synthetic poll failure")

        settings = Settings(feishu=FeishuSettings())
        discovery = private_discovery(_CHAT, _PEER, _SELF, cli=cli, search_runner=runner)
        with (closing(History(":memory:")) as history,
              patch("agent_companion.feishu_private.FeishuChannel", side_effect=capture_channel),
              patch.object(FeishuPoller, "poll_once", poll_once),
              patch("agent_companion.app.time", side_effect=lambda: clock[0]),
              patch("agent_companion.feishu_private.time", side_effect=lambda: clock[0])):
            app = asyncio.create_task(run_feishu(settings, history, _Model(), asr, discovery=discovery,
                                                 roles=demo_roles()))
            try:
                await asyncio.wait_for(notice_cleaning.wait(), 1)
                release_send.set()
                await asyncio.wait_for(send_cleaned.wait(), 1)
                self.assertFalse(app.done())
                self.assertEqual(sent, ["当前"])
                self.assertTrue(send_cancelled.is_set())
                self.assertTrue(channel._closed)
            finally:
                release_send.set()
                runner.release_notice.set()
                with self.assertRaises(ExceptionGroup) as raised:
                    await asyncio.wait_for(app, 1)
                self.assertEqual([str(exc) for exc in raised.exception.exceptions], ["synthetic poll failure"])


class FeishuPollingTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_diagnostics_distinguish_empty_filtered_duplicate_and_recorded(self):
        clock, runner = [110], _Runner()
        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        peer = _raw("om_diagnostic", 111_000, "text", {"text": "synthetic diagnostic body"})
        rows = [peer, peer, _raw("om_own", 112_000, "text", {"text": "own"}, sender=_SELF),
                _raw("om_group", 113_000, "text", {"text": "group"}, chat="oc_group"),
                _raw("om_outside", 120_000, "text", {"text": "outside"}),
                _hit(_raw("om_bot", 114_000, "text", {"text": "bot"},
                          sender="cli_synthetic", sender_type="bot")),
                _raw("om_unsupported", 115_000, "synthetic-private-kind", {"text": "unsupported"})]
        with (closing(History(":memory:")) as history,
              self.assertLogs("agent_companion.channels.feishu.channel", level="DEBUG") as logs):
            poller = _poller(history, lambda batch: None, cli, _ASR(), clock)
            await poller.poll_once()
            clock[0] = 120
            runner.pages[None] = {"items": rows[:2], "has_more": True, "page_token": "synthetic-private-cursor"}
            runner.pages["synthetic-private-cursor"] = {"items": rows[2:], "has_more": False}
            await poller.poll_once()
        summaries = [record.getMessage() for record in logs.records if record.msg.startswith("Feishu search window")]
        self.assertEqual(len(summaries), 2)
        self.assertIn("outcome=ok stage=record start_utc=1970-01-01T00:01:40+00:00", summaries[0])
        self.assertIn("end_utc=1970-01-01T00:01:50+00:00 pages=1 returned=0 filtered=0 duplicates=0", summaries[0])
        self.assertIn("filter_reasons=none", summaries[0])
        self.assertIn("pages=2 returned=7 filtered=5 duplicates=1", summaries[1])
        self.assertIn("filter_reasons=bot_sender:1,non_p2p:1,out_of_window:1,same_account:1,unsupported_type:1",
                      summaries[1])
        diagnostics = "\n".join(logs.output)
        self.assertIn("Feishu inbox recorded; source=search accepted=1 discarded=0 deferred=0 users=1", diagnostics)
        for private in (_SELF, _PEER, _CHAT, "om_diagnostic", "synthetic-private-cursor", "synthetic diagnostic body",
                        "cli_synthetic", "synthetic-private-kind"):
            self.assertNotIn(private, diagnostics)

    async def test_search_diagnostics_report_failed_stage_and_safe_cli_classification(self):
        clock, runner = [110], _Runner()
        runner.pages[None] = _failure("authorization", 99991672)

        async def reject_details(argv, data, timeout):
            if argv[3] == _MGET:
                return _failure("api", "synthetic private error code")
            return await runner(argv, data, timeout)

        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=reject_details)
        with (closing(History(":memory:")) as history,
              self.assertLogs("agent_companion.channels.feishu.channel", level="DEBUG") as logs):
            poller = _poller(history, lambda batch: None, cli, _ASR(), clock)
            with self.assertRaisesRegex(CliError, "^cli_rejected$"):
                await poller.poll_once()
            runner.pages[None] = {"items": [_raw("om_found", 101_000, "text", {"text": "synthetic body"})],
                                  "has_more": False}
            with self.assertRaisesRegex(CliError, "^cli_result_unknown$"):
                await poller.poll_once()
            self.assertEqual(history.feishu_inbound_window(0), (100, 100))
        self.assertIn("outcome=failed stage=search", logs.output[0])
        self.assertIn("category=authorization api_code=99991672", logs.output[0])
        self.assertIn("outcome=failed stage=details", logs.output[1])
        self.assertIn("category=api api_code=None", logs.output[1])
        self.assertNotIn("synthetic private", "\n".join(logs.output))

    async def test_idle_single_query_and_overlap_recovers_a_late_hit(self):
        clock, runner = [100], _Runner()
        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        with closing(History(":memory:")) as history:
            def accept(batch):
                history.consume_inbound(_SESSION, batch)

            poller = _poller(history, accept, cli, _ASR(), clock)
            await poller.poll_once()
            clock[0] = 220
            await poller.poll_once()
            self.assertEqual([argv[3] for argv, _ in runner.calls], [_SEARCH])
            clock[0] = 250
            runner.pages[None]["items"] = [_raw("om_late", 131_000, "text", {"text": "迟到"}),
                                           _raw("om_expired", 99_999, "text", {"text": "旧历史"})]
            await poller.poll_once()
            clock[0] = 251
            await poller.poll_once()
            self.assertEqual([row.message.content for row in history.tail(_SESSION, 10)], ["迟到"])
            self.assertEqual([argv[3] for argv, _ in runner.calls], [_SEARCH, _SEARCH, _MGET, _SEARCH])
            self.assertEqual(history.feishu_inbound_window(0), (100, 251))

    async def test_missing_and_mismatched_candidates_recover_after_restart_without_blocking(self):
        clock, requested = [110], []
        missing = _raw("om_missing", 101_100, "text", {"text": "迟到正文"})
        mismatch = _raw("om_mismatch", 101_200, "text", {"text": "迟到核验"})
        good = _raw("om_good", 102_000, "text", {"text": "同页正文"})
        later = _raw("om_later", 250_000, "text", {"text": "后续窗口"})
        deletion = _raw("om_deleted", 101_500, "text", {"text": "已撤回"}, deleted=True)
        search = [missing, mismatch, good, deletion, later]
        details = {row["message_id"]: row for row in (good, later, deletion)}
        details["om_mismatch"] = {**mismatch, "chat_id": "oc_wrong"}

        async def runner(argv, data, timeout):
            if argv[3] == _SEARCH:
                start, end = _search_bounds(data)
                return _success({"items": [_hit(row) for row in search if start <= int(row["create_time"]) // 1000 < end],
                                 "has_more": False})
            ids = json.loads(argv[5])["message_ids"]
            requested.append(ids)
            return _success({"items": [details[mid] for mid in ids if mid in details]})

        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        with TemporaryDirectory() as root:
            path = Path(root) / "synthetic.db"
            with closing(History(path)) as history, self.assertLogs(level="WARNING") as logs:
                poller = _poller(history, lambda batch: history.consume_inbound(_SESSION, batch), cli, _ASR(), clock)
                await poller.poll_once()
                self.assertEqual(history.feishu_inbound_window(0), (100, 110))
                self.assertEqual([row.message.content for row in history.tail(_SESSION, 20)], ["同页正文"])
                self.assertTrue(history.feishu_inbound_seen("om_deleted"))
                for mid in ("om_missing", "om_mismatch"):
                    self.assertFalse(history.feishu_inbound_seen(mid))
                    self.assertIsNotNone(history.feishu_recovery_candidate(mid))
                clock[0] = 119
                await poller.poll_once()
                self.assertEqual(len(requested), 1)
                self.assertNotIn("om_missing", " ".join(logs.output))
                self.assertNotIn("oc_wrong", " ".join(logs.output))
            with (closing(History(path)) as history,
                  patch("agent_companion.channels.feishu.channel._MAX_RECOVERY_BATCH", 1)):
                poller = _poller(history, lambda batch: history.consume_inbound(_SESSION, batch), cli, _ASR(), clock)
                clock[0] = 300  # 搜索窗口已过两个候选的原始时间。
                await poller.poll_once()
                self.assertEqual(requested[1:], [["om_later"], ["om_mismatch"]])
                self.assertEqual(history.feishu_inbound_window(0), (100, 300))
                self.assertFalse(history.feishu_inbound_seen("om_mismatch"))
                details.update(om_missing=missing, om_mismatch=mismatch)
                clock[0] = 301
                await poller.poll_once()
                clock[0] = 319
                await poller.poll_once()
                self.assertEqual(requested[-1], ["om_missing"])
                clock[0] = 320
                await poller.poll_once()
                self.assertEqual(requested[-1], ["om_mismatch"])
                self.assertEqual([row.message.content for row in history.tail(_SESSION, 20)],
                                 ["同页正文", "后续窗口", "迟到正文", "迟到核验"])
                self.assertEqual(history.due_feishu_candidates(9999, 10), [])

    async def test_single_oversized_detail_does_not_block_scan_or_other_recovery(self):
        clock, requests = [110], []
        poison = _raw("om_0_oversized", 101_000, "text", {"text": "超限后恢复"})
        recoverable = _raw("om_1_recoverable", 102_000, "text", {"text": "独立恢复"})
        regular = _raw("om_2_regular", 103_000, "text", {"text": "同页正常"})
        later = _raw("om_3_later", 115_000, "text", {"text": "后续窗口"})
        rows = {row["message_id"]: row for row in (poison, regular, recoverable, later)}

        async def runner(argv, data, timeout):
            if argv[3] == _SEARCH:
                start, end = _search_bounds(data)
                return _success({"items": [_hit(row) for row in rows.values()
                                           if start <= int(row["create_time"]) // 1000 < end], "has_more": False})
            ids = json.loads(argv[5])["message_ids"]
            requests.append(ids)
            if "om_0_oversized" in ids and clock[0] < 140:
                raise CliStdoutLimit("cli_output_limit")
            return _success({"items": [rows[mid] for mid in ids if mid != "om_1_recoverable" or clock[0] >= 120]})

        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        with TemporaryDirectory() as root:
            db = Path(root) / "synthetic.db"
            with closing(History(db)) as history, self.assertLogs(level="WARNING") as logs:
                poller = _poller(history, lambda batch: history.consume_inbound(_SESSION, batch), cli, _ASR(), clock)
                await poller.poll_once()
                self.assertEqual(requests, [["om_0_oversized", "om_2_regular", "om_1_recoverable"],
                                            ["om_0_oversized"], ["om_2_regular", "om_1_recoverable"]])
                self.assertEqual(history.feishu_inbound_window(0), (100, 110))
                self.assertIsNotNone(history.feishu_recovery_candidate("om_0_oversized"))
                self.assertFalse(history.feishu_inbound_seen("om_0_oversized"))
                self.assertEqual([row.message.content for row in history.tail(_SESSION, 10)], ["同页正常"])
                self.assertIn("single-message detail output limit", " ".join(logs.output))
                self.assertNotIn("om_0_oversized", " ".join(logs.output))
            with closing(History(db)) as history, self.assertLogs(level="WARNING") as logs:
                poller = _poller(history, lambda batch: history.consume_inbound(_SESSION, batch), cli, _ASR(), clock)
                clock[0] = 120
                await poller.poll_once()
                self.assertEqual(requests[3:], [["om_3_later"], ["om_0_oversized", "om_1_recoverable"],
                                                ["om_0_oversized"]])
                self.assertEqual(history.feishu_inbound_window(0), (100, 120))
                self.assertIn("single-message recovery output limit", " ".join(logs.output))
                self.assertFalse(history.feishu_inbound_seen("om_0_oversized"))
                self.assertEqual([hit.message_id for hit in history.due_feishu_candidates(120, 5)], ["om_1_recoverable"])
                clock[0] = 121
                await poller.poll_once()
                self.assertEqual(requests[6:], [["om_1_recoverable"]])
                self.assertEqual([row.message.content for row in history.tail(_SESSION, 10)],
                                 ["同页正常", "后续窗口", "独立恢复"])
                clock[0] = 139
                await poller.poll_once()
                self.assertEqual(len(requests), 7)
                clock[0] = 140
                await poller.poll_once()
                self.assertEqual(requests[7:], [["om_0_oversized"]])
                self.assertIsNone(history.feishu_recovery_candidate("om_0_oversized"))
                self.assertEqual([row.message.content for row in history.tail(_SESSION, 10)],
                                 ["同页正常", "后续窗口", "独立恢复", "超限后恢复"])

    async def test_corrected_recovery_metadata_is_persisted_and_reverified_once(self):
        clock, requests = [110], []
        mid = "om_corrected"
        old = _raw(mid, 101_200, "text", {"text": "旧来源"}, sender="ou_wrong", chat="oc_old")
        corrected = _raw(mid, 102_300, "text", {"text": "已更正"})
        hits, rows = [_hit(old)], []

        async def runner(argv, data, timeout):
            if argv[3] == _SEARCH:
                return _success({"items": hits, "has_more": False})
            requests.append(json.loads(argv[5])["message_ids"])
            return _success({"items": rows})

        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        with TemporaryDirectory() as root:
            db = Path(root) / "synthetic.db"
            with closing(History(db)) as history:
                poller = _poller(history, lambda batch: history.consume_inbound(_SESSION, batch), cli, _ASR(), clock)
                await poller.poll_once()
                prior = history.feishu_recovery_candidate(mid)
                self.assertEqual(prior.peer_open_id, "ou_wrong")
                clock[0] = 111
                await poller.poll_once()
                self.assertEqual(requests, [[mid]])  # 相同回看命中不提前到期。

                clock[0] = 120
                hits[:] = [_hit(corrected, p2p=False)]
                rows[:] = [corrected]
                await poller.poll_once()
                self.assertEqual(requests, [[mid], [mid]])  # 正常到期，但非 P2P 不能改写核验依据。
                self.assertEqual(history.feishu_recovery_candidate(mid), prior)
                self.assertFalse(history.feishu_inbound_seen(mid))

                clock[0] = 121
                hits[:] = [_hit({**corrected, "create_time": "121000"})]
                await poller.poll_once()  # 恰在结束边界的候选不可信。
                self.assertEqual(history.feishu_recovery_candidate(mid), prior)
                self.assertEqual(len(requests), 2)

                clock[0] = 122
                hits[:] = [_hit(old), _hit(corrected)]
                with self.assertRaisesRegex(CliError, "cli_conflicting_search_hit"):
                    await poller.poll_once()
                self.assertEqual(history.feishu_inbound_window(0), (100, 121))
                self.assertEqual(history.feishu_recovery_candidate(mid), prior)

                clock[0] = 123
                hits[:] = [_hit(corrected)]
                rows[:] = [old]
                await poller.poll_once()
                refreshed = history.feishu_recovery_candidate(mid)
                self.assertEqual((refreshed.peer_open_id, refreshed.created_second), (_PEER, 102))
                self.assertNotEqual(refreshed.chat_digest, prior.chat_digest)
                self.assertEqual(requests, [[mid], [mid], [mid]])  # 新依据提前重试，旧正文不通过。
                self.assertFalse(history.feishu_inbound_seen(mid))
                self.assertEqual(history.pending_feishu_users(), [])

                clock[0] = 124
                rows[:] = [corrected]
                await poller.poll_once()
                self.assertEqual(len(requests), 3)
            with closing(History(db)) as history:
                self.assertEqual(history.feishu_recovery_candidate(mid), refreshed)
                clock[0] = 163
                hits.clear()  # 不靠重启后再搜到该消息。
                poller = _poller(history, lambda batch: history.consume_inbound(_SESSION, batch), cli, _ASR(), clock)
                await poller.poll_once()
                self.assertTrue(history.feishu_inbound_seen(mid))
                self.assertIsNone(history.feishu_recovery_candidate(mid))
                clock[0] = 164
                hits[:] = [_hit(old), _hit(corrected)]
                await poller.poll_once()
                self.assertEqual(requests, [[mid], [mid], [mid], [mid]])
                self.assertEqual([row.message.content for row in history.tail(_SESSION, 10)], ["已更正"])

    async def test_valid_id_with_wrong_sender_type_timestamp_or_body_is_not_routed(self):
        clock = [110]
        good = _raw("om_checked", 101_400, "text", {"text": "verified"})
        variants = deque([
            {**good, "sender": {**good["sender"], "id": "ou_wrong"}},
            {**good, "sender": {**good["sender"], "sender_type": "app"}},
            {**good, "create_time": "102400"},
            {**good, "msg_type": "post"},
            {**good, "body": {}},
            good,
        ])

        async def runner(argv, data, timeout):
            if argv[3] == _SEARCH:
                return _success({"items": [_hit(good)], "has_more": False})
            return _success({"items": [variants.popleft()]})

        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        with closing(History(":memory:")) as history:
            poller = _poller(history, lambda batch: history.consume_inbound(_SESSION, batch), cli, _ASR(), clock)
            for step in range(5):
                await poller.poll_once()
                self.assertFalse(history.feishu_inbound_seen("om_checked"))
                self.assertEqual(history.tail(_SESSION, 10), [])
                clock[0] += 10 * 2 ** step
            await poller.poll_once()
            self.assertEqual([row.message.content for row in history.tail(_SESSION, 10)], ["verified"])

    async def test_details_respect_batch_limit_and_keep_already_verified_parts(self):
        clock, requested = [110], []
        rows = {f"om_{i}": _raw(f"om_{i}", 101_000 + i, "text", {"text": str(i)}) for i in range(51)}
        broken = deque(["auth", "network", "null", "omitted", "duplicate"])

        async def runner(argv, data, timeout):
            if argv[3] == _SEARCH:
                token = json.loads(argv[5]).get("page_token")
                hits = [_hit(row) for row in rows.values()]
                return _success({"items": hits[:20], "has_more": True, "page_token": "middle"} if token is None else
                                {"items": hits[20:40], "has_more": True, "page_token": "tail"} if token == "middle" else
                                {"items": hits[40:], "has_more": False})
            ids = json.loads(argv[5])["message_ids"]
            requested.append(ids)
            if broken and "om_50" in ids:
                failure = broken.popleft()
                if failure == "auth":
                    return _failure("authorization")
                if failure == "network":
                    raise CliError("cli_timeout")
                if failure == "null":
                    return _success({"items": None})
                if failure == "omitted":
                    return _success({})
                return _success({"items": [rows["om_50"], rows["om_50"]]})
            return _success({"items": [rows[mid] for mid in ids]})

        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        with closing(History(":memory:")) as history:
            poller = _poller(history, lambda batch: history.consume_inbound(_SESSION, batch), cli, _ASR(), clock)
            for error in ("cli_rejected", "cli_timeout", *(["cli_invalid_message_details"] * 3)):
                with self.assertRaisesRegex(CliError, error):
                    await poller.poll_once()
                self.assertFalse(history.feishu_inbound_seen("om_0"))
                self.assertEqual(history.feishu_inbound_window(0), (100, 100))
                self.assertEqual(history.pending_feishu_users(), [])
            await poller.poll_once()
            self.assertEqual(requested, [[f"om_{i}" for i in range(50)], *([["om_50"]] * 6)])
            self.assertEqual([row.message.content for row in history.tail(_SESSION, 60)], [str(i) for i in range(51)])

    async def test_cyclic_cursor_never_fetches_details_or_advances(self):
        clock, runner = [110], _Runner()
        row = _raw("om_seen", 101_000, "text", {"text": "合成"})
        runner.pages = {key: {"items": [row], "has_more": True, "page_token": "cycle"} for key in (None, "cycle")}
        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        with closing(History(":memory:")) as history:
            poller = _poller(history, self.fail, cli, _ASR(), clock)
            with self.assertRaisesRegex(CliError, "cli_page_cycle"):
                await poller.poll_once()
            self.assertEqual([argv[3] for argv, _ in runner.calls], [_SEARCH, _SEARCH])
            self.assertEqual(history.feishu_inbound_window(0), (100, 100))
            self.assertFalse(history.feishu_inbound_seen("om_seen"))

    async def test_across_pages_order_filter_post_and_delayed_voice(self):
        clock = [100]
        runner, asr, model = _Runner(), _ASR(), _Model()
        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        audio = _raw("om_voice", 102_000, "audio", {"file_key": "file_a"})
        rich = _raw("om_rich", 103_000, "post", {"zh_cn": {"title": "标题", "content": [[
            {"tag": "text", "text": "前"}, {"tag": "img", "image_key": "img_a"},
            {"tag": "a", "text": "后", "href": "https://example.invalid/link"}
        ], [{"tag": "text", "text": "次行"}]]}})
        runner.pages = {None: {"items": [audio, _raw("om_text", 101_000, "text", {"text": "先到"}),
                                        _raw("om_own", 103_100, "text", {"text": "自身"}, sender=_SELF),
                                        _raw("om_wrong", 102_100, "text", {"text": "其他频道"}, chat="oc_group"),
                                        _raw("om_bot", 102_100, "text", {"text": "应用"}, sender_type="app"),
                                        _raw("om_deleted", 102_100, "text", {"text": "撤回"}, deleted=True)],
                               "has_more": True, "page_token": "next"},
                        "next": {"items": [audio, rich, _raw("om_pic", 103_000, "image", {"image_key": "img_a"})],
                                 "has_more": False}}
        asr.release.clear()
        with closing(History(":memory:")) as history:
            channel = FeishuChannel(cli)
            self.addAsyncCleanup(channel.close)
            async with running_agent(Settings(), history, model, channel, session_id=_SESSION,
                                     roles=demo_roles()) as (main, _):
                submit = main.submit_persisted_batch
                poller = _poller(history, submit, cli, asr, clock)
                await poller.poll_once()
                clock[0] = 110
                finish = history.finish_feishu_raw
                commit_rejected = asyncio.Event()
                can_commit = False

                def finish_with_lock(*args):
                    if not can_commit:
                        commit_rejected.set()
                        raise sqlite3.OperationalError("database is locked")
                    return finish(*args)

                with patch.object(history, "finish_feishu_raw", side_effect=finish_with_lock):
                    polling = asyncio.create_task(poller.poll_once())
                    await asyncio.wait_for(asr.entered.wait(), 1)
                    self.assertEqual(history.feishu_inbound_window(0), (100, 110))
                    self.assertEqual(model.calls, [])
                    self.assertEqual(history.tail(_SESSION, 20), [])
                    self.assertEqual(runner.posts, [])
                    asr.release.set()
                    await asyncio.wait_for(commit_rejected.wait(), 1)
                    self.assertEqual(runner.posts, [])
                    can_commit = True
                    await asyncio.wait_for(polling, 1)
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
                self.assertFalse(history.feishu_inbound_seen("om_echo"))
                self.assertEqual([row.message.content for row in history.tail(_SESSION, 20)].count("模拟转写"), 1)
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
                cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
                with closing(History(":memory:")) as history:
                    submitted = asyncio.Event()
                    accepted = []

                    def submit(batch):
                        history.consume_inbound(_SESSION, batch)
                        accepted.append(batch)
                        submitted.set()

                    poller = _poller(history, submit, cli, asr, clock)
                    await poller.poll_once()
                    clock[0] = 110
                    await poller.poll_once()
                    await asyncio.wait_for(runner.notice_entered.wait(), 1)
                    self.assertTrue(submitted.is_set())
                    self.assertEqual([row.message.content for row in history.tail(_SESSION, 20)], ["前", "后"])
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
        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
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
            self.assertEqual(poller._window, None)
            self.assertEqual(history.feishu_inbound_window(0), (100, 110))
            await poller.close()

    async def test_page_budget_resumes_fixed_window_and_commits_only_once_complete(self):
        clock = [100]
        runner = _Runner()
        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
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
                    self.assertEqual(history.feishu_inbound_window(current_time), (100, 100))
                    self.assertEqual(accepted, [])
            self.assertEqual([json.loads(argv[5]).get("page_token") for argv, _ in runner.calls if argv[3] == _SEARCH],
                             [None, "next", "last"])
            self.assertEqual([json.loads(argv[5])["message_ids"] for argv, _ in runner.calls if argv[3] == _MGET],
                             [["om_first", "om_second", "om_third"]])
            self.assertEqual(history.feishu_inbound_window(130), (100, 110))
            self.assertEqual(len(accepted), 1)
            history.consume_inbound(_SESSION, accepted[0])
            self.assertEqual([row.message.content for row in history.tail(_SESSION, 5)], ["最先", "同一毫秒", "最后"])
            await poller.close()

    async def test_single_result_stdout_limit_does_not_skip_or_commit(self):
        clock = [100]
        source = "feishu-private"
        queries = []
        first = _raw("om_first", 101_000, "text", {"text": "可读取"})

        async def runner(argv, data, timeout):
            params = json.loads(argv[5])
            size, token = params["page_size"], params.get("page_token")
            queries.append((*_search_bounds(data), size, token))
            if size == 1 and token is None:
                return _success({"items": [_hit(first)], "has_more": True, "page_token": "blocked"})
            raise CliStdoutLimit("cli_output_limit")

        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        with closing(History(":memory:")) as history:
            poller = _poller(history, self.fail, cli, _ASR(), clock)
            await poller.poll_once()
            for current_time in (110, 120, 130, 140):
                clock[0] = current_time
                await poller.poll_once()
            for current_time in (160, 220):
                clock[0] = current_time
                with self.assertRaisesRegex(CliError, "^cli_single_result_output_limit$"):
                    await poller.poll_once()
                self.assertEqual(history.feishu_inbound_window(current_time), (100, 100))
                self.assertEqual(history.pending_inbound(_SESSION, source), [])
                self.assertFalse(history.feishu_inbound_seen("om_first"))
                sent_count = len(queries)
                clock[0] += 10
                await poller.poll_once()
                self.assertEqual(len(queries), sent_count)
            self.assertEqual(queries, [(100, 110, size, token) for size, token in (
                (20, None), (10, None), (5, None), (2, None), (1, None), (1, "blocked"), (1, "blocked"),
            )])
            await poller.close()

    async def test_partial_paginated_scan_restarts_at_durable_watermark(self):
        clock = [100]
        runner = _Runner()
        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
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
            self.assertIsNone(json.loads(runner.calls[-1][0][5]).get("page_token"))

    async def test_total_window_state_limit_can_narrow_without_skipping_history(self):
        clock = [100]
        source = "feishu-private"
        early = _raw("om_early", 105_000, "text", {"text": "第一段"})
        late = _raw("om_late", 130_000, "text", {"text": "第二段"})
        queries = []

        async def runner(argv, data, timeout):
            params = json.loads(argv[5])
            if argv[3] == _MGET:
                return _success({"items": [item for item in (early, late) if item["message_id"] in params["message_ids"]]})
            start, end = _search_bounds(data)
            token = params.get("page_token")
            queries.append((start, end, token))
            if (start, end) == (100, 140):
                return _success({"items": [] if token else [_hit(early)], "has_more": True,
                                 "page_token": "overflow" if token else "next"})
            return _success({"items": [_hit(early), _hit(late)], "has_more": False})

        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        with closing(History(":memory:")) as history, patch("agent_companion.channels.feishu.channel._MAX_WINDOW_PAGES", 2):
            collected = []

            def submit(batch_id):
                collected.append(history.consume_inbound(_SESSION, batch_id))

            poller = _poller(history, submit, cli, _ASR(), clock)
            await poller.poll_once()
            clock[0] = 140
            with self.assertRaisesRegex(CliError, "^cli_window_limit$"):
                await poller.poll_once()
            self.assertEqual(history.feishu_inbound_window(140), (100, 100))
            self.assertEqual(collected, [])
            clock[0] = 150
            await poller.poll_once()
            self.assertEqual(history.feishu_inbound_window(150), (100, 120))
            clock[0] = 160
            await poller.poll_once()
            self.assertEqual([row.message.content for row in history.tail(_SESSION, 5)], ["第一段", "第二段"])
            self.assertEqual(queries, [(100, 140, None), (100, 140, "next"), (100, 120, None), (120, 140, None)])

    async def test_dense_overlap_is_bisected_before_scanning_right(self):
        clock = [340]
        rows = [_raw(f"om_{second}", 1000 * second, "text", {"text": str(second)}) for second in (200, 265, 315)]
        queries = []

        async def runner(argv, data, timeout):
            params = json.loads(argv[5])
            if argv[3] == _MGET:
                return _success({"items": [row for row in rows if row["message_id"] in params["message_ids"]]})
            start, end = _search_bounds(data)
            token = params.get("page_token")
            queries.append((start, end, token))
            hits = [_hit(row) for row in rows if start <= int(row["create_time"]) // 1000 < end]
            if len(hits) > 1:
                return _success({"items": hits[:1], "has_more": True, "page_token": "overflow" if token else "next"})
            return _success({"items": hits, "has_more": False})

        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        with closing(History(":memory:")) as history, patch("agent_companion.channels.feishu.channel._MAX_WINDOW_PAGES", 2):
            history.feishu_inbound_window(100)
            history.record_feishu_window(300, now=300, verified={}, discarded=(), retry=())
            poller = _poller(history, lambda batch: history.consume_inbound(_SESSION, batch), cli, _ASR(), clock)
            for expected_error, through in ((True, 300), (False, 300), (True, 300), (False, 300), (False, 340)):
                if expected_error:
                    with self.assertRaisesRegex(CliError, "cli_window_limit"):
                        await poller.poll_once()
                else:
                    await poller.poll_once()
                self.assertEqual(history.feishu_inbound_window(0), (100, through))
                clock[0] += 10
            self.assertEqual(queries, [(180, 340, None), (180, 340, "next"), (180, 260, None),
                                       (260, 340, None), (260, 340, "next"), (260, 300, None), (300, 340, None)])
            self.assertEqual([row.message.content for row in history.tail(_SESSION, 5)], ["200", "265", "315"])

    async def test_dense_single_second_backs_off_without_skipping(self):
        clock = [101]
        runner = _Runner()
        queries = []

        async def capture(argv, data, timeout):
            queries.append(_search_bounds(data))
            return await runner(argv, data, timeout)

        runner.pages[None] = {"items": [], "has_more": True, "page_token": "overflow"}
        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=capture)
        with closing(History(":memory:")) as history, patch("agent_companion.channels.feishu.channel._MAX_WINDOW_PAGES", 1):
            poller = _poller(history, self.fail, cli, _ASR(), clock)
            with self.assertRaisesRegex(CliError, "cli_single_second_window_limit"):
                await poller.poll_once()
            clock[0] = 120
            await poller.poll_once()
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(history.feishu_inbound_window(0), (100, 100))
            runner.pages[None] = {"items": [], "has_more": False}
            clock[0] = 161
            await poller.poll_once()
            self.assertEqual(history.feishu_inbound_window(0), (100, 101))
            self.assertEqual(queries, [(100, 101), (100, 101)])

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
        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        runner.pages[None] = {"items": [_raw("om_next", 101_000, "text", {"text": "稍后处理"})],
                              "has_more": False}
        with closing(History(":memory:")) as history:
            channel = FeishuChannel(cli)
            self.addAsyncCleanup(channel.close)
            async with running_agent(Settings(), history, model, channel, session_id=_SESSION,
                                     roles=demo_roles()) as (main, _):
                submit = main.submit_persisted_batch
                poller = _poller(history, submit, cli, asr, clock)
                await poller.poll_once()
                main.submit_batch([Message("user", "先处理")])
                await asyncio.wait_for(started.wait(), 1)
                clock[0] = 110
                await poller.poll_once()
                self.assertEqual([row.message.content for row in history.tail(_SESSION, 5)], ["先处理"])
                clock[0] = 120
                await poller.poll_once()  # Recovery scan must not queue the same pending ID twice.
                release.set()
                await main.wait_idle()
                self.assertEqual(len(model.calls), 2)
                self.assertEqual([row.message.content for row in history.tail(_SESSION, 5)],
                                 ["先处理", "模拟回复", "稍后处理", "模拟回复"])

    async def test_restart_recovers_pending_then_deduplicates_consumed(self):
        clock = [100]
        runner, asr, model = _Runner(), _ASR(), _Model()
        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
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
                self.assertEqual(history.tail(_SESSION, 5), [])
            # 不可依赖搜索还能找到这条消息才恢复待处理输入。
            runner.pages[None] = {"items": [], "has_more": False}
            clock[0] = 400
            for expected_calls in (1, 1):
                clock[0] += 1
                with closing(History(path)) as history:
                    channel = FeishuChannel(cli)
                    self.addAsyncCleanup(channel.close)
                    async with running_agent(Settings(), history, model, channel, session_id=_SESSION,
                                             roles=demo_roles()) as (main, _):
                        submit = main.submit_persisted_batch
                        await _poller(history, submit, cli, asr, clock).poll_once()
                        await main.wait_idle()
                        self.assertEqual(len(model.calls), expected_calls)
                        self.assertEqual([row.message.content for row in history.tail(_SESSION, 20)],
                                         ["只消费一次", "模拟回复"])

    async def test_media_budget_keeps_small_image_and_text_in_one_ordered_batch(self):
        clock = [100]
        runner, asr, model = _Runner(), _ASR(), _Model()
        cli = FeishuCLI("lark-cli", ChannelUser(_PEER), timeout=2, runner=runner)
        post = _raw("om_rich_budget", 102_000, "post", {"content": [[
            {"tag": "text", "text": "前"}, {"tag": "img", "image_key": "img_a"},
            {"tag": "img", "image_key": "img_a"}, {"tag": "text", "text": "后"},
        ]]})
        with closing(History(":memory:")) as history:
            channel = FeishuChannel(cli)
            self.addAsyncCleanup(channel.close)
            async with running_agent(Settings(), history, model, channel, session_id=_SESSION,
                                     roles=demo_roles()) as (main, _):
                submit = main.submit_persisted_batch
                poller = _poller(history, submit, cli, asr, clock, batch_budget=470)
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

    def test_sample_config_offline(self):
        settings = load_settings(Path("config.example.toml"))
        self.assertIsNone(settings.feishu)
        self.assertEqual(settings.models, {})
        self.assertEqual(settings.llm_providers, {})
        self.assertIsNone(settings.asr)
        self.assertIsNone(settings.tts)

if __name__ == "__main__":
    unittest.main()
