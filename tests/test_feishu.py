import asyncio
from base64 import b64decode
from contextlib import closing
from dataclasses import replace
import io
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

from agent_companion.__main__ import _serve
from agent_companion.app import run_feishu, running_agent
from agent_companion.channels.feishu.channel import FeishuChannel, FeishuPoller
from agent_companion.channels.feishu.cli import CliError, CliStdoutLimit, CommandResult, FeishuCLI, run_command
from agent_companion.config import (ASRSettings, FeishuSettings, LLMSettings, MiniMaxConnection, Settings, TTSSettings,
                                    VoiceSettings, load_settings)
from agent_companion.protocol import (OMITTED_INPUT_IMAGE, DeliveryStatus, Image, Message, ReplyInfo,
                                      ReplyStream, Text, TextDelta)
from agent_companion.speech.asr import AsrError
from agent_companion.storage.sqlite import FeishuBindingError, History


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
        text = json.loads(json.loads(data)["content"])["text"]
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


class _Voice:
    async def deliver(self, reply):
        raise AssertionError("notification must not use the voice output")


def _poller(history, main, cli, asr, clock, mode="text"):
    return FeishuPoller(cli, asr, FeishuChannel(cli, mode=mode, voice_output=_Voice() if mode == "voice" else None),
                        chat_id=_CHAT, sender_id=_PEER, own_id=_SELF, session_id="main", history=history,
                        submit=main, clock=lambda: clock[0])


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

                channel = FeishuChannel(FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner), mode="text")
                reply = ReplyStream(ReplyInfo("reply", "main", "run", 0))
                reply.publish("第一")
                reply.publish("第二")
                reply.close()
                receipt = await channel.deliver(reply)
                self.assertEqual((receipt.reply_id, receipt.status, num), ("reply", expected, 1))
                self.assertEqual(receipt.confirmed, "第一第二" if expected == DeliveryStatus.FULL else "")

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
            async with running_agent(Settings(), history, model, FeishuChannel(cli, mode="text")) as (main, _):
                poller = _poller(history, main.submit_persisted_batch, cli, asr, clock)
                await poller.poll_once()
                clock[0] = 110
                polling = asyncio.create_task(poller.poll_once())
                await asyncio.wait_for(asr.entered.wait(), 1)
                self.assertEqual(model.calls, [])
                self.assertEqual(history.tail("main", 20), [])
                asr.release.set()
                await polling
                await main.wait_idle()
                self.assertEqual(asr.calls, 1)
                self.assertEqual(len(model.calls), 1)
                text, voice, post, image = model.calls[0][-4:]
                self.assertEqual((text.content, voice.content), ("先到", "模拟转写"))
                self.assertEqual(post.content, (Text("标题"), Text("\n"), Text("前"), image.content[0],
                                                Text("后 (https://example.invalid/link)"), Text("\n"), Text("次行")))
                self.assertIsInstance(image.content[0], Image)
                self.assertEqual(b64decode(image.content[0].url.split(",", 1)[1]), _PNG)
                clock[0] = 120
                await poller.poll_once()
                await main.wait_idle()
                self.assertEqual((len(model.calls), asr.calls), (1, 1))
                self.assertEqual(runner.sent, ["模拟回复"])
                await poller.close()

    async def test_failed_audio_never_retried_or_added_in_both_modes(self):
        for mode, error in (("text", AsrError("asr_transport")), ("voice", TimeoutError())):
            with self.subTest(mode=mode):
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

                    poller = _poller(history, submit, cli, asr, clock, mode)
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
        source = json.dumps(["feishu", _CHAT, _PEER, _SELF])
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
        source = json.dumps(["feishu", _CHAT, _PEER, _SELF])
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
        source = json.dumps(["feishu", _CHAT, _PEER, _SELF])
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
        source = json.dumps(["feishu", _CHAT, _PEER, _SELF])
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
            async with running_agent(Settings(), history, model, FeishuChannel(cli, mode="text")) as (main, _):
                poller = _poller(history, main.submit_persisted_batch, cli, asr, clock)
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
                    async with running_agent(Settings(), history, model, FeishuChannel(cli, mode="text")) as (main, _):
                        await _poller(history, main.submit_persisted_batch, cli, asr, clock).poll_once()
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
            channel = FeishuChannel(cli, mode="text")
            async with running_agent(Settings(), history, model, channel) as (main, _):
                poller = FeishuPoller(cli, asr, channel, chat_id=_CHAT, sender_id=_PEER, own_id=_SELF,
                                      session_id="main", history=history, submit=main.submit_persisted_batch,
                                      clock=lambda: clock[0], batch_budget=470)
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

    async def test_binding_rejects_changed_identities_and_unbound_history_before_startup(self):
        model, asr = _Model(), _ASR()
        runner = _Runner()
        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        identity = FeishuSettings(_CHAT, _PEER, _SELF)
        settings = Settings(feishu=identity, llm=LLMSettings("deepseek", "disabled", api_key="synthetic"),
                            asr=ASRSettings(MiniMaxConnection("synthetic"), max_audio_seconds=500))
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            path = Path(temp) / "synthetic.sqlite"
            with closing(History(path)) as history, patch.object(FeishuPoller, "run") as run:
                await run_feishu(settings, history, model, asr, cli=cli)
                run.assert_awaited_once()
                history.append("main", [Message("user", "旧会话标记")])
                history.accept_subtask("child", "queued", "main", "run", "call")
            with closing(History(path)) as history, patch("agent_companion.app.running_agent") as start:
                for field, changed in (("chat_id", "oc_other"), ("sender_open_id", "ou_other"),
                                       ("own_open_id", "ou_other")):
                    with self.subTest(field=field):
                        wrong = replace(settings, feishu=replace(identity, mode="voice", **{field: changed}))
                        with self.assertRaisesRegex(FeishuBindingError, "新的数据库路径"):
                            await run_feishu(wrong, history, model, asr, cli=cli, tts=object())
                start.assert_not_called()
                self.assertEqual(history.subtask_run("child", "queued").status, "queued")
                self.assertEqual(model.calls, [])
                self.assertEqual(runner.calls, [])
            wrong_startup = replace(settings, database=path, feishu=replace(identity, chat_id="oc_other"))
            with patch("agent_companion.__main__.httpx.AsyncClient") as open_client:
                with self.assertRaises(FeishuBindingError):
                    await _serve(wrong_startup)
                open_client.assert_not_called()
            with closing(History(path)) as history, patch.object(FeishuPoller, "run") as run:
                voice = replace(settings, feishu=replace(identity, mode="voice"))
                await run_feishu(voice, history, model, asr, cli=cli, voice_output=_Voice())
                run.assert_awaited_once()
                self.assertEqual(history.read_after("main")[0].message.content, "旧会话标记")
        with closing(History(":memory:")) as old, patch("agent_companion.app.running_agent") as start:
            old.append("main", [Message("user", "未绑定历史")])
            with self.assertRaisesRegex(FeishuBindingError, "保留旧数据库"):
                await run_feishu(settings, old, model, asr, cli=cli)
            start.assert_not_called()

    async def test_text_composition_and_voice_requires_injection(self):
        runner, asr, model = _Runner(), _ASR(), _Model()
        cli = FeishuCLI("lark-cli", _CHAT, timeout=2, runner=runner)
        clock = [100]
        runner.pages[None] = {"items": [_raw("om_startup", 100_300, "text", {"text": "验证组装"})],
                              "has_more": False}

        async def exercise(poller):
            poller._clock = lambda: clock[0]
            await poller.poll_once()
            clock[0] = 110
            await poller.poll_once()
            for _ in range(50):
                if runner.sent:
                    return
                await asyncio.sleep(0)
            self.fail("composed text channel did not send")

        with closing(History(":memory:")) as history, patch.object(FeishuPoller, "run", exercise):
            await run_feishu(Settings(feishu=FeishuSettings(_CHAT, _PEER, _SELF)), history, model, asr, cli=cli)
            self.assertEqual(runner.sent, ["模拟回复"])
            with self.assertRaisesRegex(ValueError, "requires an audio output"):
                await run_feishu(Settings(feishu=FeishuSettings(_CHAT, _PEER, _SELF, mode="voice")), history,
                                 model, asr, cli=cli)

    def test_sample_config_offline_and_id_validation(self):
        settings = load_settings(Path("config.example.toml"))
        self.assertIsNone(settings.feishu)
        self.assertIsNone(settings.llm)
        self.assertIsNone(settings.asr)
        self.assertIsNone(settings.tts)
        with self.assertRaises(ValueError):
            FeishuSettings(_CHAT, _SELF, _SELF)


class FeishuProductionBindingTests(unittest.IsolatedAsyncioTestCase):
    def settings(self, path, *, mode="text"):
        speech = MiniMaxConnection("synthetic")
        return Settings(database=path, feishu=FeishuSettings(_CHAT, _PEER, _SELF, mode=mode),
                        llm=LLMSettings("deepseek", "disabled", api_key="synthetic"),
                        asr=ASRSettings(speech, max_audio_seconds=500),
                        tts=TTSSettings(speech) if mode == "voice" else None,
                        voice=VoiceSettings(input_device=1, output_device=2, vad_model=Path("synthetic.onnx")))

    def create_legacy_db(self, path, *, used=False):
        # Deliberately missing both newer tables and newer outcome columns.
        with closing(sqlite3.connect(path)) as legacy, legacy:
            legacy.execute("CREATE TABLE reply_outcomes (reply_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,"
                           " run_id TEXT NOT NULL, model_status TEXT NOT NULL, delivery_status TEXT)")
            if used:
                legacy.execute("INSERT INTO reply_outcomes VALUES ('old', 'other', 'run', 'complete', 'full')")

    async def assert_rejected_before_startup(self, settings, message):
        with (patch("agent_companion.__main__.httpx.AsyncClient") as client,
              patch("agent_companion.channels.audio.macos.MacAudioDevice") as audio,
              patch("agent_companion.app.running_agent") as start):
            with self.assertRaisesRegex(FeishuBindingError, message):
                await _serve(settings)
            client.assert_not_called()
            audio.assert_not_called()
            start.assert_not_called()

    async def test_unbound_legacy_history_in_any_session_rejects_without_mutating_db(self):
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "legacy.sqlite"
            self.create_legacy_db(path, used=True)
            before_bytes = path.read_bytes()
            with closing(sqlite3.connect(path)) as legacy:
                before_dump = list(legacy.iterdump())
            await self.assert_rejected_before_startup(self.settings(path, mode="voice"), "保留旧数据库")
            self.assertEqual(path.read_bytes(), before_bytes)
            with closing(sqlite3.connect(path)) as legacy:
                self.assertEqual(list(legacy.iterdump()), before_dump)

    async def test_changed_main_session_cannot_bind_even_if_original_has_no_messages(self):
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "bound.sqlite"
            settings = self.settings(path)
            with (patch("agent_companion.__main__.httpx.AsyncClient"), patch.object(FeishuPoller, "run") as poll):
                await _serve(settings)
                poll.assert_awaited_once()
            with closing(sqlite3.connect(path)) as db:
                original = db.execute("SELECT * FROM feishu_bindings").fetchall()
                self.assertEqual([session_id for session_id, _ in original], [settings.main_session_id])
                self.assertEqual(db.execute("SELECT COUNT(*) FROM messages").fetchone(), (0,))
            await self.assert_rejected_before_startup(replace(settings, main_session_id="other"), "主会话或飞书身份")
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("SELECT * FROM feishu_bindings").fetchall(), original)

    async def test_empty_legacy_db_migrates_and_restarts_across_modes(self):
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "legacy.sqlite"
            self.create_legacy_db(path)
            settings = self.settings(path)
            with (patch("agent_companion.__main__.httpx.AsyncClient"),
                  patch("agent_companion.channels.audio.macos.MacAudioDevice") as audio,
                  patch("agent_companion.app.SileroVAD"), patch("agent_companion.app.VoiceChannel") as voice_channel,
                  patch.object(FeishuPoller, "run") as poll):
                await _serve(settings)
                audio.assert_not_called()
                with closing(sqlite3.connect(path)) as db:
                    original = db.execute("SELECT * FROM feishu_bindings").fetchall()
                    self.assertEqual([session_id for session_id, _ in original], [settings.main_session_id])
                    self.assertTrue({"assistant_message_id", "model_error"} <= {
                        row[1] for row in db.execute("PRAGMA table_info(reply_outcomes)")})
                await _serve(replace(self.settings(path, mode="voice"), asr_timeout=4.5))
                await _serve(settings)
                self.assertEqual(poll.await_count, 3)
                audio.assert_called_once()
                voice_channel.return_value.connected.assert_called_once()
                self.assertEqual(voice_channel.call_args.kwargs["asr_timeout"], 4.5)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("SELECT * FROM feishu_bindings").fetchall(), original)


if __name__ == "__main__":
    unittest.main()
