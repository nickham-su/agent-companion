import asyncio
from contextlib import asynccontextmanager, closing, suppress
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from agent_companion.app import run_feishu, running_agent
from agent_companion.channels.feishu.cli import CliError, CommandResult, FeishuDiscovery
from agent_companion.channels.feishu.macos_calls import DesktopCallError, MacCallDesktop, RingingWindow, WindowIdentity
from agent_companion.config import ASRSettings, FeishuSettings, MiniMaxConnection, Settings
from agent_companion.protocol import TextDelta, ToolCall, ToolReady
from agent_companion.runtime.engine import Runtime
from agent_companion.runtime.subtasks import DELEGATE_TOOL_NAME, RESULT_PREFIX
from agent_companion.speech.voice import VoiceChannel
from agent_companion.storage.sqlite import History

from test_feishu_private import _CHATS, _OWN, _SENDERS, _Runner, _chat, _eventually, _message
from test_voice import _A, _ASR, _Device, _SETTINGS, _TTS, _Vad


def _ok(data, *, identity="user"):
    return CommandResult(0, json.dumps({"ok": True, "identity": identity, "data": data}).encode(), b"")


def _meeting(sender, **changes):
    return {"meeting": {"id": "12345", "status": 1, "host_user": {"id": sender},
                        "participants": [{"id": sender}], **changes}}


class CallCLITests(unittest.IsolatedAsyncioTestCase):
    async def test_cli_returns_only_a_verified_ringing_identity(self):
        sender = _SENDERS[0]
        one = {"meetings": [{"meeting_id": "12345"}]}
        no_detail = object()
        for listing, detail, expected in (
            (_ok(one), _ok(_meeting(sender)), sender),
            (_ok({"meetings": []}), no_detail, None),
            (_ok({"meetings": None}), no_detail, None),
            (_ok({"meetings": one["meetings"] * 2}), no_detail, None),
            (_ok(one, identity="bot"), no_detail, CliError),
            (_ok({}), no_detail, CliError),
            (_ok({"meetings": [{"meeting_id": "--not-an-id"}]}), no_detail, CliError),
            (CommandResult(1, b"", b""), no_detail, CliError),
            (_ok(one), _ok(_meeting(sender, status=True)), None),
            (_ok(one), _ok(_meeting(sender, status=2)), None),
            (_ok(one), _ok(_meeting(sender, id="456")), CliError),
            (_ok(one), _ok(_meeting(_OWN)), None),
            (_ok(one), _ok(_meeting("invalid")), None),
            (_ok(one), _ok(_meeting(sender, participants=[])), None),
            (_ok(one), _ok(_meeting(sender, participants=[{"id": sender}, {"id": _OWN}])), None),
            (_ok(one), _ok(_meeting(sender, participants=[{"id": _SENDERS[1]}])), None),
        ):
            commands = []

            async def runner(argv, data, timeout):
                commands.append(argv)
                assert argv[-4:] == ("--as", "user", "--format", "json") and data is None
                if argv[1:3] == ("vc", "+meeting-list-active"):
                    return listing
                assert argv[1:-4] == ("vc", "meeting", "get", "--meeting-id", "12345", "--with-participants",
                                      "--user-id-type", "open_id")
                assert detail is not no_detail
                return detail

            discovery = FeishuDiscovery("synthetic-cli", timeout=1, runner=runner)
            with self.subTest(listing=listing, detail=detail):
                if expected is CliError:
                    with self.assertRaises(CliError):
                        await discovery.ringing_caller(_OWN)
                else:
                    caller = await discovery.ringing_caller(_OWN)
                    self.assertEqual(caller.sender_open_id if caller else None, expected)
                    if caller is not None:
                        self.assertEqual(caller.meeting_id, "12345")
                self.assertEqual(len(commands), 1 if detail is no_detail else 2)


def _cg(number, *, owner="飞书会议", title="飞书会议", pid=3, layer=25, width=368, height=86):
    return {"kCGWindowOwnerName": owner, "kCGWindowName": title, "kCGWindowOwnerPID": pid,
            "kCGWindowLayer": layer, "kCGWindowNumber": number,
            "kCGWindowBounds": {"X": 100, "Y": 50, "Width": width, "Height": height}}


class DesktopTests(unittest.TestCase):
    def test_double_records_accept_once_and_recheck_window_number_after_warp(self):
        desktop = MacCallDesktop.__new__(MacCallDesktop)
        visible = [_cg(10), _cg(11, title="")]
        desktop._windows = lambda *, visible: self.windows if visible else []
        desktop._can_post = lambda: True
        calls = []
        desktop._mouse = lambda _, event, *args: event
        desktop._set = desktop._release = lambda *args: None
        desktop._post = lambda tap, event: calls.append(event)
        self.windows = visible
        (ring,) = desktop.ringing()
        self.assertEqual(ring.identity.numbers, {10, 11})
        self.assertEqual((ring.answer_point.x, ring.answer_point.y), (433, 95))
        desktop._move = lambda point: 0
        with patch("agent_companion.channels.feishu.macos_calls.time.sleep"):
            self.assertTrue(desktop.answer(ring, before_deadline=lambda: True))
        self.assertEqual(calls, [1, 2])
        self.windows.insert(0, _cg(50, owner="遮挡窗口", pid=9))
        self.assertFalse(desktop.answer(ring, before_deadline=lambda: True))
        self.windows.pop(0)

        def replace_on_warp(point):
            self.windows = [_cg(12), _cg(13, title="")]
            return 0

        desktop._move = replace_on_warp
        self.assertFalse(desktop.answer(ring, before_deadline=lambda: True))
        self.assertEqual(calls, [1, 2])

    def test_answer_checks_deadline_before_down_but_always_releases_after_down(self):
        for expire_at in ("move", "refresh", "event_setup", "after_down"):
            with self.subTest(expire_at=expire_at):
                desktop = MacCallDesktop.__new__(MacCallDesktop)
                desktop._can_post = lambda: True
                desktop._mouse = lambda _, event, *args: event
                posts, released = [], []
                now, deadline = 0, 1
                snapshots = 0

                def windows(*, visible):
                    nonlocal now, snapshots
                    snapshots += 1
                    if expire_at == "refresh" and snapshots == 2:
                        now = deadline
                    return [_cg(10)]

                def move(point):
                    nonlocal now
                    if expire_at == "move":
                        now = deadline
                    return 0

                def set_field(event, *args):
                    nonlocal now
                    if expire_at == "event_setup" and event == 2:
                        now = deadline

                def post(tap, event):
                    nonlocal now
                    posts.append(event)
                    if expire_at == "after_down" and event == 1:
                        now = deadline

                desktop._windows = windows
                desktop._move, desktop._set, desktop._post, desktop._release = move, set_field, post, released.append
                ring = RingingWindow(WindowIdentity(3, frozenset({10})), (100, 50, 368, 86))
                with patch("agent_companion.channels.feishu.macos_calls.time.sleep"):
                    self.assertEqual(desktop.answer(ring, before_deadline=lambda: now < deadline),
                                     expire_at == "after_down")
                self.assertEqual(posts, [1, 2] if expire_at == "after_down" else [])
                self.assertEqual(released, [1, 2])

    def test_track_main_by_process_and_number_from_all_windows(self):
        desktop = MacCallDesktop.__new__(MacCallDesktop)
        full = [_cg(400, layer=0, width=920, height=710), _cg(401, title="", layer=0, width=920, height=710)]
        desktop._windows = lambda *, visible: [] if visible else full
        self.assertEqual(desktop.ringing(), ())
        (main,) = desktop.main_windows()
        full[:] = [_cg(401, title="", layer=-1, width=850, height=650)]
        self.assertTrue(desktop.window_exists(main))
        full[:] = [_cg(401, pid=4)]
        self.assertFalse(desktop.window_exists(main))

    def test_timeout_snapshot_summarizes_raw_windows_without_metadata_values(self):
        desktop = MacCallDesktop.__new__(MacCallDesktop)
        windows = [
            _cg(10), _cg(11, layer=0), _cg(12, layer=0, title=""),
            _cg(13, layer=0, title="synthetic other title"), _cg(14, layer=0),
            _cg(15, layer=3), _cg(16, layer=0, pid=4), _cg(17, layer=0, owner="synthetic other owner"),
        ]
        windows[0]["kCGWindowIsOnscreen"] = windows[1]["kCGWindowIsOnscreen"] = True
        del windows[4]["kCGWindowName"]
        desktop._windows = Mock(return_value=windows)

        snapshot = desktop.main_window_diagnostics(3)

        desktop._windows.assert_called_once_with(visible=False)
        self.assertEqual(snapshot, {
            "desktop_records": 8,
            "meeting_owner_records": 7,
            "same_pid_records": 7,
            "same_pid_meeting_owner_records": 6,
            "same_pid_onscreen_records": 2,
            "same_pid_meeting_layers": {0: 4, 3: 1, 25: 1},
            "same_pid_layer0_title_match": 1,
            "same_pid_layer0_title_missing": 2,
            "same_pid_layer0_title_other": 1,
            "other_pid_layer0_title_match": 1,
        })


class _CallRunner(_Runner):
    def __init__(self):
        super().__init__()
        self.pages[None] = ([_chat(chat, sender) for chat, sender in zip(_CHATS, _SENDERS)], None)
        self.caller = None
        self.meeting_id = None
        self.lookups = 0
        self.details_asked = asyncio.Event()
        self.release_detail = asyncio.Event()
        self.release_detail.set()

    async def __call__(self, argv, data, timeout):
        if argv[1:3] == ("vc", "+meeting-list-active"):
            self.lookups += 1
            meetings = [] if self.meeting_id is None else [{"meeting_id": self.meeting_id}]
            return _ok({"meetings": meetings})
        if argv[1:4] == ("vc", "meeting", "get"):
            caller = self.caller
            meeting_id = self.meeting_id
            self.details_asked.set()
            try:
                async with asyncio.timeout(timeout):
                    await self.release_detail.wait()
            except TimeoutError:
                raise CliError("cli_timeout") from None
            return _ok(_meeting(caller, id=meeting_id))
        return await super().__call__(argv, data, timeout)


class _Desktop:
    def __init__(self, runner):
        self.runner = runner
        self.rings = ()
        self.main = ()
        self.clicks = []
        self.error = False

    def ring(self, sender, number):
        self.runner.caller = sender
        self.runner.meeting_id = str(number)
        self.rings = (RingingWindow(WindowIdentity(3, frozenset({number})), (50, 10, 368, 86)),)

    def connect(self, number):
        self.rings = ()
        self.main = (WindowIdentity(3, frozenset({number})),)

    def ringing(self):
        return self.rings

    def main_windows(self):
        return self.main

    def main_window_diagnostics(self, pid):
        return {"desktop_records": len(self.main) + len(self.rings)}

    def window_exists(self, identity):
        if self.error and self.main:
            raise DesktopCallError("synthetic_window_failure")
        windows = self.main + tuple(ring.identity for ring in self.rings)
        return any(window.pid == identity.pid and not window.numbers.isdisjoint(identity.numbers)
                   for window in windows)

    def answer(self, ring, *, before_deadline):
        if self.rings != (ring,) or not before_deadline():
            return False
        self.clicks.append(ring.identity)
        return True


class _Echo:
    def __init__(self):
        self.calls = []

    async def stream(self, messages, tools, *, thinking):
        self.calls.append(tuple(m.content for m in messages if m.role == "user"))
        yield TextDelta(messages[-1].content + "。")


def _talk(device):
    device.frame(True, 4)
    device.frame(False, 3)


def _session(sender):
    return "feishu-private:" + sha256(sender.encode("ascii")).hexdigest()


class IncomingCallTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.runner = _CallRunner()
        self.desktop = _Desktop(self.runner)
        self.clock = 100.250
        self.devices = []
        self.asrs = []
        self.vads = []

    @asynccontextmanager
    async def _serve(self, model, *, make_asr=lambda i: _ASR(f"测试语音{i}"), make_device=_Device,
                     tts=None, copy=True):
        tts = _TTS() if tts is None else tts
        settings = Settings(feishu=FeishuSettings(_OWN, mode="voice", voice_text_copy=copy),
                            voice=replace(_SETTINGS, input_device="mock-in", output_device="mock-out",
                                          vad_model=Path("synthetic.onnx")),
                            asr=ASRSettings(MiniMaxConnection("synthetic"), max_audio_seconds=30))

        def build(_, asr, tts):
            device, vad = make_device(), _Vad()
            self.devices.append(device)
            self.vads.append(vad)
            self.asrs.append(make_asr(len(self.devices)))
            return VoiceChannel(self.asrs[-1], tts, vad, settings.voice, asr_timeout=2, max_audio_seconds=30), device

        with (closing(History(":memory:")) as history,
              patch("agent_companion.channels.feishu.macos_calls.MacCallDesktop", return_value=self.desktop),
              patch("agent_companion.app._make_voice", side_effect=build),
              patch("agent_companion.app.running_agent", wraps=running_agent) as agents,
              patch("agent_companion.app.time", side_effect=lambda: self.clock),
              patch("agent_companion.feishu_private.time", side_effect=lambda: self.clock),
              patch("agent_companion.feishu_private._DISCOVERY_SECONDS", 0.015),
              patch("agent_companion.channels.feishu.channel._POLL_SECONDS", 0.015),
              patch("agent_companion.feishu_calls._POLL_SECONDS", 0.005),
              patch("agent_companion.feishu_calls._CONNECT_SECONDS", 0.12)):
            task = asyncio.create_task(run_feishu(settings, history, model, object(), tts=tts,
                                                discovery=self.runner.discovery()))
            try:
                yield history, agents
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    async def _ring_connect(self, sender, number, count):
        self.desktop.ring(sender, number)
        await _eventually(lambda: len(self.desktop.clicks) == count)
        self.assertTrue(self.devices[-1].started.is_set())
        self.desktop.connect(number + 100)
        # 让状态机完成至少一次窗口快照，声音随后才注入。
        await asyncio.sleep(0.025)

    def _text(self, index, content):
        self.runner.messages[_CHATS[index]].append(_message(str(index), _CHATS[index], _SENDERS[index], content))
        self.clock = 102.0

    async def test_first_call_looks_up_once_after_audio_opens_and_shares_text_history(self):
        self.runner.release_detail.clear()
        model, tts = _Echo(), _TTS()
        sender = _SENDERS[0]
        async with self._serve(model, tts=tts) as (history, agents):
            await _eventually(lambda: bool(self.runner.queries))
            self.assertEqual(self.runner.lookups, 0)
            self.desktop.ring(sender, 1)
            await self.runner.details_asked.wait()
            self.assertFalse(self.desktop.clicks)
            self.assertEqual(agents.call_count, 0)
            device = self.devices[0]
            self.assertTrue(device.started.is_set())
            _talk(device)  # 查询期间打开了设备，但不能收音。
            self.runner.release_detail.set()
            await _eventually(lambda: len(self.desktop.clicks) == 1)
            _talk(device)
            await asyncio.sleep(0.03)
            self.assertFalse(self.asrs[0].files)
            self.assertFalse(tts.calls)
            self.desktop.connect(101)
            await asyncio.sleep(0.025)
            _talk(device)
            await _eventually(lambda: bool(tts.calls))
            await _eventually(lambda: any(device.pull(32)))
            await _eventually(lambda: self.runner.sent[_CHATS[0]] == ["测试语音1。"])
            self._text(0, "通话中发文字")
            await _eventually(lambda: len(self.runner.sent[_CHATS[0]]) == 2)
            self.assertEqual(self.runner.sent[_CHATS[0]][1], "通话中发文字。")
            self.assertEqual(len(tts.calls), 1)
            self.assertEqual(agents.call_count, 1)
            self.assertEqual(self.runner.lookups, 1)
            self.assertEqual([row.message.content for row in history.tail(_session(sender), 10)],
                             ["测试语音1", "测试语音1。", "通话中发文字", "通话中发文字。"])

    async def test_retries_same_window_with_fresh_lookup_after_route_becomes_ready(self):
        self.runner.pages[None] = ([], None)
        async with self._serve(_Echo()) as (_, agents):
            await _eventually(lambda: bool(self.runner.queries))
            self.desktop.ring(_SENDERS[0], 1)
            await _eventually(lambda: bool(self.devices) and self.devices[0].closed)
            self.assertEqual(agents.call_count, 0)
            self.assertFalse(self.desktop.clicks)
            lookups_before = self.runner.lookups

            self.runner.pages[None] = ([_chat(_CHATS[0], _SENDERS[0])], None)
            self.clock = 102.0
            await _eventually(lambda: bool(self.desktop.clicks))
            self.assertEqual(self.desktop.clicks[0].numbers, {1})
            self.assertGreater(self.runner.lookups, lookups_before)

    async def test_window_replaced_during_lookup_uses_next_lookup_for_the_new_window(self):
        self.runner.release_detail.clear()
        async with self._serve(_Echo()) as (_, agents):
            self.desktop.ring(_SENDERS[0], 1)
            await self.runner.details_asked.wait()
            self.desktop.ring(_SENDERS[1], 2)  # 同坐标不同窗口号，不能沿用原结果。
            self.runner.release_detail.set()
            await _eventually(lambda: bool(self.desktop.clicks))
            self.assertEqual(self.desktop.clicks[0].numbers, {2})
            self.assertEqual(self.runner.lookups, 2)
            self.assertEqual(agents.call_count, 1)
            self.assertTrue(self.devices[0].closed)

    async def test_pre_existing_meeting_prevents_preparing_audio_and_looking_up_a_caller(self):
        desktop = self.desktop
        desktop.main = (WindowIdentity(3, frozenset({90})),)
        with patch.object(desktop, "main_windows", wraps=desktop.main_windows) as main_windows:
            async with self._serve(_Echo()):
                desktop.ring(_SENDERS[0], 1)
                await _eventually(lambda: main_windows.call_count >= 2)
                self.assertFalse(self.devices)
                self.assertFalse(desktop.clicks)
                self.assertEqual(self.runner.lookups, 0)

    async def test_rejected_click_retries_same_window_with_fresh_lookup(self):
        desktop = self.desktop
        real_answer = desktop.answer
        attempts = []

        def answer(ring, *, before_deadline):
            attempts.append(ring)
            if len(attempts) == 1:
                return False
            self.assertTrue(self.devices[0].closed)
            return real_answer(ring, before_deadline=before_deadline)

        with patch.object(desktop, "answer", side_effect=answer):
            async with self._serve(_Echo()):
                desktop.ring(_SENDERS[0], 1)
                await _eventually(lambda: bool(desktop.clicks))
                self.assertEqual(attempts, [desktop.rings[0]] * 2)
                self.assertEqual(self.runner.lookups, 2)

    async def test_unconfirmed_click_retries_same_window_then_confirms_with_popup_still_present(self):
        desktop = self.desktop
        real_answer = desktop.answer
        click_times = []

        def answer(ring, *, before_deadline):
            clicked = real_answer(ring, before_deadline=before_deadline)
            if clicked:
                click_times.append(asyncio.get_running_loop().time())
                if len(click_times) == 2:
                    self.assertTrue(self.devices[0].closed)
                    desktop.main = (WindowIdentity(3, frozenset({101})),)  # 浮窗仍在。
            return clicked

        with patch.object(desktop, "answer", side_effect=answer):
            async with self._serve(_Echo()) as (history, _):
                desktop.ring(_SENDERS[0], 1)
                await _eventually(lambda: bool(desktop.clicks))
                _talk(self.devices[0])
                await _eventually(lambda: len(desktop.clicks) == 2)
                self.assertGreaterEqual(click_times[1] - click_times[0], 0.12)
                self.assertTrue(desktop.rings)
                self.assertFalse(self.asrs[0].files)
                _talk(self.devices[1])
                await _eventually(lambda: bool(history.tail(_session(_SENDERS[0]), 10)))
                self.assertEqual(self.runner.lookups, 2)

    async def test_confirmation_timeout_logs_poll_summary_and_one_snapshot(self):
        desktop = self.desktop
        real_answer = desktop.answer

        def answer(ring, *, before_deadline):
            accepted = real_answer(ring, before_deadline=before_deadline)
            if accepted:
                desktop.main = (WindowIdentity(5, frozenset({104})),)
            return accepted

        with (patch.object(desktop, "answer", side_effect=answer),
              patch.object(desktop, "main_window_diagnostics", wraps=desktop.main_window_diagnostics) as snapshot,
              self.assertLogs("agent_companion.feishu_calls", level="DEBUG") as logs):
            async with self._serve(_Echo()):
                desktop.ring(_SENDERS[0], 1)
                await _eventually(lambda: bool(self.devices) and self.devices[0].closed)
                self.assertEqual(len(desktop.clicks), 1)

        snapshot.assert_called_once_with(3)
        diagnostics = "\n".join(logs.output)
        self.assertIn("click sent; waiting up to", diagnostics)
        self.assertRegex(diagnostics, r"did not confirm connection \(reason=timeout, elapsed=\d+\.\d+s, "
                         r"polls=[1-9]\d*, peak_candidates=1, peak_same_pid_candidates=0\)")
        self.assertIn("meeting window timeout snapshot: {'desktop_records': 2}", diagnostics)

    async def test_confirms_and_tracks_multiple_meeting_windows_from_the_ringing_process(self):
        desktop = self.desktop
        real_answer = desktop.answer
        primary = WindowIdentity(3, frozenset({101, 102}))
        controls = WindowIdentity(3, frozenset({103}))
        foreign = WindowIdentity(5, frozenset({104}))

        def answer(ring, *, before_deadline):
            accepted = real_answer(ring, before_deadline=before_deadline)
            if accepted:
                desktop.main = (foreign, primary, controls)
            return accepted

        with patch.object(desktop, "answer", side_effect=answer):
            async with self._serve(_Echo()) as (history, _):
                desktop.ring(_SENDERS[0], 1)
                await _eventually(lambda: bool(desktop.clicks))
                device = self.devices[0]
                _talk(device)
                await _eventually(lambda: bool(history.tail(_session(_SENDERS[0]), 10)))

                desktop.main = (controls, foreign)
                with patch.object(desktop, "window_exists", wraps=desktop.window_exists) as probe:
                    await _eventually(lambda: probe.call_count >= 2)
                self.assertFalse(device.closed)

                desktop.main = (foreign,)
                await _eventually(lambda: device.closed)
                self.assertEqual(self.runner.lookups, 1)

    async def test_lookup_after_audio_opens_can_see_a_reused_window_has_a_new_caller(self):
        desktop, runner = self.desktop, self.runner
        other = _SENDERS[1]

        class ReusedWindow(_Device):
            def start(self, capture, render):
                super().start(capture, render)
                # PID、CG 窗口号、坐标没变，但在唯一一次查询前已经换成另一通。
                desktop.ring(other, 1)
                runner.meeting_id = "2"

        async with self._serve(_Echo(), make_device=ReusedWindow) as (history, agents):
            await self._ring_connect(_SENDERS[0], 1, 1)
            self.assertEqual(runner.lookups, 1)
            self.assertEqual(agents.call_count, 1)
            _talk(self.devices[-1])
            await _eventually(lambda: bool(history.tail(_session(other), 10)))
            self.assertFalse(history.tail(_session(_SENDERS[0]), 10))

    async def test_cli_timeout_logs_stage_timings_without_ids(self):
        runner = self.runner
        sender, meeting = _SENDERS[0], "123456789012345"
        runner.release_detail.clear()

        with self.assertLogs("agent_companion", level="DEBUG") as logs:
            async with self._serve(_Echo()):
                self.desktop.ring(sender, 1)
                runner.meeting_id = meeting
                await runner.details_asked.wait()
                self.desktop.rings = ()
                await _eventually(lambda: bool(self.devices) and self.devices[0].closed)
                self.assertEqual(runner.lookups, 1)
                self.assertFalse(self.desktop.clicks)

        messages = [record.getMessage() for record in logs.records
                    if record.name in {"agent_companion.feishu_calls", "agent_companion.channels.feishu.cli"}]
        diagnostics = "\n".join(messages)
        self.assertEqual(diagnostics.count("step=list_active, outcome=ok, elapsed="), 1)
        self.assertIn("step=meeting_detail, outcome=failed, elapsed=", diagnostics)
        self.assertRegex(diagnostics, r"caller lookup started \(audio_prepare=\d+\.\d+s, "
                         r"audio_open=\d+\.\d+s\)")
        self.assertIn("caller lookup finished (elapsed=", diagnostics)
        self.assertIn("caller lookup failed (cli_timeout)", diagnostics)
        for identifier in (sender, meeting, _OWN):
            self.assertNotIn(identifier, diagnostics)

    async def test_expired_identity_is_not_clicked_and_next_round_queries_again(self):
        loop = asyncio.get_running_loop()
        clock = loop.time
        time_spent = 0.0
        runner = self.runner

        class SlowCheckDevice(_Device):
            def check_active(self):
                nonlocal time_spent
                super().check_active()
                if runner.lookups == 1 and not time_spent:
                    # 用同一单调时钟模拟最后一次同步设备检查耗时，不真正阻塞测试。
                    time_spent = 0.6

        with (patch.object(loop, "time", new=lambda: clock() + time_spent),
              patch.object(loop, "slow_callback_duration", 2.0),  # 不把模拟时钟跳变报告为实际阻塞
              patch.object(self.desktop, "answer", wraps=self.desktop.answer) as answer):
            async with self._serve(_Echo(), make_device=SlowCheckDevice):
                self.desktop.ring(_SENDERS[0], 1)
                await _eventually(lambda: bool(self.desktop.clicks))
                self.assertEqual(runner.lookups, 2)
                self.assertEqual(time_spent, 0.6)
                self.assertTrue(self.devices[0].closed)
                answer.assert_called_once()

    async def test_audio_open_exceeding_old_total_deadline_does_not_block_fresh_lookup_and_click(self):
        loop = asyncio.get_running_loop()
        clock = loop.time
        elapsed = 0.0
        audio_opened = asyncio.Event()

        class SlowOpenDevice(_Device):
            def start(self, capture, render):
                nonlocal elapsed
                super().start(capture, render)
                elapsed = 9.0
                audio_opened.set()

        with (patch.object(loop, "time", new=lambda: clock() + elapsed),
              patch.object(loop, "slow_callback_duration", 12.0)):
            async with self._serve(_Echo(), make_device=SlowOpenDevice):
                self.desktop.ring(_SENDERS[0], 1)
                await asyncio.wait_for(audio_opened.wait(), timeout=12.0)
                await _eventually(lambda: bool(self.desktop.clicks))
                self.assertEqual(self.runner.lookups, 1)

    async def test_audio_inactive_at_start_or_while_confirming_never_opens_capture(self):
        desktop = self.desktop
        for before_click in (True, False):
            with self.subTest(before_click=before_click):
                self.devices.clear()
                desktop.clicks.clear()
                desktop.main = ()

                class Device(_Device):
                    def start(self, capture, render):
                        super().start(capture, render)
                        self.fault = before_click

                real_answer = desktop.answer

                def click_then_fail(ring, *, before_deadline):
                    accepted = real_answer(ring, before_deadline=before_deadline)
                    if accepted:
                        # 服务最终看到了正确的接通窗口，但不能把已故障的设备当成就绪。
                        self.devices[0].fault = True
                        desktop.connect(101)
                    return accepted

                with (patch.object(desktop, "answer", side_effect=click_then_fail),
                      patch.object(VoiceChannel, "enable_capture", autospec=True) as enable,
                      self.assertLogs("agent_companion.feishu_calls", level="WARNING")):
                    async with self._serve(_Echo(), make_device=Device):
                        desktop.ring(_SENDERS[0], 1)
                        await _eventually(lambda: bool(self.devices) and self.devices[0].closed)
                        self.assertEqual(len(desktop.clicks), 0 if before_click else 1)
                        if self.devices[0].started.is_set():
                            _talk(self.devices[0])
                        self.assertFalse(self.asrs[-1].files)
                        enable.assert_not_called()

    async def test_first_full_snapshot_without_original_window_seals_pcm_immediately(self):
        async with self._serve(_Echo(), tts=_TTS(size=500), copy=False):
            await self._ring_connect(_SENDERS[0], 1, 1)
            old, = self.desktop.main
            other = WindowIdentity(old.pid, frozenset({404}))
            device = self.devices[0]
            _talk(device)
            await _eventually(lambda: any(device.pull(1)))  # 旧通仍有未播 PCM。
            exists = self.desktop.window_exists
            next_sample = asyncio.Event()
            pulled = []

            def window_exists(identity):
                present = exists(identity)
                if identity == old and not present and not next_sample.is_set():
                    def read_output():
                        pulled.append(device.pull(32))
                        next_sample.set()

                    asyncio.get_running_loop().call_soon(read_output)
                return present

            self.desktop.window_exists = window_exists
            # 可见来电列表已有变化，但全量窗口仍含被遮挡/最小化的旧窗口。
            self.desktop.main = old, other
            self.assertFalse(self.desktop.ringing())
            await asyncio.sleep(0.025)
            self.assertFalse(device.closed)
            self.assertTrue(any(device.pull(1)))
            self.desktop.main = (other,)
            await asyncio.wait_for(next_sample.wait(), 2)
            self.assertEqual(pulled, [bytes(64)])
            await _eventually(lambda: device.closed)

    async def test_window_errors_and_verified_route_revocation(self):
        async with self._serve(_Echo()) as (_, agents):
            await self._ring_connect(_SENDERS[0], 1, 1)
            window = self.desktop.main
            with self.assertLogs("agent_companion.feishu_calls", level="WARNING"):
                self.desktop.error = True
                await _eventually(lambda: self.devices[0].closed)
            self.desktop.error = False
            self.desktop.main = ()
            await self._ring_connect(_SENDERS[0], 2, 2)
            self.runner.pages[None] = ([], None)
            await _eventually(lambda: self.devices[1].closed)
            self.assertEqual(len(self.desktop.main), len(window))
            self.assertEqual(agents.call_count, 1)

    async def test_old_tts_model_and_asr_do_not_leak_on_a_to_b_or_b_redial(self):
        model_wait, model_release, model_late = asyncio.Event(), asyncio.Event(), asyncio.Event()
        tts_wait, tts_cancelled, tts_release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        asr_wait, asr_cancelled, asr_release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        tts_late, asr_late = asyncio.Event(), asyncio.Event()

        class Model(_Echo):
            async def stream(self, messages, tools, *, thinking):
                if messages[-1].content == "测试语音1":
                    yield TextDelta("旧合成。")
                    model_wait.set()
                    try:
                        await model_release.wait()
                    except asyncio.CancelledError:
                        # 模拟忽略首次取消的模型适配器：仍不得向下一通输出。
                        await model_release.wait()
                    yield TextDelta("旧模型尾句。")
                    model_late.set()
                else:
                    async for event in super().stream(messages, tools, thinking=thinking):
                        yield event

        class TTS(_TTS):
            async def synthesize(self, text, *, sample_rate):
                if text == "旧合成。":
                    tts_wait.set()
                    try:
                        await tts_release.wait()
                    except asyncio.CancelledError:
                        tts_cancelled.set()
                        await tts_release.wait()
                        tts_late.set()
                    yield _A
                else:
                    async for pcm in super().synthesize(text, sample_rate=sample_rate):
                        yield pcm

        class ASR:
            async def transcribe(self, audio):
                asr_wait.set()
                try:
                    await asr_release.wait()
                except asyncio.CancelledError:
                    asr_cancelled.set()
                    await asr_release.wait()
                    asr_late.set()
                return "不应入库的旧转写"

        model, tts = Model(), TTS()
        try:
            async with self._serve(model, tts=tts, make_asr=lambda i: ASR() if i == 2 else _ASR(f"测试语音{i}")) as (history, agents):
                await self._ring_connect(_SENDERS[0], 1, 1)
                _talk(self.devices[0])
                await asyncio.wait_for(tts_wait.wait(), 2)
                await asyncio.wait_for(model_wait.wait(), 2)
                self.desktop.main = ()
                self.desktop.ring(_SENDERS[1], 2)
                await asyncio.wait_for(tts_cancelled.wait(), 2)
                self.assertFalse(any(self.devices[0].pull(32)))
                await _eventually(lambda: len(self.desktop.clicks) == 2)
                self.desktop.connect(102)
                tts_release.set()
                model_release.set()
                await asyncio.wait_for(tts_late.wait(), 1)
                await asyncio.wait_for(model_late.wait(), 1)
                await asyncio.sleep(0.025)
                self.assertFalse(any(self.devices[1].pull(32)))
                _talk(self.devices[1])
                await asyncio.wait_for(asr_wait.wait(), 2)
                self.desktop.main = ()
                self.desktop.ring(_SENDERS[1], 3)
                await asyncio.wait_for(asr_cancelled.wait(), 2)
                await _eventually(lambda: len(self.desktop.clicks) == 3)
                self.desktop.connect(103)
                asr_release.set()
                await asyncio.wait_for(asr_late.wait(), 1)
                await asyncio.sleep(0.025)
                self.assertFalse(any(self.devices[2].pull(32)))
                _talk(self.devices[2])
                await _eventually(lambda: bool(tts.calls))
                await _eventually(lambda: any(self.devices[2].pull(32)))
                await _eventually(lambda: self.runner.sent[_CHATS[1]] == ["测试语音3。"])
                self.assertFalse(self.runner.sent[_CHATS[0]])
                self.assertEqual(tts.calls, [("测试语音3。", 16000)])
                self.assertEqual(agents.call_count, 2)
                self.assertEqual(len({id(vad) for vad in self.vads}), 3)
                self.assertNotIn("不应入库的旧转写", [r.message.content for r in history.tail(_session(_SENDERS[1]), 10)])
        finally:
            tts_release.set()
            asr_release.set()
            model_release.set()

    async def test_same_sender_redial_stops_blocking_model_without_losing_played_text(self):
        model_blocked, model_cancelled = asyncio.Event(), asyncio.Event()
        new_model_inputs = []

        class Model(_Echo):
            async def stream(self, messages, tools, *, thinking):
                if messages[-1].content == "测试语音1":
                    yield TextDelta("已播放的旧句。")
                    model_blocked.set()
                    try:
                        await asyncio.Future()  # 不主动释放，不能等待整段模型超时才能让新话轮运行。
                    finally:
                        model_cancelled.set()
                else:
                    new_model_inputs.extend(m.content for m in messages)
                    async for event in super().stream(messages, tools, thinking=thinking):
                        yield event

        sender = _SENDERS[0]
        async with self._serve(Model()) as (history, agents):
            await self._ring_connect(sender, 1, 1)
            _talk(self.devices[0])
            await asyncio.wait_for(model_blocked.wait(), 2)
            await _eventually(lambda: any(self.devices[0].pull(32)))
            self.desktop.main = ()
            await self._ring_connect(sender, 2, 2)
            await asyncio.wait_for(model_cancelled.wait(), 2)
            self.assertFalse(any(self.devices[1].pull(32)))
            _talk(self.devices[1])
            await _eventually(lambda: any(self.devices[1].pull(32)))
            await _eventually(lambda: self.runner.sent[_CHATS[0]] == ["已播放的旧句。", "测试语音2。"])
            self.assertEqual(agents.call_count, 1)
            self.assertIn("已播放的旧句。", new_model_inputs)
            self.assertEqual([row.message.content for row in history.tail(_session(sender), 10)],
                             ["测试语音1", "已播放的旧句。", "测试语音2", "测试语音2。"])

    async def test_ending_call_does_not_stop_blocked_text_model(self):
        started, resume, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Model(_Echo):
            async def stream(self, messages, tools, *, thinking):
                started.set()
                try:
                    await resume.wait()
                    yield TextDelta("文字仍可回复。")
                finally:
                    finished.set()

        async with self._serve(Model()):
            await self._ring_connect(_SENDERS[0], 1, 1)
            self._text(0, "通话期间开始的文字轮")
            await asyncio.wait_for(started.wait(), 2)
            self.desktop.main = ()
            await _eventually(lambda: self.devices[0].closed)
            self.assertFalse(finished.is_set())
            resume.set()
            await _eventually(lambda: self.runner.sent[_CHATS[0]] == ["文字仍可回复。"])

    async def test_queued_voice_tool_continuation_and_delayed_child_keep_their_original_route(self):
        parent_wait, parent_release = asyncio.Event(), asyncio.Event()
        child_wait, child_release = asyncio.Event(), asyncio.Event()
        continuation = []
        voice_submitted = []
        submit = Runtime.submit_batch

        def track_submit(runtime, messages, **kwargs):
            voice_submitted.extend(m.content for m in messages)
            submit(runtime, messages, **kwargs)

        class Model(_Echo):
            async def stream(self, messages, tools, *, thinking):
                last = messages[-1]
                if not tools:
                    child_wait.set()
                    await child_release.wait()
                    yield TextDelta("旧子任务结果。")
                elif last.content == "开始委派":
                    yield ToolReady(ToolCall("voice-child", DELEGATE_TOOL_NAME, '{"prompt":"旧后台任务"}'))
                    parent_wait.set()
                    await parent_release.wait()
                elif last.role == "tool":
                    continuation.append(tuple(m.content for m in messages if m.role == "user"))
                    yield TextDelta("旧通话工具续轮。")
                else:
                    async for event in super().stream(messages, tools, thinking=thinking):
                        yield event

        sender = _SENDERS[0]
        first = _ASR("开始委派")
        tts = _TTS()
        with patch.object(Runtime, "submit_batch", track_submit):
            async with self._serve(Model(), tts=tts, copy=False,
                                   make_asr=lambda i: first if i == 1 else _ASR("重拨语音")) as (history, agents):
                await self._ring_connect(sender, 1, 1)
                _talk(self.devices[0])
                await asyncio.wait_for(parent_wait.wait(), 2)
                await asyncio.wait_for(child_wait.wait(), 2)
                first.text = "挂断前排队转写"
                _talk(self.devices[0])
                await _eventually(lambda: "挂断前排队转写" in voice_submitted)
                self._text(0, "排队的文字")
                self.desktop.main = ()
                await self._ring_connect(sender, 2, 2)
                _talk(self.devices[1])
                await _eventually(lambda: "重拨语音" in voice_submitted)
                parent_release.set()
                child_release.set()
                await _eventually(lambda: self.runner.sent[_CHATS[0]] == ["排队的文字。"])
                await _eventually(lambda: bool(tts.calls))
                await _eventually(lambda: any(self.devices[1].pull(32)))
                self.assertEqual(continuation, [])
                self.assertEqual(tts.calls, [("重拨语音。", 16000)])
                self.assertEqual(agents.call_count, 1)
                rows = history.tail(_session(sender), 20)
                accepted, = (json.loads(row.message.content) for row in rows if row.message.role == "tool")

                def child():
                    return history.subtask_run(accepted["child_session_id"], accepted["run_id"])

                await _eventually(lambda: child().status == "completed")
                self.assertIsNone(child().enqueued_at)
                texts = [row.message.content for row in rows]
                self.assertNotIn("挂断前排队转写", texts)
                self.assertFalse(any(text.startswith(RESULT_PREFIX) for text in texts if isinstance(text, str)))
