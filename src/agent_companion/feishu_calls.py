"""一次只接入一通飞书来电；来电身份和设备生命周期不改变私聊路由。"""

import asyncio
from collections.abc import Awaitable, Callable
import logging
from sqlite3 import OperationalError
from typing import Protocol

from .channels.feishu.cli import CliError, FeishuDiscovery
from .channels.feishu.macos_calls import DesktopCallError, RingingWindow, WindowIdentity
from .speech.audio import AudioDevice
from .speech.vad import VoiceSetupError
from .speech.voice import VoiceChannel
from .storage.sqlite import is_sqlite_lock_error


_LOG = logging.getLogger(__name__)
_POLL_SECONDS = 0.25
_CLICK_FRESH_SECONDS = 0.5
_CONNECT_SECONDS = 10.0


class CallDesktop(Protocol):
    def ringing(self) -> tuple[RingingWindow, ...]: ...
    def main_windows(self) -> tuple[WindowIdentity, ...]: ...
    def main_window_diagnostics(self, pid: int) -> dict[str, int | dict[int, int]]: ...
    def window_exists(self, identity: WindowIdentity) -> bool: ...
    def answer(self, ring: RingingWindow, *, before_deadline: Callable[[], bool]) -> bool: ...


class CallPeer(Protocol):
    async def prepare_agent(self, *, wait_for_lock: bool = True) -> bool: ...
    def watch_call(self, stop: Callable[[], None]) -> bool: ...
    def call_started(self, stop: Callable[[], None], voice: VoiceChannel, *, check_window: Callable[[], bool],
                     idle_timeout: float, copy_confirmed: bool) -> bool: ...
    def call_input(self, stop: Callable[[], None], text: str | None, input_active: bool | None) -> None: ...
    def call_ended(self, stop: Callable[[], None]) -> None: ...


VoiceFactory = Callable[[], Awaitable[tuple[VoiceChannel, AudioDevice]]]


def _require_audio(voice: VoiceChannel, device: AudioDevice) -> None:
    if not voice.failed:
        try:
            device.check_active()
            return
        except Exception:
            pass
    raise VoiceSetupError("feishu_call_audio_inactive")


class _CallBinding:
    def __init__(self, peer: CallPeer, voice: VoiceChannel, desktop: CallDesktop, *, idle_timeout: float,
                 copy_confirmed: bool) -> None:
        self._peer, self._voice, self._desktop = peer, voice, desktop
        self._idle_timeout, self._copy_confirmed = idle_timeout, copy_confirmed
        self._window: WindowIdentity | None = None
        self.closed = False

    def active(self) -> bool:
        return self._window is not None and not self.closed and not self._voice.failed

    def confirm(self, window: WindowIdentity) -> bool:
        if not self.closed and self._peer.call_started(
                self.stop, self._voice, check_window=self.check_window, idle_timeout=self._idle_timeout,
                copy_confirmed=self._copy_confirmed):
            self._window = window
            self._voice.enable_capture()
            return True
        return False

    def check_window(self) -> bool:
        if self.closed or self._window is None:
            return False
        try:
            if self._desktop.window_exists(self._window):
                return True
        except DesktopCallError as exc:
            _LOG.warning("Feishu auto-answer released audio after desktop error (%s)", exc)
        else:
            _LOG.debug("Feishu auto-answer disconnected: meeting window gone")
        self.stop()
        return False

    def input(self, text: str | None, input_active: bool | None) -> None:
        # 设备已经停止，直接解绑放行，不再等待桌面查询。
        if self._voice.failed:
            self.stop()
            return
        # 窗口可能在两次轮询间消失；仅在语义事件查询，不查每帧 PCM。
        if self.check_window():
            self._peer.call_input(self.stop, text, input_active)

    def stop(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self._voice.stop_io()
            finally:
                self._peer.call_ended(self.stop)


class IncomingCalls:
    def __init__(self, desktop: CallDesktop, discovery: FeishuDiscovery, voice_factory: VoiceFactory, *,
                 own_id: str, idle_timeout: float, copy_confirmed: bool) -> None:
        self._desktop, self._discovery, self._voice_factory = desktop, discovery, voice_factory
        self._own_id = own_id
        self._idle_timeout, self._copy_confirmed = idle_timeout, copy_confirmed

    async def run(self, resolve: Callable[[str], CallPeer | None]) -> None:
        desktop_error: str | None = None
        while True:
            try:
                rings = self._desktop.ringing()
                if len(rings) == 1 and not self._desktop.main_windows():
                    await self._handle(rings[0], resolve)
                desktop_error = None
            except DesktopCallError as exc:
                if desktop_error != str(exc):
                    desktop_error = str(exc)
                    _LOG.warning("Feishu auto-answer desktop unavailable (%s)", desktop_error)
            await asyncio.sleep(_POLL_SECONDS)

    async def _handle(self, ring: RingingWindow, resolve: Callable[[str], CallPeer | None]) -> None:
        loop = asyncio.get_running_loop()
        started_at = loop.time()

        try:
            voice, device = await self._voice_factory()
        except VoiceSetupError as exc:
            _LOG.warning("Feishu auto-answer audio setup failed (%s)", exc)
            return

        if self._desktop.ringing() != (ring,) or self._desktop.main_windows():
            _LOG.debug("Feishu auto-answer skipped: desktop changed during audio preparation")
            return

        binding: _CallBinding | None = None
        peer: CallPeer | None = None

        def on_input(text: str | None, input_active: bool | None) -> None:
            if binding is not None:
                binding.input(text, input_active)

        audio_open_started_at = loop.time()
        try:
            async with voice.connected(device, on_input, capture_enabled=False, wait_for_cleanup=False):
                try:
                    _require_audio(voice, device)
                    lookup_started_at = loop.time()
                    _LOG.debug("Feishu auto-answer caller lookup started (audio_prepare=%.3fs, audio_open=%.3fs)",
                               audio_open_started_at - started_at, lookup_started_at - audio_open_started_at)
                    try:
                        caller = await self._discovery.ringing_caller(self._own_id)
                    finally:
                        lookup_finished_at = loop.time()
                        _LOG.debug("Feishu auto-answer caller lookup finished (elapsed=%.3fs)",
                                   lookup_finished_at - lookup_started_at)
                    if caller is None:
                        _LOG.debug("Feishu auto-answer skipped: no unique ringing caller")
                        return
                    if self._desktop.ringing() != (ring,):
                        _LOG.debug("Feishu auto-answer skipped: ringing window changed during caller lookup")
                        return

                    # 本轮只短暂使用身份结果，之后回到桌面轮询重新查询。
                    click_deadline = lookup_finished_at + _CLICK_FRESH_SECONDS

                    def before_deadline() -> bool:
                        return loop.time() < click_deadline

                    if not before_deadline():
                        _LOG.debug("Feishu auto-answer skipped: caller lookup result expired")
                        return
                    peer = resolve(caller.sender_open_id)
                    if peer is None:
                        _LOG.debug("Feishu auto-answer skipped: private route not ready")
                        return
                    try:
                        async with asyncio.timeout_at(click_deadline):
                            ready = await peer.prepare_agent(wait_for_lock=False)
                    except TimeoutError:
                        _LOG.debug("Feishu auto-answer skipped: caller lookup result expired during private binding")
                        return
                    except OperationalError as exc:
                        if not is_sqlite_lock_error(exc):
                            raise
                        _LOG.debug("Feishu auto-answer skipped: private route storage busy")
                        return
                    if not ready:
                        _LOG.debug("Feishu auto-answer skipped: private route not ready")
                        return
                    binding = _CallBinding(peer, voice, self._desktop, idle_timeout=self._idle_timeout,
                                           copy_confirmed=self._copy_confirmed)
                    if not peer.watch_call(binding.stop):
                        return
                    if (binding.closed or resolve(caller.sender_open_id) is not peer or self._desktop.main_windows()):
                        _LOG.debug("Feishu auto-answer skipped: private route or desktop changed before click")
                        return
                    _require_audio(voice, device)
                    if not before_deadline():
                        _LOG.debug("Feishu auto-answer skipped: caller lookup result expired before click")
                        return
                    if not self._desktop.answer(ring, before_deadline=before_deadline):
                        _LOG.debug("Feishu auto-answer skipped: target window changed or deadline expired")
                        return
                    _LOG.debug("Feishu auto-answer click sent; waiting up to %.1fs for a meeting window",
                               _CONNECT_SECONDS)
                    window = await self._confirm(ring, binding, voice, device)
                    if window is None or not binding.confirm(window):
                        return
                    _LOG.debug("Feishu auto-answer connected")
                    await self._follow(binding)
                # 在 VoiceChannel 的 TaskGroup 内消化预期错误，文字服务继续运行。
                except DesktopCallError as exc:
                    if binding is not None:
                        binding.stop()
                    _LOG.warning("Feishu auto-answer released audio after desktop error (%s)", exc)
                except CliError as exc:
                    _LOG.warning("Feishu auto-answer caller lookup failed (%s)", exc)
                except VoiceSetupError as exc:
                    _LOG.warning("Feishu auto-answer audio unavailable (%s)", exc)
                finally:
                    if binding is not None:
                        binding.stop()  # 先封闸和清 PCM，再由 connected 取消请求、关闭设备。
        except VoiceSetupError as exc:
            _LOG.warning("Feishu auto-answer audio setup failed (%s)", exc)

    async def _confirm(self, ring: RingingWindow, binding: _CallBinding,
                       voice: VoiceChannel, device: AudioDevice) -> WindowIdentity | None:
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        deadline = started_at + _CONNECT_SECONDS
        pid = ring.identity.pid
        polls = peak_candidates = peak_same_pid = 0
        while not binding.closed and loop.time() < deadline:
            _require_audio(voice, device)
            candidates = self._desktop.main_windows()
            same_pid = tuple(window for window in candidates if window.pid == pid)
            polls += 1
            peak_candidates = max(peak_candidates, len(candidates))
            peak_same_pid = max(peak_same_pid, len(same_pid))
            # 点击前没有会议窗口。合并同一进程新出现的窗口，避免只跟踪大窗口或控制条。
            numbers = frozenset(number for window in same_pid for number in window.numbers)
            if numbers:
                _require_audio(voice, device)
                if loop.time() < deadline:
                    return WindowIdentity(pid, numbers)
                break
            await asyncio.sleep(min(_POLL_SECONDS, max(0, deadline - loop.time())))
        _LOG.debug("Feishu auto-answer did not confirm connection "
                   "(reason=%s, elapsed=%.3fs, polls=%d, peak_candidates=%d, peak_same_pid_candidates=%d)",
                   "binding_closed" if binding.closed else "timeout", loop.time() - started_at,
                   polls, peak_candidates, peak_same_pid)
        if not binding.closed and _LOG.isEnabledFor(logging.DEBUG):
            try:
                snapshot = self._desktop.main_window_diagnostics(pid)
            except DesktopCallError as exc:
                _LOG.debug("Feishu meeting window timeout snapshot unavailable (%s)", exc)
            else:
                _LOG.debug("Feishu meeting window timeout snapshot: %s", snapshot)
        return None

    async def _follow(self, binding: _CallBinding) -> None:
        while binding.active() and binding.check_window():
            await asyncio.sleep(_POLL_SECONDS)
