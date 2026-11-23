"""一次只接入一通飞书来电；来电身份和设备生命周期不改变私聊路由。"""

import asyncio
from collections.abc import Awaitable, Callable
import logging
from sqlite3 import OperationalError
from typing import Protocol

from .channels.feishu.cli import CliError, FeishuDiscovery
from .channels.feishu.macos_calls import DesktopCallError, RingingWindow, WindowIdentity
from .protocol import Message, ReplyInfo, ReplyRoute
from .runtime.engine import Runtime
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
    def window_exists(self, identity: WindowIdentity) -> bool: ...
    def answer(self, ring: RingingWindow, *, before_deadline: Callable[[], bool]) -> bool: ...


class CallPeer(Protocol):
    async def runtime(self, *, wait_for_lock: bool = True) -> Runtime | None: ...
    def copy_confirmed(self, info: ReplyInfo, text: str) -> None: ...
    def watch_call(self, stop: Callable[[], None]) -> bool: ...
    def unwatch_call(self, stop: Callable[[], None]) -> None: ...


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
    def __init__(self, main: Runtime, voice: VoiceChannel, *, idle_timeout: float,
                 copy_confirmed: Callable[[ReplyInfo, str], None] | None) -> None:
        self._main, self._voice = main, voice
        self._connected = False
        self.closed = False
        self._route = ReplyRoute(voice, self.active, idle_timeout, copy_confirmed)

    def active(self) -> bool:
        return self._connected and not self.closed and not self._voice.failed

    def confirm(self) -> None:
        if not self.closed:
            self._connected = True
            self._voice.enable_capture()

    def submit(self, text: str) -> None:
        if self.active():
            self._main.submit_batch((Message("user", text),), reply_route=self._route)

    def stop(self) -> None:
        if not self.closed:
            self.closed = True
            self._voice.stop_io()
            self._main.stop_reply_route(self._route)


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

        def submit(text: str) -> None:
            if binding is not None:
                binding.submit(text)

        audio_open_started_at = loop.time()
        try:
            async with voice.connected(device, submit, capture_enabled=False, wait_for_cleanup=False):
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
                            main = await peer.runtime(wait_for_lock=False)
                    except TimeoutError:
                        _LOG.debug("Feishu auto-answer skipped: caller lookup result expired during private binding")
                        return
                    except OperationalError as exc:
                        if not is_sqlite_lock_error(exc):
                            raise
                        _LOG.debug("Feishu auto-answer skipped: private route storage busy")
                        return
                    if main is None:
                        _LOG.debug("Feishu auto-answer skipped: private route not ready")
                        return
                    binding = _CallBinding(main, voice, idle_timeout=self._idle_timeout,
                                           copy_confirmed=peer.copy_confirmed if self._copy_confirmed else None)
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
                    window = await self._confirm(ring, binding, voice, device)
                    if window is None:
                        _LOG.debug("Feishu auto-answer did not confirm connection")
                        return
                    binding.confirm()
                    _LOG.debug("Feishu auto-answer connected")
                    await self._follow(window, binding)
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
                    if binding is not None and peer is not None:
                        binding.stop()  # 先封闸和清 PCM，再由 connected 取消请求、关闭设备。
                        peer.unwatch_call(binding.stop)
        except VoiceSetupError as exc:
            _LOG.warning("Feishu auto-answer audio setup failed (%s)", exc)

    async def _confirm(self, ring: RingingWindow, binding: _CallBinding,
                       voice: VoiceChannel, device: AudioDevice) -> WindowIdentity | None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _CONNECT_SECONDS
        while not binding.closed and loop.time() < deadline:
            _require_audio(voice, device)
            windows = self._desktop.main_windows()
            if len(windows) == 1 and windows[0].pid == ring.identity.pid:
                _require_audio(voice, device)
                return windows[0] if loop.time() < deadline else None
            await asyncio.sleep(min(_POLL_SECONDS, max(0, deadline - loop.time())))
        return None

    async def _follow(self, window: WindowIdentity, binding: _CallBinding) -> None:
        while binding.active():
            if not self._desktop.window_exists(window):
                binding.stop()
                _LOG.debug("Feishu auto-answer disconnected: meeting window gone")
                return
            await asyncio.sleep(_POLL_SECONDS)
