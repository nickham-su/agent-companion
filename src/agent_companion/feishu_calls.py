"""一次只接入一通飞书来电；来电身份和设备生命周期不改变私聊路由。"""

import asyncio
from collections import deque
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
_LOOKUP_SECONDS = 8.0
_CONNECT_SECONDS = 4.0


class CallDesktop(Protocol):
    def ringing(self) -> tuple[RingingWindow, ...]: ...
    def main_windows(self) -> tuple[WindowIdentity, ...]: ...
    def window_exists(self, identity: WindowIdentity) -> bool: ...
    def answer(self, ring: RingingWindow, *, before_deadline: Callable[[], bool]) -> bool: ...


class CallPeer(Protocol):
    async def runtime(self) -> Runtime | None: ...
    def copy_confirmed(self, info: ReplyInfo, text: str) -> None: ...
    def watch_call(self, stop: Callable[[], None]) -> bool: ...
    def unwatch_call(self, stop: Callable[[], None]) -> None: ...


VoiceFactory = Callable[[], Awaitable[tuple[VoiceChannel, AudioDevice]]]


class _CallBinding:
    def __init__(self, main: Runtime, voice: VoiceChannel, *, idle_timeout: float,
                 copy_confirmed: Callable[[ReplyInfo, str], None] | None) -> None:
        self._main, self._voice = main, voice
        self._connected = False
        self.closed = False
        self._route = ReplyRoute(voice, self.active, idle_timeout, copy_confirmed)

    def active(self) -> bool:
        return self._connected and not self.closed and not self._voice.failed

    def require_audio(self, device: AudioDevice) -> None:
        if not self._voice.failed:
            try:
                device.check_active()
                return
            except Exception:
                pass
        raise VoiceSetupError("feishu_call_audio_inactive")

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
        self._clicked_meetings: deque[str] = deque(maxlen=16)

    async def run(self, resolve: Callable[[str], CallPeer | None]) -> None:
        attempted: deque[WindowIdentity] = deque(maxlen=16)
        desktop_error: str | None = None
        while True:
            try:
                rings = self._desktop.ringing()
                for identity in tuple(attempted):
                    if not any(r.identity.overlaps(identity) for r in rings) and not self._desktop.window_exists(identity):
                        attempted.remove(identity)
                if len(rings) == 1 and not any(rings[0].identity.overlaps(identity) for identity in attempted):
                    attempted.append(rings[0].identity)
                    await self._handle(rings[0], resolve)
                desktop_error = None
            except DesktopCallError as exc:
                if desktop_error != str(exc):
                    desktop_error = str(exc)
                    _LOG.warning("Feishu auto-answer desktop unavailable (%s)", desktop_error)
            await asyncio.sleep(_POLL_SECONDS)

    async def _handle(self, ring: RingingWindow, resolve: Callable[[str], CallPeer | None]) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _LOOKUP_SECONDS

        def before_deadline() -> bool:
            return loop.time() < deadline

        try:
            async with asyncio.timeout_at(deadline):
                caller = await self._discovery.ringing_caller(self._own_id)
                if caller is None:
                    _LOG.debug("Feishu auto-answer skipped: no unique ringing caller")
                    return
                if caller.meeting_id in self._clicked_meetings:
                    return  # 同一通话的浮窗被客户端重建也不再尝试点击。
                while self._desktop.ringing() == (ring,):
                    peer = resolve(caller.sender_open_id)
                    if peer is not None:
                        try:
                            main = await peer.runtime()
                        except OperationalError as exc:
                            if not is_sqlite_lock_error(exc):
                                raise
                        else:
                            if main is not None:
                                break
                    await asyncio.sleep(_POLL_SECONDS)
                else:
                    return
                # 首版不尝试接入另一个已开的通话，避免虚拟线路对应错会话。
                if self._desktop.main_windows():
                    _LOG.debug("Feishu auto-answer skipped: pre-existing meeting window")
                    return
                voice, device = await self._voice_factory()
        except CliError as exc:
            _LOG.warning("Feishu auto-answer caller lookup failed (%s)", exc)
            return
        except TimeoutError:
            _LOG.debug("Feishu auto-answer skipped: ringing lookup timed out")
            return
        except VoiceSetupError as exc:
            _LOG.warning("Feishu auto-answer audio setup failed (%s)", exc)
            return

        binding = _CallBinding(main, voice, idle_timeout=self._idle_timeout,
                               copy_confirmed=peer.copy_confirmed if self._copy_confirmed else None)
        if not peer.watch_call(binding.stop):
            return
        try:
            async with voice.connected(device, binding.submit, capture_enabled=False, wait_for_cleanup=False):
                try:
                    if binding.closed or not before_deadline():
                        return
                    binding.require_audio(device)
                    # 同一窗口号也可能被客户端复用；打开设备后用剩余判定时间复查来电身份。
                    async with asyncio.timeout_at(deadline):
                        verified = await self._discovery.ringing_caller(self._own_id)
                    if verified != caller:
                        _LOG.debug("Feishu auto-answer skipped: ringing identity changed or became unavailable")
                        return
                    if (binding.closed or resolve(caller.sender_open_id) is not peer or self._desktop.main_windows()
                            or not before_deadline()):
                        return
                    binding.require_audio(device)
                    if not before_deadline():
                        _LOG.debug("Feishu auto-answer skipped: deadline expired during audio check")
                        return
                    self._clicked_meetings.append(caller.meeting_id)
                    if not self._desktop.answer(ring, before_deadline=before_deadline):
                        _LOG.debug("Feishu auto-answer skipped: target window changed or deadline expired")
                        return
                    window = await self._confirm(ring, binding, device)
                    if window is None:
                        _LOG.debug("Feishu auto-answer did not confirm connection")
                        return
                    binding.confirm()
                    _LOG.debug("Feishu auto-answer connected")
                    await self._follow(window, binding)
                # 在 VoiceChannel 的 TaskGroup 内消化预期错误，文字服务继续运行。
                except DesktopCallError as exc:
                    binding.stop()
                    _LOG.warning("Feishu auto-answer released binding after desktop error (%s)", exc)
                except CliError as exc:
                    _LOG.warning("Feishu auto-answer caller recheck failed (%s)", exc)
                except TimeoutError:
                    _LOG.debug("Feishu auto-answer skipped: ringing recheck timed out")
                except VoiceSetupError as exc:
                    _LOG.warning("Feishu auto-answer audio unavailable (%s)", exc)
                finally:
                    binding.stop()  # 先封闸和清 PCM，再由 connected 取消请求、关闭设备。
        except VoiceSetupError as exc:
            _LOG.warning("Feishu auto-answer audio setup failed (%s)", exc)
        finally:
            binding.stop()
            peer.unwatch_call(binding.stop)

    async def _confirm(self, ring: RingingWindow, binding: _CallBinding,
                       device: AudioDevice) -> WindowIdentity | None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _CONNECT_SECONDS
        while not binding.closed and loop.time() < deadline:
            binding.require_audio(device)
            windows = self._desktop.main_windows()
            if len(windows) == 1 and windows[0].pid == ring.identity.pid and not self._desktop.ringing():
                binding.require_audio(device)
                return windows[0]
            await asyncio.sleep(_POLL_SECONDS)
        return None

    async def _follow(self, window: WindowIdentity, binding: _CallBinding) -> None:
        while binding.active():
            if not self._desktop.window_exists(window):
                binding.stop()
                _LOG.debug("Feishu auto-answer disconnected: meeting window gone")
                return
            await asyncio.sleep(_POLL_SECONDS)
