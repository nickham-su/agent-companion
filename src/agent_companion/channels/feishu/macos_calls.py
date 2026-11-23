"""0002 的 macOS 桌面适配器：双记录归并、命中校验和窗口身份跟踪。"""

from collections.abc import Callable
import ctypes as C
from dataclasses import dataclass
import plistlib
import sys
import time
from typing import Any

from ...speech.vad import VoiceSetupError


_FRAMEWORKS = "/System/Library/Frameworks"
_MEETING_OWNER = "飞书会议"
_POPUP_LAYER = 25
_POPUP_SIZE = (368, 86)
_EXCLUDE_DESKTOP = 1 << 4
_ON_SCREEN_ONLY = 1


class DesktopCallError(RuntimeError):
    """消息只使用固定诊断码，不包含窗口标题或其它桌面数据。"""


class _Point(C.Structure):
    _fields_ = [("x", C.c_double), ("y", C.c_double)]


@dataclass(frozen=True, slots=True)
class WindowIdentity:
    pid: int
    numbers: frozenset[int]

    def overlaps(self, other: "WindowIdentity") -> bool:
        return self.pid == other.pid and not self.numbers.isdisjoint(other.numbers)


@dataclass(frozen=True, slots=True)
class RingingWindow:
    identity: WindowIdentity
    bounds: tuple[float, float, float, float]

    @property
    def answer_point(self) -> _Point:
        left, top, width, _ = self.bounds
        return _Point(left + width - 35, top + 45)


_Window = dict[str, Any]


def _bounds(window: _Window) -> tuple[float, float, float, float]:
    bounds = window["kCGWindowBounds"]
    return bounds["X"], bounds["Y"], bounds["Width"], bounds["Height"]


def _groups(windows: list[_Window], layer: int) -> list[tuple[RingingWindow, bool]]:
    groups: dict[tuple[int, tuple[float, float, float, float]], tuple[set[int], bool]] = {}
    for window in windows:
        if window.get("kCGWindowOwnerName") != _MEETING_OWNER or window.get("kCGWindowLayer") != layer:
            continue
        key = window["kCGWindowOwnerPID"], _bounds(window)
        numbers, named = groups.get(key, (set(), False))
        numbers.add(window["kCGWindowNumber"])
        groups[key] = numbers, named or window.get("kCGWindowName") == _MEETING_OWNER
    return [(RingingWindow(WindowIdentity(pid, frozenset(numbers)), bounds), named)
            for (pid, bounds), (numbers, named) in groups.items()]


def _rings(windows: list[_Window]) -> tuple[RingingWindow, ...]:
    return tuple(window for window, _ in _groups(windows, _POPUP_LAYER) if window.bounds[2:] == _POPUP_SIZE)


def _front_at(windows: list[_Window], point: _Point) -> _Window | None:
    for window in windows:
        left, top, width, height = _bounds(window)
        if (window.get("kCGWindowAlpha", 1) > 0
                and left <= point.x < left + width and top <= point.y < top + height):
            return window
    return None


def _is_target(windows: list[_Window], ring: RingingWindow, point: _Point) -> bool:
    if _rings(windows) != (ring,):
        return False
    front = _front_at(windows, point)
    return (front is not None and front["kCGWindowOwnerPID"] == ring.identity.pid
            and front["kCGWindowNumber"] in ring.identity.numbers)


def _function(library, name, result, *arguments):
    function = getattr(library, name)
    function.restype, function.argtypes = result, arguments
    return function


class MacCallDesktop:
    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise VoiceSetupError("feishu_auto_answer_requires_macos_desktop")
        cg = C.CDLL(f"{_FRAMEWORKS}/CoreGraphics.framework/CoreGraphics")
        cf = C.CDLL(f"{_FRAMEWORKS}/CoreFoundation.framework/CoreFoundation")
        pointer = C.c_void_p
        self._copy = _function(cg, "CGWindowListCopyWindowInfo", pointer, C.c_uint32, C.c_uint32)
        self._can_post = _function(cg, "CGPreflightPostEventAccess", C.c_bool)
        self._move = _function(cg, "CGWarpMouseCursorPosition", C.c_int32, _Point)
        self._mouse = _function(cg, "CGEventCreateMouseEvent", pointer, pointer, C.c_uint32, _Point, C.c_uint32)
        self._set = _function(cg, "CGEventSetIntegerValueField", None, pointer, C.c_uint32, C.c_int64)
        self._post = _function(cg, "CGEventPost", None, C.c_uint32, pointer)
        self._plist = _function(cf, "CFPropertyListCreateData", pointer, pointer, pointer, C.c_long, C.c_ulong, pointer)
        self._bytes = _function(cf, "CFDataGetBytePtr", pointer, pointer)
        self._length = _function(cf, "CFDataGetLength", C.c_long, pointer)
        self._release = _function(cf, "CFRelease", None, pointer)
        if not self._can_post():
            raise VoiceSetupError("feishu_auto_answer_mouse_permission_missing")
        try:
            self._windows(visible=True)
        except DesktopCallError as exc:
            raise VoiceSetupError(str(exc)) from None

    def _windows(self, *, visible: bool) -> list[_Window]:
        options = _EXCLUDE_DESKTOP | (_ON_SCREEN_ONLY if visible else 0)
        windows = self._copy(options, 0)
        if not windows:
            raise DesktopCallError("desktop_window_list_unavailable")
        try:
            data = self._plist(None, windows, 200, 0, None)  # kCFPropertyListBinaryFormat_v1_0
            if not data:
                raise DesktopCallError("desktop_window_data_unavailable")
            try:
                result = plistlib.loads(C.string_at(self._bytes(data), self._length(data)))
            except (ValueError, plistlib.InvalidFileException):
                raise DesktopCallError("desktop_window_data_invalid") from None
            finally:
                self._release(data)
        finally:
            self._release(windows)
        if not isinstance(result, list):
            raise DesktopCallError("desktop_window_data_invalid")
        for window in result:
            bounds = window.get("kCGWindowBounds") if isinstance(window, dict) else None
            if (not isinstance(bounds, dict) or not all(type(bounds.get(key)) in (int, float)
                                                       for key in ("X", "Y", "Width", "Height"))
                    or not all(type(window.get(key)) is int for key in ("kCGWindowOwnerPID", "kCGWindowNumber"))):
                raise DesktopCallError("desktop_window_data_invalid")
        return result

    def ringing(self) -> tuple[RingingWindow, ...]:
        return _rings(self._windows(visible=True))

    def main_windows(self) -> tuple[WindowIdentity, ...]:
        return tuple(window.identity for window, named in _groups(self._windows(visible=False), 0) if named)

    def window_exists(self, identity: WindowIdentity) -> bool:
        return any(window["kCGWindowOwnerPID"] == identity.pid and window["kCGWindowNumber"] in identity.numbers
                   for window in self._windows(visible=False))

    def answer(self, ring: RingingWindow, *, before_deadline: Callable[[], bool]) -> bool:
        point = ring.answer_point
        if not _is_target(self._windows(visible=True), ring, point) or not before_deadline():
            return False
        if not self._can_post() or self._move(point):
            raise DesktopCallError("desktop_mouse_event_unavailable")
        if not _is_target(self._windows(visible=True), ring, point):
            return False
        down, up = self._mouse(None, 1, point, 0), self._mouse(None, 2, point, 0)
        try:
            if not down or not up:
                raise DesktopCallError("desktop_mouse_event_unavailable")
            self._set(down, 1, 1)  # kCGMouseEventClickState
            self._set(up, 1, 1)
            # 窗口枚举、移动和事件准备都是同步操作，发送按下前才做最终截止检查。
            if not before_deadline():
                return False
            self._post(0, down)  # kCGHIDEventTap
            try:
                time.sleep(0.05)
            finally:
                self._post(0, up)  # 已按下就必须释放，不再检查时限。
        finally:
            for event in (down, up):
                if event:
                    self._release(event)
        return True
