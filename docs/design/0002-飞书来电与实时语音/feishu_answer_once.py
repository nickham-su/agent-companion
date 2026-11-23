#!/usr/bin/env python3
"""检测当前 Mac 上的飞书来电浮窗，模拟点击一次接听。"""

import argparse
import ctypes as C
import plistlib
import sys
import time
from dataclasses import dataclass

FRAMEWORKS = "/System/Library/Frameworks"
MEETING_OWNER = "飞书会议"
POPUP_LAYER = 25
POPUP_WIDTH = 368
POPUP_HEIGHT = 86
FROM_RIGHT = 35
FROM_TOP = 45
VISIBLE_WINDOWS = 17
BINARY_PROPERTY_LIST = 200
LEFT_DOWN, LEFT_UP, LEFT_BUTTON = 1, 2, 0
CLICK_STATE, HID_EVENT_TAP = 1, 0


class Point(C.Structure):
    _fields_ = [("x", C.c_double), ("y", C.c_double)]


@dataclass(frozen=True)
class Popup:
    pid: int
    left: float
    top: float

    @property
    def answer_point(self) -> Point:
        return Point(self.left + POPUP_WIDTH - FROM_RIGHT, self.top + FROM_TOP)


def bind(library, name, result, *arguments):
    function = getattr(library, name)
    function.restype = result
    function.argtypes = arguments
    return function


class Desktop:
    def __init__(self):
        cg = C.CDLL(f"{FRAMEWORKS}/CoreGraphics.framework/CoreGraphics")
        cf = C.CDLL(f"{FRAMEWORKS}/CoreFoundation.framework/CoreFoundation")
        pointer = C.c_void_p
        self.copy_windows = bind(cg, "CGWindowListCopyWindowInfo", pointer, C.c_uint32, C.c_uint32)
        self.can_post = bind(cg, "CGPreflightPostEventAccess", C.c_bool)
        self.move = bind(cg, "CGWarpMouseCursorPosition", C.c_int32, Point)
        self.mouse_event = bind(cg, "CGEventCreateMouseEvent", pointer, pointer, C.c_uint32, Point, C.c_uint32)
        self.set_integer_field = bind(cg, "CGEventSetIntegerValueField", None, pointer, C.c_uint32, C.c_int64)
        self.post = bind(cg, "CGEventPost", None, C.c_uint32, pointer)
        self.plist_data = bind(
            cf, "CFPropertyListCreateData", pointer, pointer, pointer, C.c_long, C.c_ulong, pointer
        )
        self.data_pointer = bind(cf, "CFDataGetBytePtr", pointer, pointer)
        self.data_length = bind(cf, "CFDataGetLength", C.c_long, pointer)
        self.release = bind(cf, "CFRelease", None, pointer)

    def visible_windows(self):
        windows = self.copy_windows(VISIBLE_WINDOWS, 0)
        if not windows:
            raise RuntimeError("无法读取当前桌面会话的窗口列表。")
        try:
            data = self.plist_data(None, windows, BINARY_PROPERTY_LIST, 0, None)
            if not data:
                raise RuntimeError("无法读取窗口位置。")
            try:
                return plistlib.loads(C.string_at(self.data_pointer(data), self.data_length(data)))
            finally:
                self.release(data)
        finally:
            self.release(windows)

    def click(self, point: Point):
        down = self.mouse_event(None, LEFT_DOWN, point, LEFT_BUTTON)
        up = self.mouse_event(None, LEFT_UP, point, LEFT_BUTTON)
        try:
            if not down or not up:
                raise RuntimeError("macOS 未创建完整的鼠标点击事件。")
            self.set_integer_field(down, CLICK_STATE, 1)
            self.set_integer_field(up, CLICK_STATE, 1)
            self.post(HID_EVENT_TAP, down)
            try:
                time.sleep(0.05)
            finally:
                self.post(HID_EVENT_TAP, up)
        finally:
            for event in (down, up):
                if event:
                    self.release(event)


def popup_from(window) -> Popup | None:
    if window.get("kCGWindowOwnerName") != MEETING_OWNER or window.get("kCGWindowLayer") != POPUP_LAYER:
        return None
    bounds = window["kCGWindowBounds"]
    if (bounds["Width"], bounds["Height"]) != (POPUP_WIDTH, POPUP_HEIGHT):
        return None
    return Popup(window["kCGWindowOwnerPID"], bounds["X"], bounds["Y"])


def find_popup(windows) -> Popup | None:
    # 飞书为同一个提示浮窗创建两个重叠窗口，按进程和位置合并。
    candidates = {popup for window in windows if (popup := popup_from(window)) is not None}
    return next(iter(candidates)) if len(candidates) == 1 else None


def front_window_at(windows, point: Point):
    for window in windows:
        bounds = window["kCGWindowBounds"]
        left, top = bounds["X"], bounds["Y"]
        if window.get("kCGWindowAlpha", 1) > 0 and (
            left <= point.x < left + bounds["Width"] and top <= point.y < top + bounds["Height"]
        ):
            return window
    return {}


def answer_once(desktop: Desktop, seconds: float):
    desktop.visible_windows()
    if not desktop.can_post():
        raise RuntimeError("macOS 当前未授权发送鼠标事件。")
    print("已开始监听飞书来电浮窗。", flush=True)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        windows = desktop.visible_windows()
        popup = find_popup(windows)
        if popup is not None:
            point = popup.answer_point
            if popup_from(front_window_at(windows, point)) == popup:
                if desktop.move(point):
                    raise RuntimeError("macOS 未能将鼠标移动到接听按钮。")
                windows = desktop.visible_windows()
                if find_popup(windows) == popup and popup_from(front_window_at(windows, point)) == popup:
                    desktop.click(point)
                    print("已向接听位置发送一次点击。", flush=True)
                    return 0
        time.sleep(0.25)
    print("监听结束，没有点击。", flush=True)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=90, help="等待来电的秒数，默认 90")
    args = parser.parse_args()
    if not 0 < args.seconds <= 300:
        parser.error("等待时间应大于 0 秒且不超过 300 秒。")
    if sys.platform != "darwin":
        parser.exit(2, "请在 macOS 桌面会话中运行。\n")
    try:
        return answer_once(Desktop(), args.seconds)
    except RuntimeError as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
