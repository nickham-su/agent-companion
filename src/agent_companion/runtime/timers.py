"""主会话的进程内定时器；到期仅向当前绑定的调度器交接内部输入。"""

from asyncio import TimerHandle, get_running_loop
from collections.abc import Callable
from dataclasses import dataclass, field
import json
import logging
from math import isfinite
from secrets import token_hex
from typing import Any

from ..protocol import InternalInput, TimerFired, ToolSpec
from .tools import Tool, ToolContext


_LOG = logging.getLogger(__name__)
SET_TIMER_TOOL_NAME = "set_timer"
CANCEL_TIMER_TOOL_NAME = "cancel_timer"
_INVALID_ARGUMENTS = '{"error":"invalid_arguments"}'
_MIN_REPEAT_INTERVAL = 1
_MAX_REPEAT_FIRES = 100
_MAX_ACTIVE_TIMERS = 8
_SET_TIMER = ToolSpec(
    SET_TIMER_TOOL_NAME,
    "登记会话定时器，立即返回 timer_id，不等待到期。到期后模型会收到内部定时器事件，"
    f"可稍晚处理；只有纳入真实用户输入的运行可以登记。同一会话最多 {_MAX_ACTIVE_TIMERS} 个活跃定时器。",
    {
        "type": "object",
        "properties": {
            "interval_seconds": {
                "type": "number", "exclusiveMinimum": 0,
                "description": f"到第一次触发的间隔秒数；循环时也是后续间隔，且至少 {_MIN_REPEAT_INTERVAL} 秒",
            },
            "mode": {"type": "string", "enum": ["once", "repeat"], "default": "once"},
            "max_fires": {"type": "integer", "minimum": 1, "maximum": _MAX_REPEAT_FIRES,
                          "description": "循环模式必填的最大触发次数；单次模式不要传"},
            "message": {"type": "string", "description": "可选，到期时传给模型的内容"},
        },
        "required": ["interval_seconds"],
        "additionalProperties": False,
    },
)
_CANCEL_TIMER = ToolSpec(
    CANCEL_TIMER_TOOL_NAME,
    "用 timer_id 取消本会话定时器的后续触发，不撤回已经投递的事件。"
    "定时器事件触发的运行也可调用；已完成、已取消或找不到时返回 not_active。",
    {
        "type": "object",
        "properties": {"timer_id": {"type": "string", "minLength": 1}},
        "required": ["timer_id"],
        "additionalProperties": False,
    },
)


@dataclass(slots=True)
class _Timer:
    interval: float
    repeat: bool
    max_fires: int
    message: str | None = field(repr=False)
    fire_count: int = 0
    handle: TimerHandle | None = None


class Timers:
    """一条会话绑定下的工具和回调，仅由同一个事件循环驱动。"""

    def __init__(self, session_id: str, submit: Callable[[InternalInput], bool]) -> None:
        self._session_id = session_id
        self._submit = submit
        self._loop = get_running_loop()
        self._id_prefix = f"t-{token_hex(5)}"
        self._next_id = 1
        self._active: dict[str, _Timer] = {}
        self._closed = False
        self.tools = (Tool(_SET_TIMER, self._set), Tool(_CANCEL_TIMER, self._cancel))

    async def _set(self, args: dict[str, Any], context: ToolContext) -> str:
        if context.session_id != self._session_id:
            return '{"error":"not_parent_session"}'
        if self._closed:
            return '{"error":"timers_closed"}'
        if not context.has_user_input:
            return '{"error":"timer_requires_user_input"}'

        interval = args.get("interval_seconds")
        mode = args.get("mode", "once")
        if (args.keys() - {"interval_seconds", "mode", "max_fires", "message"}
                or type(interval) not in (int, float) or mode not in ("once", "repeat")):
            return _INVALID_ARGUMENTS
        try:
            interval = float(interval)
        except OverflowError:
            return _INVALID_ARGUMENTS
        if not isfinite(interval) or interval <= 0:
            return _INVALID_ARGUMENTS
        repeat = mode == "repeat"
        if repeat:
            max_fires = args.get("max_fires")
            if (interval < _MIN_REPEAT_INTERVAL or type(max_fires) is not int
                    or not 1 <= max_fires <= _MAX_REPEAT_FIRES):
                return _INVALID_ARGUMENTS
        else:
            if "max_fires" in args:
                return _INVALID_ARGUMENTS
            max_fires = 1
        message = args.get("message")
        if "message" in args and not isinstance(message, str):
            return _INVALID_ARGUMENTS
        if len(self._active) >= _MAX_ACTIVE_TIMERS:
            return '{"error":"timer_capacity_reached"}'

        timer_id = f"{self._id_prefix}-{self._next_id}"
        self._next_id += 1
        timer = _Timer(interval, repeat, max_fires, message)
        timer.handle = self._loop.call_later(timer.interval, self._fire, timer_id)
        self._active[timer_id] = timer
        return json.dumps({"status": "accepted", "timer_id": timer_id})

    def _fire(self, timer_id: str) -> None:
        timer = self._active.get(timer_id)
        if timer is None:
            return
        timer.handle = None
        count = timer.fire_count + 1
        event = TimerFired(timer_id, count, timer.max_fires, timer.repeat, timer.message)
        try:
            accepted = self._submit(InternalInput(event))
        except Exception as exc:
            _LOG.warning("timer event submission failed; error_type=%s", type(exc).__name__)
            accepted = False
        if not accepted or count == timer.max_fires:
            self._active.pop(timer_id, None)
        elif self._active.get(timer_id) is timer:
            timer.fire_count = count
            # 从实际交接开始等下一段间隔；延误后不补发积压的触发。
            timer.handle = self._loop.call_later(timer.interval, self._fire, timer_id)

    async def _cancel(self, args: dict[str, Any], context: ToolContext) -> str:
        timer_id = args.get("timer_id")
        if args.keys() != {"timer_id"} or not isinstance(timer_id, str) or not timer_id.strip():
            return _INVALID_ARGUMENTS
        timer = self._active.pop(timer_id, None) if context.session_id == self._session_id else None
        if timer is None:
            return '{"status":"not_active"}'
        if timer.handle is not None:
            timer.handle.cancel()
        return '{"status":"cancelled"}'

    def close(self) -> None:
        self._closed = True
        for timer in self._active.values():
            if timer.handle is not None:
                timer.handle.cancel()
        self._active.clear()
