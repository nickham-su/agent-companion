"""进程内静态注册；函数不会由模型动态加载。"""

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
import json
import logging
from typing import Any

from ..protocol import ToolCall, ToolSpec
from ..workspace import Workspace
from ._tool_output import limit_tool_output


TOOL_CANCELLED = '{"error":"tool_cancelled"}'
_LOG = logging.getLogger(__name__)


def _log_rejected(context: "ToolContext", reason: str, *, name: str = "-") -> None:
    # name 只能来自已注册的定义；不回显模型提供的未知工具名、参数或调用 ID。
    _LOG.debug("tool dispatch rejected; run=%s reply=%s tool=%s reason=%s",
               context.run_id, context.reply_id or "-", name, reason)


@dataclass(frozen=True, slots=True)
class ToolContext:
    session_id: str
    run_id: str
    call_id: str
    agent_id: str | None = None
    workspace: Workspace | None = None
    reply_id: str | None = None


@dataclass(frozen=True, slots=True)
class Tool:
    spec: ToolSpec
    handler: Callable[[dict[str, Any], ToolContext], Awaitable[str]]
    serial_per_request: bool = False
    reject_duplicate_keys: bool = False
    record_result_on_shutdown: bool = False


class Tools:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.spec.name in self._tools:
                raise ValueError("duplicate tool name")
            self._tools[tool.spec.name] = tool
        self.specs = tuple(tool.spec for tool in self._tools.values())

    def request_locks(self) -> dict[str, asyncio.Lock]:
        """每次模型请求独立分配；不同工具名之间不互相等待。"""
        return {name: asyncio.Lock() for name, tool in self._tools.items() if tool.serial_per_request}

    def records_result_on_shutdown(self, name: str) -> bool:
        """只保留显式启用的工具已返回的结果，不改变工具任务的取消方式。"""
        tool = self._tools.get(name)
        return tool is not None and tool.record_result_on_shutdown

    async def execute(self, call: ToolCall, context: ToolContext, *,
                      request_locks: Mapping[str, asyncio.Lock] | None = None,
                      can_start_serial: Callable[[], bool] | None = None) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            _log_rejected(context, "unknown_tool")
            return '{"error":"unknown_tool"}'
        if request_locks is not None and (lock := request_locks.get(call.name)) is not None:
            # asyncio.Lock 按等待顺序放行；Runtime 按工具帧顺序创建执行任务。
            async with lock:
                # 排队期间请求可能已中断；只禁止未开始的串行调用。
                if can_start_serial is not None and not can_start_serial():
                    _log_rejected(context, "tool_cancelled", name=tool.spec.name)
                    return TOOL_CANCELLED
                return await self._invoke(tool, call, context)
        return await self._invoke(tool, call, context)

    @staticmethod
    async def _invoke(tool: Tool, call: ToolCall, context: ToolContext) -> str:
        try:
            args = json.loads(call.arguments, parse_constant=_invalid_constant,
                              object_pairs_hook=_unique_object if tool.reject_duplicate_keys else None)
            if not isinstance(args, dict):
                raise ValueError
        except (ValueError, RecursionError):
            _log_rejected(context, "invalid_arguments", name=tool.spec.name)
            return '{"error":"invalid_arguments"}'
        try:
            result = await tool.handler(args, context)
            if not isinstance(result, str):
                raise TypeError
            return await limit_tool_output(result, context.workspace)
        except Exception:
            # 工具错误本身作为终态结果，不把可能含有数据的异常正文投影给模型。
            return '{"error":"tool_failed"}'


def _invalid_constant(_: str) -> None:
    raise ValueError("invalid JSON constant")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result
