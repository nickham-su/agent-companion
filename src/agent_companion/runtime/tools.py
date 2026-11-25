"""进程内静态注册；函数不会由模型动态加载。"""

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
import json
from typing import Any

from ..protocol import ToolCall, ToolSpec
from ..workspace import Workspace


TOOL_CANCELLED = '{"error":"tool_cancelled"}'


@dataclass(frozen=True, slots=True)
class ToolContext:
    session_id: str
    run_id: str
    call_id: str
    has_user_input: bool
    agent_id: str | None = None
    has_agent_input: bool = False
    workspace: Workspace | None = None


@dataclass(frozen=True, slots=True)
class Tool:
    spec: ToolSpec
    handler: Callable[[dict[str, Any], ToolContext], Awaitable[str]]
    advertised: bool = True
    serial_per_request: bool = False


class Tools:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.spec.name in self._tools:
                raise ValueError("duplicate tool name")
            self._tools[tool.spec.name] = tool
        self.specs = tuple(tool.spec for tool in self._tools.values() if tool.advertised)


    def request_locks(self) -> dict[str, asyncio.Lock]:
        """每次模型请求独立分配；不同工具名之间不互相等待。"""
        return {name: asyncio.Lock() for name, tool in self._tools.items() if tool.serial_per_request}

    async def execute(self, call: ToolCall, context: ToolContext, *,
                      request_locks: Mapping[str, asyncio.Lock] | None = None,
                      can_start_serial: Callable[[], bool] | None = None) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            return '{"error":"unknown_tool"}'
        if request_locks is not None and (lock := request_locks.get(call.name)) is not None:
            # asyncio.Lock 按等待顺序放行；Runtime 按工具帧顺序创建执行任务。
            async with lock:
                # 排队期间请求可能已中断；只禁止未开始的串行调用。
                if can_start_serial is not None and not can_start_serial():
                    return TOOL_CANCELLED
                return await self._invoke(tool, call, context)
        return await self._invoke(tool, call, context)

    @staticmethod
    async def _invoke(tool: Tool, call: ToolCall, context: ToolContext) -> str:
        try:
            args = json.loads(call.arguments, parse_constant=_invalid_constant)
            if not isinstance(args, dict):
                raise ValueError
        except (ValueError, RecursionError):
            return '{"error":"invalid_arguments"}'
        try:
            result = await tool.handler(args, context)
            if not isinstance(result, str):
                raise TypeError
            return result
        except Exception:
            # 工具错误本身作为终态结果，不把可能含有数据的异常正文投影给模型。
            return '{"error":"tool_failed"}'


def _invalid_constant(_: str) -> None:
    raise ValueError("invalid JSON constant")
