"""进程内静态注册；函数不会由模型动态加载。"""

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
import json
from typing import Any

from ..protocol import ReplyRoute, ToolCall, ToolSpec


@dataclass(frozen=True, slots=True)
class ToolContext:
    session_id: str
    run_id: str
    call_id: str
    has_user_input: bool
    reply_route: ReplyRoute | None = None


@dataclass(frozen=True, slots=True)
class Tool:
    spec: ToolSpec
    handler: Callable[[dict[str, Any], ToolContext], Awaitable[str]]


class Tools:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.spec.name in self._tools:
                raise ValueError("duplicate tool name")
            self._tools[tool.spec.name] = tool
        self.specs = tuple(tool.spec for tool in self._tools.values())

    async def execute(self, call: ToolCall, context: ToolContext) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            return '{"error":"unknown_tool"}'
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
