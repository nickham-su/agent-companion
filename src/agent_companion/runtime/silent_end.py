"""模型可请求无正文地结束本轮；最终决策由 Runtime 做出。"""

from typing import Any

from ..protocol import ToolSpec
from .tools import Tool, ToolContext


SILENT_END_TOOL_NAME = "silent_end"
SILENT_END_ACCEPTED = '{"status":"accepted"}'


async def _request_silent_end(args: dict[str, Any], _: ToolContext) -> str:
    if args:
        return '{"error":"invalid_arguments"}'
    return SILENT_END_ACCEPTED


SILENT_END_TOOL = Tool(ToolSpec(
    SILENT_END_TOOL_NAME,
    "沉默：如果本轮没有需要告知的新信息，也没有待答复的问题或到期应发送的提醒，可调用此工具。"
    "无需参数，可与其他工具一起调用；同一次请求有非空白正文时不生效，否则其他工具结算后结束本轮。",
    {"type": "object", "properties": {}, "additionalProperties": False},
), _request_silent_end)
