"""Agent 树共享的主动文字发送入口及角色工具。"""

import asyncio
import json
from typing import Any

from ..protocol import TextSender, TextSendResult, ToolSpec
from .tools import Tool, ToolContext


SEND_TEXT_TOOL_NAME = "send_text"
_MAX_TEXT_CHARS = 1_000


class TextSendBinding:
    """只绑定一次；撤销后不再接受新发送，也不会指向重开的会话。"""

    def __init__(self, sender: TextSender | None) -> None:
        self._sender = sender

    def revoke(self) -> None:
        self._sender = None

    async def __call__(self, text: str) -> TextSendResult:
        sender = self._sender
        if sender is None:
            return TextSendResult.UNAVAILABLE
        try:
            return await sender(text)
        except asyncio.CancelledError:
            if self._sender is not None:
                raise
            # 根会话已撤销入口，但这个调用已经进入发送端，不能断言没有发出。
            return TextSendResult.UNCONFIRMED


def make_send_text_tool(sender: TextSender | None) -> Tool:
    async def send_text(args: dict[str, Any], _: ToolContext) -> str:
        text = args.get("text")
        if args.keys() != {"text"} or not isinstance(text, str):
            result = TextSendResult.INVALID_TEXT
        elif len(text) > _MAX_TEXT_CHARS:
            result = TextSendResult.TEXT_TOO_LONG
        elif sender is None:
            result = TextSendResult.UNAVAILABLE
        else:
            result = await sender(text)
        if result is TextSendResult.SENT:
            return '{"ok":true}'
        return json.dumps({"ok": False, "reason": result.value}, separators=(",", ":"))

    return Tool(ToolSpec(
        SEND_TEXT_TOOL_NAME,
        "把链接、代码、路径或其他纯文本发给当前会话已绑定的用户。发送未确认不代表没有发出。",
        {
            "type": "object",
            "properties": {"text": {"type": "string", "maxLength": _MAX_TEXT_CHARS, "description": "纯文本正文"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    ), send_text, reject_duplicate_keys=True, record_result_on_shutdown=True)
