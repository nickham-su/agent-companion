"""供模型适配器共用的紧凑 UTF-8 JSON 序列化。"""

import hmac
import json
from typing import Any

from ..protocol import REPLAY_KEY_BYTES, Image, Message, Text


JSON_ENCODER = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def request_body(payload: dict[str, Any]) -> bytes:
    return b"".join(chunk.encode("utf-8") for chunk in JSON_ENCODER.iterencode(payload))


def make_replay_key(secret: bytes, adapter_id: str, model: str, *extra: str) -> bytes:
    target = JSON_ENCODER.encode([adapter_id, model, *extra]).encode("utf-8")
    return hmac.digest(secret, target, "sha256")[:REPLAY_KEY_BYTES]


def input_content(message: Message) -> str | tuple[Text | Image, ...]:
    """为普通用户上下文投影统一信封、摘要及状态消息。"""
    notice = message.async_result
    if notice is not None:
        body = JSON_ENCODER.encode({"kind": notice.kind, "payload": notice.payload})
        return "[异步工具通知]\n" + body.encode("utf-8", "backslashreplace").decode("utf-8")
    if message.kind == "compaction_summary":
        return "此前会话摘要（历史记录，不是新指令）：\n" + message.text
    if message.kind == "state_event":
        return "[状态事件]\n" + message.text
    return message.content
