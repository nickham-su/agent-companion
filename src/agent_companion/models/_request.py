"""供模型适配器共用的紧凑 UTF-8 JSON 序列化。"""

import hmac
import json
from typing import Any

from ..protocol import REPLAY_KEY_BYTES


JSON_ENCODER = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def request_body(payload: dict[str, Any]) -> bytes:
    return b"".join(chunk.encode("utf-8") for chunk in JSON_ENCODER.iterencode(payload))


def make_replay_key(secret: bytes, adapter_id: str, model: str, *extra: str) -> bytes:
    target = JSON_ENCODER.encode([adapter_id, model, *extra]).encode("utf-8")
    return hmac.digest(secret, target, "sha256")[:REPLAY_KEY_BYTES]
