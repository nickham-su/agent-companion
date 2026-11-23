"""供模型适配器共用的紧凑 UTF-8 JSON 序列化。"""

import json
from typing import Any


JSON_ENCODER = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def request_body(payload: dict[str, Any]) -> bytes:
    return b"".join(chunk.encode("utf-8") for chunk in JSON_ENCODER.iterencode(payload))
