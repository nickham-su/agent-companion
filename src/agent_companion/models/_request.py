"""UTF-8 JSON 的准确请求字节限额，供模型适配器共用。"""

import json
from typing import Any

from ..protocol import RequestTooLarge


JSON_ENCODER = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _text_chars(value: Any) -> int:
    """廉价下限：大图片以 ASCII 内联，先用长度拒绝明显超限的请求。"""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_text_chars(key) + _text_chars(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return sum(map(_text_chars, value))
    return 0


def request_body(payload: dict | list, limit: int, *, output: bytearray | None = None) -> int:
    if _text_chars(payload) > limit:
        raise RequestTooLarge("request_body_too_large")
    remaining = limit
    for chunk in JSON_ENCODER.iterencode(payload):
        if len(chunk) > remaining:
            raise RequestTooLarge("request_body_too_large")
        if output is not None or not chunk.isascii():
            encoded = chunk.encode("utf-8")
            remaining -= len(encoded)
            if remaining < 0:
                raise RequestTooLarge("request_body_too_large")
            if output is not None:
                output.extend(encoded)
        else:
            remaining -= len(chunk)
    return remaining
