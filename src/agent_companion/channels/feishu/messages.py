"""将飞书 OpenAPI 原始消息规范化为有序文本和图片。"""

from base64 import b64encode
from dataclasses import dataclass, field, replace
import json
import re
from typing import Any

from ...protocol import OMITTED_INPUT_IMAGE, ChannelUser, Image, Message, Part, Text
from ...speech.asr import AsrError, AudioFile, FileASR, MAX_AUDIO_BYTES
from .cli import CliError, FeishuCLI, parse_open_id


SUPPORTED_TYPES = {"audio", "image", "post", "text"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_BATCH_BYTES = 32 * 1024 * 1024  # 给模型工具声明和协议封装留余量。
_MAX_POST_NODES = 1000


class MessageFormatError(Exception):
    """无正文、未知结构或不支持的图片，错误不包含原始消息。"""


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    message_id: str
    created_ms: int
    kind: str
    content: str = field(repr=False)


def incoming(raw: Any, *, chat_id: str, user: ChannelUser, account: ChannelUser) -> IncomingMessage | None:
    if not isinstance(raw, dict) or raw.get("chat_id") != chat_id or raw.get("deleted") is True:
        return None
    sender = raw.get("sender")
    if (not isinstance(sender, dict) or sender.get("sender_type") != "user"
            or sender.get("id_type") != "open_id"):
        return None
    author = parse_open_id(sender.get("id"))
    if author is None or author == account or author != user:
        return None
    kind, message_id, created = raw.get("msg_type"), raw.get("message_id"), raw.get("create_time")
    if (not isinstance(kind, str) or kind not in SUPPORTED_TYPES or not isinstance(message_id, str)
            or not re.fullmatch(r"om_[\w-]{1,160}", message_id, flags=re.ASCII)
            or not isinstance(created, str) or not re.fullmatch(r"[0-9]{1,16}", created)):
        return None
    body = raw.get("body")
    content = body.get("content") if isinstance(body, dict) else None
    return IncomingMessage(message_id, int(created), kind, content if isinstance(content, str) else "")


def _read_content(encoded: str) -> dict[str, Any]:
    try:
        content = json.loads(encoded)
    except (ValueError, RecursionError):
        raise MessageFormatError("invalid_message_content") from None
    if not isinstance(content, dict):
        raise MessageFormatError("invalid_message_content")
    return content


def _required(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MessageFormatError("missing_message_content")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise MessageFormatError("invalid_message_encoding") from None
    return value


def _post_body(content: dict[str, Any]) -> dict[str, Any]:
    # 读取接口通常已选语言；也接受包含多个语言的原始 post 对象。
    content = content.get("post", content)
    if isinstance(content, dict):
        if isinstance(content.get("content"), list):
            return content
        candidates = (content.get("zh_cn"), content.get("en_us"), *content.values())
        for candidate in candidates:
            if isinstance(candidate, dict) and isinstance(candidate.get("content"), list):
                return candidate
    raise MessageFormatError("invalid_post_content")


def _image_media_type(data: bytes) -> str:
    if (len(data) >= 45 and data.startswith(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
            and data.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82")):
        return "image/png"
    if len(data) >= 12 and data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9"):
        return "image/jpeg"
    if len(data) >= 14 and data.startswith((b"GIF87a", b"GIF89a")) and data.endswith(b";"):
        return "image/gif"
    if len(data) >= 20 and data.startswith(b"RIFF") and data[8:12] == b"WEBP" and data[12:16] in (
        b"VP8 ", b"VP8L", b"VP8X"
    ) and int.from_bytes(data[4:8], "little") + 8 == len(data):
        return "image/webp"
    raise MessageFormatError("unsupported_image_format")


def _text_bytes(value: str) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _image_bytes(encoded_length: int) -> int:
    return encoded_length + 64  # 含 data URL 及图片内容块的 JSON 包装。


class BatchMediaBudget:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self.remaining = limit

    def reserve_image(self, encoded_length: int) -> bool:
        cost = _image_bytes(encoded_length)
        if cost > self.remaining:
            return False
        self.remaining -= cost
        return True

    def finish(self, items: list[tuple[str, Message | None]]) -> list[tuple[str, Message | None]]:
        # 先为同批所有文字与占位留空间，再按原顺序保留可装下的图片。
        placeholder_cost = _text_bytes(OMITTED_INPUT_IMAGE.text) + 32
        body = sum(64 + (_text_bytes(message.content) if isinstance(message.content, str) else
                        sum(_text_bytes(part.text) + 32 if isinstance(part, Text) else placeholder_cost
                            for part in message.content))
                   for _, message in items if message is not None)
        remaining = self._limit - body
        final = []
        for external_id, message in items:
            if message is not None and not isinstance(message.content, str):
                parts: list[Part] = []
                for part in message.content:
                    if isinstance(part, Image):
                        cost = _image_bytes(len(part.url)) - placeholder_cost
                        if cost > remaining:
                            part = OMITTED_INPUT_IMAGE
                        else:
                            remaining -= cost
                    parts.append(part)
                message = replace(message, content=tuple(parts))
            final.append((external_id, message))
        return final


class MessageNormalizer:
    def __init__(self, cli: FeishuCLI, asr: FileASR) -> None:
        self._cli = cli
        self._asr = asr

    async def normalize(self, raw: IncomingMessage, budget: BatchMediaBudget) -> Message:
        content = _read_content(raw.content)
        if raw.kind == "text":
            return Message("user", _required(content.get("text")))
        if raw.kind == "image":
            image = await self._image(raw.message_id, _required(content.get("image_key")), budget)
            return Message("user", (image,))
        if raw.kind == "audio":
            data = await self._cli.download(raw.message_id, _required(content.get("file_key")), "file",
                                            limit=MAX_AUDIO_BYTES)
            return Message("user", _required(await self._asr.transcribe(AudioFile(data))))
        return Message("user", await self._post(raw.message_id, _post_body(content), budget))

    async def _image(self, message_id: str, key: str, budget: BatchMediaBudget) -> Part:
        if budget.remaining <= _image_bytes(23):
            return OMITTED_INPUT_IMAGE
        data = await self._cli.download(message_id, key, "image", limit=MAX_IMAGE_BYTES)
        mime = _image_media_type(data)
        if not budget.reserve_image(len(f"data:{mime};base64,") + 4 * ((len(data) + 2) // 3)):
            return OMITTED_INPUT_IMAGE
        return Image(f"data:{mime};base64,{b64encode(data).decode('ascii')}")

    async def _post(self, message_id: str, post: dict[str, Any], budget: BatchMediaBudget) -> tuple[Part, ...]:
        parts: list[Part] = []
        title = post.get("title")
        if isinstance(title, str) and title.strip():
            parts.append(Text(_required(title)))
        count = 0
        for paragraph in post["content"]:
            if not isinstance(paragraph, list):
                raise MessageFormatError("invalid_post_paragraph")
            count += len(paragraph)
            if count > _MAX_POST_NODES:
                raise MessageFormatError("post_node_limit")
            line: list[Part] = []
            for node in paragraph:
                if not isinstance(node, dict):
                    raise MessageFormatError("invalid_post_node")
                tag = node.get("tag")
                if tag == "img":
                    line.append(await self._image(message_id, _required(node.get("image_key")), budget))
                elif tag == "a":
                    label, target = node.get("text"), node.get("href")
                    label = _required(label) if isinstance(label, str) and label.strip() else ""
                    target = _required(target) if isinstance(target, str) and target.strip() else ""
                    if label or target:
                        # 保留目标作为普通文本；不解析、访问或下载外部链接。
                        line.append(Text(f"{label} ({target})" if label and target and target != label else label or target))
                elif tag in ("text", "at", "code_block", "md"):
                    value = node.get("text")
                    if isinstance(value, str) and value:
                        line.append(Text(_required(value) if value.strip() else value))
            if line:
                if parts:
                    parts.append(Text("\n"))
                parts.extend(line)
        if not parts:
            raise MessageFormatError("empty_post")
        return tuple(parts)


NORMALIZE_ERRORS = (MessageFormatError, CliError, AsrError, TimeoutError)
