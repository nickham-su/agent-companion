"""将飞书 OpenAPI 原始消息规范化为有序文本和图片。"""

from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any, Protocol, TypeAlias

from ...images import MAX_IMAGE_BYTES, UNAVAILABLE_IMAGE, image_media_type, image_request_bytes, run_image_io, save_image
from ...protocol import OMITTED_INPUT_IMAGE, ChannelUser, Message, Part, Text
from ...speech.asr import AsrError, AudioFile, FileASR, MAX_AUDIO_BYTES
from ...workspace import Workspace
from .cli import CliError, parse_open_id


SUPPORTED_TYPES = {"audio", "image", "post", "text"}
_MAX_POST_NODES = 1000
_LOG = logging.getLogger(__name__)


class MessageFormatError(Exception):
    """无正文、未知结构或不支持的图片，错误不包含原始消息。"""


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    message_id: str
    created_ms: int
    kind: str
    content: str = field(repr=False)


def incoming(raw: Any, *, user: ChannelUser, account: ChannelUser) -> IncomingMessage | None:
    if not isinstance(raw, dict) or raw.get("deleted") is True:
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


def _text_bytes(value: str) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class _ImageResource:
    message_id: str = field(repr=False)
    key: str = field(repr=False)


_PostPart: TypeAlias = Text | _ImageResource
NormalizedContent: TypeAlias = str | _ImageResource | tuple[_PostPart, ...]


class ResourceDownloader(Protocol):
    async def download(self, message_id: str, key: str, kind: str, *, limit: int) -> bytes: ...


class MessageNormalizer:
    def __init__(self, cli: ResourceDownloader, asr: FileASR) -> None:
        self._cli = cli
        self._asr = asr

    async def normalize(self, raw: IncomingMessage) -> NormalizedContent:
        content = _read_content(raw.content)
        if raw.kind == "text":
            return _required(content.get("text"))
        if raw.kind == "image":
            return _ImageResource(raw.message_id, _required(content.get("image_key")))
        if raw.kind == "audio":
            data = await self._cli.download(raw.message_id, _required(content.get("file_key")), "file",
                                            limit=MAX_AUDIO_BYTES)
            return _required(await self._asr.transcribe(AudioFile(data)))
        return self._post(raw.message_id, _post_body(content))

    async def finish(self, items: list[NormalizedContent], workspace: Workspace, *, budget: int) -> list[Message]:
        # 先为整批文字与图片占位留空间。按原序下载和保存，一张成功才扣它的编码增量。
        placeholder_cost = _text_bytes(OMITTED_INPUT_IMAGE.text) + 32

        def body_bytes(content: NormalizedContent) -> int:
            if isinstance(content, str):
                return _text_bytes(content)
            if isinstance(content, _ImageResource):
                return placeholder_cost
            if len(content) == 1 and isinstance(content[0], Text):
                return _text_bytes(content[0].text)
            return sum(_text_bytes(part.text) + 32 if isinstance(part, Text) else placeholder_cost for part in content)

        remaining = budget - sum(64 + body_bytes(content) for content in items)
        final = []
        failed = False
        for content in items:
            if isinstance(content, str):
                final.append(Message("user", content))
                continue
            standalone = isinstance(content, _ImageResource)
            parts: list[Part] = []
            for part in (content,) if standalone else content:
                if isinstance(part, Text):
                    parts.append(part)
                    continue
                if image_request_bytes(0, "image/jpeg") - placeholder_cost >= remaining:
                    parts.append(OMITTED_INPUT_IMAGE)
                    continue
                try:
                    data = await self._cli.download(part.message_id, part.key, "image", limit=MAX_IMAGE_BYTES)
                    mime = image_media_type(data)
                    if mime is None:
                        raise MessageFormatError("unsupported_image_format")
                    cost = image_request_bytes(len(data), mime) - placeholder_cost
                    if cost > remaining:
                        parts.append(OMITTED_INPUT_IMAGE)
                        continue
                    image = await run_image_io(lambda: save_image(workspace, data), settle_write=True)
                except (MessageFormatError, CliError, TimeoutError, OSError):
                    failed = True
                    parts.append(UNAVAILABLE_IMAGE)
                else:
                    parts.append(image)
                    remaining -= cost
            if standalone and parts[0] is UNAVAILABLE_IMAGE:
                remaining += 64 + placeholder_cost  # 未交付的单图不保留占位预算。
            else:
                final.append(Message("user", tuple(parts)))
        if failed:
            _LOG.warning("Feishu input image could not be downloaded, validated or saved")
        return final

    def _post(self, message_id: str, post: dict[str, Any]) -> tuple[_PostPart, ...]:
        parts: list[_PostPart] = []
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
            line: list[_PostPart] = []
            for node in paragraph:
                if not isinstance(node, dict):
                    raise MessageFormatError("invalid_post_node")
                tag = node.get("tag")
                if tag == "img":
                    try:
                        key = _required(node.get("image_key"))
                    except MessageFormatError:
                        line.append(UNAVAILABLE_IMAGE)
                    else:
                        line.append(_ImageResource(message_id, key))
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
