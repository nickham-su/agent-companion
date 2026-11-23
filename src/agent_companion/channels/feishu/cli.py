"""lark-cli v1.0.100 的 raw API 与资源下载协议，固定使用用户身份。"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import stat
import sys
from tempfile import TemporaryDirectory
from typing import Any

from ...protocol import DeliveryStatus


_MESSAGES = "/open-apis/im/v1/messages"
MIN_PAGE_SIZE = 1
MAX_PAGE_SIZE = 50
_JSON_LIMIT = 4 * 1024 * 1024
_ERROR_LIMIT = 64 * 1024
_TEXT_LIMIT = 150_000
_FILE_LIMIT_ENTRY = Path(__file__).with_name("_file_limit.py")
# 飞书登录会话由 CLI 自行管理；不把 Agent 的模型凭据交给 CLI 子进程。
_CLI_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "XDG_CONFIG_HOME",
            "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME")
# Generic API errors and policy checks that may run after the request cannot prove non-delivery.
_DEFINITE_ERRORS = {"validation", "config", "authentication", "authorization", "confirmation"}


class CliError(Exception):
    """只带由本模块定义的诊断码；不得附带子进程输入、输出或原始异常。"""

    def __init__(self, reason: str, *, definite: bool = False) -> None:
        super().__init__(reason)
        self.definite = definite


class CliStdoutLimit(CliError):
    """CLI 标准输出超过上限；分页调用可改用更小的页。"""


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: bytes = field(repr=False)
    stderr: bytes = field(repr=False)


CommandRunner = Callable[[tuple[str, ...], bytes | None, float], Awaitable[CommandResult]]


def as_object(raw: bytes) -> dict[str, Any]:
    try:
        result = json.loads(raw)
    except (ValueError, RecursionError):
        return {}
    return result if isinstance(result, dict) else {}


class _CommandProtocol(asyncio.SubprocessProtocol):
    """Bound output without letting full or inherited pipes hold up cancellation."""

    def __init__(self, data: bytes | None) -> None:
        self.data = data
        self.done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.failure: CliError | None = None
        self.transport: asyncio.SubprocessTransport | None = None

    def connection_made(self, transport: asyncio.SubprocessTransport) -> None:
        self.transport = transport
        if self.data is not None:
            pipe = transport.get_pipe_transport(0)
            pipe.write(self.data)
            pipe.close()  # Flush the bounded request to stdin, then signal EOF.

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        if self.failure is not None:
            return
        buffer, limit = (self.stdout, _JSON_LIMIT) if fd == 1 else (self.stderr, _ERROR_LIMIT)
        if len(buffer) + len(data) > limit:
            self.failure = (CliStdoutLimit if fd == 1 else CliError)("cli_output_limit")
            self.transport.close()
        else:
            buffer.extend(data)

    def pipe_connection_lost(self, fd: int, exc: Exception | None) -> None:
        # A CLI that rejected the request may close stdin early; use its typed stderr.
        if fd != 0 and exc is not None and self.failure is None:
            self.failure = CliError("cli_io_failed")
            self.transport.close()

    def connection_lost(self, exc: Exception | None) -> None:
        if exc is not None and self.failure is None:
            self.failure = CliError("cli_io_failed")
        if not self.done.done():
            self.done.set_result(None)


async def run_command(argv: tuple[str, ...], data: bytes | None, timeout: float) -> CommandResult:
    env = {name: os.environ[name] for name in _CLI_ENV if name in os.environ}
    protocol = _CommandProtocol(data)
    transport = None
    try:
        async with asyncio.timeout(timeout):
            transport, _ = await asyncio.get_running_loop().subprocess_exec(
                lambda: protocol, *argv, env=env,
                stdin=asyncio.subprocess.PIPE if data is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await asyncio.shield(protocol.done)
            if protocol.failure is not None:
                raise protocol.failure
            returncode = transport.get_returncode()
            if returncode is None:
                raise CliError("cli_io_failed")
            return CommandResult(returncode, bytes(protocol.stdout), bytes(protocol.stderr))
    except TimeoutError:
        raise CliError("cli_timeout") from None
    except (OSError, ValueError):
        raise CliError("cli_could_not_start" if transport is None else "cli_io_failed",
                       definite=transport is None) from None
    finally:
        if transport is not None:
            transport.close()  # Closes all three pipes and kills an in-flight CLI.
            # 下载的调用方随即清理目录；必须先收回进程，不能让它继续写文件。
            if not protocol.done.done():
                await protocol.done


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def _check_exit(result: CommandResult) -> None:
    if result.returncode == 0:
        return
    envelope = as_object(result.stderr)
    error = envelope.get("error")
    definite = False
    if envelope.get("ok") is False and isinstance(error, dict):
        category = error.get("type")
        definite = isinstance(category, str) and category in _DEFINITE_ERRORS
    raise CliError("cli_rejected" if definite else "cli_result_unknown", definite=definite)


def _success_data(result: CommandResult) -> dict[str, Any]:
    envelope = as_object(result.stdout)
    data = envelope.get("data")
    if envelope.get("ok") is not True or envelope.get("identity") != "user" or not isinstance(data, dict):
        raise CliError("cli_invalid_response")
    return data


class FeishuCLI:
    def __init__(self, executable: str, chat_id: str, *, timeout: float,
                 runner: CommandRunner = run_command) -> None:
        self._executable = executable
        self._chat_id = chat_id
        self._timeout = timeout
        self._runner = runner

    async def _run(self, args: tuple[str, ...], *, data: bytes | None = None,
                   file_limit: int | None = None) -> CommandResult:
        argv = (self._executable, *args, "--as", "user", "--format", "json")
        if file_limit is not None:
            argv = (sys.executable, "-I", "-S", str(_FILE_LIMIT_ENTRY), str(file_limit), *argv)
        result = await self._runner(argv, data, self._timeout)
        _check_exit(result)
        return result

    async def list_page(self, start: int, end: int, token: str | None, *,
                        page_size: int = MAX_PAGE_SIZE) -> tuple[list, str | None]:
        if type(page_size) is not int or not MIN_PAGE_SIZE <= page_size <= MAX_PAGE_SIZE:
            raise ValueError("invalid Feishu page size")
        params = {"container_id_type": "chat", "container_id": self._chat_id,
                  "start_time": str(start), "end_time": str(end), "sort_type": "ByCreateTimeAsc",
                  "page_size": page_size}
        if token is not None:
            params["page_token"] = token
        data = _success_data(await self._run(("api", "GET", _MESSAGES, "--params", _json_bytes(params).decode())))
        items = data.get("items", [])
        has_more = data.get("has_more")
        page_token = data.get("page_token")
        if not isinstance(items, list) or type(has_more) is not bool or (has_more and (
            not isinstance(page_token, str) or not page_token or len(page_token) > 2048
        )):
            raise CliError("cli_invalid_page")
        return items, page_token if has_more else None

    async def download(self, message_id: str, key: str, kind: str, *, limit: int) -> bytes:
        prefix = {"image": "img", "file": "file"}.get(kind)
        if prefix is None or not re.fullmatch(r"om_[\w-]{1,160}", message_id, flags=re.ASCII) or not re.fullmatch(
            rf"{prefix}_[\w-]{{1,160}}", key, flags=re.ASCII
        ):
            raise CliError("invalid_resource_reference", definite=True)
        try:
            with TemporaryDirectory(prefix=".feishu-", dir=Path.cwd()) as tmp:
                # 指定扩展名，使该命令不会采用远端提供的文件名；不信任 CLI 返回的路径。
                target = Path(tmp) / "media.bin"
                result = await self._run(("im", "+messages-resources-download", "--message-id", message_id,
                                          "--file-key", key, "--type", kind, "--output", str(target)), file_limit=limit)
                reported_size = _success_data(result).get("size_bytes")
                info = target.lstat()
                if (not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit
                        or type(reported_size) is not int or reported_size != info.st_size):
                    raise CliError("invalid_resource_size", definite=True)
                with target.open("rb") as file:
                    data = file.read(limit + 1)
                if len(data) != reported_size:
                    raise CliError("invalid_resource_size", definite=True)
                return data
        except OSError:
            raise CliError("resource_file_unavailable", definite=True) from None

    async def send_text(self, text: str) -> DeliveryStatus:
        content = json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":"))
        try:
            oversized = len(content.encode("utf-8")) > _TEXT_LIMIT
        except UnicodeError:
            return DeliveryStatus.FAILED
        if oversized:
            return DeliveryStatus.FAILED
        payload = _json_bytes({"receive_id": self._chat_id, "msg_type": "text", "content": content})
        try:
            result = await self._run(("api", "POST", _MESSAGES, "--params", '{"receive_id_type":"chat_id"}',
                                      "--data", "-"), data=payload)
            data = _success_data(result)
            message_id = data.get("message_id")
            if (not isinstance(message_id, str) or not re.fullmatch(r"om_[\w-]{1,160}", message_id, flags=re.ASCII)
                    or data.get("chat_id", self._chat_id) != self._chat_id):
                return DeliveryStatus.UNKNOWN
        except CliError as exc:
            return DeliveryStatus.FAILED if exc.definite else DeliveryStatus.UNKNOWN
        return DeliveryStatus.FULL
