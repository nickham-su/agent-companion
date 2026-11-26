"""lark-cli raw API、来电识别与资源下载，固定使用用户身份。"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import stat
import sys
from tempfile import TemporaryDirectory
from typing import Any

from ...protocol import ChannelUser, DeliveryStatus


_LOG = logging.getLogger(__name__)
_MESSAGES = "/open-apis/im/v1/messages"
MIN_PAGE_SIZE = 1
MAX_PAGE_SIZE = 50
DEFAULT_SEARCH_PAGE_SIZE = 20
MAX_MGET_SIZE = 50
_JSON_LIMIT = 4 * 1024 * 1024
_ERROR_LIMIT = 64 * 1024
_TEXT_LIMIT = 150_000
_POST_LIMIT = 30_000
_FILE_LIMIT_ENTRY = Path(__file__).with_name("_file_limit.py")
# 飞书登录会话由 CLI 自行管理；不把 Agent 的模型凭据交给 CLI 子进程。
_CLI_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "XDG_CONFIG_HOME",
            "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME")
# Generic API errors and policy checks that may run after the request cannot prove non-delivery.
_DEFINITE_ERRORS = {"validation", "config", "authentication", "authorization", "confirmation"}
_DIAGNOSTIC_ERRORS = _DEFINITE_ERRORS | {"api", "network", "internal", "rate_limit"}


class CliError(Exception):
    """只带诊断码和经过筛选的错误分类；不得附带子进程输入、输出或原始异常。"""

    def __init__(self, reason: str, *, definite: bool = False, category: str | None = None,
                 api_code: int | None = None) -> None:
        super().__init__(reason)
        self.definite = definite
        self.category = category
        self.api_code = api_code


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


def _json_bytes(value: dict, *, ensure_ascii: bool = True) -> bytes:
    return json.dumps(value, ensure_ascii=ensure_ascii, separators=(",", ":")).encode("utf-8")


def _check_exit(result: CommandResult) -> None:
    if result.returncode == 0:
        return
    envelope = as_object(result.stderr)
    error = envelope.get("error")
    category = None
    api_code = None
    if envelope.get("ok") is False and isinstance(error, dict):
        kind = error.get("type")
        if isinstance(kind, str) and kind in _DIAGNOSTIC_ERRORS:
            category = kind
            code = error.get("code")
            if type(code) is int and 0 < code <= 99_999_999:
                api_code = code
    definite = category in _DEFINITE_ERRORS
    raise CliError("cli_rejected" if definite else "cli_result_unknown", definite=definite,
                   category=category, api_code=api_code)


def _success_data(result: CommandResult) -> dict[str, Any]:
    envelope = as_object(result.stdout)
    data = envelope.get("data")
    if envelope.get("ok") is not True or envelope.get("identity") != "user" or not isinstance(data, dict):
        raise CliError("cli_invalid_response")
    return data


class _UserCLI:
    def __init__(self, executable: str, *, timeout: float, runner: CommandRunner = run_command) -> None:
        self._executable = executable
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


@dataclass(frozen=True, slots=True, repr=False)
class RingingCaller:
    meeting_id: str
    user: ChannelUser


def _valid_id(value: Any, prefix: str) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"{prefix}_[\w-]{{1,160}}", value, flags=re.ASCII) is not None


def parse_open_id(value: Any) -> ChannelUser | None:
    """在飞书协议边界将合法的 open_id 转为通道用户。"""
    return ChannelUser(value) if _valid_id(value, "ou") else None


class FeishuDiscovery(_UserCLI):
    async def account_user(self) -> ChannelUser:
        data = _success_data(await self._run(("api", "GET", "/open-apis/authen/v1/user_info")))
        user = parse_open_id(data.get("open_id"))
        if user is None:
            raise CliError("cli_invalid_account")
        return user

    async def search_page(self, start: int, end: int, token: str | None, *,
                          page_size: int = DEFAULT_SEARCH_PAGE_SIZE) -> tuple[list, str | None]:
        if type(page_size) is not int or not MIN_PAGE_SIZE <= page_size <= MAX_PAGE_SIZE:
            raise ValueError("invalid Feishu search page size")
        params = {"page_size": page_size, "user_id_type": "open_id"}
        if token is not None:
            params["page_token"] = token
        payload = {"query": "", "filter": {"chat_type": "p2p", "from_types": ["user"], "time_range": {
            "start_time": datetime.fromtimestamp(start, timezone.utc).isoformat(),
            "end_time": datetime.fromtimestamp(end, timezone.utc).isoformat(),
        }}}
        result = await self._run(("api", "POST", _MESSAGES + "/search", "--params", _json_bytes(params).decode(),
                                  "--data", "-"), data=_json_bytes(payload))
        data = _success_data(result)
        items, more, next_token = data.get("items"), data.get("has_more"), data.get("page_token")
        # 飞书的空结果可以是 null 或省略 items；只有明确完成的页才接受这种形态。
        if items is None and more is False:
            items = []
        if not isinstance(items, list) or type(more) is not bool or (more and (
            not isinstance(next_token, str) or not 0 < len(next_token) <= 2048 or "\x00" in next_token
        )):
            raise CliError("cli_invalid_search_page")
        return items, next_token if more else None

    async def messages(self, message_ids: tuple[str, ...]) -> list:
        if not 0 < len(message_ids) <= MAX_MGET_SIZE or not all(_valid_id(mid, "om") for mid in message_ids):
            raise ValueError("invalid Feishu message batch")
        # raw CLI 将 JSON 数组编码为重复的 message_ids 查询参数；路径本身不能带查询串。
        params = _json_bytes({"message_ids": list(message_ids), "user_id_type": "open_id"}).decode()
        data = _success_data(await self._run(("api", "GET", _MESSAGES + "/mget", "--params", params)))
        items = data.get("items")
        # 未确认 mget 契约允许 null / 省略 items；不能把无法核验的整批响应当作合法缺项。
        if not isinstance(items, list):
            raise CliError("cli_invalid_message_details")
        return items

    async def _ringing_query(self, step: str, args: tuple[str, ...]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        outcome = "failed"
        try:
            data = _success_data(await self._run(args))
            outcome = "ok"
            return data
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            # 只记录固定步骤和耗时，不记录参数或响应（含会议和参与者身份）。
            _LOG.debug("Feishu ringing CLI query (step=%s, outcome=%s, elapsed=%.3fs)",
                       step, outcome, loop.time() - started_at)

    async def ringing_caller(self, account: ChannelUser) -> RingingCaller | None:
        """只接受当前实测的一对一响铃形态，不将其当作会议来源的证明。"""
        data = await self._ringing_query("list_active", ("vc", "+meeting-list-active"))
        meetings = data.get("meetings")
        if meetings is None and "meetings" in data:
            return None
        if not isinstance(meetings, list):
            raise CliError("cli_invalid_active_meetings")
        if len(meetings) != 1:
            return None
        row = meetings[0]
        meeting_id = row.get("meeting_id") if isinstance(row, dict) else None
        if not isinstance(meeting_id, str) or re.fullmatch(r"[0-9]{1,32}", meeting_id) is None:
            raise CliError("cli_invalid_meeting_id")
        data = await self._ringing_query("meeting_detail", ("vc", "meeting", "get", "--meeting-id", meeting_id,
                                                            "--with-participants", "--user-id-type", "open_id"))
        meeting = data.get("meeting")
        if not isinstance(meeting, dict) or meeting.get("id") != meeting_id:
            raise CliError("cli_invalid_meeting_detail")
        status, host, participants = meeting.get("status"), meeting.get("host_user"), meeting.get("participants")
        if type(status) is not int or status != 1 or not isinstance(participants, list) or len(participants) != 1:
            return None
        caller = parse_open_id(host.get("id")) if isinstance(host, dict) else None
        participant = participants[0]
        if (caller is None or caller == account
                or not isinstance(participant, dict) or parse_open_id(participant.get("id")) != caller):
            return None
        return RingingCaller(meeting_id, caller)

    def for_user(self, user: ChannelUser) -> "FeishuCLI":
        return FeishuCLI(self._executable, user, timeout=self._timeout, runner=self._runner)


class FeishuCLI(_UserCLI):
    def __init__(self, executable: str, user: ChannelUser, *, timeout: float,
                 runner: CommandRunner = run_command) -> None:
        super().__init__(executable, timeout=timeout, runner=runner)
        if parse_open_id(user.key) is None:
            raise ValueError("invalid Feishu recipient")
        self._open_id = user.key

    async def send_text(self, text: str) -> DeliveryStatus:
        content = json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":"))
        try:
            oversized = len(content.encode("utf-8")) > _TEXT_LIMIT
        except UnicodeError:
            return DeliveryStatus.FAILED
        if oversized:
            return DeliveryStatus.FAILED
        payload = _json_bytes({"receive_id": self._open_id, "msg_type": "text", "content": content})
        return await self._send_message(payload)

    def prepare_transcript(self, text: str) -> bytes | None:
        """将正文作为字面富文本编码，并按最终发送的请求大小校验。"""
        if len(text) > _POST_LIMIT:
            return None
        content = json.dumps({"zh_cn": {"content": [[{"tag": "text", "text": text, "style": ["bold"]}]]}},
                             ensure_ascii=False, separators=(",", ":"))
        try:
            payload = _json_bytes({"receive_id": self._open_id, "msg_type": "post", "content": content},
                                  ensure_ascii=False)
        except UnicodeError:
            return None
        return payload if len(payload) <= _POST_LIMIT else None

    async def send_prepared_transcript(self, payload: bytes) -> DeliveryStatus:
        """直接发送已校验的回显请求，不重复编码。"""
        return await self._send_message(payload)

    async def _send_message(self, payload: bytes) -> DeliveryStatus:
        try:
            result = await self._run(("api", "POST", _MESSAGES, "--params", '{"receive_id_type":"open_id"}',
                                      "--data", "-"), data=payload)
            data = _success_data(result)
            message_id = data.get("message_id")
            if not _valid_id(message_id, "om") or ("chat_id" in data and not _valid_id(data["chat_id"], "oc")):
                return DeliveryStatus.UNKNOWN
        except CliError as exc:
            return DeliveryStatus.FAILED if exc.definite else DeliveryStatus.UNKNOWN
        return DeliveryStatus.FULL
