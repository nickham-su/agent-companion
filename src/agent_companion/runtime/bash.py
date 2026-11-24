"""在用户工作区中运行 Bash，按 Agent 树串行执行。"""

import asyncio
from collections.abc import Coroutine
from errno import ECHILD, ESRCH
import fcntl
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

from ..protocol import ToolSpec
from ._bash_launcher import READY, WORKSPACE_ERROR
from ._child_exit import child_exited
from .tools import Tool, ToolContext


BASH_TOOL_NAME = "bash"
_LOG = logging.getLogger(__name__)
_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 120
_MAX_COMMAND_BYTES = 32 * 1024
_MAX_OUTPUT_BYTES = 32 * 1024  # stdout / stderr 合计，按原始字节计算。
_PIPE_CLOSE_GRACE = 0.25
_CHILD_POLL_INTERVAL = 0.01
_GROUP_CLOSE_TIMEOUT = 1.0
_GROUP_POLL_INTERVAL = 0.05
_PROBE_TIMEOUT = 0.25
_PROBE_MAX_OUTPUT = 128 * 1024
_LINUX_PROBE = Path(__file__).with_name("_proc_group_probe.py").resolve()
_BASH_LAUNCHER = Path(__file__).with_name("_bash_launcher.py").resolve()
_CLEANUP_ERROR = '{"error":"bash_cleanup_unconfirmed"}'
_CLOSED_ERROR = '{"error":"bash_closed"}'
# 清理不确定时，退役再绑定也不绕过；不对旧 PGID 重新发信号。
_BLOCKED_WORKSPACES: set[Path] = set()
_UNREAPED_CHILDREN: set[subprocess.Popen[bytes]] = set()
_SPEC = ToolSpec(
    BASH_TOOL_NAME,
    "在用户工作目录中运行 Bash 并等待终态。每次从同一目录开始；目录不是沙箱。",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "非空的 Bash 命令"},
            "timeout": {"type": "integer", "minimum": 1, "maximum": _MAX_TIMEOUT,
                        "description": "执行超时秒数，默认 30"},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
)


class Bash:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._closed = False
        self.tool = Tool(_SPEC, self._execute)

    def stop_accepting(self) -> None:
        """同步禁止新启动，不在此处取消已启动的工具。"""
        self._closed = True

    async def _execute(self, args: dict[str, Any], context: ToolContext) -> str:
        command, timeout = args.get("command"), args.get("timeout", _DEFAULT_TIMEOUT)
        if (args.keys() - {"command", "timeout"} or not isinstance(command, str) or not command.strip()
                or "\x00" in command or type(timeout) is not int or not 1 <= timeout <= _MAX_TIMEOUT):
            return '{"error":"invalid_arguments"}'
        try:
            if len(command.encode("utf-8")) > _MAX_COMMAND_BYTES:
                return '{"error":"invalid_arguments"}'
        except UnicodeEncodeError:
            return '{"error":"invalid_arguments"}'
        if context.workspace is None:
            return '{"error":"workspace_unavailable"}'
        if self._closed:
            return _CLOSED_ERROR
        async with self._lock:
            if self._closed:
                return _CLOSED_ERROR  # 等锁期间可能已经撤销路由。
            workspace_key = context.workspace.directory
            if workspace_key in _BLOCKED_WORKSPACES:
                return _CLEANUP_ERROR
            return await _run_command(command, workspace_key, timeout)


class _Child:
    """始终先持有 PID 和两路管道，再配置非阻塞 I/O。"""

    def __init__(self, argv: tuple[str, ...], *, env: dict[str, str], capture_stderr: bool,
                 output_limit: int, pass_fds: tuple[int, ...] = ()) -> None:
        for previous in tuple(_UNREAPED_CHILDREN):
            # 仅回收已结束信号阶段的句柄；之后不会再用它们的 PGID。
            if previous.poll() is not None:
                _UNREAPED_CHILDREN.discard(previous)
        self.stdout, self.stderr = bytearray(), bytearray()
        self.output_limit = output_limit
        self.output_limit_exceeded = self.timed_out = self.io_failed = False
        self.pipes_drained = False
        self._stopped = False
        self._ownership_lost = False
        self._group_verified = False
        self.process = subprocess.Popen(
            # 不设置 cwd：用户工作区的 chdir 不能出现在 Popen 的 fork/exec 等待内。
            argv, env=env, start_new_session=True, bufsize=0, pass_fds=pass_fds,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
        )

    def stop(self, *, verified: bool = False) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._ownership_lost:
            return
        if not verified:
            try:
                self._exited()  # 超时、取消或观察异常后，先无回收地核验组长仍归我们。
            except OSError:
                self.io_failed = True
                return
        self._group_verified = True
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            self.io_failed = True
            try:
                os.kill(self.process.pid, signal.SIGKILL)
            except OSError:
                pass  # 仍在有界等待中做回收确认。

    def _lose_ownership(self) -> None:
        self._ownership_lost = True
        if self.process.returncode is None:
            # 没有可取得的退出码。终结 Popen 句柄，避免析构时再次 waitpid；不对外返回假退出码。
            self.process.returncode = 255

    def _exited(self) -> bool:
        if self.process.returncode is not None:
            self._lose_ownership()
            raise ChildProcessError(ECHILD, "child already reaped")
        try:
            return child_exited(self.process.pid)
        except OSError as exc:
            if exc.errno in (ECHILD, ESRCH):
                self._lose_ownership()
            raise

    async def collect(self, deadline: float) -> None:
        pipes = [(stream.fileno(), output) for stream, output in
                 ((self.process.stdout, self.stdout), (self.process.stderr, self.stderr)) if stream is not None]
        for fd, _ in pipes:
            os.set_blocking(fd, False)
        clock = asyncio.get_running_loop().time
        pipe_deadline = None
        while True:
            now = clock()
            if pipe_deadline is None:
                exited = now < deadline and self._exited()
                now = clock()
                if now >= deadline:
                    self.timed_out = True  # 未在截止前观察到退出，晚到的退出/EOF 不算成功。
                    return
                if exited:
                    self.stop(verified=True)  # 不回收组长，先终止仍持有管道的同组进程。
                    pipe_deadline = now + _PIPE_CLOSE_GRACE
            elif now >= pipe_deadline:
                return

            for fd, output in tuple(pipes):
                room = self.output_limit - len(self.stdout) - len(self.stderr)
                try:
                    data = os.read(fd, room + 1)
                except BlockingIOError:
                    continue
                if not data:
                    pipes.remove((fd, output))
                    continue
                output.extend(data[:room])
                if len(data) > room:
                    if pipe_deadline is None and clock() >= deadline:
                        self.timed_out = True
                    else:
                        self.output_limit_exceeded = True
                    return
            self.pipes_drained = not pipes
            if pipe_deadline is not None and self.pipes_drained:
                return
            wake = deadline if pipe_deadline is None else pipe_deadline
            await asyncio.sleep(max(0, min(_CHILD_POLL_INTERVAL, wake - clock())))

    async def reap(self) -> bool:
        self.stop()
        for stream in (self.process.stdout, self.process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    self.io_failed = True
        if self._ownership_lost:
            return False
        clock = asyncio.get_running_loop().time
        deadline = clock() + _GROUP_CLOSE_TIMEOUT
        while True:
            # 仅在不会再发送组信号后回收。waitpid 保留 ECHILD，不把丢失句柄解释成退出码 0。
            try:
                pid, status = os.waitpid(self.process.pid, os.WNOHANG)
            except OSError as exc:
                if exc.errno in (ECHILD, ESRCH):
                    self._lose_ownership()
                else:
                    _UNREAPED_CHILDREN.add(self.process)
                return False
            if pid:
                self.process.returncode = os.waitstatus_to_exitcode(status)
                return self._group_verified
            remaining = deadline - clock()
            if remaining <= 0:
                # 内核不回应时拒绝判定成功，保留句柄供后续子进程创建时再次回收。
                _UNREAPED_CHILDREN.add(self.process)
                return False
            await asyncio.sleep(min(_CHILD_POLL_INTERVAL, remaining))


async def _finish(cleanup: Coroutine[Any, Any, bool]) -> bool:
    task = asyncio.create_task(cleanup)
    interrupted = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            # Runtime 可重复取消；收尾的所有权不可随调用方取消丢失。
            interrupted = exc
    result = task.result()
    if interrupted is not None:
        raise interrupted
    return result


def _probe_command(group: int) -> tuple[str, ...] | None:
    if sys.platform == "linux":
        return sys.executable, "-I", "-S", str(_LINUX_PROBE), str(group)
    if sys.platform == "darwin":
        return "/bin/ps", "-A", "-o", "pgid=", "-o", "stat="
    return None


def _macos_live_members(group: int, snapshot: bytes) -> bool | None:
    found_zombie = False
    for row in snapshot.splitlines():
        if not row.strip():
            continue
        pgid, state = row.split()
        if int(pgid) == group:
            if not state.startswith(b"Z"):
                return True
            found_zombie = True
    return False if found_zombie else None


async def _inspect_group(group: int, deadline: float) -> tuple[bool, bool | None]:
    """返回探测器是否已收尾、以及目标组是否仍有活进程（未知为 None）。"""
    argv = _probe_command(group)
    if argv is None:
        return True, None
    try:
        child = _Child(argv, env={"LC_ALL": "C"}, capture_stderr=False, output_limit=_PROBE_MAX_OUTPUT)
    except (OSError, ValueError):
        return True, None
    try:
        await child.collect(deadline)
    except OSError:
        child.io_failed = True
    finally:
        reaped = await _finish(child.reap())
    if (not reaped or child.io_failed or child.timed_out or child.output_limit_exceeded
            or not child.pipes_drained or child.process.returncode != 0):
        return reaped, None
    if sys.platform == "linux":
        return True, {b"live": True, b"zombies": False}.get(bytes(child.stdout).strip())
    try:
        return True, _macos_live_members(group, bytes(child.stdout))
    except ValueError:
        return True, None


async def _confirm_group_exit(group: int) -> bool:
    """组消失或只剩不能运行的僵尸；不可观察的组不视为已停止。"""
    clock = asyncio.get_running_loop().time
    deadline = clock() + _GROUP_CLOSE_TIMEOUT
    while (now := clock()) < deadline:
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        reaped, live = await _inspect_group(group, min(deadline, now + _PROBE_TIMEOUT))
        if not reaped:
            return False
        if live is False:
            return True
        remaining = deadline - clock()
        if remaining > 0:
            await asyncio.sleep(min(_GROUP_POLL_INTERVAL, remaining))
    return False


async def _finish_command(child: _Child, workspace_key: Path) -> bool:
    if await child.reap() and await _confirm_group_exit(child.process.pid):
        return True
    _BLOCKED_WORKSPACES.add(workspace_key)
    _LOG.warning("bash process group cleanup unconfirmed; workspace Bash disabled until restart")
    return False


def _open_launch_status() -> tuple[int, int]:
    status_read, status_write = os.pipe()
    if status_write >= 3:
        return status_read, status_write

    # 服务可能没有完整的标准描述符。只搬移要传给助手的写端，避免被 Popen 重定向覆盖。
    moved_write = None
    try:
        try:
            moved_write = fcntl.fcntl(status_write, fcntl.F_DUPFD_CLOEXEC, 3)
        finally:
            os.close(status_write)
    except BaseException:
        try:
            os.close(status_read)
        finally:
            if moved_write is not None:
                os.close(moved_write)
        raise
    return status_read, moved_write


def _read_launch_status(fd: int) -> bytes | None:
    # 助手至多写两个字节，正常 exec 或退出会关闭写端。没有 EOF 也视为异常。
    os.set_blocking(fd, False)
    status = b""
    try:
        while len(status) < 3:
            data = os.read(fd, 3 - len(status))
            if not data:
                return status
            status += data
    except BlockingIOError:
        pass
    return None


async def _run_command(command: str, workspace_key: Path, timeout: int) -> str:
    deadline = asyncio.get_running_loop().time() + timeout
    # 私有管道区分准备失败和 Bash 自身的非零退出；不占用 stdout / stderr。
    try:
        status_read, status_write = _open_launch_status()
    except OSError:
        return '{"error":"bash_start_failed"}'
    try:
        child = None
        try:
            try:
                argv = (sys.executable, "-I", "-S", "-B", str(_BASH_LAUNCHER),
                        str(status_write), str(workspace_key), command)
                child = _Child(argv, env={"LC_ALL": "C"}, capture_stderr=True, output_limit=_MAX_OUTPUT_BYTES,
                               pass_fds=(status_write,))
            finally:
                os.close(status_write)
            await child.collect(deadline)
        except OSError:
            if child is None:
                return '{"error":"bash_start_failed"}'
            child.io_failed = True
        except ValueError:
            if child is None:
                return '{"error":"bash_start_failed"}'
            raise
        finally:
            if child is not None:
                confirmed = await _finish(_finish_command(child, workspace_key))
        if not confirmed:
            return _CLEANUP_ERROR
        if child.io_failed:
            return '{"error":"bash_io_failed"}'
        status = _read_launch_status(status_read)
        if not child.timed_out:
            if status == WORKSPACE_ERROR:
                return '{"error":"workspace_unavailable"}'
            if status != READY:
                return '{"error":"bash_start_failed"}'
    except OSError:
        return '{"error":"bash_io_failed"}'
    finally:
        os.close(status_read)
    return json.dumps({
        "exit_code": child.process.returncode,
        "stdout": child.stdout.decode("utf-8", errors="replace") if status == READY else "",
        "stderr": child.stderr.decode("utf-8", errors="replace") if status == READY else "",
        "timed_out": child.timed_out,
        "output_limit_exceeded": child.output_limit_exceeded,
    })
