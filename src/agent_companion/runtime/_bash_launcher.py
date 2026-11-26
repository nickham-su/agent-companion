"""在已受控的子进程中准备工作区，然后以同一 PID 执行 Bash。"""

from errno import ELOOP, ENOTDIR
import fcntl
import os
from pathlib import Path
import signal
import stat
import sys


READY = b"R"
WORKSPACE_ERROR = b"W"
EXEC_ERROR = b"E"
RC_PENDING = b"B"
RC_UNREADABLE = b"U"
RC_SYMLINK = b"L"
RC_FDS_UNAVAILABLE = b"Q"
RC_FAILED = b"F"
RC_LOADED = b"D"
RC_RUN = b"G"
RC_HANDOFF = b"H"
_MIN_SHELL_FD = 64


class _RcSymlink(OSError):
    pass


def _open_rc() -> int | None:
    """仅加载工作区里的普通文件；FIFO 等特殊文件不能阻塞启动。"""
    from agent_companion.workspace import open_child_directory

    try:
        agent_fd = open_child_directory(None, ".agent", create=False, search_only=True)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in (ENOTDIR, ELOOP):
            try:
                mode = os.stat(".agent", follow_symlinks=False).st_mode
            except FileNotFoundError:
                return None
            if stat.S_ISREG(mode):
                return None  # 普通文件没有 bashrc 子路径。
            if stat.S_ISLNK(mode):
                raise _RcSymlink() from None
        raise
    try:
        rc_fd = os.open("bashrc", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=agent_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == ELOOP:
            raise _RcSymlink() from None
        raise
    finally:
        os.close(agent_fd)
    try:
        if not stat.S_ISREG(os.fstat(rc_fd).st_mode):
            raise OSError("not a regular file")
    except BaseException:
        os.close(rc_fd)
        raise
    return rc_fd


def _reserve_shell_fd(fd: int) -> int:
    # 避免占用脚本常用的数字 FD，不因低 FD 限额回退到这些编号。
    if fd >= _MIN_SHELL_FD:
        return fd
    reserved = fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, _MIN_SHELL_FD)
    os.close(fd)
    return reserved


def _rc_command(status_fd: int, rc_fd: int) -> str:
    # 整块先解析，不受 bashrc 新定义的别名影响。不要在 if 条件内 source，
    # 否则 Bash 会忽略脚本自己的 errexit。保留控制 FD，避免脚本的动态 FD 分配占用它。
    return f"""{{
    __agent_bash_command=$1
    shift
    builtin source /dev/fd/{rc_fd}
    __agent_bash_rc_status=$?
    command exec {rc_fd}<&-
    if (( __agent_bash_rc_status != 0 )); then
        builtin printf {RC_FAILED.decode()} >&{status_fd}
        builtin exit "$__agent_bash_rc_status"
    fi
    builtin unset __agent_bash_rc_status
    builtin printf {RC_LOADED.decode()} >&{status_fd}
    if ! builtin read -r -d {RC_RUN.decode()} __agent_bash_rc_go <&{status_fd}; then
        builtin exit 1
    fi
    builtin unset __agent_bash_rc_go
    builtin printf {RC_HANDOFF.decode()} >&{status_fd} || builtin exit 1
    command exec {status_fd}>&- || builtin exit 1
    builtin eval $'builtin unset __agent_bash_command\\n'"$__agent_bash_command"
}}"""


def main(status_fd: int, workspace: str, command: str) -> int:
    # -I -S 不读取用户路径；只从本助手所在的可信安装目录加载共用逻辑。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from agent_companion.workspace import Workspace

    os.set_inheritable(status_fd, False)
    try:
        with Workspace(Path(workspace)).open_directory(search_only=True) as (_, workspace_fd):
            os.fchdir(workspace_fd)
            # 如果目录刚被重命名，HOME 跟随已打开的目录，不重新使用旧名称。
            home = os.getcwd()
    except (OSError, ValueError):
        os.write(status_fd, WORKSPACE_ERROR)
        return 1

    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": home, "LANG": "C"}
    rc_fd = None
    try:
        # Python 忽略这些信号；恢复为直接由 Popen 启动 Bash 时的默认行为。
        for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ"):
            if (sig := getattr(signal, name, None)) is not None:
                signal.signal(sig, signal.SIG_DFL)
        try:
            rc_fd = _open_rc()
        except _RcSymlink:
            os.write(status_fd, RC_SYMLINK)
            return 1
        except OSError:
            os.write(status_fd, RC_UNREADABLE)
            return 1
        argv = ("/bin/bash", "--noprofile", "--norc", "-c")
        if rc_fd is None:
            os.write(status_fd, READY)
            os.execve(argv[0], (*argv, command), env)
        else:
            # 路径检查与 source 使用同一个已打开的文件，避免换成链接或 FIFO。
            try:
                rc_fd = _reserve_shell_fd(rc_fd)
                status_fd = _reserve_shell_fd(status_fd)
                os.set_inheritable(rc_fd, True)
                os.set_inheritable(status_fd, True)
            except OSError:
                os.write(status_fd, RC_FDS_UNAVAILABLE)
                return 1
            os.write(status_fd, READY + RC_PENDING)
            os.execve(argv[0], (*argv, _rc_command(status_fd, rc_fd), argv[0], command), env)
    except BaseException:
        # READY 只意味着准备完成。exec 失败必须覆盖它，不能把助手异常当作命令输出。
        os.write(status_fd, EXEC_ERROR)
        return 1
    finally:
        if rc_fd is not None:
            os.close(rc_fd)


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]), sys.argv[2], sys.argv[3]))
