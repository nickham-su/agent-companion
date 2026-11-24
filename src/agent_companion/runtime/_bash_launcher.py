"""在已受控的子进程中准备工作区，然后以同一 PID 执行 Bash。"""

import os
from pathlib import Path
import signal
import sys


READY = b"R"
WORKSPACE_ERROR = b"W"
EXEC_ERROR = b"E"


def main(status_fd: int, workspace: str, command: str) -> int:
    # -I -S 不读取用户路径；只从本助手所在的可信安装目录加载共用逻辑。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from agent_companion.workspace import Workspace

    os.set_inheritable(status_fd, False)  # Bash 命令拿不到启动诊断管道。
    try:
        directory = Workspace(Path(workspace)).ensure_directory()
        os.chdir(directory)
    except (OSError, ValueError):
        os.write(status_fd, WORKSPACE_ERROR)
        return 1

    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(directory), "LANG": "C"}
    try:
        # Python 忽略这些信号；恢复为直接由 Popen 启动 Bash 时的默认行为。
        for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ"):
            if (sig := getattr(signal, name, None)) is not None:
                signal.signal(sig, signal.SIG_DFL)
        os.write(status_fd, READY)
        os.execve("/bin/bash", ("/bin/bash", "--noprofile", "--norc", "-c", command), env)
    except BaseException:
        # READY 只意味着准备完成。exec 失败必须覆盖它，不能把助手异常当作命令输出。
        os.write(status_fd, EXEC_ERROR)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]), sys.argv[2], sys.argv[3]))
