"""POSIX 专用下载子入口：先设文件硬上限，再用原 argv 替换为 CLI。"""

import os
import signal
import sys


def main() -> int:
    if len(sys.argv) < 3:
        return 126
    try:
        import resource

        requested = int(sys.argv[1])
        command = sys.argv[2:]
        if requested <= 0 or not command:
            return 126
        existing = resource.getrlimit(resource.RLIMIT_FSIZE)
        limit = min([requested, *(value for value in existing if value != resource.RLIM_INFINITY)])
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
        os.execvp(command[0], command)
    except (ImportError, OSError, ValueError, OverflowError):
        # 不向调用方输出环境、路径、argv 或原始异常；未成功限制时禁止启动下载。
        return 126
    return 126


if __name__ == "__main__":
    raise SystemExit(main())
