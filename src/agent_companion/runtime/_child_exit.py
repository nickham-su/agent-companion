"""只观察直属子进程退出；调用者发完进程组信号之前不得回收组长。"""

import ctypes
from errno import EIO, ENOSYS
from functools import lru_cache
import os
import sys


# Darwin 的 <sys/wait.h>；值不等同于 Linux 的 waitid 选项。
_DARWIN_P_PID = 1
_DARWIN_WAIT_OPTIONS = 0x01 | 0x04 | 0x20  # WNOHANG | WEXITED | WNOWAIT


class _DarwinSiginfo(ctypes.Structure):
    """Darwin <sys/signal.h> 的 siginfo_t；只读取 si_pid。"""

    _fields_ = [
        ("si_signo", ctypes.c_int), ("si_errno", ctypes.c_int), ("si_code", ctypes.c_int),
        ("si_pid", ctypes.c_int), ("si_uid", ctypes.c_uint), ("si_status", ctypes.c_int),
        ("si_addr", ctypes.c_void_p), ("si_value", ctypes.c_void_p), ("si_band", ctypes.c_long),
        ("_pad", ctypes.c_ulong * 7),
    ]


@lru_cache(maxsize=1)
def _darwin_waitid():
    # 部分 macOS Python 不导出 os.waitid；直接使用 libSystem 的 POSIX 接口。
    try:
        waitid = ctypes.CDLL(None, use_errno=True).waitid
    except AttributeError:
        raise OSError(ENOSYS, "non-reaping child observation unavailable") from None
    waitid.argtypes = (ctypes.c_int, ctypes.c_uint, ctypes.POINTER(_DarwinSiginfo), ctypes.c_int)
    waitid.restype = ctypes.c_int
    return waitid


def child_exited(pid: int) -> bool:
    waitid = getattr(os, "waitid", None)
    if waitid is not None and all(hasattr(os, flag) for flag in ("P_PID", "WEXITED", "WNOHANG", "WNOWAIT")):
        try:
            return waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
        except NotImplementedError:
            pass
        except OSError as exc:
            if exc.errno != ENOSYS:
                raise
    if sys.platform != "darwin":
        raise OSError(ENOSYS, "non-reaping child observation unavailable")

    info = _DarwinSiginfo()  # 无可用退出状态时，si_pid 保持为 0。
    if _darwin_waitid()(_DARWIN_P_PID, pid, ctypes.byref(info), _DARWIN_WAIT_OPTIONS) == -1:
        raise OSError(ctypes.get_errno(), "child exit observation failed")
    if info.si_pid not in (0, pid):
        raise OSError(EIO, "unexpected child exit status")
    return info.si_pid == pid
