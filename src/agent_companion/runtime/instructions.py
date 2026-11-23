"""只加载固定工作目录下的本地系统提示词。"""

import errno
import os
from pathlib import Path
import stat

from ..protocol import ModelError


MAX_AGENTS_BYTES = 32 * 1024


class InstructionsError(ModelError):
    """本地文件不可用；诊断不得带文件内容或路径。"""


def read_agents(path: Path) -> str:
    def safe_open(name: str, flags: int) -> int:
        return os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK)

    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise InstructionsError("agents_md_invalid_file")
        with open(path, "rb", opener=safe_open) as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise InstructionsError("agents_md_invalid_file")
            data = stream.read(MAX_AGENTS_BYTES + 1)
    except FileNotFoundError:
        return ""
    except OSError as exc:
        reason = "agents_md_invalid_file" if exc.errno in (errno.ELOOP, errno.EISDIR) else "agents_md_unreadable"
        raise InstructionsError(reason) from None

    if len(data) > MAX_AGENTS_BYTES:
        raise InstructionsError("agents_md_too_large")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise InstructionsError("agents_md_invalid_utf8") from None
    return text if text.strip() else ""
