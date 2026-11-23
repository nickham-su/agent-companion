"""仅配置应用自己的标准输出日志，不打开第三方库的调试输出。"""

from collections.abc import Iterator
from contextlib import contextmanager
import logging
import sys
from threading import Lock
from time import monotonic


_WARNING_INTERVAL = 30.0


class _WarningThrottle(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self._last: dict[tuple[str, str], float] = {}
        self._lock = Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.WARNING or not isinstance(record.msg, str):
            return True
        # 应用日志使用固定模板。按模板限频，不保存正文或每次请求的参数。
        key = record.name, record.msg
        with self._lock:
            now = monotonic()
            last = self._last.get(key)
            if last is not None and now - last < _WARNING_INTERVAL:
                return False
            self._last[key] = now
            return True


@contextmanager
def stdout_logging(level: str) -> Iterator[None]:
    logger = logging.getLogger("agent_companion")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    if level != "DEBUG":
        handler.addFilter(_WarningThrottle())
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.setLevel(level)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
