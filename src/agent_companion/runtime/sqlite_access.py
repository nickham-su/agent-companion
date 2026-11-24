"""在事件循环上执行单次 SQLite 纯读或原子写入。"""

import asyncio
from collections.abc import Callable
from sqlite3 import OperationalError
from typing import ParamSpec, TypeVar

from ..storage.sqlite import RetryableInputWrite, is_sqlite_lock_error


_P = ParamSpec("_P")
_T = TypeVar("_T")
_LOCK_RETRY_SECONDS = 0.05
_INPUT_LOCK_RETRIES = 100
_INPUT_IO_RETRIES = 1


class SQLiteAccess:
    def __init__(self, *, retry_locks: bool, stopped: Callable[[], bool]) -> None:
        self._retry_locks = retry_locks
        self._stopped = stopped

    async def __call__(self, operation: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        """只重试返回完整数据的纯读，或失败时已退出事务并回滚的写入。"""
        if not self._retry_locks:
            return operation(*args, **kwargs)
        while not self._stopped():
            try:
                return operation(*args, **kwargs)
            except OperationalError as exc:
                if not is_sqlite_lock_error(exc):
                    raise
            # 不跨 await 持有事务；睡眠可被路由清理取消。
            await asyncio.sleep(_LOCK_RETRY_SECONDS)
        raise asyncio.CancelledError

    async def write_input(self, operation: Callable[[], _T]) -> _T:
        """仅在安全的边界有界重试输入；持久故障交给 Runtime 停住队首。"""
        lock_retries = _INPUT_LOCK_RETRIES if self._retry_locks else 0
        io_retries = _INPUT_IO_RETRIES
        while not self._stopped():
            try:
                return operation()
            except RetryableInputWrite:
                if not io_retries:
                    raise
                io_retries -= 1
            except OperationalError as exc:
                if not lock_retries or not is_sqlite_lock_error(exc):
                    raise
                lock_retries -= 1
            await asyncio.sleep(_LOCK_RETRY_SECONDS)
        raise asyncio.CancelledError
