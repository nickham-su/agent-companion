"""协调同库的在途回复写入；进程退出后由系统释放锁，不恢复执行。"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import os
from pathlib import Path
import sqlite3
from threading import RLock


@dataclass(eq=False, slots=True)
class _LockFile:
    fd: int | None

    def flock(self, mode: int) -> None:
        assert self.fd is not None
        fcntl.flock(self.fd, mode | fcntl.LOCK_NB)


@dataclass(slots=True)
class _Writers:
    file: _LockFile | None
    replies: set[str] = field(default_factory=set)


_lock = RLock()
_writers: dict[Path, _Writers] = {}
_files: set[_LockFile] = set()


def _before_fork() -> None:
    # 文件的打开、登记和关闭都在此锁内；fork 不会截断另一线程的这些操作。
    _lock.acquire()


def _after_fork_parent() -> None:
    _lock.release()


def _after_fork_child() -> None:
    global _lock
    try:
        # 只关闭子进程持有的副本，不显式解锁，否则也会解除父进程的 flock。
        for file in tuple(_files):
            _close_lock(file)
        _writers.clear()
    finally:
        # 平衡本轮 fork 的加锁；若当前线程正处于 observe，旧上下文仍会退出旧锁。
        _lock.release()
        _lock = RLock()


os.register_at_fork(before=_before_fork, after_in_parent=_after_fork_parent, after_in_child=_after_fork_child)


def _open_lock(path: Path) -> _LockFile:
    # 单独锁文件：额外打开并关闭 SQLite 主文件会影响 SQLite 的 POSIX 锁。
    # 不删除锁文件，避免多个进程在不同 inode 上获得排他锁。
    file = _LockFile(os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600))
    _files.add(file)
    return file


def _close_lock(file: _LockFile) -> None:
    if file.fd is not None:
        # 子进程为新锁重用 fd 编号后，旧上下文的 finally 不能再关闭它。
        fd, file.fd = file.fd, None
        _files.remove(file)
        os.close(fd)


class ReplyWriters:
    def __init__(self, database_file: str) -> None:
        # SQLite 返回实际的主库路径；内存库只需保护同一连接上的 Runtime。
        self._path = Path(f"{Path(database_file).resolve()}.runtime-lock") if database_file else None
        self._memory = _Writers(None)

    @contextmanager
    def own(self, reply_id: str) -> Iterator[None]:
        path = self._path
        with _lock:
            writer = self._memory if path is None else _writers.get(path)
            if writer is None:
                assert path is not None
                file = _open_lock(path)
                try:
                    file.flock(fcntl.LOCK_EX)
                except BaseException as exc:
                    _close_lock(file)
                    if isinstance(exc, BlockingIOError):
                        raise sqlite3.OperationalError("database is locked") from None
                    raise
                writer = _writers[path] = _Writers(file)
            if reply_id in writer.replies:
                raise ValueError("reply already has a writer")
            writer.replies.add(reply_id)
        try:
            yield
        finally:
            with _lock:
                writer.replies.discard(reply_id)
                if path is not None and not writer.replies and _writers.get(path) is writer:
                    del _writers[path]
                    assert writer.file is not None
                    _close_lock(writer.file)

    def owns(self, reply_id: str) -> bool:
        with _lock:
            writer = self._memory if self._path is None else _writers.get(self._path)
            return writer is not None and reply_id in writer.replies

    @contextmanager
    def observe(self) -> Iterator[set[str] | None]:
        """横跨同步读取/摘要提交；None 表示外部进程可能持有任何未结算回复。"""
        with _lock:
            path = self._path
            if path is None:
                yield self._memory.replies
            elif (writer := _writers.get(path)) is not None:
                yield writer.replies
            else:
                file = _open_lock(path)
                try:
                    try:
                        file.flock(fcntl.LOCK_SH)
                    except BlockingIOError:
                        yield None
                    else:
                        # 快照期间没有外部写入者；历史缺失结果才可判为未知。
                        yield set()
                finally:
                    _close_lock(file)
