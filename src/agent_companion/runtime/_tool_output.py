"""在工具结果进入历史和模型上下文之前，限制预览并归档完整文本。"""

import asyncio
from contextlib import ExitStack, contextmanager
import os
import re
import stat
from threading import Lock
from time import monotonic, time
from uuid import uuid4

from ..workspace import Workspace, open_child_directory
from ._darwin_acl import require_no_acl_grants


_RESULT_LIMIT = 8 * 1024
_PREVIEW_LIMIT = 4 * 1024
_DAY = 24 * 60 * 60
_ARCHIVE_NAME = re.compile(r"result-[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}\.txt")
_RECORD_SUFFIX = ".meta"
_RECORD_LIMIT = 256
_CHECK_LOCK = Lock()
_LAST_CHECK: dict[tuple[int, int], float] = {}


async def limit_tool_output(result: str, workspace: Workspace | None) -> str:
    raw = result.encode("utf-8")
    if len(raw) <= _RESULT_LIMIT:
        return result

    # raw 是完整有效的 UTF-8；切片后至多丢掉末尾不完整的一个字符。
    preview = raw[:_PREVIEW_LIMIT].decode("utf-8", errors="ignore")
    note = "\n\n[公共工具层已截断此工具结果。"
    if workspace is None:
        return preview + note + "没有工作区，未归档。]"
    try:
        archive = await asyncio.to_thread(_archive, workspace, raw)
    except (OSError, ValueError):
        return preview + note + "无法写入归档。]"
    return preview + note + f"截断前的工具结果：{archive}]"


def _directory_info(fd: int, *, ancestor: bool = False) -> os.stat_result:
    info = os.fstat(fd)
    owners = (0, os.geteuid()) if ancestor else (os.geteuid(),)
    # 可信所有者的粘滞祖先目录（如 /tmp）只允许所有者删除子项；打开子项时也验证所有者。
    sticky_ancestor = ancestor and info.st_mode & stat.S_ISVTX
    if info.st_uid not in owners or (info.st_mode & 0o022 and not sticky_ancestor):
        raise OSError("untrusted tool archive directory")
    require_no_acl_grants(fd)
    return info


def _check_ancestor(fd: int) -> None:
    _directory_info(fd, ancestor=True)


@contextmanager
def _open_cache(workspace: Workspace, *, create: bool):
    with ExitStack() as directories:
        _, workspace_fd = directories.enter_context(workspace.open_directory(create=create, check_ancestor=_check_ancestor))
        workspace_info = _directory_info(workspace_fd)
        agent_fd = open_child_directory(workspace_fd, ".agent", create=create)
        directories.callback(os.close, agent_fd)
        _directory_info(agent_fd)
        archive_fd = open_child_directory(agent_fd, "tool-output", create=create)
        directories.callback(os.close, archive_fd)
        cache_info = _directory_info(archive_fd)
        yield workspace_info, archive_fd, cache_info


def _archive(workspace: Workspace, raw: bytes) -> str:
    with _open_cache(workspace, create=True) as (workspace_info, archive_fd, cache_info):
        name = f"result-{uuid4().hex}.txt"
        record_name = name + _RECORD_SUFFIX
        archived = _write_private_file(archive_fd, name, raw)
        record = None
        try:
            # 归档成功关闭后才写来源记录。只记录文件名和身份，不复制正文。
            record = _write_private_file(archive_fd, record_name, _record_bytes(name, archived))
            if _claim_cleanup(_inode(workspace_info)):
                try:
                    _clean_expired(archive_fd, time() - 7 * _DAY, keep=name)
                except OSError:
                    pass  # 缓存清理失败不能使已写入的工具结果失败。
            # 不返回已经被移走的目录或文件。打开的目录 FD 本身不会随路径替换指向新目录。
            with _open_cache(workspace, create=False) as (_, current_fd, current_info):
                if (_inode(current_info) != _inode(cache_info)
                        or not _same_file(current_fd, name, archived)
                        or not _same_file(current_fd, record_name, record)):
                    raise OSError("tool archive path changed")
        except OSError:
            if record is not None:
                _discard(archive_fd, record_name, record)
            _discard(archive_fd, name, archived)
            raise
        return f".agent/tool-output/{name}"


def _inode(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _identity(info: os.stat_result) -> tuple[int, int, int, int]:
    # ctime 和大小让记录不会在同名文件被替换且 inode 复用后误认新文件。
    return info.st_dev, info.st_ino, info.st_ctime_ns, info.st_size


def _record_bytes(name: str, info: os.stat_result) -> bytes:
    dev, ino, ctime, size = _identity(info)
    return f"v1 {name} {dev} {ino} {ctime} {size}\n".encode("ascii")


def _private_file(info: os.stat_result) -> bool:
    return (stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
            and info.st_nlink == 1 and not info.st_mode & 0o077)


def _same_file(directory: int, name: str, expected: os.stat_result) -> bool:
    current = os.stat(name, dir_fd=directory, follow_symlinks=False)
    return _private_file(current) and _identity(current) == _identity(expected)


def _discard(directory: int, name: str, expected: os.stat_result) -> None:
    try:
        if _same_file(directory, name, expected):
            os.unlink(name, dir_fd=directory)
    except OSError:
        pass


def _write_private_file(directory: int, name: str, raw: bytes) -> os.stat_result:
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    try:
        try:
            # 归档和来源记录都必须在写入任何内容前检查继承的 Darwin ACL。
            require_no_acl_grants(fd)
            # 用副本写入；写入或关闭失败后仍可在原 FD 上核对该文件身份。
            writer_fd = os.dup(fd)
            try:
                file = os.fdopen(writer_fd, "wb")
            except BaseException:
                os.close(writer_fd)
                raise
            with file:
                file.write(raw)
            return os.fstat(fd)
        except OSError:
            _discard(directory, name, os.fstat(fd))
            raise
    finally:
        os.close(fd)


def _claim_cleanup(workspace: tuple[int, int]) -> bool:
    with _CHECK_LOCK:
        now = monotonic()
        last = _LAST_CHECK.get(workspace)
        if last is not None and now - last < _DAY:
            return False
        _LAST_CHECK[workspace] = now
        return True


def _clean_expired(archive_fd: int, cutoff: float, *, keep: str) -> None:
    with os.scandir(archive_fd) as entries:
        for entry in entries:
            if entry.name == keep or not _ARCHIVE_NAME.fullmatch(entry.name):
                continue
            try:
                info = entry.stat(follow_symlinks=False)
                if not _private_file(info) or info.st_mtime >= cutoff:
                    continue
                record_name = entry.name + _RECORD_SUFFIX
                record_fd = os.open(record_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=archive_fd)
                try:
                    record = os.fstat(record_fd)
                    if not _private_file(record) or record.st_size > _RECORD_LIMIT:
                        continue
                    require_no_acl_grants(record_fd)
                    if os.read(record_fd, _RECORD_LIMIT) != _record_bytes(entry.name, info):
                        continue
                    if not _same_file(archive_fd, record_name, record) or not _same_file(archive_fd, entry.name, info):
                        continue
                    os.unlink(entry.name, dir_fd=archive_fd)
                    _discard(archive_fd, record_name, record)
                finally:
                    os.close(record_fd)
            except OSError:
                continue  # 来源未知、文件消失或不可清理，不妨碍检查其他归档。
    # 孤立记录（如进程在两次 unlink 间退出）和不完整记录不能证明任何归档，留存而不删除。
