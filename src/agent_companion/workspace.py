"""用户的工作目录描述；不是文件系统访问权限边界。"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import stat
import sys


_PACKAGE_DIR = Path(__file__).resolve().parent


def _directory_flags(search_only: bool) -> int:
    mode = os.O_RDONLY
    if search_only:
        if sys.platform == "linux":
            mode = getattr(os, "O_PATH", mode)
        elif sys.platform == "darwin":
            # XNU 的 O_SEARCH 等于 O_EXEC | O_DIRECTORY；只用 Python 从 SDK 暴露的标志。
            mode = getattr(os, "O_SEARCH", getattr(os, "O_EXEC", mode))
    return mode | os.O_DIRECTORY | os.O_NOFOLLOW


def open_child_directory(parent_fd: int | None, name: str, *, create: bool, search_only: bool = False) -> int:
    """打开单一目录分量，不跟随链接；并发创建后仍以相同规则打开。"""
    flags = _directory_flags(search_only)
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        return os.open(name, flags, dir_fd=parent_fd)


def _check_package_copy(fd: int) -> None:
    for name in ("__init__.py", "workspace.py"):
        try:
            mode = os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode
        except FileNotFoundError:
            return
        if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
            return
    raise OSError("workspace must be outside the program package")


@dataclass(frozen=True, slots=True)
class Workspace:
    directory: Path

    @classmethod
    def for_user(cls, root: Path, user_key: str) -> "Workspace":
        """计算已核验用户的工作区，不创建目录或会话。"""
        return cls(root.absolute() / sha256(user_key.encode("utf-8")).hexdigest())

    def resolved_directory(self) -> Path:
        """按文件系统语义解析配置中的祖先（含 link/..），不解析工作区末级。"""
        try:
            parent = self.directory.parent.resolve()
        except RuntimeError:
            raise OSError("workspace path cannot be resolved") from None
        directory = parent.parent if self.directory.name == ".." else parent / self.directory.name
        if directory.is_relative_to(_PACKAGE_DIR):
            raise OSError("workspace must be outside the program package")
        return directory

    @contextmanager
    def open_directory(self, *, create: bool = True, search_only: bool = False,
                       check_ancestor: Callable[[int], object] | None = None) -> Iterator[tuple[Path, int]]:
        """逐分量无链接打开规范路径，调用方在 FD 有效期间使用目录。"""
        directory = self.resolved_directory()
        fd = os.open(directory.anchor, _directory_flags(search_only))
        try:
            for component in directory.parts[1:]:
                if check_ancestor is not None:
                    check_ancestor(fd)
                next_fd = open_child_directory(fd, component, create=create, search_only=search_only)
                os.close(fd)
                fd = next_fd
                if component == "agent_companion":
                    _check_package_copy(fd)
            yield directory, fd
        finally:
            os.close(fd)

    def ensure_directory(self) -> Path:
        """由真正需要目录的工具调用；绑定路由时不创建。"""
        with self.open_directory() as (directory, _):
            return directory
