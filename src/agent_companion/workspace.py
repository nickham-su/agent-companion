"""用户的工作目录描述；不是文件系统访问权限边界。"""

from dataclasses import dataclass
from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class Workspace:
    directory: Path

    def ensure_directory(self) -> Path:
        """由真正需要目录的工具调用；绑定路由时不创建。"""
        if self.directory.is_symlink():
            raise OSError("workspace directory must not be a symlink")
        try:
            directory = self.directory.resolve()
        except RuntimeError:
            raise OSError("workspace path cannot be resolved") from None
        # 当前运行包，以及使用已安装程序时可能仍位于部署目录内的源码副本。
        if directory.is_relative_to(_PACKAGE_DIR) or any(
            parent.name == "agent_companion" and (parent / "__init__.py").is_file()
            and (parent / "workspace.py").is_file()
            for parent in (directory, *directory.parents)
        ):
            raise OSError("workspace must be outside the program package")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        return directory
