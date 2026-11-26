"""启动时加载可信工作目录下的角色配置。"""

import errno
import os
from pathlib import Path
import re
import stat
import tomllib

from ..config import ConfigError, valid_config_id
from .subtasks import AGENT_TOOL_NAMES, AgentRole


MAX_ROLE_BYTES = 32 * 1024
_MISSING_MAIN = "缺少 agents/main.md，请从 agents/main.example.md 复制并配置"
_OPENING = re.compile(r"\+\+\+\r?\n")
_CLOSING = re.compile(r"^\+\+\+(?:\r?\n|\Z)", re.MULTILINE)
_TOML_LOCATION = re.compile(r"\(at line (\d+), column (\d+)\)\Z")


def _invalid_role(name: str, reason: str) -> ConfigError:
    # 只标识文件及固定诊断，不回显字段值、正文或解析器异常。
    return ConfigError(f"角色文件 {'agents/' + name!r}：{reason}")


def _read_role_file(directory: int, name: str) -> str:
    def safe_open(path: str, flags: int) -> int:
        return os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)

    try:
        if not stat.S_ISREG(os.stat(name, dir_fd=directory, follow_symlinks=False).st_mode):
            raise _invalid_role(name, "必须是普通文件，不接受符号链接")
        with open(name, "rb", opener=safe_open) as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise _invalid_role(name, "必须是普通文件，不接受符号链接")
            data = stream.read(MAX_ROLE_BYTES + 1)
    except FileNotFoundError:
        if name == "main.md":
            raise ConfigError(_MISSING_MAIN) from None
        raise _invalid_role(name, "文件不存在") from None
    except OSError as exc:
        reason = "必须是普通文件，不接受符号链接" if exc.errno in (errno.ELOOP, errno.EISDIR) else "无法读取文件"
        raise _invalid_role(name, reason) from None

    if len(data) > MAX_ROLE_BYTES:
        raise _invalid_role(name, "文件超过 32 KiB")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise _invalid_role(name, "文件必须是 UTF-8 编码") from None


def _parse_role(name: str, text: str) -> AgentRole:
    opening = _OPENING.match(text)
    if opening is None:
        raise _invalid_role(name, "首行必须为独占一行的 +++")
    closing = _CLOSING.search(text, opening.end())
    if closing is None:
        raise _invalid_role(name, "缺少独占一行的结束分隔线 +++")
    header = text[opening.end():closing.start()]
    try:
        fields = tomllib.loads(header)
    except tomllib.TOMLDecodeError as exc:
        # Python 3.11 只在异常消息末尾提供行列号；不将整条解析异常带入诊断。
        position = _TOML_LOCATION.search(str(exc))
        if position is None:
            raise _invalid_role(name, "头部 TOML 格式错误（检查文件头末尾）") from None
        line, column = position.groups()
        raise _invalid_role(name, f"头部 TOML 格式错误（第 {int(line) + 1} 行，第 {column} 列）") from None
    except (ValueError, RecursionError):
        raise _invalid_role(name, "头部 TOML 格式错误") from None
    if fields.keys() - {"description", "tools", "model_id"}:
        raise _invalid_role(name, "头部有未知字段，仅支持 description、tools 和 model_id")
    model_id = fields.get("model_id")
    if not valid_config_id(model_id):
        raise _invalid_role(name, "model_id 必须显式设置为非空模型配置 ID，请参考 [models.llm.<ID>]")
    description = fields.get("description", "")
    if not isinstance(description, str) or (name != "main.md" and not description.strip()):
        raise _invalid_role(name, "description 必须为字符串，子角色须填写非空描述")
    tools = fields.get("tools")
    if not isinstance(tools, list) or any(not isinstance(tool, str) for tool in tools):
        raise _invalid_role(name, "tools 必须为工具名数组")
    if set(tools) - AGENT_TOOL_NAMES:
        raise _invalid_role(name, "tools 包含未知工具名")
    if len(set(tools)) != len(tools):
        raise _invalid_role(name, "tools 不得重复声明工具")
    # 二进制读取和偏移切片保留 CRLF、正文分隔线及其他字符。
    return AgentRole(text[closing.end():], tuple(tools), description, model_id)


def _role_id(name: str) -> str:
    role_id = name.removesuffix(".md")
    if not role_id.strip():
        raise _invalid_role(name, "角色文件名须有非空的 UTF-8 名称")
    try:
        role_id.encode("utf-8")
    except UnicodeEncodeError:
        raise _invalid_role(name, "角色文件名须有非空的 UTF-8 名称") from None
    return role_id


def load_roles(working_directory: Path) -> dict[str, AgentRole]:
    """一次性加载本地角色，启动后不再读取文件。"""
    try:
        directory = os.open(working_directory / "agents", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        raise ConfigError(_MISSING_MAIN) from None
    except OSError:
        raise ConfigError("无法打开 agents/：须为可读取的普通目录，不接受符号链接") from None

    try:
        try:
            names = os.listdir(directory)
        except OSError:
            raise ConfigError("无法列出 agents/ 中的角色文件") from None
        if "main.md" not in names:
            raise ConfigError(_MISSING_MAIN)
        return {
            _role_id(name): _parse_role(name, _read_role_file(directory, name))
            for name in sorted(names) if name.endswith(".md") and not name.endswith(".example.md")
        }
    finally:
        os.close(directory)
