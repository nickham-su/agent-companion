"""SQLite 中的 Agent、会话历史、摘要和投递记录。"""

from base64 import b64decode, b64encode
from collections.abc import Callable, Iterable, Iterator, Sequence, Set
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
import json
from os import fspath
from pathlib import Path
from secrets import token_bytes, token_hex
import sqlite3
from time import time
from uuid import uuid4

from ..images import UNAVAILABLE_IMAGE, image_filename
from ..protocol import (AsyncToolResult, DeliveryStatus, Image, Message, MessageDelivery, Part, PrivateData,
                        Reasoning, Text, ToolCall, UserSource)
from ._reply_writers import ReplyWriters


_SCHEMA_VERSION = 5
_AGENT_ID_ATTEMPTS = 8
_FEISHU_INBOX = ("feishu-inbox", "feishu-private")
_FEISHU_INITIAL_RETRY_SECONDS = 10
_FEISHU_MAX_RETRY_SECONDS = 900
_MODEL_ORIGINS = """
CREATE TABLE IF NOT EXISTS reply_model_origins (
    reply_id TEXT PRIMARY KEY REFERENCES reply_outcomes(reply_id) ON DELETE CASCADE,
    assistant_message_id INTEGER UNIQUE REFERENCES messages(message_id) ON DELETE CASCADE,
    fingerprint BLOB NOT NULL CHECK (length(fingerprint) = 32)
)
"""
_MESSAGE_COLUMNS = (
    "m.message_id, m.session_id, m.created_at, m.kind, m.parts, m.source, m.async_result, "
    "(SELECT json_group_object(call_id, output) FROM tool_results "
    "WHERE assistant_message_id = m.message_id) AS tool_results"
)
_UNSETTLED_REPLY = (
    "m.reply_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM reply_outcomes o WHERE o.reply_id = m.reply_id)"
)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('user', 'assistant', 'state_event', 'async_tool_result')),
    parts TEXT NOT NULL,
    source TEXT,
    async_result TEXT,
    reply_id TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS tool_results (
    assistant_message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    call_id TEXT NOT NULL,
    output TEXT NOT NULL,
    PRIMARY KEY (assistant_message_id, call_id)
);
CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, message_id);
CREATE VIRTUAL TABLE IF NOT EXISTS message_search USING fts5(body);
CREATE TABLE IF NOT EXISTS summaries (
    session_id TEXT PRIMARY KEY,
    covers_through INTEGER NOT NULL REFERENCES messages(message_id),
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reply_outcomes (
    reply_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    model_status TEXT NOT NULL,
    delivery_status TEXT,
    assistant_message_id INTEGER REFERENCES messages(message_id),
    model_error TEXT,
    input_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS reply_outcomes_session_status ON reply_outcomes(session_id, model_status);
CREATE TABLE IF NOT EXISTS model_replay_salt (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    salt BLOB NOT NULL CHECK (length(salt) = 32)
);
CREATE TABLE IF NOT EXISTS inbound_windows (
    session_id TEXT NOT NULL,
    source TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    scanned_until INTEGER NOT NULL,
    PRIMARY KEY (session_id, source)
);
CREATE TABLE IF NOT EXISTS inbound_seen (
    session_id TEXT NOT NULL,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    PRIMARY KEY (session_id, source, external_id)
);
CREATE TABLE IF NOT EXISTS feishu_raw_batches (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    peer_open_id TEXT NOT NULL,
    has_media INTEGER NOT NULL,
    contents TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS feishu_raw_pending ON feishu_raw_batches(peer_open_id, batch_id);
CREATE TABLE IF NOT EXISTS feishu_recovery (
    message_id TEXT PRIMARY KEY NOT NULL,
    peer_open_id TEXT NOT NULL,
    chat_digest TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_second INTEGER NOT NULL,
    retry_at INTEGER NOT NULL,
    retry_delay INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS feishu_recovery_due ON feishu_recovery(retry_at, message_id);
CREATE TABLE IF NOT EXISTS feishu_private_account (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    owner_fingerprint TEXT NOT NULL,
    started_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT UNIQUE NOT NULL,
    role_id TEXT NOT NULL,
    parent_agent_id TEXT REFERENCES agents(agent_id),
    CHECK ((parent_agent_id IS NULL) = (role_id = 'main')),
    CHECK (parent_agent_id IS NULL OR parent_agent_id <> agent_id)
);
CREATE INDEX IF NOT EXISTS agents_parent ON agents(parent_agent_id);
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY NOT NULL,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id),
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'interrupted')),
    started_at REAL NOT NULL,
    finished_at REAL,
    result TEXT,
    error TEXT,
    UNIQUE (agent_id, run_id)
);
CREATE TABLE IF NOT EXISTS delegations (
    delegation_id TEXT PRIMARY KEY NOT NULL,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id),
    parent_agent_id TEXT NOT NULL,
    parent_run_id TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    accepted_at REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'consumed', 'interrupted')),
    run_id TEXT,
    error TEXT,
    CHECK ((status = 'consumed') = (run_id IS NOT NULL)),
    FOREIGN KEY (parent_agent_id, parent_run_id) REFERENCES agent_runs(agent_id, run_id),
    FOREIGN KEY (agent_id, run_id) REFERENCES agent_runs(agent_id, run_id)
);
CREATE INDEX IF NOT EXISTS delegations_by_run ON delegations(run_id);
"""


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    message_id: int
    session_id: str
    created_at: float
    message: Message
    active_reply: bool = False


@dataclass(frozen=True, slots=True)
class Summary:
    covers_through: int
    text: str


@dataclass(frozen=True, slots=True)
class ReplyOutcome:
    reply_id: str
    session_id: str
    run_id: str
    model_status: str
    delivery_status: DeliveryStatus | None = None
    assistant_message_id: int | None = None
    model_error: str | None = None
    input_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Agent:
    agent_id: str
    session_id: str
    role_id: str
    parent_agent_id: str | None


@dataclass(frozen=True, slots=True)
class Delegation:
    delegation_id: str
    agent_id: str
    parent_agent_id: str
    parent_run_id: str
    tool_call_id: str
    accepted_at: float
    status: str
    run_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRun:
    run_id: str
    agent_id: str
    status: str
    started_at: float
    finished_at: float | None
    result: str | None = field(repr=False)
    error: str | None


def _has_reply_text(value: str | None) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _encode_parts(parts: Sequence[Part]) -> str:
    encoded = []
    for part in parts:
        if isinstance(part, Text):
            item = {"type": "text", "text": part.text}
        elif isinstance(part, Image):
            item = ({"type": "image", "url": part.url} if image_filename(part.url) is not None else
                    {"type": "text", "text": UNAVAILABLE_IMAGE.text})
        elif isinstance(part, ToolCall):
            # result_image 是出站请求副本中的字段，不进入历史。
            item = {"type": "tool_call", "id": part.id, "name": part.name, "arguments": part.arguments}
        else:
            item = {"type": "reasoning"}
        if not isinstance(part, Image) and part.private is not None:
            item["private"] = [part.private.compatibility_key.hex(), b64encode(part.private.payload).decode("ascii")]
        encoded.append(item)
    return _dump(encoded)


def _decode_private(data: object) -> PrivateData | None:
    if not isinstance(data, list) or len(data) != 2 or not all(isinstance(value, str) for value in data):
        return None
    try:
        return PrivateData(bytes.fromhex(data[0]), b64decode(data[1], validate=True))
    except ValueError:
        # 私有数据损坏不影响同一消息中可独立使用的正文和调用结果。
        return None


def _decode_parts(encoded: str, *, private: bool = False, results: dict[str, str] | None = None) -> tuple[Part, ...]:
    parts: list[Part] = []
    for item in json.loads(encoded):
        metadata = _decode_private(item.get("private")) if private else None
        match item["type"]:
            case "text":
                parts.append(Text(item["text"], metadata))
            case "image":
                parts.append(Image(item["url"]))
            case "tool_call":
                result = results.get(item["id"]) if results is not None else None
                parts.append(ToolCall(item["id"], item["name"], item["arguments"], result, metadata))
            case "reasoning":
                if metadata is not None:
                    parts.append(Reasoning(metadata))
            case _:
                raise ValueError("unknown stored part type")
    return tuple(parts)


def _call_identity(message: Message) -> dict[str, tuple[str, str]]:
    return {call.id: (call.name, call.arguments) for call in message.tool_calls}


def _decode(row: sqlite3.Row, *, private: bool = False) -> HistoryEntry:
    source = json.loads(row["source"]) if row["source"] is not None else None
    result = json.loads(row["async_result"]) if row["async_result"] is not None else None
    return HistoryEntry(
        row["message_id"], row["session_id"], row["created_at"],
        Message(row["kind"], _decode_parts(row["parts"], private=private, results=json.loads(row["tool_results"])),
                source=UserSource(**source) if source is not None else None,
                async_result=AsyncToolResult(**result) if result is not None else None),
    )


class FeishuBindingError(ValueError):
    """飞书账号或会话的数据库归属无法确认。"""


class DatabaseFormatError(ValueError):
    """数据库不是空库或当前支持的版本。"""


class SummaryWriteError(Exception):
    """摘要写入未提交，可以继续使用已存历史。"""


class RetryableInputWrite(Exception):
    """输入写语句遇到 IOERR，且提交前已成功回滚。"""


class InputCommitUnknown(RuntimeError):
    """不能仅凭提交报错判定事务可重试。"""


class FeishuRawConfirmationUnresolved(RuntimeError):
    """连接仍可用，但不能确认该用户的原始批次已删除。"""


class InputRollbackFailed(RuntimeError):
    """输入事务回滚失败；共享连接已失效。"""


def is_sqlite_lock_error(exc: sqlite3.OperationalError) -> bool:
    """按 SQLite 主错误码（含扩展码）判定；兼容未携带错误码的合成异常。"""
    code = getattr(exc, "sqlite_errorcode", None)
    if code is not None:
        return isinstance(code, int) and (code & 0xFF) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
    return str(exc) in {"database is locked", "database table is locked", "database schema is locked",
                        "database is busy"}


@dataclass(frozen=True, slots=True, repr=False)
class FeishuCandidate:
    message_id: str
    peer_open_id: str
    chat_digest: str
    kind: str
    created_second: int


FeishuRaw = tuple[str, int, str, str]  # 消息 ID、毫秒时间、类型、原始正文（不含来源字段）


class History:
    def __init__(self, path: str | Path, *, feishu_private_owner: str | None = None) -> None:
        self._connection: sqlite3.Connection | None = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        try:
            database_file = self._db.execute("PRAGMA database_list").fetchone()[2]
            if not database_file and fspath(path) not in ("", ":memory:"):
                # SQLite 可把文件 URI 打开为共享内存库，但不给出可用的主库身份。
                # 在建表前拒绝；普通 :memory: 和匿名临时库都只对当前连接可见。
                raise DatabaseFormatError("不支持 URI 内存数据库；请使用普通 :memory: 或 SQLite 文件路径。")
            # 旧格式直接拒绝，无需争用写锁。初始化仍会在写事务内复查，防止并发建表。
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, _SCHEMA_VERSION):
                raise DatabaseFormatError("不支持当前 SQLite 数据库格式；请保留原库并配置新的数据库路径。")
            self._db.execute("PRAGMA foreign_keys = ON")
            with self._db:
                # 串行化首次初始化。检查格式后才可写入 schema，并且版本标记与建表一起提交。
                self._db.execute("BEGIN IMMEDIATE")
                version = self._db.execute("PRAGMA user_version").fetchone()[0]
                if version != _SCHEMA_VERSION:
                    if version != 0 or self._db.execute("SELECT 1 FROM sqlite_schema LIMIT 1").fetchone():
                        raise DatabaseFormatError("不支持当前 SQLite 数据库格式；请保留原库并配置新的数据库路径。")
                    # _SCHEMA 仅包含独立 DDL；不能用 executescript，否则会隐式提交上面的事务。
                    for statement in _SCHEMA.split(";"):
                        if statement.strip():
                            self._db.execute(statement)
                    self._db.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                if feishu_private_owner is not None:
                    # 打开时只预检已有归属；空库由绑定事务认领。
                    self.check_feishu_private(feishu_private_owner)
                self._db.execute(_MODEL_ORIGINS)
            self._reply_writers = ReplyWriters(database_file)
        except BaseException:
            self.close()
            raise

    @property
    def _db(self) -> sqlite3.Connection:
        connection = self._connection
        if connection is None:
            raise sqlite3.ProgrammingError("storage connection is closed")
        return connection

    def close(self) -> None:
        # 先摘除持有的连接；即使 close 抛错，也不能再从本 History 发出读写。
        connection = self._connection
        self._connection = None
        if connection is not None:
            connection.close()

    def replay_salt(self) -> bytes:
        """首次使用建立库级随机盐；重启和不同会话使用同一值。"""
        row = self._db.execute("SELECT salt FROM model_replay_salt WHERE singleton = 1").fetchone()
        if row is None:
            with self._db:
                self._db.execute("INSERT INTO model_replay_salt VALUES (1, ?) ON CONFLICT DO NOTHING",
                                 (token_bytes(32),))
                row = self._db.execute("SELECT salt FROM model_replay_salt WHERE singleton = 1").fetchone()
        return row["salt"]

    def _has_history(self) -> bool:
        return self._db.execute("""
            SELECT 1 FROM messages
            UNION ALL SELECT 1 FROM agents
            UNION ALL SELECT 1 FROM summaries
            UNION ALL SELECT 1 FROM reply_outcomes
            UNION ALL SELECT 1 FROM inbound_windows
            UNION ALL SELECT 1 FROM inbound_seen
            UNION ALL SELECT 1 FROM feishu_raw_batches
            UNION ALL SELECT 1 FROM feishu_recovery
            UNION ALL SELECT 1 FROM message_search
            LIMIT 1
            """).fetchone() is not None

    def bind_feishu_private(self, owner_fingerprint: str, *, started_ms: int) -> int:
        """账号核验后，在同一事务重新检查并保存首次启用水位。"""
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            prior = self.check_feishu_private(owner_fingerprint)
            if prior is None:
                self._db.execute("INSERT INTO feishu_private_account VALUES (1, ?, ?)", (owner_fingerprint, started_ms))
        # 多私聊的轮询在事件循环上执行，不能用 SQLite 的同步忙等阻塞其他路由。
        self._db.execute("PRAGMA busy_timeout = 0")
        return started_ms if prior is None else prior

    def check_feishu_private(self, owner_fingerprint: str) -> int | None:
        """只读检查已有账号归属；不认领空库。"""
        row = self._db.execute("SELECT owner_fingerprint, started_ms FROM feishu_private_account").fetchone()
        if row is not None:
            if row["owner_fingerprint"] != owner_fingerprint:
                raise FeishuBindingError("飞书服务账号与数据库绑定不一致；请切回原 CLI 登录账号和应用，或使用新的数据库路径。")
            return row["started_ms"]
        if self._has_history():
            raise FeishuBindingError("现有数据库有历史但没有飞书账号绑定；请保留旧数据库并配置新的数据库路径。")
        return None

    def ensure_main_agent(self, session_id: str) -> Agent:
        """按内部 Session 键建立可复用的主 Agent。"""
        with self._input_transaction(retryable_io=False):
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute("SELECT * FROM agents WHERE session_id = ?", (session_id,)).fetchone()
            if row is None:
                return self._insert_agent(session_id, "main", None)
            if row["parent_agent_id"] is not None:
                raise ValueError("session belongs to a child agent")
            return Agent(**dict(row))

    def _insert_agent(self, session_id: str, role_id: str, parent_agent_id: str | None) -> Agent:
        for attempt in range(1, _AGENT_ID_ATTEMPTS + 1):
            agent_id = f"a-{token_hex(4)}"
            # 在运行的父 Agent 已占用这个 ID；直接写入会先触发禁止自引用的 CHECK。
            if agent_id == parent_agent_id:
                continue
            try:
                self._db.execute(
                    "INSERT INTO agents (agent_id, session_id, role_id, parent_agent_id) VALUES (?, ?, ?, ?)",
                    (agent_id, session_id, role_id, parent_agent_id))
            except sqlite3.IntegrityError as exc:
                if (attempt == _AGENT_ID_ATTEMPTS or exc.sqlite_errorcode != sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY
                        or str(exc) != "UNIQUE constraint failed: agents.agent_id"):
                    raise
            else:
                return Agent(agent_id, session_id, role_id, parent_agent_id)
        raise sqlite3.IntegrityError("could not allocate a unique agents.agent_id")

    def agent(self, agent_id: str) -> Agent | None:
        row = self._db.execute("SELECT * FROM agents WHERE agent_id = ?", (agent_id,)).fetchone()
        return Agent(**dict(row)) if row is not None else None

    def accept_delegation(self, parent_agent_id: str, parent_run_id: str, tool_call_id: str, *,
                          role_id: str | None = None, agent_id: str | None = None) -> tuple[Agent, Delegation] | None:
        """原子建立子 Agent 和委派，或复用直属子 Agent；非直属/未知目标返回 None。"""
        if (role_id is None) == (agent_id is None) or role_id in ("", "main"):
            raise ValueError("invalid delegation target")
        with self._input_transaction(retryable_io=False):
            self._db.execute("BEGIN IMMEDIATE")
            if not self._agent_run_active(parent_agent_id, parent_run_id):
                raise ValueError("delegation requires an active parent run")
            if role_id is not None:
                child = self._insert_agent(f"child-{uuid4().hex}", role_id, parent_agent_id)
            else:
                assert agent_id is not None
                child = self.agent(agent_id)
                if child is None or child.parent_agent_id != parent_agent_id:
                    return None
            delegation = Delegation(f"delegation-{uuid4().hex}", child.agent_id, parent_agent_id, parent_run_id,
                                    tool_call_id, time(), "queued")
            self._db.execute("INSERT INTO delegations"
                             " (delegation_id, agent_id, parent_agent_id, parent_run_id, tool_call_id, accepted_at, status)"
                             " VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (delegation.delegation_id, child.agent_id, parent_agent_id, parent_run_id, tool_call_id,
                              delegation.accepted_at, delegation.status))
            return child, delegation

    def delegation(self, delegation_id: str) -> Delegation | None:
        row = self._db.execute("SELECT * FROM delegations WHERE delegation_id = ?", (delegation_id,)).fetchone()
        return Delegation(**dict(row)) if row is not None else None

    def run_delegation_ids(self, run_id: str) -> tuple[str, ...]:
        return tuple(row[0] for row in self._db.execute(
            "SELECT delegation_id FROM delegations WHERE run_id = ? ORDER BY rowid", (run_id,)))

    def interrupt_delegation(self, delegation_id: str, *, reason: str) -> None:
        """提交后若入口已关闭，将尚未进入运行的委派标记中断。"""
        with self._input_transaction(retryable_io=False):
            self._db.execute("UPDATE delegations SET status = 'interrupted', error = ?"
                             " WHERE delegation_id = ? AND status = 'queued'", (reason, delegation_id))

    def _agent_run_active(self, agent_id: str, run_id: str) -> bool:
        return self._db.execute("SELECT 1 FROM agent_runs WHERE agent_id = ? AND run_id = ? AND status = 'running'",
                                (agent_id, run_id)).fetchone() is not None

    def agent_run(self, run_id: str) -> AgentRun | None:
        row = self._db.execute("SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)).fetchone()
        return AgentRun(**dict(row)) if row is not None else None

    def _append(self, session_id: str, messages: Iterable[Message], *, reply_id: str | None = None,
                staged: bool = False) -> list[int]:
        created_at = time()
        ids = []
        for message in messages:
            if message.kind != "assistant" and any(isinstance(part, (Reasoning, ToolCall)) or
                    isinstance(part, Text) and part.private is not None for part in message.parts):
                raise ValueError("only assistant messages may carry calls or private data")
            cursor = self._db.execute(
                "INSERT INTO messages (session_id, created_at, kind, parts, source, async_result, reply_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, created_at, message.kind, _encode_parts(message.parts),
                 _dump(asdict(message.source)) if message.source is not None else None,
                 _dump(asdict(message.async_result)) if message.async_result is not None else None, reply_id),
            )
            message_id = cursor.lastrowid
            for call in message.tool_calls:
                if call.result is None:
                    if staged:
                        continue
                    raise ValueError("settled tool calls require a result")
                self._insert_tool_result(message_id, call.id, call.result)
            self._index_message(message_id, message)
            ids.append(message_id)
        return ids

    def _insert_tool_result(self, assistant_message_id: int, call_id: str, output: str) -> None:
        self._db.execute("INSERT INTO tool_results VALUES (?, ?, ?)", (assistant_message_id, call_id, output))

    def _reindex_message(self, message_id: int, message: Message) -> None:
        self._db.execute("DELETE FROM message_search WHERE rowid = ?", (message_id,))
        self._index_message(message_id, message)

    def _index_message(self, message_id: int, message: Message) -> None:
        if message.async_result is not None:
            searchable = json.dumps(asdict(message.async_result), ensure_ascii=False).encode(
                "utf-8", "backslashreplace").decode("utf-8")
        else:
            content = message.content
            text = content if isinstance(content, str) else "\n".join(
                part.text for part in content if isinstance(part, Text))
            searchable = "\n".join(filter(None, (text, *(call.result for call in message.tool_calls))))
        if searchable:
            self._db.execute("INSERT INTO message_search (rowid, body) VALUES (?, ?)", (message_id, searchable))

    @contextmanager
    def _input_transaction(self, *, retryable_io: bool,
                           commit_unknown_on_db_error: bool = False) -> Iterator[None]:
        """区分写语句与提交；飞书原批确认可要求核验非锁提交错误。"""
        connection = self._db
        try:
            yield
        except BaseException as exc:
            self._rollback_input(connection, exc)
            if (retryable_io and isinstance(exc, sqlite3.OperationalError)
                    and getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_IOERR):
                raise RetryableInputWrite from exc
            raise
        else:
            try:
                connection.commit()
            except BaseException as exc:
                locked = isinstance(exc, sqlite3.OperationalError) and is_sqlite_lock_error(exc)
                unknown = ((locked and not connection.in_transaction)
                           or (commit_unknown_on_db_error and not locked
                               and isinstance(exc, sqlite3.DatabaseError)))
                self._rollback_input(connection, exc)
                if unknown:
                    raise InputCommitUnknown("input commit outcome is unknown") from exc
                raise

    def _rollback_input(self, connection: sqlite3.Connection, trigger: BaseException) -> None:
        try:
            connection.rollback()
        except BaseException as exc:
            # 不允许另一会话在同一个连接上提交未成功回滚的事务。
            try:
                self.close()
            except Exception:
                # 持有者已摘除连接，仍按回滚失败报告，不记录底层异常正文。
                pass
            if not isinstance(exc, Exception):
                raise
            if isinstance(trigger, Exception):
                raise InputRollbackFailed("input transaction rollback failed") from exc
            # 原异常是取消或其他 BaseException，由事务调用点继续抛出。

    def append(self, session_id: str, messages: Iterable[Message], *, retryable_io: bool = False) -> list[int]:
        with self._input_transaction(retryable_io=retryable_io):
            return self._append(session_id, messages)

    def inbound_window(self, session_id: str, source: str, now: int) -> tuple[int, int]:
        key = (session_id, source)
        with self._db:
            self._db.execute("INSERT INTO inbound_windows VALUES (?, ?, ?, ?)"
                             " ON CONFLICT DO NOTHING", (*key, now, now))
            row = self._db.execute("SELECT started_at, scanned_until FROM inbound_windows"
                                   " WHERE session_id = ? AND source = ?", key).fetchone()
        return row["started_at"], row["scanned_until"]

    def inbound_seen(self, session_id: str, source: str, external_id: str) -> bool:
        return self._db.execute("SELECT 1 FROM inbound_seen WHERE session_id = ? AND source = ? AND external_id = ?",
                                (session_id, source, external_id)).fetchone() is not None

    def _mark_inbound_seen(self, key: tuple[str, str], message_id: str) -> bool:
        return self._db.execute("INSERT INTO inbound_seen VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                                (*key, message_id)).rowcount == 1

    def _advance_inbound(self, key: tuple[str, str], through: int) -> None:
        self._db.execute("UPDATE inbound_windows SET scanned_until = max(scanned_until, ?)"
                         " WHERE session_id = ? AND source = ?", (through, *key))

    def feishu_inbound_window(self, initial: int) -> tuple[int, int]:
        return self.inbound_window(*_FEISHU_INBOX, initial)

    def feishu_inbound_seen(self, message_id: str) -> bool:
        return self.inbound_seen(*_FEISHU_INBOX, message_id)

    def feishu_recovery_candidate(self, message_id: str) -> FeishuCandidate | None:
        row = self._db.execute(
            "SELECT message_id, peer_open_id, chat_digest, kind, created_second FROM feishu_recovery"
            " WHERE message_id = ?", (message_id,)).fetchone()
        return None if row is None else FeishuCandidate(*row)

    def reconcile_feishu_search(self, hits: Iterable[FeishuCandidate], *, now: int) -> list[FeishuCandidate]:
        """返回新 ID；待恢复项仅在新搜索依据有变化时刷新并提前到期。"""
        fresh, changed = [], []
        for hit in hits:
            prior = self.feishu_recovery_candidate(hit.message_id)
            if prior is None:
                # 固定窗口可能跨多轮分页，其间旧候选可能已经恢复。
                if not self.feishu_inbound_seen(hit.message_id):
                    fresh.append(hit)
            elif prior != hit:
                changed.append(hit)
        if changed:
            with self._input_transaction(retryable_io=False):
                self._db.executemany(
                    "UPDATE feishu_recovery SET peer_open_id = ?, chat_digest = ?, kind = ?, created_second = ?,"
                    " retry_at = min(retry_at, ?) WHERE message_id = ?",
                    ((hit.peer_open_id, hit.chat_digest, hit.kind, hit.created_second, now, hit.message_id)
                     for hit in changed))
        return fresh

    def due_feishu_candidates(self, now: int, limit: int) -> list[FeishuCandidate]:
        return [FeishuCandidate(*row) for row in self._db.execute(
            "SELECT message_id, peer_open_id, chat_digest, kind, created_second FROM feishu_recovery"
            " WHERE retry_at <= ? ORDER BY retry_at, message_id LIMIT ?", (now, limit))]

    def record_feishu_window(self, through: int | None, *, now: int,
                             verified: dict[str, list[FeishuRaw]], discarded: Sequence[str],
                             retry: Sequence[FeishuCandidate]) -> None:
        """原始输入、去重、恢复项和水位同事务；恢复本身不改变主水位。"""
        with self._input_transaction(retryable_io=False):
            for hit in retry:
                self._db.execute(
                    "INSERT INTO feishu_recovery VALUES (?, ?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(message_id) DO UPDATE SET retry_delay = min(retry_delay * 2, ?),"
                    " retry_at = ? + min(retry_delay * 2, ?)",
                    (hit.message_id, hit.peer_open_id, hit.chat_digest, hit.kind, hit.created_second,
                     now + _FEISHU_INITIAL_RETRY_SECONDS, _FEISHU_INITIAL_RETRY_SECONDS,
                     _FEISHU_MAX_RETRY_SECONDS, now, _FEISHU_MAX_RETRY_SECONDS))
            for mid in discarded:
                self._accept_feishu_candidate(mid)
            for peer, batch in verified.items():
                fresh = [row for row in batch if self._accept_feishu_candidate(row[0])]
                if fresh:
                    self._db.execute("INSERT INTO feishu_raw_batches (peer_open_id, has_media, contents)"
                                     " VALUES (?, ?, ?)", (peer, any(row[2] != "text" for row in fresh),
                                                           json.dumps(fresh)))
            if through is not None:
                self._advance_inbound(_FEISHU_INBOX, through)

    def _accept_feishu_candidate(self, mid: str) -> bool:
        self._db.execute("DELETE FROM feishu_recovery WHERE message_id = ?", (mid,))
        return self._mark_inbound_seen(_FEISHU_INBOX, mid)

    def pending_feishu_users(self) -> list[str]:
        return [row[0] for row in self._db.execute(
            "SELECT DISTINCT peer_open_id FROM feishu_raw_batches")]

    def has_pending_feishu_input(self, peer: str) -> bool:
        """查询飞书尚未确认的原始批次，不读取正文。"""
        return self._db.execute("SELECT 1 FROM feishu_raw_batches WHERE peer_open_id = ? LIMIT 1",
                                (peer,)).fetchone() is not None

    def next_feishu_raw(self, peer: str) -> tuple[int, bool] | None:
        row = self._db.execute("SELECT batch_id, has_media FROM feishu_raw_batches WHERE peer_open_id = ?"
                               " ORDER BY batch_id LIMIT 1", (peer,)).fetchone()
        return None if row is None else (row["batch_id"], bool(row["has_media"]))

    def read_feishu_raw(self, peer: str, batch_id: int) -> list[FeishuRaw]:
        row = self._db.execute("SELECT contents FROM feishu_raw_batches WHERE batch_id = ? AND peer_open_id = ?",
                               (batch_id, peer)).fetchone()
        if row is None:
            raise ValueError("missing Feishu raw batch")
        return [tuple(item) for item in json.loads(row[0])]

    def finish_feishu_raw(self, peer: str, batch_id: int) -> None:
        """确认本用户已整理的原始批次；消息随后直接交调度器。"""
        try:
            with self._input_transaction(retryable_io=False, commit_unknown_on_db_error=True):
                removed = self._db.execute("DELETE FROM feishu_raw_batches WHERE batch_id = ? AND peer_open_id = ?",
                                           (batch_id, peer)).rowcount
                if removed != 1:
                    raise ValueError("missing Feishu raw batch")
        except InputCommitUnknown as exc:
            # 仅在安全回滚之后核验这次删除；不是查询外部去重键，也不重试确认事务。
            try:
                remaining = self._db.execute("SELECT 1 FROM feishu_raw_batches"
                                             " WHERE batch_id = ? AND peer_open_id = ?", (batch_id, peer)).fetchone()
            except sqlite3.ProgrammingError:
                raise  # 共享连接失效不是单用户交接故障。
            except sqlite3.DatabaseError as lookup_error:
                raise FeishuRawConfirmationUnresolved("Feishu raw confirmation could not be verified") from lookup_error
            if remaining is not None:
                raise FeishuRawConfirmationUnresolved("Feishu raw batch remains after commit error") from exc

    def _start_agent_run(self, agent_id: str, run_id: str) -> None:
        self._db.execute("INSERT INTO agent_runs (run_id, agent_id, status, started_at)"
                         " VALUES (?, ?, 'running', ?)", (run_id, agent_id, time()))

    def start_agent_run(self, agent_id: str, run_id: str, *, can_start: Callable[[], bool]) -> bool:
        """拿到写锁后确认唤醒资格，只写运行元数据，不新增对话消息。"""
        with self._input_transaction(retryable_io=False):
            self._db.execute("BEGIN IMMEDIATE")
            if not can_start():
                return False
            self._start_agent_run(agent_id, run_id)
        return True

    def append_agent_inputs(self, agent_id: str, run_id: str,
                            deliveries: Sequence[MessageDelivery], *, start: bool) -> tuple[list[int], bool]:
        """按序原子写入整批，返回消息 ID 及是否创建了运行。"""
        ids: list[int] = []
        started = False
        with self._input_transaction(retryable_io=True):
            self._db.execute("BEGIN IMMEDIATE")
            agent = self.agent(agent_id)
            if agent is None:
                raise ValueError("unknown input agent")
            if not start and not self._agent_run_active(agent_id, run_id):
                raise ValueError("cannot append to an inactive agent run")
            for delivery in deliveries:
                message = delivery.message
                delegation = delivery.delegation
                if start and not started and delivery.intent == "trigger":
                    # 委派有运行外键。先在同一写锁下核对首个候选，再创建运行、消费和写正文。
                    if delegation is not None and not self._db.execute(
                            "SELECT 1 FROM delegations WHERE delegation_id = ? AND agent_id = ? AND status = 'queued'",
                            (delegation.delegation_id, agent_id)).fetchone():
                        continue
                    self._start_agent_run(agent_id, run_id)
                    started = True
                if delegation is not None:
                    if not self._consume_delegation(agent_id, run_id, delegation.delegation_id):
                        continue
                    message = replace(message, source=UserSource("delegation"))
                ids.extend(self._append(agent.session_id, (message,)))
        return ids, started

    def _consume_delegation(self, agent_id: str, run_id: str, delegation_id: str) -> bool:
        cursor = self._db.execute("UPDATE delegations SET run_id = ?, status = 'consumed'"
                                  " WHERE delegation_id = ? AND agent_id = ? AND status = 'queued'",
                                  (run_id, delegation_id, agent_id))
        return cursor.rowcount == 1

    @contextmanager
    def own_reply(self, reply_id: str) -> Iterator[None]:
        """持有到工具和回复均不再写入；此所有权不是工具执行状态。"""
        with self._reply_writers.own(reply_id):
            yield

    def _unsettled_reply(self, session_id: str, reply_id: str) -> HistoryEntry | None:
        row = self._db.execute(f"SELECT {_MESSAGE_COLUMNS} FROM messages m WHERE m.reply_id = ?"
                               f" AND {_UNSETTLED_REPLY}", (reply_id,)).fetchone()
        if row is None:
            return None
        if row["session_id"] != session_id or row["kind"] != "assistant" or not self._reply_writers.owns(reply_id):
            raise ValueError("reply is not owned by this writer")
        return _decode(row)

    def stage_tool_calls(self, session_id: str, reply_id: str, message: Message) -> None:
        """在任何工具开始前记下接受的调用，不写未经确认的正文。"""
        calls = _call_identity(message)
        if (not self._reply_writers.owns(reply_id) or message.kind != "assistant" or not calls
                or len(calls) != len(message.tool_calls) or any(
                    not isinstance(part, (ToolCall, Reasoning)) or isinstance(part, ToolCall) and part.result is not None
                    for part in message.parts)):
            raise ValueError("invalid staged tool calls")
        # 交给 Runtime 在锁冲突时异步重试，不能卡住同事件循环上的语音回执。
        with self._without_busy_wait(), self._input_transaction(retryable_io=False):
            self._db.execute("BEGIN IMMEDIATE")
            existing = self._unsettled_reply(session_id, reply_id)
            if existing is None:
                self._append(session_id, (message,), reply_id=reply_id, staged=True)
            else:
                if not _call_identity(existing.message).items() <= calls.items():
                    raise ValueError("cannot replace a staged tool call")
                self._db.execute("UPDATE messages SET parts = ? WHERE message_id = ?",
                                 (_encode_parts(message.parts), existing.message_id))

    def record_tool_result(self, session_id: str, reply_id: str, call_id: str, output: str) -> None:
        """只补写对应调用的真实文本，包括空字符串。"""
        with self._without_busy_wait(), self._input_transaction(retryable_io=False):
            self._db.execute("BEGIN IMMEDIATE")
            existing = self._unsettled_reply(session_id, reply_id)
            if existing is None or not any(call.id == call_id for call in existing.message.tool_calls):
                raise ValueError("result has no staged tool call")
            self._insert_tool_result(existing.message_id, call_id, output)
            parts = tuple(replace(part, result=output) if isinstance(part, ToolCall) and part.id == call_id else part
                          for part in existing.message.parts)
            self._reindex_message(existing.message_id, Message("assistant", parts))

    def finalize_reply(self, outcome: ReplyOutcome, messages: Sequence[Message], *,
                       model_origin: bytes | None = None) -> list[int]:
        # 已预写调用的回复保留其消息 ID；确认正文与投递结算仍一次提交。
        if len(messages) > 1 or (messages and messages[0].kind != "assistant"):
            raise ValueError("reply history must contain at most one assistant message")
        with self._without_busy_wait(), self._input_transaction(retryable_io=False):
            self._db.execute("BEGIN IMMEDIATE")
            existing = self._unsettled_reply(outcome.session_id, outcome.reply_id)
            if existing is None:
                ids = self._append(outcome.session_id, messages, reply_id=outcome.reply_id)
            else:
                if not messages:
                    raise ValueError("cannot discard staged tool calls")
                message = messages[0]
                stored = {call.id: call.result for call in existing.message.tool_calls}
                if len(stored) != len(message.tool_calls) or _call_identity(message) != _call_identity(existing.message):
                    raise ValueError("cannot change the staged tool calls when settling a reply")
                for call in message.tool_calls:
                    result = stored[call.id]
                    if call.result is None:
                        continue  # 仅结算正文；保留库内结果，包括提交成功但调用方未获确认的结果。
                    if result is None:
                        self._insert_tool_result(existing.message_id, call.id, call.result)
                        stored[call.id] = call.result
                    elif result != call.result:
                        raise ValueError("cannot replace a recorded result")
                message = replace(message, parts=tuple(
                    replace(part, result=stored[part.id]) if isinstance(part, ToolCall) else part
                    for part in message.parts))
                ids = [existing.message_id]
                self._db.execute("UPDATE messages SET parts = ? WHERE message_id = ?",
                                 (_encode_parts(message.parts), existing.message_id))
                self._reindex_message(existing.message_id, message)
            self._db.execute("INSERT INTO reply_outcomes"
                             " (reply_id, session_id, run_id, model_status, delivery_status,"
                             " assistant_message_id, model_error, input_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
                outcome.reply_id, outcome.session_id, outcome.run_id, outcome.model_status,
                outcome.delivery_status, ids[0] if ids else None, outcome.model_error, outcome.input_tokens,
            ))
            if model_origin is not None and outcome.model_status == "complete" and outcome.input_tokens is not None:
                self._db.execute("INSERT INTO reply_model_origins VALUES (?, ?, ?)", (
                    outcome.reply_id, ids[0] if ids else None, model_origin,
                ))
        return ids

    def finish_agent_run(self, run_id: str, *, result: str | None = None, error: str | None = None) -> None:
        """持久化整轮终态；子 Agent 必须有正文或错误，提交成功后才能回投。"""
        if result is not None and error is not None:
            raise ValueError("a run cannot finish with both result and error")
        if result is not None and not isinstance(result, str):
            raise ValueError("a run result must be text")
        if error is not None and not _has_reply_text(error):
            raise ValueError("a run error must be non-blank text")
        has_reply = _has_reply_text(result) or error is not None
        with self._input_transaction(retryable_io=False):
            cursor = self._db.execute("UPDATE agent_runs SET status = ?, finished_at = ?, result = ?, error = ?"
                                      " WHERE run_id = ? AND status = 'running'"
                                      " AND (? OR EXISTS (SELECT 1 FROM agents AS a"
                                      " WHERE a.agent_id = agent_runs.agent_id AND a.parent_agent_id IS NULL))",
                                      ("failed" if error is not None else "completed", time(), result, error, run_id,
                                       has_reply))
            if cursor.rowcount != 1:
                raise ValueError("invalid agent run state transition or missing child reply")

    def interrupt_agent_work(self, *, reason: str, agent_id: str | None = None) -> None:
        """中断单个 Agent 的排队委派及运行，或在应用启动/关闭时收尾全库。"""
        scope = " AND agent_id = ?" if agent_id is not None else ""
        args = (agent_id,) if agent_id is not None else ()
        with self._input_transaction(retryable_io=False):
            self._db.execute("UPDATE delegations SET status = 'interrupted', error = ? WHERE status = 'queued'" + scope,
                             (reason, *args))
            self._db.execute("UPDATE agent_runs SET status = 'interrupted', finished_at = ?, error = ?"
                             " WHERE status = 'running'" + scope, (time(), reason, *args))

    def interrupt_agent_tree(self, root_id: str, *, reason: str) -> None:
        """撤销单条已核验路由时，只收尾这棵树。"""
        tree = """WITH RECURSIVE tree(id) AS (
            SELECT agent_id FROM agents WHERE agent_id = ?
            UNION ALL SELECT a.agent_id FROM agents AS a JOIN tree ON a.parent_agent_id = tree.id
        ) """
        # Python 的 SQLite 不会为以 WITH 开头的 UPDATE 隐式启动事务。
        with self._input_transaction(retryable_io=False):
            self._db.execute("BEGIN IMMEDIATE")
            self._db.execute(tree + "UPDATE delegations SET status = 'interrupted', error = ?"
                             " WHERE status = 'queued' AND agent_id IN tree", (root_id, reason))
            self._db.execute(tree + "UPDATE agent_runs SET status = 'interrupted', finished_at = ?, error = ?"
                             " WHERE status = 'running' AND agent_id IN tree", (root_id, time(), reason))

    def outcomes(self, session_id: str) -> list[ReplyOutcome]:
        return [ReplyOutcome(row["reply_id"], row["session_id"], row["run_id"], row["model_status"],
                             DeliveryStatus(row["delivery_status"]) if row["delivery_status"] else None,
                             row["assistant_message_id"], row["model_error"], row["input_tokens"])
                for row in self._db.execute(
                    "SELECT * FROM reply_outcomes WHERE session_id = ? ORDER BY rowid", (session_id,))]

    def last_input_tokens(self, session_id: str, *, model_origin: bytes | None = None) -> int | None:
        row = self._db.execute(
            "SELECT o.input_tokens, p.fingerprint FROM reply_outcomes o"
            " LEFT JOIN reply_model_origins p ON p.reply_id = o.reply_id"
            " WHERE o.session_id = ? AND o.model_status = 'complete' ORDER BY o.rowid DESC LIMIT 1", (session_id,),
        ).fetchone()
        # 不向前找匹配记录：最新普通请求没有可比较用量时，本轮不据此触发压缩。
        return (row["input_tokens"] if row is not None and
                (model_origin is None or row["fingerprint"] == model_origin) else None)

    def read_after(self, session_id: str, after: int = 0, *,
                   active_replies: Set[str] | None = frozenset()) -> list[HistoryEntry]:
        # 只有 context_snapshot 会判定所有权；普通历史读取返回已经记录的事实。
        rows = self._db.execute(
            f"SELECT {_MESSAGE_COLUMNS}, CASE WHEN {_UNSETTLED_REPLY} THEN m.reply_id END AS pending_id FROM messages m"
            " WHERE m.session_id = ? AND m.message_id > ? ORDER BY m.message_id", (session_id, after))
        entries = []
        for row in rows:
            pending = row["pending_id"]
            entry = _decode(row, private=True)
            entries.append(replace(entry, active_reply=True) if pending is not None and (
                active_replies is None or pending in active_replies) else entry)
        return entries

    def context_snapshot(self, session_id: str) -> tuple[Summary | None, list[HistoryEntry]]:
        """摘要、续尾及不透明协议数据来自同一个 SQLite 读事务。"""
        with self._reply_writers.observe() as active_replies:
            self._db.execute("BEGIN")
            with self._db:
                summary = self.summary(session_id)
                entries = self.read_after(session_id, summary.covers_through if summary else 0,
                                          active_replies=active_replies)
        return summary, entries

    def try_context_snapshot(self, session_id: str) -> tuple[Summary | None, list[HistoryEntry]] | None:
        """压缩 CAS 失利后的单次刷新；遇锁时调用方保留原快照。"""
        with self._without_busy_wait():
            try:
                return self.context_snapshot(session_id)
            except sqlite3.OperationalError as exc:
                if not is_sqlite_lock_error(exc):
                    raise
                return None

    def tail(self, session_id: str, limit: int) -> list[HistoryEntry]:
        if limit < 1:
            raise ValueError("limit must be positive")
        recent = self._db.execute(
            f"SELECT {_MESSAGE_COLUMNS} FROM messages m WHERE session_id = ? ORDER BY message_id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [_decode(row) for row in reversed(recent)]

    def summary(self, session_id: str) -> Summary | None:
        row = self._db.execute("SELECT covers_through, text FROM summaries WHERE session_id = ?",
                               (session_id,)).fetchone()
        return Summary(*row) if row else None

    def save_summary(self, session_id: str, summary: Summary) -> None:
        found = self._db.execute("SELECT 1 FROM messages WHERE session_id = ? AND message_id = ?",
                                 (session_id, summary.covers_through)).fetchone()
        if found is None:
            raise ValueError("summary coverage must point to a message in the same session")
        if not self._write_summary(
            "INSERT INTO summaries VALUES (?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET"
            " covers_through = excluded.covers_through, text = excluded.text"
            " WHERE excluded.covers_through > summaries.covers_through",
            (session_id, summary.covers_through, summary.text),
            session_id, summary.covers_through,
        ):
            raise ValueError("summary coverage must advance without crossing an active reply")

    def try_save_summary(self, session_id: str, summary: Summary, *, covered: int, protected_from: int) -> bool:
        """在发包前单次尝试；锁冲突不等待，也不改变连接的常规锁策略。"""
        if not 0 <= covered < summary.covers_through < protected_from:
            return False
        with self._without_busy_wait():
            return self._write_summary(
                "INSERT INTO summaries (session_id, covers_through, text) SELECT ?, ?, ?"
                " WHERE COALESCE((SELECT covers_through FROM summaries WHERE session_id = ?), 0) = ?"
                " AND EXISTS (SELECT 1 FROM messages WHERE session_id = ? AND message_id = ?)"
                " ON CONFLICT(session_id) DO UPDATE SET covers_through = excluded.covers_through, text = excluded.text",
                (session_id, summary.covers_through, summary.text, session_id, covered,
                 session_id, summary.covers_through),
                session_id, summary.covers_through,
            )

    @contextmanager
    def _without_busy_wait(self) -> Iterator[None]:
        connection = self._db
        timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
        connection.execute("PRAGMA busy_timeout = 0")
        try:
            yield
        finally:
            if self._connection is connection:
                connection.execute(f"PRAGMA busy_timeout = {timeout:d}")

    def _write_summary(self, sql: str, params: tuple[str | int, ...], session_id: str, covers_through: int) -> bool:
        try:
            with self._reply_writers.observe() as active, self._db:
                # 与 CAS 同一个写事务；不能使用候选生成时的在途快照替代此检查。
                self._db.execute("BEGIN IMMEDIATE")
                if active is None or active:
                    encoded = None if active is None else _dump(sorted(active))
                    blocked = self._db.execute(
                        "SELECT 1 FROM messages m WHERE m.session_id = ? AND m.message_id <= ?"
                        f" AND {_UNSETTLED_REPLY}"
                        " AND (? IS NULL OR m.reply_id IN (SELECT value FROM json_each(?))) LIMIT 1",
                        (session_id, covers_through, encoded, encoded),
                    ).fetchone()
                    if blocked:
                        return False
                rowcount = self._db.execute(sql, params).rowcount
            return rowcount == 1
        except sqlite3.DatabaseError as exc:
            code = getattr(exc, "sqlite_errorcode", 0)
            # 只转换写入被拒绝或无法持久化的故障；保留结构错误及覆盖点不变量错误。
            if code == sqlite3.SQLITE_CONSTRAINT_TRIGGER or (code & 0xff) in {
                sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_READONLY,
                sqlite3.SQLITE_IOERR, sqlite3.SQLITE_FULL,
            }:
                raise SummaryWriteError("summary_write_failed") from None
            raise

    def search(self, session_id: str, *, keyword: str | None = None,
               since: float | None = None, until: float | None = None,
               limit: int = 50) -> list[HistoryEntry]:
        if limit < 1:
            raise ValueError("limit must be positive")
        terms = (keyword or "").split()
        join = " JOIN message_search ON message_search.rowid = m.message_id" if terms else ""
        clauses, args = ["m.session_id = ?"], [session_id]
        if terms:
            clauses.append("message_search MATCH ?")
            args.append(" AND ".join('"' + word.replace('"', '""') + '"' for word in terms))
        if since is not None:
            clauses.append("m.created_at >= ?")
            args.append(since)
        if until is not None:
            clauses.append("m.created_at < ?")
            args.append(until)
        args.append(limit)
        sql = f"SELECT {_MESSAGE_COLUMNS} FROM messages AS m" + join + " WHERE " + " AND ".join(clauses)
        return [_decode(row) for row in self._db.execute(sql + " ORDER BY m.message_id DESC LIMIT ?", args)]
