"""SQLite 中的 Agent、会话历史、摘要和投递记录。"""

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from secrets import token_bytes
import sqlite3
from time import time
from uuid import uuid4

from ..protocol import (MAX_REPLAY_BYTES, DeliveryStatus, Image, Message, ProviderReplay, ReplayItem,
                        ReplayScope, Text, ToolCall)


_SCHEMA_VERSION = 3
_FEISHU_INBOX = ("feishu-inbox", "feishu-private")
_FEISHU_INITIAL_RETRY_SECONDS = 10
_FEISHU_MAX_RETRY_SECONDS = 900
# v1 的可选扩展；旧库不需要回填，旧程序也可继续写入没有来源的记录。
_MODEL_ORIGINS = """
CREATE TABLE IF NOT EXISTS reply_model_origins (
    reply_id TEXT PRIMARY KEY REFERENCES reply_outcomes(reply_id) ON DELETE CASCADE,
    assistant_message_id INTEGER UNIQUE REFERENCES messages(message_id) ON DELETE CASCADE,
    fingerprint BLOB NOT NULL CHECK (length(fingerprint) = 32)
)
"""
_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_calls TEXT NOT NULL,
    tool_call_id TEXT,
    reasoning TEXT
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
CREATE TABLE IF NOT EXISTS model_replays (
    assistant_message_id INTEGER PRIMARY KEY REFERENCES messages(message_id) ON DELETE CASCADE,
    protocol TEXT NOT NULL,
    fingerprint BLOB NOT NULL CHECK (length(fingerprint) = 32),
    reasoning_effort TEXT NOT NULL,
    items TEXT NOT NULL
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
CREATE TABLE IF NOT EXISTS inbound_batches (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    source TEXT NOT NULL,
    contents TEXT NOT NULL,
    peer_open_id TEXT
);
CREATE INDEX IF NOT EXISTS inbound_pending ON inbound_batches(session_id, source, batch_id);
CREATE INDEX IF NOT EXISTS inbound_feishu_peers ON inbound_batches(peer_open_id) WHERE peer_open_id IS NOT NULL;
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
    reply_enqueued_at REAL,
    reply_consumed_run_id TEXT REFERENCES agent_runs(run_id),
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
    reply_enqueued_at: float | None
    reply_consumed_run_id: str | None


@dataclass(frozen=True, slots=True)
class DelegatedInput:
    delegation_id: str
    message: Message = field(repr=False)


@dataclass(frozen=True, slots=True)
class AgentReplyInput:
    source_agent_id: str
    source_run_id: str
    message: Message = field(repr=False)


def _has_reply_text(value: str | None) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _encode(message: Message) -> tuple[str, str]:
    if isinstance(message.content, str):
        return json.dumps(message.content, ensure_ascii=False), message.content
    parts, searchable = [], []
    for part in message.content:
        if isinstance(part, Text):
            parts.append({"text": part.text})
            searchable.append(part.text)
        else:
            parts.append({"image": part.url})
    return json.dumps(parts, ensure_ascii=False), "\n".join(searchable)


def _decode_content(encoded: str) -> str | tuple[Text | Image, ...]:
    value = json.loads(encoded)
    return value if isinstance(value, str) else tuple(
        Text(item["text"]) if "text" in item else Image(item["image"]) for item in value
    )


def _decode_replay(row: sqlite3.Row, scope: ReplayScope) -> ProviderReplay | None:
    if row["replay_items"] is None:
        return None
    try:
        items = json.loads(row["replay_items"].decode("utf-8"))
        if not isinstance(items, list):
            raise ValueError
        parsed = []
        for item in items:
            if (type(item) is not dict or type(item["parts"]) is not list or type(item["summary"]) is not list
                    or any(type(part) is not list for part in item["parts"])):
                raise ValueError
            cipher, summary = item["encrypted_content"], tuple(item["summary"])
            # JSON 转义可以藏有孤立代理码点；此类记录不能被写入 UTF-8 请求。
            for text in ((cipher,) if cipher is not None else ()) + summary:
                if not isinstance(text, str):
                    raise ValueError
                text.encode("utf-8")
            parsed.append(ReplayItem(
                kind=item["kind"], item_id=item["item_id"], call_id=item["call_id"], phase=item["phase"],
                parts=tuple(tuple(part) for part in item["parts"]), encrypted_content=cipher, summary=summary,
            ))
        return ProviderReplay(scope, tuple(parsed))
    except (ValueError, TypeError, KeyError, RecursionError):
        return None  # 私有记录损坏时，仍可正常读取通用会话历史。


def _decode(row: sqlite3.Row, replay: ProviderReplay | None = None, *, include_reasoning: bool = True) -> HistoryEntry:
    return HistoryEntry(
        row["message_id"], row["session_id"], row["created_at"],
        Message(row["role"], _decode_content(row["content"]),
                tuple(ToolCall(**item) for item in json.loads(row["tool_calls"])), row["tool_call_id"],
                row["reasoning"] if include_reasoning else None, replay=replay),
    )


class FeishuBindingError(ValueError):
    """飞书账号或会话的数据库归属无法确认。"""


class DatabaseFormatError(ValueError):
    """数据库不是空库或当前支持的版本。"""


class SummaryWriteError(Exception):
    """摘要写入未提交，可以继续使用已存历史。"""


class RetryableInputWrite(Exception):
    """输入写语句遇到 IOERR，且提交前已成功回滚。"""


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
            UNION ALL SELECT 1 FROM inbound_batches
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
            return self._ensure_main_agent(session_id)

    def _ensure_main_agent(self, session_id: str) -> Agent:
        self._db.execute("INSERT INTO agents (agent_id, session_id, role_id) VALUES (?, ?, 'main')"
                         " ON CONFLICT(session_id) DO NOTHING", (f"agent-{uuid4().hex}", session_id))
        row = self._db.execute("SELECT * FROM agents WHERE session_id = ?", (session_id,)).fetchone()
        if row["parent_agent_id"] is not None:
            raise ValueError("session belongs to a child agent")
        return Agent(**dict(row))

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
                child = Agent(f"agent-{uuid4().hex}", f"child-{uuid4().hex}", role_id, parent_agent_id)
                self._db.execute("INSERT INTO agents VALUES (?, ?, ?, ?)",
                                 (child.agent_id, child.session_id, child.role_id, child.parent_agent_id))
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

    def _append(self, session_id: str, messages: Iterable[Message]) -> list[int]:
        created_at = time()
        ids = []
        for message in messages:
            content, searchable = _encode(message)
            cursor = self._db.execute(
                "INSERT INTO messages (session_id, created_at, role, content, tool_calls, tool_call_id, reasoning)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, created_at, message.role, content,
                 json.dumps([asdict(call) for call in message.tool_calls], ensure_ascii=False),
                 message.tool_call_id, message.reasoning),
            )
            message_id = cursor.lastrowid
            if searchable:
                self._db.execute("INSERT INTO message_search (rowid, body) VALUES (?, ?)",
                                 (message_id, searchable))
            ids.append(message_id)
        return ids

    @contextmanager
    def _input_transaction(self, *, retryable_io: bool) -> Iterator[None]:
        """区分写语句与提交阶段；不能重试提交结果未知的 IOERR。"""
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
                self._rollback_input(connection, exc)
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

    def record_inbound(self, session_id: str, source: str, through: int,
                       items: Sequence[tuple[str, Message | None]]) -> set[str]:
        """同一事务记录水位、去重键和待接收的批次；None 表示已处理但没有输入。"""
        key = (session_id, source)
        with self._db:
            recorded = self._record_inbound_items(key, session_id, items)
            self._advance_inbound(key, through)
        return recorded

    def _mark_inbound_seen(self, key: tuple[str, str], message_id: str) -> bool:
        return self._db.execute("INSERT INTO inbound_seen VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                                (*key, message_id)).rowcount == 1

    def _record_inbound_items(self, seen_key: tuple[str, str], session_id: str,
                              items: Sequence[tuple[str, Message | None]]) -> set[str]:
        recorded: set[str] = set()
        messages = []
        for external_id, message in items:
            if self._mark_inbound_seen(seen_key, external_id):
                recorded.add(external_id)
                if message is not None:
                    messages.append(message)
        self._insert_inbound_batch(session_id, seen_key[1], messages)
        return recorded

    def _insert_inbound_batch(self, session_id: str, source: str, messages: Sequence[Message], *,
                              peer_open_id: str | None = None) -> None:
        contents = []
        for message in messages:
            if (message.role != "user" or message.tool_calls or message.tool_call_id is not None
                    or message.reasoning is not None or message.replay is not None):
                raise ValueError("inbound batches require plain user messages")
            contents.append(_encode(message)[0])
        if contents:
            self._db.execute("INSERT INTO inbound_batches (session_id, source, contents, peer_open_id)"
                             " VALUES (?, ?, ?, ?)", (session_id, source, json.dumps(contents, ensure_ascii=False), peer_open_id))

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
            "SELECT peer_open_id FROM feishu_raw_batches UNION SELECT peer_open_id FROM inbound_batches"
            " WHERE source = ? AND peer_open_id IS NOT NULL", (_FEISHU_INBOX[1],))]

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

    def finish_feishu_raw(self, peer: str, batch_id: int, session_id: str, messages: Sequence[Message]) -> None:
        with self._input_transaction(retryable_io=False):
            removed = self._db.execute("DELETE FROM feishu_raw_batches WHERE batch_id = ? AND peer_open_id = ?",
                                       (batch_id, peer)).rowcount
            if removed != 1:
                raise ValueError("missing Feishu raw batch")
            self._insert_inbound_batch(session_id, _FEISHU_INBOX[1], messages, peer_open_id=peer)

    def pending_inbound(self, session_id: str, source: str) -> list[int]:
        rows = self._db.execute("SELECT batch_id FROM inbound_batches WHERE session_id = ? AND source = ?"
                                " ORDER BY batch_id", (session_id, source))
        return [row["batch_id"] for row in rows]

    def consume_inbound(self, session_id: str, batch_id: int) -> list[int]:
        """原子移交：提交原始用户历史与移出待处理批次不可分开。"""
        with self._input_transaction(retryable_io=True):
            return self._consume_inbound(session_id, batch_id)

    def append_with_inbound(self, session_id: str, items: Sequence[Message | int]) -> tuple[list[int], bool]:
        """按给定顺序写入内存消息与飞书批次，返回消息 ID 和是否确实消费了持久化消息。"""
        ids = []
        consumed = False
        with self._input_transaction(retryable_io=True):
            for item in items:
                if isinstance(item, int):
                    inbound_ids = self._consume_inbound(session_id, item)
                    ids.extend(inbound_ids)
                    consumed = consumed or bool(inbound_ids)
                else:
                    ids.extend(self._append(session_id, (item,)))
        return ids, consumed

    def append_agent_inputs(self, agent_id: str, run_id: str,
                            items: Sequence[Message | int | DelegatedInput | AgentReplyInput], *,
                            start: bool) -> tuple[list[int], bool, bool]:
        """按顺序写输入；返回消息 ID、是否消费飞书输入、是否消费可信 Agent 输入。"""
        ids: list[int] = []
        consumed_inbound = consumed_agent = False
        with self._input_transaction(retryable_io=True):
            self._db.execute("BEGIN IMMEDIATE")
            agent = self.agent(agent_id)
            if agent is None:
                raise ValueError("unknown input agent")
            if start:
                self._db.execute("INSERT INTO agent_runs (run_id, agent_id, status, started_at)"
                                 " VALUES (?, ?, 'running', ?)", (run_id, agent_id, time()))
            elif not self._agent_run_active(agent_id, run_id):
                raise ValueError("cannot append to an inactive agent run")
            for item in items:
                if isinstance(item, int):
                    inbound_ids = self._consume_inbound(agent.session_id, item)
                    ids.extend(inbound_ids)
                    consumed_inbound = consumed_inbound or bool(inbound_ids)
                elif isinstance(item, (DelegatedInput, AgentReplyInput)):
                    if self._consume_agent_input(agent_id, run_id, item):
                        ids.extend(self._append(agent.session_id, (item.message,)))
                        consumed_agent = True
                else:
                    ids.extend(self._append(agent.session_id, (item,)))
            if start and not ids:
                # 作废的批次或重复回投不产生运行；删除仍在同一事务内。
                self._db.execute("DELETE FROM agent_runs WHERE run_id = ?", (run_id,))
        return ids, consumed_inbound, consumed_agent

    def _consume_agent_input(self, agent_id: str, run_id: str, item: DelegatedInput | AgentReplyInput) -> bool:
        if isinstance(item, DelegatedInput):
            cursor = self._db.execute("UPDATE delegations SET run_id = ?, status = 'consumed'"
                                      " WHERE delegation_id = ? AND agent_id = ? AND status = 'queued'",
                                      (run_id, item.delegation_id, agent_id))
        else:
            source = self._db.execute("""
                SELECT r.status, r.result, r.error FROM agent_runs AS r
                JOIN agents AS a ON a.agent_id = r.agent_id
                WHERE r.agent_id = ? AND r.run_id = ? AND a.parent_agent_id = ?
                  AND r.status IN ('completed', 'failed') AND r.reply_consumed_run_id IS NULL
                """, (item.source_agent_id, item.source_run_id, agent_id)).fetchone()
            if source is None:
                return False
            result, error = source["result"], source["error"]
            if source["status"] == "completed":
                valid = error is None and _has_reply_text(result)
            else:
                valid = result is None and _has_reply_text(error)
            if not valid:
                return False
            # 调用方已持有写事务；校验、去重标记与消息写入不会被拆开。
            cursor = self._db.execute("UPDATE agent_runs SET reply_consumed_run_id = ? WHERE run_id = ?",
                                      (run_id, item.source_run_id))
        return cursor.rowcount == 1

    def _consume_inbound(self, session_id: str, batch_id: int) -> list[int]:
        key = (session_id, batch_id)
        row = self._db.execute("SELECT contents FROM inbound_batches WHERE session_id = ? AND batch_id = ?",
                               key).fetchone()
        if row is None:
            return []
        if self._db.execute("DELETE FROM inbound_batches WHERE session_id = ? AND batch_id = ?", key).rowcount != 1:
            return []
        return self._append(session_id, (Message("user", _decode_content(content))
                                         for content in json.loads(row["contents"])))

    def finalize_reply(self, outcome: ReplyOutcome, messages: Sequence[Message], *,
                       model_origin: bytes | None = None) -> list[int]:
        # 原子提交响应及对应结果，不在崩溃后暴露只有调用、没有结果的组。
        if messages and messages[0].role != "assistant":
            raise ValueError("reply history must start with an assistant message")
        with self._input_transaction(retryable_io=False):
            ids = self._append(outcome.session_id, messages)
            replay = messages[0].replay if messages else None
            if replay is not None and outcome.model_status == "complete":
                items = json.dumps([asdict(item) for item in replay.items], ensure_ascii=False, separators=(",", ":"))
                if len(items.encode("utf-8")) > MAX_REPLAY_BYTES:
                    raise ValueError("provider replay exceeds storage limit")
                scope = replay.scope
                self._db.execute("INSERT INTO model_replays VALUES (?, ?, ?, ?, ?)", (
                    ids[0], scope.protocol, scope.fingerprint, scope.reasoning_effort, items,
                ))
            self._db.execute("INSERT INTO reply_outcomes"
                             " (reply_id, session_id, run_id, model_status, delivery_status,"
                             " assistant_message_id, model_error, input_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
                outcome.reply_id, outcome.session_id, outcome.run_id, outcome.model_status,
                outcome.delivery_status, ids[0] if ids else None, outcome.model_error, outcome.input_tokens,
            ))
            if model_origin is not None and outcome.model_status == "complete" and (
                    outcome.input_tokens is not None or bool(messages and messages[0].reasoning is not None)):
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

    def mark_agent_reply_enqueued(self, run_id: str) -> None:
        """记录已回投，不参与消费判断；父级可能先消费才写入此诊断。"""
        with self._input_transaction(retryable_io=False):
            self._db.execute("UPDATE agent_runs SET reply_enqueued_at = ? WHERE run_id = ?"
                             " AND status IN ('completed', 'failed') AND reply_enqueued_at IS NULL", (time(), run_id))

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
                   replay_scope: ReplayScope | None = None, model_origin: bytes | None = None) -> list[HistoryEntry]:
        columns, joins, select_args, join_args = ["m.*"], [], [], []
        if replay_scope is not None:
            columns.append("CASE WHEN length(CAST(r.items AS BLOB)) <= ? THEN CAST(r.items AS BLOB) END AS replay_items")
            select_args.append(MAX_REPLAY_BYTES)
            joins.append("LEFT JOIN model_replays r ON r.assistant_message_id = m.message_id"
                         " AND r.protocol = ? AND r.fingerprint = ? AND r.reasoning_effort = ?")
            join_args.extend((replay_scope.protocol, replay_scope.fingerprint, replay_scope.reasoning_effort))
        if model_origin is not None:
            columns.append("p.fingerprint IS NOT NULL AS matching_origin")
            joins.append("LEFT JOIN reply_model_origins p ON p.assistant_message_id = m.message_id AND p.fingerprint = ?")
            join_args.append(model_origin)
        sql = (f"SELECT {', '.join(columns)} FROM messages m {' '.join(joins)}"
               " WHERE m.session_id = ? AND m.message_id > ? ORDER BY m.message_id")
        return [_decode(row, _decode_replay(row, replay_scope) if replay_scope is not None else None,
                        include_reasoning=model_origin is None or bool(row["matching_origin"]))
                for row in self._db.execute(sql, (*select_args, *join_args, session_id, after))]

    def context_snapshot(
        self, session_id: str, *, replay_scope: ReplayScope | None = None, model_origin: bytes | None = None,
    ) -> tuple[Summary | None, list[HistoryEntry]]:
        """摘要、续尾及私有回放来自同一个 SQLite 读事务。"""
        self._db.execute("BEGIN")
        with self._db:
            summary = self.summary(session_id)
            options = {"model_origin": model_origin} if model_origin is not None else {}
            entries = self.read_after(session_id, summary.covers_through if summary else 0,
                                      replay_scope=replay_scope, **options)
        return summary, entries

    def try_context_snapshot(
        self, session_id: str, *, replay_scope: ReplayScope | None = None, model_origin: bytes | None = None,
    ) -> tuple[Summary | None, list[HistoryEntry]] | None:
        """压缩 CAS 失利后的单次刷新；遇锁时调用方保留原快照。"""
        with self._without_busy_wait():
            try:
                options = {"model_origin": model_origin} if model_origin is not None else {}
                return self.context_snapshot(session_id, replay_scope=replay_scope, **options)
            except sqlite3.OperationalError as exc:
                if not is_sqlite_lock_error(exc):
                    raise
                return None

    def tail(self, session_id: str, limit: int) -> list[HistoryEntry]:
        if limit < 1:
            raise ValueError("limit must be positive")
        recent = self._db.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY message_id DESC LIMIT ?",
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
        ):
            raise ValueError("summary coverage must advance")

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
            )

    @contextmanager
    def _without_busy_wait(self) -> Iterator[None]:
        timeout = self._db.execute("PRAGMA busy_timeout").fetchone()[0]
        self._db.execute("PRAGMA busy_timeout = 0")
        try:
            yield
        finally:
            self._db.execute(f"PRAGMA busy_timeout = {timeout:d}")

    def _write_summary(self, sql: str, params: tuple[str | int, ...]) -> bool:
        try:
            with self._db:
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
        sql = "SELECT m.* FROM messages AS m" + join + " WHERE " + " AND ".join(clauses)
        return [_decode(row) for row in self._db.execute(sql + " ORDER BY m.message_id DESC LIMIT ?", args)]
