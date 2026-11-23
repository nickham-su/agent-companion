"""SQLite 中的原始会话历史、摘要与独立投递记录。"""

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from secrets import token_bytes
import sqlite3
from time import time

from ..protocol import (MAX_REPLAY_BYTES, DeliveryStatus, Image, Message, ProviderReplay, ReplayItem,
                        ReplayScope, Text, ToolCall)


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
CREATE TABLE IF NOT EXISTS subtask_runs (
    run_id TEXT PRIMARY KEY,
    child_session_id TEXT NOT NULL,
    parent_session_id TEXT NOT NULL,
    parent_run_id TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'interrupted')),
    accepted_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    enqueued_at REAL,
    result TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS subtask_parent_child ON subtask_runs(parent_session_id, child_session_id);
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
    contents TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS inbound_pending ON inbound_batches(session_id, source, batch_id);
CREATE TABLE IF NOT EXISTS feishu_private_account (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    owner_fingerprint TEXT NOT NULL,
    started_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS feishu_private_routes (
    session_id TEXT PRIMARY KEY,
    chat_fingerprint TEXT UNIQUE NOT NULL
);
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
class SubtaskRun:
    run_id: str
    child_session_id: str
    parent_session_id: str
    parent_run_id: str
    tool_call_id: str
    status: str
    accepted_at: float
    started_at: float | None
    finished_at: float | None
    enqueued_at: float | None
    result: str | None = field(repr=False)
    error: str | None


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


def _decode(row: sqlite3.Row, replay: ProviderReplay | None = None) -> HistoryEntry:
    return HistoryEntry(
        row["message_id"], row["session_id"], row["created_at"],
        Message(row["role"], _decode_content(row["content"]),
                tuple(ToolCall(**item) for item in json.loads(row["tool_calls"])), row["tool_call_id"], row["reasoning"],
                replay=replay),
    )


class FeishuBindingError(ValueError):
    """飞书账号或会话的数据库归属无法确认。"""


class SummaryWriteError(Exception):
    """摘要写入未提交，可以继续使用已存历史。"""


def is_sqlite_lock_error(exc: sqlite3.OperationalError) -> bool:
    """按 SQLite 主错误码（含扩展码）判定；兼容未携带错误码的合成异常。"""
    code = getattr(exc, "sqlite_errorcode", None)
    if code is not None:
        return isinstance(code, int) and (code & 0xFF) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
    return str(exc) in {"database is locked", "database table is locked", "database schema is locked",
                        "database is busy"}


class History:
    def __init__(self, path: str | Path, *, feishu_private_owner: str | None = None) -> None:
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        try:
            self._db.execute("PRAGMA foreign_keys = ON")
            with self._db:
                # BEGIN 放在脚本内，避免 executescript 的隐式提交拆开归属校验和迁移。
                self._db.executescript("BEGIN IMMEDIATE;\n" + _SCHEMA)
                if feishu_private_owner is not None:
                    # 只预检已有归属；空库须等实际 CLI 账号核验后绑定。
                    self.check_feishu_private(feishu_private_owner)
                columns = {row["name"] for row in self._db.execute("PRAGMA table_info(reply_outcomes)")}
                if "assistant_message_id" not in columns:
                    self._db.execute("ALTER TABLE reply_outcomes ADD COLUMN"
                                     " assistant_message_id INTEGER REFERENCES messages(message_id)")
                if "model_error" not in columns:
                    self._db.execute("ALTER TABLE reply_outcomes ADD COLUMN model_error TEXT")
                if "input_tokens" not in columns:
                    self._db.execute("ALTER TABLE reply_outcomes ADD COLUMN input_tokens INTEGER")
        except BaseException:
            self._db.close()
            raise

    def close(self) -> None:
        self._db.close()

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
            UNION ALL SELECT 1 FROM summaries
            UNION ALL SELECT 1 FROM reply_outcomes
            UNION ALL SELECT 1 FROM subtask_runs
            UNION ALL SELECT 1 FROM inbound_windows
            UNION ALL SELECT 1 FROM inbound_seen
            UNION ALL SELECT 1 FROM inbound_batches
            UNION ALL SELECT 1 FROM message_search
            UNION ALL SELECT 1 FROM feishu_private_routes
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
                raise FeishuBindingError("飞书服务账号与数据库绑定不一致；请恢复原配置或使用新的数据库路径。")
            return row["started_ms"]
        if self._has_history():
            raise FeishuBindingError("现有数据库有历史但没有飞书账号绑定；请保留旧数据库并配置新的数据库路径。")
        return None

    def bind_feishu_private_route(self, session_id: str, chat_fingerprint: str) -> bool:
        """已见过的发送者和私聊之间必须仍是原来的一对一关系，禁止重绑。"""
        with self._db:
            self._db.execute("INSERT INTO feishu_private_routes VALUES (?, ?) ON CONFLICT DO NOTHING",
                             (session_id, chat_fingerprint))
            return self._db.execute("SELECT 1 FROM feishu_private_routes WHERE session_id = ? AND chat_fingerprint = ?",
                                    (session_id, chat_fingerprint)).fetchone() is not None

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

    def append(self, session_id: str, messages: Iterable[Message]) -> list[int]:
        with self._db:
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
        recorded: set[str] = set()
        contents: list[str] = []
        key = (session_id, source)
        with self._db:
            for external_id, message in items:
                cursor = self._db.execute("INSERT INTO inbound_seen VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                                          (*key, external_id))
                if cursor.rowcount == 1:
                    recorded.add(external_id)
                    if message is not None:
                        if (message.role != "user" or message.tool_calls or message.tool_call_id is not None
                                or message.reasoning is not None or message.replay is not None):
                            raise ValueError("inbound batches require plain user messages")
                        contents.append(_encode(message)[0])
            if contents:
                self._db.execute("INSERT INTO inbound_batches (session_id, source, contents) VALUES (?, ?, ?)",
                                 (*key, json.dumps(contents, ensure_ascii=False)))
            self._db.execute("UPDATE inbound_windows SET scanned_until = max(scanned_until, ?)"
                             " WHERE session_id = ? AND source = ?", (through, *key))
        return recorded

    def pending_inbound(self, session_id: str, source: str) -> list[int]:
        rows = self._db.execute("SELECT batch_id FROM inbound_batches WHERE session_id = ? AND source = ?"
                                " ORDER BY batch_id", (session_id, source))
        return [row["batch_id"] for row in rows]

    def consume_inbound(self, session_id: str, batch_id: int) -> list[int]:
        """原子移交：提交原始用户历史与移出待处理批次不可分开。"""
        with self._db:
            key = (session_id, batch_id)
            row = self._db.execute("SELECT contents FROM inbound_batches WHERE session_id = ? AND batch_id = ?",
                                   key).fetchone()
            if row is None:
                return []
            if self._db.execute("DELETE FROM inbound_batches WHERE session_id = ? AND batch_id = ?", key).rowcount != 1:
                return []
            return self._append(session_id, (Message("user", _decode_content(content))
                                             for content in json.loads(row["contents"])))

    def finalize_reply(self, outcome: ReplyOutcome, messages: Sequence[Message]) -> list[int]:
        # 原子提交响应及对应结果，不在崩溃后暴露只有调用、没有结果的组。
        if messages and messages[0].role != "assistant":
            raise ValueError("reply history must start with an assistant message")
        with self._db:
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
        return ids

    def outcomes(self, session_id: str) -> list[ReplyOutcome]:
        return [ReplyOutcome(row["reply_id"], row["session_id"], row["run_id"], row["model_status"],
                             DeliveryStatus(row["delivery_status"]) if row["delivery_status"] else None,
                             row["assistant_message_id"], row["model_error"], row["input_tokens"])
                for row in self._db.execute(
                    "SELECT * FROM reply_outcomes WHERE session_id = ? ORDER BY rowid", (session_id,))]

    def last_input_tokens(self, session_id: str) -> int | None:
        row = self._db.execute(
            "SELECT input_tokens FROM reply_outcomes WHERE session_id = ? AND model_status = 'complete'"
            " ORDER BY rowid DESC LIMIT 1", (session_id,),
        ).fetchone()
        return row["input_tokens"] if row is not None else None

    def run_text(self, session_id: str, run_id: str) -> str | None:
        row = self._db.execute(
            "SELECT m.content FROM reply_outcomes AS r JOIN messages AS m"
            " ON m.message_id = r.assistant_message_id"
            " WHERE r.session_id = ? AND r.run_id = ? AND r.model_status = 'complete'"
            " ORDER BY r.rowid DESC LIMIT 1", (session_id, run_id),
        ).fetchone()
        if row is None:
            return None
        content = json.loads(row[0])
        return content if isinstance(content, str) and content.strip() else None

    def can_resume_subtask(self, parent_session_id: str, child_session_id: str) -> bool:
        return self._db.execute(
            "SELECT 1 FROM subtask_runs WHERE parent_session_id = ? AND child_session_id = ? LIMIT 1",
            (parent_session_id, child_session_id),
        ).fetchone() is not None

    def accept_subtask(self, child_session_id: str, run_id: str, parent_session_id: str,
                       parent_run_id: str, tool_call_id: str) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO subtask_runs (child_session_id, run_id, parent_session_id, parent_run_id,"
                " tool_call_id, status, accepted_at) VALUES (?, ?, ?, ?, ?, 'queued', ?)",
                (child_session_id, run_id, parent_session_id, parent_run_id, tool_call_id, time()),
            )

    def start_subtask(self, run_id: str) -> None:
        self._update_subtask(
            "UPDATE subtask_runs SET status = 'running', started_at = ? WHERE run_id = ? AND status = 'queued'",
            (time(), run_id),
        )

    def finish_subtask(self, run_id: str, *, result: str | None, error: str | None) -> None:
        if (result is None) == (error is None):
            raise ValueError("subtask requires exactly one result or error")
        self._update_subtask(
            "UPDATE subtask_runs SET status = ?, finished_at = ?, result = ?, error = ?"
            " WHERE run_id = ? AND status = 'running'",
            ("failed" if error is not None else "completed", time(), result, error, run_id),
        )

    def mark_subtask_enqueued(self, run_id: str) -> None:
        self._update_subtask(
            "UPDATE subtask_runs SET enqueued_at = ? WHERE run_id = ? AND enqueued_at IS NULL"
            " AND status IN ('completed', 'failed')", (time(), run_id),
        )

    def interrupt_subtasks(self, parent_session_id: str, *, reason: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE subtask_runs SET status = 'interrupted', finished_at = ?, error = ?"
                " WHERE parent_session_id = ? AND status IN ('queued', 'running')",
                (time(), reason, parent_session_id),
            )

    def subtask_run(self, child_session_id: str, run_id: str) -> SubtaskRun | None:
        row = self._db.execute("SELECT * FROM subtask_runs WHERE child_session_id = ? AND run_id = ?",
                               (child_session_id, run_id)).fetchone()
        return SubtaskRun(**dict(row)) if row is not None else None

    def _update_subtask(self, sql: str, params: tuple) -> None:
        with self._db:
            if self._db.execute(sql, params).rowcount != 1:
                raise ValueError("invalid subtask state transition")

    def read_after(self, session_id: str, after: int = 0, *,
                   replay_scope: ReplayScope | None = None) -> list[HistoryEntry]:
        if replay_scope is not None:
            return [_decode(row, _decode_replay(row, replay_scope)) for row in self._db.execute(
                "SELECT m.*,"
                " CASE WHEN length(CAST(r.items AS BLOB)) <= ? THEN CAST(r.items AS BLOB) END AS replay_items"
                " FROM messages m LEFT JOIN model_replays r ON r.assistant_message_id = m.message_id"
                " AND r.protocol = ? AND r.fingerprint = ? AND r.reasoning_effort = ?"
                " WHERE m.session_id = ? AND m.message_id > ? ORDER BY m.message_id",
                (MAX_REPLAY_BYTES, replay_scope.protocol, replay_scope.fingerprint, replay_scope.reasoning_effort,
                 session_id, after))]
        return [_decode(row) for row in self._db.execute(
            "SELECT * FROM messages WHERE session_id = ? AND message_id > ? ORDER BY message_id",
            (session_id, after))]

    def context_snapshot(
        self, session_id: str, *, replay_scope: ReplayScope | None = None,
    ) -> tuple[Summary | None, list[HistoryEntry]]:
        """摘要、续尾及私有回放来自同一个 SQLite 读事务。"""
        self._db.execute("BEGIN")
        with self._db:
            summary = self.summary(session_id)
            entries = self.read_after(session_id, summary.covers_through if summary else 0, replay_scope=replay_scope)
        return summary, entries

    def try_context_snapshot(
        self, session_id: str, *, replay_scope: ReplayScope | None = None,
    ) -> tuple[Summary | None, list[HistoryEntry]] | None:
        """压缩 CAS 失利后的单次刷新；遇锁时调用方保留原快照。"""
        with self._without_busy_wait():
            try:
                return self.context_snapshot(session_id, replay_scope=replay_scope)
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
