"""SQLite 中的原始会话历史、摘要与独立投递记录。"""

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sqlite3
from time import time

from ..protocol import DeliveryStatus, Image, Message, Text, ToolCall


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
    model_error TEXT
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
CREATE TABLE IF NOT EXISTS feishu_bindings (
    session_id TEXT PRIMARY KEY,
    identity_fingerprint TEXT NOT NULL
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


def _decode(row: sqlite3.Row) -> HistoryEntry:
    return HistoryEntry(
        row["message_id"], row["session_id"], row["created_at"],
        Message(row["role"], _decode_content(row["content"]),
                tuple(ToolCall(**item) for item in json.loads(row["tool_calls"])), row["tool_call_id"], row["reasoning"]),
    )


class FeishuBindingError(ValueError):
    """主会话或飞书身份不匹配，或旧会话归属无法确定。"""


class SummaryWriteError(Exception):
    """摘要写入未提交，可以继续使用已存历史。"""


class History:
    def __init__(self, path: str | Path, *, feishu_binding: tuple[str, str] | None = None) -> None:
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        try:
            self._db.execute("PRAGMA foreign_keys = ON")
            with self._db:
                # BEGIN 放在脚本内，避免 executescript 的隐式提交拆开归属校验和迁移。
                self._db.executescript("BEGIN IMMEDIATE;\n" + _SCHEMA)
                if feishu_binding is not None:
                    self._bind_feishu(*feishu_binding)
                columns = {row["name"] for row in self._db.execute("PRAGMA table_info(reply_outcomes)")}
                if "assistant_message_id" not in columns:
                    self._db.execute("ALTER TABLE reply_outcomes ADD COLUMN"
                                     " assistant_message_id INTEGER REFERENCES messages(message_id)")
                if "model_error" not in columns:
                    self._db.execute("ALTER TABLE reply_outcomes ADD COLUMN model_error TEXT")
        except BaseException:
            self._db.close()
            raise

    def close(self) -> None:
        self._db.close()

    def bind_feishu(self, session_id: str, identity_fingerprint: str) -> None:
        """在已打开的历史库校验唯一主会话。生产入口则与建表、迁移一起完成。"""
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            self._bind_feishu(session_id, identity_fingerprint)

    def _bind_feishu(self, session_id: str, identity_fingerprint: str) -> None:
        existing = self._db.execute("SELECT session_id, identity_fingerprint FROM feishu_bindings LIMIT 2").fetchall()
        if existing:
            if len(existing) != 1 or tuple(existing[0]) != (session_id, identity_fingerprint):
                raise FeishuBindingError("主会话或飞书身份与数据库绑定不一致；请恢复原配置或使用新的数据库路径。")
            return
        used = self._db.execute("""
            SELECT 1 FROM messages
            UNION ALL SELECT 1 FROM summaries
            UNION ALL SELECT 1 FROM reply_outcomes
            UNION ALL SELECT 1 FROM subtask_runs
            UNION ALL SELECT 1 FROM inbound_windows
            UNION ALL SELECT 1 FROM inbound_seen
            UNION ALL SELECT 1 FROM inbound_batches
            UNION ALL SELECT 1 FROM message_search
            LIMIT 1
            """).fetchone()
        if used is not None:
            raise FeishuBindingError("现有数据库有历史但没有飞书绑定；请保留旧数据库并配置新的数据库路径。")
        self._db.execute("INSERT INTO feishu_bindings VALUES (?, ?)", (session_id, identity_fingerprint))

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
                                or message.reasoning is not None):
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

    def finalize_reply(self, outcome: ReplyOutcome, messages: Sequence[Message]) -> None:
        # 原子提交响应及对应结果，不在崩溃后暴露只有调用、没有结果的组。
        if messages and messages[0].role != "assistant":
            raise ValueError("reply history must start with an assistant message")
        with self._db:
            ids = self._append(outcome.session_id, messages)
            self._db.execute("INSERT INTO reply_outcomes"
                             " (reply_id, session_id, run_id, model_status, delivery_status, assistant_message_id, model_error)"
                             " VALUES (?, ?, ?, ?, ?, ?, ?)", (
                outcome.reply_id, outcome.session_id, outcome.run_id, outcome.model_status,
                outcome.delivery_status, ids[0] if ids else None, outcome.model_error,
            ))

    def outcomes(self, session_id: str) -> list[ReplyOutcome]:
        return [ReplyOutcome(row["reply_id"], row["session_id"], row["run_id"], row["model_status"],
                             DeliveryStatus(row["delivery_status"]) if row["delivery_status"] else None,
                             row["assistant_message_id"], row["model_error"])
                for row in self._db.execute(
                    "SELECT * FROM reply_outcomes WHERE session_id = ? ORDER BY rowid", (session_id,))]

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

    def read_after(self, session_id: str, after: int = 0) -> list[HistoryEntry]:
        return [_decode(row) for row in self._db.execute(
            "SELECT * FROM messages WHERE session_id = ? AND message_id > ? ORDER BY message_id",
            (session_id, after))]

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
        try:
            with self._db:
                cursor = self._db.execute(
                    "INSERT INTO summaries VALUES (?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET"
                    " covers_through = excluded.covers_through, text = excluded.text"
                    " WHERE excluded.covers_through > summaries.covers_through",
                    (session_id, summary.covers_through, summary.text),
                )
                if cursor.rowcount != 1:
                    raise ValueError("summary coverage must advance")
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
