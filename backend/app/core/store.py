"""数据集 / 会话 / 报告 / 图表产物的 SQLite 元数据存储。

每次调用独立开连接 + WAL，天然线程安全（FastAPI 线程池里多请求并发也不打架）。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from ..config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    filename      TEXT NOT NULL,
    original_path TEXT NOT NULL,
    working_path  TEXT NOT NULL,
    sheet         TEXT,
    encoding      TEXT,
    rows          INTEGER DEFAULT 0,
    cols          INTEGER DEFAULT 0,
    size_bytes    INTEGER DEFAULT 0,
    revision      INTEGER DEFAULT 0,
    profile       TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    dataset_id  TEXT NOT NULL,
    title       TEXT,
    turns       INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id              TEXT PRIMARY KEY,
    dataset_id      TEXT NOT NULL,
    conversation_id TEXT,
    title           TEXT,
    markdown        TEXT,
    html            TEXT,
    markdown_path   TEXT,
    html_path       TEXT,
    payload         TEXT,
    generated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    id              TEXT PRIMARY KEY,
    dataset_id      TEXT NOT NULL,
    conversation_id TEXT,
    kind            TEXT,
    title           TEXT,
    description     TEXT,
    file_path       TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_dataset ON conversations(dataset_id);
CREATE INDEX IF NOT EXISTS idx_reports_dataset ON reports(dataset_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_dataset ON artifacts(dataset_id);
"""


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _decode_dataset_row(row: dict[str, Any]) -> dict[str, Any]:
    """把 datasets 表里的 profile 从 JSON 字符串还原成 dict。"""

    raw = row.get("profile")
    if isinstance(raw, str):
        try:
            row["profile"] = json.loads(raw)
        except json.JSONDecodeError:
            row["profile"] = {}
    elif raw is None:
        row["profile"] = {}
    return row


class Store:
    """轻量元数据仓库。"""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else get_settings().meta_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.init_schema()

    # ---------------- 基础设施 ----------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _exec(self, sql: str, params: tuple = ()) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(sql, params)

    def _query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def _query_one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    # ---------------- 数据集 ----------------
    def save_dataset(
        self,
        *,
        dataset_id: str,
        name: str,
        filename: str,
        original_path: Path,
        working_path: Path,
        sheet: str | None,
        encoding: str | None,
        rows: int,
        cols: int,
        size_bytes: int,
        profile: dict[str, Any],
    ) -> None:
        ts = now_iso()
        self._exec(
            """
            INSERT INTO datasets (id, name, filename, original_path, working_path, sheet,
                                  encoding, rows, cols, size_bytes, revision, profile,
                                  created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                rows=excluded.rows, cols=excluded.cols, size_bytes=excluded.size_bytes,
                profile=excluded.profile, updated_at=excluded.updated_at
            """,
            (
                dataset_id, name, filename, str(original_path), str(working_path), sheet,
                encoding, rows, cols, size_bytes,
                json.dumps(profile, ensure_ascii=False), ts, ts,
            ),
        )

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        row = self._query_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        return _decode_dataset_row(row) if row else None

    def list_datasets(self) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM datasets ORDER BY created_at DESC")
        return [_decode_dataset_row(r) for r in rows]

    def update_working_copy(
        self,
        dataset_id: str,
        *,
        working_path: Path,
        rows: int,
        cols: int,
        profile: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE datasets
                   SET working_path = ?, rows = ?, cols = ?,
                       revision = revision + 1, updated_at = ?
                 WHERE id = ?
                """,
                (str(working_path), rows, cols, now_iso(), dataset_id),
            )
            if profile is not None:
                conn.execute(
                    "UPDATE datasets SET profile = ? WHERE id = ?",
                    (json.dumps(profile, ensure_ascii=False), dataset_id),
                )

    def delete_dataset(self, dataset_id: str) -> None:
        with self._lock, self._connect() as conn:
            for table in ("datasets", "conversations", "reports", "artifacts"):
                conn.execute(f"DELETE FROM {table} WHERE {'id' if table == 'datasets' else 'dataset_id'} = ?", (dataset_id,))

    # ---------------- 会话 ----------------
    def ensure_conversation(self, conversation_id: str, dataset_id: str, title: str | None = None) -> str:
        ts = now_iso()
        self._exec(
            """
            INSERT INTO conversations (id, dataset_id, title, turns, created_at, updated_at)
            VALUES (?,?,?,0,?,?)
            ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (conversation_id, dataset_id, title, ts, ts),
        )
        return conversation_id

    def touch_conversation(self, conversation_id: str) -> None:
        self._exec(
            "UPDATE conversations SET turns = turns + 1, updated_at = ? WHERE id = ?",
            (now_iso(), conversation_id),
        )

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        return self._query_one("SELECT * FROM conversations WHERE id = ?", (conversation_id,))

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        """改会话标题，返回是否命中记录。"""

        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now_iso(), conversation_id),
            )
            return cur.rowcount > 0

    def purge_conversation(self, conversation_id: str) -> dict[str, Any]:
        """删除会话及其报告、图表记录，返回待清理的文件路径。

        文件本身交给调用方在确认路径安全后删除。
        """

        with self._lock, self._connect() as conn:
            artifacts = [
                dict(r)
                for r in conn.execute(
                    "SELECT id, file_path FROM artifacts WHERE conversation_id = ?",
                    (conversation_id,),
                )
            ]
            reports = [
                dict(r)
                for r in conn.execute(
                    "SELECT id, markdown_path, html_path FROM reports WHERE conversation_id = ?",
                    (conversation_id,),
                )
            ]
            conn.execute("DELETE FROM artifacts WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM reports WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

        files: list[str] = [r["file_path"] for r in artifacts if r.get("file_path")]
        for r in reports:
            files.extend(p for p in (r.get("markdown_path"), r.get("html_path")) if p)
        return {"artifacts": len(artifacts), "reports": len(reports), "files": files}

    def list_conversations(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        # 同一秒内创建的两个会话时间戳相同，SQLite 的排序会变得不确定；
        # 追加 rowid 作为第二排序键，保证"后建的排前面"。
        if dataset_id:
            return self._query(
                "SELECT * FROM conversations WHERE dataset_id = ?"
                " ORDER BY updated_at DESC, rowid DESC",
                (dataset_id,),
            )
        return self._query(
            "SELECT * FROM conversations ORDER BY updated_at DESC, rowid DESC"
        )

    def delete_conversation(self, conversation_id: str) -> None:
        self._exec("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    # ---------------- 报告 ----------------
    def save_report(
        self,
        *,
        report_id: str,
        dataset_id: str,
        conversation_id: str | None,
        title: str,
        markdown: str,
        html: str,
        markdown_path: Path | None,
        html_path: Path | None,
        payload: dict[str, Any],
    ) -> None:
        self._exec(
            """
            INSERT INTO reports (id, dataset_id, conversation_id, title, markdown, html,
                                 markdown_path, html_path, payload, generated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                markdown=excluded.markdown, html=excluded.html, payload=excluded.payload
            """,
            (
                report_id, dataset_id, conversation_id, title, markdown, html,
                str(markdown_path) if markdown_path else None,
                str(html_path) if html_path else None,
                json.dumps(payload, ensure_ascii=False), now_iso(),
            ),
        )

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        row = self._query_one("SELECT * FROM reports WHERE id = ?", (report_id,))
        if row and row.get("payload"):
            try:
                row["payload"] = json.loads(row["payload"])
            except json.JSONDecodeError:
                row["payload"] = {}
        return row

    def list_reports(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        if dataset_id:
            return self._query(
                "SELECT id, dataset_id, conversation_id, title, generated_at "
                "FROM reports WHERE dataset_id = ? ORDER BY generated_at DESC",
                (dataset_id,),
            )
        return self._query(
            "SELECT id, dataset_id, conversation_id, title, generated_at "
            "FROM reports ORDER BY generated_at DESC"
        )

    # ---------------- 图表产物 ----------------
    def save_artifact(
        self,
        *,
        artifact_id: str,
        dataset_id: str,
        conversation_id: str | None,
        kind: str,
        title: str,
        description: str,
        file_path: Path,
    ) -> None:
        self._exec(
            """
            INSERT INTO artifacts (id, dataset_id, conversation_id, kind, title,
                                   description, file_path, created_at)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET title=excluded.title, description=excluded.description
            """,
            (artifact_id, dataset_id, conversation_id, kind, title, description, str(file_path), now_iso()),
        )

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        return self._query_one("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))

    def list_artifacts(
        self,
        dataset_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM artifacts WHERE 1=1"
        params: list[Any] = []
        if dataset_id:
            sql += " AND dataset_id = ?"
            params.append(dataset_id)
        if conversation_id:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)
        sql += " ORDER BY created_at DESC"
        return self._query(sql, tuple(params))


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store()
