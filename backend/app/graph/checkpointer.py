"""SQLite 持久化 Checkpoint。

LangGraph 自带的 ``InMemorySaver`` 只把状态放在内存里，进程一重启多轮上下文就没了。
这里在它的基础上做"写穿透 + 启动回放"：所有读取语义完全复用官方实现，只在
``put`` / ``put_writes`` / ``delete_thread`` 三个写入点上同步落库，启动时再把三张表
装回内存。这样既拿到了持久化，也不用重写一套 checkpointer 的读取逻辑。
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.base import SerializerProtocol

from ..config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id            TEXT NOT NULL,
    checkpoint_ns        TEXT NOT NULL DEFAULT '',
    checkpoint_id        TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    cp_type              TEXT NOT NULL,
    cp                   BLOB NOT NULL,
    md_type              TEXT NOT NULL,
    md                   BLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id     TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel       TEXT NOT NULL,
    version       TEXT NOT NULL,
    type          TEXT NOT NULL,
    data          BLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id     TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    idx           INTEGER NOT NULL,
    channel       TEXT NOT NULL,
    type          TEXT NOT NULL,
    value         BLOB NOT NULL,
    task_path     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

CREATE INDEX IF NOT EXISTS idx_cp_thread ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_cpw_thread ON checkpoint_writes(thread_id);
"""


def _encode_version(version: Any) -> str:
    return str(version)


def _decode_version(text: str) -> Any:
    """把存成文本的 channel 版本号还原成原来的类型（int / float / str）。"""

    try:
        return int(text)
    except (TypeError, ValueError):
        pass
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


class SqliteCheckpointSaver(InMemorySaver):
    """带 SQLite 持久化的 checkpointer。"""

    def __init__(
        self,
        db_path: Path | str,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        self._restore()

    # ---------------- 基础设施 ----------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _restore(self) -> None:
        """启动时把磁盘上的 checkpoint 回放到内存结构。"""

        with self._lock, self._connect() as conn:
            for tid, ns, cid, parent, cp_type, cp, md_type, md in conn.execute(
                "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,"
                " cp_type, cp, md_type, md FROM checkpoints"
            ):
                self.storage[tid][ns][cid] = ((cp_type, bytes(cp)), (md_type, bytes(md)), parent)

            for tid, ns, channel, version, btype, data in conn.execute(
                "SELECT thread_id, checkpoint_ns, channel, version, type, data FROM checkpoint_blobs"
            ):
                self.blobs[(tid, ns, channel, _decode_version(version))] = (btype, bytes(data))

            for tid, ns, cid, task_id, idx, channel, wtype, value, task_path in conn.execute(
                "SELECT thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel,"
                " type, value, task_path FROM checkpoint_writes"
            ):
                self.writes[(tid, ns, cid)][(task_id, idx)] = (
                    task_id, channel, (wtype, bytes(value)), task_path,
                )

    # ---------------- 写穿透 ----------------
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        result = super().put(config, checkpoint, metadata, new_versions)

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        values = checkpoint.get("channel_values", {}) or {}
        saved = self.storage[thread_id][checkpoint_ns][checkpoint["id"]]
        (cp_type, cp_blob), (md_type, md_blob), parent_id = saved

        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id,"
                " parent_checkpoint_id, cp_type, cp, md_type, md) VALUES (?,?,?,?,?,?,?,?)",
                (
                    thread_id, checkpoint_ns, checkpoint["id"], parent_id,
                    cp_type, cp_blob, md_type, md_blob,
                ),
            )
            rows = []
            for channel, version in new_versions.items():
                blob = self.blobs.get((thread_id, checkpoint_ns, channel, version))
                if blob is None:
                    blob = (
                        self.serde.dumps_typed(values[channel])
                        if channel in values
                        else ("empty", b"")
                    )
                rows.append(
                    (thread_id, checkpoint_ns, channel, _encode_version(version), blob[0], blob[1])
                )
            if rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO checkpoint_blobs (thread_id, checkpoint_ns, channel,"
                    " version, type, data) VALUES (?,?,?,?,?,?)",
                    rows,
                )
        return result

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        super().put_writes(config, writes, task_id, task_path)

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        stored = self.writes.get((thread_id, checkpoint_ns, checkpoint_id), {})

        rows = [
            (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, blob[0], blob[1], path)
            for (tid, idx), (task_id, channel, blob, path) in stored.items()
        ]
        if rows:
            with self._lock, self._connect() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO checkpoint_writes (thread_id, checkpoint_ns,"
                    " checkpoint_id, task_id, idx, channel, type, value, task_path)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    rows,
                )

    def delete_thread(self, thread_id: str) -> None:
        super().delete_thread(thread_id)
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,))

    async def adelete_thread(self, thread_id: str) -> None:  # pragma: no cover
        self.delete_thread(thread_id)

    # ---------------- 运维辅助 ----------------
    def threads(self) -> list[dict[str, Any]]:
        """列出磁盘上已有的会话线程，便于前端展示历史。"""

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT thread_id, COUNT(*) AS checkpoints, MAX(checkpoint_id) AS last_id"
                " FROM checkpoints GROUP BY thread_id ORDER BY last_id DESC"
            ).fetchall()
        return [
            {"thread_id": tid, "checkpoints": int(count), "last_checkpoint": last}
            for tid, count, last in rows
        ]

    def stats(self) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            cp = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
            blobs = conn.execute("SELECT COUNT(*) FROM checkpoint_blobs").fetchone()[0]
            writes = conn.execute("SELECT COUNT(*) FROM checkpoint_writes").fetchone()[0]
        return {"checkpoints": int(cp), "blobs": int(blobs), "writes": int(writes)}

    def purge_all(self) -> None:
        """清空全部 checkpoint（危险操作，仅用于测试/重置）。"""

        super().__init__(serde=self.serde)
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM checkpoints")
            conn.execute("DELETE FROM checkpoint_blobs")
            conn.execute("DELETE FROM checkpoint_writes")


@lru_cache(maxsize=1)
def get_checkpointer() -> SqliteCheckpointSaver:
    return SqliteCheckpointSaver(get_settings().checkpoint_db_path)


def reset_checkpointer() -> None:
    get_checkpointer.cache_clear()


__all__ = [
    "SqliteCheckpointSaver",
    "get_checkpointer",
    "reset_checkpointer",
    "BaseCheckpointSaver",
]
