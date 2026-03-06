"""Persistent LangGraph checkpoint saver backed by DuckDB."""

from __future__ import annotations

import os
import random
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import duckdb
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

DEFAULT_CHECKPOINT_DB_PATH = "mao_checkpoints.duckdb"
_CHECKPOINT_SAVERS: dict[str, "DuckDBSaver"] = {}
_CHECKPOINT_SAVERS_LOCK = threading.Lock()


def get_checkpoint_db_path() -> str:
    return os.environ.get("MAO_CHECKPOINT_DB_PATH", DEFAULT_CHECKPOINT_DB_PATH)


def get_checkpointer() -> "DuckDBSaver":
    db_path = os.path.abspath(get_checkpoint_db_path())
    with _CHECKPOINT_SAVERS_LOCK:
        if db_path not in _CHECKPOINT_SAVERS:
            _CHECKPOINT_SAVERS[db_path] = DuckDBSaver(db_path)
        return _CHECKPOINT_SAVERS[db_path]


class DuckDBSaver(BaseCheckpointSaver[str]):
    """DuckDB-backed checkpoint saver compatible with LangGraph checkpointers."""

    def __init__(self, db_path: str) -> None:
        super().__init__()
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._lock = threading.RLock()
        self._setup()

    def _setup(self) -> None:
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id VARCHAR NOT NULL,
                    checkpoint_ns VARCHAR NOT NULL,
                    checkpoint_id VARCHAR NOT NULL,
                    checkpoint_type VARCHAR NOT NULL,
                    checkpoint_blob BLOB NOT NULL,
                    metadata_type VARCHAR NOT NULL,
                    metadata_blob BLOB NOT NULL,
                    parent_checkpoint_id VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                    thread_id VARCHAR NOT NULL,
                    checkpoint_ns VARCHAR NOT NULL,
                    channel_name VARCHAR NOT NULL,
                    channel_version VARCHAR NOT NULL,
                    value_type VARCHAR NOT NULL,
                    value_blob BLOB NOT NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, channel_name, channel_version)
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_writes (
                    thread_id VARCHAR NOT NULL,
                    checkpoint_ns VARCHAR NOT NULL,
                    checkpoint_id VARCHAR NOT NULL,
                    task_id VARCHAR NOT NULL,
                    write_idx INTEGER NOT NULL,
                    channel_name VARCHAR NOT NULL,
                    value_type VARCHAR NOT NULL,
                    value_blob BLOB NOT NULL,
                    task_path VARCHAR NOT NULL DEFAULT '',
                    PRIMARY KEY (
                        thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx
                    )
                )
                """
            )

    def _load_blobs(
        self, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
    ) -> dict[str, Any]:
        channel_values: dict[str, Any] = {}
        for channel_name, channel_version in versions.items():
            row = self.conn.execute(
                """
                SELECT value_type, value_blob
                FROM checkpoint_blobs
                WHERE thread_id = ? AND checkpoint_ns = ? AND channel_name = ? AND channel_version = ?
                """,
                [thread_id, checkpoint_ns, channel_name, str(channel_version)],
            ).fetchone()
            if row and row[0] != "empty":
                channel_values[channel_name] = self.serde.loads_typed((row[0], row[1]))
        return channel_values

    def _row_to_checkpoint_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        row: tuple[Any, ...],
    ) -> CheckpointTuple:
        checkpoint_id, checkpoint_type, checkpoint_blob, metadata_type, metadata_blob, parent_checkpoint_id = row
        checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_blob))
        metadata = self.serde.loads_typed((metadata_type, metadata_blob))
        writes = self.conn.execute(
            """
            SELECT task_id, channel_name, value_type, value_blob, task_path
            FROM checkpoint_writes
            WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            ORDER BY task_id, write_idx
            """,
            [thread_id, checkpoint_ns, checkpoint_id],
        ).fetchall()
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint={
                **checkpoint,
                "channel_values": self._load_blobs(
                    thread_id, checkpoint_ns, checkpoint["channel_versions"]
                ),
            },
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
            pending_writes=[
                (task_id, channel_name, self.serde.loads_typed((value_type, value_blob)))
                for task_id, channel_name, value_type, value_blob, _task_path in writes
            ],
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        with self._lock:
            if checkpoint_id:
                row = self.conn.execute(
                    """
                    SELECT checkpoint_id, checkpoint_type, checkpoint_blob, metadata_type, metadata_blob, parent_checkpoint_id
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    """,
                    [thread_id, checkpoint_ns, checkpoint_id],
                ).fetchone()
            else:
                row = self.conn.execute(
                    """
                    SELECT checkpoint_id, checkpoint_type, checkpoint_blob, metadata_type, metadata_blob, parent_checkpoint_id
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ?
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                    """,
                    [thread_id, checkpoint_ns],
                ).fetchone()
            if row is None:
                return None
            return self._row_to_checkpoint_tuple(thread_id, checkpoint_ns, row)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        filter = filter or {}
        thread_ids = (
            [config["configurable"]["thread_id"]]
            if config
            else [r[0] for r in self.conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()]
        )
        config_checkpoint_ns = (
            config["configurable"].get("checkpoint_ns") if config else None
        )
        before_checkpoint_id = get_checkpoint_id(before) if before else None
        yielded = 0
        with self._lock:
            for thread_id in thread_ids:
                namespaces = (
                    [config_checkpoint_ns]
                    if config_checkpoint_ns is not None
                    else [r[0] for r in self.conn.execute(
                        "SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE thread_id = ?",
                        [thread_id],
                    ).fetchall()]
                )
                for checkpoint_ns in namespaces:
                    if checkpoint_ns is None:
                        continue
                    rows = self.conn.execute(
                        """
                        SELECT checkpoint_id, checkpoint_type, checkpoint_blob, metadata_type, metadata_blob, parent_checkpoint_id
                        FROM checkpoints
                        WHERE thread_id = ? AND checkpoint_ns = ?
                        ORDER BY checkpoint_id DESC
                        """,
                        [thread_id, checkpoint_ns],
                    ).fetchall()
                    for row in rows:
                        checkpoint_id = row[0]
                        if before_checkpoint_id and checkpoint_id >= before_checkpoint_id:
                            continue
                        checkpoint_tuple = self._row_to_checkpoint_tuple(thread_id, checkpoint_ns, row)
                        if filter and not all(
                            checkpoint_tuple.metadata.get(k) == v for k, v in filter.items()
                        ):
                            continue
                        yield checkpoint_tuple
                        yielded += 1
                        if limit is not None and yielded >= limit:
                            return

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        c = checkpoint.copy()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]
        with self._lock:
            for channel_name, channel_version in new_versions.items():
                typed_value = (
                    self.serde.dumps_typed(values[channel_name])
                    if channel_name in values
                    else ("empty", b"")
                )
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoint_blobs
                    (thread_id, checkpoint_ns, channel_name, channel_version, value_type, value_blob)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        thread_id,
                        checkpoint_ns,
                        channel_name,
                        str(channel_version),
                        typed_value[0],
                        typed_value[1],
                    ],
                )
            checkpoint_payload = self.serde.dumps_typed(c)
            metadata_payload = self.serde.dumps_typed(
                get_checkpoint_metadata(config, metadata)
            )
            self.conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                (thread_id, checkpoint_ns, checkpoint_id, checkpoint_type, checkpoint_blob, metadata_type, metadata_blob, parent_checkpoint_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    checkpoint_payload[0],
                    checkpoint_payload[1],
                    metadata_payload[0],
                    metadata_payload[1],
                    config["configurable"].get("checkpoint_id"),
                ],
            )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        with self._lock:
            for idx, (channel_name, value) in enumerate(writes):
                write_idx = WRITES_IDX_MAP.get(channel_name, idx)
                if write_idx >= 0:
                    existing = self.conn.execute(
                        """
                        SELECT 1 FROM checkpoint_writes
                        WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                          AND task_id = ? AND write_idx = ?
                        """,
                        [thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx],
                    ).fetchone()
                    if existing:
                        continue
                typed_value = self.serde.dumps_typed(value)
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoint_writes
                    (thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx, channel_name, value_type, value_blob, task_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        write_idx,
                        channel_name,
                        typed_value[0],
                        typed_value[1],
                        task_path,
                    ],
                )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(str(current).split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
