import base64
import random
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence, Tuple

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import ChannelProtocol
from redis import Redis

from .base import (
    _make_redis_checkpoint_key,
    _dump_writes,
    _make_redis_checkpoint_writes_key,
    _parse_redis_checkpoint_writes_key,
    _load_writes,
    _parse_redis_checkpoint_data,
    _filter_keys,
    _parse_redis_checkpoint_key,
)

__all__ = [
    "_make_redis_checkpoint_key",
    "_dump_writes",
    "_make_redis_checkpoint_writes_key",
    "_parse_redis_checkpoint_writes_key",
    "_load_writes",
    "_parse_redis_checkpoint_data",
    "_filter_keys",
    "_parse_redis_checkpoint_key",
]
REDIS_KEY_SEPARATOR = ":"


class RedisSaver(BaseCheckpointSaver[str]):
    """Redis-based checkpoint saver implementation."""

    conn: Redis

    def __init__(self, conn: Redis):
        super().__init__()
        self.conn = conn

    @classmethod
    @contextmanager
    def from_conn_info(
            cls, *, host: str, port: int, db: int, password: str
    ) -> Iterator["RedisSaver"]:
        conn = Redis(host=host, port=port, db=db, password=password)
        try:
            yield RedisSaver(conn)
        finally:
            conn.close()

    def put(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint with subgraph support to Redis."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("parent_checkpoint_id")
        key = _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = self.serde.dumps(metadata)

        # Encode checkpoint and metadata in base64
        serialized_checkpoint_b64 = base64.b64encode(serialized_checkpoint).decode(
            "utf-8"
        )
        serialized_metadata_b64 = base64.b64encode(serialized_metadata).decode("utf-8")

        data = {
            "checkpoint": serialized_checkpoint_b64,
            "type": type_,
            "metadata": serialized_metadata_b64,
            "parent_checkpoint_id": parent_checkpoint_id or "",
        }

        self.conn.hset(key, mapping=data)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "parent_checkpoint_id": parent_checkpoint_id,
            }
        }

    def put_writes(
            self,
            config: RunnableConfig,
            writes: Sequence[Tuple[str, Any]],
            task_id: str,
    ) -> RunnableConfig:
        """Store intermediate writes with subgraph support."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        for idx, data in enumerate(_dump_writes(self.serde, writes)):
            key = _make_redis_checkpoint_writes_key(
                thread_id, checkpoint_ns, checkpoint_id, task_id, idx
            )
            self.conn.hset(key, mapping=data)
        return config

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple with subgraph support from Redis."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if not checkpoint_id:
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }

        checkpoint_key = self._get_checkpoint_key(
            self.conn, thread_id, checkpoint_ns, checkpoint_id
        )
        if not checkpoint_key:
            return None
        checkpoint_data = self.conn.hgetall(checkpoint_key)

        # Load pending writes
        writes_key = _make_redis_checkpoint_writes_key(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            "*",
            None,  # 在 task_id 位置使用 "*" 来匹配
        )
        matching_keys = self.conn.keys(pattern=writes_key)
        parsed_keys = [
            _parse_redis_checkpoint_writes_key(key.decode()) for key in matching_keys
        ]
        pending_writes = _load_writes(
            self.serde,
            {
                (parsed_key["task_id"], parsed_key["idx"]): self.conn.hgetall(key)
                for key, parsed_key in sorted(
                zip(matching_keys, parsed_keys), key=lambda x: x[1]["idx"]
            )
            },
        )
        return _parse_redis_checkpoint_data(
            self.serde, checkpoint_key, checkpoint_data, pending_writes=pending_writes
        )

    def list(
            self,
            config: Optional[RunnableConfig],
            *,
            filter: Optional[dict[str, Any]] = None,
            before: Optional[RunnableConfig] = None,
            limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints with subgraph support from Redis."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        pattern = _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")
        keys = _filter_keys(self.conn.keys(pattern), before, limit)
        for key in keys:
            data = self.conn.hgetall(key)
            if data and b"checkpoint" in data and b"metadata" in data:
                yield _parse_redis_checkpoint_data(self.serde, key.decode(), data)

    def _get_checkpoint_key(
            self, conn, thread_id: str, checkpoint_ns: str, checkpoint_id: Optional[str]
    ) -> Optional[str]:
        """Determine the Redis key for a checkpoint."""
        if checkpoint_id:
            return _make_redis_checkpoint_key(
                thread_id, checkpoint_ns, checkpoint_id
            )

        all_keys = conn.keys(
            _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")
        )
        if not all_keys:
            return None

        latest_key = max(
            all_keys,
            key=lambda k: _parse_redis_checkpoint_key(k.decode())["checkpoint_id"],
        )
        return latest_key.decode()

    def get_next_version(self, current: Optional[str], channel: ChannelProtocol) -> str:
        """Get the next version for a channel using Redis."""
        # Fetch the current version from Redis if it's not provided
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            # Split on "." to get the integer part of the version
            current_v = int(current.split(".")[0])

        # Increment the version
        next_v = current_v + 1

        # Generate a random hash value for the next version
        next_h = random.random()

        # Return the new version in the same format
        return f"{next_v:032}.{next_h:016}"
