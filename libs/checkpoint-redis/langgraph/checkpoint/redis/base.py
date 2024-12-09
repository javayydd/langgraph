import base64
from typing import Optional, List, Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import WRITES_IDX_MAP, PendingWrite, CheckpointTuple
from langgraph.checkpoint.serde.base import SerializerProtocol

REDIS_KEY_SEPARATOR = ":"
def _make_redis_checkpoint_key(
    thread_id: str, checkpoint_ns: str, checkpoint_id: str
) -> str:
    return REDIS_KEY_SEPARATOR.join(
        ["checkpoint", thread_id, checkpoint_ns, checkpoint_id]
    )


def _make_redis_checkpoint_writes_key(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    idx: Optional[int],
) -> str:
    # Use empty strings to replace None values if necessary
    thread_id = thread_id or ""
    checkpoint_ns = checkpoint_ns or ""
    checkpoint_id = checkpoint_id or ""
    task_id = task_id or ""

    if idx is None:
        return REDIS_KEY_SEPARATOR.join(
            ["writes", thread_id, checkpoint_ns, checkpoint_id, task_id]
        )

    return REDIS_KEY_SEPARATOR.join(
        ["writes", thread_id, checkpoint_ns, checkpoint_id, task_id, str(idx)]
    )


def _parse_redis_checkpoint_key(redis_key: str) -> dict:
    parts = redis_key.split(REDIS_KEY_SEPARATOR)

    if parts[0] != "checkpoint":
        raise ValueError("Expected checkpoint key to start with 'checkpoint'")

    # namespace is always the first part, checkpoint_id is always the last part
    namespace = parts[0]
    checkpoint_id = parts[-1]

    # Handle cases with varying numbers of middle components
    if len(parts) == 4:
        _, thread_id, checkpoint_ns, checkpoint_id = parts
    elif len(parts) > 4:
        thread_id = parts[1]
        checkpoint_ns = ":".join(
            parts[2:-1]
        )  # Capture everything between thread_id and checkpoint_id
    else:
        raise ValueError(f"Invalid redis key format: {redis_key}")

    return {
        "namespace": namespace,
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
    }


def _parse_redis_checkpoint_writes_key(redis_key: str) -> dict:
    parts = redis_key.split(REDIS_KEY_SEPARATOR)

    if parts[0] != "writes":
        raise ValueError("Expected writes key to start with 'writes'")

    idx = parts[-1]
    task_id = parts[-2]

    # Handle cases with varying numbers of middle components
    if len(parts) == 6:
        _, thread_id, checkpoint_ns, checkpoint_id, task_id, idx = parts
    elif len(parts) > 6:
        thread_id = parts[1]
        checkpoint_ns = ":".join(
            parts[2:-3]
        )  # Capture everything between thread_id and checkpoint_id
        checkpoint_id = parts[-3]
    else:
        raise ValueError(f"Invalid writes key format: {redis_key}")

    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
        "task_id": task_id,
        "idx": idx,
    }


def _filter_keys(
    keys: List[str], before: Optional[RunnableConfig], limit: Optional[int]
) -> list:
    """Filter and sort Redis keys based on optional criteria."""
    if before:
        keys = [
            k
            for k in keys
            if _parse_redis_checkpoint_key(k.decode())["checkpoint_id"]
            < before["configurable"]["checkpoint_id"]
        ]

    keys = sorted(
        keys,
        key=lambda k: _parse_redis_checkpoint_key(k.decode())["checkpoint_id"],
        reverse=True,
    )
    if limit:
        keys = keys[:limit]
    return keys


def _dump_writes(serde: SerializerProtocol, writes: tuple[str, Any]) -> list[dict]:
    """Serialize pending writes."""
    serialized_writes = []
    for channel, value in writes:
        type_, serialized_value = serde.dumps_typed(value)
        serialized_writes.append(
            {"channel": channel, "type": type_, "value": serialized_value}
        )
    return serialized_writes


def _load_writes(
    serde: SerializerProtocol, writes_dict: dict[tuple[str, int], dict]
) -> WRITES_IDX_MAP:
    """Deserialize pending writes."""
    loaded_writes = []
    for _, value in sorted(writes_dict.items()):
        # 使用 get 方法检查 'channel' 键是否存在
        channel = value.get("channel")
        if channel is None:
            continue
        channel = (
            value["channel"].decode()
            if isinstance(value["channel"], bytes)
            else value["channel"]
        )
        serialized_type = (
            value["type"].decode()
            if isinstance(value["type"], bytes)
            else value["type"]
        )

        # Handle the potential types of value["value"]

        serialized_value = value["value"]
        if isinstance(serialized_value, bytes):
            serialized_value = serialized_value.decode()
        elif serialized_value is None:
            serialized_value = ""  # or handle None case as needed

        loaded_writes.append(
            (
                channel,
                serde.loads_typed(serialized_type, serialized_value),
            )
        )
    return loaded_writes


def _parse_redis_checkpoint_data(
    serde: SerializerProtocol,
    key: str,
    data: dict,
    pending_writes: Optional[List[PendingWrite]] = None,
) -> Optional[CheckpointTuple]:
    """Parse checkpoint data retrieved from Redis."""
    if not data:
        return None

    parsed_key = _parse_redis_checkpoint_key(key)
    thread_id = parsed_key["thread_id"]
    checkpoint_ns = parsed_key["checkpoint_ns"]
    checkpoint_id = parsed_key["checkpoint_id"]
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }

    # Decode the checkpoint without converting to UTF-8, keep it as bytes
    checkpoint_type = data[b"type"].decode()  # Assume type is a string
    checkpoint = serde.loads_typed(
        (checkpoint_type, base64.b64decode(data[b"checkpoint"]))
    )

    # Decode the metadata if it exists, handle missing metadata gracefully
    metadata_b64 = data.get(b"metadata")
    if metadata_b64:
        metadata = serde.loads(base64.b64decode(metadata_b64.decode("utf-8")))
    else:
        metadata = None  # or handle the missing metadata case

    parent_checkpoint_id = data.get(b"parent_checkpoint_id", b"").decode()
    parent_config = (
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": parent_checkpoint_id,
            }
        }
        if parent_checkpoint_id
        else None
    )
    return CheckpointTuple(
        config=config,
        checkpoint=checkpoint,
        metadata=metadata,
        parent_config=parent_config,
        pending_writes=pending_writes,
    )