"""Tests for Redis checkpoint implementation."""

import pytest
from redis.asyncio import Redis as AsyncRedis

from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.checkpoint.serde.json import JsonSerializer

@pytest.fixture
async def redis_conn():
    """Create a Redis connection for testing."""
    conn = AsyncRedis(host="localhost", port=6379, db=0)
    yield conn
    await conn.flushdb()
    await conn.aclose()

@pytest.fixture
def serde():
    """Create a JSON serializer for testing."""
    return JsonSerializer()

@pytest.mark.asyncio
async def test_redis_checkpoint_basic(redis_conn, serde):
    """Test basic checkpoint operations."""
    saver = AsyncRedisSaver(redis_conn, serde=serde)
    config = {"configurable": {"thread_id": "test", "checkpoint_ns": "test"}}
    checkpoint = {"id": "test", "data": "test"}
    metadata = {"test": "test"}
    new_versions = {}

    # Test saving checkpoint
    result = await saver.aput(config, checkpoint, metadata, new_versions)
    assert result["configurable"]["checkpoint_id"] == "test"

    # Test retrieving checkpoint
    checkpoint_tuple = await saver.aget_tuple(result)
    assert checkpoint_tuple is not None
    assert checkpoint_tuple.checkpoint["id"] == "test"
    assert checkpoint_tuple.metadata["test"] == "test"

@pytest.mark.asyncio
async def test_redis_checkpoint_writes(redis_conn, serde):
    """Test checkpoint writes operations."""
    saver = AsyncRedisSaver(redis_conn, serde=serde)
    config = {
        "configurable": {
            "thread_id": "test",
            "checkpoint_ns": "test",
            "checkpoint_id": "test"
        }
    }
    writes = [("channel1", "data1"), ("channel2", "data2")]
    task_id = "task1"

    # Test saving writes
    result = await saver.aput_writes(config, writes, task_id)
    assert result["configurable"]["checkpoint_id"] == "test"

    # Test retrieving checkpoint with writes
    checkpoint_tuple = await saver.aget_tuple(result)
    assert checkpoint_tuple is not None
    assert len(checkpoint_tuple.pending_writes) == 2
    assert checkpoint_tuple.pending_writes[0][0] == "channel1"
    assert checkpoint_tuple.pending_writes[1][0] == "channel2"

@pytest.mark.asyncio
async def test_redis_checkpoint_list(redis_conn, serde):
    """Test listing checkpoints."""
    saver = AsyncRedisSaver(redis_conn, serde=serde)
    config = {"configurable": {"thread_id": "test", "checkpoint_ns": "test"}}
    
    # Create multiple checkpoints
    checkpoints = []
    for i in range(3):
        checkpoint = {"id": f"test{i}", "data": f"test{i}"}
        metadata = {"test": f"test{i}"}
        new_versions = {}
        result = await saver.aput(config, checkpoint, metadata, new_versions)
        checkpoints.append(result)

    # Test listing all checkpoints
    async for checkpoint_tuple in saver.alist(config):
        assert checkpoint_tuple is not None
        assert checkpoint_tuple.checkpoint["id"].startswith("test")
        assert checkpoint_tuple.metadata["test"].startswith("test")

    # Test listing with limit
    count = 0
    async for checkpoint_tuple in saver.alist(config, limit=2):
        count += 1
    assert count == 2

@pytest.mark.asyncio
async def test_redis_checkpoint_version(redis_conn, serde):
    """Test version generation."""
    saver = AsyncRedisSaver(redis_conn, serde=serde)
    
    # Test version generation from None
    v1 = saver.get_next_version(None, "test")
    assert v1 is not None
    assert "." in v1
    
    # Test version increment
    v2 = saver.get_next_version(v1, "test")
    assert v2 is not None
    assert v2 > v1
