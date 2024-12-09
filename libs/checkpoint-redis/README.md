# LangGraph Redis Checkpoint

Redis-based checkpoint implementation for LangGraph.

## Installation

```bash
pip install langgraph-checkpoint-redis
```

## Usage

```python
from langgraph.checkpoint.redis import AsyncRedisSaver

# Create a Redis checkpoint saver
async with AsyncRedisSaver.from_conn_info(
    host="localhost",
    port=6379,
    db=0,
    password="your-password"
) as saver:
    # Use the saver in your LangGraph application
    config = {"configurable": {"thread_id": "1"}}
    checkpoint = await saver.aget_tuple(config)
```

## Development

This package uses poetry for dependency management. To set up a development environment:

```bash
poetry install
poetry run pytest
```
