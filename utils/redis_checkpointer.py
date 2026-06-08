"""
Redis Checkpointer for LangGraph (Official)
============================================
Uses the official `langgraph-checkpoint-redis` package which is maintained
by the LangGraph team and provides:
  - Fast orjson serialization
  - RediSearch-backed metadata filtering
  - RedisJSON human-readable storage
  - Full async support

Requirements
------------
- Redis Stack running (see Docker_redis/docker-compose.yml)
  Redis Stack bundles RedisJSON + RediSearch — required by this library.
- Install: pip install langgraph-checkpoint-redis

Usage
-----
Sync:
    from utils.redis_checkpointer import get_sync_saver

    with get_sync_saver() as saver:
        graph = builder.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "user-42"}}
        result = graph.invoke({"messages": [...]}, config)

Async:
    from utils.redis_checkpointer import get_async_saver

    async with get_async_saver() as saver:
        graph = builder.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "user-42"}}
        result = await graph.ainvoke({"messages": [...]}, config)
"""

from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator

from langgraph.checkpoint.redis import AsyncRedisSaver, RedisSaver

REDIS_URL = "redis://localhost:6379"


@contextmanager
def get_sync_saver(url: str = REDIS_URL) -> Iterator[RedisSaver]:
    """
    Yield a ready-to-use synchronous RedisSaver.

    Calls .setup() automatically on first use to create the required
    RediSearch indices (safe to call multiple times — it's idempotent).
    """
    with RedisSaver.from_conn_string(url) as saver:
        saver.setup()
        yield saver


@asynccontextmanager
async def get_async_saver(url: str = REDIS_URL) -> AsyncIterator[AsyncRedisSaver]:
    """
    Yield a ready-to-use asynchronous AsyncRedisSaver.

    Calls .asetup() automatically on first use to create the required
    RediSearch indices (safe to call multiple times — it's idempotent).
    """
    async with AsyncRedisSaver.from_conn_string(url) as saver:
        await saver.asetup()
        yield saver
