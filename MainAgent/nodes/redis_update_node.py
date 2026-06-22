"""
MainAgent/nodes/redis_update_node.py
--------------------------------------
Graph node: redis_update_node

Refreshes the Redis conversation cache after all persistence is complete.

This node always runs — even if summary_node was a no-op — to ensure
the newly persisted user and assistant messages appear in the cache for
the next request.

Operations
----------
1. Fetch the latest 20 messages from PostgreSQL (authoritative source).
2. Call set_conversation_cache() with the current summary and fresh messages.

The cache key is: conversation:{conv_id}
TTL is refreshed on every write (7 days).
"""

from __future__ import annotations

import asyncio

from MainAgent.db.postgres_client import fetch_recent_messages
from MainAgent.db.redis_client import set_conversation_cache
from MainAgent.state import OrchestratorState
from utils.tracing import traceable


@traceable(name="redis_update_node", tags=["main-agent", "node"])
async def redis_update_node(state: OrchestratorState) -> dict:
    """
    Refresh the Redis conversation cache with the latest persisted messages.

    Always runs at the end of the pipeline so the next request is served
    directly from cache without a PostgreSQL round-trip.

    Returns an empty dict — this node has no state fields to update.
    """
    conv_id = state["conv_id"]

    # ── Fetch the 20 most recent messages from DB ─────────────────────────────
    recent = await asyncio.to_thread(fetch_recent_messages, conv_id, 20)

    # ── Write refreshed cache entry ───────────────────────────────────────────
    await asyncio.to_thread(
        set_conversation_cache,
        conv_id,
        state["summary"],   # may have been updated by summary_node
        recent,
    )

    return {}
