"""
MainAgent/nodes/load_memory_node.py
-------------------------------------
Graph node: load_memory_node

Loads conversation memory for the current turn:
  - Running summary
  - Last 20 messages (10 turns)
  - Unsummarized token count
  - Last summarized message sequence number

Strategy
--------
  unsummarized_token_count and last_summarized_message_seq are always
  read from PostgreSQL (lightweight single-row query; not cached in Redis).

  summary + recent_messages use Redis-first with PostgreSQL fallback.
  On a Redis miss the values are written back to Redis before returning.

  A conversation_memory row is guaranteed to exist after this node via
  ensure_conversation_memory_exists() (INSERT … ON CONFLICT DO NOTHING).
"""

from __future__ import annotations

import asyncio

from MainAgent.db.postgres_client import (
    ensure_conversation_memory_exists,
    fetch_conversation_memory,
    fetch_recent_messages,
)
from MainAgent.db.redis_client import get_conversation_cache, set_conversation_cache
from MainAgent.state import OrchestratorState
from utils.tracing import traceable


@traceable(name="load_memory_node", tags=["main-agent", "node"])
async def load_memory_node(state: OrchestratorState) -> dict:
    """
    Load conversation memory from Redis (summary + messages) and
    PostgreSQL (token counts / summarization bookmarks).

    Returns partial state update with:
      - ``summary``
      - ``recent_messages``
      - ``unsummarized_token_count``
      - ``last_summarized_message_seq``
    """
    conv_id = state["conv_id"]

    # ── 1. Ensure a conversation_memory row exists (idempotent) ───────────────
    await asyncio.to_thread(ensure_conversation_memory_exists, conv_id)

    # ── 2. Always load token counters from DB (not stored in Redis) ───────────
    memory_row = await asyncio.to_thread(fetch_conversation_memory, conv_id)

    unsummarized_token_count   = 0
    last_summarized_message_seq = 0

    if memory_row:
        unsummarized_token_count    = memory_row.get("unsummarized_token_count")    or 0
        last_summarized_message_seq = memory_row.get("last_summarized_message_seq") or 0

    # ── 3. Try Redis for summary + messages ───────────────────────────────────
    cached = await asyncio.to_thread(get_conversation_cache, conv_id)
    if cached is not None:
        return {
            "summary":                    cached.get("summary", ""),
            "recent_messages":            cached.get("messages", []),
            "unsummarized_token_count":   unsummarized_token_count,
            "last_summarized_message_seq": last_summarized_message_seq,
        }

    # ── 4. Redis miss — load from PostgreSQL ──────────────────────────────────
    summary = (memory_row.get("summary") or "") if memory_row else ""

    recent_messages = await asyncio.to_thread(fetch_recent_messages, conv_id, 20)

    # ── 5. Populate Redis cache ───────────────────────────────────────────────
    await asyncio.to_thread(
        set_conversation_cache,
        conv_id,
        summary,
        recent_messages,
    )

    return {
        "summary":                    summary,
        "recent_messages":            recent_messages,
        "unsummarized_token_count":   unsummarized_token_count,
        "last_summarized_message_seq": last_summarized_message_seq,
    }
