"""
MainAgent/db/redis_client.py
-----------------------------
Redis caching layer for the MainAgent pipeline.

Two cache namespaces
--------------------
  template:{template_id}   — template config (behaviour_prompt, tool_information)
  conversation:{conv_id}   — {"summary": "...", "messages": [...]}

A module-level ConnectionPool is shared across all calls so new TCP sockets
are not opened per invocation.

All public functions are synchronous; async nodes call them via
``asyncio.to_thread()`` to avoid blocking the event loop.

Environment
-----------
  REDIS_URL   e.g. redis://localhost:6379  (default used if not set)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from redis import ConnectionPool, Redis

# ---------------------------------------------------------------------------
# Bootstrap — load .env from project root
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # AdapterAI/
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Connection pool (module-level singleton)
# ---------------------------------------------------------------------------

_pool: Optional[ConnectionPool] = None

_TTL_TEMPLATE     = 60 * 60            # 1 hour
_TTL_CONVERSATION = 60 * 60 * 24 * 7  # 7 days


def _get_client() -> Redis:
    """Return a Redis client backed by the shared connection pool."""
    global _pool
    if _pool is None:
        url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        _pool = ConnectionPool.from_url(url, decode_responses=True)
    return Redis(connection_pool=_pool)


# ---------------------------------------------------------------------------
# Template cache  —  template:{template_id}
# ---------------------------------------------------------------------------

def get_template_cache(template_id: str) -> Optional[Dict]:
    """
    Retrieve cached template configuration.

    Returns
    -------
    dict with keys ``behavior_prompt`` and ``custom_tool_information``,
    or ``None`` on a cache miss.
    """
    client = _get_client()
    raw = client.get(f"template:{template_id}")
    if raw is None:
        return None
    return json.loads(raw)


def set_template_cache(template_id: str, data: Dict) -> None:
    """
    Store template configuration in Redis.

    TTL is 1 hour — short enough to pick up template edits without
    requiring a manual cache invalidation.
    """
    client = _get_client()
    client.setex(
        f"template:{template_id}",
        _TTL_TEMPLATE,
        json.dumps(data),
    )


# ---------------------------------------------------------------------------
# Conversation cache  —  conversation:{conv_id}
# ---------------------------------------------------------------------------

def get_conversation_cache(conv_id: str) -> Optional[Dict]:
    """
    Retrieve cached conversation state.

    Returns
    -------
    dict with keys:
      - ``summary``  (str)       : the current running summary
      - ``messages`` (list[dict]): last 20 message dicts
    or ``None`` on a cache miss.
    """
    client = _get_client()
    raw = client.get(f"conversation:{conv_id}")
    if raw is None:
        return None
    return json.loads(raw)


def set_conversation_cache(
    conv_id: str,
    summary: str,
    messages: List[Dict],
) -> None:
    """
    Overwrite the conversation cache entry.

    Only the last 20 messages are stored — older messages are dropped.
    The TTL is refreshed on every write.

    Parameters
    ----------
    conv_id : str
        Conversation identifier.
    summary : str
        Current running summary (may be empty string).
    messages : list[dict]
        Full recent message list; only the last 20 are kept.
    """
    client = _get_client()
    payload = {
        "summary":  summary,
        "messages": messages[-20:],  # keep last 20 (= 10 turns)
    }
    client.setex(
        f"conversation:{conv_id}",
        _TTL_CONVERSATION,
        json.dumps(payload, default=str),
    )
