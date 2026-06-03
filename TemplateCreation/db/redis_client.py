"""
db/redis_client.py
------------------
Redis conversation cache for the TemplateCreation service.

Each template's conversation history is stored as a JSON array under:

    tmpl_conv:{template_id}

A module-level ``ConnectionPool`` is shared across all calls so that
new TCP sockets are not opened on every invocation.

Environment
-----------
  REDIS_URL   e.g. redis://localhost:6379
"""

import json
import os
from typing import List, Dict, Optional

import redis
from redis import ConnectionPool, Redis


# ---------------------------------------------------------------------------
# Connection pool (module-level singleton)
# ---------------------------------------------------------------------------

_pool: Optional[ConnectionPool] = None


def _get_client() -> Redis:
    """Return a Redis client backed by the shared connection pool."""
    global _pool
    if _pool is None:
        url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        _pool = ConnectionPool.from_url(url, decode_responses=True)
    return Redis(connection_pool=_pool)


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

_KEY_PREFIX = "tmpl_conv"
_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days


def _key(template_id: str) -> str:
    return f"{_KEY_PREFIX}:{template_id}"


# ---------------------------------------------------------------------------
# Public cache API
# ---------------------------------------------------------------------------

def get_conversation_cache(template_id: str) -> Optional[List[Dict]]:
    """
    Retrieve the cached conversation for ``template_id``.

    Returns
    -------
    list[dict] or None
        The cached message list on a hit, or ``None`` on a miss.
        Each dict has keys: ``role``, ``content``, ``sequence_number``,
        ``token_count``.
    """
    client = _get_client()
    raw = client.get(_key(template_id))
    if raw is None:
        return None
    return json.loads(raw)


def set_conversation_cache(template_id: str, messages: List[Dict]) -> None:
    """
    Overwrite the entire cached conversation for ``template_id``.

    The TTL is reset on every write.
    """
    client = _get_client()
    client.setex(_key(template_id), _TTL_SECONDS, json.dumps(messages, default=str))


def append_to_conversation_cache(
    template_id: str,
    new_messages: List[Dict],
) -> None:
    """
    Append ``new_messages`` to an existing cache entry.

    If no cache entry exists yet, one is created from ``new_messages``
    alone.  The TTL is refreshed on every write.
    """
    client = _get_client()
    raw = client.get(_key(template_id))
    existing: List[Dict] = json.loads(raw) if raw else []
    existing.extend(new_messages)
    client.setex(_key(template_id), _TTL_SECONDS, json.dumps(existing, default=str))


def invalidate_conversation_cache(template_id: str) -> None:
    """Delete the cache entry for ``template_id``."""
    client = _get_client()
    client.delete(_key(template_id))
