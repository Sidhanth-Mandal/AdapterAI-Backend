"""
redis_client.py — Redis error-history layer for the ToolGeneration pipeline.

Each pipeline run maintains a Redis List keyed by a temporary run key
(until the tool_id is known) and then renamed to the final tool_id.

List layout
-----------
Index 0  : {"type": "prompt",   "content": "<tool_generation_prompt>"}
Index 1+ : {"type": "validation_error" | "execution_error" | "repair",
             ...stage-specific fields...}

Functions
---------
get_redis()                              → redis.Redis client
init_error_history(key, prompt_text)     → None
append_error_history(key, entry: dict)   → None
get_error_history(key)                   → list[dict]
rename_key(old_key, new_key)             → None
"""

from __future__ import annotations

import json
import os
from typing import Any

import redis
from dotenv import load_dotenv

load_dotenv()

_REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def get_redis() -> redis.Redis:
    """Return a Redis client connected via REDIS_URL from .env."""
    return redis.from_url(_REDIS_URL, decode_responses=True)


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------

def init_error_history(key: str, prompt_text: str) -> None:
    """
    Initialise (or reset) an error-history list in Redis.

    Deletes any pre-existing key, then pushes the tool_generation_prompt
    as the mandatory first entry.

    Parameters
    ----------
    key         : str   Redis key (temporary run key)
    prompt_text : str   The tool_generation_prompt fetched from the DB
    """
    r = get_redis()
    r.delete(key)
    first_entry: dict[str, Any] = {
        "type": "prompt",
        "content": prompt_text,
    }
    r.rpush(key, json.dumps(first_entry))


def append_error_history(key: str, entry: dict) -> None:
    """
    Append a JSON-serialisable entry to the history list.

    Typical entry shapes
    --------------------
    Validation failure:
        {
            "type":   "validation_error",
            "errors": [...],
            "stage":  "Schema Integrity"
        }
    Execution failure:
        {
            "type":   "execution_error",
            "error":  "..."
        }
    Repair applied:
        {
            "type":  "repair",
            "cause": ["...", "..."],
            "fix":   ["...", "..."]
        }

    Parameters
    ----------
    key   : str   Redis key
    entry : dict  Must be JSON-serialisable
    """
    r = get_redis()
    r.rpush(key, json.dumps(entry))


def get_error_history(key: str) -> list[dict]:
    """
    Return the full error history for this key as a list of dicts.

    Returns an empty list if the key does not exist.
    """
    r = get_redis()
    raw_entries = r.lrange(key, 0, -1)
    return [json.loads(e) for e in raw_entries]


def rename_key(old_key: str, new_key: str) -> None:
    """
    Atomically move the error-history list from its temporary run key
    to the permanent tool_id key.

    If ``old_key`` does not exist this is a no-op (avoids crashing when
    the run failed before any history was written).

    Parameters
    ----------
    old_key : str   Temporary run key, e.g. "gen:tem00001:a1b2c3d4"
    new_key : str   Final tool_id,       e.g. "to00003"
    """
    r = get_redis()
    if r.exists(old_key):
        # If new_key already exists, delete it first to avoid RENAME collision
        r.delete(new_key)
        r.rename(old_key, new_key)
