"""
MainAgent/nodes/load_template_node.py
--------------------------------------
Graph node: load_template_node

Loads template configuration (behavior_prompt, custom_tool_information)
using a Redis-first, PostgreSQL-fallback strategy.

Redis key : template:{template_id}
DB table  : templates (columns: behaviour_prompt, tool_information)

On a cache miss the loaded values are written back to Redis before
the function returns so the next request is served from cache.
"""

from __future__ import annotations

import asyncio

from MainAgent.db.postgres_client import fetch_template
from MainAgent.db.redis_client import get_template_cache, set_template_cache
from MainAgent.state import OrchestratorState
from utils.tracing import traceable


@traceable(name="load_template_node", tags=["main-agent", "node"])
async def load_template_node(state: OrchestratorState) -> dict:
    """
    Load template configuration.

    Returns partial state update with:
      - ``behavior_prompt``        : custom persona / behaviour instructions
      - ``custom_tool_information``: description of the custom tool(s)

    If the template is not found in Redis or PostgreSQL, both fields
    default to empty strings so the pipeline continues gracefully.
    """
    template_id = state["template_id"]

    # ── 1. Try Redis ──────────────────────────────────────────────────────────
    cached = await asyncio.to_thread(get_template_cache, template_id)
    if cached is not None:
        return {
            "behavior_prompt":         cached.get("behavior_prompt", ""),
            "custom_tool_information": cached.get("custom_tool_information", ""),
        }

    # ── 2. Fallback to PostgreSQL ─────────────────────────────────────────────
    row = await asyncio.to_thread(fetch_template, template_id)

    if row is None:
        # Unknown template — continue with empty config rather than raising
        return {
            "behavior_prompt":         "",
            "custom_tool_information": "",
        }

    # ── 3. Populate Redis for future requests ─────────────────────────────────
    await asyncio.to_thread(set_template_cache, template_id, row)

    return {
        "behavior_prompt":         row["behavior_prompt"],
        "custom_tool_information": row["custom_tool_information"],
    }
