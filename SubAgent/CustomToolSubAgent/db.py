"""
SubAgent/CustomToolSubAgent/db.py — Minimal DB layer for the CustomToolSubAgent.

Two queries:
  1. fetch_template_id_for_conv(conv_id)  → str
         Conversations.conv_id → Conversations.template_id

  2. fetch_tool_json_for_template(template_id) → dict
         Tools.template_id → Tools.tool_json  (JSONB column, returned as dict)

Both use POSTGRES_DSN from the project .env file.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # AdapterAI/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

_DSN: str = os.environ["POSTGRES_DSN"]


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _get_connection() -> psycopg2.extensions.connection:
    """Return a fresh psycopg2 connection using POSTGRES_DSN."""
    return psycopg2.connect(_DSN)


# ---------------------------------------------------------------------------
# Public queries
# ---------------------------------------------------------------------------

def fetch_template_id_for_conv(conv_id: str) -> str:
    """
    Look up the template_id associated with a conversation.

    Parameters
    ----------
    conv_id : str
        Primary key of the Conversations row.

    Returns
    -------
    str
        The template_id linked to that conversation.

    Raises
    ------
    ValueError
        If conv_id is not found in the Conversations table.
    """
    sql = """
        SELECT template_id
        FROM   conversations
        WHERE  conv_id = %s
        LIMIT  1
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (conv_id,))
            row = cur.fetchone()

    if row is None:
        raise ValueError(
            f"No conversation found with conv_id='{conv_id}'. "
            "Ensure the conversation exists in the Conversations table."
        )
    return row[0]


def fetch_tool_json_for_template(template_id: str) -> dict:
    """
    Retrieve the tool_json JSONB column from the Tools table for a given template.

    If multiple tool rows exist for the same template_id (e.g. multiple
    versions), the most recently updated one is returned.

    Parameters
    ----------
    template_id : str
        FK → Templates.template_id

    Returns
    -------
    dict
        The parsed tool JSON schema dict (the content of the tool_json column).

    Raises
    ------
    ValueError
        If no Tools row exists for this template_id, or tool_json is NULL.
    """
    sql = """
        SELECT tool_json
        FROM   tools
        WHERE  template_id = %s
        ORDER  BY updated_at DESC
        LIMIT  1
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (template_id,))
            row = cur.fetchone()

    if row is None:
        raise ValueError(
            f"No tool found for template_id='{template_id}'. "
            "Ensure a tool has been generated and stored for this template."
        )

    tool_json = row[0]
    if tool_json is None:
        raise ValueError(
            f"tool_json column is NULL for template_id='{template_id}'."
        )

    # psycopg2 returns JSONB columns as dicts already; handle string fallback
    if isinstance(tool_json, dict):
        return tool_json
    try:
        return json.loads(tool_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            f"tool_json for template_id='{template_id}' could not be parsed as JSON: {exc}"
        ) from exc
