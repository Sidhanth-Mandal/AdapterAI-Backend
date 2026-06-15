"""
db.py — PostgreSQL data-access layer for the ToolGeneration pipeline.

All SQL operations are isolated here so the rest of the pipeline never
touches psycopg2 directly.

Functions
---------
get_connection()                          → psycopg2 connection
fetch_template(template_id)               → dict | None
get_next_tool_id()                        → str  (e.g. "to00003")
insert_tool(...)                          → None
update_template_tool_information(...)     → None
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

_DSN: str = os.environ["POSTGRES_DSN"]


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection() -> psycopg2.extensions.connection:
    """Return a new psycopg2 connection using POSTGRES_DSN from .env."""
    return psycopg2.connect(_DSN)


# ---------------------------------------------------------------------------
# Template queries
# ---------------------------------------------------------------------------

def fetch_template(template_id: str) -> dict | None:
    """
    Fetch a single template row from the Templates table.

    Returns
    -------
    dict
        Row as a plain dict, or None if not found.
    """
    sql = """
        SELECT
            template_id,
            name,
            description,
            behaviour_prompt,
            tool_generation_prompt,
            tool_information,
            created_by,
            created_at,
            updated_at
        FROM templates
        WHERE template_id = %s
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (template_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def update_template_tool_information(
    template_id: str,
    tool_information: str,
) -> None:
    """
    Write the generated tool capability summary back to the originating
    template row so the orchestrator can find it quickly.

    Parameters
    ----------
    template_id : str
        Primary key of the template to update.
    tool_information : str
        Multi-line capability summary produced by extract_tool_metadata().
    """
    sql = """
        UPDATE templates
        SET    tool_information = %s,
               updated_at      = NOW()
        WHERE  template_id = %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (tool_information, template_id))
        conn.commit()


# ---------------------------------------------------------------------------
# Tool ID generation
# ---------------------------------------------------------------------------

def get_next_tool_id() -> str:
    """
    Return the next incremental tool_id in the format ``toNNNNN``.

    Scans all existing tool_ids that match ``to<digits>``, finds the
    maximum integer suffix, and returns suffix + 1 zero-padded to 5 digits.

    Examples
    --------
    No tools yet     → "to00001"
    Max is "to00003" → "to00004"
    """
    sql = "SELECT tool_id FROM tools WHERE tool_id ~ '^to[0-9]{5}$'"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    if not rows:
        return "to00001"

    max_n = 0
    for (tid,) in rows:
        m = re.fullmatch(r"to(\d{5})", tid)
        if m:
            max_n = max(max_n, int(m.group(1)))

    return f"to{max_n + 1:05d}"


# ---------------------------------------------------------------------------
# Tool insert
# ---------------------------------------------------------------------------

def insert_tool(
    tool_id: str,
    template_id: str,
    name: str,
    description: str,
    language: str,
    tool_json: dict,
    version: str = "1.0.0",
) -> None:
    """
    Insert a newly generated tool record into the Tools table.

    Parameters
    ----------
    tool_id     : str   e.g. "to00003"
    template_id : str   FK → Templates.template_id
    name        : str   from extract_tool_metadata()
    description : str   from extract_tool_metadata()
    language    : str   e.g. "python"
    tool_json   : dict  full generated tool dict (stored as JSONB)
    version     : str   default "1.0.0"
    """
    sql = """
        INSERT INTO tools (
            tool_id,
            template_id,
            name,
            description,
            language,
            tool_json,
            version,
            created_at,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                tool_id,
                template_id,
                name,
                description,
                language,
                json.dumps(tool_json),
                version,
            ))
        conn.commit()
