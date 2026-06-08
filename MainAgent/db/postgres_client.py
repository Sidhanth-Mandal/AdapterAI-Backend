"""
MainAgent/db/postgres_client.py
--------------------------------
All PostgreSQL operations required by the MainAgent pipeline.

Uses a module-level ThreadedConnectionPool (psycopg2) — same pattern as
TemplateCreation/db/postgres_client.py — so connections are reused across
calls without opening a new TCP socket per invocation.

All public functions are synchronous; async nodes call them via
``asyncio.to_thread()`` to avoid blocking the event loop.

Tables accessed
---------------
  templates           — template configuration (behaviour_prompt, tool_information)
  conversations       — last_message_at update
  messages            — message persistence and retrieval
  conversation_memory — summary, token count, and summarization bookmarks

Environment
-----------
  POSTGRES_DSN   e.g. postgresql://postgres:password@localhost:5432/app_db
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from psycopg2.pool import ThreadedConnectionPool

# ---------------------------------------------------------------------------
# Bootstrap — load .env from project root
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # AdapterAI/
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Connection pool (module-level singleton)
# ---------------------------------------------------------------------------

_pool: Optional[ThreadedConnectionPool] = None


def _get_pool() -> ThreadedConnectionPool:
    """Return the shared connection pool, initialising it on first call."""
    global _pool
    if _pool is None:
        dsn = os.environ["POSTGRES_DSN"]
        _pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=dsn)
    return _pool


@contextmanager
def _get_connection() -> Iterator[psycopg2.extensions.connection]:
    """
    Yield a psycopg2 connection from the pool.

    Commits on success, rolls back on exception, and always returns
    the connection to the pool.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
# Token counting helper (used by persist_messages_node)
# ---------------------------------------------------------------------------

try:
    import tiktoken as _tiktoken

    _ENCODING = _tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        """Return the cl100k_base token count for ``text``."""
        return len(_ENCODING.encode(text))

except ImportError:
    # Graceful degradation — word-count heuristic if tiktoken is absent
    def count_tokens(text: str) -> int:  # type: ignore[misc]
        return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

def fetch_template(template_id: str) -> Optional[Dict]:
    """
    Load template configuration from the Templates table.

    Maps the DB column names to the spec's Python field names:
      behaviour_prompt  → behavior_prompt
      tool_information  → custom_tool_information

    Returns
    -------
    dict with keys ``behavior_prompt`` and ``custom_tool_information``,
    or ``None`` if the template_id does not exist.
    """
    sql = """
        SELECT behaviour_prompt, tool_information
        FROM   templates
        WHERE  template_id = %s
        LIMIT  1
    """
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (template_id,))
            row = cur.fetchone()

    if row is None:
        return None

    return {
        "behavior_prompt":        row["behaviour_prompt"] or "",
        "custom_tool_information": row["tool_information"] or "",
    }


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------

def fetch_conversation_memory(conv_id: str) -> Optional[Dict]:
    """
    Fetch the conversation_memory row for ``conv_id``.

    Returns
    -------
    dict with keys: summary, last_summarized_message_id,
    last_summarized_message_seq, unsummarized_token_count
    — or ``None`` if no row exists yet.
    """
    sql = """
        SELECT summary,
               last_summarized_message_id,
               last_summarized_message_seq,
               unsummarized_token_count
        FROM   conversation_memory
        WHERE  conversation_id = %s
        LIMIT  1
    """
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (conv_id,))
            row = cur.fetchone()

    return dict(row) if row is not None else None


def ensure_conversation_memory_exists(conv_id: str) -> None:
    """
    Guarantee that a conversation_memory row exists for ``conv_id``.

    Uses INSERT … ON CONFLICT DO NOTHING — safe to call every request.
    """
    sql = """
        INSERT INTO conversation_memory
            (conversation_id, summary, last_summarized_message_seq,
             unsummarized_token_count)
        VALUES (%s, '', 0, 0)
        ON CONFLICT (conversation_id) DO NOTHING
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (conv_id,))


def update_conversation_memory(
    conv_id: str,
    new_summary: str,
    last_message_id: str,
    last_message_seq: int,
    unsummarized_token_count: int,
) -> None:
    """
    Write an updated summary and reset the unsummarized token counter.

    Called by summary_node after a successful summarization run.
    """
    sql = """
        UPDATE conversation_memory
        SET
            summary                      = %s,
            last_summarized_message_id   = %s,
            last_summarized_message_seq  = %s,
            unsummarized_token_count     = %s,
            updated_at                   = NOW()
        WHERE conversation_id = %s
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                new_summary,
                last_message_id,
                last_message_seq,
                unsummarized_token_count,
                conv_id,
            ))


def update_unsummarized_token_count(conv_id: str, new_count: int) -> None:
    """
    Update only the unsummarized_token_count for a conversation.

    Called by persist_messages_node after each successful turn to keep
    the running total accurate without triggering a full summary update.
    """
    sql = """
        UPDATE conversation_memory
        SET
            unsummarized_token_count = %s,
            updated_at               = NOW()
        WHERE conversation_id = %s
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (new_count, conv_id))


# ---------------------------------------------------------------------------
# Message retrieval
# ---------------------------------------------------------------------------

def fetch_recent_messages(conv_id: str, limit: int = 20) -> List[Dict]:
    """
    Fetch the most recent ``limit`` messages for a conversation.

    Rows are fetched in DESC order then reversed so the returned list
    is chronological (oldest first) — ready to pass to the LLM.

    Returns
    -------
    list[dict] — each dict has: message_id, conv_id, role, content,
    token_count, sequence_number, created_at
    """
    sql = """
        SELECT message_id, conv_id, role, content,
               token_count, sequence_number, created_at
        FROM   messages
        WHERE  conv_id = %s
        ORDER  BY sequence_number DESC
        LIMIT  %s
    """
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (conv_id, limit))
            rows = cur.fetchall()

    # Reverse to chronological order
    return [dict(r) for r in reversed(rows)]


def fetch_unsummarized_messages(conv_id: str, after_seq: int) -> List[Dict]:
    """
    Fetch all messages with sequence_number > ``after_seq``, in order.

    Used by summary_node to collect the messages that need to be
    incorporated into the next summary update.

    Returns
    -------
    list[dict] — each dict has: message_id, role, content,
    token_count, sequence_number, created_at
    """
    sql = """
        SELECT message_id, role, content,
               token_count, sequence_number, created_at
        FROM   messages
        WHERE  conv_id        = %s
          AND  sequence_number > %s
        ORDER  BY sequence_number ASC
    """
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (conv_id, after_seq))
            rows = cur.fetchall()

    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Message persistence
# ---------------------------------------------------------------------------

def get_next_sequence_number(conv_id: str) -> int:
    """
    Return the next available sequence number for a conversation.

    Returns 1 when no messages exist yet.
    """
    sql = """
        SELECT COALESCE(MAX(sequence_number), 0) + 1
        FROM   messages
        WHERE  conv_id = %s
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (conv_id,))
            result = cur.fetchone()

    return result[0] if result else 1


def insert_message(
    conv_id: str,
    role: str,
    content: str,
    token_count: int,
    sequence_number: int,
) -> str:
    """
    Insert a single message into the Messages table.

    Returns
    -------
    str
        The auto-generated ``message_id`` (UUID4).
    """
    message_id = str(uuid.uuid4())
    sql = """
        INSERT INTO messages
            (message_id, conv_id, role, content, token_count, sequence_number)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (message_id, conv_id, role, content, token_count, sequence_number),
            )

    return message_id


def update_conversation_last_message_at(conv_id: str) -> None:
    """
    Stamp the current time as last_message_at on the Conversations row.

    Called once per turn after both user and assistant messages are persisted.
    """
    sql = """
        UPDATE conversations
        SET    last_message_at = NOW()
        WHERE  conv_id = %s
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (conv_id,))
