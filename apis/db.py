"""
apis/db.py
----------
Shared PostgreSQL connection pool for the API layer.

Uses the same psycopg2 ThreadedConnectionPool pattern as
MainAgent/db/postgres_client.py and TemplateCreation/db/postgres_client.py.

All public functions are synchronous; FastAPI async endpoints call them
via asyncio.to_thread() to avoid blocking the event loop.

Tables accessed
---------------
  users           — authentication (user_id, password_hash)
  conversations   — conv_id, user_id, template_id, title
  messages        — chat history for a given conv_id
  templates       — templates owned by a user (for loadtemplate/)
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

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # AdapterAI/
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
# Auth helpers
# ---------------------------------------------------------------------------

def fetch_user_by_username(username: str) -> Optional[Dict]:
    """
    Return the user row for ``username``, or None if not found.

    Returns dict with keys: user_id, username, email, password_hash.
    """
    sql = """
        SELECT user_id, username, email, password_hash
        FROM   users
        WHERE  username = %s
        LIMIT  1
    """
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (username,))
            row = cur.fetchone()
    return dict(row) if row else None


def create_user(username: str, email: str, password_hash: str) -> str:
    """
    Insert a new user into the Users table.
    Returns the newly created user_id.
    """
    user_id = str(uuid.uuid4())
    sql = """
        INSERT INTO users (user_id, username, email, password_hash)
        VALUES (%s, %s, %s, %s)
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, username, email, password_hash))
    return user_id


def fetch_user_by_id(user_id: str) -> Optional[Dict]:
    """Return the user row for ``user_id``, or None if not found."""
    sql = """
        SELECT user_id, username, email
        FROM   users
        WHERE  user_id = %s
        LIMIT  1
    """
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (user_id,))
            row = cur.fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def conversation_exists(conv_id: str) -> bool:
    """Return True if a row for ``conv_id`` exists in the Conversations table."""
    sql = "SELECT 1 FROM conversations WHERE conv_id = %s LIMIT 1"
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (conv_id,))
            return cur.fetchone() is not None


def create_conversation(
    conv_id: str,
    user_id: str,
    template_id: str,
    title: Optional[str] = None,
) -> None:
    """
    Insert a new row into the Conversations table.

    Safe to call only when the conv_id does not yet exist.
    """
    sql = """
        INSERT INTO conversations (conv_id, user_id, template_id, title)
        VALUES (%s, %s, %s, %s)
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (conv_id, user_id, template_id, title))


def fetch_conversations_for_user(user_id: str) -> List[Dict]:
    """
    Return all conversations for ``user_id``, ordered newest-first.

    Each dict contains: conv_id, title, template_id, created_at, last_message_at.
    """
    sql = """
        SELECT conv_id, title, template_id, created_at, last_message_at
        FROM   conversations
        WHERE  user_id = %s
        ORDER  BY last_message_at DESC NULLS LAST
    """
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (user_id,))
            rows = cur.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def fetch_messages_for_conversation(conv_id: str) -> List[Dict]:
    """
    Return all messages for ``conv_id`` in chronological order.

    Each dict contains: message_id, conv_id, role, content,
    token_count, sequence_number, created_at.
    """
    sql = """
        SELECT message_id, conv_id, role, content,
               token_count, sequence_number, created_at
        FROM   messages
        WHERE  conv_id = %s
        ORDER  BY sequence_number ASC
    """
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (conv_id,))
            rows = cur.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def fetch_templates_for_user(user_id: str) -> List[Dict]:
    """
    Return all templates created by ``user_id``, ordered newest-first.

    Each dict contains: template_id, name, description, created_at, updated_at.
    """
    sql = """
        SELECT template_id, name, description, created_at, updated_at
        FROM   templates
        WHERE  created_by = %s
        ORDER  BY created_at DESC
    """
    with _get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (user_id,))
            rows = cur.fetchall()
    return [dict(r) for r in rows]
