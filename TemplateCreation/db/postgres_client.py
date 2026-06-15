"""
db/postgres_client.py
---------------------
Centralised PostgreSQL client for the TemplateCreation service.

Uses a ThreadedConnectionPool (psycopg2) so connections are reused
across calls without opening a new TCP connection per invocation.

Tables managed here
-------------------
  TEMP_MESSAGES  — conversation turns during Phase 1 gathering
  Templates      — final generated template artifacts from Phase 2

Environment
-----------
  POSTGRES_DSN   e.g. postgresql://postgres:password@localhost:5432/app_db
"""

import os
import uuid
from contextlib import contextmanager
from typing import Iterator, List, Dict, Optional

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool


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
def get_connection() -> Iterator[psycopg2.extensions.connection]:
    """
    Yield a psycopg2 connection from the pool.

    Automatically commits on success and rolls back on exception,
    then returns the connection to the pool.
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
# DDL
# ---------------------------------------------------------------------------

_DDL_TEMP_MESSAGES = """
CREATE TABLE IF NOT EXISTS TEMP_MESSAGES (
    message_id      VARCHAR PRIMARY KEY,
    template_id     VARCHAR NOT NULL,
    role            VARCHAR NOT NULL,
    content         TEXT    NOT NULL,
    token_count     INT,
    sequence_number INT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_temp_messages_template_id
    ON TEMP_MESSAGES (template_id);
"""

_DDL_TEMPLATES = """
CREATE TABLE IF NOT EXISTS Templates (
    template_id             VARCHAR PRIMARY KEY,
    name                    VARCHAR NOT NULL,
    description             TEXT,
    behaviour_prompt        TEXT,
    tool_generation_prompt  TEXT,
    tool_information        TEXT,
    created_by              VARCHAR NOT NULL,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def ensure_tables() -> None:
    """
    Create TEMP_MESSAGES and Templates tables if they do not exist.

    Safe to call multiple times — all statements use IF NOT EXISTS.
    Intended to be called once at service startup.
    """
    print("[TC:pg]   ensure_tables() — verifying TEMP_MESSAGES and Templates tables exist …")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL_TEMP_MESSAGES)
            cur.execute(_DDL_TEMPLATES)
    print("[TC:pg]   ensure_tables() — tables OK")


# ---------------------------------------------------------------------------
# TEMP_MESSAGES helpers
# ---------------------------------------------------------------------------

def insert_message(
    template_id: str,
    role: str,
    content: str,
    sequence_number: int,
    token_count: Optional[int] = None,
) -> str:
    """
    Insert a single message into TEMP_MESSAGES.

    Parameters
    ----------
    template_id : str
        The template this message belongs to.
    role : str
        One of ``user``, ``assistant``, or ``system``.
    content : str
        The full message text.
    sequence_number : int
        Monotonically increasing turn index within the conversation.
    token_count : int, optional
        Pre-computed token count; stored as-is (no validation).

    Returns
    -------
    str
        The auto-generated ``message_id`` (UUID4).
    """
    message_id = str(uuid.uuid4())
    sql = """
        INSERT INTO TEMP_MESSAGES
            (message_id, template_id, role, content, token_count, sequence_number)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    print(f"[TC:pg]   insert_message | template_id={template_id!r} role={role!r} seq={sequence_number} tokens={token_count}")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (message_id, template_id, role, content, token_count, sequence_number),
            )
    print(f"[TC:pg]   insert_message done | message_id={message_id!r}")
    return message_id


def get_messages(template_id: str) -> List[Dict]:
    """
    Fetch all messages for a template ordered by ``sequence_number`` ascending.

    Returns
    -------
    list[dict]
        Each dict contains: ``message_id``, ``role``, ``content``,
        ``token_count``, ``sequence_number``, ``created_at``.
    """
    sql = """
        SELECT message_id, role, content, token_count, sequence_number, created_at
        FROM TEMP_MESSAGES
        WHERE template_id = %s
        ORDER BY sequence_number ASC
    """
    print(f"[TC:pg]   get_messages | template_id={template_id!r}")
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (template_id,))
            rows = cur.fetchall()
    print(f"[TC:pg]   get_messages returned {len(rows)} row(s)")
    return [dict(r) for r in rows]


def get_next_sequence_number(template_id: str) -> int:
    """
    Return the next available sequence number for ``template_id``.

    Returns 1 when no messages exist yet.
    """
    sql = """
        SELECT COALESCE(MAX(sequence_number), 0) + 1
        FROM TEMP_MESSAGES
        WHERE template_id = %s
    """
    print(f"[TC:pg]   get_next_sequence_number | template_id={template_id!r}")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (template_id,))
            result = cur.fetchone()
    seq = result[0] if result else 1
    print(f"[TC:pg]   next sequence number: {seq}")
    return seq


# ---------------------------------------------------------------------------
# Templates helpers
# ---------------------------------------------------------------------------

def is_template_finalized(template_id: str) -> bool:
    """
    Return ``True`` if a row for ``template_id`` already exists in the
    Templates table, meaning Phase 2 has completed and the template is
    locked for editing.

    This is used by ``chat_template()`` to reject further conversational
    turns after the template has been generated.
    """
    sql = "SELECT 1 FROM Templates WHERE template_id = %s LIMIT 1"
    print(f"[TC:pg]   is_template_finalized | template_id={template_id!r}")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (template_id,))
            finalized = cur.fetchone() is not None
    print(f"[TC:pg]   is_template_finalized result: {finalized}")
    return finalized


def insert_template(
    template_id: str,
    user_id: str,
    name: str,
    description: str,
    behaviour_prompt: str,
    tool_generation_prompt: str,
) -> None:
    """
    Upsert a generated template into the Templates table.

    ``tool_information`` is intentionally omitted from this INSERT —
    it is populated by a separate downstream pipeline.

    An ON CONFLICT clause updates all mutable fields if the template
    already exists (idempotent re-runs are safe).
    """
    sql = """
        INSERT INTO Templates
            (template_id, created_by, name, description,
             behaviour_prompt, tool_generation_prompt)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (template_id) DO UPDATE SET
            name                   = EXCLUDED.name,
            description            = EXCLUDED.description,
            behaviour_prompt       = EXCLUDED.behaviour_prompt,
            tool_generation_prompt = EXCLUDED.tool_generation_prompt,
            updated_at             = CURRENT_TIMESTAMP
    """
    print(f"[TC:pg]   insert_template | template_id={template_id!r} user_id={user_id!r} name={name!r}")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                template_id,
                user_id,
                name,
                description,
                behaviour_prompt,
                tool_generation_prompt,
            ))
    print(f"[TC:pg]   insert_template done | template '{template_id}' upserted successfully")
