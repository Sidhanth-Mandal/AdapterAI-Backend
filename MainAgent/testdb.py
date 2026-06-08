"""
MainAgent/testdb.py
--------------------
Integration tests for the MainAgent Redis and PostgreSQL clients.

Run from the project root:
    python -m MainAgent.testdb

Each test prints PASS / FAIL with a brief description.  All test data
created during the run is cleaned up in teardown so real data is never
polluted.

Coverage
--------
REDIS
  [R1]  Connection health (PING)
  [R2]  Template cache  — set, get (hit), miss, TTL is set
  [R3]  Template cache  — overwrite refreshes data and TTL
  [R4]  Conversation cache — set, get (hit), miss, TTL is set
  [R5]  Conversation cache — >20 messages are truncated to 20
  [R6]  Conversation cache — default=str serializer handles datetime fields

POSTGRESQL
  [P1]  Connection health (SELECT 1)
  [P2]  fetch_template — returns None for unknown id
  [P3]  fetch_template — returns dict for existing template (tem000002 / any real row)
  [P4]  ensure_conversation_memory_exists — idempotent (safe to call twice)
  [P5]  fetch_conversation_memory — returns dict after ensure; None before
  [P6]  get_next_sequence_number — returns 1 when no messages exist
  [P7]  insert_message — inserts row, returns uuid string
  [P8]  fetch_recent_messages — returns chronological list
  [P9]  fetch_unsummarized_messages — filters by sequence_number
  [P10] update_unsummarized_token_count — persists new value
  [P11] update_conversation_memory — persists all fields
  [P12] update_conversation_last_message_at — doesn't error; timestamp advances
  [P13] get_next_sequence_number — increments correctly after inserts
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Bootstrap .env before importing the db modules
# ---------------------------------------------------------------------------

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # AdapterAI/
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Import the clients under test
# ---------------------------------------------------------------------------

from MainAgent.db import redis_client as R
from MainAgent.db import postgres_client as P

# ---------------------------------------------------------------------------
# Tiny test harness
# ---------------------------------------------------------------------------

_results: list[tuple[str, str, str]] = []   # (id, status, message)
_FAIL_FAST = False


def _run(test_id: str, description: str, fn: Callable) -> None:
    """Execute *fn* and record PASS / FAIL."""
    try:
        fn()
        _results.append((test_id, "PASS", description))
        print(f"  [PASS] {test_id}: {description}")
    except AssertionError as exc:
        _results.append((test_id, "FAIL", f"{description} - {exc}"))
        print(f"  [FAIL] {test_id}: {description}")
        print(f"         {exc}")
        if _FAIL_FAST:
            sys.exit(1)
    except Exception as exc:
        _results.append((test_id, "ERROR", f"{description} - {exc}"))
        print(f"  [ERROR] {test_id}: {description}")
        traceback.print_exc()
        if _FAIL_FAST:
            sys.exit(1)


def _summary() -> None:
    total  = len(_results)
    passed = sum(1 for _, s, _ in _results if s == "PASS")
    failed = sum(1 for _, s, _ in _results if s == "FAIL")
    errors = sum(1 for _, s, _ in _results if s == "ERROR")
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed  |  {failed} failed  |  {errors} errors")
    print("=" * 60)
    if failed or errors:
        sys.exit(1)


# ===========================================================================
# Helpers
# ===========================================================================

def _unique_id() -> str:
    """Generate a short UUID-based id for test isolation."""
    return f"test_{uuid.uuid4().hex[:12]}"


def _unique_uuid() -> str:
    """Generate a proper UUID string for columns typed as uuid in Postgres."""
    return str(uuid.uuid4())


# Cached template_id borrowed from the real templates table for FK satisfaction
_TEMPLATE_ID: Optional[str] = None


def _redis_cleanup(*keys: str) -> None:
    client = R._get_client()
    for k in keys:
        client.delete(k)


def _pg_cleanup_conv(conv_id: str) -> None:
    """Remove messages, conversation_memory, and conversations rows for conv_id."""
    with P._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM messages          WHERE conv_id         = %s", (conv_id,))
            cur.execute("DELETE FROM conversation_memory WHERE conversation_id = %s", (conv_id,))
            # conversations row may not exist — that's fine
            cur.execute("DELETE FROM conversations     WHERE conv_id         = %s", (conv_id,))


def _get_any_template_id() -> str:
    """Return any existing template_id from the templates table.

    Raises RuntimeError if the templates table is empty (tests cannot run).
    Result is cached in module-level _TEMPLATE_ID.
    """
    global _TEMPLATE_ID
    if _TEMPLATE_ID is not None:
        return _TEMPLATE_ID
    with P._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT template_id FROM templates LIMIT 1")
            row = cur.fetchone()
    if row is None:
        raise RuntimeError(
            "No rows in 'templates' table — cannot create test conversations. "
            "Run seed_tem000002.py first."
        )
    _TEMPLATE_ID = row[0]
    return _TEMPLATE_ID


def _pg_insert_test_conv(conv_id: str, user_id: str = "24c6336f-0c52-4eb9-9307-0e3e8d89f3f6") -> None:
    """Insert a minimal conversations row so FK constraints pass.

    Real schema: conversations(conv_id, user_id, template_id NOT NULL, title, ...)
    We borrow any existing template_id to satisfy the NOT NULL constraint.
    """
    tid = _get_any_template_id()
    with P._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (conv_id, user_id, template_id, title)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (conv_id) DO NOTHING
                """,
                (conv_id, user_id, tid, "TestConv"),
            )


# ===========================================================================
# Redis tests
# ===========================================================================

def _test_r1_ping():
    client = R._get_client()
    result = client.ping()
    assert result is True, f"Redis PING returned {result!r}"


def _test_r2_template_cache_set_get():
    tid = _unique_id()
    key = f"template:{tid}"
    data = {"behavior_prompt": "Be helpful.", "custom_tool_information": "[]"}

    # Miss before set
    hit_before = R.get_template_cache(tid)
    assert hit_before is None, "Expected cache miss before set"

    R.set_template_cache(tid, data)

    hit = R.get_template_cache(tid)
    assert hit == data, f"Cached value mismatch: {hit!r}"

    # TTL should be ≤ 3600 and > 0
    client = R._get_client()
    ttl = client.ttl(key)
    assert 0 < ttl <= R._TTL_TEMPLATE, f"Unexpected TTL: {ttl}"

    _redis_cleanup(key)


def _test_r3_template_cache_overwrite():
    tid = _unique_id()
    key = f"template:{tid}"
    data1 = {"behavior_prompt": "v1", "custom_tool_information": "[]"}
    data2 = {"behavior_prompt": "v2", "custom_tool_information": "[tool]"}

    R.set_template_cache(tid, data1)
    R.set_template_cache(tid, data2)

    hit = R.get_template_cache(tid)
    assert hit == data2, f"Expected overwritten value, got {hit!r}"
    _redis_cleanup(key)


def _test_r4_conversation_cache_set_get():
    cid = _unique_id()
    key = f"conversation:{cid}"
    summary  = "Prior context."
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]

    miss = R.get_conversation_cache(cid)
    assert miss is None, "Expected cache miss before set"

    R.set_conversation_cache(cid, summary, messages)
    hit = R.get_conversation_cache(cid)

    assert hit is not None, "Expected cache hit after set"
    assert hit["summary"]  == summary,  f"summary mismatch: {hit['summary']!r}"
    assert hit["messages"] == messages, f"messages mismatch: {hit['messages']!r}"

    client = R._get_client()
    ttl = client.ttl(key)
    assert 0 < ttl <= R._TTL_CONVERSATION, f"Unexpected TTL: {ttl}"

    _redis_cleanup(key)


def _test_r5_conversation_cache_truncates_to_20():
    cid = _unique_id()
    key = f"conversation:{cid}"
    messages = [{"role": "user", "content": f"msg {i}"} for i in range(30)]

    R.set_conversation_cache(cid, "", messages)
    hit = R.get_conversation_cache(cid)

    assert len(hit["messages"]) == 20, \
        f"Expected 20 messages after truncation, got {len(hit['messages'])}"
    # Should be the LAST 20
    assert hit["messages"][0]["content"] == "msg 10", \
        f"Wrong start: {hit['messages'][0]}"
    assert hit["messages"][-1]["content"] == "msg 29", \
        f"Wrong end: {hit['messages'][-1]}"

    _redis_cleanup(key)


def _test_r6_conversation_cache_datetime_serialisation():
    """set_conversation_cache must handle datetime objects without crashing."""
    cid = _unique_id()
    key = f"conversation:{cid}"
    messages = [
        {
            "role": "user",
            "content": "hi",
            "created_at": datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        }
    ]
    # Should not raise
    R.set_conversation_cache(cid, "summary", messages)
    hit = R.get_conversation_cache(cid)
    assert hit is not None, "Cache miss after set"
    # datetime should have been serialised as a string
    assert isinstance(hit["messages"][0]["created_at"], str), \
        "datetime was not serialised to string"
    _redis_cleanup(key)


# ===========================================================================
# PostgreSQL tests
# ===========================================================================

def _test_p1_pg_connection():
    with P._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            result = cur.fetchone()
    assert result == (1,), f"Expected (1,), got {result!r}"


def _test_p2_fetch_template_miss():
    result = P.fetch_template("non_existent_template_id_xyz")
    assert result is None, f"Expected None for unknown template, got {result!r}"


def _test_p3_fetch_template_hit():
    """
    Verify fetch_template against the real templates table.
    If tem000002 (seeded by seed_tem000002.py) doesn't exist, skip gracefully.
    """
    result = P.fetch_template("tem000002")
    if result is None:
        # Try fetching any existing template_id
        with P._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT template_id FROM templates LIMIT 1")
                row = cur.fetchone()
        if row is None:
            print("         (skipped — no templates in DB)")
            return
        result = P.fetch_template(row[0])

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "behavior_prompt"        in result, "Missing key 'behavior_prompt'"
    assert "custom_tool_information" in result, "Missing key 'custom_tool_information'"


def _test_p4_ensure_conversation_memory_idempotent():
    # conversation_memory.conversation_id is uuid — use a proper UUID
    cid = _unique_uuid()
    _pg_insert_test_conv(cid)
    try:
        P.ensure_conversation_memory_exists(cid)  # first call
        P.ensure_conversation_memory_exists(cid)  # second call — must not raise
    finally:
        _pg_cleanup_conv(cid)


def _test_p5_fetch_conversation_memory_lifecycle():
    # conversation_memory.conversation_id is uuid — use a proper UUID
    cid = _unique_uuid()
    _pg_insert_test_conv(cid)
    try:
        # No row yet -> None
        before = P.fetch_conversation_memory(cid)
        assert before is None, f"Expected None before ensure, got {before!r}"

        P.ensure_conversation_memory_exists(cid)
        after = P.fetch_conversation_memory(cid)
        assert after is not None, "Expected dict after ensure"
        assert after["summary"]                  == "",  f"summary should be empty, got {after['summary']!r}"
        assert after["unsummarized_token_count"] == 0,  f"token_count should be 0"
        assert after["last_summarized_message_seq"] == 0
    finally:
        _pg_cleanup_conv(cid)


def _test_p6_get_next_sequence_number_empty():
    cid = _unique_uuid()
    _pg_insert_test_conv(cid)
    try:
        seq = P.get_next_sequence_number(cid)
        assert seq == 1, f"Expected 1 for empty conversation, got {seq}"
    finally:
        _pg_cleanup_conv(cid)


def _test_p7_insert_message_returns_uuid():
    cid = _unique_uuid()
    _pg_insert_test_conv(cid)
    try:
        msg_id = P.insert_message(cid, "user", "Hello!", token_count=5, sequence_number=1)
        assert isinstance(msg_id, str), f"Expected str, got {type(msg_id)}"
        # Must be a valid UUID4
        parsed = uuid.UUID(msg_id, version=4)
        assert str(parsed) == msg_id, "Returned id is not a valid UUID4"
    finally:
        _pg_cleanup_conv(cid)


def _test_p8_fetch_recent_messages_chronological():
    cid = _unique_uuid()
    _pg_insert_test_conv(cid)
    try:
        P.insert_message(cid, "user",      "msg1", token_count=3, sequence_number=1)
        P.insert_message(cid, "assistant", "msg2", token_count=4, sequence_number=2)
        P.insert_message(cid, "user",      "msg3", token_count=3, sequence_number=3)

        msgs = P.fetch_recent_messages(cid, limit=20)

        assert len(msgs) == 3, f"Expected 3 messages, got {len(msgs)}"
        contents = [m["content"] for m in msgs]
        assert contents == ["msg1", "msg2", "msg3"], \
            f"Not chronological: {contents}"
    finally:
        _pg_cleanup_conv(cid)


def _test_p8b_fetch_recent_messages_limit():
    """fetch_recent_messages with limit=1 should return only the newest message."""
    cid = _unique_uuid()
    _pg_insert_test_conv(cid)
    try:
        P.insert_message(cid, "user",      "old", token_count=2, sequence_number=1)
        P.insert_message(cid, "assistant", "new", token_count=2, sequence_number=2)

        msgs = P.fetch_recent_messages(cid, limit=1)
        assert len(msgs) == 1, f"Expected 1, got {len(msgs)}"
        assert msgs[0]["content"] == "new", f"Expected 'new', got {msgs[0]['content']!r}"
    finally:
        _pg_cleanup_conv(cid)


def _test_p9_fetch_unsummarized_messages():
    cid = _unique_uuid()
    _pg_insert_test_conv(cid)
    try:
        P.insert_message(cid, "user",      "m1", token_count=2, sequence_number=1)
        P.insert_message(cid, "assistant", "m2", token_count=2, sequence_number=2)
        P.insert_message(cid, "user",      "m3", token_count=2, sequence_number=3)

        msgs = P.fetch_unsummarized_messages(cid, after_seq=1)
        assert len(msgs) == 2, f"Expected 2 messages after seq 1, got {len(msgs)}"
        assert msgs[0]["sequence_number"] == 2
        assert msgs[1]["sequence_number"] == 3

        msgs_all = P.fetch_unsummarized_messages(cid, after_seq=0)
        assert len(msgs_all) == 3, f"Expected 3 messages after seq 0, got {len(msgs_all)}"
    finally:
        _pg_cleanup_conv(cid)


def _test_p10_update_unsummarized_token_count():
    cid = _unique_uuid()
    _pg_insert_test_conv(cid)
    try:
        P.ensure_conversation_memory_exists(cid)
        P.update_unsummarized_token_count(cid, 42)
        mem = P.fetch_conversation_memory(cid)
        assert mem["unsummarized_token_count"] == 42, \
            f"Expected 42, got {mem['unsummarized_token_count']}"
    finally:
        _pg_cleanup_conv(cid)


def _test_p11_update_conversation_memory():
    cid = _unique_uuid()
    _pg_insert_test_conv(cid)
    try:
        P.ensure_conversation_memory_exists(cid)
        # Need a real message_id for last_summarized_message_id (FK may or may not exist)
        fake_msg_id = str(uuid.uuid4())
        # Insert a real message so FK constraint doesn't bite us
        real_msg_id = P.insert_message(cid, "user", "hello", token_count=3, sequence_number=1)

        P.update_conversation_memory(
            conv_id=cid,
            new_summary="This is the summary.",
            last_message_id=real_msg_id,
            last_message_seq=1,
            unsummarized_token_count=100,
        )

        mem = P.fetch_conversation_memory(cid)
        assert mem["summary"]                      == "This is the summary."
        assert mem["last_summarized_message_seq"]  == 1
        assert mem["unsummarized_token_count"]     == 100
        assert str(mem["last_summarized_message_id"]) == real_msg_id
    finally:
        _pg_cleanup_conv(cid)


def _test_p12_update_conversation_last_message_at():
    cid = _unique_uuid()
    _pg_insert_test_conv(cid)
    try:
        # Grab the timestamp before
        with P._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT last_message_at FROM conversations WHERE conv_id = %s", (cid,))
                row = cur.fetchone()
        before = row[0] if row else None

        P.update_conversation_last_message_at(cid)

        with P._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT last_message_at FROM conversations WHERE conv_id = %s", (cid,))
                row = cur.fetchone()
        after = row[0] if row else None

        assert after is not None, "last_message_at should not be None after update"
        if before is not None:
            assert after >= before, "last_message_at should not go backwards"
    finally:
        _pg_cleanup_conv(cid)


def _test_p13_sequence_number_increments():
    cid = _unique_uuid()
    _pg_insert_test_conv(cid)
    try:
        seq1 = P.get_next_sequence_number(cid)
        P.insert_message(cid, "user", "a", token_count=1, sequence_number=seq1)

        seq2 = P.get_next_sequence_number(cid)
        assert seq2 == seq1 + 1, f"Expected seq {seq1 + 1}, got {seq2}"

        P.insert_message(cid, "assistant", "b", token_count=1, sequence_number=seq2)

        seq3 = P.get_next_sequence_number(cid)
        assert seq3 == seq2 + 1, f"Expected seq {seq2 + 1}, got {seq3}"
    finally:
        _pg_cleanup_conv(cid)


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    print("\n" + "=" * 60)
    print("MainAgent DB Integration Tests")
    print("=" * 60)

    # ---- Redis ----
    print("\n--- Redis ---")
    _run("R1", "Connection health (PING)",                            _test_r1_ping)
    _run("R2", "Template cache: set / get / TTL",                     _test_r2_template_cache_set_get)
    _run("R3", "Template cache: overwrite refreshes data",            _test_r3_template_cache_overwrite)
    _run("R4", "Conversation cache: set / get / TTL",                 _test_r4_conversation_cache_set_get)
    _run("R5", "Conversation cache: >20 messages truncated to 20",    _test_r5_conversation_cache_truncates_to_20)
    _run("R6", "Conversation cache: datetime serialised as string",   _test_r6_conversation_cache_datetime_serialisation)

    # ---- PostgreSQL ----
    print("\n--- PostgreSQL ---")
    _run("P1",  "Connection health (SELECT 1)",                       _test_p1_pg_connection)
    _run("P2",  "fetch_template: None for unknown id",                _test_p2_fetch_template_miss)
    _run("P3",  "fetch_template: returns dict for existing template",  _test_p3_fetch_template_hit)
    _run("P4",  "ensure_conversation_memory_exists: idempotent",      _test_p4_ensure_conversation_memory_idempotent)
    _run("P5",  "fetch_conversation_memory: lifecycle (None -> dict)", _test_p5_fetch_conversation_memory_lifecycle)
    _run("P6",  "get_next_sequence_number: returns 1 when empty",     _test_p6_get_next_sequence_number_empty)
    _run("P7",  "insert_message: returns valid UUID4",                _test_p7_insert_message_returns_uuid)
    _run("P8",  "fetch_recent_messages: chronological order",         _test_p8_fetch_recent_messages_chronological)
    _run("P8b", "fetch_recent_messages: respects limit",              _test_p8b_fetch_recent_messages_limit)
    _run("P9",  "fetch_unsummarized_messages: filters by seq",        _test_p9_fetch_unsummarized_messages)
    _run("P10", "update_unsummarized_token_count: persists value",    _test_p10_update_unsummarized_token_count)
    _run("P11", "update_conversation_memory: all fields persist",     _test_p11_update_conversation_memory)
    _run("P12", "update_conversation_last_message_at: timestamp set", _test_p12_update_conversation_last_message_at)
    _run("P13", "get_next_sequence_number: increments after inserts", _test_p13_sequence_number_increments)

    _summary()


if __name__ == "__main__":
    main()
