"""
test.py
-------
End-to-end sanity check for the TemplateCreation service layer.

Tests (in order)
----------------
  1.  Environment variables present
  2.  PostgreSQL connectivity
  3.  Redis connectivity
  4.  LangGraph graph compilation
  5.  TEMP_MESSAGES insert / read / sequence number
  6.  Redis cache  set / get / append / invalidate
  7.  chat_template()   -- Phase 1 (one real LLM turn)
  8.  create_template() -- Phase 2 (direct call)
  9.  Finalization guard (chat_template() after Phase 2 must return error)
 10.  Cleanup (delete all test rows)

Run from TemplateCreation/
    python test.py
"""

import os
import sys
import io
import uuid
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Force UTF-8 stdout so LLM responses with special chars don't crash on
# Windows cp1252 terminals
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Ensure TemplateCreation/ is importable
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
_ROOT_ENV = _HERE.parent / ".env"
load_dotenv(dotenv_path=_ROOT_ENV if _ROOT_ENV.exists() else None, override=False)

# ---------------------------------------------------------------------------
# Colour helpers (plain ASCII fallback on Windows cp1252)
# ---------------------------------------------------------------------------
PASS  = "[PASS]"
FAIL  = "[FAIL]"
INFO  = "[INFO]"
SEP   = "-" * 60


def _safe(text: str) -> str:
    """Strip / replace characters that can't be printed on this terminal."""
    return text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
        sys.stdout.encoding or "utf-8"
    )

# ---------------------------------------------------------------------------
# Unique IDs for this test run (avoids collisions with real data)
# ---------------------------------------------------------------------------
TEST_TEMPLATE_ID = f"test_{uuid.uuid4().hex[:12]}"
TEST_USER_ID     = None   # resolved in test_postgres_connectivity()


# ===========================================================================
# Helper
# ===========================================================================

def _header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _ok(msg: str) -> bool:
    print(f"  {PASS}  {msg}")
    return True


def _fail(msg: str, exc: Exception = None) -> bool:
    print(f"  {FAIL}  {msg}")
    if exc:
        print(f"         {type(exc).__name__}: {exc}")
    return False


# ===========================================================================
# Test 1 — Environment variables
# ===========================================================================

def test_env() -> bool:
    _header("1. Environment Variables")
    all_ok = True
    for var in ("GROQ_API_KEY", "POSTGRES_DSN", "REDIS_URL"):
        val = os.environ.get(var)
        if val:
            preview = val[:12] + "..." if len(val) > 12 else val
            _ok(f"{var} = {preview}")
        else:
            _fail(f"{var} is NOT set")
            all_ok = False
    return all_ok


# ===========================================================================
# Test 2 — PostgreSQL connectivity + fetch test user_id
# ===========================================================================

def test_postgres_connectivity() -> bool:
    global TEST_USER_ID
    _header("2. PostgreSQL Connectivity")
    try:
        import psycopg2
        conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
        cur  = conn.cursor()

        # Verify required tables exist
        for table in ("users", "templates", "temp_messages"):
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name=%s",
                (table,),
            )
            if cur.fetchone():
                _ok(f"Table '{table}' exists")
            else:
                _fail(f"Table '{table}' NOT found")
                conn.close()
                return False

        # Grab a real user_id for FK-safe template inserts
        cur.execute("SELECT user_id FROM Users LIMIT 1")
        row = cur.fetchone()
        if row:
            TEST_USER_ID = row[0]
            _ok(f"Test user_id resolved: {TEST_USER_ID[:24]}...")
        else:
            _fail("Users table is empty — run seed_users.py first")
            conn.close()
            return False

        cur.close()
        conn.close()
        return True
    except Exception as e:
        return _fail("Could not connect to PostgreSQL", e)


# ===========================================================================
# Test 3 — Redis connectivity
# ===========================================================================

def test_redis_connectivity() -> bool:
    _header("3. Redis Connectivity")
    try:
        import redis as _redis
        r = _redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        pong = r.ping()
        if pong:
            return _ok("Redis PING -> PONG")
        return _fail("Redis PING returned False")
    except Exception as e:
        return _fail("Could not connect to Redis", e)


# ===========================================================================
# Test 4 — Graph compilation
# ===========================================================================

def test_graph_compilation() -> bool:
    _header("4. LangGraph Compilation")
    try:
        from graph import build_graph
        g = build_graph()
        _ok(f"Graph compiled: {type(g).__name__}")

        # Check expected nodes exist
        node_names = list(g.nodes)
        for node in ("chatbot_node", "planner_node"):
            if node in node_names:
                _ok(f"Node '{node}' registered")
            else:
                _fail(f"Node '{node}' missing from graph")
                return False
        return True
    except Exception as e:
        return _fail("Graph compilation failed", e)


# ===========================================================================
# Test 5 — TEMP_MESSAGES CRUD
# ===========================================================================

def test_temp_messages_crud() -> bool:
    _header("5. TEMP_MESSAGES CRUD")
    try:
        from db.postgres_client import (
            insert_message,
            get_messages,
            get_next_sequence_number,
        )

        # Insert 3 messages
        roles_contents = [
            ("user",      "Hello, I need a Python coding assistant."),
            ("assistant", "Great! What kind of Python tasks do you need help with?"),
            ("user",      "Mainly debugging and code reviews."),
        ]
        for i, (role, content) in enumerate(roles_contents, start=1):
            mid = insert_message(TEST_TEMPLATE_ID, role, content, i, token_count=len(content.split()))
            _ok(f"Inserted message seq={i}  role={role}  id={mid[:8]}...")

        # Read back
        rows = get_messages(TEST_TEMPLATE_ID)
        assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
        _ok(f"get_messages() returned {len(rows)} rows in correct order")

        # Sequence number
        nxt = get_next_sequence_number(TEST_TEMPLATE_ID)
        assert nxt == 4, f"Expected next_seq=4, got {nxt}"
        _ok(f"get_next_sequence_number() = {nxt}")

        return True
    except AssertionError as e:
        return _fail(f"Assertion failed: {e}")
    except Exception as e:
        return _fail("TEMP_MESSAGES CRUD error", e)


# ===========================================================================
# Test 6 — Redis cache operations
# ===========================================================================

def test_redis_cache() -> bool:
    _header("6. Redis Cache Operations")
    try:
        from db.redis_client import (
            get_conversation_cache,
            set_conversation_cache,
            append_to_conversation_cache,
            invalidate_conversation_cache,
        )

        key = f"test_{uuid.uuid4().hex[:8]}"
        msgs = [
            {"role": "user",      "content": "Hi",     "sequence_number": 1, "token_count": 1},
            {"role": "assistant", "content": "Hello!", "sequence_number": 2, "token_count": 2},
        ]

        # Miss
        assert get_conversation_cache(key) is None
        _ok("Cache miss on fresh key")

        # Set
        set_conversation_cache(key, msgs)
        cached = get_conversation_cache(key)
        assert cached is not None and len(cached) == 2
        _ok(f"set + get: {len(cached)} messages cached")

        # Append
        append_to_conversation_cache(key, [
            {"role": "user", "content": "More", "sequence_number": 3, "token_count": 1}
        ])
        cached = get_conversation_cache(key)
        assert len(cached) == 3
        _ok(f"append_to_conversation_cache: now {len(cached)} messages")

        # Invalidate
        invalidate_conversation_cache(key)
        assert get_conversation_cache(key) is None
        _ok("invalidate_conversation_cache: key gone")

        return True
    except AssertionError as e:
        return _fail(f"Assertion failed: {e}")
    except Exception as e:
        return _fail("Redis cache error", e)


# ===========================================================================
# Test 7 — chat_template() Phase 1 (real LLM call)
# ===========================================================================

def test_chat_template_phase1() -> bool:
    _header("7. chat_template() -- Phase 1 (real LLM call)")
    print(f"  {INFO}  template_id = {TEST_TEMPLATE_ID}")
    print(f"  {INFO}  This makes a real Groq API call -- may take a few seconds...")
    try:
        from service import chat_template

        response = chat_template(
            template_id=TEST_TEMPLATE_ID,
            user_id=TEST_USER_ID,
            user_prompt=(
                "Hello! I need a coding assistant that specialises in Python. "
                "It should help with debugging, code reviews, and best practices."
            ),
        )

        assert isinstance(response, str) and len(response) > 0, "Empty response"
        _ok(f"Got AI response ({len(response)} chars)")

        # Preview first 120 chars
        preview = _safe(response[:120].replace("\n", " "))
        print(f"\n  {INFO}  Response preview:\n         \"{preview}...\"\n")

        # Verify message was persisted
        from db.postgres_client import get_messages
        rows = get_messages(TEST_TEMPLATE_ID)
        # We already inserted 3 rows in test 5, now expect at least 5 (3 + user + assistant)
        assert len(rows) >= 5, f"Expected >=5 rows after chat turn, got {len(rows)}"
        _ok(f"TEMP_MESSAGES now has {len(rows)} rows (user + AI persisted)")

        return True
    except AssertionError as e:
        return _fail(f"Assertion failed: {e}")
    except Exception as e:
        return _fail("chat_template() Phase 1 error", e)


# ===========================================================================
# Test 8 — create_template() Phase 2 (direct call)
# ===========================================================================

def test_create_template_phase2() -> bool:
    _header("8. create_template() -- Phase 2 (real LLM call)")
    print(f"  {INFO}  This calls planner_node + name/description Groq call...")
    try:
        from service import create_template
        from db.postgres_client import get_messages, is_template_finalized

        # Use a separate template_id to avoid the guard blocking us
        phase2_id = f"test_{uuid.uuid4().hex[:12]}"

        conv_history = [
            {"role": "user",      "content": "I need a Python coding assistant.",    "sequence_number": 1, "token_count": 9},
            {"role": "assistant", "content": "What kind of Python tasks?",            "sequence_number": 2, "token_count": 7},
            {"role": "user",      "content": "Debugging, code reviews, type hints.",  "sequence_number": 3, "token_count": 7},
            {"role": "assistant", "content": "Any specific frameworks like FastAPI?",  "sequence_number": 4, "token_count": 7},
            {"role": "user",      "content": "Yes, FastAPI and SQLAlchemy.",           "sequence_number": 5, "token_count": 6},
        ]

        create_template(
            user_id=TEST_USER_ID,
            template_id=phase2_id,
            template_conv_history=conv_history,
        )

        # Verify row in Templates table
        assert is_template_finalized(phase2_id), "Template not found in Templates table after create_template()"
        _ok("Template row inserted into Templates table")

        # Check name / description were generated
        import psycopg2
        conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
        cur  = conn.cursor()
        cur.execute("SELECT name, description, system_prompt, tool_generation_prompt FROM Templates WHERE template_id=%s", (phase2_id,))
        row = cur.fetchone()
        conn.close()

        assert row, "No row returned from Templates"
        name, desc, sp, tgp = row
        _ok(f"name                  = {name}")
        _ok(f"description           = {(desc or '')[:80]}")
        _ok(f"system_prompt         = {len(sp or '')} chars")
        _ok(f"tool_generation_prompt= {len(tgp or '')} chars")

        # Cleanup this phase2_id
        import psycopg2 as _pg
        conn2 = _pg.connect(os.environ["POSTGRES_DSN"])
        cur2  = conn2.cursor()
        cur2.execute("DELETE FROM Templates WHERE template_id=%s", (phase2_id,))
        conn2.commit()
        conn2.close()
        _ok("Phase 2 test row cleaned up")

        return True
    except AssertionError as e:
        return _fail(f"Assertion failed: {e}")
    except Exception as e:
        return _fail("create_template() Phase 2 error", e)


# ===========================================================================
# Test 9 — Finalization guard
# ===========================================================================

def test_finalization_guard() -> bool:
    _header("9. Finalization Guard")
    print(f"  {INFO}  Inserting a dummy Templates row then calling chat_template()...")
    try:
        from service import chat_template
        from db.postgres_client import insert_template, is_template_finalized

        guard_id = f"test_{uuid.uuid4().hex[:12]}"

        # Mark template as finalised
        insert_template(
            template_id=guard_id,
            user_id=TEST_USER_ID,
            name="Guard Test Template",
            description="Created to verify the finalization guard.",
            system_prompt="You are a test assistant.",
            tool_generation_prompt="Generate no tools.",
        )
        assert is_template_finalized(guard_id)
        _ok("Template pre-finalised in DB")

        # chat_template() must return the error string, not invoke the graph
        response = chat_template(
            template_id=guard_id,
            user_id=TEST_USER_ID,
            user_prompt="Can I still edit this?",
        )

        assert "no longer" in response.lower() or "already" in response.lower(), \
            f"Unexpected response: {response}"
        _ok(f"Guard triggered correctly: \"{response}\"")

        # Cleanup
        import psycopg2
        conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
        cur  = conn.cursor()
        cur.execute("DELETE FROM Templates WHERE template_id=%s", (guard_id,))
        conn.commit()
        conn.close()
        _ok("Guard test row cleaned up")

        return True
    except AssertionError as e:
        return _fail(f"Assertion failed: {e}")
    except Exception as e:
        return _fail("Finalization guard test error", e)


# ===========================================================================
# Test 10 — Cleanup
# ===========================================================================

def test_cleanup() -> bool:
    _header("10. Cleanup (removing test rows)")
    try:
        import psycopg2
        conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
        cur  = conn.cursor()

        cur.execute("DELETE FROM TEMP_MESSAGES WHERE template_id=%s", (TEST_TEMPLATE_ID,))
        deleted_msgs = cur.rowcount
        _ok(f"Deleted {deleted_msgs} rows from TEMP_MESSAGES")

        cur.execute("DELETE FROM Templates WHERE template_id=%s", (TEST_TEMPLATE_ID,))
        deleted_tmpl = cur.rowcount
        _ok(f"Deleted {deleted_tmpl} rows from Templates (if any)")

        conn.commit()
        conn.close()

        # Invalidate Redis cache for test template
        from db.redis_client import invalidate_conversation_cache
        invalidate_conversation_cache(TEST_TEMPLATE_ID)
        _ok("Redis cache invalidated for test template_id")

        return True
    except Exception as e:
        return _fail("Cleanup error", e)


# ===========================================================================
# Runner
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 60)
    print("  TemplateCreation -- Service Layer Test Suite")
    print("=" * 60)
    print(f"  test_template_id : {TEST_TEMPLATE_ID}")

    tests = [
        ("Environment Variables",        test_env),
        ("PostgreSQL Connectivity",       test_postgres_connectivity),
        ("Redis Connectivity",            test_redis_connectivity),
        ("Graph Compilation",             test_graph_compilation),
        ("TEMP_MESSAGES CRUD",            test_temp_messages_crud),
        ("Redis Cache Operations",        test_redis_cache),
        ("chat_template() Phase 1",       test_chat_template_phase1),
        ("create_template() Phase 2",     test_create_template_phase2),
        ("Finalization Guard",            test_finalization_guard),
        ("Cleanup",                       test_cleanup),
    ]

    results = []
    for name, fn in tests:
        try:
            passed = fn()
        except Exception:
            print(f"\n  {FAIL}  Unhandled exception in '{name}':")
            traceback.print_exc()
            passed = False
        results.append((name, passed))

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("  All tests passed.")
    else:
        failed = sum(1 for _, p in results if not p)
        print(f"  {failed} test(s) failed. See output above for details.")
    print("=" * 60 + "\n")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
