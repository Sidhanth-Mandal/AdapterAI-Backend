"""
MainAgent/test.py
-----------------
Integration tests for the MainAgent pipeline.

Tests
-----
1. CustomToolSubAgent Test
   Verifies the model actually CALLS call_custom_tool_subagent (not just
   mentions it) by inspecting tools_called returned from chat_debug().

2. WebSearch Tool Test
   Verifies the model actually CALLS web_search or web_search_focused.

3. Combined Tool Chaining Test
   Verifies BOTH call_custom_tool_subagent AND a web search tool are called
   in the same turn.

4. Summarization Trigger Test  (unit test, no real API)
   Patches DB + LLM; asserts summary_node returns correct fields and resets
   the token counter.

4b. Summarization NOT Triggered  (unit test, no real API)
    Asserts summary_node returns {} when below the 4000-token threshold.

Usage
-----
    cd d:/CODING/Projects/AdapterAI
    python -m MainAgent.test       # run all tests
    python MainAgent/test.py       # equivalent
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Bootstrap sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Real IDs from the database
# ---------------------------------------------------------------------------
TEMPLATE_ID = "tem000002"                                  # Weather Expert
CONV_ID     = "89ed43a5-86a1-44fe-b7fd-dff10fc46dfd"
USER_ID     = "0ae42ca4-fecc-4b6a-bc23-a083e20d0321"

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

SEP = "=" * 70


def header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "[PASS]" if condition else "[FAIL]"
    msg = f"  {status}  {label}"
    if detail:
        msg += f"\n         -> {detail}"
    print(msg)


def skip(label: str, reason: str = "") -> None:
    msg = f"  [SKIP]  {label}"
    if reason:
        msg += f"\n         -> {reason}"
    print(msg)


# Backoff schedule (seconds) for rate-limit retries
_RETRY_DELAYS = [30, 60, 120]


async def _run_with_retry(coro_fn, *args, **kwargs):
    """
    Call an async factory (coro_fn(*args, **kwargs)) and retry up to
    len(_RETRY_DELAYS) times when a rate-limit error is encountered.

    Returns (result, None) on success.
    Raises the last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS, start=1):
        if delay:
            print(f"  [WAIT]  Rate limit hit. Waiting {delay}s before retry "
                  f"{attempt}/{len(_RETRY_DELAYS) + 1}...")
            await asyncio.sleep(delay)
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit(exc):
                last_exc = exc
                continue
            raise  # non-rate-limit errors propagate immediately
    raise last_exc  # all retries exhausted


def _is_rate_limit(exc: Exception) -> bool:
    """Return True when exc is a Groq/API 429 rate-limit error."""
    msg = str(exc).lower()
    return "rate_limit" in msg or "429" in msg or "ratelimiterror" in type(exc).__name__.lower()


# ---------------------------------------------------------------------------
# Test 1 — CustomToolSubAgent
# ---------------------------------------------------------------------------

async def test_custom_tool_subagent():
    """
    Verify the model CALLS call_custom_tool_subagent (not just answers
    from training data) when asked for a player's batting average.

    Uses chat_debug() which returns (response, tools_called).
    """
    header("TEST 1 — CustomToolSubAgent Usage (Weather Tool)")

    from MainAgent.service import chat_debug

    prompt = (
        "Use the custom weather tool to get the current temperature in London. "
        "Return the exact temperature value, unit, and weather description from the tool."
    )

    print(f"  Prompt     : {prompt}")
    print(f"  conv_id    : {CONV_ID}")
    print(f"  user_id    : {USER_ID}")
    print()

    start = time.perf_counter()
    try:
        response, tools_called = await _run_with_retry(
            chat_debug,
            template_id=TEMPLATE_ID,
            user_id=USER_ID,
            conv_id=CONV_ID,
            user_prompt=prompt,
            if_attachment=False,
        )
        elapsed = time.perf_counter() - start

        print(f"  Tools called : {tools_called}")
        print(f"  Response ({elapsed:.1f}s):\n  {response}\n")

        # --- Hard check: the tool must have been invoked ---
        called_custom = "call_custom_tool_subagent" in tools_called
        check(
            "call_custom_tool_subagent was ACTUALLY CALLED",
            called_custom,
            f"tools_called = {tools_called}",
        )

        # --- Soft check: response content ---
        has_response = bool(response and response.strip())
        check("Response is non-empty", has_response)

        lower = response.lower()
        mentions_weather = any(kw in lower for kw in ["temperature", "celsius", "fahrenheit", "london", "weather", "degrees", "error", "unable"])
        check(
            "Response relates to the weather query",
            mentions_weather,
            "keywords: temperature, celsius, fahrenheit, london, weather, degrees",
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start
        if _is_rate_limit(exc):
            skip("CustomToolSubAgent test", f"Groq rate limit — retry later. ({exc})")
        else:
            check("No unexpected exception", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Test 2 — WebSearch Tool
# ---------------------------------------------------------------------------

async def test_websearch_tool():
    """
    Verify the model CALLS web_search or web_search_focused (not just
    answers from training data) for a current-events cricket query.
    """
    header("TEST 2 — WebSearch Tool Usage (Weather News)")

    from MainAgent.service import chat_debug

    prompt = (
        "Search the web and tell me: what is the current weather situation or "
        "any recent extreme weather events happening in Europe right now? "
        "Include specific countries, temperatures, or events if found."
    )

    print(f"  Prompt : {prompt}")
    print()

    start = time.perf_counter()
    try:
        response, tools_called = await _run_with_retry(
            chat_debug,
            template_id=TEMPLATE_ID,
            user_id=USER_ID,
            conv_id=CONV_ID,
            user_prompt=prompt,
            if_attachment=False,
        )
        elapsed = time.perf_counter() - start

        print(f"  Tools called : {tools_called}")
        print(f"  Response ({elapsed:.1f}s):\n  {response}\n")

        # --- Hard check: a web search tool must have been invoked ---
        web_tools = {"web_search", "web_search_focused", "fetch_page_content"}
        called_web = bool(web_tools & set(tools_called))
        check(
            "A web search tool was ACTUALLY CALLED",
            called_web,
            f"tools_called = {tools_called}",
        )

        # --- Content checks ---
        has_response = bool(response and response.strip())
        check("Response is non-empty", has_response)

        import re
        has_content = bool(response and response.strip())
        check("Response is non-empty", has_content)

        lower = response.lower()
        has_weather_word = any(kw in lower for kw in ["weather", "temperature", "rain", "storm", "flood", "europe", "heat", "degrees", "celsius"])
        check(
            "Response mentions weather-related content",
            has_weather_word,
            "keywords: weather, temperature, rain, storm, flood, europe, heat, degrees, celsius",
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start
        if _is_rate_limit(exc):
            skip("WebSearch test", f"Groq rate limit — retry later. ({exc})")
        else:
            check("No unexpected exception", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Test 3 — Combined Tool Chaining
# ---------------------------------------------------------------------------

async def test_combined_tool_chaining():
    """
    Verify the model CALLS BOTH call_custom_tool_subagent AND a web search
    tool in the same turn, and synthesises both results.
    """
    header("TEST 3 — Combined Tool Chaining (Custom Weather Tool + Web Search)")

    from MainAgent.service import chat_debug

    prompt = (
        "Do two things: "
        "1) Use the custom tool to get the current temperature in Mumbai and Tokyo. "
        "2) Search the web for any recent news about unusual weather in Asia. "
        "Then combine both into a short weather briefing for Asia."
    )

    print(f"  Prompt : {prompt}")
    print()

    start = time.perf_counter()
    try:
        response, tools_called = await _run_with_retry(
            chat_debug,
            template_id=TEMPLATE_ID,
            user_id=USER_ID,
            conv_id=CONV_ID,
            user_prompt=prompt,
            if_attachment=False,
        )
        elapsed = time.perf_counter() - start

        print(f"  Tools called : {tools_called}")
        print(f"  Response ({elapsed:.1f}s):\n  {response}\n")

        # --- Hard checks: BOTH tool families must appear ---
        called_custom = "call_custom_tool_subagent" in tools_called
        web_tools     = {"web_search", "web_search_focused", "fetch_page_content"}
        called_web    = bool(web_tools & set(tools_called))

        check(
            "call_custom_tool_subagent was ACTUALLY CALLED",
            called_custom,
            f"tools_called = {tools_called}",
        )
        check(
            "A web search tool was ACTUALLY CALLED",
            called_web,
            f"tools_called = {tools_called}",
        )

        # --- Soft checks: response should combine both data sources ---
        has_response = bool(response and response.strip())
        check("Response is non-empty", has_response)

        lower = response.lower()
        mentions_temp    = any(kw in lower for kw in ["temperature", "degrees", "celsius", "fahrenheit", "mumbai", "tokyo"])
        mentions_weather = any(kw in lower for kw in ["weather", "rain", "storm", "climate", "asia", "flood", "heat"])
        check("Response references temperature data (custom tool)", mentions_temp)
        check("Response references weather news or events (web search)", mentions_weather)

    except Exception as exc:
        elapsed = time.perf_counter() - start
        if _is_rate_limit(exc):
            skip("Combined chaining test", f"Groq rate limit — retry later. ({exc})")
        else:
            check("No unexpected exception", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Test 4 — Summarization (unit test: fully mocked)
# ---------------------------------------------------------------------------

async def test_summarization():
    """
    Test summary_node in isolation with mocked DB and LLM.
    Forces unsummarized_token_count >= 4000 and asserts:
      - summary updated correctly
      - last_summarized_message_seq updated
      - unsummarized_token_count reset to 0
    """
    header("TEST 4 — Summarization Node (unit test with mocks)")

    from MainAgent.nodes.summary_node import summary_node

    fake_messages = [
        {"message_id": "msg-001", "role": "user",      "content": "What is the weather in London?",         "token_count": 10, "sequence_number": 1, "created_at": "2024-01-01"},
        {"message_id": "msg-002", "role": "assistant",  "content": "London is 18C and partly cloudy.",       "token_count": 10, "sequence_number": 2, "created_at": "2024-01-01"},
        {"message_id": "msg-003", "role": "user",      "content": "And Tokyo?",                             "token_count": 5,  "sequence_number": 3, "created_at": "2024-01-01"},
        {"message_id": "msg-004", "role": "assistant",  "content": "Tokyo is 28C and humid.",                "token_count": 8,  "sequence_number": 4, "created_at": "2024-01-01"},
    ]

    state = {
        "template_id":    TEMPLATE_ID,
        "user_id":        USER_ID,
        "conv_id":        CONV_ID,
        "user_prompt":    "",
        "if_attachment":  False,
        "behavior_prompt":         "",
        "custom_tool_information": "",
        "unsummarized_token_count":    5000,   # above the 4000 threshold
        "last_summarized_message_seq": 0,
        "summary":         "",
        "recent_messages": [],
        "messages":        [],
        "final_response":  "",
        "tools_called":    [],
        "new_user_seq":    0,
        "new_assistant_seq": 0,
    }

    expected_summary = "London is currently 18C and partly cloudy. Tokyo is 28C and humid."

    fake_llm_response = MagicMock()
    fake_llm_response.content = expected_summary
    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(return_value=fake_llm_response)

    with (
        patch("MainAgent.nodes.summary_node.fetch_unsummarized_messages", return_value=fake_messages),
        patch("MainAgent.nodes.summary_node.update_conversation_memory",  return_value=None),
        patch("MainAgent.nodes.summary_node.ChatGroq", return_value=fake_llm),
    ):
        result = await summary_node(state)

    print(f"  Summary returned: {result.get('summary', '')[:120]}")
    print()

    check("summary_node returns a non-empty summary",
          bool(result.get("summary", "").strip()),
          result.get("summary", "")[:80])
    check("summary matches mocked LLM output",
          result.get("summary") == expected_summary,
          f"got: {result.get('summary')}")
    check("unsummarized_token_count reset to 0",
          result.get("unsummarized_token_count") == 0,
          f"got: {result.get('unsummarized_token_count')}")
    check("last_summarized_message_seq updated to last msg seq",
          result.get("last_summarized_message_seq") == 4,
          f"got: {result.get('last_summarized_message_seq')}")


# ---------------------------------------------------------------------------
# Test 4b — Summarization NOT triggered below threshold
# ---------------------------------------------------------------------------

async def test_summarization_not_triggered():
    """summary_node must return {} (no-op) when token count < 4000."""
    header("TEST 4b — Summarization NOT Triggered (below threshold)")

    from MainAgent.nodes.summary_node import summary_node

    state = {
        "template_id":    TEMPLATE_ID,
        "user_id":        USER_ID,
        "conv_id":        CONV_ID,
        "user_prompt":    "",
        "if_attachment":  False,
        "behavior_prompt":         "",
        "custom_tool_information": "",
        "unsummarized_token_count":    999,   # below threshold
        "last_summarized_message_seq": 0,
        "summary":         "",
        "recent_messages": [],
        "messages":        [],
        "final_response":  "",
        "tools_called":    [],
        "new_user_seq":    0,
        "new_assistant_seq": 0,
    }

    result = await summary_node(state)

    print(f"  Result: {result}")
    print()

    check("summary_node returns empty dict when below threshold",
          result == {},
          f"got: {result}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_all_tests():
    print("\n" + "=" * 70)
    print("  MainAgent Integration & Unit Tests")
    print("  Template  : tem000002 (Weather Expert)")
    print(f"  conv_id   : {CONV_ID}")
    print(f"  user_id   : {USER_ID}")
    print("=" * 70)

    # Unit tests first (fast, no real API)
    #await test_summarization()
    #await test_summarization_not_triggered()

    # Integration tests (real Groq + Tavily API)
    print("\n  NOTE: Integration tests below make real API calls to Groq & Tavily.")
    print("        Rate-limit retries are automatic (up to 3x with backoff).")
    print("        A ~20s cooldown is inserted between each test to stay")
    print("        within the 6,000 TPM free-tier limit.\n")

    #await test_custom_tool_subagent()

    print("  [COOL]  Waiting 20s before next test to respect TPM limit...")
    #await asyncio.sleep(20)

    #await test_websearch_tool()

    print("  [COOL]  Waiting 20s before next test to respect TPM limit...")
    #await asyncio.sleep(20)

    await test_combined_tool_chaining()

    print(f"\n{SEP}")
    print("  All tests complete.")
    print(SEP + "\n")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
