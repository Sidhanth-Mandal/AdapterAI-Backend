"""
service.py
----------
Primary API entry point for the TemplateCreation pipeline.

Exposes two callable functions designed to be invoked from an API layer
(e.g., the ``templatechat`` router):

    chat_template(template_id, user_id, user_prompt) -> str
        Handles one conversational turn.  Loads conversation history
        from Redis (or falls back to PostgreSQL), invokes the LangGraph
        chatbot, persists every new message, and — when the chatbot
        signals satisfaction — automatically triggers Phase 2.

    create_template(user_id, template_id, template_conv_history) -> None
        Runs the Phase 2 planner directly, generates a name and
        description via a Groq call, and persists the final template
        record to the Templates table.

Design notes
------------
* LangGraph's built-in checkpointing is NOT used.  Conversation memory
  is managed exclusively through Redis (cache) and PostgreSQL (source
  of truth).
* The graph is compiled fresh on every chat_template() call — there is
  no persistent graph state between invocations.
* main.py has been removed.  This module is the sole entry point.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ToolGeneration.pipeline import generate_tool

# ---------------------------------------------------------------------------
# Ensure TemplateCreation/ is on sys.path so bare internal imports work
# (state, graph, nodes.*, utils.*, db.*)
# ---------------------------------------------------------------------------
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

# ---------------------------------------------------------------------------
# Load .env from the project root before any env-dependent imports
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

_ROOT_ENV = _MODULE_DIR.parent / ".env"
load_dotenv(dotenv_path=_ROOT_ENV if _ROOT_ENV.exists() else None, override=False)

# ---------------------------------------------------------------------------
# LangSmith tracing (must come after .env is loaded)
# ---------------------------------------------------------------------------
from utils.tracing import traceable  # noqa: E402

# ---------------------------------------------------------------------------
# Internal imports (env vars must be available first)
# ---------------------------------------------------------------------------
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from .graph import build_graph
from .nodes.planner_node import planner_node
from .db.postgres_client import (
    ensure_tables,
    get_messages,
    get_next_sequence_number,
    insert_message,
    insert_template,
    is_template_finalized,
)
from .db.redis_client import (
    append_to_conversation_cache,
    get_conversation_cache,
    set_conversation_cache,
)

# ---------------------------------------------------------------------------
# Token counting
# Token counting is done with tiktoken (cl100k_base).  tiktoken works
# entirely from an in-memory vocabulary — there are no network calls and
# latency impact is negligible (<1 ms per message).
# ---------------------------------------------------------------------------
try:
    import tiktoken as _tiktoken

    _ENCODING = _tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        """Return the cl100k_base token count for ``text``."""
        return len(_ENCODING.encode(text))

except ImportError:
    # Graceful degradation: word-count heuristic if tiktoken is absent.
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return int(len(text.split()) * 1.3)

# ---------------------------------------------------------------------------
# Ensure DB tables exist at import time (idempotent DDL)
# ---------------------------------------------------------------------------
ensure_tables()


# ===========================================================================
# Internal helpers
# ===========================================================================

def _to_langchain_messages(records: List[Dict]):
    """
    Convert a list of message dicts (from DB / Redis) into LangChain
    message objects suitable for passing to the graph state.

    Supported role values: user / human → HumanMessage
                           assistant / ai → AIMessage
                           system → SystemMessage
    """
    _role_map = {
        "user":      HumanMessage,
        "human":     HumanMessage,
        "assistant": AIMessage,
        "ai":        AIMessage,
        "system":    SystemMessage,
    }
    result = []
    for rec in records:
        cls = _role_map.get(rec["role"].lower(), HumanMessage)
        result.append(cls(content=rec["content"]))
    return result


def _to_cache_dicts(records: List[Dict]) -> List[Dict]:
    """
    Strip non-JSON-serialisable fields (e.g. datetime objects) from
    Postgres rows so they can be stored in Redis safely.
    """
    return [
        {
            "role":            rec["role"],
            "content":         rec["content"],
            "sequence_number": rec.get("sequence_number"),
            "token_count":     rec.get("token_count"),
        }
        for rec in records
    ]


def _load_history(template_id: str) -> List[Dict]:
    """
    Return conversation history as a list of dicts.

    Read path
    ---------
    1. Try Redis → return immediately on hit.
    2. On miss: load from PostgreSQL, populate Redis, return.
    """
    cached = get_conversation_cache(template_id)
    if cached is not None:
        return cached

    # Cache miss — hydrate from PostgreSQL
    rows = get_messages(template_id)
    safe = _to_cache_dicts(rows)
    if safe:
        set_conversation_cache(template_id, safe)
    return safe


def _persist_message(
    template_id: str,
    role: str,
    content: str,
    seq: int,
) -> Dict:
    """
    Write a single message to PostgreSQL.

    Returns a cache-ready dict (matching the Redis schema) for the
    caller to append to the Redis cache.
    """
    tokens = _count_tokens(content)
    insert_message(
        template_id=template_id,
        role=role,
        content=content,
        sequence_number=seq,
        token_count=tokens,
    )
    return {
        "role":            role,
        "content":         content,
        "sequence_number": seq,
        "token_count":     tokens,
    }


def _generate_name_and_description(
    conv_history: List[Dict],
) -> Tuple[str, str]:
    """
    Derive a template name and one-sentence description from the
    conversation history using a Groq LLM call.

    Uses ``llama-3.3-70b-versatile`` at low temperature for
    deterministic, structured output.  Only the first 15 turns of the
    conversation are sent to keep the prompt short.

    Returns
    -------
    tuple[str, str]
        ``(name, description)``
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=128,
        streaming=False,
        api_key=os.environ["GROQ_API_KEY"],
    )

    # Build a compact transcript (at most 15 turns)
    lines: List[str] = []
    for m in conv_history[:15]:
        label = "USER" if m["role"].lower() in ("user", "human") else "ASSISTANT"
        lines.append(f"[{label}]: {m['content']}")
    transcript = "\n".join(lines)

    prompt = (
        "Based on the following requirements gathering conversation, generate:\n"
        "1. A concise template name (3–6 words)\n"
        "2. A single-sentence description of what this AI assistant does\n\n"
        "Respond ONLY in this exact format — no preamble, no extra lines:\n"
        "NAME: <template name>\n"
        "DESCRIPTION: <one sentence>\n\n"
        f"Conversation:\n{transcript}"
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    name = "AI Assistant Template"
    description = "A custom AI assistant template."
    for line in response.content.strip().splitlines():
        if line.startswith("NAME:"):
            name = line[5:].strip()
        elif line.startswith("DESCRIPTION:"):
            description = line[12:].strip()

    return name, description


# ===========================================================================
# Public API
# ===========================================================================

@traceable(
    name="template_creation_chat",
    tags=["template-creation", "chat"],
    metadata={"pipeline": "TemplateCreation"},
)
def chat_template(
    template_id: str,
    user_id: str,
    user_prompt: str,
) -> str:
    """
    Handle one conversational turn for the template being built.

    Parameters
    ----------
    template_id : str
        Unique identifier for the template.  All messages for this
        session are keyed on this value in both Redis and PostgreSQL.
    user_id : str
        The user owning this template (stored with the final record).
    user_prompt : str
        The latest message from the user.

    Returns
    -------
    str
        The AI assistant's response for this turn, or an error message
        if the template has already been finalised.

    Side-effects
    ------------
    * Persists the incoming ``user_prompt`` to PostgreSQL + Redis before
      invoking the graph.
    * Persists every new AI / system message generated by the graph.
    * When ``state["satisfied"]`` becomes ``True``, automatically calls
      ``create_template()`` to run Phase 2 and store the final template.
    """
    # -------------------------------------------------------------------
    # 0. Guard: reject further turns once the template is finalised
    #
    #    As soon as create_template() completes, a row is upserted into
    #    the Templates table.  Any subsequent call to chat_template() for
    #    the same template_id hits this check and returns early — the
    #    graph is never invoked and no messages are persisted.
    # -------------------------------------------------------------------
    print(f"[TC:service] ► chat_template called | template_id={template_id!r} user_id={user_id!r}")
    print(f"[TC:service]   user_prompt preview: {user_prompt[:80]!r}{'...' if len(user_prompt) > 80 else ''}")

    if is_template_finalized(template_id):
        print(f"[TC:service] ✗ Template '{template_id}' is already finalised — rejecting turn.")
        return (
            "This template has already been created and is no longer "
            "available for editing."
        )

    # -------------------------------------------------------------------
    # 1. Load existing conversation history (Redis → Postgres fallback)
    # -------------------------------------------------------------------
    history = _load_history(template_id)
    print(f"[TC:service]   history loaded: {len(history)} message(s) for template '{template_id}'")

    # -------------------------------------------------------------------
    # 2. Determine next sequence number
    # -------------------------------------------------------------------
    next_seq = (history[-1]["sequence_number"] + 1) if history else 1
    print(f"[TC:service]   next sequence number: {next_seq}")

    # -------------------------------------------------------------------
    # 3. Persist and cache the incoming user message
    # -------------------------------------------------------------------
    print(f"[TC:service]   persisting user message (seq={next_seq}) …")
    user_record = _persist_message(template_id, "user", user_prompt, next_seq)
    append_to_conversation_cache(template_id, [user_record])
    print(f"[TC:service]   user message persisted (token_count={user_record.get('token_count')})")

    # Full history now includes the user message we just stored
    history = history + [user_record]

    # -------------------------------------------------------------------
    # 4. Reconstruct LangChain messages and build graph state
    # -------------------------------------------------------------------
    lc_messages = _to_langchain_messages(history)

    state: dict = {
        "messages":            lc_messages,
        "phase":               "gathering",
        "satisfied":           False,
        "requirements":        {},
        "tool_creation_prompt": "",
        "system_prompt":       "",
    }

    # -------------------------------------------------------------------
    # 5. Invoke the graph fresh — no LangGraph checkpointer is used
    #
    #    Graph routing recap:
    #      START → chatbot_node → route_after_chatbot
    #        "gather" (satisfied=False) → END
    #        "plan"   (satisfied=True)  → planner_node → END
    #
    #    result["messages"] = all input messages + new messages from
    #    this turn (LangGraph's add_messages reducer accumulates them).
    # -------------------------------------------------------------------
    print(f"[TC:service]   building graph and invoking (Phase 1 chatbot) …")
    graph = build_graph()
    result = graph.invoke(state)
    print(f"[TC:service]   graph invocation complete | satisfied={result.get('satisfied')} phase={result.get('phase')}")

    # -------------------------------------------------------------------
    # 6. Extract newly generated messages
    #    Everything after index len(lc_messages) is new this turn.
    # -------------------------------------------------------------------
    input_count = len(lc_messages)
    new_lc_messages = result["messages"][input_count:]
    print(f"[TC:service]   new messages generated this turn: {len(new_lc_messages)}")

    # -------------------------------------------------------------------
    # 7. Persist new AI / system messages to PostgreSQL + Redis
    # -------------------------------------------------------------------
    new_cache_records: List[Dict] = []
    ai_response = ""

    for i, msg in enumerate(new_lc_messages):
        role    = "assistant" if isinstance(msg, AIMessage) else "system"
        content = msg.content
        seq     = next_seq + 1 + i

        print(f"[TC:service]   persisting {role} message (seq={seq}, tokens≈{_count_tokens(content)}) …")
        record = _persist_message(template_id, role, content, seq)
        new_cache_records.append(record)

        if isinstance(msg, AIMessage):
            ai_response = content   # keep the last AI message as the return value

    if new_cache_records:
        append_to_conversation_cache(template_id, new_cache_records)
        print(f"[TC:service]   {len(new_cache_records)} new message(s) written to Redis cache")

    # -------------------------------------------------------------------
    # 8. Phase transition: if satisfied, trigger Phase 2 automatically and Generation of tool also starts
    # -------------------------------------------------------------------
    if result.get("satisfied", False):
        print(f"[TC:service] ✔ satisfied=True — Phase 1 complete. Triggering Phase 2 (create_template) …")
        full_history = _load_history(template_id)
        print(f"[TC:service]   full history for planner: {len(full_history)} message(s)")
        create_template(user_id, template_id, full_history)
        print(f"[TC:service] ✔ Phase 2 complete. Triggering ToolGeneration pipeline (generate_tool) …")
        generate_tool(template_id)
        print(f"[TC:service] ✔ generate_tool dispatched for template_id='{template_id}'")
    else:
        print(f"[TC:service]   satisfied=False — continuing Phase 1 (gathering requirements)")

    print(f"[TC:service] ◄ chat_template returning | response length={len(ai_response)} chars")
    return ai_response


@traceable(
    name="template_creation_plan",
    tags=["template-creation", "planner"],
    metadata={"pipeline": "TemplateCreation"},
)
def create_template(
    user_id: str,
    template_id: str,
    template_conv_history: List[Dict],
) -> None:
    """
    Run Phase 2: generate template artifacts and persist to Templates.

    This function is called automatically by ``chat_template()`` when
    ``state["satisfied"]`` becomes ``True``.  It can also be called
    independently if needed.

    Parameters
    ----------
    user_id : str
        Owner of the template (stored in ``Templates.created_by``).
    template_id : str
        The template being finalised.
    template_conv_history : list[dict]
        Complete conversation history from TEMP_MESSAGES.
        Each dict must have at least: ``role``, ``content``.

    Side-effects
    ------------
    * Calls ``planner_node`` directly with the reconstructed state.
    * Makes a Groq LLM call to derive ``name`` and ``description``.
    * Upserts the result into the ``Templates`` table.
    """
    # -------------------------------------------------------------------
    # 1. Reconstruct LangGraph state from the full conversation history
    # -------------------------------------------------------------------
    print(f"[TC:service] ► create_template called | template_id={template_id!r} user_id={user_id!r}")
    print(f"[TC:service]   conversation history: {len(template_conv_history)} message(s)")
    lc_messages = _to_langchain_messages(template_conv_history)
    print(f"[TC:service]   reconstructed {len(lc_messages)} LangChain message(s) for planner")

    state: dict = {
        "messages":            lc_messages,
        "phase":               "planning",
        "satisfied":           True,
        "requirements":        {},
        "tool_creation_prompt": "",
        "system_prompt":       "",
    }

    # -------------------------------------------------------------------
    # 2. Invoke planner_node directly — no full graph re-run needed
    # -------------------------------------------------------------------
    print(f"[TC:service]   invoking planner_node (Phase 2) …")
    planner_result = planner_node(state)
    print(f"[TC:service]   planner_node complete | phase={planner_result.get('phase')}")

    tool_generation_prompt = planner_result.get("tool_creation_prompt", "")
    behaviour_prompt       = planner_result.get("system_prompt",         "")
    print(f"[TC:service]   tool_creation_prompt length: {len(tool_generation_prompt)} chars")
    print(f"[TC:service]   system_prompt length:        {len(behaviour_prompt)} chars")

    # -------------------------------------------------------------------
    # 3. Generate name + description via Groq
    # -------------------------------------------------------------------
    print(f"[TC:service]   generating template name + description via Groq …")
    name, description = _generate_name_and_description(template_conv_history)
    print(f"[TC:service]   template name: {name!r}")
    print(f"[TC:service]   description:   {description!r}")

    # -------------------------------------------------------------------
    # 4. Persist to Templates table
    #    tool_information is intentionally omitted — it is populated
    #    by a separate downstream pipeline.
    # -------------------------------------------------------------------
    print(f"[TC:service]   inserting template record into Templates table …")
    insert_template(
        template_id=template_id,
        user_id=user_id,
        name=name,
        description=description,
        behaviour_prompt=behaviour_prompt,
        tool_generation_prompt=tool_generation_prompt,
    )
    print(f"[TC:service] ✔ create_template complete — template '{template_id}' persisted to DB")
