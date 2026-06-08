"""
MainAgent/nodes/summary_node.py
---------------------------------
Graph node: summary_node

Conditionally generates an updated conversation summary.

Trigger condition
-----------------
  state["unsummarized_token_count"] >= 4000

When triggered
--------------
1. Fetch all messages with sequence_number > last_summarized_message_seq.
2. Send (existing summary + new messages) to the LLM.
3. Write the updated summary, last_summarized_message_id,
   last_summarized_message_seq, and reset unsummarized_token_count = 0
   back to PostgreSQL.
4. Return updated summary fields in state.

When NOT triggered
------------------
Returns an empty dict — no state changes, no DB writes.
"""

from __future__ import annotations

import asyncio
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from MainAgent.db.postgres_client import (
    fetch_unsummarized_messages,
    update_conversation_memory,
)
from MainAgent.prompts import SUMMARIZATION_SYSTEM_PROMPT
from MainAgent.state import OrchestratorState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUMMARIZATION_THRESHOLD = 4000
_MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

async def summary_node(state: OrchestratorState) -> dict:
    """
    Generate an updated running summary if the unsummarized token count
    has crossed the 4 000-token threshold.

    Returns partial state update (empty dict if no summarization ran):
      - ``summary``                    : updated summary text
      - ``last_summarized_message_seq``: new bookmark sequence number
      - ``unsummarized_token_count``   : reset to 0
    """
    # ── Guard: only run when threshold is exceeded ────────────────────────────
    if state["unsummarized_token_count"] < _SUMMARIZATION_THRESHOLD:
        return {}

    conv_id  = state["conv_id"]
    last_seq = state["last_summarized_message_seq"]

    # ── 1. Fetch unsummarized messages from DB ────────────────────────────────
    msgs = await asyncio.to_thread(fetch_unsummarized_messages, conv_id, last_seq)

    if not msgs:
        return {}

    # ── 2. Build the summarization prompt ─────────────────────────────────────
    turn_lines = []
    for m in msgs:
        role_label = "User" if m["role"] in ("user", "human") else "Assistant"
        turn_lines.append(f"[{role_label}]: {m['content']}")

    user_content = (
        f"EXISTING SUMMARY:\n"
        f"{state['summary'] or '(no summary yet)'}\n\n"
        f"NEW MESSAGES TO INCORPORATE:\n"
        + "\n".join(turn_lines)
    )

    # ── 3. Call LLM ───────────────────────────────────────────────────────────
    llm = ChatGroq(
        model=_MODEL,
        temperature=0.0,
        max_tokens=1024,
        api_key=os.environ["GROQ_API_KEY"],
    )

    response = await llm.ainvoke([
        SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])

    new_summary = response.content.strip()
    last_msg    = msgs[-1]

    # ── 4. Persist updated summary to PostgreSQL ──────────────────────────────
    await asyncio.to_thread(
        update_conversation_memory,
        conv_id,
        new_summary,
        last_msg["message_id"],
        last_msg["sequence_number"],
        0,  # reset unsummarized_token_count
    )

    return {
        "summary":                    new_summary,
        "last_summarized_message_seq": last_msg["sequence_number"],
        "unsummarized_token_count":   0,
    }
