"""
MainAgent/nodes/persist_messages_node.py
-----------------------------------------
Graph node: persist_messages_node

Persists the current turn's user message and assistant response to
PostgreSQL and updates the running unsummarized token counter.

Operations (in order)
---------------------
1. Count tokens for both messages.
2. Get the next available sequence numbers.
3. Insert user message   (sequence N).
4. Insert assistant message (sequence N+1).
5. Stamp last_message_at on the Conversations row.
6. Increment unsummarized_token_count in conversation_memory.

Returns partial state update with:
  - ``new_user_seq``             : assigned sequence number for user message
  - ``new_assistant_seq``        : assigned sequence number for assistant message
  - ``unsummarized_token_count`` : updated running total
"""

from __future__ import annotations

import asyncio

from MainAgent.db.postgres_client import (
    count_tokens,
    get_next_sequence_number,
    insert_message,
    update_conversation_last_message_at,
    update_unsummarized_token_count,
)
from MainAgent.state import OrchestratorState


async def persist_messages_node(state: OrchestratorState) -> dict:
    """
    Persist user + assistant messages and update conversation metadata.

    Returns partial state update with new sequence numbers and updated
    unsummarized_token_count.
    """
    conv_id        = state["conv_id"]
    user_prompt    = state["user_prompt"]
    final_response = state["final_response"]

    # ── 1. Count tokens ───────────────────────────────────────────────────────
    user_tokens      = count_tokens(user_prompt)
    assistant_tokens = count_tokens(final_response)

    # ── 2. Reserve two consecutive sequence numbers ───────────────────────────
    next_seq      = await asyncio.to_thread(get_next_sequence_number, conv_id)
    user_seq      = next_seq
    assistant_seq = next_seq + 1

    # ── 3. Persist user message ───────────────────────────────────────────────
    await asyncio.to_thread(
        insert_message,
        conv_id, "user", user_prompt, user_tokens, user_seq,
    )

    # ── 4. Persist assistant message ──────────────────────────────────────────
    await asyncio.to_thread(
        insert_message,
        conv_id, "assistant", final_response, assistant_tokens, assistant_seq,
    )

    # ── 5. Stamp last_message_at on the Conversations table ───────────────────
    await asyncio.to_thread(update_conversation_last_message_at, conv_id)

    # ── 6. Update running unsummarized token count ────────────────────────────
    new_unsummarized = (
        state["unsummarized_token_count"] + user_tokens + assistant_tokens
    )
    await asyncio.to_thread(
        update_unsummarized_token_count,
        conv_id,
        new_unsummarized,
    )

    return {
        "new_user_seq":             user_seq,
        "new_assistant_seq":        assistant_seq,
        "unsummarized_token_count": new_unsummarized,
    }
