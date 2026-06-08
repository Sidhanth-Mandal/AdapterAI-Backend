"""
MainAgent/state.py
------------------
Defines OrchestratorState — the TypedDict that flows through every node
in the MainAgent LangGraph pipeline.

Field groups
------------
Request context      : template_id, user_id, conv_id, user_prompt, if_attachment
Template config      : behavior_prompt, custom_tool_information
Conversation memory  : summary, recent_messages, unsummarized_token_count,
                       last_summarized_message_seq
Agent messages       : messages  (LangChain message objects, accumulated by add_messages)
Output               : final_response, tools_called
Persistence tracking : new_user_seq, new_assistant_seq
"""

from __future__ import annotations

from typing import Annotated, Dict, List, TypedDict

from langgraph.graph.message import add_messages


class OrchestratorState(TypedDict):
    # ── Request context ───────────────────────────────────────────────────────
    template_id: str
    """Template identifier — used to load behavior prompt and tool info."""

    user_id: str
    """Caller identity — passed through to tools and persistence."""

    conv_id: str
    """Conversation identifier — keys all memory and message storage."""

    user_prompt: str
    """The raw message sent by the user this turn."""

    if_attachment: bool
    """True when the conversation contains uploaded files/attachments."""

    # ── Template configuration (populated by load_template_node) ──────────────
    behavior_prompt: str
    """Custom persona / behaviour instructions from the template."""

    custom_tool_information: str
    """Natural-language description of the custom tool(s) bound to this template."""

    # ── Conversation memory (populated by load_memory_node) ───────────────────
    summary: str
    """Running compressed summary of the conversation history."""

    recent_messages: List[Dict]
    """Last 20 raw message dicts (role + content + metadata) from Redis/DB."""

    unsummarized_token_count: int
    """Cumulative token count of messages not yet included in summary."""

    last_summarized_message_seq: int
    """sequence_number of the last message captured in the current summary."""

    # ── LangChain message list (built by build_context_node, extended by agent) ─
    messages: Annotated[list, add_messages]
    """
    LangChain message objects passed to / returned from the LLM.
    The add_messages reducer appends new messages and deduplicates by ID.
    """

    # ── Orchestrator output ───────────────────────────────────────────────────
    final_response: str
    """The assistant's final text response for this turn."""

    tools_called: List[str]
    """Ordered list of tool names actually invoked during this turn.
    Populated by orchestrator_agent_node from ToolMessage objects.
    Useful for testing and observability.
    """

    # ── Persistence tracking (set by persist_messages_node) ───────────────────
    new_user_seq: int
    """sequence_number assigned to the persisted user message."""

    new_assistant_seq: int
    """sequence_number assigned to the persisted assistant message."""
