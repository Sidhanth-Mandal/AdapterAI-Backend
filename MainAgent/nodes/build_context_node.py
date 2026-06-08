"""
MainAgent/nodes/build_context_node.py
---------------------------------------
Graph node: build_context_node

Assembles the LangChain message list that will be passed to the
orchestrator LLM.  This is the only node that writes to the
``messages`` field in state.

Message layout
--------------
  SystemMessage  : orchestrator persona + full context block
                   (behaviour prompt, custom tool info, summary,
                    recent turns, attachment status)
  HumanMessage   : the current user prompt

The context block is built by prompts.build_full_context(), which
omits any section that has no meaningful content.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from MainAgent.prompts import ORCHESTRATOR_SYSTEM_PROMPT, build_full_context
from MainAgent.state import OrchestratorState


async def build_context_node(state: OrchestratorState) -> dict:
    """
    Build the initial message list for the orchestrator agent.

    Returns a partial state update containing one entry for ``messages``.
    The add_messages reducer will append these to the (currently empty)
    messages list in state.
    """
    # ── Assemble structured context block ─────────────────────────────────────
    context_block = build_full_context(
        behavior_prompt=state["behavior_prompt"],
        custom_tool_information=state["custom_tool_information"],
        summary=state["summary"],
        recent_messages=state["recent_messages"],
        if_attachment=state["if_attachment"],
        user_prompt=state["user_prompt"],
    )

    # ── Combine persona + context into the system message ─────────────────────
    system_content = ORCHESTRATOR_SYSTEM_PROMPT.rstrip() + "\n\n" + context_block

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=state["user_prompt"]),
    ]

    return {"messages": messages}
