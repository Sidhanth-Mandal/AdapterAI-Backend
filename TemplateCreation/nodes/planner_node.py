"""
nodes/planner_node.py
---------------------
Phase 2 — Tool Creation Prompt & System Prompt Generator Node

After the requirements chatbot (Phase 1) signals satisfaction, this node
takes over. It reads the full conversation history and uses a dedicated
Groq call to produce two structured outputs:

  1. Tool Creation Prompt  — detailed implementation spec for a tool-building AI
  2. System Prompt         — full behavioural specification for the final assistant

Both outputs are extracted from the LLM response using delimiter parsing
and stored back in state for the service layer to persist.

Note: all terminal print output has been removed.
      This node is designed to be called from the service layer, not a CLI.
"""

import os
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from TemplateCreation.state import GraphState
from utils.extraction import extract_planner_outputs


# ---------------------------------------------------------------------------
# Load planner system prompt from file
# ---------------------------------------------------------------------------
_PROMPTS_DIR   = Path(__file__).resolve().parent.parent / "prompts"
_SYSTEM_PROMPT = (_PROMPTS_DIR / "planner_system.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Groq LLM — higher context, non-streaming (structured output)
# ---------------------------------------------------------------------------
def _build_llm() -> ChatGroq:
    """Instantiate the Groq LLM for Phase 2 planning."""
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.4,
        max_tokens=4096,
        streaming=False,
        api_key=os.environ["GROQ_API_KEY"],
    )


# ---------------------------------------------------------------------------
# Build the planner user message from conversation history
# ---------------------------------------------------------------------------
def _build_planner_message(state: GraphState) -> str:
    """
    Construct the planner's user-turn message by formatting the full
    conversation history into a readable transcript.

    Parameters
    ----------
    state : GraphState
        Current graph state containing the full message history.

    Returns
    -------
    str
        A formatted transcript string that the planner LLM will process.
    """
    lines = [
        "Below is the complete requirements gathering conversation transcript.",
        "Read it carefully and generate the two outputs as instructed.",
        "",
        "=" * 60,
        "CONVERSATION TRANSCRIPT",
        "=" * 60,
        "",
    ]

    for msg in state["messages"]:
        role = getattr(msg, "type", "unknown").upper()
        # Normalize role labels for readability
        if role == "HUMAN":
            role = "USER"
        elif role == "AI":
            role = "ASSISTANT"
        lines.append(f"[{role}]")
        lines.append(msg.content)
        lines.append("")

    lines += [
        "=" * 60,
        "",
        "Now generate the TOOL CREATION PROMPT and SYSTEM PROMPT using the exact format specified.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------
def planner_node(state: GraphState) -> dict:
    """
    LangGraph node for Phase 2 — planning and output generation.

    Reads the full conversation transcript from state, calls the Groq
    planner LLM, and extracts the two structured outputs:
    - tool_creation_prompt
    - system_prompt

    Parameters
    ----------
    state : GraphState
        The current shared graph state.

    Returns
    -------
    dict
        Partial state update containing tool_creation_prompt, system_prompt,
        and phase set to "done".
    """
    llm = _build_llm()

    planner_message = _build_planner_message(state)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=planner_message),
    ]

    response    = llm.invoke(messages)
    raw_output  = response.content

    # -----------------------------------------------------------------------
    # Extract the two delimited sections
    # -----------------------------------------------------------------------
    tool_creation_prompt, system_prompt = extract_planner_outputs(raw_output)

    # -----------------------------------------------------------------------
    # Fallback: if parsing fails, store the raw output for inspection
    # -----------------------------------------------------------------------
    if not tool_creation_prompt:
        tool_creation_prompt = (
            "[PARSE ERROR] Could not extract Tool Creation Prompt.\n\n"
            "Raw planner output:\n" + raw_output
        )
    if not system_prompt:
        system_prompt = (
            "[PARSE ERROR] Could not extract System Prompt.\n\n"
            "Raw planner output:\n" + raw_output
        )

    return {
        "tool_creation_prompt": tool_creation_prompt,
        "system_prompt":        system_prompt,
        "phase":                "done",
    }
