"""
nodes/chatbot_node.py
---------------------
Phase 1 — Requirements Gathering Chatbot Node

This LangGraph node powers the conversational requirements analyst.
It uses Groq + openai/gpt-oss-120b to conduct a deep discovery
conversation with the user, asking intelligent follow-up questions until
it has gathered all the information needed to design a custom AI assistant.

The node sets state["satisfied"] = True and state["phase"] = "planning"
when it detects the [REQUIREMENTS_COMPLETE] signal in its response.

Note: all terminal print / streaming output has been removed.
      This node is designed to be called from the service layer, not a CLI.
"""

import os
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from TemplateCreation.state import GraphState
from TemplateCreation.utils.extraction import check_satisfaction_signal, clean_chatbot_response


# ---------------------------------------------------------------------------
# Load system prompt from file
# ---------------------------------------------------------------------------
_PROMPTS_DIR   = Path(__file__).resolve().parent.parent / "prompts"
_SYSTEM_PROMPT = (_PROMPTS_DIR / "chatbot_system.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Groq LLM
# ---------------------------------------------------------------------------
def _build_llm() -> ChatGroq:
    """Instantiate the Groq LLM for Phase 1."""
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.7,
        max_tokens=1024,
        streaming=False,
        api_key=os.environ["GROQ_API_KEY"],
    )


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------
def chatbot_node(state: GraphState) -> dict:
    """
    LangGraph node for Phase 1 requirements gathering.

    Reads the current conversation history from state, calls the Groq LLM
    with the analyst system prompt, and updates the state with the new
    assistant message.

    If the response contains [REQUIREMENTS_COMPLETE], sets:
        - state["satisfied"] = True
        - state["phase"]     = "planning"

    Parameters
    ----------
    state : GraphState
        The current shared graph state.

    Returns
    -------
    dict
        Partial state update with new messages and updated phase / satisfied flags.
    """
    llm = _build_llm()

    # Build message list: system prompt + conversation history
    messages = [SystemMessage(content=_SYSTEM_PROMPT)] + list(state["messages"])
    print(f"[TC:chatbot] ► chatbot_node fired | total messages (incl. system): {len(messages)}")

    print(f"[TC:chatbot]   invoking Groq LLM (Phase 1 chatbot) …")
    response = llm.invoke(messages)
    full_response = response.content
    print(f"[TC:chatbot]   LLM responded | raw response length: {len(full_response)} chars")

    # -----------------------------------------------------------------------
    # Check for satisfaction signal and strip it from the visible response
    # -----------------------------------------------------------------------
    satisfied        = check_satisfaction_signal(full_response)
    visible_response = clean_chatbot_response(full_response)
    print(f"[TC:chatbot]   satisfaction signal detected: {satisfied}")
    if satisfied:
        print("[TC:chatbot]   [REQUIREMENTS_COMPLETE] found — phase will transition to 'planning'")
    print(f"[TC:chatbot]   visible response length (after cleaning): {len(visible_response)} chars")

    # -----------------------------------------------------------------------
    # Build state update
    # -----------------------------------------------------------------------
    new_message = AIMessage(content=visible_response)
    print(f"[TC:chatbot] ◄ chatbot_node returning | phase={'planning' if satisfied else 'gathering'}")

    return {
        "messages":  [new_message],
        "satisfied": satisfied,
        "phase":     "planning" if satisfied else "gathering",
    }
