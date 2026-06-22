"""
nodes/planner_node.py
---------------------
Phase 2 — Tool Creation Prompt & System Prompt Generator Node

After the requirements chatbot (Phase 1) signals satisfaction, this node
takes over. It reads the full conversation history and uses a dedicated
Anthropic call to produce two structured outputs via `with_structured_output`:

  1. tool_creation_prompt  — detailed implementation spec for a tool-building AI
  2. behavior_prompt       — full behavioural specification for the final assistant

The LLM is bound to the `PlannerOutput` Pydantic schema, so the response is
a validated dict — no delimiter parsing or regex extraction needed.

Note: all terminal print output has been removed.
      This node is designed to be called from the service layer, not a CLI.
"""

import os
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from TemplateCreation.state import GraphState
from TemplateCreation.utils.extraction import PlannerOutput
from utils.tracing import traceable  # noqa: E402


# ---------------------------------------------------------------------------
# Load planner system prompt from file
# ---------------------------------------------------------------------------
_PROMPTS_DIR   = Path(__file__).resolve().parent.parent / "prompts"
_SYSTEM_PROMPT = (_PROMPTS_DIR / "planner_system.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Anthropic LLM bound to structured output schema
# ---------------------------------------------------------------------------
def _build_llm() -> ChatAnthropic:
    """Instantiate the Anthropic LLM for Phase 2 planning with structured output."""
    llm = ChatAnthropic(
        model="claude-haiku-4-5",
        temperature=0.4,
        max_tokens=8096,
        streaming=False,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    # Bind the Pydantic schema so the model always returns a validated
    # {tool_creation_prompt, behavior_prompt} dict — no parsing required.
    return llm.with_structured_output(PlannerOutput)


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
        "Now generate the TOOL CREATION PROMPT and BEHAVIOR PROMPT based on the conversation above.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------
@traceable(
    name="tc_planner_node",
    tags=["template-creation", "node", "planner"],
    metadata={"pipeline": "TemplateCreation"},
)
def planner_node(state: GraphState) -> dict:
    """
    LangGraph node for Phase 2 — planning and output generation.

    Reads the full conversation transcript from state, calls the Anthropic
    planner LLM with structured output, and unpacks the two validated fields:
    - tool_creation_prompt
    - behavior_prompt

    The model is bound to the `PlannerOutput` Pydantic schema via
    `with_structured_output`, so the result is always a well-typed dict
    regardless of how the model formats its internal text.

    Parameters
    ----------
    state : GraphState
        The current shared graph state.

    Returns
    -------
    dict
        Partial state update containing tool_creation_prompt, behavior_prompt,
        and phase set to \"done\".
    """
    llm = _build_llm()

    planner_message = _build_planner_message(state)
    print(f"[TC:planner] ► planner_node fired | conversation messages: {len(state['messages'])}")
    print(f"[TC:planner]   planner input transcript length: {len(planner_message)} chars")

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=planner_message),
    ]

    print("[TC:planner]   invoking Anthropic LLM (Phase 2 planner — structured output) …")
    # `llm` is already bound to PlannerOutput via with_structured_output.
    # The result is a PlannerOutput Pydantic object (not a raw AIMessage).
    result: PlannerOutput = llm.invoke(messages)

    # Unpack the validated structured fields — no regex or fallback needed.
    tool_creation_prompt: str = result.tool_creation_prompt
    behavior_prompt: str      = result.behavior_prompt

    print(
        f"[TC:planner]   tool_creation_prompt: {len(tool_creation_prompt)} chars "
        f"{'(OK)' if tool_creation_prompt else '(EMPTY)'}"
    )
    print(
        f"[TC:planner]   behavior_prompt:      {len(behavior_prompt)} chars "
        f"{'(OK)' if behavior_prompt else '(EMPTY)'}"
    )
    print("[TC:planner] ◄ planner_node returning | phase='done'")

    return {
        "tool_creation_prompt": tool_creation_prompt,
        "behavior_prompt":      behavior_prompt,
        "phase":                "done",
    }
