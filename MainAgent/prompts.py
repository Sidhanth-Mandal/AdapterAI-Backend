"""
MainAgent/prompts.py
--------------------
All prompt templates and context-assembly helpers for the MainAgent pipeline.

Exports
-------
ORCHESTRATOR_SYSTEM_PROMPT   : Core orchestrator persona injected as the system message.
SUMMARIZATION_SYSTEM_PROMPT  : Instruction for the summarization LLM call.
build_full_context()         : Assembles the structured context block from state fields.
"""

from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------------------------
# Orchestrator system prompt
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are an intelligent AI assistant and orchestrator. Your responsibilities:

1. UNDERSTAND the user's intent from their message and conversation context.
2. DECIDE which tools (if any) are needed to fulfill the request.
3. EXECUTE tools iteratively — you may call multiple tools in sequence.
4. SYNTHESISE a final, complete, and well-formatted response.

TOOL USAGE RULES
----------------
- Use web search tools when the user needs current facts, news, or external knowledge.
- Use the retrieval tool ONLY when the user asks about uploaded files or attachments.
- Use the custom tool subagent when the user's request can be served by the
  custom tools built for this conversation's template.
- You may chain tool calls freely: e.g. search → retrieve → custom tool → final answer.
- When you have gathered sufficient information, respond directly and helpfully.
- Always respond in the same language the user used.
- Do NOT call tools unnecessarily — if you can answer from context, do so directly.
"""


# ---------------------------------------------------------------------------
# Summarization system prompt
# ---------------------------------------------------------------------------

SUMMARIZATION_SYSTEM_PROMPT = """\
You are a precise conversation summarizer. Your task is to produce an updated
running summary of a conversation.

You will be given:
  1. The existing summary (may be empty on first summarization).
  2. New, unsummarized messages to incorporate.

Produce an UPDATED summary that:
  - Integrates key information from the new messages.
  - Preserves important context from the existing summary.
  - Captures: topics discussed, decisions made, user goals, important facts.
  - Is concise but complete — avoid padding or repetition.

Respond with ONLY the updated summary text. No labels, no preamble.
"""


# ---------------------------------------------------------------------------
# Context assembler
# ---------------------------------------------------------------------------

def build_full_context(
    behavior_prompt: str,
    custom_tool_information: str,
    summary: str,
    recent_messages: List[Dict],
    if_attachment: bool,
    user_prompt: str,
) -> str:
    """
    Assemble the structured context block injected into the system message.

    The sections are included only when they contain meaningful content.
    Empty / whitespace-only values are silently omitted.

    Parameters
    ----------
    behavior_prompt : str
        Custom persona / behaviour instructions from the template.
    custom_tool_information : str
        Description of the custom tool(s) available for this conversation.
    summary : str
        Running compressed summary of conversation history.
    recent_messages : list[dict]
        Last N message dicts with keys ``role`` and ``content``.
    if_attachment : bool
        Whether the conversation has uploaded attachments.
    user_prompt : str
        The current user message (referenced in the attachment hint).

    Returns
    -------
    str
        A multi-section context block ready to append to the system prompt.
    """
    sections: List[str] = []

    # ── Behavior instructions ─────────────────────────────────────────────────
    if behavior_prompt.strip():
        sections.append(
            "## BEHAVIOR INSTRUCTIONS\n"
            + behavior_prompt.strip()
        )

    # ── Custom tool information ───────────────────────────────────────────────
    if custom_tool_information.strip():
        sections.append(
            "## CUSTOM TOOL INFORMATION\n"
            "The following custom functions are available for this conversation.\n"
            "IMPORTANT: Do NOT call these functions directly by name. They are NOT\n"
            "in your tool list. You MUST invoke them exclusively through the\n"
            "`call_custom_tool_subagent` tool by describing the task in natural language.\n\n"
            + custom_tool_information.strip()
        )

    # ── Conversation summary ──────────────────────────────────────────────────
    if summary.strip():
        sections.append(
            "## CONVERSATION SUMMARY\n"
            + summary.strip()
        )

    # ── Recent conversation history ───────────────────────────────────────────
    if recent_messages:
        turn_lines: List[str] = []
        for m in recent_messages:
            role = m.get("role", "user").lower()
            label = "User" if role in ("user", "human") else "Assistant"
            turn_lines.append(f"[{label}]: {m.get('content', '')}")
        sections.append(
            "## RECENT CONVERSATION\n"
            + "\n".join(turn_lines)
        )

    # ── Attachment status ─────────────────────────────────────────────────────
    if if_attachment:
        sections.append(
            "## ATTACHMENT STATUS\n"
            "The user has uploaded one or more files in this conversation. "
            "If the current query relates to document content (PDFs, spreadsheets, "
            "reports, etc.), use the retrieval tool to search the uploaded documents."
        )
    else:
        sections.append(
            "## ATTACHMENT STATUS\n"
            "No files or attachments have been uploaded in this conversation. "
            "Do NOT call the retrieval tool."
        )

    return "\n\n".join(sections)
