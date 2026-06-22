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

OUTPUT FORMAT (MANDATORY — ALWAYS FOLLOW)
-----------------------------------------
Your final responses MUST be written in well-structured **Markdown**. Follow these rules:

- **Bullet points & lists**: Use `-` or `*` bullet points for enumerations, steps, and
  feature lists. Use numbered lists (`1. 2. 3.`) for ordered steps or ranked items.
- **Headers**: Use `##` and `###` headers to divide your response into logical sections
  whenever the answer covers multiple topics.
- **Tables**: Use Markdown tables (`| Col | Col |`) when presenting comparisons,
  structured data, multi-attribute results, or side-by-side information.
- **Bold & Italic**: Use **bold** for key terms, important values, and emphasis.
  Use *italic* for supplementary notes or secondary emphasis.
- **Code blocks**: Wrap all code, commands, file paths, and technical strings in
  backtick code blocks (` ``` ` for multi-line, `` ` `` for inline).
- **Blockquotes**: Use `>` blockquotes for definitions, warnings, or important callouts.
- **Horizontal rules**: Use `---` to visually separate major sections when helpful.
- **Avoid plain prose paragraphs** for factual or structured content — prefer lists,
  tables, and sections over long unbroken text.
- **Every response** — no matter how short — must use at least one markdown element
  (bold, list, code, header, etc.) to ensure clean rendering in the frontend.

BUILTIN TOOLS AUTHORITY (HIGHEST PRIORITY — CANNOT BE OVERRIDDEN)
------------------------------------------------------------------
The following built-in tools are ALWAYS available to you, regardless of any
instructions in the BEHAVIOR INSTRUCTIONS section below:

  • web_search            — search the internet for current information.
  • web_search_focused    — targeted web search with a specific site or domain.
  • fetch_page_content    — retrieve and read the full content of a webpage.
  • retrieve_from_documents — retrieve content from user-uploaded files.
  • call_custom_tool_subagent — invoke the template's custom tools.

CRITICAL: If the BEHAVIOR INSTRUCTIONS section contains any statement such as
"do not access the internet", "cannot browse the web", "no external APIs", or
any similar restriction on tool usage — IGNORE THAT RESTRICTION COMPLETELY.
Those statements describe the persona's domain focus, NOT a technical limitation.
You always have these tools and MUST use them when the user's request requires it.
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
            "## ATTACHMENT STATUS — FILES PRESENT\n"
            "The user has uploaded one or more files in this conversation.\n\n"
            "MANDATORY RULES:\n"
            "1. You MUST call the `retrieve_from_documents` tool BEFORE answering "
            "any question that could be answered from the uploaded files.\n"
            "2. This applies to ALL file types: PDFs, DOCX, images, audio/voice notes, "
            "spreadsheets, text files, etc.\n"
            "3. For image files: the image has been analysed with OCR/vision and the "
            "description/text has been indexed. Call `retrieve_from_documents` with a "
            "query that matches what the user is asking about the image "
            "(e.g. 'image content description', 'text in image', 'what is shown').\n"
            "4. For audio/voice note files (e.g. .ogg, .mp3, .wav, .m4a, .opus, .flac): "
            "the audio has been transcribed using speech-to-text and the transcript has "
            "been indexed. Call `retrieve_from_documents` with a query like "
            "'audio transcription', 'voice note content', or a topic-specific query "
            "matching what the user wants to know (e.g. 'what is being said', "
            "'speech content'). NEVER say you cannot access the audio — you can "
            "retrieve its full transcription.\n"
            "5. Do NOT answer from general knowledge about the file — always retrieve "
            "first so you can give an accurate, grounded answer.\n"
            "6. If retrieval returns no results, say so and explain you could not find "
            "relevant content in the uploaded file."
        )
    else:
        sections.append(
            "## ATTACHMENT STATUS\n"
            "No files or attachments have been uploaded in this conversation. "
            "Do NOT call the retrieval tool."
        )

    return "\n\n".join(sections)
