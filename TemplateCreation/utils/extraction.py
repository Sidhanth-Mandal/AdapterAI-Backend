"""
utils/extraction.py
-------------------
Helpers for parsing and extracting structured data from the planner node's
raw LLM response.
"""

import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Delimiter constants — must match planner_system.txt exactly
# ---------------------------------------------------------------------------
_TCP_START  = "--- TOOL CREATION PROMPT ---"
_TCP_END    = "--- END TOOL CREATION PROMPT ---"
_SP_START   = "--- SYSTEM PROMPT ---"
_SP_END     = "--- END SYSTEM PROMPT ---"


def extract_planner_outputs(raw_response: str) -> Tuple[str, str]:
    """
    Parse the planner node's raw LLM output and return the two sections.

    Parameters
    ----------
    raw_response : str
        The complete text response from the planner LLM call.

    Returns
    -------
    Tuple[str, str]
        (tool_creation_prompt, system_prompt)
        Each is a stripped string. Returns ("", "") if parsing fails.
    """
    tool_creation_prompt = _extract_section(raw_response, _TCP_START, _TCP_END)
    system_prompt        = _extract_section(raw_response, _SP_START,  _SP_END)
    return tool_creation_prompt, system_prompt


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    """
    Extract the text between two delimiter markers.

    Parameters
    ----------wdd
    text : str
        Full raw string to search within.
    start_marker : str
        The opening delimiter.
    end_marker : str
        The closing delimiter.

    Returns
    -------
    str
        Content between the markers, stripped of leading/trailing whitespace.
        Returns empty string if markers are not found.
    """
    # Use regex with DOTALL so '.' matches newlines
    pattern = re.compile(
        re.escape(start_marker) + r"(.*?)" + re.escape(end_marker),
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return ""


def check_satisfaction_signal(content: str) -> bool:
    """
    Check whether the chatbot's response contains the satisfaction signal.

    The chatbot system prompt instructs the model to include the exact
    token `[REQUIREMENTS_COMPLETE]` on its own line when it is satisfied
    with the gathered requirements.

    Parameters
    ----------
    content : str
        The latest assistant message content.

    Returns
    -------
    bool
        True if the satisfaction signal is present, False otherwise.
    """
    return "[REQUIREMENTS_COMPLETE]" in content


def clean_chatbot_response(content: str) -> str:
    """
    Remove the internal satisfaction signal token from the chatbot's
    visible response so users never see the raw marker.

    Parameters
    ----------
    content : str
        Raw assistant message content.

    Returns
    -------
    str
        Cleaned content with the signal token removed and whitespace normalized.
    """
    cleaned = content.replace("[REQUIREMENTS_COMPLETE]", "").strip()
    # Collapse any triple+ blank lines that result from the removal
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# MCQ question parser
# ---------------------------------------------------------------------------

def parse_mcq_questions(response_text: str) -> List[Dict]:
    """
    Parse the chatbot's response and extract structured MCQ question blocks.

    The chatbot is instructed to output questions in this exact format::

        **<Question text>**
        a) <option A>
        b) <option B>
        c) <option C>
        d) <option D>
        __________________ (or tell us in your own words)

    Parameters
    ----------
    response_text : str
        The visible (already-cleaned) assistant response.

    Returns
    -------
    list[dict]
        Each dict has the shape::

            {
                "question": "<question text>",
                "options": [
                    {"label": "a", "text": "<option A>"},
                    {"label": "b", "text": "<option B>"},
                    {"label": "c", "text": "<option C>"},
                    {"label": "d", "text": "<option D>"},
                    {"label": "custom", "text": ""}
                ]
            }

        Returns an empty list if no properly-formatted question blocks are found.
    """
    questions: List[Dict] = []

    # Match a question block: bold header + a)/b)/c)/d) lines + __ line
    # The bold header pattern: **...**  (possibly with leading whitespace)
    block_pattern = re.compile(
        r"\*\*(.+?)\*\*"          # **Question text**
        r"\s*\n"
        r"a\)\s*(.+?)\n"          # a) ...
        r"b\)\s*(.+?)\n"          # b) ...
        r"c\)\s*(.+?)\n"          # c) ...
        r"d\)\s*(.+?)\n"          # d) ...
        r"_{3,}.*",               # __________________ line
        re.DOTALL,
    )

    for m in block_pattern.finditer(response_text):
        question_text = m.group(1).strip()
        options = [
            {"label": "a", "text": m.group(2).strip()},
            {"label": "b", "text": m.group(3).strip()},
            {"label": "c", "text": m.group(4).strip()},
            {"label": "d", "text": m.group(5).strip()},
            {"label": "custom", "text": ""},
        ]
        questions.append({"question": question_text, "options": options})

    return questions


def extract_preamble(response_text: str) -> str:
    """
    Return only the non-question preamble text from a chatbot response.

    Everything before the first ``**...**`` question block is considered
    preamble (acknowledgement / summary text that the frontend can render
    as plain prose above the MCQ cards).

    Parameters
    ----------
    response_text : str
        The full visible assistant response.

    Returns
    -------
    str
        Preamble text stripped of trailing whitespace.  May be an empty
        string if the response starts immediately with a question block.
    """
    first_question = re.search(r"\*\*", response_text)
    if first_question:
        return response_text[: first_question.start()].strip()
    return response_text.strip()
