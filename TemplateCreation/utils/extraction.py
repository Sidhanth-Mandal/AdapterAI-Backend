"""
utils/extraction.py
-------------------
Helpers for parsing and extracting structured data from the planner node's
raw LLM response.
"""

import re
from typing import Tuple


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
