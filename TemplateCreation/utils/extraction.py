"""
utils/extraction.py
-------------------
Helpers for parsing and extracting structured data from the planner node's
raw LLM response.
"""

import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic schema for structured LLM output — replaces delimiter parsing
# ---------------------------------------------------------------------------

class PlannerOutput(BaseModel):
    """
    Structured output schema for the planner LLM call.

    The LLM is instructed to populate this schema directly, so no regex
    parsing of delimiters is required.  Each field maps to a downstream
    consumer:

    - ``tool_creation_prompt`` → passed to the ToolGeneration pipeline
    - ``behavior_prompt``      → stored as the assistant's behavioral spec
    """

    tool_creation_prompt: str = Field(
        description=(
            "A full production-grade Tool Creation Prompt: a detailed "
            "specification document (NOT Python code) for a downstream AI "
            "that will write the Python tool functions for this assistant. "
            "Must include context, per-tool specs, architecture requirements, "
            "recommended libraries, and quality standards."
        )
    )
    behavior_prompt: str = Field(
        description=(
            "The complete behavioral System Prompt for the final AI assistant. "
            "Written as a direct instruction document (starting with 'You are…'). "
            "Must cover identity, personality, communication style, domain "
            "specialisation, tool-usage philosophy, clarification behaviour, "
            "and safety rules."
        )
    )


# ---------------------------------------------------------------------------
# Delimiter constants — must match planner_system.txt exactly
# ---------------------------------------------------------------------------
_TCP_START  = "--- TOOL CREATION PROMPT ---"
_TCP_END    = "--- END TOOL CREATION PROMPT ---"
_SP_START   = "--- SYSTEM PROMPT ---"
_SP_END     = "--- END SYSTEM PROMPT ---"


def extract_planner_outputs(raw_response: str) -> Tuple[str, str]:
    """
    .. deprecated::
        This function relied on delimiter markers in the raw LLM text and was
        fragile.  The planner node now uses ``with_structured_output`` so the
        model returns a validated :class:`PlannerOutput` dict directly.
        This function is retained only for backward-compatibility with tests.

    Parameters
    ----------
    raw_response : str
        The complete text response from the planner LLM call.

    Returns
    -------
    Tuple[str, str]
        (tool_creation_prompt, behavior_prompt)
        Each is a stripped string. Returns ("", "") if parsing fails.
    """
    tool_creation_prompt = _extract_section(raw_response, _TCP_START, _TCP_END)
    behavior_prompt      = _extract_section(raw_response, _SP_START,  _SP_END)
    return tool_creation_prompt, behavior_prompt


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
        r"_{3,}[^\n]*",           # __________________ line
        # NOTE: re.DOTALL is intentionally omitted so that '.' does NOT
        # match newlines.  Each option group must stay on a single line;
        # using DOTALL caused (.+?) to span lines and merge all question
        # blocks into one match, so only the last question was captured.
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


#testing
reponse_text = '''{'template_id': 'Testing', 'preamble': "Welcome! I'm excited to help you design your personalized AI assistant. Let's get started with a few quick questions to understand exactly what you need.", 'questions': [{'question': 'What is the main purpose of your assistant? What problem should it solve for you?', 'options': [{'label': 'a', 'text': 'Help me with research and information gathering'}, {'label': 'b', 'text': 'Assist with writing, editing, or content creation'}, {'label': 'c', 'text': 'Manage tasks, scheduling, or productivity'}, {'label': 'd', 'text': 'Answer questions in a specific domain (e.g., finance, health, law, tech)'}, {'label': 'custom', 'text': ''}]}], 'response': "Welcome! I'm excited to help you design your personalized AI assistant. Let's get started with a few quick questions to understand exactly what you need.\n\n**What is the main purpose of your assistant? What problem should it solve for you?**\na) Help me with research and information gathering\nb) Assist with writing, editing, or content creation\nc) Manage tasks, scheduling, or productivity\nd) Answer questions in a specific domain (e.g., finance, health, law, tech)\n__________________ (or tell us in your own words)\n\n**What subject area or domain should your assistant specialize in?**\na) General-purpose (no specific domain)\nb) Business, finance, or marketing\nc) Science, technology, or engineering\nd) Health, wellness, or lifestyle\n__________________ (or tell us in your own words)\n\n**What kind of information or data should your assistant have access to?**\na) Only my own documents or files I provide\nb) Real-time web search and current news/data\nc) Specific databases, APIs, or platforms (e.g., stock data, weather, social media)\nd) A combination of the above\n__________________ (or tell us in your own words)\n\n**How deep should your assistant's expertise go?**\na) Beginner-friendly — simple explanations, no jargon\nb) Intermediate — assumes some background knowledge\nc) Expert-level — technical, detailed, research-grade responses\nd) Adaptive — adjust based on how I phrase my questions\n__________________ (or tell us in your own words)"}'''

print(parse_mcq_questions(response_text=reponse_text))