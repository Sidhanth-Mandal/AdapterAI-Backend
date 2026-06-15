"""
generator.py — LLM generation and repair layer for the ToolGeneration pipeline.

Two public functions
--------------------
generate_tool_json(user_prompt)
    Call Groq with the existing system_prompt + ToolSchema JSON schema.
    Return a validated tool dict.

repair_tool_json(tool_json, error_history, latest_error, error_type)
    Build a structured repair prompt, call Groq, extract the corrected
    tool dict plus cause / fix bullet points.
    Returns (corrected_dict, cause_list, fix_list).
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic


from ToolGeneration.schemas import ToolSchema
from ToolGeneration.SystemPrompt import system_prompt

load_dotenv()

# ---------------------------------------------------------------------------
# Shared Groq client
# ---------------------------------------------------------------------------

_MODEL_NAME = "claude-sonnet-4-6"
_model = ChatAnthropic(model=_MODEL_NAME)

_SCHEMA_STR: str = json.dumps(ToolSchema.model_json_schema(), indent=2)

_JSON_INSTRUCTION = f"""
IMPORTANT: Respond with ONLY a valid JSON object that matches this exact schema.
No markdown, no explanation, no code fences.

Schema:
{_SCHEMA_STR}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_fences(raw: str) -> str:
    """Remove markdown code fences if the model wraps its output in them."""
    raw = raw.strip()
    # Handle ```json ... ``` or ``` ... ``` anywhere in the response
    if "```" in raw:
        parts = raw.split("```")
        # parts: [before, content, after] for a single fence pair
        if len(parts) >= 3:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            raw = inner.strip()
    return raw


def _sanitise_control_chars(raw: str) -> str:
    """
    Escape bare ASCII control characters (0x00-0x1F, except valid JSON
    whitespace \t, \n, \r) that appear inside JSON string values.

    The LLM sometimes returns literal newlines / tabs / carriage returns
    embedded inside code-string values rather than the escaped forms
    (\\n, \\t) required by the JSON spec — causing json.loads to raise
    "Invalid control character".

    Strategy: walk the raw string character-by-character, tracking whether
    we are inside a JSON string (handling escaped quotes).  Replace any
    bare control character found inside a string with its \\uXXXX form.
    """
    result: list[str] = []
    in_string = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if in_string:
            if ch == "\\":          # escape sequence — keep as-is
                result.append(ch)
                i += 1
                if i < len(raw):    # keep the escaped character too
                    result.append(raw[i])
                    i += 1
                continue
            elif ch == '"':         # closing quote
                in_string = False
                result.append(ch)
            elif ord(ch) < 0x20:   # bare control character inside a string
                result.append(f"\\u{ord(ch):04x}")
            else:
                result.append(ch)
        else:
            if ch == '"':           # opening quote
                in_string = True
                result.append(ch)
            else:
                result.append(ch)
        i += 1
    return "".join(result)


def _parse_json(raw: str) -> dict:
    """Try json_repair first, fall back to stdlib json.loads with sanitisation."""
    raw = _strip_fences(raw)
    try:
        import json_repair  # type: ignore
        result = json_repair.loads(raw)
        # json_repair.loads can return a string if it cannot parse at all
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    # First attempt: direct parse (fast path)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Second attempt: sanitise bare control characters then retry
    sanitised = _sanitise_control_chars(raw)
    return json.loads(sanitised)


def _summarise_history(error_history: list[dict]) -> str:
    """
    Render error history into a compact multi-line text block suitable
    for injection into a repair prompt.

    The first entry (type='prompt') is intentionally skipped — the model
    already sees the original prompt in the system context.
    """
    lines: list[str] = []
    for i, entry in enumerate(error_history):
        etype = entry.get("type", "unknown")

        if etype == "prompt":
            continue  # already in context

        if etype == "validation_error":
            lines.append(f"[Attempt {i}] Validation error — stage: {entry.get('stage', '?')}")
            for e in entry.get("errors", []):
                lines.append(f"  • {e}")

        elif etype == "execution_error":
            lines.append(f"[Attempt {i}] Execution error:")
            lines.append(f"  {entry.get('error', '?')}")

        elif etype == "repair":
            lines.append(f"[Attempt {i}] Repair applied:")
            for c in entry.get("cause", []):
                lines.append(f"  cause • {c}")
            for f_ in entry.get("fix", []):
                lines.append(f"  fix   • {f_}")

    return "\n".join(lines) if lines else "(no prior errors)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_tool_json(user_prompt: str) -> dict:
    """
    Call Groq with the standard system prompt and return a validated tool dict.

    Parameters
    ----------
    user_prompt : str
        The tool_generation_prompt fetched from the Templates table.

    Returns
    -------
    dict
        Validated tool dict that conforms to ToolSchema.

    Raises
    ------
    ValueError
        If the model response cannot be parsed or validated against ToolSchema.
    """
    response = _model.invoke([
        SystemMessage(content=system_prompt + _JSON_INSTRUCTION),
        HumanMessage(content=user_prompt),
    ])

    with open("response.txt" , "w" , encoding= 'utf-8') as file:
        file.write('response :\n\n' + f'{response.content}' + '\n\n parsed_json\n\n' + f"{_parse_json(response.content)}")

    # print(SystemMessage(content=system_prompt + _JSON_INSTRUCTION))
    # print(HumanMessage(content=user_prompt))
    # with open("example.txt", "w", encoding='utf-8') as file:
    #     file.write(system_prompt + _JSON_INSTRUCTION + user_prompt )
    data = _parse_json(response.content)
    tool = ToolSchema.model_validate(data)
    return tool.model_dump()


def repair_tool_json(
    tool_json: dict,
    error_history: list[dict],
    latest_error: str,
    error_type: str = "validation_error",
) -> tuple[dict, list[str], list[str]]:
    """
    Ask the model to self-correct a broken tool.json.

    Parameters
    ----------
    tool_json     : dict        Current (broken) tool dict.
    error_history : list[dict]  Full Redis history for this run.
    latest_error  : str         The most recent error message (raw string).
    error_type    : str         "validation_error" or "execution_error".

    Returns
    -------
    (corrected_tool_json, cause_list, fix_list)
        corrected_tool_json : dict       New tool dict to retry with.
        cause_list          : list[str]  Bullet points explaining why it failed.
        fix_list            : list[str]  Bullet points describing the correction.

    Raises
    ------
    ValueError
        If the model response cannot be parsed or is missing required fields.
    """
    history_summary = _summarise_history(error_history)

    repair_system = textwrap.dedent(f"""
        You are an expert Python tool debugger for an AI agent platform.

        You will receive a broken tool JSON that failed {error_type} and must
        return a corrected version.

        The tool JSON must conform to this schema:
        {_SCHEMA_STR}

        IMPORTANT RULES:
        - Fix ONLY what is broken; preserve all working functions.
        - Do NOT add placeholders or stub values.
        - Do NOT leave TODO comments.
        - Return ONLY a valid JSON object with this exact shape — no markdown, no fences:
        {{
            "corrected_tool_json": {{ ... }},
            "cause": ["...", "..."],
            "fix":   ["...", "..."]
        }}
        where:
          cause = concise bullet points explaining WHY validation/execution failed
          fix   = concise bullet points describing the correction you applied
    """).strip()

    repair_user = textwrap.dedent(f"""
        === CURRENT TOOL JSON ===
        {json.dumps(tool_json, indent=2)}

        === PRIOR ERROR HISTORY ===
        {history_summary}

        === LATEST ERROR ({error_type.upper()}) ===
        {latest_error}

        Return the corrected tool JSON and your cause/fix analysis.
    """).strip()

    response = _model.invoke([
        SystemMessage(content=repair_system),
        HumanMessage(content=repair_user),
    ])

    raw = _parse_json(response.content)

    corrected = raw.get("corrected_tool_json")
    cause: list[str] = raw.get("cause", [])
    fix: list[str] = raw.get("fix", [])

    if not isinstance(corrected, dict):
        raise ValueError(
            f"repair_tool_json: model did not return 'corrected_tool_json' dict. "
            f"Got: {type(corrected)}"
        )

    # Validate corrected JSON against schema before returning
    validated = ToolSchema.model_validate(corrected)
    return validated.model_dump(), cause, fix
