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

from ToolGeneration.schemas import ToolSchema
from ToolGeneration.SystemPrompt import system_prompt

load_dotenv()

# ---------------------------------------------------------------------------
# Shared Groq client
# ---------------------------------------------------------------------------

_MODEL_NAME = "llama-3.3-70b-versatile"
_model = ChatGroq(model=_MODEL_NAME)

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
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    return raw


def _parse_json(raw: str) -> dict:
    """Try json_repair first, fall back to stdlib json.loads."""
    raw = _strip_fences(raw)
    try:
        import json_repair  # type: ignore
        return json_repair.loads(raw)
    except ImportError:
        return json.loads(raw)


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
