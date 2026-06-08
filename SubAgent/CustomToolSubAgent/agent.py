"""
SubAgent/CustomToolSubAgent/agent.py — Groq-powered tool-calling subagent.

This module implements a ReAct-style agentic loop that:
  1. Receives a task (query) and a configurable dict containing conv_id.
  2. Looks up the conversation's template_id from the Conversations table.
  3. Fetches the tool_json schema from the Tools table using that template_id.
  4. Uses a Groq LLM to reason over the available functions and decide which
     to call, with what arguments, in what order (and potentially in parallel).
  5. Sends those calls to the Docker executor via execution_call().
  6. Feeds results back into the conversation history.
  7. Repeats until the LLM declares the task fully done.
  8. Returns a final synthesised answer string.

Public API
----------
    from SubAgent.CustomToolSubAgent.agent import run_custom_tool_subagent

    answer = run_custom_tool_subagent(
        query="What is the current temperature in London and Paris?",
        configurable={"conv_id": "conv-abc123"},
    )

Environment
-----------
    GROQ_API_KEY    — required
    POSTGRES_DSN    — required  (postgresql://user:pass@host:port/db)
    TOOL_RUNNER_URL — optional  (default http://localhost:8000)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap: make the project root importable regardless of cwd
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # AdapterAI/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

try:
    from groq import Groq
except ImportError as exc:
    raise ImportError(
        "groq is not installed. Run: pip install groq"
    ) from exc

_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not _GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Add it to your .env file or export it."
    )

_groq_client = Groq(api_key=_GROQ_API_KEY)

# Model to use — must match the MainAgent model
_MODEL = "openai/gpt-oss-120b"

# Maximum agentic iterations to prevent infinite loops
_MAX_ITERATIONS = 6

# ---------------------------------------------------------------------------
# Executor + DB imports
# ---------------------------------------------------------------------------

from SubAgent.CustomToolSubAgent.executer import execution_call  # noqa: E402
from SubAgent.CustomToolSubAgent.db import (               # noqa: E402
    fetch_template_id_for_conv,
    fetch_tool_json_for_template,
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise tool-calling agent. You have been given a task and a set of
tool functions defined in JSON. Your job is to accomplish the task by calling
the appropriate functions, inspecting their outputs, and synthesising a final
answer.

TOOL SCHEMA
-----------
{tool_schema}

RULES
-----
1. To call functions, respond with a JSON object (and NOTHING else) in this
   exact format:
   {{
     "action": "call_functions",
     "calls": {{
       "function_name_1": <args_as_list_or_dict>,
       "function_name_2": <args_as_list_or_dict>
     }}
   }}
   - args_as_list  → positional args, e.g. ["London", "celsius"]
   - args_as_dict  → keyword args,   e.g. {{"city": "London", "unit": "celsius"}}
   - You may call MULTIPLE functions in a single round (they execute in parallel).

2. When you have gathered enough information to fully answer the task, respond
   with a JSON object in this format:
   {{
     "action": "final_answer",
     "answer": "<your complete, well-formatted answer here>"
   }}

3. Never include prose outside the JSON object — your entire response must be
   valid JSON with either "call_functions" or "final_answer" as the action.

4. Only call functions that exist in the tool schema above.

5. If a function call returns an error, decide whether to retry with different
   arguments or skip that function and proceed with what you have.
"""

_TASK_MESSAGE = """\
TASK
----
{query}

Begin by analysing what functions you need to call to complete this task.
"""

# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------

def run_custom_tool_subagent(
    query: str,
    configurable: dict,
    max_iterations: int = _MAX_ITERATIONS,
) -> str:
    """
    Run the Groq-powered tool-calling subagent loop.

    The tool JSON schema is fetched automatically from PostgreSQL using the
    conv_id found in `configurable`:
      conv_id → Conversations.template_id → Tools.tool_json

    Parameters
    ----------
    query : str
        The task / question the agent must answer.
    configurable : dict
        A dict that must contain at least ``conv_id`` (the conversation
        primary key). Typically the same ``configurable`` dict passed by
        LangGraph / the main agent:
            {
                "conv_id":   "conv-abc123",
                "user_id":   "user-xyz",   # optional extra keys are ignored
                "thread_id": "...",
            }
    max_iterations : int
        Maximum number of call→observe rounds before forcing a final answer.

    Returns
    -------
    str
        The synthesised final answer produced by the LLM.

    Raises
    ------
    ValueError
        If ``conv_id`` is missing from configurable, or if the DB lookup fails.
    """
    # ── Extract conv_id ───────────────────────────────────────────────────────
    conv_id = configurable.get("conv_id", "")
    if not conv_id:
        raise ValueError(
            "conv_id is missing from configurable. "
            "Pass it as: configurable={'conv_id': '<id>', ...}"
        )

    # ── Fetch tool schema from DB ─────────────────────────────────────────────
    template_id = fetch_template_id_for_conv(conv_id)
    tool_json   = fetch_tool_json_for_template(template_id)

    # ── Build system prompt with embedded schema ──────────────────────────────
    schema_summary = _summarise_schema(tool_json)
    system_prompt  = _SYSTEM_PROMPT.format(tool_schema=schema_summary)

    # ── Initialise conversation history ───────────────────────────────────────
    history: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": _TASK_MESSAGE.format(query=query)},
    ]

    # ── Agentic loop ──────────────────────────────────────────────────────────
    for iteration in range(1, max_iterations + 1):
        # Call Groq LLM
        response       = _chat(history)
        assistant_text = response.strip()

        # Append assistant turn to history
        history.append({"role": "assistant", "content": assistant_text})

        # Parse JSON action
        action_obj = _parse_action(assistant_text)

        if action_obj is None:
            # Unparseable — prompt LLM to correct itself
            history.append({
                "role": "user",
                "content": (
                    "ERROR: Your last response was not valid JSON. "
                    "Please respond with a valid JSON object using the "
                    '"call_functions" or "final_answer" action.'
                ),
            })
            continue

        action = action_obj.get("action", "")

        # ── Final answer ──────────────────────────────────────────────────────
        if action == "final_answer":
            return str(action_obj.get("answer", ""))

        # ── Function calls ────────────────────────────────────────────────────
        if action == "call_functions":
            calls: dict = action_obj.get("calls", {})
            if not calls:
                history.append({
                    "role": "user",
                    "content": 'ERROR: "calls" was empty. Provide at least one function call.',
                })
                continue

            # Execute via Docker executor
            execution_result = _execute_calls(calls, tool_json)

            # Feed result back as a user (tool-observation) message
            observation = json.dumps(execution_result, indent=2, ensure_ascii=False)
            history.append({
                "role": "user",
                "content": (
                    f"TOOL RESULTS (iteration {iteration}):\n"
                    f"```json\n{observation}\n```\n\n"
                    "Now review the results and either call more functions if "
                    "needed, or provide your final answer."
                ),
            })
            continue

        # Unknown action
        history.append({
            "role": "user",
            "content": (
                f'ERROR: Unknown action "{action}". '
                'Use "call_functions" or "final_answer".'
            ),
        })

    # ── Force a final answer after max iterations ─────────────────────────────
    history.append({
        "role": "user",
        "content": (
            "You have reached the maximum number of iterations. "
            "Based on everything gathered so far, respond with a "
            '"final_answer" JSON object containing your best synthesised answer.'
        ),
    })
    response   = _chat(history)
    action_obj = _parse_action(response.strip())
    if action_obj and action_obj.get("action") == "final_answer":
        return str(action_obj.get("answer", ""))

    # Absolute fallback — return raw LLM text
    return response.strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chat(history: list[dict]) -> str:
    """Send messages to Groq and return the assistant content string."""
    completion = _groq_client.chat.completions.create(
        model=_MODEL,
        messages=history,
        temperature=0.0,
        max_tokens=4096,
    )
    return completion.choices[0].message.content or ""


def _parse_action(text: str) -> dict | None:
    """
    Try to extract and parse a JSON object from the LLM response.
    The model is instructed to respond with pure JSON, but we strip
    code-fence wrappers just in case.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        cleaned = "\n".join(inner).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Last-ditch: find the outermost {...} block
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _execute_calls(calls: dict, tool_json: dict) -> dict:
    """
    Forward function calls to the Docker executor.
    On HTTP or validation errors, wrap them per-function so the LLM can recover.
    """
    try:
        return execution_call(function_calls=calls, tool_json=tool_json)
    except Exception as exc:  # noqa: BLE001
        return {fn: {"error": str(exc)} for fn in calls}


def _summarise_schema(tool_json: dict) -> str:
    """
    Build a compact, LLM-readable summary of the available functions.
    Full details (parameter types, descriptions, output fields) are included
    so the model can construct correct calls.
    """
    lines: list[str] = [
        f"Tool: {tool_json.get('tool_name', 'unknown')}",
        f"Description: {tool_json.get('tool_description', '')}",
        "",
        "Available functions:",
    ]

    for fn in tool_json.get("functions", []):
        lines.append(f"\n  • {fn['name']}")
        lines.append(f"    Description: {fn.get('description', '')}")

        params = fn.get("parameters", [])
        if params:
            lines.append("    Parameters:")
            for p in params:
                req        = "required" if p.get("required") else "optional"
                example    = p.get("example", "")
                example_str = f" (example: {json.dumps(example)})" if example != "" else ""
                lines.append(
                    f"      - {p['name']} ({p.get('type', 'any')}, {req}): "
                    f"{p.get('description', '')}{example_str}"
                )

        outputs = fn.get("outputs", [])
        if outputs:
            lines.append("    Returns:")
            for o in outputs:
                lines.append(
                    f"      - {o['name']} ({o.get('type', 'any')}): "
                    f"{o.get('description', '')}"
                )

    return "\n".join(lines)
