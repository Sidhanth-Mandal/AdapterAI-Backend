"""
executer.py — Client-side dispatcher for the Docker tool executor.

Contains execution_call() which sends a function call request to the
Docker container running Docker_exec/api.py, receives the result, and
returns it as-is.

Usage
-----
    from codeexecuter.executer import execution_call

    # Single function
    result = execution_call(
        function_calls={"get_player_stats": ["Virat Kohli"]},
    )

    # Multiple functions in one call
    result = execution_call(
        function_calls={
            "get_player_stats": ["Virat Kohli"],
            "get_team_stats":   ["India"],
        },
    )

    # Pass args as a dict (keyword args) instead of a list (positional)
    result = execution_call(
        function_calls={"get_player_stats": {"player_name": "Virat Kohli"}},
    )

Environment
-----------
    TOOL_RUNNER_URL  — base URL of the Docker executor (default: http://localhost:8000)

Flow
----
    execution_call()
         │  POST /execute
         │  { function_calls: {fn: args}, tool_json: {...} }
         ▼
    Docker container  (Docker_exec/api.py  port 8000)
         │  tool_compiler.py  →  writes code to file, installs deps
         │  runner.py         →  imports file, calls function(s)
         ▼
    { fn_name: result, ... }  returned as-is
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_RUNNER_URL      = os.getenv("TOOL_RUNNER_URL", "http://localhost:8000")
_DEFAULT_TOOL_JSON_PATH  = Path("codeexecuter/generated_tool.json")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execution_call(
    function_calls: dict,
    tool_json: Union[dict, str, Path] = _DEFAULT_TOOL_JSON_PATH,
    runner_url: Union[str, None] = None,
    timeout: int = 120,
) -> dict:
    """
    Send one or more function-call requests to the Docker executor and return
    the results exactly as received.

    Parameters
    ----------
    function_calls : dict
        Mapping of function name → arguments.
        Arguments can be:
          - list  : passed as positional args  →  fn(*args)
          - dict  : passed as keyword args     →  fn(**kwargs)
          - any   : passed as single positional arg

        Examples::

            {"get_player_stats": ["Virat Kohli"]}
            {"get_player_stats": {"player_name": "Virat Kohli"}}
            {"get_player_stats": ["Virat Kohli"], "get_team_stats": ["India"]}

    tool_json : dict | str | Path
        Either the full tool JSON dict, a path to ``generated_tool.json``,
        or a raw JSON string.
        Default: ``codeexecuter/generated_tool.json``

    runner_url : str | None
        Base URL of the Docker executor.
        Falls back to the ``TOOL_RUNNER_URL`` env var, then
        ``http://localhost:8000``.

    timeout : int
        HTTP request timeout in seconds. Default: 120.

    Returns
    -------
    dict
        ``{ function_name: result }`` exactly as returned by the Docker runner.
        If a single function was requested the dict still has one key — the
        caller can unpack it with ``result["get_player_stats"]``.

    Raises
    ------
    FileNotFoundError
        If ``tool_json`` is a path that does not exist.
    requests.HTTPError
        If the Docker runner returns a non-2xx HTTP status.
    ValueError
        If ``function_calls`` is empty or not a dict.
    """
    # ── Validate ─────────────────────────────────────────────────────────────
    if not isinstance(function_calls, dict) or not function_calls:
        raise ValueError(
            "function_calls must be a non-empty dict: "
            '{"function_name": [args], ...}'
        )

    # ── Resolve tool_json ────────────────────────────────────────────────────
    resolved_tool_json = _load_tool_json(tool_json)

    # ── Validate function names against schema ───────────────────────────────
    available_fns = {
        fn["name"] for fn in resolved_tool_json.get("functions", [])
    }
    unknown = set(function_calls) - available_fns
    if unknown:
        raise ValueError(
            f"Unknown function(s): {sorted(unknown)}. "
            f"Available in tool JSON: {sorted(available_fns)}"
        )

    # ── Build request payload ────────────────────────────────────────────────
    payload = {
        "function_calls": function_calls,
        "tool_json":      resolved_tool_json,
    }

    # ── POST to Docker ───────────────────────────────────────────────────────
    url = (runner_url or _DEFAULT_RUNNER_URL).rstrip("/")
    endpoint = f"{url}/execute"

    resp = requests.post(endpoint, json=payload, timeout=timeout)

    # Surface runner errors with detail
    if not resp.ok:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise requests.HTTPError(
            f"Docker executor returned HTTP {resp.status_code}: {detail}",
            response=resp,
        )

    return resp.json()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_tool_json(tool_json: Union[dict, str, Path]) -> dict:
    """Resolve tool_json input to a plain dict."""
    if isinstance(tool_json, dict):
        return tool_json

    p = Path(tool_json)
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    # Last-ditch: try to parse as a raw JSON string
    try:
        return json.loads(str(tool_json))
    except json.JSONDecodeError:
        raise FileNotFoundError(
            f"tool_json path does not exist and is not valid JSON: '{tool_json}'"
        )
