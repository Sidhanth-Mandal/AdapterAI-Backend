"""
Docker_exec/api.py — FastAPI server running inside the Docker container.

Receives POST /execute requests from execution_call(), orchestrates
tool_compiler.py and runner.py, and returns the function results.

Endpoint
--------
    POST /execute
    Body: {
        "function_calls": {"get_player_stats": ["Virat Kohli"]},
        "tool_json":      { ...full generated_tool.json content... }
    }

    GET /health
    Returns: {"status": "ok"}

Start
-----
    python api.py
    # or via Dockerfile:  CMD ["python", "api.py"]
"""

from __future__ import annotations

import traceback
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from tool_compiler import compile_tool
from runner import run_function

app = FastAPI(
    title="Tool Executor",
    description="Compiles generated_tool.json and executes requested functions.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class ExecuteRequest(BaseModel):
    function_calls: dict   # { function_name: args (list | dict) }
    tool_json:      dict   # full generated_tool.json content


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/execute")
async def execute(req: ExecuteRequest):
    """
    1. Compile tool_json → executable Python file (+ install deps).
    2. For each function in function_calls, run it via runner.py.
    3. Return { function_name: result } dict.
    """
    if not req.function_calls:
        raise HTTPException(status_code=400, detail="function_calls must not be empty")

    # ── Step 1: compile ──────────────────────────────────────────────────────
    try:
        compiled_path = compile_tool(req.tool_json)
    except Exception as e:
        tb = traceback.format_exc(limit=5)
        raise HTTPException(
            status_code=400,
            detail={
                "stage":     "compilation",
                "error":     str(e),
                "type":      type(e).__name__,
                "traceback": tb,
            },
        )

    # ── Step 2: execute each function ────────────────────────────────────────
    results: dict[str, Any] = {}

    for fn_name, args in req.function_calls.items():
        try:
            result = run_function(compiled_path, fn_name, args)
            results[fn_name] = result
        except Exception as e:
            tb = traceback.format_exc(limit=5)
            results[fn_name] = {
                "error":     str(e),
                "type":      type(e).__name__,
                "function":  fn_name,
                "args":      args,
                "traceback": tb,
            }

    return results


@app.get("/health")
def health():
    """Simple health-check endpoint."""
    return {"status": "ok", "service": "tool-executor"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
