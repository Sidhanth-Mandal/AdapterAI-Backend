"""
pipeline.py — Main entry point for the ToolGeneration pipeline.

Public API
----------
generate_tool(template_id: str) -> dict
    Full end-to-end pipeline:
      1. Fetch template from PostgreSQL
      2. Generate tool.json via Groq
      3. Initialise Redis error history
      4. Validate → self-repair loop (max 6 attempts)
      5. Execution test → self-repair loop (within same attempt budget)
      6. Extract metadata
      7. Persist to PostgreSQL (insert tool, update template)
      8. Move Redis key to final tool_id
      9. Return success/failure dict

Success response:
    {"status": "success",  "tool_id": "to00003", "template_id": "tem00001"}

Failure response:
    {"status": "failed",   "template_id": "tem00001",
     "attempts": 6,        "reason": "..."}
"""

from __future__ import annotations

import json
import logging
import traceback
import uuid
from typing import Any

import requests as _requests  # aliased to avoid shadowing

import ToolGeneration.db as db
import ToolGeneration.generator as generator
import ToolGeneration.metadata as meta_module
import ToolGeneration.redis_client as redis_client
from .executer import execution_call
from .validator import validate_tool

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_ATTEMPTS: int = 6
LANGUAGE: str = "python"
VERSION: str = "1.0.0"

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_function_calls(tool_json: dict) -> dict:
    """
    Build the function_calls dict for execution_call() by reading the
    example values embedded in each function's parameters.

    If a parameter has no example, a safe stub is used so the call can
    still be attempted.
    """
    _TYPE_STUBS: dict[str, Any] = {
        "str": "test_value",
        "int": 1,
        "float": 1.0,
        "bool": True,
        "list": [],
        "dict": {},
    }

    function_calls: dict[str, Any] = {}
    for fn in tool_json.get("functions", []):
        fn_name: str = fn.get("name", "")
        if not fn_name:
            continue

        kwargs: dict[str, Any] = {}
        for param in fn.get("parameters", []):
            p_name = param.get("name")
            if not p_name:
                continue
            example = param.get("example")
            if example is not None:
                kwargs[p_name] = example
            else:
                kwargs[p_name] = _TYPE_STUBS.get(param.get("type", "str"), "stub")

        function_calls[fn_name] = kwargs

    return function_calls


def _collect_validation_errors(report) -> str:
    """Format a ValidationReport into a single error string for the repair prompt."""
    lines: list[str] = []
    for stage in report.stages:
        if not stage.passed:
            lines.append(f"Stage: {stage.name}")
            for err in stage.errors:
                lines.append(f"  ERROR: {err}")
    return "\n".join(lines) if lines else "Unknown validation error"


def _latest_stage_name(report) -> str:
    """Return the name of the first failing stage."""
    for stage in report.stages:
        if not stage.passed:
            return stage.name
    return "Unknown"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_tool(template_id: str) -> dict:
    """
    End-to-end tool generation pipeline.

    Parameters
    ----------
    template_id : str
        Primary key of the template row in the Templates table.
        The tool_generation_prompt column of this row is used as the
        user prompt for the LLM.

    Returns
    -------
    dict
        Success: {"status": "success",  "tool_id": str, "template_id": str}
        Failure: {"status": "failed",   "template_id": str,
                  "attempts": int,      "reason": str}
    """

    # ── 1. Fetch template ────────────────────────────────────────────────────
    log.info(f"[Pipeline] Fetching template: {template_id}")
    template = db.fetch_template(template_id)

    if template is None:
        log.error(f"[Pipeline] Template '{template_id}' not found in database.")
        return {
            "status": "failed",
            "template_id": template_id,
            "attempts": 0,
            "reason": f"Template '{template_id}' not found in the database.",
        }

    tool_generation_prompt: str = template.get("tool_generation_prompt") or ""
    if not tool_generation_prompt.strip():
        log.error("[Pipeline] Template has empty tool_generation_prompt.")
        return {
            "status": "failed",
            "template_id": template_id,
            "attempts": 0,
            "reason": "Template 'tool_generation_prompt' is empty.",
        }

    # ── 2. Initial tool generation ───────────────────────────────────────────
    log.info("[Pipeline] Generating initial tool.json …")
    try:
        tool_json: dict = generator.generate_tool_json(tool_generation_prompt)
    except Exception as exc:
        log.error(f"[Pipeline] Initial generation failed: {exc}")
        return {
            "status": "failed",
            "template_id": template_id,
            "attempts": 1,
            "reason": f"Initial tool generation failed: {exc}",
        }

    # ── 3. Initialise Redis error history ────────────────────────────────────
    run_key: str = f"gen:{template_id}:{uuid.uuid4().hex[:8]}"
    log.info(f"[Pipeline] Redis run key: {run_key}")
    redis_client.init_error_history(run_key, tool_generation_prompt)

    # ── 4 + 5. Validate → Repair → Execute → Repair loop ────────────────────
    attempts: int = 0

    while attempts < MAX_ATTEMPTS:
        attempts += 1
        log.info(f"[Pipeline] Attempt {attempts}/{MAX_ATTEMPTS}")

        # ── Validation ───────────────────────────────────────────────────────
        log.info("[Pipeline] Running validator …")
        report = validate_tool(tool_json)

        if not report.passed or not report.safe:
            error_str = _collect_validation_errors(report)
            stage_name = _latest_stage_name(report)

            if not report.safe:
                error_str = "Tool flagged as UNSAFE — dangerous patterns detected.\n" + error_str

            log.warning(f"[Pipeline] Validation FAILED (attempt {attempts}):\n{error_str}")

            # Record validation failure
            redis_client.append_error_history(run_key, {
                "type":   "validation_error",
                "stage":  stage_name,
                "errors": report.errors,
            })

            if attempts >= MAX_ATTEMPTS:
                break

            # Ask model to repair
            log.info("[Pipeline] Requesting repair from model …")
            history = redis_client.get_error_history(run_key)
            try:
                tool_json, cause, fix = generator.repair_tool_json(
                    tool_json=tool_json,
                    error_history=history,
                    latest_error=error_str,
                    error_type="validation_error",
                )
            except Exception as exc:
                log.error(f"[Pipeline] Repair call failed: {exc}")
                redis_client.append_error_history(run_key, {
                    "type":  "repair",
                    "cause": [f"Repair call raised exception: {exc}"],
                    "fix":   [],
                })
                continue

            redis_client.append_error_history(run_key, {
                "type":  "repair",
                "cause": cause,
                "fix":   fix,
            })
            log.info("[Pipeline] Repair applied — retrying validation …")
            continue  # retry validation with repaired json

        # ── Validation passed — run execution test ───────────────────────────
        log.info("[Pipeline] Validation passed. Running execution test …")
        function_calls = _build_function_calls(tool_json)

        if not function_calls:
            log.warning("[Pipeline] No functions found in tool_json — skipping execution test.")
            break  # treat as passing if there are no functions to test

        try:
            exec_result = execution_call(
                function_calls=function_calls,
                tool_json=tool_json,
            )
            log.info(f"[Pipeline] Execution succeeded: {list(exec_result.keys())}")
            break  # all good — exit the loop

        except _requests.exceptions.ConnectionError as conn_err:
            # Docker executor container is unreachable — hard abort
            log.error(f"[Pipeline] Docker executor is unreachable: {conn_err}")
            redis_client.append_error_history(run_key, {
                "type":  "execution_error",
                "error": f"Docker executor unreachable: {conn_err}",
            })
            return {
                "status": "failed",
                "template_id": template_id,
                "attempts": attempts,
                "reason": (
                    "Docker executor container is not running (connection refused on port 8000). "
                    "Tool cannot be execution-tested — aborting."
                ),
            }

        except Exception as exec_err:
            exec_error_str = f"{type(exec_err).__name__}: {exec_err}"
            log.warning(f"[Pipeline] Execution FAILED (attempt {attempts}): {exec_error_str}")

            redis_client.append_error_history(run_key, {
                "type":  "execution_error",
                "error": exec_error_str,
            })

            if attempts >= MAX_ATTEMPTS:
                break

            # Ask model to repair based on execution failure
            log.info("[Pipeline] Requesting repair from model (execution error) …")
            history = redis_client.get_error_history(run_key)
            try:
                tool_json, cause, fix = generator.repair_tool_json(
                    tool_json=tool_json,
                    error_history=history,
                    latest_error=exec_error_str,
                    error_type="execution_error",
                )
            except Exception as exc:
                log.error(f"[Pipeline] Repair call failed: {exc}")
                redis_client.append_error_history(run_key, {
                    "type":  "repair",
                    "cause": [f"Repair call raised exception: {exc}"],
                    "fix":   [],
                })
                continue

            redis_client.append_error_history(run_key, {
                "type":  "repair",
                "cause": cause,
                "fix":   fix,
            })
            log.info("[Pipeline] Repair applied — restarting from validation …")
            # loop continues — will re-validate before re-executing

    else:
        # Exhausted MAX_ATTEMPTS via the while condition
        pass

    # ── Check final state ────────────────────────────────────────────────────
    final_report = validate_tool(tool_json)
    if not final_report.passed or not final_report.safe:
        log.error(f"[Pipeline] Exhausted {MAX_ATTEMPTS} attempts — tool still invalid.")
        return {
            "status": "failed",
            "template_id": template_id,
            "attempts": attempts,
            "reason": "Unable to generate a valid executable tool — maximum correction attempts exceeded.",
        }

    # ── 6. Extract metadata ──────────────────────────────────────────────────
    log.info("[Pipeline] Extracting tool metadata …")
    metadata: dict = meta_module.extract_tool_metadata(tool_json)

    # ── 7. Generate incremental tool_id ─────────────────────────────────────
    tool_id: str = db.get_next_tool_id()
    log.info(f"[Pipeline] Assigned tool_id: {tool_id}")

    # ── 8. Persist tool to PostgreSQL ────────────────────────────────────────
    log.info(f"[Pipeline] Inserting tool into database …")
    try:
        db.insert_tool(
            tool_id=tool_id,
            template_id=template_id,
            name=metadata["name"],
            description=metadata["description"],
            language=LANGUAGE,
            tool_json=tool_json,
            version=VERSION,
        )
    except Exception as exc:
        log.error(f"[Pipeline] DB insert failed: {exc}")
        return {
            "status": "failed",
            "template_id": template_id,
            "attempts": attempts,
            "reason": f"Database insert failed: {exc}",
        }

    # ── 9. Update template's tool_information ───────────────────────────────
    log.info("[Pipeline] Updating Templates.tool_information …")
    try:
        db.update_template_tool_information(template_id, metadata["tool_information"])
    except Exception as exc:
        # Non-fatal — tool is already persisted; log and continue
        log.warning(f"[Pipeline] Could not update template tool_information: {exc}")

    # ── 10. Move Redis history to permanent tool_id key ──────────────────────
    log.info(f"[Pipeline] Moving Redis history: {run_key} → {tool_id}")
    try:
        redis_client.rename_key(run_key, tool_id)
    except Exception as exc:
        # Non-fatal — history is still under the run key
        log.warning(f"[Pipeline] Redis rename failed: {exc}")

    log.info(f"[Pipeline] ✓ Tool '{tool_id}' generated successfully.")
    return {
        "status": "success",
        "tool_id": tool_id,
        "template_id": template_id,
    }
