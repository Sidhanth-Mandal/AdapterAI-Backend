"""
SubAgent/CustomToolSubAgent/calling.py — Callable-tool wrapper for the CustomToolSubAgent.

Exposes call_custom_tool_subagent as a LangChain @tool-decorated function so
the main agent can bind it just like any other builtin tool.

The main agent passes:
  • query         — the natural-language task
  • configurable  — JSON string with at least {"conv_id": "<id>"}
                    (the same configurable dict the main agent already holds)

The subagent uses conv_id to look up the tool schema from PostgreSQL, so the
main agent does NOT need to know anything about which tool is attached.

Usage in the main agent
-----------------------
    from SubAgent.CustomToolSubAgent.calling import TOOLS

    model = ChatGroq(...).bind_tools(TOOLS)

Direct Python call (no LangChain required)
-------------------------------------------
    from SubAgent.CustomToolSubAgent.calling import call_tool_subagent_direct

    result = call_tool_subagent_direct(
        query="What is the temperature in London?",
        configurable={"conv_id": "conv-abc123"},
    )
"""

from __future__ import annotations

import json

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from SubAgent.CustomToolSubAgent.agent import run_custom_tool_subagent


# ---------------------------------------------------------------------------
# LangChain @tool-decorated wrapper
# ---------------------------------------------------------------------------

@tool
def call_custom_tool_subagent(
    query: str,
    config: RunnableConfig,
) -> str:
    """Invoke the CustomToolSubAgent to complete a task using the conversation's tool.

    The subagent automatically discovers which tool is linked to the current
    conversation by reading conv_id from the LangGraph config. It then fetches
    the tool schema from the database, decides which functions to call (and how
    many times), executes them via the Docker tool executor, and synthesises a
    final answer — all autonomously.

    Use this tool whenever the user's request should be handled by the custom
    tool that was generated for this conversation's template.

    Args:
        query: A clear description of the task the subagent must complete.
               Be specific — include all relevant parameters or context the
               subagent needs to select and call the right tool functions.
               Example: "Get the current temperature in London and Paris and
                          compare them."
        config: Injected automatically by LangGraph. Must contain
                config["configurable"]["conv_id"] so the subagent can look up
                the correct tool schema from the database.

    Returns:
        A complete, synthesised answer string produced by the subagent after
        executing the necessary tool functions. If the subagent encounters an
        unrecoverable error, a descriptive error message is returned instead.
    """
    try:
        configurable: dict = (config or {}).get("configurable", {})
        if not configurable.get("conv_id"):
            return (
                "[CustomToolSubAgent ERROR] conv_id is missing from "
                "config['configurable']. Ensure it is set when invoking the graph."
            )

        return run_custom_tool_subagent(
            query=query,
            configurable=configurable,
        )

    except Exception as exc:  # noqa: BLE001
        return f"[CustomToolSubAgent ERROR] {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Direct (non-LangChain) entry-point
# ---------------------------------------------------------------------------

def call_tool_subagent_direct(
    query: str,
    configurable: dict,
) -> str:
    """
    Thin wrapper around run_custom_tool_subagent for direct Python calls
    (no LangChain / RunnableConfig needed).

    Parameters
    ----------
    query : str
        The task the subagent should complete.
    configurable : dict
        Must contain at least ``conv_id``.
        Example: {"conv_id": "conv-abc123", "user_id": "user-xyz"}

    Returns
    -------
    str
        The final synthesised answer.
    """
    return run_custom_tool_subagent(
        query=query,
        configurable=configurable,
    )


# ---------------------------------------------------------------------------
# Tool registry (main agent imports this)
# ---------------------------------------------------------------------------

TOOLS = [call_custom_tool_subagent]
