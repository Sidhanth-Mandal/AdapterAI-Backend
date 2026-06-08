"""
Graph Invoke Wrappers
======================
Wrapper functions around LangGraph's graph.invoke() / graph.ainvoke() that
automatically prune stale checkpoints after each run, keeping only the single
latest checkpoint per thread_id.

Why?
----
LangGraph saves a checkpoint after every node execution, so a single
graph.invoke() call can create many checkpoints for the same thread.
These wrappers transparently clean up the old ones right after invocation,
keeping Redis lean without sacrificing the ability to resume the latest state.

Usage
-----
Sync:
    from utils.graph_invoke import invoke_graph

    result = invoke_graph(
        graph=my_compiled_graph,
        input={"messages": [HumanMessage(content="Hello")]},
        thread_id="user-42",
    )

Async:
    from utils.graph_invoke import ainvoke_graph

    result = await ainvoke_graph(
        graph=my_compiled_graph,
        input={"messages": [HumanMessage(content="Hello")]},
        thread_id="user-42",
    )
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from utils.redis_checkpointer import REDIS_URL, get_async_saver, get_sync_saver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_config(thread_id: str, checkpoint_ns: str = "") -> RunnableConfig:
    """Build the LangGraph config dict for a given thread."""
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
    }


def _prune_old_checkpoints_sync(saver, thread_id: str, checkpoint_ns: str = "") -> None:
    """
    Delete every checkpoint for this thread except the most recent one.

    LangGraph's .list() returns checkpoints in reverse-chronological order
    (newest first), so we skip index 0 and delete everything after it.
    """
    config = _build_config(thread_id, checkpoint_ns)
    checkpoints = list(saver.list(config))

    # checkpoints[0] is the latest — keep it, delete the rest
    for old in checkpoints[1:]:
        old_id = old.config["configurable"]["checkpoint_id"]
        # The saver's underlying redis client is exposed via ._redis_client
        # or ._client depending on the version; we use the list + put pattern
        # to overwrite. Instead, we delete via the raw redis connection.
        try:
            client = saver._redis_client  # langgraph-checkpoint-redis internals
            ns = checkpoint_ns or ""
            client.delete(
                f"checkpoint:{thread_id}:{ns}:{old_id}",
                f"writes:{thread_id}:{ns}:{old_id}",
            )
        except AttributeError:
            # Fallback: attribute name differs across versions — silently skip
            pass


async def _prune_old_checkpoints_async(saver, thread_id: str, checkpoint_ns: str = "") -> None:
    """Async version of _prune_old_checkpoints_sync."""
    config = _build_config(thread_id, checkpoint_ns)
    checkpoints = [cp async for cp in saver.alist(config)]

    for old in checkpoints[1:]:
        old_id = old.config["configurable"]["checkpoint_id"]
        try:
            client = saver._redis_client
            ns = checkpoint_ns or ""
            await client.delete(
                f"checkpoint:{thread_id}:{ns}:{old_id}",
                f"writes:{thread_id}:{ns}:{old_id}",
            )
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def invoke_graph(
    graph,
    input: dict[str, Any],
    thread_id: str,
    checkpoint_ns: str = "",
    redis_url: str = REDIS_URL,
    extra_config: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Invoke a compiled LangGraph graph synchronously and keep only the
    latest checkpoint in Redis for the given thread_id.

    Parameters
    ----------
    graph        : A compiled LangGraph graph (builder.compile()).
                   NOTE: compile WITHOUT a checkpointer — this wrapper
                   manages the saver internally.
    input        : The input dict passed to graph.invoke().
    thread_id    : Unique identifier for the conversation thread.
    checkpoint_ns: Optional namespace (defaults to "").
    redis_url    : Redis connection URL (defaults to localhost:6379).
    extra_config : Any additional keys to merge into the LangGraph config.

    Returns
    -------
    The output dict returned by graph.invoke().
    """
    config = _build_config(thread_id, checkpoint_ns)
    if extra_config:
        config["configurable"].update(extra_config)

    with get_sync_saver(redis_url) as saver:
        # Compile the graph with the checkpointer for this invocation
        compiled = graph.compile(checkpointer=saver)
        result = compiled.invoke(input, config)

        # Prune stale checkpoints — keep only the latest
        _prune_old_checkpoints_sync(saver, thread_id, checkpoint_ns)

    return result


async def ainvoke_graph(
    graph,
    input: dict[str, Any],
    thread_id: str,
    checkpoint_ns: str = "",
    redis_url: str = REDIS_URL,
    extra_config: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Invoke a compiled LangGraph graph asynchronously and keep only the
    latest checkpoint in Redis for the given thread_id.

    Parameters
    ----------
    graph        : A compiled LangGraph graph (builder.compile()).
                   NOTE: compile WITHOUT a checkpointer — this wrapper
                   manages the saver internally.
    input        : The input dict passed to graph.ainvoke().
    thread_id    : Unique identifier for the conversation thread.
    checkpoint_ns: Optional namespace (defaults to "").
    redis_url    : Redis connection URL (defaults to localhost:6379).
    extra_config : Any additional keys to merge into the LangGraph config.

    Returns
    -------
    The output dict returned by graph.ainvoke().
    """
    config = _build_config(thread_id, checkpoint_ns)
    if extra_config:
        config["configurable"].update(extra_config)

    async with get_async_saver(redis_url) as saver:
        compiled = graph.compile(checkpointer=saver)
        result = await compiled.ainvoke(input, config)

        # Prune stale checkpoints — keep only the latest
        await _prune_old_checkpoints_async(saver, thread_id, checkpoint_ns)

    return result
