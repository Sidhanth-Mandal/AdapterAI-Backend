"""
MainAgent/service.py
---------------------
Primary entry point for the MainAgent orchestration module.

Exposes a single async callable:

    async def chat(
        template_id: str,
        user_id: str,
        conv_id: str,
        user_prompt: str,
        if_attachment: bool,
    ) -> str

This function is the only interface the API layer needs to call.
No CLI entrypoint exists — the API layer invokes chat() directly.

Design notes
------------
* The LangGraph pipeline is compiled fresh on every call.  There is no
  persistent graph state between invocations; all state lives in Redis
  and PostgreSQL.
* graph.ainvoke() is used (fully async) — all nodes are async functions.
* The RunnableConfig carries user_id, conv_id, and thread_id so that
  LangChain tools requiring identity context receive it automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is on sys.path so cross-module imports work
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # AdapterAI/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# LangSmith tracing (must be imported after .env is loaded)
# ---------------------------------------------------------------------------
from utils.tracing import traceable  # noqa: E402

# ---------------------------------------------------------------------------
# Internal imports (env vars must be loaded first)
# ---------------------------------------------------------------------------

from MainAgent.graph import build_graph
from MainAgent.state import OrchestratorState


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@traceable(
    name="main_agent_chat",
    tags=["main-agent"],
    metadata={"pipeline": "MainAgent"},
)
async def chat(
    template_id: str,
    user_id: str,
    conv_id: str,
    user_prompt: str,
    if_attachment: bool,
) -> str:
    """
    Handle one user turn through the full MainAgent pipeline.

    The pipeline executes the following nodes in sequence:
      load_template_node → load_memory_node → build_context_node →
      orchestrator_agent_node → persist_messages_node →
      summary_node → redis_update_node

    Parameters
    ----------
    template_id : str
        Identifier of the template that configures this AI's behaviour.
        Used to load behavior_prompt and custom_tool_information.
    user_id : str
        Identifier of the user sending the message.
        Passed through to tools and stored with persisted messages.
    conv_id : str
        Identifier of the active conversation.
        Keys all memory, message storage, and Redis caching.
    user_prompt : str
        The user's current message text.
    if_attachment : bool
        Set True when the conversation contains uploaded files.
        Enables the retrieval tool and adjusts the agent's context.

    Returns
    -------
    str
        The assistant's final response text for this turn.

    Raises
    ------
    Exception
        Propagated from the pipeline on unrecoverable errors
        (e.g. missing GROQ_API_KEY, DB connection failure).
    """
    graph = build_graph()

    initial_state: OrchestratorState = {
        # Request context
        "template_id":    template_id,
        "user_id":        user_id,
        "conv_id":        conv_id,
        "user_prompt":    user_prompt,
        "if_attachment":  if_attachment,
        # Template config — populated by load_template_node
        "behavior_prompt":         "",
        "custom_tool_information": "",
        # Conversation memory — populated by load_memory_node
        "summary":                     "",
        "recent_messages":             [],
        "unsummarized_token_count":    0,
        "last_summarized_message_seq": 0,
        # LangChain messages — populated by build_context_node, extended by agent
        "messages":       [],
        # Orchestrator output — populated by orchestrator_agent_node
        "final_response": "",
        "tools_called":   [],
        # Persistence tracking — populated by persist_messages_node
        "new_user_seq":       0,
        "new_assistant_seq":  0,
    }

    # Pass identity context to tools via RunnableConfig
    config = {
        "configurable": {
            "user_id":   user_id,
            "conv_id":   conv_id,
            "thread_id": f"{user_id}:{conv_id}",
        }
    }

    result = await graph.ainvoke(initial_state, config=config)

    return result["final_response"]


@traceable(
    name="main_agent_chat_debug",
    tags=["main-agent", "debug"],
    metadata={"pipeline": "MainAgent"},
)
async def chat_debug(
    template_id: str,
    user_id: str,
    conv_id: str,
    user_prompt: str,
    if_attachment: bool,
) -> tuple[str, list[str]]:
    """
    Same as chat() but also returns the list of tool names that were
    actually invoked during the turn.

    Returns
    -------
    (final_response, tools_called)
        final_response : str   — the assistant's reply
        tools_called   : list  — ordered list of tool names that ran,
                                   e.g. ['call_custom_tool_subagent', 'web_search']
                                   Empty list means no tools were called.
    """
    graph = build_graph()

    initial_state: OrchestratorState = {
        "template_id":    template_id,
        "user_id":        user_id,
        "conv_id":        conv_id,
        "user_prompt":    user_prompt,
        "if_attachment":  if_attachment,
        "behavior_prompt":         "",
        "custom_tool_information": "",
        "summary":                     "",
        "recent_messages":             [],
        "unsummarized_token_count":    0,
        "last_summarized_message_seq": 0,
        "messages":       [],
        "final_response": "",
        "tools_called":   [],
        "new_user_seq":       0,
        "new_assistant_seq":  0,
    }

    config = {
        "configurable": {
            "user_id":   user_id,
            "conv_id":   conv_id,
            "thread_id": f"{user_id}:{conv_id}",
        }
    }

    result = await graph.ainvoke(initial_state, config=config)
    return result["final_response"], result.get("tools_called", [])


# ---------------------------------------------------------------------------
# Streaming public API
# ---------------------------------------------------------------------------

from typing import AsyncGenerator  # noqa: E402

from MainAgent.nodes.load_template_node import load_template_node
from MainAgent.nodes.load_memory_node import load_memory_node
from MainAgent.nodes.build_context_node import build_context_node
from MainAgent.nodes.orchestrator_agent_node import orchestrator_agent_node_stream
from MainAgent.nodes.persist_messages_node import persist_messages_node
from MainAgent.nodes.summary_node import summary_node
from MainAgent.nodes.redis_update_node import redis_update_node


async def chat_stream(
    template_id: str,
    user_id: str,
    conv_id: str,
    user_prompt: str,
    if_attachment: bool,
) -> AsyncGenerator[dict, None]:
    """
    Streaming variant of chat().

    Runs the full MainAgent pipeline but yields structured event dicts
    progressively so that the caller can forward them to the client over
    Server-Sent Events (SSE) as they happen.

    Pipeline execution order
    ------------------------
    1. load_template_node    — synchronous setup (Redis/DB)
    2. load_memory_node      — synchronous setup (Redis/DB)
    3. build_context_node    — synchronous setup (assembles message list)
    4. orchestrator_agent_node_stream
                             — streaming: yields tool_call events BEFORE
                               each tool batch executes, then yields token
                               events as the final LLM answer is generated.
    5. persist_messages_node — post-processing (DB writes)
    6. summary_node          — post-processing (optional compression)
    7. redis_update_node     — post-processing (cache refresh)

    Yields (event dicts)
    --------------------
    ``{"type": "tool_call", "tool": <str>, "args": <dict>}``
        Emitted once per tool call, BEFORE the tool is executed.
        Multiple tool calls in a single LLM step are emitted sequentially.

    ``{"type": "token", "content": <str>}``
        Emitted for each text chunk streamed from the LLM during the final
        (non-tool-calling) response.  Accumulate these to reconstruct the
        full response text.

    ``{"type": "done", "conv_id": <str>}``
        Emitted once at the very end, after all post-processing has finished.
        Signals that the stream is complete and the response has been
        persisted to the database.

    Error handling
    --------------
    If an unrecoverable exception occurs during setup or post-processing,
    a ``{"type": "error", "detail": <str>}`` event is yielded and the
    generator exits.  Errors inside the orchestrator are propagated from
    ``orchestrator_agent_node_stream`` (which itself yields an error token
    via the emit callback for LLM-level recoverable errors).

    Parameters
    ----------
    template_id, user_id, conv_id, user_prompt, if_attachment
        Same semantics as chat().
    """
    # ── Build initial state (identical to chat()) ─────────────────────────────
    state: OrchestratorState = {
        "template_id":    template_id,
        "user_id":        user_id,
        "conv_id":        conv_id,
        "user_prompt":    user_prompt,
        "if_attachment":  if_attachment,
        "behavior_prompt":         "",
        "custom_tool_information": "",
        "summary":                     "",
        "recent_messages":             [],
        "unsummarized_token_count":    0,
        "last_summarized_message_seq": 0,
        "messages":       [],
        "final_response": "",
        "tools_called":   [],
        "new_user_seq":       0,
        "new_assistant_seq":  0,
    }

    # ── Phase 1: Setup nodes (non-streaming, sequential) ─────────────────────
    try:
        template_update = await load_template_node(state)
        state.update(template_update)

        memory_update = await load_memory_node(state)
        state.update(memory_update)

        context_update = await build_context_node(state)
        # build_context_node returns {"messages": [...]}, which must be merged
        # (the LangGraph add_messages reducer appends; here we do the same)
        state["messages"] = state["messages"] + context_update.get("messages", [])
    except Exception as setup_exc:
        yield {"type": "error", "detail": f"Pipeline setup error: {setup_exc}"}
        return

    # ── Phase 2: Streaming orchestrator ──────────────────────────────────────
    # Collect emitted events via an async queue so the generator can yield them.
    import asyncio as _asyncio

    event_queue: _asyncio.Queue = _asyncio.Queue()
    _SENTINEL = object()  # marks end of events

    async def _emit(event: dict) -> None:
        await event_queue.put(event)

    async def _run_orchestrator():
        try:
            result = await orchestrator_agent_node_stream(state, _emit)
            await event_queue.put(result)   # final dict goes into queue
        except Exception as orch_exc:
            await event_queue.put({"type": "error", "detail": f"Orchestrator error: {orch_exc}"})
        finally:
            await event_queue.put(_SENTINEL)

    # Launch the orchestrator concurrently; drain its events as they arrive.
    orchestrator_task = _asyncio.ensure_future(_run_orchestrator())

    orchestrator_result: dict = {}
    while True:
        item = await event_queue.get()
        if item is _SENTINEL:
            break
        if isinstance(item, dict):
            if item.get("type") in ("tool_call", "token"):
                # Stream events directly to the caller
                yield item
            elif item.get("type") == "error":
                yield item
                await orchestrator_task
                return
            else:
                # This is the orchestrator result dict (no "type" key)
                orchestrator_result = item

    await orchestrator_task  # ensure any exceptions are propagated

    # ── Merge orchestrator output into state ──────────────────────────────────
    if orchestrator_result:
        # Append new messages (mirrors the add_messages reducer behaviour)
        state["messages"] = state["messages"] + orchestrator_result.get("messages", [])
        state["final_response"] = orchestrator_result.get("final_response", "")
        state["tools_called"]   = orchestrator_result.get("tools_called", [])

    # ── Phase 3: Post-processing nodes (non-streaming) ────────────────────────
    try:
        persist_update = await persist_messages_node(state)
        state.update(persist_update)

        summary_update = await summary_node(state)
        state.update(summary_update)

        await redis_update_node(state)
    except Exception as post_exc:
        # Post-processing failure: the response was already streamed to the
        # client, so we log and still emit done (data is partially persisted).
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "[chat_stream] Post-processing error (response already streamed): %s",
            post_exc,
        )

    yield {"type": "done", "conv_id": conv_id}

