"""
MainAgent/nodes/orchestrator_agent_node.py
-------------------------------------------
Graph node: orchestrator_agent_node

The core reasoning and tool-execution loop.

Responsibilities
----------------
1. Bind the correct set of tools to the LLM (conditionally adding the
   retrieval tool when if_attachment is True).
2. Invoke the LLM iteratively until:
     a. The LLM returns a response with no tool calls  →  final answer.
     b. MAX_TOOL_ITERATIONS is exhausted  →  force a final answer.
3. Execute all tool calls from each LLM response concurrently.
4. Return only the NEW messages produced this turn (the add_messages
   reducer in OrchestratorState handles appending them to state).
5. Write the assistant's final text into state["final_response"].

Tools available
---------------
  Always bound:
    web_search             (builtintools.websearch)
    web_search_focused     (builtintools.websearch)
    fetch_page_content     (builtintools.websearch)
    call_custom_tool_subagent  (SubAgent.CustomToolSubAgent.calling)

  Conditionally bound (if_attachment == True):
    retrieve_from_documents    (builtintools.retrieval)

Config propagation
------------------
The RunnableConfig containing user_id, conv_id, and thread_id is passed
to every tool invocation so that tools requiring identity context
(retrieve_from_documents, call_custom_tool_subagent) receive it correctly.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

try:
    # groq >= 0.9  — BadRequestError is the 400 class we want to catch
    from groq import BadRequestError as _GroqBadRequestError
except ImportError:  # older groq version or not installed
    _GroqBadRequestError = None

from MainAgent.state import OrchestratorState
from utils.tracing import traceable

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # AdapterAI/
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TOOL_ITERATIONS = 10
#_MODEL = "openai/gpt-oss-120b"    # switched from llama-3.3-70b-versatile
_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Tool imports (lazy-style: imported at module level but grouped clearly)
# ---------------------------------------------------------------------------

from builtintools.websearch import TOOLS as _WEB_SEARCH_TOOLS        # noqa: E402
from builtintools.retrieval import TOOLS as _RETRIEVAL_TOOLS          # noqa: E402
from SubAgent.CustomToolSubAgent.calling import TOOLS as _CUSTOM_TOOLS  # noqa: E402


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

_DBG_SEP  = "-" * 60
_DBG_THIN = "." * 60

# Max chars kept from a tool result before truncating (prevents 413 errors)
_MAX_TOOL_RESULT_CHARS = 3000


def _safe_print(*args, **kwargs) -> None:
    """
    Print that survives Windows cp1252 consoles AND flushes immediately
    so output appears in real-time even when stdout is piped/buffered.
    Any character that cannot be encoded in the terminal's codec is
    replaced with '?' so we never crash on Unicode from LLM responses.
    """
    import sys
    enc = sys.stdout.encoding or "utf-8"
    text = " ".join(str(a) for a in args)
    safe = text.encode(enc, errors="replace").decode(enc)
    print(safe, flush=True, **kwargs)


def _extract_text(content) -> str:
    """
    Safely extract a plain-text string from an LLM response's ``content``
    field, which may be either:

    * A plain ``str`` — returned as-is.
    * A ``list`` of content-block dicts (Anthropic/Claude style), e.g.:
        [
            {"type": "text",     "text": "Thinking …"},
            {"type": "tool_use", "id":   "…", …},
        ]
      In this case all ``text`` blocks are joined with newlines and
      returned.  Non-text blocks (tool_use, image, etc.) are ignored.

    Returns an empty string for ``None`` or unexpected types.
    """
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
            if not isinstance(block, dict) or block.get("type") != "tool_use"
        ]
        return "\n".join(p for p in parts if p).strip()
    # Fallback for any other type
    return str(content)

def _dbg_llm_response(iteration: int, response) -> None:
    """Print the raw LLM response for a given iteration."""
    _safe_print(f"\n{_DBG_SEP}")
    _safe_print(f"[LLM] Iteration {iteration} — model response")
    _safe_print(_DBG_THIN)

    if response.content:
        content_str = _extract_text(response.content)
        preview = content_str[:500] + (" ..." if len(content_str) > 500 else "")
        _safe_print(f"  content : {preview}")
    else:
        _safe_print("  content : (empty — model is making tool calls)")

    if response.tool_calls:
        _safe_print(f"  tool calls ({len(response.tool_calls)}):")
        for tc in response.tool_calls:
            args_preview = json.dumps(tc.get("args", {}), ensure_ascii=False)
            if len(args_preview) > 200:
                args_preview = args_preview[:200] + " ..."
            _safe_print(f"    -> {tc['name']}({args_preview})")
    else:
        _safe_print("  tool calls : none")

    _safe_print(_DBG_SEP)


def _dbg_tool_start(tool_name: str, args: dict) -> None:
    """Print before a tool is executed."""
    args_preview = json.dumps(args, ensure_ascii=False)
    if len(args_preview) > 300:
        args_preview = args_preview[:300] + " ..."
    _safe_print(f"\n[TOOL >>>] Calling  : {tool_name}")
    _safe_print(f"           Args     : {args_preview}")


def _dbg_tool_result(tool_name: str, result: str | Exception) -> None:
    """Print the result returned from a tool."""
    if isinstance(result, Exception):
        _safe_print(f"[TOOL <<<] {tool_name} — ERROR: {type(result).__name__}: {result}")
    else:
        total = len(str(result))
        preview = str(result)[:500] + (" ..." if total > 500 else "")
        _safe_print(f"[TOOL <<<] {tool_name} — result ({total} chars):")
        _safe_print(f"           {preview}")


def _dbg_final(response_text: str, reason: str) -> None:
    """Print the final answer and why we stopped."""
    _safe_print(f"\n{_DBG_SEP}")
    _safe_print(f"[FINAL] Stop reason : {reason}")
    _safe_print(_DBG_THIN)
    preview = response_text[:600] + (" ..." if len(response_text) > 600 else "")
    _safe_print(f"  {preview}")
    _safe_print(_DBG_SEP)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tools(if_attachment: bool) -> List[BaseTool]:
    """Assemble the tool list based on whether attachments are present."""
    tools: List[BaseTool] = list(_WEB_SEARCH_TOOLS) + list(_CUSTOM_TOOLS)
    if if_attachment:
        tools += list(_RETRIEVAL_TOOLS)
    return tools


async def _invoke_tool(
    tool_call: Dict,
    tool_map: Dict[str, BaseTool],
    config: RunnableConfig,
) -> str:
    """
    Invoke a single tool and return a string result.

    Uses tool.ainvoke() — LangChain wraps synchronous tool functions in a
    thread executor automatically, so blocking tools (e.g. the custom tool
    subagent) do not block the event loop.

    On any failure a descriptive error string is returned rather than
    raising, so the LLM can reason about the failure and recover.
    """
    tool = tool_map.get(tool_call["name"])
    if tool is None:
        return f"[Error] Tool '{tool_call['name']}' is not available in this context."

    try:
        result = await tool.ainvoke(tool_call["args"], config=config)
        if not isinstance(result, str):
            result = json.dumps(result, indent=2, default=str)
        
        if len(result) > _MAX_TOOL_RESULT_CHARS:
            result = result[:_MAX_TOOL_RESULT_CHARS] + " ... (truncated)"
        return result
    except Exception as exc:  # noqa: BLE001
        return f"[Tool Error] {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@traceable(name="orchestrator_agent_node", tags=["main-agent", "node"])
async def orchestrator_agent_node(state: OrchestratorState) -> dict:
    """
    Core orchestration loop: invoke LLM → execute tools → repeat until done.

    Returns partial state update with:
      - ``messages``       : new LangChain messages produced this turn
                             (LLM responses + tool messages)
      - ``final_response`` : the assistant's final text content
      - ``tools_called``   : ordered list of tool names invoked this turn
    """
    tools    = _build_tools(state["if_attachment"])
    tool_map = {t.name: t for t in tools}

    _safe_print(f"\n{'='*60}")
    _safe_print(f"[AGENT] Starting orchestrator  model={_MODEL}")
    _safe_print(f"        Available tools : {list(tool_map.keys())}")
    _safe_print(f"        User prompt     : {state['user_prompt'][:200]}")
    _safe_print(f"{'='*60}")

    # ── LLM setup ─────────────────────────────────────────────────────────────
    # llm = ChatGroq(
    #     model=_MODEL,
    #     temperature=0.0,
    #     max_tokens=4096,
    #     api_key=os.environ["GROQ_API_KEY"],
    # )
    llm = ChatAnthropic(model_name= _MODEL ,
                        max_tokens=4096 ,
                        api_key=os.environ["ANTHROPIC_API_KEY"])
    
    llm_with_tools = llm.bind_tools(tools)

    # ── Config carries identity for tools that need it ────────────────────────
    runnable_config = RunnableConfig(configurable={
        "user_id":   state["user_id"],
        "conv_id":   state["conv_id"],
        "thread_id": f"{state['user_id']}:{state['conv_id']}",
    })

    # ── Snapshot state message count so we return ONLY new messages ───────────
    initial_count = len(state["messages"])
    messages = list(state["messages"])  # working copy

    final_response = ""
    tools_called: List[str] = []   # track every tool invocation this turn

    # ── Iterative tool-calling loop ───────────────────────────────────────────
    for iteration in range(MAX_TOOL_ITERATIONS):
        _safe_print(f"\n[AGENT] --- Loop iteration {iteration + 1}/{MAX_TOOL_ITERATIONS} ---")

        # Call LLM — only catch genuine Groq 400 BadRequestError (malformed
        # tool-call generation). Rate limits, network errors, etc. propagate.
        try:
            response = await llm_with_tools.ainvoke(messages, config=runnable_config)
        except Exception as llm_exc:  # noqa: BLE001
            # Only swallow genuine "invalid tool call format" 400 errors.
            err_str = str(llm_exc).lower()
            is_bad_request = (
                "400" in err_str
                or "tool_use_failed" in err_str
                or "invalid_request" in err_str
                or (_GroqBadRequestError and isinstance(llm_exc, _GroqBadRequestError))
            )
            if not is_bad_request:
                _safe_print(f"[AGENT] Non-400 error -- re-raising: {type(llm_exc).__name__}: {llm_exc}")
                raise

            # For malformed tool-call 400s: fall back to plain LLM once.
            _safe_print(f"[AGENT] Malformed tool-call 400 -- falling back to plain LLM")
            _safe_print(f"        Error: {llm_exc}")
            error_hint = HumanMessage(content=(
                f"A tool-call formatting error occurred: {llm_exc}. "
                "Please provide a direct, helpful answer based on what you know, "
                "without calling any tools."
            ))
            messages.append(error_hint)
            fallback = await llm.ainvoke(messages, config=runnable_config)
            messages.append(fallback)
            final_response = _extract_text(fallback.content)
            _dbg_final(final_response, "400 fallback to plain LLM")
            break

        # ── Print the LLM response ────────────────────────────────────────────
        _dbg_llm_response(iteration + 1, response)
        messages.append(response)

        # No tool calls → final answer reached
        if not response.tool_calls:
            final_response = _extract_text(response.content)
            _dbg_final(final_response, "no more tool calls — done")
            break

        # ── Execute all tool calls from this response concurrently ────────────
        _safe_print(f"\n[AGENT] Dispatching {len(response.tool_calls)} tool call(s) concurrently...")
        for tc in response.tool_calls:
            _dbg_tool_start(tc["name"], tc.get("args", {}))

        tool_tasks = [
            _invoke_tool(tc, tool_map, runnable_config)
            for tc in response.tool_calls
        ]
        results = await asyncio.gather(*tool_tasks, return_exceptions=True)

        # ── Build ToolMessages, log results, record names ─────────────────────
        tool_messages: List[ToolMessage] = []
        for tc, result in zip(response.tool_calls, results):
            tools_called.append(tc["name"])
            _dbg_tool_result(tc["name"], result)

            if isinstance(result, Exception):
                content = f"[Tool Error] {type(result).__name__}: {result}"
            else:
                content = str(result)

            tool_messages.append(ToolMessage(
                content=content,
                tool_call_id=tc["id"],
                name=tc["name"],
            ))

        messages.extend(tool_messages)

    else:
        # ── Max iterations reached: force a final answer ──────────────────────
        _safe_print(f"\n[AGENT] Max iterations ({MAX_TOOL_ITERATIONS}) reached — forcing final answer")
        messages.append(HumanMessage(content=(
            "You have reached the maximum number of tool call iterations. "
            "Based on all information gathered so far, provide your best "
            "final answer now. Do not make any more tool calls."
        )))
        forced = await llm.ainvoke(messages, config=runnable_config)
        messages.append(forced)
        final_response = _extract_text(forced.content)
        _dbg_final(final_response, f"max iterations ({MAX_TOOL_ITERATIONS}) reached")

    _safe_print(f"\n[AGENT] Turn complete. tools_called={tools_called}")

    # ── Return only messages added this turn ──────────────────────────────────
    new_messages = messages[initial_count:]

    return {
        "messages":       new_messages,
        "final_response": final_response,
        "tools_called":   tools_called,
    }


# ---------------------------------------------------------------------------
# Streaming variant
# ---------------------------------------------------------------------------

from typing import AsyncGenerator, Callable, Awaitable  # noqa: E402 (already imported above)


@traceable(name="orchestrator_agent_node_stream", tags=["main-agent", "node", "streaming"])
async def orchestrator_agent_node_stream(
    state: OrchestratorState,
    emit: Callable[[dict], Awaitable[None]],
) -> AsyncGenerator[None, None]:
    """
    Streaming variant of orchestrator_agent_node.

    Instead of returning all output at the end, this coroutine progressively
    *emits* structured events via the ``emit`` async callback so that a
    caller (e.g. the streaming API endpoint) can forward them to the client
    over SSE as they happen.

    Event types emitted
    -------------------
    ``tool_call``
        Emitted **before** each tool batch is dispatched.
        One event per tool call in the batch.
        Payload: ``{"type": "tool_call", "tool": <name>, "args": <dict>}``

    ``token``
        Emitted for each text chunk produced by the LLM when streaming
        the **final** response (i.e. the turn where the model decides to
        stop calling tools and reply directly).
        Payload: ``{"type": "token", "content": <str>}``

    Parameters
    ----------
    state : OrchestratorState
        The full current pipeline state (must include ``messages``,
        ``user_id``, ``conv_id``, ``if_attachment``).
    emit : async callable
        ``await emit(event_dict)`` is called for every event.

    Returns
    -------
    dict
        Identical shape to ``orchestrator_agent_node``:
        ``{"messages": [...], "final_response": str, "tools_called": [...]}``
    """
    tools    = _build_tools(state["if_attachment"])
    tool_map = {t.name: t for t in tools}

    _safe_print(f"\n{'='*60}")
    _safe_print(f"[AGENT-STREAM] Starting orchestrator  model={_MODEL}")
    _safe_print(f"               Available tools : {list(tool_map.keys())}")
    _safe_print(f"               User prompt     : {state['user_prompt'][:200]}")
    _safe_print(f"{'='*60}")

    # ── LLM setup ─────────────────────────────────────────────────────────────
    llm = ChatAnthropic(
        model_name=_MODEL,
        max_tokens=4096,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    llm_with_tools = llm.bind_tools(tools)

    # ── Config carries identity for tools that need it ────────────────────────
    runnable_config = RunnableConfig(configurable={
        "user_id":   state["user_id"],
        "conv_id":   state["conv_id"],
        "thread_id": f"{state['user_id']}:{state['conv_id']}",
    })

    # ── Snapshot state message count so we return ONLY new messages ───────────
    initial_count = len(state["messages"])
    messages = list(state["messages"])  # working copy

    final_response = ""
    tools_called: List[str] = []

    # ── Iterative tool-calling loop ───────────────────────────────────────────
    for iteration in range(MAX_TOOL_ITERATIONS):
        _safe_print(f"\n[AGENT-STREAM] --- Loop iteration {iteration + 1}/{MAX_TOOL_ITERATIONS} ---")

        # ── LLM call (blocking — we need to know whether tools are requested) ─
        try:
            response = await llm_with_tools.ainvoke(messages, config=runnable_config)
        except Exception as llm_exc:  # noqa: BLE001
            err_str = str(llm_exc).lower()
            is_bad_request = (
                "400" in err_str
                or "tool_use_failed" in err_str
                or "invalid_request" in err_str
                or (_GroqBadRequestError and isinstance(llm_exc, _GroqBadRequestError))
            )
            if not is_bad_request:
                _safe_print(f"[AGENT-STREAM] Non-400 error -- re-raising: {type(llm_exc).__name__}: {llm_exc}")
                raise

            # Fallback: plain LLM with streaming for the recovery response
            _safe_print(f"[AGENT-STREAM] Malformed tool-call 400 -- falling back to plain LLM (streaming)")
            error_hint = HumanMessage(content=(
                f"A tool-call formatting error occurred: {llm_exc}. "
                "Please provide a direct, helpful answer based on what you know, "
                "without calling any tools."
            ))
            messages.append(error_hint)

            # Stream the fallback response
            accumulated = ""
            async for chunk in llm.astream(messages, config=runnable_config):
                chunk_text = _extract_text(chunk.content)
                if chunk_text:
                    accumulated += chunk_text
                    await emit({"type": "token", "content": chunk_text})

            # Build a synthetic AIMessage so the message list stays consistent
            from langchain_core.messages import AIMessage
            fallback_msg = AIMessage(content=accumulated)
            messages.append(fallback_msg)
            final_response = accumulated
            _dbg_final(final_response, "400 fallback to plain LLM (streaming)")
            break

        # ── Log what the LLM returned ─────────────────────────────────────────
        _dbg_llm_response(iteration + 1, response)
        messages.append(response)

        # ── No tool calls → this IS the final answer — emit it as tokens ──────
        if not response.tool_calls:
            _safe_print(f"[AGENT-STREAM] No tool calls — emitting ainvoke response as tokens …")

            # We already have the complete response from ainvoke above.
            # Emit it as token events directly — no second LLM call needed.
            # Split on words to give the frontend a smooth streaming effect
            # while keeping the total number of emit() calls reasonable.
            final_text = _extract_text(response.content)
            final_response = final_text

            # Emit in ~10-char chunks to simulate streaming granularity
            chunk_size = 10
            for i in range(0, len(final_text), chunk_size):
                chunk = final_text[i:i + chunk_size]
                await emit({"type": "token", "content": chunk})

            _dbg_final(final_response, "no more tool calls — emitted from ainvoke")
            break

        # ── Emit tool_call events BEFORE dispatching ──────────────────────────
        _safe_print(f"\n[AGENT-STREAM] Emitting {len(response.tool_calls)} tool_call event(s) …")
        for tc in response.tool_calls:
            _dbg_tool_start(tc["name"], tc.get("args", {}))
            await emit({
                "type": "tool_call",
                "tool": tc["name"],
                "args": tc.get("args", {}),
            })

        # ── Execute all tool calls concurrently ───────────────────────────────
        tool_tasks = [
            _invoke_tool(tc, tool_map, runnable_config)
            for tc in response.tool_calls
        ]
        results = await asyncio.gather(*tool_tasks, return_exceptions=True)

        # ── Build ToolMessages, log results, record names ─────────────────────
        tool_messages: List[ToolMessage] = []
        for tc, result in zip(response.tool_calls, results):
            tools_called.append(tc["name"])
            _dbg_tool_result(tc["name"], result)

            if isinstance(result, Exception):
                content = f"[Tool Error] {type(result).__name__}: {result}"
            else:
                content = str(result)

            tool_messages.append(ToolMessage(
                content=content,
                tool_call_id=tc["id"],
                name=tc["name"],
            ))

        messages.extend(tool_messages)

    else:
        # ── Max iterations reached: force a final answer (streamed) ──────────
        _safe_print(f"\n[AGENT-STREAM] Max iterations ({MAX_TOOL_ITERATIONS}) reached — forcing final answer (streaming)")
        messages.append(HumanMessage(content=(
            "You have reached the maximum number of tool call iterations. "
            "Based on all information gathered so far, provide your best "
            "final answer now. Do not make any more tool calls."
        )))

        accumulated = ""
        async for chunk in llm.astream(messages, config=runnable_config):
            chunk_text = _extract_text(chunk.content)
            if chunk_text:
                accumulated += chunk_text
                await emit({"type": "token", "content": chunk_text})

        from langchain_core.messages import AIMessage
        forced_msg = AIMessage(content=accumulated)
        messages.append(forced_msg)
        final_response = accumulated
        _dbg_final(final_response, f"max iterations ({MAX_TOOL_ITERATIONS}) reached — streamed")

    _safe_print(f"\n[AGENT-STREAM] Turn complete. tools_called={tools_called}")

    # ── Return only messages added this turn ──────────────────────────────────
    new_messages = messages[initial_count:]

    return {
        "messages":       new_messages,
        "final_response": final_response,
        "tools_called":   tools_called,
    }
