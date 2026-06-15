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
# Internal imports (env vars must be loaded first)
# ---------------------------------------------------------------------------

from MainAgent.graph import build_graph
from MainAgent.state import OrchestratorState


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
