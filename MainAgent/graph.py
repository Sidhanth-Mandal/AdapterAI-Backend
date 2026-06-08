"""
MainAgent/graph.py
-------------------
Builds and compiles the MainAgent LangGraph StateGraph.

Pipeline topology (linear — no conditional edges)
-------------------------------------------------

  START
    │
    ▼
  load_template_node     ← Redis-first template config load
    │
    ▼
  load_memory_node       ← Redis-first conversation memory load
    │
    ▼
  build_context_node     ← Assemble LangChain message list
    │
    ▼
  orchestrator_agent_node ← ReAct tool-calling loop (up to 10 iterations)
    │
    ▼
  persist_messages_node  ← Write user + assistant messages to PostgreSQL
    │
    ▼
  summary_node           ← Conditionally compress conversation history
    │
    ▼
  redis_update_node      ← Refresh conversation cache in Redis
    │
    ▼
  END

Notes
-----
* No LangGraph checkpointer is used.  All state is managed explicitly
  through Redis and PostgreSQL, consistent with the TemplateCreation module.
* The graph is compiled fresh on each chat() call — there is no persistent
  graph object between invocations.
* All nodes are async; the graph must be run via graph.ainvoke().
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from MainAgent.nodes.build_context_node import build_context_node
from MainAgent.nodes.load_memory_node import load_memory_node
from MainAgent.nodes.load_template_node import load_template_node
from MainAgent.nodes.orchestrator_agent_node import orchestrator_agent_node
from MainAgent.nodes.persist_messages_node import persist_messages_node
from MainAgent.nodes.redis_update_node import redis_update_node
from MainAgent.nodes.summary_node import summary_node
from MainAgent.state import OrchestratorState


def build_graph():
    """
    Construct and compile the full MainAgent LangGraph pipeline.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready to invoke with ``await graph.ainvoke()``.
    """
    builder = StateGraph(OrchestratorState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("load_template_node",      load_template_node)
    builder.add_node("load_memory_node",        load_memory_node)
    builder.add_node("build_context_node",      build_context_node)
    builder.add_node("orchestrator_agent_node", orchestrator_agent_node)
    builder.add_node("persist_messages_node",   persist_messages_node)
    builder.add_node("summary_node",            summary_node)
    builder.add_node("redis_update_node",       redis_update_node)

    # ── Wire edges (linear pipeline) ─────────────────────────────────────────
    builder.add_edge(START,                      "load_template_node")
    builder.add_edge("load_template_node",       "load_memory_node")
    builder.add_edge("load_memory_node",         "build_context_node")
    builder.add_edge("build_context_node",       "orchestrator_agent_node")
    builder.add_edge("orchestrator_agent_node",  "persist_messages_node")
    builder.add_edge("persist_messages_node",    "summary_node")
    builder.add_edge("summary_node",             "redis_update_node")
    builder.add_edge("redis_update_node",        END)

    return builder.compile()
