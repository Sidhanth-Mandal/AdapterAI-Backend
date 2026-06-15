"""
graph.py
--------
Defines and compiles the LangGraph StateGraph for the Requirements Chatbot
+ Planner pipeline.

Graph structure
---------------

    START
      │
      ▼
  [chatbot_node]  ◄─────────────────────┐
      │                                 │
      ▼ (route_after_chatbot)           │
  satisfied? ── No (gather) ────────────┘
      │
     Yes (plan)
      │
      ▼
  [planner_node]
      │
      ▼
     END
"""

from langgraph.graph import StateGraph, START, END

from .state import GraphState
from .nodes.chatbot_node import chatbot_node
from .nodes.planner_node import planner_node


# ---------------------------------------------------------------------------
# Routing function — conditional edge after chatbot_node
# ---------------------------------------------------------------------------
def route_after_chatbot(state: GraphState) -> str:
    """
    Determine the next node after the chatbot responds.

    Returns
    -------
    str
        "plan"   → route to planner_node  (requirements satisfied)
        "gather" → route back to chatbot_node entry (waiting for next user input)
                   In practice the graph will pause at the user_input interrupt.
    """
    if state.get("satisfied", False):
        print("[TC:graph]   route_after_chatbot → 'plan' (satisfied=True, routing to planner_node)")
        return "plan"
    print("[TC:graph]   route_after_chatbot → 'gather' (satisfied=False, waiting for next user input)")
    return "gather"


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------
def build_graph():
    """
    Construct and compile the full LangGraph pipeline.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready to invoke.
    """
    builder = StateGraph(GraphState)

    # --- Register nodes ---
    builder.add_node("chatbot_node", chatbot_node)
    builder.add_node("planner_node", planner_node)

    # --- Entry edge ---
    builder.add_edge(START, "chatbot_node")

    # --- Conditional routing after chatbot ---
    builder.add_conditional_edges(
        "chatbot_node",
        route_after_chatbot,
        {
            "plan":   "planner_node",
            "gather": END,           # Graph pauses here; main.py re-invokes with next user message
        },
    )

    # --- Planner always terminates ---
    builder.add_edge("planner_node", END)

    compiled = builder.compile()
    print("[TC:graph]   build_graph() compiled successfully")
    return compiled
