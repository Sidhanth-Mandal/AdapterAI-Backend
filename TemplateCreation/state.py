"""
state.py
--------
Defines the shared TypedDict state that flows through the entire LangGraph
pipeline for the Requirements Gathering Chatbot.
"""

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """
    Shared state for the Requirements Chatbot LangGraph workflow.

    Fields
    ------
    messages : list
        Full conversation history. Uses LangGraph's `add_messages` reducer
        so that each node can append new messages without overwriting history.
    phase : str
        Current phase of the workflow.
        - "gathering"  → Phase 1 (requirements chatbot is active)
        - "planning"   → Phase 2 (planner node is running)
        - "done"       → Workflow complete
    satisfied : bool
        Set to True by the chatbot node when it has gathered enough information
        to confidently design tools and a system prompt.
    requirements : dict
        Structured dictionary of extracted requirements populated after the
        chatbot declares satisfaction. Used as additional context for the
        planner node.
    tool_creation_prompt : str
        The detailed tool-generation instruction prompt produced by the planner.
        Empty string until Phase 2 completes.
    behavior_prompt : str
        The final assistant behavior/system prompt produced by the planner.
        Empty string until Phase 2 completes.
    """

    messages: Annotated[list, add_messages]
    phase: str
    satisfied: bool
    requirements: dict
    tool_creation_prompt: str
    behavior_prompt: str
