"""
SubAgent/CustomToolSubAgent — Groq-powered tool-calling subagent.

The subagent resolves its tool schema from PostgreSQL using conv_id:
  conv_id → Conversations.template_id → Tools.tool_json

Public exports
--------------
    run_custom_tool_subagent   : Core agentic loop (agent.py)
    call_custom_tool_subagent  : LangChain @tool wrapper (calling.py)
    call_tool_subagent_direct  : Direct Python wrapper (calling.py)
    TOOLS                      : List[BaseTool] for bind_tools() (calling.py)
"""

from SubAgent.CustomToolSubAgent.agent import run_custom_tool_subagent
from SubAgent.CustomToolSubAgent.calling import (
    TOOLS,
    call_custom_tool_subagent,
    call_tool_subagent_direct,
)

__all__ = [
    "run_custom_tool_subagent",
    "call_custom_tool_subagent",
    "call_tool_subagent_direct",
    "TOOLS",
]
