"""
MainAgent/__init__.py
---------------------
Public API for the MainAgent orchestration module.

Usage
-----
    from MainAgent import chat

    response = await chat(
        template_id="tmpl-abc123",
        user_id="user-xyz",
        conv_id="conv-def456",
        user_prompt="Hello, how can you help me?",
        if_attachment=False,
    )
"""

from MainAgent.service import chat

__all__ = ["chat"]
