"""
apis/routers/chat_router.py
----------------------------
POST /chat/
-----------
Sends one user turn to the MainAgent pipeline.

Behaviour:
  • If the conv_id does NOT yet exist in the Conversations table, a new row
    is created before invoking the agent (requires template_id in the body).
  • The agent runs asynchronously (graph.ainvoke is already async).
  • Returns the agent's final response.

JWT-protected: requires a valid Bearer token in the Authorization header.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from apis.auth import get_current_user
from apis.db import (
    conversation_exists,
    create_conversation,
    fetch_user_by_id,
)
from apis.schemas import ChatRequest, ChatResponse

# MainAgent public API
from MainAgent.service import chat as main_agent_chat

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "/",
    response_model=ChatResponse,
    summary="Send a message to the MainAgent",
    description=(
        "Sends one user turn through the full MainAgent LangGraph pipeline. "
        "If the supplied `conv_id` does not yet exist in the database, a new "
        "Conversation row is created automatically before the agent is invoked. "
        "Requires a valid JWT Bearer token."
    ),
)
async def chat_endpoint(
    body: ChatRequest,
    user_id: Annotated[str, Depends(get_current_user)],
) -> ChatResponse:
    # -----------------------------------------------------------------------
    # 1. Verify the authenticated user actually exists
    # -----------------------------------------------------------------------
    user = await asyncio.to_thread(fetch_user_by_id, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Authenticated user not found in database.",
        )

    # -----------------------------------------------------------------------
    # 2. Ensure the conversation row exists
    #    - If conv_id is new → create it with the supplied template_id.
    #    - If it already exists → proceed (the agent will read from DB/Redis).
    # -----------------------------------------------------------------------
    exists = await asyncio.to_thread(conversation_exists, body.conv_id)
    if not exists:
        await asyncio.to_thread(
            create_conversation,
            body.conv_id,
            user_id,
            body.template_id,
            title=None,  # title can be set/updated separately
        )

    # -----------------------------------------------------------------------
    # 3. Invoke the MainAgent pipeline (fully async)
    # -----------------------------------------------------------------------
    try:
        response_text = await main_agent_chat(
            template_id=body.template_id,
            user_id=user_id,
            conv_id=body.conv_id,
            user_prompt=body.user_prompt,
            if_attachment=body.if_attachment,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent pipeline error: {exc}",
        )

    return ChatResponse(conv_id=body.conv_id, response=response_text)
