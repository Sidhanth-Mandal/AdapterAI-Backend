"""
apis/routers/history_router.py
-------------------------------
GET /loadchathistory/
---------------------
Returns the full message history for a given conversation.

Query parameter:
  conv_id (required) — the conversation whose messages to fetch.

JWT-protected: we verify that the conversation belongs to the authenticated
user before returning any data.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from apis.auth import get_current_user
from apis.db import fetch_messages_for_conversation, _get_connection
from apis.schemas import ChatHistoryResponse, MessageRecord

router = APIRouter(prefix="/loadchathistory", tags=["History"])


def _fetch_conversation_owner(conv_id: str) -> Optional[str]:
    """Return the user_id that owns ``conv_id``, or None if not found."""
    sql = "SELECT user_id FROM conversations WHERE conv_id = %s LIMIT 1"
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (conv_id,))
            row = cur.fetchone()
    return row[0] if row else None


@router.get(
    "/",
    response_model=ChatHistoryResponse,
    summary="Load full chat history for a conversation",
    description=(
        "Returns every message in the conversation identified by `conv_id`, "
        "ordered chronologically (oldest first). "
        "The conversation must belong to the authenticated user. "
        "Requires a valid JWT Bearer token."
    ),
)
async def load_chat_history(
    conv_id: str = Query(..., description="The conversation ID to load history for"),
    user_id: Annotated[str, Depends(get_current_user)] = None,
) -> ChatHistoryResponse:
    # -----------------------------------------------------------------------
    # 1. Verify the conversation exists and belongs to the authenticated user
    # -----------------------------------------------------------------------
    owner = await asyncio.to_thread(_fetch_conversation_owner, conv_id)
    if owner is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conv_id}' not found.",
        )
    if owner != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this conversation.",
        )

    # -----------------------------------------------------------------------
    # 2. Fetch and return all messages
    # -----------------------------------------------------------------------
    rows = await asyncio.to_thread(fetch_messages_for_conversation, conv_id)
    messages = [MessageRecord(**r) for r in rows]
    return ChatHistoryResponse(conv_id=conv_id, messages=messages)
