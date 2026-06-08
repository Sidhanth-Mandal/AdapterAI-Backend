"""
apis/routers/conversation_router.py
-------------------------------------
GET /loadconv/
--------------
Returns all conversations (conv_id + title) that belong to the authenticated
user, ordered by the most recent activity (last_message_at DESC).

No query parameters are required — user identity comes from the JWT token.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends

from apis.auth import get_current_user
from apis.db import fetch_conversations_for_user
from apis.schemas import ConversationListResponse, ConversationRecord

router = APIRouter(prefix="/loadconv", tags=["Conversations"])


@router.get(
    "/",
    response_model=ConversationListResponse,
    summary="Load all conversations for the authenticated user",
    description=(
        "Returns every conversation owned by the authenticated user, "
        "including `conv_id`, `title`, `template_id`, `created_at`, and "
        "`last_message_at`. Results are ordered by most-recent activity. "
        "Requires a valid JWT Bearer token."
    ),
)
async def load_conversations(
    user_id: Annotated[str, Depends(get_current_user)],
) -> ConversationListResponse:
    rows = await asyncio.to_thread(fetch_conversations_for_user, user_id)
    conversations = [ConversationRecord(**r) for r in rows]
    return ConversationListResponse(user_id=user_id, conversations=conversations)
