"""
apis/routers/conversation_router.py
-------------------------------------
GET  /loadconv/
    Returns all conversations (conv_id + title) that belong to the
    authenticated user, ordered by the most recent activity (last_message_at DESC).

DELETE /loadconv/{conv_id}
    Fully deletes a conversation:
      1. Removes all Pinecone vectors for (user_id, conv_id).
      2. Removes all Supabase bucket files linked to the conversation's attachments.
      3. Deletes DB rows in FK order: attachments → messages → conversations.

No query parameters are required — user identity comes from the JWT token.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from apis.auth import get_current_user
from apis.db import (
    delete_conversation,
    fetch_attachments_for_conversation,
    fetch_conversations_for_user,
)
from apis.schemas import (
    ConversationListResponse,
    ConversationRecord,
    DeleteConversationResponse,
)
from apis.supabase_storage import delete_files_from_supabase
from vector_store.pinecone_client import delete_by_conversation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/loadconv", tags=["Conversations"])


# ---------------------------------------------------------------------------
# GET /loadconv/  — list all conversations for the current user
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DELETE /loadconv/{conv_id}  — fully delete a conversation
# ---------------------------------------------------------------------------

@router.delete(
    "/{conv_id}",
    response_model=DeleteConversationResponse,
    summary="Delete a conversation and all associated data",
    description=(
        "Permanently deletes a conversation and every piece of data tied to it:\n\n"
        "1. **Pinecone** — all vectors whose metadata matches `(user_id, conv_id)` "
        "are removed from the vector store.\n"
        "2. **Supabase Storage** — every file attached to any message in this "
        "conversation is deleted from the bucket.\n"
        "3. **Database** — attachment rows, message rows, and the conversation row "
        "itself are deleted in FK-safe order.\n\n"
        "Returns `404` if the conversation does not exist or belongs to another user."
    ),
)
async def delete_conversation_endpoint(
    conv_id: str,
    user_id: Annotated[str, Depends(get_current_user)],
) -> DeleteConversationResponse:
    # ── 1. Fetch attachment URLs before deleting anything ──────────────────
    attachments: list[dict] = await asyncio.to_thread(
        fetch_attachments_for_conversation, conv_id
    )
    storage_urls: list[str] = [a["storage_url"] for a in attachments]

    # ── 2. Remove Pinecone vectors ─────────────────────────────────────────
    vector_store_cleaned = False
    try:
        await asyncio.to_thread(delete_by_conversation, user_id, conv_id)
        vector_store_cleaned = True
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to delete Pinecone vectors for conv_id=%s user_id=%s",
            conv_id, user_id,
        )

    # ── 3. Remove Supabase Storage files ──────────────────────────────────
    supabase_result: dict = {"deleted": [], "errors": []}
    if storage_urls:
        try:
            supabase_result = await asyncio.to_thread(
                delete_files_from_supabase, storage_urls
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to delete Supabase files for conv_id=%s", conv_id
            )

    # ── 4. Delete DB rows (attachments → messages → conversation) ──────────
    deleted: bool = await asyncio.to_thread(delete_conversation, conv_id, user_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conv_id}' not found or access denied.",
        )

    supabase_errors = [str(e) for e in supabase_result.get("errors", [])]

    return DeleteConversationResponse(
        conv_id=conv_id,
        deleted=True,
        vector_store_cleaned=vector_store_cleaned,
        supabase_files_deleted=len(supabase_result.get("deleted", [])),
        supabase_errors=supabase_errors,
    )
