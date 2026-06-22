"""
apis/routers/chat_router.py
----------------------------
POST /chat/
-----------
Sends one user turn to the MainAgent pipeline.

Behaviour:
  • Accepts a JSON body (ChatRequest).
  • If the request body contains an ``attachment`` object
    (file_name, mime_type, file_content_b64), the endpoint will:
      1. Decode the base64 file content.
      2. Upload the raw bytes to Cloudflare R2 via r2_client.upload_file_to_r2().
         The returned ``attachment_id`` (UUID) becomes the R2 object-key prefix
         AND the Attachments table PK.
      3. Insert a placeholder message row so the FK constraint on
         Attachments.message_id is satisfied, then insert the Attachments row.
      4. Force ``if_attachment = True`` so the agent knows files are present.
  • If the conv_id does NOT yet exist in the Conversations table, a new row
    is created before invoking the agent (requires template_id in the body).
  • The agent runs asynchronously (graph.ainvoke is already async).
  • Returns the agent's final response plus the attachment_id (if any).

JWT-protected: requires a valid Bearer token in the Authorization header.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from typing import Annotated, AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from apis.auth import get_current_user
from apis.db import (
    conversation_exists,
    create_attachment,
    create_conversation,
    fetch_user_by_id,
    update_conversation_title,
)
from apis.title_generator import generate_title
from apis.supabase_storage import upload_file_to_supabase
from apis.schemas import ChatRequest, ChatResponse

# MainAgent public API
from MainAgent.service import chat as main_agent_chat
from MainAgent.service import chat_stream as main_agent_chat_stream

# RAG ingest pipeline
from vector_store import ingest_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "/",
    response_model=ChatResponse,
    summary="Send a message (optionally with a file attachment) to the MainAgent",
    description=(
        "Sends one user turn through the full MainAgent LangGraph pipeline. "
        "If the supplied ``conv_id`` does not yet exist in the database, a new "
        "Conversation row is created automatically before the agent is invoked. "
        "An optional ``attachment`` object (file_name, mime_type, file_content_b64) "
        "can be included; the file is uploaded to Cloudflare R2 and recorded in the "
        "Attachments table before the agent is invoked. "
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
    # -----------------------------------------------------------------------
    exists = await asyncio.to_thread(conversation_exists, body.conv_id)
    is_new = not exists  # True only for the very first turn — used for title gen
    if not exists:
        await asyncio.to_thread(
            create_conversation,
            body.conv_id,
            user_id,
            body.template_id,
            title=None,
        )

    # -----------------------------------------------------------------------
    # 3. Handle optional file attachment
    #    a) Decode base64 content
    #    b) Upload to Supabase Storage  →  attachment_id, storage_url
    #    c) Create a stub message row (required by the FK on attachments)
    #    d) Insert into the Attachments table
    #    e) Run RAG ingest pipeline (text extraction → chunking → Pinecone)
    # -----------------------------------------------------------------------
    attachment_id: Optional[str] = None
    # Track locally so we never depend on body mutation order
    has_attachment = body.attachment is not None

    if has_attachment:
        att = body.attachment

        # ── Set if_attachment=True IMMEDIATELY so it is never skipped by a
        #    later exception.  We force it here rather than at the end of the
        #    block so every code-path that exits this block sees True.
        body = body.model_copy(update={"if_attachment": True})
        logger.info(
            "[chat_router] Attachment detected: '%s' (%s). "
            "if_attachment forced to True before DB/ingest steps.",
            att.file_name, att.mime_type,
        )

        # 3a. Decode base64
        try:
            file_bytes = base64.b64decode(att.file_content_b64)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid base64 encoding in attachment.file_content_b64: {exc}",
            )

        # 3b. Upload to Supabase Storage
        try:
            attachment_id, storage_url = await asyncio.to_thread(
                upload_file_to_supabase,
                file_bytes,
                att.file_name,
                att.mime_type,
            )
            logger.info(
                "[chat_router] Supabase upload OK — attachment_id=%s", attachment_id
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to upload file to Supabase Storage: {exc}",
            )

        # 3c. We need a message_id to satisfy the FK.
        #     We create a synthetic "user" message row representing this upload.
        message_id = str(uuid.uuid4())
        try:
            await asyncio.to_thread(
                _insert_message_row,
                message_id,
                body.conv_id,
                att.file_name,
            )
        except Exception as exc:
            logger.warning(
                "[chat_router] _insert_message_row failed for '%s': %s — continuing.",
                att.file_name, exc,
            )

        # 3d. Insert into Attachments table
        try:
            await asyncio.to_thread(
                create_attachment,
                attachment_id,
                message_id,
                att.file_name,
                att.mime_type,
                storage_url,
            )
            logger.info("[chat_router] Attachments DB row inserted OK.")
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to record attachment in database: {exc}",
            )

        # 3e. Run the RAG ingest pipeline (text extraction → chunking → Pinecone).
        #     Errors here are non-fatal: the file is already safely stored in
        #     Supabase, so we log a warning and let the chat continue.
        #     if_attachment is already True (set above) regardless of ingest outcome.
        try:
            logger.info(
                "[chat_router] Starting RAG ingest for '%s' (mime=%s) …",
                att.file_name, att.mime_type,
            )
            ingest_result = await asyncio.to_thread(
                ingest_file,
                file_bytes,
                att.file_name,
                user_id,
                body.conv_id,
            )
            logger.info(
                "[chat_router] RAG ingest complete for '%s': "
                "%d chunks → %d vectors upserted.",
                att.file_name,
                ingest_result["num_chunks"],
                len(ingest_result["vector_ids"]),
            )
        except Exception as rag_exc:
            logger.warning(
                "[chat_router] RAG ingest FAILED for '%s' "
                "(file still stored in Supabase; if_attachment stays True): %s",
                att.file_name,
                rag_exc,
            )

    # -----------------------------------------------------------------------
    # 4. Invoke the MainAgent pipeline (fully async)
    # -----------------------------------------------------------------------
    logger.info(
        "[chat_router] Invoking main_agent_chat — "
        "conv_id=%s  if_attachment=%s  has_attachment=%s",
        body.conv_id, body.if_attachment, has_attachment,
    )
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

    # -----------------------------------------------------------------------
    # 5. Generate and persist a title for brand-new conversations
    # -----------------------------------------------------------------------
    if is_new:
        try:
            title = await asyncio.to_thread(generate_title, body.user_prompt)
            await asyncio.to_thread(update_conversation_title, body.conv_id, title)
            logger.info(
                "[chat_router] Title set for new conv_id=%s: %r", body.conv_id, title
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[chat_router] Title generation failed for conv_id=%s — title stays null.",
                body.conv_id,
            )

    return ChatResponse(
        conv_id=body.conv_id,
        response=response_text,
        attachment_id=attachment_id,
    )


# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/stream",
    summary="Stream a message response with live tool-call events (SSE)",
    description=(
        "Sends one user turn through the full MainAgent pipeline and streams the "
        "response as Server-Sent Events (SSE).  "
        "Before the final answer begins streaming, one ``tool_call`` event is emitted "
        "for every tool the agent decides to invoke (with tool name and arguments). "
        "The final LLM response is then streamed token-by-token as ``token`` events. "
        "A ``done`` event is emitted at the very end once the response has been "
        "persisted to the database.  "
        "Same attachment handling as POST /chat/.  "
        "Requires a valid JWT Bearer token.  "
        "Response Content-Type: text/event-stream."
    ),
    response_class=StreamingResponse,
)
async def chat_stream_endpoint(
    body: ChatRequest,
    user_id: Annotated[str, Depends(get_current_user)],
) -> StreamingResponse:
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
    # -----------------------------------------------------------------------
    exists = await asyncio.to_thread(conversation_exists, body.conv_id)
    is_new = not exists  # True only for the very first turn — used for title gen
    if not exists:
        await asyncio.to_thread(
            create_conversation,
            body.conv_id,
            user_id,
            body.template_id,
            title=None,
        )

    # -----------------------------------------------------------------------
    # 3. Handle optional file attachment (identical to chat_endpoint)
    # -----------------------------------------------------------------------
    attachment_id: Optional[str] = None
    has_attachment = body.attachment is not None

    if has_attachment:
        att = body.attachment
        body = body.model_copy(update={"if_attachment": True})
        logger.info(
            "[chat_stream_router] Attachment detected: '%s' (%s). "
            "if_attachment forced to True before DB/ingest steps.",
            att.file_name, att.mime_type,
        )

        try:
            file_bytes = base64.b64decode(att.file_content_b64)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid base64 encoding in attachment.file_content_b64: {exc}",
            )

        try:
            attachment_id, storage_url = await asyncio.to_thread(
                upload_file_to_supabase,
                file_bytes,
                att.file_name,
                att.mime_type,
            )
            logger.info(
                "[chat_stream_router] Supabase upload OK — attachment_id=%s", attachment_id
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to upload file to Supabase Storage: {exc}",
            )

        message_id = str(uuid.uuid4())
        try:
            await asyncio.to_thread(
                _insert_message_row,
                message_id,
                body.conv_id,
                att.file_name,
            )
        except Exception as exc:
            logger.warning(
                "[chat_stream_router] _insert_message_row failed for '%s': %s — continuing.",
                att.file_name, exc,
            )

        try:
            await asyncio.to_thread(
                create_attachment,
                attachment_id,
                message_id,
                att.file_name,
                att.mime_type,
                storage_url,
            )
            logger.info("[chat_stream_router] Attachments DB row inserted OK.")
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to record attachment in database: {exc}",
            )

        try:
            logger.info(
                "[chat_stream_router] Starting RAG ingest for '%s' (mime=%s) …",
                att.file_name, att.mime_type,
            )
            ingest_result = await asyncio.to_thread(
                ingest_file,
                file_bytes,
                att.file_name,
                user_id,
                body.conv_id,
            )
            logger.info(
                "[chat_stream_router] RAG ingest complete for '%s': "
                "%d chunks → %d vectors upserted.",
                att.file_name,
                ingest_result["num_chunks"],
                len(ingest_result["vector_ids"]),
            )
        except Exception as rag_exc:
            logger.warning(
                "[chat_stream_router] RAG ingest FAILED for '%s' "
                "(file still stored in Supabase; if_attachment stays True): %s",
                att.file_name,
                rag_exc,
            )

    # -----------------------------------------------------------------------
    # 4. Build and return a StreamingResponse (SSE)
    # -----------------------------------------------------------------------
    logger.info(
        "[chat_stream_router] Starting SSE stream — "
        "conv_id=%s  if_attachment=%s  has_attachment=%s",
        body.conv_id, body.if_attachment, has_attachment,
    )

    # Capture these for the closure below (body may not be picklable)
    _template_id    = body.template_id
    _conv_id        = body.conv_id
    _user_prompt    = body.user_prompt
    _if_attachment  = body.if_attachment
    _attachment_id  = attachment_id
    _is_new         = is_new  # carry new-conversation flag into the closure

    async def _event_generator() -> AsyncGenerator[str, None]:
        """
        Wraps main_agent_chat_stream and serialises each event dict as an
        SSE ``data:`` line.  The ``done`` event is enriched with the
        attachment_id before forwarding so the client knows the file ID.
        For brand-new conversations, a title is generated and persisted
        to the DB once the ``done`` event is received.
        """
        try:
            async for event in main_agent_chat_stream(
                template_id=_template_id,
                user_id=user_id,
                conv_id=_conv_id,
                user_prompt=_user_prompt,
                if_attachment=_if_attachment,
            ):
                # Enrich the done event with attachment_id
                if event.get("type") == "done" and _attachment_id is not None:
                    event = {**event, "attachment_id": _attachment_id}

                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                # Generate and persist title once the stream is fully done
                if event.get("type") == "done" and _is_new:
                    try:
                        title = await asyncio.to_thread(generate_title, _user_prompt)
                        await asyncio.to_thread(
                            update_conversation_title, _conv_id, title
                        )
                        logger.info(
                            "[chat_stream_router] Title set for new conv_id=%s: %r",
                            _conv_id, title,
                        )
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "[chat_stream_router] Title generation failed for "
                            "conv_id=%s — title stays null.",
                            _conv_id,
                        )
        except Exception as exc:
            error_event = {"type": "error", "detail": f"Stream error: {exc}"}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _insert_message_row(message_id: str, conv_id: str, file_name: str) -> None:
    """
    Insert a lightweight 'user' message row that acts as the parent for an
    Attachments FK reference.  The content is a short description of the
    uploaded file so the chat history stays readable.
    """
    from apis.db import _get_connection  # avoid top-level circular import

    sql = """
        INSERT INTO messages (message_id, conv_id, role, content, sequence_number)
        VALUES (
            %s, %s, 'user',
            %s,
            COALESCE(
                (SELECT MAX(sequence_number) FROM messages WHERE conv_id = %s),
                0
            ) + 1
        )
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (message_id, conv_id, f"[Attachment uploaded: {file_name}]", conv_id),
            )
