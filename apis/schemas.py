"""
apis/schemas.py
---------------
Pydantic request/response models for all AdapterAI API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------------------
# Auth schemas
# ---------------------------------------------------------------------------

class TokenResponse(BaseModel):
    """Returned by POST /auth/token on successful login."""
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    """Body for POST /auth/login (JSON-based alternative to OAuth2 form)."""
    username: str
    password: str = Field(..., max_length=72)


class SignupRequest(BaseModel):
    """Body for POST /auth/signup."""
    username: str
    email: EmailStr
    password: str = Field(..., max_length=72)


# ---------------------------------------------------------------------------
# Chat schemas
# ---------------------------------------------------------------------------

class AttachmentInfo(BaseModel):
    """
    Optional file attachment carried with a chat message.

    Clients that send a file must include all three fields.
    The attachment is uploaded to Cloudflare R2 before the agent
    is invoked; the resulting ``attachment_id`` (a UUID) is used as
    the R2 object-key prefix AND the Attachments table primary key.
    """
    file_name: str = Field(..., description="Original file name, e.g. 'report.pdf'")
    mime_type: str = Field(..., description="MIME type, e.g. 'application/pdf'")
    # Base64-encoded file bytes supplied by the client
    file_content_b64: str = Field(..., description="Base64-encoded file content")


class ChatRequest(BaseModel):
    """
    Body for POST /chat/

    conv_id      — existing conversation to continue, or a new UUID to create.
    template_id  — template that drives the agent's behaviour.
    user_prompt  — the user's message text.
    attachment   — optional file attachment (file_name, mime_type, file_content_b64).
    if_attachment — auto-set to True by the endpoint when an attachment is present;
                    clients can also set it explicitly.
    """
    conv_id: str
    template_id: str
    user_prompt: str
    attachment: Optional[AttachmentInfo] = None
    if_attachment: bool = False


class ChatResponse(BaseModel):
    """Returned by POST /chat/"""
    conv_id: str
    response: str
    attachment_id: Optional[str] = None  # set when a file was uploaded


# ---------------------------------------------------------------------------
# Template chat schemas
# ---------------------------------------------------------------------------

class TempChatRequest(BaseModel):
    """
    Body for POST /tempchat/

    template_id  — the template being built (acts as session key).
    user_prompt  — the user's message text.
    """
    template_id: str
    user_prompt: str


class MCQOption(BaseModel):
    """
    A single selectable option within a MCQ question.

    label  — one of "a", "b", "c", "d", or "custom".
             "custom" is the free-text write-in slot (text will always be
             an empty string from the server; the frontend fills it at runtime).
    text   — the display text for the option.  Empty string for "custom".
    """
    label: str = Field(..., description="'a'|'b'|'c'|'d'|'custom'")
    text: str  = Field(..., description="Display text; empty for 'custom' (write-in) option")


class MCQQuestion(BaseModel):
    """
    A single MCQ question block with four labelled options plus a free-text slot.
    """
    question: str             = Field(..., description="The question text (markdown bold stripped)")
    options:  List[MCQOption] = Field(..., description="Always 5 items: a, b, c, d, custom")


class TempChatResponse(BaseModel):
    """
    Returned by POST /tempchat/

    Fields
    ------
    template_id : str
        Echoes back the template ID from the request.
    preamble : str
        The acknowledgement / summary text that precedes the questions.
        May be empty on the very first turn or when the agent goes straight
        to questions.  Render this as plain prose above the MCQ cards.
    questions : list[MCQQuestion]
        Structured question blocks (0–4 per turn).  Each block contains the
        question text and five options (a, b, c, d, custom).  An empty list
        means the agent produced a closing/informational message with no
        questions (e.g. the final satisfaction message).
    response : str
        The full raw assistant response text (preamble + question blocks
        concatenated).  Kept for backward-compatibility and for fallback
        rendering if the frontend cannot parse ``questions``.
    """
    template_id: str
    preamble:    str                = Field("", description="Prose before the MCQ questions")
    questions:   List[MCQQuestion]  = Field(default_factory=list, description="Parsed MCQ blocks")
    response:    str                = Field(..., description="Full raw assistant response (fallback)")


# ---------------------------------------------------------------------------
# Chat history schemas
# ---------------------------------------------------------------------------

class MessageRecord(BaseModel):
    """A single message row from the Messages table."""
    message_id: str
    conv_id: str
    role: str
    content: str
    token_count: Optional[int] = None
    sequence_number: Optional[int] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ChatHistoryResponse(BaseModel):
    """Returned by GET /loadchathistory/"""
    conv_id: str
    messages: List[MessageRecord]


# ---------------------------------------------------------------------------
# Template list schemas
# ---------------------------------------------------------------------------

class TemplateRecord(BaseModel):
    """A single row from the Templates table."""
    template_id: str
    name: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TemplateListResponse(BaseModel):
    """Returned by GET /loadtemplate/"""
    user_id: str
    templates: List[TemplateRecord]


# ---------------------------------------------------------------------------
# Conversation list schemas
# ---------------------------------------------------------------------------

class ConversationRecord(BaseModel):
    """A single row from the Conversations table."""
    conv_id: str
    title: Optional[str] = None
    template_id: str
    created_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ConversationListResponse(BaseModel):
    """Returned by GET /loadconv/"""
    user_id: str
    conversations: List[ConversationRecord]


# ---------------------------------------------------------------------------
# Delete response schemas
# ---------------------------------------------------------------------------

class DeleteConversationResponse(BaseModel):
    """Returned by DELETE /loadconv/{conv_id}"""
    conv_id: str
    deleted: bool
    vector_store_cleaned: bool
    supabase_files_deleted: int
    supabase_errors: List[str] = []


class DeleteTemplateResponse(BaseModel):
    """Returned by DELETE /loadtemplate/{template_id}"""
    template_id: str
    deleted: bool
