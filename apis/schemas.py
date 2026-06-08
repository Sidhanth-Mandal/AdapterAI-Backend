"""
apis/schemas.py
---------------
Pydantic request/response models for all AdapterAI API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

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

class ChatRequest(BaseModel):
    """
    Body for POST /chat/

    conv_id      — existing conversation to continue, or a new UUID to create.
    template_id  — template that drives the agent's behaviour.
    user_prompt  — the user's message text.
    if_attachment — set True when an uploaded file is part of the context.
    """
    conv_id: str
    template_id: str
    user_prompt: str
    if_attachment: bool = False


class ChatResponse(BaseModel):
    """Returned by POST /chat/"""
    conv_id: str
    response: str


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


class TempChatResponse(BaseModel):
    """Returned by POST /tempchat/"""
    template_id: str
    response: str


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
