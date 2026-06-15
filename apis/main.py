"""
apis/main.py
------------
AdapterAI FastAPI application entry point.

Mounts all routers and configures global middleware.

Endpoints
---------
  POST   /auth/token               — OAuth2 password-flow login
  POST   /auth/login               — JSON-body login

  POST   /chat/                    — MainAgent chat (auto-creates conversation if new)
  POST   /tempchat/                — TemplateCreation chat

  GET    /loadchathistory/         — Full message history for a conv_id
  GET    /loadtemplate/            — All templates for the authenticated user
  DELETE /loadtemplate/{id}        — Delete a template (DB only)
  GET    /loadconv/                — All conv_id + title for the authenticated user
  DELETE /loadconv/{conv_id}       — Delete a conversation (DB + Pinecone + Supabase)

Running
-------
From the project root (AdapterAI/):

    uvicorn apis.main:app --reload --host 0.0.0.0 --port 8002

Interactive docs:
    http://localhost:8002/docs   (Swagger UI)
    http://localhost:8002/redoc  (ReDoc)
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Ensure the project root (AdapterAI/) is on sys.path so that cross-module
# imports like `from MainAgent.service import chat` work when this file
# is launched via `uvicorn apis.main:app`.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Router imports
# ---------------------------------------------------------------------------
from apis.routers.auth_router import router as auth_router
from apis.routers.chat_router import router as chat_router
from apis.routers.tempchat_router import router as tempchat_router
from apis.routers.history_router import router as history_router
from apis.routers.template_router import router as template_router
from apis.routers.conversation_router import router as conversation_router

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AdapterAI API",
    description=(
        "REST API for AdapterAI — a multi-agent AI platform.\n\n"
        "All protected endpoints require a Bearer JWT obtained from "
        "`POST /auth/token` or `POST /auth/login`."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow all origins in development; tighten in production
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Mount routers
# ---------------------------------------------------------------------------
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(tempchat_router)
app.include_router(history_router)
app.include_router(template_router)
app.include_router(conversation_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"], summary="Health check")
async def root() -> dict:
    return {"status": "ok", "service": "AdapterAI API"}
