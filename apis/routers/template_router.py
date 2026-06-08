"""
apis/routers/template_router.py
--------------------------------
GET /loadtemplate/
------------------
Returns all templates that belong to the authenticated user.

No query parameters are required — user identity comes from the JWT token.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends

from apis.auth import get_current_user
from apis.db import fetch_templates_for_user
from apis.schemas import TemplateListResponse, TemplateRecord

router = APIRouter(prefix="/loadtemplate", tags=["Templates"])


@router.get(
    "/",
    response_model=TemplateListResponse,
    summary="Load all templates for the authenticated user",
    description=(
        "Returns every template created by the authenticated user, "
        "ordered newest-first. "
        "Requires a valid JWT Bearer token."
    ),
)
async def load_templates(
    user_id: Annotated[str, Depends(get_current_user)],
) -> TemplateListResponse:
    rows = await asyncio.to_thread(fetch_templates_for_user, user_id)
    templates = [TemplateRecord(**r) for r in rows]
    return TemplateListResponse(user_id=user_id, templates=templates)
