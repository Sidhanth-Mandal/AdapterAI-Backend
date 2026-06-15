"""
apis/routers/template_router.py
--------------------------------
GET  /loadtemplate/
    Returns all templates that belong to the authenticated user.

DELETE /loadtemplate/{template_id}
    Permanently deletes a template from the database.
    Only the user who created the template (created_by) can delete it.

No query parameters are required — user identity comes from the JWT token.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from apis.auth import get_current_user
from apis.db import delete_template, fetch_templates_for_user
from apis.schemas import (
    DeleteTemplateResponse,
    TemplateListResponse,
    TemplateRecord,
)

router = APIRouter(prefix="/loadtemplate", tags=["Templates"])


# ---------------------------------------------------------------------------
# GET /loadtemplate/  — list all templates for the current user
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DELETE /loadtemplate/{template_id}  — delete a template
# ---------------------------------------------------------------------------

@router.delete(
    "/{template_id}",
    response_model=DeleteTemplateResponse,
    summary="Delete a template",
    description=(
        "Permanently removes a template from the database.\n\n"
        "Only the user who originally created the template (`created_by`) "
        "may delete it. Returns `404` if the template does not exist or "
        "belongs to another user.\n\n"
        "**Note:** Deleting a template does **not** delete existing conversations "
        "that were started with it. Use `DELETE /loadconv/{conv_id}` for that."
    ),
)
async def delete_template_endpoint(
    template_id: str,
    user_id: Annotated[str, Depends(get_current_user)],
) -> DeleteTemplateResponse:
    deleted: bool = await asyncio.to_thread(delete_template, template_id, user_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found or access denied.",
        )

    return DeleteTemplateResponse(template_id=template_id, deleted=True)
