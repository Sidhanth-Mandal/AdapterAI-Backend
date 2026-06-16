"""
apis/routers/tempchat_router.py
--------------------------------
POST /tempchat/
---------------
Sends one user turn to the TemplateCreation pipeline.

TemplateCreation/service.chat_template() is synchronous (it calls a
synchronous LangGraph graph.invoke and psycopg2 directly), so we dispatch
it via asyncio.to_thread() to avoid blocking the FastAPI event loop.

JWT-protected: requires a valid Bearer token in the Authorization header.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from apis.auth import get_current_user
from apis.schemas import MCQOption, MCQQuestion, TempChatRequest, TempChatResponse

# TemplateCreation uses bare internal imports (e.g. `from graph import …`)
# which require its own directory to be on sys.path. Patch it here so that
# importing service.py works regardless of the CWD at server start-up.
_TC_DIR = Path(__file__).resolve().parents[2] / "TemplateCreation"
if str(_TC_DIR) not in sys.path:
    sys.path.insert(0, str(_TC_DIR))

from TemplateCreation.service import chat_template  # noqa: E402
from TemplateCreation.utils.extraction import extract_preamble, parse_mcq_questions  # noqa: E402

router = APIRouter(prefix="/tempchat", tags=["Template Chat"])


@router.post(
    "/",
    response_model=TempChatResponse,
    summary="Send a message to the TemplateCreation agent",
    description=(
        "Sends one user turn through the TemplateCreation LangGraph pipeline. "
        "The `template_id` acts as the session key — all messages for a given "
        "template build are keyed on this value in Redis and PostgreSQL. "
        "When the agent is satisfied with the gathered requirements, Phase 2 "
        "(template generation) is triggered automatically. "
        "Requires a valid JWT Bearer token."
    ),
)
async def tempchat_endpoint(
    body: TempChatRequest,
    user_id: Annotated[str, Depends(get_current_user)],
) -> TempChatResponse:
    # chat_template is synchronous — run it in a thread pool
    try:
        response_text: str = await asyncio.to_thread(
            chat_template,
            body.template_id,
            user_id,
            body.user_prompt,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TemplateCreation pipeline error: {exc}",
        )

    # -----------------------------------------------------------------------
    # Parse the structured MCQ blocks out of the raw assistant response so
    # the frontend can render interactive tick-mark cards instead of raw text.
    # -----------------------------------------------------------------------
    raw_questions = parse_mcq_questions(response_text)
    preamble      = extract_preamble(response_text)

    questions = [
        MCQQuestion(
            question=q["question"],
            options=[MCQOption(label=opt["label"], text=opt["text"]) for opt in q["options"]],
        )
        for q in raw_questions
    ]

    return TempChatResponse(
        template_id=body.template_id,
        preamble=preamble,
        questions=questions,
        response=response_text,
    )
