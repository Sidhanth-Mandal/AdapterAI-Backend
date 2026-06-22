"""
apis/title_generator.py
------------------------
Generates a concise 5-6 word conversation title from the user's first message.

Uses the Groq API (llama-3.3-70b-versatile) — a fast, lightweight model ideal
for this micro-task (~200 ms latency, minimal token cost).

The function is synchronous and is meant to be called via asyncio.to_thread()
from async FastAPI endpoints so it never blocks the event loop.

Fallback behaviour
------------------
If the Groq call fails for any reason (network error, API error, etc.) the
function falls back to truncating the first 60 characters of the user prompt
so the chat endpoint never breaks because of a title generation failure.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Bootstrap — load .env from project root
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # AdapterAI/
load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq client (module-level singleton — thread-safe for reads)
# ---------------------------------------------------------------------------

_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    """Return the shared Groq client, initialising it on first call."""
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in the environment.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ---------------------------------------------------------------------------
# Title generation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a conversation title generator. "
    "Given the user's first message, respond with ONLY a concise 5-6 word title "
    "that captures the topic. "
    "Rules: no punctuation at the end, no surrounding quotes, title-case only, "
    "no extra explanation — just the title."
)

_MODEL = "llama-3.3-70b-versatile"
_MAX_TOKENS = 20  # a 5-6 word title never needs more than 20 tokens


def generate_title(user_prompt: str) -> str:
    """
    Generate a 5-6 word title for a new conversation.

    Parameters
    ----------
    user_prompt : str
        The user's first message in the conversation.

    Returns
    -------
    str
        A short, human-readable title (5-6 words).
        Falls back to a truncated version of ``user_prompt`` if the LLM
        call fails for any reason.
    """
    # ── Fallback title (used if LLM call fails) ────────────────────────────
    fallback = user_prompt.strip()[:57]
    if len(user_prompt.strip()) > 57:
        fallback += "..."

    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            temperature=0.4,   # low temperature → consistent, focused titles
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt.strip()},
            ],
        )
        title = response.choices[0].message.content.strip()

        # Strip any stray quotes the model might add despite instructions
        title = title.strip('"').strip("'").strip()

        if not title:
            logger.warning("[title_generator] Empty title returned; using fallback.")
            return fallback

        logger.info("[title_generator] Generated title: %r", title)
        return title

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[title_generator] Groq call failed (%s); using fallback title.", exc
        )
        return fallback
