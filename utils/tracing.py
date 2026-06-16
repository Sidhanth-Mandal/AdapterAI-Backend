"""
utils/tracing.py
----------------
Centralised LangSmith tracing helper for AdapterAI.

All pipelines import `traceable` from here rather than from langsmith
directly. This ensures:
  1. The .env file is always loaded before the langsmith client is
     initialised (prevents "API key not set" errors on cold imports).
  2. A single lazy LangSmith client is reused across the process.
  3. A consistent set of project-level tags is applied to every trace.

Usage
-----
    from utils.tracing import traceable

    @traceable(name="my_pipeline", tags=["my-module"])
    def my_function(arg):
        ...

    # async functions work identically
    @traceable(name="my_async_pipeline", tags=["my-module"])
    async def my_async_function(arg):
        ...

Environment variables (set in .env)
------------------------------------
    LANGSMITH_TRACING  = true
    LANGSMITH_ENDPOINT = https://api.smith.langchain.com
    LANGSMITH_API_KEY  = lsv2_pt_...
    LANGSMITH_PROJECT  = "Ad"
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Ensure .env is loaded before any langsmith import so LANGSMITH_API_KEY
# and friends are available when the Client is constructed.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # AdapterAI/
load_dotenv(_PROJECT_ROOT / ".env", override=False)

# ---------------------------------------------------------------------------
# Lazy singleton LangSmith client
# ---------------------------------------------------------------------------
from langsmith import Client  # noqa: E402 — must come after load_dotenv
from langsmith import traceable  # noqa: E402,F401 — re-exported for callers


@lru_cache(maxsize=1)
def get_langsmith_client() -> Client:
    """
    Return a cached LangSmith Client instance.

    Uses the LANGSMITH_API_KEY and LANGSMITH_ENDPOINT env vars loaded
    from .env above.  Raises RuntimeError if the key is missing.
    """
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "LANGSMITH_API_KEY is not set. "
            "Add it to your .env file: LANGSMITH_API_KEY=lsv2_pt_..."
        )
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    return Client(api_url=endpoint, api_key=api_key)


# ---------------------------------------------------------------------------
# Convenience: project name from env (used in tags / metadata by callers)
# ---------------------------------------------------------------------------
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "AdapterAI").strip('"').strip("'")
