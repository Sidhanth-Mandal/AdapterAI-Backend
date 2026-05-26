"""
embeddings.py
-------------
Generates text embeddings using Cloudflare AI Workers.

Model  : @cf/baai/bge-base-en-v1.5
Docs   : https://developers.cloudflare.com/workers-ai/models/bge-base-en-v1.5/

Environment variables required:
  CLOUDFLARE_API_TOKEN   – Cloudflare API token with Workers AI permission
  CLOUDFLARE_ACCOUNT_ID  – Your Cloudflare account ID

The REST endpoint is:
  POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/baai/bge-base-en-v1.5
"""

from __future__ import annotations

import os
from typing import List

import httpx


# ── config ────────────────────────────────────────────────────────────────────
CF_API_TOKEN   = os.getenv("CLOUDFLARE_API_TOKEN", "")
CF_ACCOUNT_ID  = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CF_EMBED_MODEL = "@cf/baai/bge-base-en-v1.5"

_CF_EMBED_URL  = (
    f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}"
    f"/ai/run/{CF_EMBED_MODEL}"
)

# Cloudflare BGE endpoint accepts up to 100 texts per request
_CF_BATCH_LIMIT = 100

EMBEDDING_DIM = 768   # bge-base-en-v1.5 output dimension


# ── internal helpers ───────────────────────────────────────────────────────────

def _build_headers() -> dict:
    token = CF_API_TOKEN or os.getenv("CLOUDFLARE_API_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "CLOUDFLARE_API_TOKEN is not set. "
            "Add it to your .env file or environment."
        )
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Send a single batch (≤ _CF_BATCH_LIMIT texts) to Cloudflare AI and
    return a list of float vectors.
    """
    account_id = CF_ACCOUNT_ID or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/ai/run/{CF_EMBED_MODEL}"
    )
    payload = {"text": texts}

    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, headers=_build_headers(), json=payload)

    if response.status_code != 200:
        raise RuntimeError(
            f"Cloudflare embedding API error {response.status_code}: {response.text}"
        )

    data = response.json()

    # Response shape: { "result": { "data": [[...], [...]] }, "success": true }
    if not data.get("success"):
        errors = data.get("errors", [])
        raise RuntimeError(f"Cloudflare API returned errors: {errors}")

    vectors: List[List[float]] = data["result"]["data"]
    return vectors


# ── public API ─────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Automatically batches requests to stay within Cloudflare's per-request
    limit of _CF_BATCH_LIMIT texts.

    Parameters
    ----------
    texts : List of strings to embed. Empty strings are replaced with a
            single space so the API never receives an empty input.

    Returns
    -------
    List[List[float]] : One 768-dimensional vector per input text,
                        in the same order as `texts`.
    """
    if not texts:
        return []

    # Guard against empty strings
    safe_texts = [t if t.strip() else " " for t in texts]

    all_vectors: List[List[float]] = []
    for i in range(0, len(safe_texts), _CF_BATCH_LIMIT):
        batch = safe_texts[i : i + _CF_BATCH_LIMIT]
        all_vectors.extend(_embed_batch(batch))

    return all_vectors


def embed_query(query: str) -> List[float]:
    """
    Convenience wrapper — embed a single query string and return its vector.

    Parameters
    ----------
    query : The user query to embed.

    Returns
    -------
    List[float] : 768-dimensional embedding vector.
    """
    vectors = embed_texts([query])
    return vectors[0]
