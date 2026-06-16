"""
retrieve.py
-----------
Semantic retrieval from Pinecone, scoped to a specific user + conversation.

Usage
-----
    from vector_store.retrieve import retrieve_chunks

    results = retrieve_chunks(
        query   = "What does the contract say about payment terms?",
        user_id = "user_abc",
        conv_id = "conv_xyz",
        top_k   = 5,
    )

    for r in results:
        print(r["score"], r["text"])
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .embeddings     import embed_query
from .pinecone_client import query_vectors

# ── LangSmith tracing ─────────────────────────────────────────────────────────
from utils.tracing import traceable  # noqa: E402


# ── public API ─────────────────────────────────────────────────────────────────

@traceable(
    name="vector_store_retrieve",
    tags=["vector-store", "retrieve"],
    metadata={"pipeline": "VectorStore"},
)
def retrieve_chunks(
    query:      str,
    user_id:    str,
    conv_id:    str,
    top_k:      int  = 5,
    min_score:  float = 0.0,
    namespace:  str  = "",
) -> List[Dict[str, Any]]:
    """
    Embed `query` and retrieve the top-k semantically similar chunks from
    Pinecone, filtered strictly to `user_id` + `conv_id`.

    Parameters
    ----------
    query      : Natural-language query string.
    user_id    : Filter — only search within this user's documents.
    conv_id    : Filter — only search within this conversation's documents.
    top_k      : Maximum number of results to return (default 5).
    min_score  : Minimum cosine similarity score to include in results (0–1).
                 Set to 0.0 to return all results regardless of score.
    namespace  : Pinecone namespace (optional; defaults to empty string).

    Returns
    -------
    List[Dict] where each dict contains:
      "id"        – Pinecone vector ID
      "score"     – cosine similarity score (float, higher = more similar)
      "text"      – the original chunk text
      "source"    – original filename the chunk came from
      "chunk_idx" – position of this chunk within the source document
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty.")
    if not user_id:
        raise ValueError("user_id must not be empty.")
    if not conv_id:
        raise ValueError("conv_id must not be empty.")

    # 1. Embed the query via Cloudflare AI
    query_vector = embed_query(query)

    # 2. Query Pinecone with user_id + conv_id filter
    hits = query_vectors(
        query_vector = query_vector,
        user_id      = user_id,
        conv_id      = conv_id,
        top_k        = top_k,
        namespace    = namespace,
    )

    # 3. Apply optional minimum score filter
    if min_score > 0.0:
        hits = [h for h in hits if h["score"] >= min_score]

    return hits


@traceable(
    name="vector_store_retrieve_context",
    tags=["vector-store", "retrieve"],
    metadata={"pipeline": "VectorStore"},
)
def retrieve_context_string(
    query:      str,
    user_id:    str,
    conv_id:    str,
    top_k:      int   = 5,
    min_score:  float = 0.0,
    separator:  str   = "\n\n---\n\n",
    namespace:  str   = "",
) -> str:
    """
    Convenience wrapper — retrieves chunks and joins their text into a single
    string ready to be injected into an LLM prompt as context.

    Parameters
    ----------
    separator  : String used to join chunks (default: a markdown horizontal rule).

    Returns
    -------
    str : Concatenated chunk texts, or empty string if no results found.
    """
    hits = retrieve_chunks(
        query     = query,
        user_id   = user_id,
        conv_id   = conv_id,
        top_k     = top_k,
        min_score = min_score,
        namespace = namespace,
    )

    if not hits:
        return ""

    return separator.join(h["text"] for h in hits)
