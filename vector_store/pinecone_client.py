"""
pinecone_client.py
------------------
Thin wrapper around the Pinecone Python SDK (v3+).

Environment variables required:
  PINECONE_API_KEY   – Your Pinecone API key
  PINECONE_INDEX     – Name of the Pinecone index to use

Index must be created with dimension=768 (bge-base-en-v1.5 output dim)
and metric="cosine" (or "dotproduct").

Each vector is upserted with metadata:
  {
    "user_id" : str,
    "conv_id"  : str,
    "text"     : str,   # original chunk text
    "source"   : str,   # original filename / source label
    "chunk_idx": int,   # position of chunk within source document
  }

Filtering during retrieval is done on `user_id` and `conv_id`.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pinecone import Pinecone, ServerlessSpec


# ── config ────────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "adapterai-index")
VECTOR_DIM       = 768     # @cf/baai/bge-base-en-v1.5


# ── singleton client ───────────────────────────────────────────────────────────

_pc: Optional[Pinecone]     = None
_index                       = None   # pinecone.Index instance


def _get_client() -> Pinecone:
    global _pc
    if _pc is None:
        api_key = PINECONE_API_KEY or os.getenv("PINECONE_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "PINECONE_API_KEY is not set. Add it to your .env file."
            )
        _pc = Pinecone(api_key=api_key)
    return _pc


def _get_index():
    global _index
    if _index is None:
        pc   = _get_client()
        name = PINECONE_INDEX or os.getenv("PINECONE_INDEX", "adapterai-index")

        existing = [i.name for i in pc.list_indexes()]
        if name not in existing:
            # Auto-create a serverless index if it doesn't exist yet.
            # Change cloud/region to match your Pinecone project settings.
            pc.create_index(
                name=name,
                dimension=VECTOR_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        _index = pc.Index(name)
    return _index


# ── public API ─────────────────────────────────────────────────────────────────

def upsert_vectors(
    vectors: List[List[float]],
    chunks:  List[str],
    user_id: str,
    conv_id: str,
    source:  str = "unknown",
    namespace: str = "",
) -> List[str]:
    """
    Upsert a batch of vectors + their metadata into Pinecone.

    Parameters
    ----------
    vectors   : List of 768-d float vectors (one per chunk).
    chunks    : Corresponding raw text chunks.
    user_id   : User identifier — stored as metadata for filtering.
    conv_id   : Conversation identifier — stored as metadata for filtering.
    source    : Original filename or label (e.g. "invoice.pdf").
    namespace : Pinecone namespace (optional; defaults to empty string).

    Returns
    -------
    List[str] : The generated vector IDs that were upserted.
    """
    if len(vectors) != len(chunks):
        raise ValueError(
            f"vectors ({len(vectors)}) and chunks ({len(chunks)}) must have equal length."
        )

    index   = _get_index()
    records = []
    ids     = []

    for chunk_idx, (vec, text) in enumerate(zip(vectors, chunks)):
        vid = str(uuid.uuid4())
        ids.append(vid)
        records.append({
            "id":     vid,
            "values": vec,
            "metadata": {
                "user_id"  : user_id,
                "conv_id"  : conv_id,
                "text"     : text,
                "source"   : source,
                "chunk_idx": chunk_idx,
            },
        })

    # Pinecone recommends batches of ≤ 100
    batch_size = 100
    for i in range(0, len(records), batch_size):
        index.upsert(vectors=records[i : i + batch_size], namespace=namespace)

    return ids


def query_vectors(
    query_vector: List[float],
    user_id: str,
    conv_id: str,
    top_k: int = 5,
    namespace: str = "",
) -> List[Dict[str, Any]]:
    """
    Query Pinecone for the top-k most similar vectors, filtered by
    user_id AND conv_id.

    Parameters
    ----------
    query_vector : 768-d query embedding.
    user_id      : Filter – only return vectors from this user.
    conv_id      : Filter – only return vectors from this conversation.
    top_k        : Number of results to return (default 5).
    namespace    : Pinecone namespace (optional).

    Returns
    -------
    List[Dict]   : Each dict has keys: id, score, text, source, chunk_idx
    """
    index = _get_index()

    pinecone_filter = {
        "$and": [
            {"user_id": {"$eq": user_id}},
            {"conv_id": {"$eq": conv_id}},
        ]
    }

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=pinecone_filter,
        namespace=namespace,
    )

    hits = []
    for match in result.get("matches", []):
        meta = match.get("metadata", {})
        hits.append({
            "id"        : match["id"],
            "score"     : match["score"],
            "text"      : meta.get("text", ""),
            "source"    : meta.get("source", ""),
            "chunk_idx" : meta.get("chunk_idx", -1),
        })

    return hits


def delete_by_conversation(
    user_id: str,
    conv_id: str,
    namespace: str = "",
) -> None:
    """
    Delete all vectors belonging to a specific user + conversation.
    Uses Pinecone's delete-by-metadata filter (requires a paid plan or
    index with filter support).
    """
    index  = _get_index()
    f      = {"$and": [{"user_id": {"$eq": user_id}}, {"conv_id": {"$eq": conv_id}}]}
    index.delete(filter=f, namespace=namespace)
