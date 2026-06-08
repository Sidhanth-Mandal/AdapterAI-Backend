"""
builtintools/retrieval.py — RAG retrieval tool for LangGraph agents.

Performs semantic search over the conversation's uploaded documents stored
in Pinecone, scoped strictly to the current user and conversation.

The tool takes only ONE explicit input (the search query).
user_id and conv_id are injected automatically from the LangGraph
RunnableConfig.configurable dict — the caller never needs to pass them.

Expected config shape (set when you call graph.invoke / astream):
    config = {
        "configurable": {
            "thread_id": f"{user_id}:{conv_id}",
            "user_id":   user_id,
            "conv_id":   conv_id,
        }
    }

Usage in your LangGraph agent
------------------------------
    from builtintools.retrieval import TOOLS
    model = ChatGroq(...).bind_tools(TOOLS)

    # Only enable when the attachment flag is on, e.g.:
    tools = TOOLS if has_attachments else []
    model_with_tools = base_model.bind_tools(tools)

Environment variables required (inherited from vector_store modules)
--------------------------------------------------------------------
    PINECONE_API_KEY        — Pinecone API key
    PINECONE_INDEX          — Pinecone index name (default: adapterai-index)
    CLOUDFLARE_API_TOKEN    — Cloudflare Workers AI token (for embeddings)
    CLOUDFLARE_ACCOUNT_ID   — Cloudflare account ID
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from vector_store.retrieve import retrieve_chunks


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _extract_ids(config: RunnableConfig) -> tuple[str, str]:
    """
    Pull user_id and conv_id out of config["configurable"].

    Raises
    ------
    ValueError
        If either key is missing or empty — prevents silent mis-scoping
        of vector queries.
    """
    configurable: Dict[str, Any] = (config or {}).get("configurable", {})

    user_id = configurable.get("user_id", "")
    conv_id = configurable.get("conv_id", "")

    if not user_id:
        raise ValueError(
            "user_id is missing from config['configurable']. "
            "Make sure you pass it when invoking the graph:\n"
            "  config = {'configurable': {'user_id': ..., 'conv_id': ...}}"
        )
    if not conv_id:
        raise ValueError(
            "conv_id is missing from config['configurable']. "
            "Make sure you pass it when invoking the graph:\n"
            "  config = {'configurable': {'user_id': ..., 'conv_id': ...}}"
        )

    return user_id, conv_id


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool
def retrieve_from_documents(
    query: str,
    config: RunnableConfig,
    top_k: int = 5,
    min_score: float = 0.35,
) -> Dict[str, Any]:
    """Search the user's uploaded documents for information relevant to the query.

    Use this tool whenever the user asks about content in files or documents
    they have shared in the current conversation (invoices, reports, contracts,
    PDFs, spreadsheets, etc.).  Do NOT use it for general knowledge questions
    that don't relate to uploaded files.

    The search is automatically scoped to the current user and conversation —
    results from other users or sessions are never returned.

    Args:
        query: A natural-language description of what you are looking for.
               Be specific and phrase it as a question or a key phrase from
               the document.
               Examples:
                 "What are the payment terms in the contract?"
                 "Total invoice amount"
                 "Employee names listed in the HR report"
        top_k: Maximum number of document chunks to retrieve (default 5,
               range 1–20).  Increase when you need broader coverage;
               decrease for a tighter, more focused answer.
        min_score: Minimum cosine-similarity threshold (0.0–1.0, default 0.35).
                   Chunks scoring below this value are discarded.  Lower values
                   return more results but may include noise; higher values
                   keep only the most relevant passages.

    Returns:
        dict with keys:
          - query (str)          : The original query
          - total_results (int)  : Number of chunks returned
          - results (list)       : Each result contains:
              - rank (int)       : 1-based rank (most similar first)
              - score (float)    : Cosine similarity (0.0–1.0, higher = better)
              - text (str)       : The matching document excerpt
              - source (str)     : Filename / label the chunk came from
              - chunk_idx (int)  : Position of the chunk within its source file
          - message (str)        : Human-readable summary (useful when 0 results)
          - error (str | None)   : Error description on failure, else null
    """
    # Clamp parameters to sane ranges
    top_k     = max(1, min(20, int(top_k)))
    min_score = max(0.0, min(1.0, float(min_score)))

    # --- Validate query -------------------------------------------------------
    if not query or not query.strip():
        return {
            "query":         query,
            "total_results": 0,
            "results":       [],
            "message":       "Query was empty — please provide a search term.",
            "error":         "Empty query.",
        }

    # --- Pull identity from config --------------------------------------------
    try:
        user_id, conv_id = _extract_ids(config)
    except ValueError as exc:
        return {
            "query":         query,
            "total_results": 0,
            "results":       [],
            "message":       str(exc),
            "error":         str(exc),
        }

    # --- Retrieve from Pinecone -----------------------------------------------
    try:
        hits = retrieve_chunks(
            query     = query.strip(),
            user_id   = user_id,
            conv_id   = conv_id,
            top_k     = top_k,
            min_score = min_score,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "query":         query,
            "total_results": 0,
            "results":       [],
            "message":       f"Retrieval failed: {exc}",
            "error":         str(exc),
        }

    # --- Format output --------------------------------------------------------
    if not hits:
        return {
            "query":         query,
            "total_results": 0,
            "results":       [],
            "message": (
                "No relevant document chunks were found for this query. "
                "The uploaded documents may not contain information about this topic, "
                "or the similarity score threshold may be too high."
            ),
            "error": None,
        }

    formatted: List[Dict[str, Any]] = [
        {
            "rank":      rank,
            "score":     round(hit["score"], 4),
            "text":      hit["text"],
            "source":    hit["source"],
            "chunk_idx": hit["chunk_idx"],
        }
        for rank, hit in enumerate(hits, start=1)
    ]

    return {
        "query":         query,
        "total_results": len(formatted),
        "results":       formatted,
        "message":       f"Found {len(formatted)} relevant chunk(s).",
        "error":         None,
    }


# ---------------------------------------------------------------------------
# Tool registry — import this in your LangGraph agent
# ---------------------------------------------------------------------------

TOOLS = [retrieve_from_documents]
