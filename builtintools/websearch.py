"""
builtintools/websearch.py — Tavily-powered web search tool for LangGraph agents.

Provides three @tool-decorated functions:
  - web_search          : General web search, returns ranked results with snippets
  - web_search_focused  : Topic-scoped search with domain filtering and date controls
  - fetch_page_content  : Retrieve and extract clean text from a specific URL

Usage in your LangGraph agent
------------------------------
    from builtintools.websearch import TOOLS
    model = ChatGroq(...).bind_tools(TOOLS)

Environment
-----------
    TAVILY_API_KEY  — required (get one at https://tavily.com)
"""

from __future__ import annotations

import os
from typing import Annotated, Any, Optional
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

_TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ---------------------------------------------------------------------------
# Lazy client — instantiated once on first use
# ---------------------------------------------------------------------------

_client = None

def _get_client():
    global _client
    if _client is None:
        if not _TAVILY_API_KEY:
            raise EnvironmentError(
                "TAVILY_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )
        try:
            from tavily import TavilyClient
        except ImportError:
            raise ImportError(
                "tavily-python is not installed. Run: pip install tavily-python"
            )
        _client = TavilyClient(api_key=_TAVILY_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Helper — clean and format raw Tavily results
# ---------------------------------------------------------------------------

def _format_results(raw_results: dict, include_raw_content: bool = False) -> dict:
    """
    Convert raw Tavily response into a clean, agent-friendly structure.

    Returns
    -------
    dict with keys:
        query           : the original query string
        answer          : Tavily's AI-generated direct answer (may be empty)
        results         : list of result dicts with url, title, snippet, score
        total_results   : number of results returned
        follow_up_questions : suggested follow-up queries (if available)
    """
    results = []
    for r in raw_results.get("results", []):
        entry: dict[str, Any] = {
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "snippet": r.get("content", ""),
            "score":   round(r.get("score", 0.0), 4),
            "published_date": r.get("published_date", ""),
        }
        if include_raw_content and r.get("raw_content"):
            entry["raw_content"] = r["raw_content"][:4000]  # cap at 4k chars
        results.append(entry)

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query":               raw_results.get("query", ""),
        "answer":              raw_results.get("answer", ""),
        "results":             results,
        "total_results":       len(results),
        "follow_up_questions": raw_results.get("follow_up_questions", []),
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def web_search(
    query: str,
    max_results: int = 5,
    include_answer: bool = True,
) -> dict:
    """Search the web using Tavily and return ranked results with snippets.

    Use this for general-purpose information retrieval, current events,
    facts, news, and any topic that benefits from live web data.

    Args:
        query: The search query string. Be specific for better results.
               Example: "Python asyncio best practices 2024"
        max_results: Number of results to return (1-10). Default is 5.
        include_answer: If True, Tavily generates a direct AI answer
                        synthesised from the top results. Default is True.

    Returns:
        dict with keys:
          - query (str): The original query
          - answer (str): AI-synthesised direct answer (empty if include_answer=False)
          - results (list): Ranked list of results, each containing:
              - title (str): Page title
              - url (str): Source URL
              - snippet (str): Relevant excerpt from the page
              - score (float): Relevance score 0.0–1.0
              - published_date (str): Publication date if available
          - total_results (int): Number of results returned
          - follow_up_questions (list): Suggested follow-up queries
    """
    client = _get_client()
    max_results = max(1, min(10, max_results))

    raw = client.search(
        query=query,
        search_depth="basic",
        max_results=max_results,
        include_answer=include_answer,
        include_follow_up_questions=True,
    )
    return _format_results(raw)


@tool
def web_search_focused(
    query: str,
    topic: str = "general",
    days: int = 0,
    include_domains: str = "",
    exclude_domains: str = "",
    max_results: int = 5,
) -> dict:
    """Perform a focused web search with topic scoping, domain filtering, and date controls.

    Use this when you need more precise control over the search — e.g. only
    recent news, results from specific trusted domains, or a particular topic area.

    Args:
        query: The search query string.
        topic: Search topic scope. One of:
               - "general"  : broad web search (default)
               - "news"     : focuses on news articles and recent reporting
        days: If > 0, restrict results to articles published within the last N days.
              Set to 0 to search all time. Example: days=7 for the past week.
        include_domains: Comma-separated list of domains to INCLUDE exclusively.
                         Example: "reuters.com,bbc.com,nature.com"
                         Leave empty to search all domains.
        exclude_domains: Comma-separated list of domains to EXCLUDE.
                         Example: "reddit.com,quora.com"
                         Leave empty to exclude nothing.
        max_results: Number of results to return (1-10). Default is 5.

    Returns:
        dict with keys:
          - query (str): The original query
          - answer (str): AI-synthesised direct answer
          - results (list): Ranked list of results (title, url, snippet, score, published_date)
          - total_results (int): Number of results returned
          - follow_up_questions (list): Suggested follow-up queries
    """
    client = _get_client()
    max_results = max(1, min(10, max_results))

    # Parse domain lists
    inc_domains = [d.strip() for d in include_domains.split(",") if d.strip()] or None
    exc_domains = [d.strip() for d in exclude_domains.split(",") if d.strip()] or None

    kwargs: dict[str, Any] = dict(
        query=query,
        search_depth="advanced",
        topic=topic if topic in {"general", "news"} else "general",
        max_results=max_results,
        include_answer=True,
        include_follow_up_questions=True,
    )
    if days and days > 0:
        kwargs["days"] = days
    if inc_domains:
        kwargs["include_domains"] = inc_domains
    if exc_domains:
        kwargs["exclude_domains"] = exc_domains

    raw = client.search(**kwargs)
    return _format_results(raw)


@tool
def fetch_page_content(url: str) -> dict:
    """Retrieve and extract the full clean text content of a specific web page.

    Use this when you have a URL from search results and need to read the
    complete article, documentation, or page — not just the snippet.

    Args:
        url: The full URL of the page to fetch.
             Example: "https://docs.python.org/3/library/asyncio.html"

    Returns:
        dict with keys:
          - url (str): The fetched URL
          - title (str): Page title
          - content (str): Extracted clean text (up to ~8000 characters)
          - raw_content (str): Longer raw content if available
          - failed_results (list): Any URLs that could not be fetched
    """
    client = _get_client()

    raw = client.extract(urls=[url])

    extracted = raw.get("results", [])
    failed = raw.get("failed_results", [])

    if not extracted:
        return {
            "url": url,
            "title": "",
            "content": "",
            "raw_content": "",
            "failed_results": failed,
            "error": "Page could not be fetched or had no extractable content.",
        }

    page = extracted[0]
    content = page.get("raw_content", "")

    return {
        "url":          page.get("url", url),
        "title":        page.get("title", ""),
        "content":      content[:8000],          # clean truncated excerpt
        "raw_content":  content,                 # full content for agent use
        "failed_results": failed,
    }


# ---------------------------------------------------------------------------
# Tool registry — import this in your LangGraph agent
# ---------------------------------------------------------------------------

TOOLS = [web_search, web_search_focused, fetch_page_content]
