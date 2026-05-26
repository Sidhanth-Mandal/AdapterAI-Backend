"""
chunking.py
-----------
Splits raw text into overlapping chunks suitable for embedding.

Strategy:
  - Split on sentence / paragraph boundaries first (greedy)
  - Hard-cap each chunk at `chunk_size` characters
  - Overlap by `overlap` characters to preserve context across boundaries
"""

from __future__ import annotations

import re
from typing import List


# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 512      # target characters per chunk
DEFAULT_OVERLAP    = 64       # characters shared between adjacent chunks


# ── helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using a lightweight regex."""
    # Split after . ! ? followed by whitespace or end-of-string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _merge_sentences(sentences: List[str], chunk_size: int) -> List[str]:
    """
    Greedily merge sentences into blocks that fit within chunk_size chars.
    Any single sentence longer than chunk_size is kept as-is (split later).
    """
    blocks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        # +1 for the space we'll add between sentences
        if current_len + sentence_len + (1 if current else 0) > chunk_size and current:
            blocks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sentence)
        current_len += sentence_len + (1 if len(current) > 1 else 0)

    if current:
        blocks.append(" ".join(current))

    return blocks


def _hard_split(text: str, chunk_size: int) -> List[str]:
    """Hard-split a single string that is longer than chunk_size."""
    return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]


# ── public API ─────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """
    Convert `text` into a list of overlapping text chunks.

    Parameters
    ----------
    text       : The raw text to chunk.
    chunk_size : Maximum characters per chunk (default 512).
    overlap    : Characters of overlap between consecutive chunks (default 64).

    Returns
    -------
    List[str]  : Ordered list of text chunks.
    """
    if not text or not text.strip():
        return []

    # 1. Split into sentences → merge into blocks ≤ chunk_size
    sentences = _split_sentences(text)
    blocks = _merge_sentences(sentences, chunk_size)

    # 2. Hard-split any block that still exceeds chunk_size
    raw_chunks: List[str] = []
    for block in blocks:
        if len(block) > chunk_size:
            raw_chunks.extend(_hard_split(block, chunk_size))
        else:
            raw_chunks.append(block)

    if not raw_chunks:
        return []

    # 3. Apply overlap: prepend tail of previous chunk to current chunk
    if overlap <= 0 or len(raw_chunks) == 1:
        return raw_chunks

    final_chunks: List[str] = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = raw_chunks[i - 1][-overlap:]
        merged    = prev_tail + " " + raw_chunks[i]
        # Hard-cap if overlap caused the chunk to exceed limit
        if len(merged) > chunk_size + overlap:
            merged = merged[: chunk_size + overlap]
        final_chunks.append(merged)

    return final_chunks
