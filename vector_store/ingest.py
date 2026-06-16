"""
ingest.py
---------
Converts uploaded files (PDF, DOCX, image, audio) into text,
chunks the text, embeds it via Cloudflare AI, and stores it in Pinecone.

Supported file types
--------------------
  PDF   → pdfplumber  (text extraction)
  DOCX  → python-docx (text extraction)
  Image → Cloudflare AI Workers (OCR / image-to-text)
          model: @cf/llava-phi-3  (vision)
  Audio → Cloudflare AI Workers (ASR / speech-to-text)
          model: @cf/openai/whisper

Environment variables required (see embeddings.py / pinecone_client.py):
  CLOUDFLARE_API_TOKEN
  CLOUDFLARE_ACCOUNT_ID
  PINECONE_API_KEY
  PINECONE_INDEX
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import List, Optional, Union

import httpx

# ── local modules ─────────────────────────────────────────────────────────────
from .chunking       import chunk_text
from .embeddings     import embed_texts
from .pinecone_client import upsert_vectors

# ── LangSmith tracing ─────────────────────────────────────────────────────────
from utils.tracing import traceable  # noqa: E402


# ── Cloudflare config ─────────────────────────────────────────────────────────
CF_API_TOKEN  = os.getenv("CLOUDFLARE_API_TOKEN", "")
CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")

# Cloudflare vision model for image → text
# Docs: https://developers.cloudflare.com/workers-ai/models/llava-1.5-7b-hf/
_CF_VISION_MODEL  = "@cf/llava-hf/llava-1.5-7b-hf"
# Cloudflare Whisper model for audio → text (turbo = faster, less timeout risk)
# Docs: https://developers.cloudflare.com/workers-ai/models/whisper-large-v3-turbo/
_CF_WHISPER_MODEL = "@cf/openai/whisper-large-v3-turbo"


def _cf_url(model: str) -> str:
    account_id = CF_ACCOUNT_ID or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
    return (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/ai/run/{model}"
    )


def _cf_headers() -> dict:
    token = CF_API_TOKEN or os.getenv("CLOUDFLARE_API_TOKEN", "")
    if not token:
        raise EnvironmentError("CLOUDFLARE_API_TOKEN is not set.")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


# ── per-format text extraction ─────────────────────────────────────────────────

def _extract_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Install pdfplumber: pip install pdfplumber")

    text_parts: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


def _extract_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")

    doc   = Document(io.BytesIO(file_bytes))
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paras)


def _extract_image_via_cf(file_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """
    Send an image to Cloudflare AI (LLaVA) for OCR / description.
    Returns the model's text output.

    The REST API for @cf/llava-hf/llava-1.5-7b-hf expects:
      { "image": [<byte int>, ...], "prompt": "..." }
    where "image" is a plain list of unsigned byte integers (0-255),
    NOT a base64 data URI.
    """
    # Convert bytes → list of unsigned ints (what the CF REST API expects)
    image_array = list(file_bytes)

    payload = {
        "image":  image_array,
        "prompt": (
            "Extract all text visible in this image verbatim. "
            "If there is no readable text, describe the image in detail."
        ),
        "max_tokens": 1024,
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            _cf_url(_CF_VISION_MODEL),
            headers=_cf_headers(),
            json=payload,
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Cloudflare vision API error {response.status_code}: {response.text}"
        )

    data = response.json()
    if not data.get("success"):
        raise RuntimeError(f"Cloudflare vision API errors: {data.get('errors')}")

    # LLaVA response: { "result": { "description": "..." } }
    result = data.get("result", {})
    return result.get("description", result.get("response", ""))


def _extract_audio_via_cf(file_bytes: bytes) -> str:
    """
    Send audio bytes to Cloudflare AI Whisper for transcription.
    Returns the transcribed text.

    The REST API for @cf/openai/whisper-large-v3-turbo expects JSON:
      { "audio": "<base64-encoded audio string>" }
    Sending raw binary (octet-stream) causes 408 timeouts on the REST API.
    """
    token = CF_API_TOKEN or os.getenv("CLOUDFLARE_API_TOKEN", "")
    if not token:
        raise EnvironmentError("CLOUDFLARE_API_TOKEN is not set.")

    # Encode audio as base64 string — required by the CF REST API
    audio_b64 = base64.b64encode(file_bytes).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"audio": audio_b64}

    with httpx.Client(timeout=300.0) as client:
        response = client.post(
            _cf_url(_CF_WHISPER_MODEL),
            headers=headers,
            json=payload,
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Cloudflare Whisper API error {response.status_code}: {response.text}"
        )

    data = response.json()
    if not data.get("success"):
        raise RuntimeError(f"Cloudflare Whisper API errors: {data.get('errors')}")

    # Whisper response: { "result": { "text": "..." } }
    return data.get("result", {}).get("text", "")


# ── mime-type detection ───────────────────────────────────────────────────────

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus"}

_EXT_TO_MIME = {
    ".jpg"  : "image/jpeg",
    ".jpeg" : "image/jpeg",
    ".png"  : "image/png",
    ".gif"  : "image/gif",
    ".bmp"  : "image/bmp",
    ".webp" : "image/webp",
    ".tiff" : "image/tiff",
}


def _detect_and_extract(filename: str, file_bytes: bytes) -> str:
    """
    Detect file type from extension and extract text using the appropriate method.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(file_bytes)

    if ext == ".docx":
        return _extract_docx(file_bytes)

    if ext in _IMAGE_EXTS:
        mime = _EXT_TO_MIME.get(ext, "image/jpeg")
        return _extract_image_via_cf(file_bytes, mime_type=mime)

    if ext in _AUDIO_EXTS:
        return _extract_audio_via_cf(file_bytes)

    # Fallback: try to decode as plain text (e.g. .txt, .md, .csv)
    try:
        return file_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        raise ValueError(f"Unsupported file type '{ext}': {exc}") from exc


# ── main public API ────────────────────────────────────────────────────────────

@traceable(
    name="vector_store_ingest",
    tags=["vector-store", "ingest"],
    metadata={"pipeline": "VectorStore"},
)
def ingest_file(
    file_bytes: bytes,
    filename:   str,
    user_id:    str,
    conv_id:    str,
    chunk_size: int = 512,
    overlap:    int = 64,
    namespace:  str = "",
) -> dict:
    """
    Full ingest pipeline:
      file_bytes → text → chunks → embeddings → Pinecone upsert.

    Parameters
    ----------
    file_bytes  : Raw file content (bytes).
    filename    : Original filename including extension (used for type detection
                  and stored as metadata `source`).
    user_id     : User identifier (stored in Pinecone metadata for filtering).
    conv_id     : Conversation identifier (stored in Pinecone metadata for filtering).
    chunk_size  : Max characters per chunk (default 512).
    overlap     : Overlap characters between consecutive chunks (default 64).
    namespace   : Optional Pinecone namespace.

    Returns
    -------
    dict with keys:
      source      – filename
      text_length – total characters extracted
      num_chunks  – number of chunks produced
      vector_ids  – list of Pinecone vector IDs upserted
    """
    if not user_id:
        raise ValueError("user_id must not be empty.")
    if not conv_id:
        raise ValueError("conv_id must not be empty.")
    if not file_bytes:
        raise ValueError("file_bytes must not be empty.")

    # 1. Extract text from file
    raw_text = _detect_and_extract(filename, file_bytes)
    if not raw_text or not raw_text.strip():
        raise ValueError(f"No text could be extracted from '{filename}'.")

    # 2. Chunk the text
    chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("Text chunking produced no chunks.")

    # 3. Embed all chunks via Cloudflare AI
    vectors = embed_texts(chunks)

    # 4. Upsert into Pinecone with user_id + conv_id metadata
    vector_ids = upsert_vectors(
        vectors   = vectors,
        chunks    = chunks,
        user_id   = user_id,
        conv_id   = conv_id,
        source    = filename,
        namespace = namespace,
    )

    return {
        "source"     : filename,
        "text_length": len(raw_text),
        "num_chunks" : len(chunks),
        "vector_ids" : vector_ids,
    }


@traceable(
    name="vector_store_ingest_path",
    tags=["vector-store", "ingest"],
    metadata={"pipeline": "VectorStore"},
)
def ingest_file_path(
    filepath:  Union[str, Path],
    user_id:   str,
    conv_id:   str,
    chunk_size: int = 512,
    overlap:    int = 64,
    namespace:  str = "",
) -> dict:
    """
    Convenience wrapper — reads a file from disk and ingests it.

    Parameters
    ----------
    filepath  : Absolute or relative path to the file.
    user_id   : User identifier.
    conv_id   : Conversation identifier.

    Returns
    -------
    Same dict as ingest_file().
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    file_bytes = path.read_bytes()
    return ingest_file(
        file_bytes  = file_bytes,
        filename    = path.name,
        user_id     = user_id,
        conv_id     = conv_id,
        chunk_size  = chunk_size,
        overlap     = overlap,
        namespace   = namespace,
    )
