"""
apis/supabase_storage.py
------------------------
Supabase Storage upload utilities.

Uses the official ``supabase-py`` client to upload files to a
Supabase Storage bucket.

Required .env keys
------------------
  SUPABASE_URL         — project URL, e.g. https://xyzxyz.supabase.co
  SUPABASE_SERVICE_KEY — service-role secret key (bypasses RLS for server-side uploads)
  SUPABASE_BUCKET      — target bucket name, e.g. attachments

All functions are synchronous; call them via asyncio.to_thread() from
async FastAPI endpoints.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # AdapterAI/
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Lazy Supabase client (module-level singleton)
# ---------------------------------------------------------------------------

_supabase: Client | None = None


def _get_supabase() -> Client:
    """Return the shared Supabase client, initialising it on first call."""
    global _supabase
    if _supabase is None:
        _supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_KEY"],
        )
    return _supabase


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_file_to_supabase(
    file_content: bytes,
    file_name: str,
    mime_type: str,
    *,
    folder: str = "attachments",
) -> tuple[str, str]:
    """
    Upload *file_content* to Supabase Storage and return
    ``(attachment_id, storage_url)``.

    Parameters
    ----------
    file_content : bytes
        Raw file bytes.
    file_name : str
        Original file name (used to build the storage path suffix).
    mime_type : str
        MIME type passed as ``content-type`` to Supabase.
    folder : str
        Top-level "folder" inside the bucket (default ``attachments``).

    Returns
    -------
    attachment_id : str
        A UUID4 string that uniquely identifies this attachment.
        This is the primary key in the Attachments table and forms
        part of the storage path.
    storage_url : str
        Full public URL to the uploaded object.
    """
    attachment_id = str(uuid.uuid4())

    # Sanitise file name to avoid path-traversal issues
    safe_name = Path(file_name).name
    storage_path = f"{folder}/{attachment_id}/{safe_name}"

    bucket = os.environ["SUPABASE_BUCKET"]
    client = _get_supabase()

    client.storage.from_(bucket).upload(
        path=storage_path,
        file=file_content,
        file_options={"content-type": mime_type},
    )

    # Build the public URL
    # get_public_url returns the full URL string for public buckets.
    storage_url: str = client.storage.from_(bucket).get_public_url(storage_path)

    return attachment_id, storage_url


def delete_files_from_supabase(storage_urls: list[str]) -> dict:
    """
    Remove a list of objects from the Supabase Storage bucket.

    Parameters
    ----------
    storage_urls : list[str]
        Full public URLs as stored in the ``storage_url`` column of the
        Attachments table.  The bucket and project URL prefix are stripped
        to derive the object path that Supabase expects.

    Returns
    -------
    dict
        ``{"deleted": [...], "errors": [...]}``
        *deleted* — storage paths that were successfully removed.
        *errors*  — paths that could not be removed (with reason).
    """
    if not storage_urls:
        return {"deleted": [], "errors": []}

    bucket = os.environ["SUPABASE_BUCKET"]
    client = _get_supabase()

    # Derive the object path from each URL.
    # Public URL format:  <SUPABASE_URL>/storage/v1/object/public/<bucket>/<path>
    base_prefix = (
        f"{os.environ['SUPABASE_URL'].rstrip('/')}"
        f"/storage/v1/object/public/{bucket}/"
    )

    paths_to_delete: list[str] = []
    for url in storage_urls:
        if url.startswith(base_prefix):
            paths_to_delete.append(url[len(base_prefix):])
        else:
            # Fallback: store the raw URL so callers know it was skipped
            paths_to_delete.append(url)

    deleted: list[str] = []
    errors: list[dict] = []

    # supabase-py remove() accepts a list of paths
    try:
        client.storage.from_(bucket).remove(paths_to_delete)
        deleted = paths_to_delete
    except Exception as exc:  # noqa: BLE001
        errors.append({"paths": paths_to_delete, "reason": str(exc)})

    return {"deleted": deleted, "errors": errors}
