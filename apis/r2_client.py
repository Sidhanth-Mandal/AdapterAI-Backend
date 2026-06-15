"""
apis/r2_client.py
-----------------
Cloudflare R2 upload utilities.

R2 exposes an S3-compatible endpoint, so we use boto3 with a custom
endpoint_url pointing at:
    https://<ACCOUNT_ID>.r2.cloudflarestorage.com

Required .env keys
------------------
  R2_ACCOUNT_ID        — Cloudflare account ID
  R2_ACCESS_KEY_ID     — R2 API token (Access Key ID)
  R2_SECRET_ACCESS_KEY — R2 API token (Secret Access Key)
  R2_BUCKET_NAME       — target bucket name
  R2_PUBLIC_URL        — public base URL (e.g. https://pub-xxx.r2.dev)

All functions are synchronous; call them via asyncio.to_thread() from
async FastAPI endpoints.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # AdapterAI/
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Lazy S3 client (R2 endpoint)
# ---------------------------------------------------------------------------

_s3_client = None


def _get_r2_client():
    """Return a boto3 S3 client pointed at the R2 endpoint (singleton)."""
    global _s3_client
    if _s3_client is None:
        account_id = os.environ["R2_ACCOUNT_ID"]
        _s3_client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
    return _s3_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_file_to_r2(
    file_content: bytes,
    file_name: str,
    mime_type: str,
    *,
    folder: str = "attachments",
) -> tuple[str, str]:
    """
    Upload *file_content* to R2 and return ``(attachment_id, storage_url)``.

    Parameters
    ----------
    file_content : bytes
        Raw file bytes.
    file_name : str
        Original file name (used to build the object key suffix).
    mime_type : str
        MIME type sent as ``ContentType`` to R2.
    folder : str
        Prefix / "folder" inside the bucket (default ``attachments``).

    Returns
    -------
    attachment_id : str
        A UUID4 string that uniquely identifies this attachment.
        This becomes the primary key in the Attachments table and
        also forms part of the R2 object key.
    storage_url : str
        Full public URL to the uploaded object.
    """
    attachment_id = str(uuid.uuid4())
    # Sanitise file name to avoid path injection
    safe_name = Path(file_name).name
    object_key = f"{folder}/{attachment_id}/{safe_name}"

    bucket = os.environ["R2_BUCKET_NAME"]
    client = _get_r2_client()

    client.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=file_content,
        ContentType=mime_type,
    )

    public_base = os.environ["R2_PUBLIC_URL"].rstrip("/")
    storage_url = f"{public_base}/{object_key}"

    return attachment_id, storage_url
