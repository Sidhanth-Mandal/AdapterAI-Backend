"""
apis/auth.py
------------
JWT authentication utilities for the AdapterAI API.

Provides:
  - Password hashing / verification (bcrypt via passlib)
  - JWT token creation and decoding (python-jose)
  - FastAPI dependency: get_current_user()

Environment variables (loaded from .env)
-----------------------------------------
  JWT_SECRET_KEY   — secret used to sign tokens (generate a strong random key)
  JWT_ALGORITHM    — algorithm, defaults to HS256
  JWT_EXPIRE_MINS  — token lifetime in minutes, defaults to 60

If JWT_SECRET_KEY is not set in .env, a hardcoded fallback is used for
development ONLY. Set a proper secret in production.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SECRET_KEY: str = os.environ.get(
    "JWT_SECRET_KEY",
    "CHANGE_ME_use_a_real_random_secret_in_production_32bytes_min",
)
ALGORITHM: str = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
    os.environ.get("JWT_EXPIRE_MINS", "60")
)

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Return the bcrypt hash of ``plain``."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if ``plain`` matches the stored ``hashed`` password."""
    return _pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def create_access_token(
    user_id: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT access token.

    Parameters
    ----------
    user_id : str
        Value stored in the ``sub`` claim.
    expires_delta : timedelta, optional
        Custom lifetime; defaults to ACCESS_TOKEN_EXPIRE_MINUTES.

    Returns
    -------
    str
        Encoded JWT string.
    """
    delta = expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.now(timezone.utc) + delta
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> str:
    """
    Decode a JWT token and return the ``sub`` (user_id) claim.

    Raises
    ------
    HTTPException 401
        If the token is invalid, expired, or missing the ``sub`` claim.
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise credentials_exc
        return user_id
    except JWTError:
        raise credentials_exc


# ---------------------------------------------------------------------------
# FastAPI OAuth2 scheme + dependency
# ---------------------------------------------------------------------------

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def get_current_user(token: str = Depends(_oauth2_scheme)) -> str:
    """
    FastAPI dependency that extracts and validates the Bearer token.

    Returns
    -------
    str
        The ``user_id`` encoded in the token.

    Raises
    ------
    HTTPException 401
        If the token is missing, malformed, or expired.
    """
    return decode_token(token)
