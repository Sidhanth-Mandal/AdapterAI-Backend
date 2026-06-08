"""
apis/routers/auth_router.py
----------------------------
Authentication endpoints.

POST /auth/token   — OAuth2 password-flow login (returns JWT)
POST /auth/login   — JSON-body login (returns same JWT, easier to call from JS)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from apis.auth import create_access_token, verify_password, hash_password
from apis.db import fetch_user_by_username, create_user
from apis.schemas import LoginRequest, TokenResponse, SignupRequest
import psycopg2

router = APIRouter(prefix="/auth", tags=["Auth"])


# ---------------------------------------------------------------------------
# Shared login logic
# ---------------------------------------------------------------------------

def _authenticate(username: str, password: str) -> str:
    """
    Validate credentials and return user_id.

    Raises HTTPException 401 on failure.
    """
    user = fetch_user_by_username(username)
    if user is None or not verify_password(password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user["user_id"]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/token",
    response_model=TokenResponse,
    summary="OAuth2 password-flow login",
    description=(
        "Standard OAuth2 `password` grant. Submit `username` and `password` "
        "as form fields. Returns a Bearer JWT for use in the Authorization header."
    ),
)
async def login_oauth2(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> TokenResponse:
    user_id = _authenticate(form_data.username, form_data.password)
    token = create_access_token(user_id=user_id)
    return TokenResponse(access_token=token)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="JSON-body login",
    description=(
        "Alternative login endpoint that accepts JSON instead of a form body. "
        "Returns the same Bearer JWT as `/auth/token`."
    ),
)
async def login_json(body: LoginRequest) -> TokenResponse:
    user_id = _authenticate(body.username, body.password)
    token = create_access_token(user_id=user_id)
    return TokenResponse(access_token=token)


@router.post(
    "/signup",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="User Registration",
    description=(
        "Registers a new user and returns a Bearer JWT. "
        "Requires username, email, and password."
    ),
)
async def signup_json(body: SignupRequest) -> TokenResponse:
    # Hash the password before saving
    hashed = hash_password(body.password)
    try:
        user_id = create_user(body.username, body.email, hashed)
    except psycopg2.IntegrityError as e:
        # Check if it's a unique constraint violation for username or email
        error_msg = str(e).lower()
        if "username" in error_msg:
            detail = "Username already registered"
        elif "email" in error_msg:
            detail = "Email already registered"
        else:
            detail = "Integrity error: user could not be created"
            
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}",
        )

    # Automatically log the user in after signup
    token = create_access_token(user_id=user_id)
    return TokenResponse(access_token=token)
