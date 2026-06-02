from datetime import datetime, timedelta
from typing import Any
import json
import secrets
import urllib.request
import urllib.error
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import User


pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
security = HTTPBearer()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_access_token(subject: str) -> str:
    expires = datetime.utcnow() + timedelta(minutes=settings.access_token_ttl_minutes)
    payload: dict[str, Any] = {"sub": subject, "exp": expires}
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def verify_google_id_token(credential: str) -> dict[str, str]:
    """
    Verify a Google Identity Services ID token (the `credential` returned by the
    "Sign in with Google" button). Checks Google's signature, the audience
    (== our OAuth client id), the issuer, and that the email is verified.
    Returns {'sub': google-subject-id, 'email': lowercased email}.
    Raises 401 on any failure.
    """
    if not settings.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google sign-in is not configured on the server.",
        )
    # Imported lazily so the rest of the app loads even if google-auth is absent
    # in a stripped environment (e.g. a test that never touches Google login).
    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google auth library not installed on the server.",
        ) from exc

    try:
        info = google_id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            settings.google_client_id,
            clock_skew_in_seconds=10,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Google token"
        ) from exc

    iss = info.get("iss", "")
    if iss not in ("accounts.google.com", "https://accounts.google.com"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Google token issuer")

    email = (info.get("email") or "").lower().strip()
    sub = info.get("sub", "")
    if not email or not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Google token missing email")
    if info.get("email_verified") not in (True, "true"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Google email not verified")

    return {"sub": sub, "email": email}


def _decode_supabase_token(token: str) -> dict[str, Any]:
    """
    Validate a Supabase user JWT by calling the Supabase /auth/v1/user endpoint.
    Returns a dict with 'sub' (user UUID) and 'email'.
    """
    url = f"{settings.supabase_url}/auth/v1/user"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "apikey": settings.supabase_service_role_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    user_id = data.get("id", "")
    email = (data.get("email") or "").lower().strip()
    if not user_id or not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    return {"sub": user_id, "email": email}


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    token = creds.credentials
    payload: dict[str, Any] | None = None
    is_supabase = False
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        try:
            payload = _decode_supabase_token(token)
            is_supabase = True
        except HTTPException:
            raise

    user_id = payload.get("sub") if payload else None
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if user:
        return user

    if is_supabase:
        email = (payload.get("email") or "").lower().strip()
        if not email:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            return existing
        try:
            user_uuid = UUID(str(user_id))
        except ValueError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        temp_password = secrets.token_urlsafe(32)
        user = User(id=user_uuid, email=email, password_hash=hash_password(temp_password))
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
