from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from uuid import UUID

from app.config import settings
from app.db import get_db
from app.models import User
import secrets


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


@lru_cache(maxsize=1)
def _supabase_jwk_client() -> jwt.PyJWKClient:
    jwks_url = f"{settings.supabase_url}/auth/v1/keys"
    return jwt.PyJWKClient(jwks_url)


def _decode_supabase_token(token: str) -> dict[str, Any]:
    try:
        signing_key = _supabase_jwk_client().get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    issuer = payload.get("iss")
    expected_issuer = f"{settings.supabase_url}/auth/v1"
    if issuer != expected_issuer:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return payload


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
