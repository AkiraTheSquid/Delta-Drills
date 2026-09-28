"""Signs the signed-in learner into the Quackback feedback widget (the
bottom-right bubble in ``Local_Deployed_Shared/feedback-widget/``).

Delta Drills has its OWN self-hosted Quackback
(``https://drills.163-192-116-252.sslip.io``), separate from Delta Note's, so
its posts, roadmap and changelog never mix with Delta Note's.

Quackback accepts only a server-signed identity: an HS256 JWT made with the
widget signing secret, passed to the widget as ``identify({ ssoToken })``. The
browser cannot hold that secret, so this route mints the token for whoever
the app JWT (or Supabase token) says is asking.

Claims: ``sub`` = the Delta Drills user id (a uuid, stable across an email
change), ``email``, and a 5-minute ``exp`` so a leaked token cannot be
replayed later.

Guest accounts are real rows (``guest-…@guest.delta-drills.app``) with a
random address nobody reads; they get no token, so a guest stays signed out of
the widget and the board asks them to sign in before posting.

Secret (Fly ``delta-drills-backend``): ``QUACKBACK_SIGNING_SECRET``, from the
Drills Quackback admin → Settings → Widget → Verified identity only →
Regenerate. Unset → ``ok: False``; the widget stays signed out and nothing
else notices.
"""
from __future__ import annotations

import os
import time

import jwt
from fastapi import APIRouter, Depends

from app.auth import get_current_user
from app.models import User

TOKEN_TTL_SECONDS = 5 * 60
GUEST_EMAIL_DOMAIN = "guest.delta-drills.app"

router = APIRouter(prefix="/api/quackback", tags=["quackback"])


def signing_secret() -> str | None:
    """The widget signing secret, or None when it is not configured."""
    return os.getenv("QUACKBACK_SIGNING_SECRET", "").strip() or None


def is_guest_email(email: str) -> bool:
    return str(email or "").lower().strip().endswith("@" + GUEST_EMAIL_DOMAIN)


def mint_sso_token(secret: str, *, sub: str, email: str, now: float | None = None) -> str:
    """A Quackback widget ssoToken. Pure: the clock is a parameter."""
    at = int(now if now is not None else time.time())
    claims = {"sub": sub, "email": email, "iat": at, "exp": at + TOKEN_TTL_SECONDS}
    return jwt.encode(claims, secret, algorithm="HS256")


def sso_token(*, user_id: str, email: str, secret: str | None, now: float | None = None) -> dict:
    """What the widget needs to identify the caller: ``{ok, ssoToken}``."""
    if secret is None:
        return {"ok": False, "error": "Feedback sign-in is not configured."}
    email = str(email or "").lower().strip()
    if not user_id or not email:
        return {"ok": False, "error": "Your sign-in has no user id or email."}
    if is_guest_email(email):
        return {"ok": False, "error": "Guests post feedback after signing in."}
    return {"ok": True, "ssoToken": mint_sso_token(secret, sub=str(user_id), email=email, now=now)}


@router.post("/sso")
def quackback_sso(user: User = Depends(get_current_user)) -> dict:
    return sso_token(user_id=str(user.id), email=user.email, secret=signing_secret())
