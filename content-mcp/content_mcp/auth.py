"""Password gate for content writes.

Reads are open: anyone pointed at this repo can list lessons, read a KP, or
search the drill bank. Anything that changes a file on disk goes through
`require()`, which needs a live session opened with the shared editing
password.

The password is stored as a salted SHA-256 digest in `content-mcp/auth.json`,
never in plaintext. Change it with `dd-content set-password`, or point
DELTA_DRILLS_CONTENT_PASSWORD_FILE at a different auth.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from pathlib import Path

from . import paths

SESSION_TTL_SECONDS = 12 * 60 * 60
_ITERATIONS = 200_000


class AuthError(RuntimeError):
    """Raised when a write is attempted without a live session."""


def _auth_file() -> Path:
    configured = os.environ.get("DELTA_DRILLS_CONTENT_PASSWORD_FILE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "auth.json"


def _session_file() -> Path:
    return paths.state_dir() / "session.json"


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _ITERATIONS
    ).hex()


def write_password(password: str, current: str | None = None) -> Path:
    """(Re)write the stored digest.

    Once a password exists, changing it requires proving you know the current
    one — otherwise the write gate is only as strong as "can you run the CLI",
    which is no gate at all. The first call (no digest on disk yet) bootstraps
    without a credential.
    """
    existing = _stored()
    if existing:
        if current is None:
            raise AuthError(
                "A password is already set. Pass the CURRENT password to change it."
            )
        if not check_password(current):
            raise AuthError("Current password is wrong — password not changed.")
    salt = secrets.token_hex(16)
    record = {
        "algorithm": "pbkdf2_hmac_sha256",
        "iterations": _ITERATIONS,
        "salt": salt,
        "digest": hash_password(password, salt),
        "note": "Shared editing password for Delta Drills content. Reads need no password.",
    }
    target = _auth_file()
    target.write_text(json.dumps(record, indent=2) + "\n")
    return target


def _stored() -> dict | None:
    target = _auth_file()
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def check_password(password: str) -> bool:
    record = _stored()
    if not record:
        raise AuthError(
            f"No password is configured. Run: dd-content set-password  (writes {_auth_file()})"
        )
    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        (password or "").encode("utf-8"),
        record["salt"].encode("utf-8"),
        int(record.get("iterations", _ITERATIONS)),
    ).hex()
    return secrets.compare_digest(candidate, record["digest"])


def login(password: str) -> dict:
    if not check_password(password):
        raise AuthError("Wrong content password.")
    token = secrets.token_urlsafe(24)
    expires = time.time() + SESSION_TTL_SECONDS
    target = _session_file()
    # Created 0600 by open(), not chmod-ed afterwards: write_text would create
    # the file under the process umask first, exposing a bearer token to any
    # local user for the moment in between.
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(json.dumps({"token": token, "expires": expires}))
    return {"token": token, "expires_epoch": expires, "expires_in_hours": SESSION_TTL_SECONDS / 3600}


def logout() -> bool:
    target = _session_file()
    if target.exists():
        target.unlink()
        return True
    return False


def _live_session() -> dict | None:
    target = _session_file()
    if not target.exists():
        return None
    try:
        record = json.loads(target.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if float(record.get("expires", 0)) < time.time():
        return None
    return record


def status() -> dict:
    session = _live_session()
    env_password = os.environ.get("DELTA_DRILLS_CONTENT_PASSWORD", "")
    env_ok = bool(env_password) and check_password(env_password)
    return {
        "authenticated": bool(session) or env_ok,
        "via": "env" if (env_ok and not session) else ("session" if session else None),
        "expires_in_minutes": (
            round((session["expires"] - time.time()) / 60) if session else None
        ),
        "password_configured": _stored() is not None,
        "auth_file": str(_auth_file()),
    }


def require(token: str | None = None) -> None:
    """Gate for every mutating op. Raises AuthError when not logged in.

    Three ways in, in order of precedence: an explicit token argument (so one
    MCP call can authenticate itself), a live session file (the normal case,
    opened by `content_login`), or DELTA_DRILLS_CONTENT_PASSWORD in the
    environment (for CI and non-interactive scripts).
    """
    if token:
        session = _live_session()
        if session and secrets.compare_digest(token, session["token"]):
            return
        if check_password(token):  # a caller may pass the password itself
            return
        raise AuthError("Token is not valid or has expired. Run content_login again.")
    if _live_session():
        return
    env_password = os.environ.get("DELTA_DRILLS_CONTENT_PASSWORD", "")
    if env_password and check_password(env_password):
        return
    raise AuthError(
        "Content editing is password protected. Call content_login "
        "(MCP) or `dd-content login` (CLI) first."
    )
