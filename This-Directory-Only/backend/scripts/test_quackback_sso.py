#!/usr/bin/env python3
"""POST /api/quackback/sso signs a real learner into the feedback widget, and
nobody else.

Run: .venv/bin/python scripts/test_quackback_sso.py   (from backend/)

Quackback only trusts an HS256 ssoToken signed with the widget secret. The
route must (1) sign `{sub, email, iat, exp}` for a signed-in learner, (2) give a
guest account nothing — its random address would become a feedback author
nobody can reach, (3) say `ok: False` instead of failing when the secret is
unset, and (4) refuse a caller with no valid token at all.
"""
import os
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

os.environ["USER_DATA_DIR"] = tempfile.mkdtemp(prefix="quackback_sso_test_")
os.environ.setdefault("KERNEL_BACKEND", "fork")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jwt  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import auth  # noqa: E402
from app.main import app  # noqa: E402

SECRET = "wgt_test_" + "0" * 56
client = TestClient(app)


def as_user(email):
    user = SimpleNamespace(id=uuid.uuid4(), email=email)
    app.dependency_overrides[auth.get_current_user] = lambda: user
    return user


def main():
    os.environ["QUACKBACK_SIGNING_SECRET"] = SECRET

    user = as_user("Learner@Example.com")
    body = client.post("/api/quackback/sso").json()
    assert body["ok"] is True, body
    claims = jwt.decode(body["ssoToken"], SECRET, algorithms=["HS256"])
    assert claims["sub"] == str(user.id), claims
    assert claims["email"] == "learner@example.com", claims
    assert claims["exp"] - claims["iat"] == 300, claims
    try:
        jwt.decode(body["ssoToken"], "wrong-secret", algorithms=["HS256"])
        raise AssertionError("token verified under the wrong secret")
    except jwt.InvalidSignatureError:
        pass

    as_user("guest-ab12cd34@guest.delta-drills.app")
    body = client.post("/api/quackback/sso").json()
    assert body["ok"] is False and "ssoToken" not in body, body

    as_user("learner@example.com")
    del os.environ["QUACKBACK_SIGNING_SECRET"]
    body = client.post("/api/quackback/sso").json()
    assert body == {"ok": False, "error": "Feedback sign-in is not configured."}, body

    app.dependency_overrides.clear()
    status = client.post("/api/quackback/sso").status_code
    assert status in (401, 403), status
    status = client.post("/api/quackback/sso", headers={"Authorization": "Bearer nope"}).status_code
    assert status == 401, status

    print("PASS test_quackback_sso: learner signed, guest refused, unset secret ok:false, no token 401/403")


if __name__ == "__main__":
    main()
