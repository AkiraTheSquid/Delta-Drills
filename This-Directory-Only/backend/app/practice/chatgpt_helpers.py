"""
Shared OpenAI helpers used by the practice routers.

Two-step strategy mirrors ChatGPT.py: try the Responses API first,
then fall back to Chat Completions. API key resolution checks
(in order) the user record, the user_settings table, the
OPENAI_API_KEY env var, then the global settings.
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models import User


def load_chatgpt_api_key(user: Optional[User], db: Optional[Session] = None) -> str:
    """Load the OpenAI API key for the given user."""
    if user is not None and user.openai_api_key:
        return user.openai_api_key
    if user is not None and db is not None:
        try:
            row = db.execute(
                text("SELECT openai_api_key FROM user_settings WHERE user_email = :email"),
                {"email": user.email},
            ).fetchone()
            if row and row[0]:
                return row[0]
        except Exception:
            pass
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    from app.config import settings
    if settings.openai_api_key:
        return settings.openai_api_key
    raise ValueError("No OpenAI API key available.")


def call_chatgpt(
    prompt: str,
    model: str,
    user: Optional[User] = None,
    db: Optional[Session] = None,
) -> str:
    """Call OpenAI: Responses API first, Chat Completions as fallback."""
    api_key = load_chatgpt_api_key(user, db)
    client = OpenAI(api_key=api_key)
    try:
        resp = client.responses.create(model=model, input=prompt, temperature=1)
        answer = getattr(resp, "output_text", None) or ""
        if not answer:
            first_output = getattr(resp, "output", None)
            if isinstance(first_output, list) and first_output:
                first_content = getattr(first_output[0], "content", None)
                if isinstance(first_content, list) and first_content:
                    maybe_text = getattr(first_content[0], "text", None)
                    if isinstance(maybe_text, str):
                        answer = maybe_text
        return answer
    except Exception:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
        )
        return completion.choices[0].message.content or "" if completion.choices else ""
