"""POST /api/conceptual/chat — one streamed assistant turn.

The client is stateless and replays the visible thread on every turn, like
the practice tutor does. The server owns everything that must not be the
client's call: the system prompt, which model and door (provider.py), the
trim, and the spend limits (limits.py).

Wire format is Server-Sent Events, one JSON object per `data:` line:

    data: {"delta": "partial text"}     zero or more
    data: {"error": "human message"}    at most one, then the stream ends
    data: [DONE]                        always last

A refusal that can be decided before the model is called (bad thread, limit
reached) is a plain HTTP error instead, so the client can tell "you can't send
this" from "the model broke mid-answer".
"""

from __future__ import annotations

import json
import logging
from typing import Iterator, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.auth import get_current_user
from app.conceptual.limits import check_and_spend
from app.conceptual.prompts import build_system_prompt
from app.conceptual.provider import resolve_provider
from app.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conceptual", tags=["conceptual"])

MAX_TURNS = 24
MAX_CHARS = 8000

# Bounds on what a VALID body may hold: an oversized thread is a 422 before
# any quota is spent or model is called — the trims below only cut what is
# SENT. Generous against the client (24 turns, answers a few thousand chars).
# 🔴 NOT a memory guard: FastAPI reads and decodes the whole body before
# pydantic sees it. A byte cap belongs in front of the app (ASGI middleware
# or the proxy), and would cover every route, not just this one.
MAX_BODY_MESSAGES = 64
MAX_BODY_CHARS = 16000


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_BODY_CHARS)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list, max_length=MAX_BODY_MESSAGES)
    concept: str | None = Field(default=None, max_length=200)


def _sse(obj: object) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _stream(messages: list[dict], user_id: str) -> Iterator[str]:
    try:
        provider = resolve_provider()
        stream = provider.client.chat.completions.create(
            model=provider.model,
            messages=messages,
            stream=True,
        )
    except Exception as exc:  # no key, proxy down, model refused
        logger.warning("conceptual chat open failed for %s: %s", user_id, exc)
        yield _sse({"error": "The tutor is unavailable right now. Try again in a minute."})
        yield "data: [DONE]\n\n"
        return
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                yield _sse({"delta": text})
    except Exception as exc:
        logger.warning("conceptual chat stream broke for %s: %s", user_id, exc)
        yield _sse({"error": "The answer was cut off. Send your message again."})
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    yield "data: [DONE]\n\n"


@router.post("/chat")
def conceptual_chat(payload: ChatRequest, user: User = Depends(get_current_user)) -> StreamingResponse:
    thread = [m for m in payload.messages if m.content.strip()][-MAX_TURNS:]
    if not thread or thread[-1].role != "user":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The last message must be from you.")
    refusal = check_and_spend(str(user.id))
    if refusal:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, refusal)
    messages = [{"role": "system", "content": build_system_prompt(payload.concept)}] + [
        {"role": m.role, "content": m.content[:MAX_CHARS]} for m in thread
    ]
    return StreamingResponse(
        _stream(messages, str(user.id)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
