"""Which model answers the conceptual chat, and through which door.

Two doors, picked by config and nothing else, so moving between them is a
Fly secret rather than a code change:

  CONCEPTUAL_CHAT_BASE_URL set    -> an openai-oauth proxy (Seth's ChatGPT
                                     subscription). The proxy speaks the
                                     OpenAI Chat Completions API, so the same
                                     `openai` client drives it; the key is
                                     ignored by the proxy but the client
                                     refuses to start without one.
  CONCEPTUAL_CHAT_BASE_URL unset  -> the OpenAI API with the server key that
                                     practice/chatgpt_helpers.py already
                                     resolves.

🔴 NO SAMPLING CONTROLS. The ChatGPT/Codex backend behind openai-oauth rejects
`temperature` / `top_p`, so the request never sends them on either door; a
door that behaves differently per provider is the bug this module exists to
prevent.

🔴 THE PROXY HAS NO AUTH. Point BASE_URL only at a private address (Fly's
`.internal` network or 127.0.0.1), never at a public one: anyone who can reach
it spends the subscription.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI

from app.config import settings
from app.practice.chatgpt_helpers import load_chatgpt_api_key

# 🔴 The proxy only serves the Codex models Seth's PLAN lists at /v1/models
# (2026-09-25: gpt-6-astra/sol/luna, gpt-5.6-sol/terra/luna, gpt-5.5), and the
# list moves. A retired id fails every message; override with
# CONCEPTUAL_CHAT_MODEL rather than editing this.
PROXY_DEFAULT_MODEL = "gpt-6-luna"
API_DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class ChatProvider:
    client: OpenAI
    model: str
    name: str  # "openai-oauth" | "openai-api" — reported to the client, never a secret


# One client per door, reused: each OpenAI() owns an HTTP connection pool, and
# building one per message throws the pool (and its keep-alive) away. Keyed on
# the config so a changed secret still takes effect.
@lru_cache(maxsize=4)
def _client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI(api_key=api_key)


def resolve_provider() -> ChatProvider:
    base_url = (settings.conceptual_chat_base_url or "").strip()
    override = (settings.conceptual_chat_model or "").strip()
    if base_url:
        return ChatProvider(
            client=_client(base_url, "openai-oauth"),
            model=override or PROXY_DEFAULT_MODEL,
            name="openai-oauth",
        )
    return ChatProvider(
        client=_client("", load_chatgpt_api_key(None)),
        model=override or API_DEFAULT_MODEL,
        name="openai-api",
    )
