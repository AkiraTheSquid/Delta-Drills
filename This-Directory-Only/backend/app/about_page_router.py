"""Public About-page content, with one server-enforced editor account.

The frontend is static, so the published copy lives on Fly's mounted
``USER_DATA_DIR`` rather than in the deployed image.  Reading is public;
writing requires both a valid app JWT and the configured editor email.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import bleach
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.adaptive import DATA_DIR
from app.auth import get_current_user
from app.models import User


router = APIRouter(prefix="/site-content", tags=["site-content"])

EDITOR_EMAIL = "sethbgibson@gmail.com"
CONTENT_FILE = DATA_DIR / "about-page.json"
MAX_HTML_LENGTH = 250_000

# Keep the app's existing About-page structure working after an edit while
# refusing scripts, inline event handlers, styles, and unsafe URL schemes.
# The page has been the AISC write-up since 2026-09-24, which adds a section
# nav (header/nav), a comparison table, a definition list and a footer. Its
# live figures (svg, input, output) stay disallowed: the client restores those
# from the static page instead of trusting saved markup for them.
ALLOWED_TAGS = [
    "a", "article", "b", "blockquote", "br", "button", "code", "dd", "details",
    "div", "dl", "dt", "em", "figcaption", "figure", "footer", "h1", "h2", "h3",
    "h4", "header", "hr", "i", "img", "kbd", "li", "main", "mark", "nav", "ol",
    "p", "pre", "section", "small", "span", "strong", "sub", "summary", "sup",
    "table", "tbody", "td", "th", "thead", "tr", "ul",
]


def _allowed_attribute(tag: str, name: str, value: str) -> bool:
    if name in {"class", "id", "role", "title", "open", "hidden", "type", "disabled"}:
        return True
    if name.startswith(("aria-", "data-")):
        return True
    if tag == "a" and name in {"href", "target", "rel"}:
        return True
    if tag == "img" and name in {"src", "alt", "width", "height", "loading"}:
        return True
    return False


class AboutPageUpdate(BaseModel):
    html: str = Field(min_length=1, max_length=MAX_HTML_LENGTH)


class AboutPageContent(BaseModel):
    html: str | None = None
    updated_at: str | None = None
    updated_by: str | None = None


def _clean_html(html: str) -> str:
    return bleach.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=_allowed_attribute,
        protocols={"http", "https", "mailto"},
        strip=True,
        strip_comments=True,
    ).strip()


def _read_content() -> AboutPageContent:
    if not CONTENT_FILE.exists():
        return AboutPageContent()
    try:
        raw = json.loads(CONTENT_FILE.read_text(encoding="utf-8"))
        return AboutPageContent(**raw)
    except (OSError, json.JSONDecodeError, ValueError):
        # A malformed on-disk override must never make the public About page
        # disappear; the static HTML remains its fallback.
        return AboutPageContent()


def _write_content(content: AboutPageContent) -> None:
    CONTENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=CONTENT_FILE.parent, delete=False
    ) as tmp:
        json.dump(content.model_dump(), tmp, ensure_ascii=False)
        tmp.write("\n")
        temp_path = Path(tmp.name)
    try:
        os.replace(temp_path, CONTENT_FILE)
    finally:
        temp_path.unlink(missing_ok=True)


def _require_editor(user: User) -> None:
    if (user.email or "").strip().lower() != EDITOR_EMAIL:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the About-page editor may update this content.",
        )


@router.get("/about", response_model=AboutPageContent)
def get_about_page() -> AboutPageContent:
    """Return the saved override, or ``html: null`` for the static fallback."""
    return _read_content()


@router.put("/about", response_model=AboutPageContent)
def update_about_page(
    payload: AboutPageUpdate,
    user: User = Depends(get_current_user),
) -> AboutPageContent:
    _require_editor(user)
    html = _clean_html(payload.html)
    if not html:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="About page cannot be empty.")
    content = AboutPageContent(
        html=html,
        updated_at=datetime.now(timezone.utc).isoformat(),
        updated_by=EDITOR_EMAIL,
    )
    _write_content(content)
    return content
