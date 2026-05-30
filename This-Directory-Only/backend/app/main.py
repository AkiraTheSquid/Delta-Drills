from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth_router import router as auth_router
from app.chapters_router import router as chapters_router
from app.jobs_router import router as jobs_router
from app.practice import router as practice_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Split Tool Backend", version="0.1.0")


@app.on_event("startup")
def _ensure_schema() -> None:
    """Create tables if they don't exist (idempotent). Needed when pointing at a
    fresh Postgres (e.g. Neon) that wasn't provisioned out-of-band like the old
    Supabase DB. Safe no-op when the schema already exists."""
    try:
        from app.db import engine, Base
        import app.models  # noqa: F401 — register table metadata
        Base.metadata.create_all(bind=engine)
        logger.info("Schema ensured (create_all) on %s", engine.url.render_as_string(hide_password=True))
    except Exception as exc:  # never block startup on this
        logger.warning("Schema bootstrap skipped: %s", exc)


app.include_router(practice_router)
app.include_router(auth_router)
app.include_router(jobs_router)
app.include_router(chapters_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
