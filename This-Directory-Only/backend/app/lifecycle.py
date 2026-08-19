"""
Process lifecycle hooks.

main.py is the thin assembly layer (its watch.py caps it at 60 LOC), so the
startup/shutdown work lives here and is attached in one call. Everything in
here is about the PROCESS — schema, warm torch, forked kernels — never about a
request; endpoint logic belongs in a *_router.py.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def register_lifecycle(app: FastAPI) -> None:
    """Attach every startup/shutdown handler to the app."""

    @app.on_event("startup")
    def _ensure_schema() -> None:
        """Create tables if they don't exist (idempotent). Needed when pointing at a
        fresh Postgres (e.g. Neon) that wasn't provisioned out-of-band like the old
        Supabase DB. Safe no-op when the schema already exists."""
        try:
            from app.db import engine, Base
            import app.models  # noqa: F401 — register table metadata
            Base.metadata.create_all(bind=engine)
            logger.info("Schema ensured (create_all) on %s",
                        engine.url.render_as_string(hide_password=True))
        except Exception as exc:  # never block startup on this
            logger.warning("Schema bootstrap skipped: %s", exc)

    @app.on_event("startup")
    def _preload_torch_runner() -> None:
        """Preimport torch so the fork runner grades torch drills in-process
        (milliseconds per run instead of a doomed cold import), and so notebook
        kernels forked later inherit it through copy-on-write. Non-fatal when
        torch is absent — those drills fall back to Colab routing."""
        from app.code_runner import preload_torch
        preload_torch()

    @app.on_event("shutdown")
    def _shutdown_notebook_kernels() -> None:
        """Stop every live notebook kernel. They are forked children of THIS
        process, so a worker that exits without killing them leaves orphans
        holding a share of a 2gb box."""
        from app.kernel_runner import shutdown_all
        shutdown_all()
