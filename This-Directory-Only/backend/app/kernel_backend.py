"""
Which notebook kernel implementation this process uses.

  modal  — `modal_kernel`: one Modal sandbox per learner, real IPython, magics
           and rich output. Chosen whenever Modal credentials are present.
  fork   — `kernel_runner`: a hardened fork of the API process. The fallback
           for a box with no Modal token (local dev without one, or Modal
           down at import time).

`KERNEL_BACKEND` (`auto` | `modal` | `fork`) forces the choice. The two modules
share one public surface — `run_cell`, `reset_kernel`, `kernel_status`,
`shutdown_all`, `DEFAULT_TIMEOUT_SECONDS` — so the router imports from here
and never learns which it got.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


def _modal_credentialed() -> bool:
    if settings.modal_token_id and settings.modal_token_secret:
        # pydantic loaded them from .env; the modal client reads os.environ.
        os.environ.setdefault("MODAL_TOKEN_ID", settings.modal_token_id)
        os.environ.setdefault("MODAL_TOKEN_SECRET", settings.modal_token_secret)
        return True
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True
    return (Path.home() / ".modal.toml").exists()


def _choose():
    wanted = (settings.kernel_backend or "auto").lower()
    if wanted in ("auto", "modal") and _modal_credentialed():
        try:
            import modal  # noqa: F401
            from app import modal_kernel
            logger.info("notebook kernels: modal backend (%s)", modal_kernel.APP_NAME)
            return modal_kernel, "modal"
        except ImportError as exc:
            if wanted == "modal":
                raise
            logger.warning("notebook kernels: modal requested but unavailable (%s); using fork", exc)
    elif wanted == "modal":
        raise RuntimeError("KERNEL_BACKEND=modal but no Modal credentials "
                           "(MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or ~/.modal.toml)")
    from app import kernel_runner
    logger.info("notebook kernels: fork backend")
    return kernel_runner, "fork"


backend, BACKEND_NAME = _choose()

DEFAULT_TIMEOUT_SECONDS = backend.DEFAULT_TIMEOUT_SECONDS
run_cell = backend.run_cell
reset_kernel = backend.reset_kernel
kernel_status = backend.kernel_status
shutdown_all = backend.shutdown_all


def startup() -> None:
    """Per-backend housekeeping at process start."""
    sweep = getattr(backend, "sweep_orphans", None)
    if sweep:
        sweep()
