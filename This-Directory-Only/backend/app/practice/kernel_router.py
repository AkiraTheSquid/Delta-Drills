"""
Persistent notebook kernel endpoints.

Endpoints (mounted under /api/practice by the parent router):
  POST /kernel/exec
  POST /kernel/reset
  GET  /kernel/status

`/run-code` (ai_router) stays exactly as it is: grading wants a fresh process
per submission, and nothing about a notebook session should be able to leak
into a graded run. These endpoints are the other half — a session that PERSISTS
between cells, which is what makes the default edition behave like Colab.

Authenticated. A kernel is a scarce resource on a 2gb box (four of them), so an
anonymous caller must not be able to reserve one. The session key is derived
from the signed-in user, never taken from the request: a client-chosen id would
let one learner attach to another learner's live namespace.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import get_current_user
from app.kernel_runner import (
    DEFAULT_TIMEOUT_SECONDS,
    kernel_status,
    reset_kernel,
    run_cell,
)
from app.models import User

router = APIRouter()

MAX_TIMEOUT_SECONDS = 60
MAX_CONTEXT_CHARS = 200


class KernelExecRequest(BaseModel):
    code: str
    # Installed once, on a kernel the server had to create for this call. The
    # cell harness (last-expression echo, `<cell N>` tracebacks) lives in
    # practice/notebook.js and is shipped here rather than duplicated server
    # side, so the two cannot drift.
    bootstrap: str = ""
    # Shown in tracebacks — the client sends `<cell 8>` so a line number means
    # the line in that cell.
    filename: str = "<cell>"
    # Which notebook these cells belong to. Switching notebooks restarts the
    # kernel instead of carrying the last lesson's names into this one.
    context: str = ""
    timeout: int = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1, le=MAX_TIMEOUT_SECONDS)
    # "If you had to build the kernel, run nothing and just tell me." The
    # client sends this for any cell but the first, because its answer to a
    # fresh kernel is to replay cells 1..N — and running the clicked cell here
    # as well would run it twice.
    skip_on_fresh: bool = False


class KernelExecResponse(BaseModel):
    stdout: str
    stderr: str
    success: bool
    # True when this call had to create the kernel. The client's cue that
    # whatever state it thought it had is gone — it was evicted, idled out, or
    # the box was redeployed — and that earlier cells need re-running.
    fresh: bool
    exec_count: int


def _session_id(user: User) -> str:
    return f"u{user.id}"


@router.post("/kernel/exec", response_model=KernelExecResponse)
def kernel_exec(payload: KernelExecRequest,
                user: User = Depends(get_current_user)) -> KernelExecResponse:
    """Run one cell in this learner's live session."""
    try:
        run = run_cell(
            _session_id(user),
            payload.code,
            bootstrap=payload.bootstrap,
            filename=payload.filename or "<cell>",
            context=(payload.context or "")[:MAX_CONTEXT_CHARS],
            timeout=payload.timeout,
            skip_on_fresh=payload.skip_on_fresh,
        )
    except RuntimeError as exc:
        # Busy kernel or a full box — both are "come back in a moment", not a
        # broken request. 409 so the client can retry rather than surface a
        # traceback to the learner.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return KernelExecResponse(
        stdout=run.result.stdout,
        stderr=run.result.stderr,
        success=run.result.success,
        fresh=run.fresh,
        exec_count=run.exec_count,
    )


@router.post("/kernel/reset")
def kernel_reset(user: User = Depends(get_current_user)) -> dict:
    """Restart this learner's kernel — the notebook's "Restart runtime"."""
    return {"restarted": reset_kernel(_session_id(user))}


@router.get("/kernel/status")
def kernel_status_endpoint(user: User = Depends(get_current_user)) -> dict:
    """This learner's own kernel, plus how full the box is.

    Scoped to the caller deliberately. Signed in is not privileged: the
    unscoped listing hands any learner every other learner's session id, the
    lesson they are on and how long they have been idle.
    """
    return kernel_status(_session_id(user))
