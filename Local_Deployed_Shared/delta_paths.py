from __future__ import annotations

import os
from pathlib import Path


SHARED_DIR = Path(__file__).resolve().parent
REPO_DIR = SHARED_DIR.parent
THIS_DIR_ONLY = REPO_DIR / "This-Directory-Only"
CSV_DIR = THIS_DIR_ONLY / "csv files of problems"


def get_chatgpt_code_dir() -> Path:
    configured = os.environ.get("DELTA_CHATGPT_CODE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (SHARED_DIR / "chatgpt").resolve()


def get_chatgpt_runtime_dir() -> Path:
    configured = os.environ.get("DELTA_CHATGPT_RUNTIME_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (THIS_DIR_ONLY / "chatgpt").resolve()


def get_backend_python() -> Path:
    return THIS_DIR_ONLY / "backend" / ".venv" / "bin" / "python3"


def torch_importable() -> bool:
    """Cheap: resolves the spec without importing torch (which costs ~1 s)."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def ensure_torch_python(*, required: bool = True) -> None:
    """Put the running SCRIPT on an interpreter that has torch.

    Every content check that executes drill or lesson code needs torch — the
    bank is 100% PyTorch dialect — but the venv that carries it lives under
    This-Directory-Only/backend/.venv and bare `python3` does not have it.
    Run under bare python the checks did not fail, they went quiet:
    audit_question_bank.py filed 1111 of 1243 questions as `torch_unavailable`
    ("question unscanned"), export_questions_json.py kept stale
    expected_output values, validate_lessons.py called every torch fence
    broken. A check that cannot run the code is not a check.

    So: if torch resolves here, do nothing. Otherwise re-exec this same
    script (same argv) under the backend venv. If that venv is missing, or
    we are already inside it and torch still does not resolve, say so on
    stderr and exit (`required=True`) or carry on (`required=False`).

    Call it from `main()` / the `__main__` block ONLY. `os.execv` replaces the
    whole process, and a watcher that merely *imports* the script (Modulario
    runs watch.py under system python) must not be swapped out from under
    its runner — lessons/watch.py already subprocesses into the venv for
    exactly that reason.
    """
    import sys

    if torch_importable():
        return
    backend = get_backend_python()
    script = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    # The guard breaks the loop when the venv itself has lost torch: without
    # it a broken venv re-execs into itself forever.
    already = os.environ.get("DELTA_TORCH_REEXEC") == "1"
    if backend.exists() and script is not None and script.is_file() and not already:
        os.environ["DELTA_TORCH_REEXEC"] = "1"
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(str(backend), [str(backend), str(script), *sys.argv[1:]])
    where = f"{backend} (venv python, still no torch)" if already else (
        f"{backend} (backend venv not found)" if not backend.exists() else sys.executable)
    msg = (
        f"torch is not importable from {where}; this check executes torch code "
        "and cannot run without it. Create the backend venv "
        "(This-Directory-Only/backend/.venv) or run under an interpreter that has torch."
    )
    if required:
        sys.exit(msg)
    print(f"WARNING: {msg}", file=sys.stderr)
