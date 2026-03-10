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
    return (THIS_DIR_ONLY / "chatgpt").resolve()


def get_chatgpt_runtime_dir() -> Path:
    configured = os.environ.get("DELTA_CHATGPT_RUNTIME_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return get_chatgpt_code_dir()


def get_backend_python() -> Path:
    return THIS_DIR_ONLY / "backend" / ".venv" / "bin" / "python3"
