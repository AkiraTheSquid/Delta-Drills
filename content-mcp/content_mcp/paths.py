"""Where the Delta Drills content lives.

Single source of repo layout for the content MCP. Every other module in this
package imports paths from here rather than recomputing them, so moving a
content file is a one-line change instead of a grep.
"""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    configured = os.environ.get("DELTA_DRILLS_REPO", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    # content-mcp/content_mcp/paths.py -> content-mcp -> repo
    return Path(__file__).resolve().parents[2]


REPO = _repo_root()

SHARED = REPO / "Local_Deployed_Shared"
LESSONS = SHARED / "lessons"
NOTES = LESSONS / "notes"
KC_REGISTRY = LESSONS / "kc_registry.json"
QMATRIX = LESSONS / "qmatrix_tags.json"
LESSONS_STRUCTURED = LESSONS / "lessons_structured.json"
AUTHORING = LESSONS / "AUTHORING.md"
GLOSSARY = LESSONS / "glossary.js"

QUESTIONS_STRUCTURED = SHARED / "questions_structured.json"
QUESTIONS_FLAT = SHARED / "questions.json"

PIPELINE = SHARED / "pipeline"
RETIRED_IDS = PIPELINE / "retired_question_ids.json"
EXPORT_SCRIPT = PIPELINE / "export_questions_json.py"
BANK_AUDIT = PIPELINE / "audit_question_bank.py"

SCRIPTS = REPO / "scripts"
VALIDATE_SCRIPT = SCRIPTS / "validate_lessons.py"
COMPILE_SCRIPT = SCRIPTS / "compile_lessons.py"
QMATRIX_SCRIPT = SCRIPTS / "build_qmatrix.py"
WEB_NOTEBOOKS_SCRIPT = SCRIPTS / "compile_web_notebooks.py"

THIS_ONLY = REPO / "This-Directory-Only"
CSV_DIR = THIS_ONLY / "csv files of problems"
CURATED_CSV = CSV_DIR / "curated_additions.csv"
CHATGPT = THIS_ONLY / "chatgpt"
CURATED_OVERRIDES = CHATGPT / "curated_overrides.jsonl"

BACKEND_PY = THIS_ONLY / "backend" / ".venv" / "bin" / "python3"

# Courses = every lessons/ subdirectory that actually holds KP markdown.
def courses() -> list[str]:
    if not LESSONS.is_dir():
        return []
    found = []
    for child in sorted(LESSONS.iterdir()):
        if child.is_dir() and any(child.glob("kp-*.md")):
            found.append(child.name)
    return found


def kp_files() -> list[Path]:
    out: list[Path] = []
    for course in courses():
        out.extend(sorted((LESSONS / course).glob("kp-*.md")))
    return out


def state_dir() -> Path:
    configured = os.environ.get("DELTA_DRILLS_CONTENT_STATE", "").strip()
    root = Path(configured).expanduser().resolve() if configured else REPO / ".content-mcp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def python_for_content() -> str:
    """The interpreter the content pipeline must run under.

    Bare python3 has no torch: validate_lessons.py then reports every torch
    drill broken and export_questions_json.py leaves expected_output stale.
    """
    if BACKEND_PY.exists():
        return str(BACKEND_PY)
    return os.environ.get("DELTA_DRILLS_PYTHON", "python3")


# Everything a content author can break, and therefore everything the daily
# snapshot has to carry. Each entry is (repo-relative path, include globs or
# None for "everything under it"). Secrets and build junk are excluded by
# BACKUP_EXCLUDE below — the chatgpt/ folder holds an api_key.txt beside the
# override layers, and a backup is not a place to copy credentials into.
CONTENT_PATHS: list[tuple[str, list[str] | None]] = [
    ("Local_Deployed_Shared/lessons", None),
    ("Local_Deployed_Shared/questions.json", None),
    ("Local_Deployed_Shared/questions_structured.json", None),
    ("Local_Deployed_Shared/pipeline/retired_question_ids.json", None),
    ("This-Directory-Only/csv files of problems", ["*.csv"]),
    ("This-Directory-Only/chatgpt", ["*.jsonl", "function_mode_*.json"]),
]

BACKUP_EXCLUDE = {"__pycache__", ".venv", "api_key.txt", ".git", "node_modules"}


def json_indent_of(path, default: int = 2) -> int:
    """The indent width a JSON file already uses.

    Rewriting `kc_registry.json` (indent 1) with the json module's default
    turned a one-node addition into a 414-line diff — unreviewable, and a
    guaranteed conflict with any other session touching the file.
    """
    try:
        for line in path.read_text().splitlines()[:40]:
            stripped = line.lstrip(" ")
            if stripped and stripped != line:
                return len(line) - len(stripped)
    except OSError:
        pass
    return default
