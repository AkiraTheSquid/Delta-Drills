"""Running the repo's own content pipeline.

Nothing here reimplements a check. Every entry shells out to the script the
repo already gates its deploys on, under the backend venv interpreter —
`python3` on this machine has no torch, and without torch validate_lessons.py
reports every torch drill broken and export_questions_json.py leaves
`expected_output` stale instead of recomputing it.

`check_all` is the order a content change has to survive: validate the pages,
compile them, rebuild the Q-matrix, re-export the bank, then audit the bank.
"""

from __future__ import annotations

import subprocess
import time

from . import paths

DEFAULT_TIMEOUT = 900


def _run(argv: list[str], cwd=None, timeout: int = DEFAULT_TIMEOUT) -> dict:
    started = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd or paths.REPO),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "command": " ".join(argv), "timed_out": True,
                "seconds": round(time.time() - started, 1)}
    except FileNotFoundError as err:
        return {"ok": False, "command": " ".join(argv), "error": str(err)}
    return {
        "ok": proc.returncode == 0,
        "command": " ".join(argv),
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 1),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _script(path, *args: str) -> dict:
    return _run([paths.python_for_content(), str(path), *args])


STEPS = {
    "validate": lambda: _script(paths.VALIDATE_SCRIPT, "--coverage"),
    "compile": lambda: _script(paths.COMPILE_SCRIPT),
    "qmatrix": lambda: _script(paths.QMATRIX_SCRIPT),
    "export": lambda: _script(paths.EXPORT_SCRIPT),
    "audit_bank": lambda: _script(paths.BANK_AUDIT, "--gate"),
    "notebooks": lambda: _script(paths.WEB_NOTEBOOKS_SCRIPT),
}

AUDITS = {
    "symbol_coverage": "audit_symbol_coverage.py",
    "solution_prereqs": "audit_solution_prereqs.py",
    "arena_grounding": "audit_arena_grounding.py",
    "graph_structure": "audit_graph_structure.py",
    "lesson_syntax": "audit_lesson_syntax.py",
    "question_syntax": "audit_question_syntax.py",
    "prose_prereqs": "audit_prose_prereqs.py",
    "ladder_pairing": "audit_ladder_pairing.py",
}


def step(name: str) -> dict:
    if name not in STEPS:
        raise KeyError(f"Unknown step '{name}'. Known: {', '.join(STEPS)}")
    return STEPS[name]()


def audit(name: str, extra_args: list[str] | None = None) -> dict:
    if name not in AUDITS:
        raise KeyError(f"Unknown audit '{name}'. Known: {', '.join(AUDITS)}")
    return _script(paths.SCRIPTS / AUDITS[name], *(extra_args or []))


def check_all(fast: bool = False) -> dict:
    """The whole gate, in order, stopping at the first failure.

    `fast` skips the two slow steps (fence execution and the bank export) —
    useful while iterating on registry structure, never sufficient to ship.
    """
    order = ["compile", "qmatrix"] if fast else ["validate", "compile", "qmatrix", "export", "audit_bank"]
    results = []
    for name in order:
        result = step(name)
        results.append({"step": name, **result})
        if not result.get("ok"):
            return {"ok": False, "failed_at": name, "steps": results}
    return {"ok": True, "steps": results}


def folder_watchers(folders: list[str] | None = None) -> dict:
    """Run the Modulario watch.py health checks for the content folders.

    These carry the standing content guards (prerequisite order, ARENA
    grounding, symbol coverage), so a drill reaching for an untaught function
    is refused where it was written rather than found at deploy time.
    """
    targets = folders or [
        "Local_Deployed_Shared/lessons",
        "scripts",
        "Local_Deployed_Shared/pipeline",
    ]
    results = []
    for folder in targets:
        watcher = paths.REPO / folder / "watch.py"
        if not watcher.exists():
            results.append({"folder": folder, "ok": True, "skipped": "no watch.py"})
            continue
        results.append({"folder": folder, **_run([paths.python_for_content(), str(watcher)])})
    return {"ok": all(r.get("ok", True) for r in results), "watchers": results}


def python_in_use() -> dict:
    return {
        "interpreter": paths.python_for_content(),
        "is_backend_venv": paths.BACKEND_PY.exists(),
        "warning": None if paths.BACKEND_PY.exists() else (
            "Backend venv not found — running under bare python3. Torch drills will "
            "validate as broken and expected_output will not be recomputed."
        ),
    }
