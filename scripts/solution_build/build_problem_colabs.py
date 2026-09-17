#!/usr/bin/env python3
"""Assemble per-question PROBLEM Colab notebooks for torch bank questions.

Torch drills can't run in the in-app sandbox (no `import torch` within the time
limit), so the app routes them to Colab. The learner needs the PROBLEM there —
the starter code, NO answer — kept SEPARATE from the solution notebook that
build_solution_colabs.py emits (that one shows the reference answer).

Outputs (additive — does NOT touch the existing q<ID>-<sub>.ipynb solutions):
  arena-procedural-drills/solutions/<topic-slug>/q<ID>-<sub>.problem.ipynb
  This-Directory-Only/backend/app/data/question_problem_notebooks.jsonl  (id -> path)

Only torch questions get a problem notebook — everything else runs in-app. The
starter is the EFFECTIVE starter: function_mode_overrides_*.jsonl merged over the
bank export, matching exactly what the backend serves (and already sanitized so
the TODO comments don't spell out the answer).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
QUESTIONS = {q["id"]: q for q in json.loads((HERE / "dd_questions.json").read_text())}
AUTHORED_DIR = HERE / "authored"
CHATGPT = REPO / "This-Directory-Only" / "chatgpt"
SOLUTIONS_DIR = REPO / "arena-procedural-drills" / "solutions"
DATA = REPO / "This-Directory-Only" / "backend" / "app" / "data"

# Same precedence questions.py uses when layering starter_code overrides.
OVERRIDE_FILES = [
    "function_mode_overrides.jsonl",
    "function_mode_overrides_round2.jsonl",
    "function_mode_overrides_round3.jsonl",
]

PIP = {
    "torch": "%pip install -q numpy torch --index-url https://download.pytorch.org/whl/cpu",
    "einops": "%pip install -q numpy einops torch --index-url https://download.pytorch.org/whl/cpu",
}


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return s[:48] or "x"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def _effective_starter(qid: int, q: dict) -> str:
    """Bank starter with function-mode overrides layered on (same as the API)."""
    starter = q.get("starter_code") or ""
    for fn in OVERRIDE_FILES:
        path = CHATGPT / fn
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("id") == qid and rec.get("starter_code"):
                starter = rec["starter_code"]
    return starter


def _is_torch(q: dict) -> bool:
    if q.get("primary_library") == "torch":
        return True
    blob = f"{q.get('answer_code', '')}\n{q.get('starter_code', '')}"
    return bool(re.search(r"(?m)^\s*(?:import\s+torch\b|from\s+torch[\s.])", blob))


def build_problem(q: dict, starter: str, hint: str) -> dict:
    pip = PIP.get(q.get("primary_library", "torch"), PIP["torch"])
    sub = q.get("subtopic") or ""
    title = sub if sub.lower().startswith((q.get("topic") or "").lower()) else f"{q.get('topic')}: {sub}"
    cells = [
        md(f"# {title}\n\n"
           f"**Practice notebook — Delta Drills #{q['id']}**\n\n"
           "This drill uses PyTorch, which the Delta Drills in-app runner can't execute. "
           "Work it here, run your code, then return to Delta Drills and rate yourself. "
           "The full worked answer is in the separate *solution* notebook (Show solution in the app).\n"),
        md("## Problem\n\n" + (q.get("question_text") or "").strip() + "\n"),
        md("<details><summary>💡 Hint (click to reveal)</summary>\n\n"
           + (hint or "_No hint._").strip() + "\n\n</details>\n"),
        code(pip),
        md("## Your solution\n\nEdit the cell below and run it. Print exactly what the problem asks.\n"),
        code((starter or "# Write your solution here\n").rstrip() + "\n"),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def main() -> None:
    hints: dict[int, str] = {}
    for f in sorted(AUTHORED_DIR.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                if obj.get("hint"):
                    hints[obj["id"]] = obj["hint"]

    DATA.mkdir(parents=True, exist_ok=True)
    manifest, skipped = [], []
    for qid, q in QUESTIONS.items():
        if not _is_torch(q):
            continue
        starter = _effective_starter(qid, q)
        topic_dir = SOLUTIONS_DIR / slug(q["topic"])
        topic_dir.mkdir(parents=True, exist_ok=True)
        nb_path = topic_dir / f"q{qid}-{slug(q['subtopic'])}.problem.ipynb"
        nb_path.write_text(json.dumps(build_problem(q, starter, hints.get(qid, "")), indent=1))
        manifest.append({"id": qid, "path": nb_path.relative_to(REPO).as_posix()})

    manifest.sort(key=lambda m: m["id"])
    (DATA / "question_problem_notebooks.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest) + "\n")
    print(f"torch problem notebooks: {len(manifest)}")
    print(f"manifest: {(DATA / 'question_problem_notebooks.jsonl').relative_to(REPO)}")


if __name__ == "__main__":
    main()
