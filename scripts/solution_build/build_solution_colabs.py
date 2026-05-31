#!/usr/bin/env python3
"""Assemble per-question solution Colab notebooks from validated authored JSONL.

Inputs:
  scripts/solution_build/dd_questions.json         (bank export)
  scripts/solution_build/authored/*.jsonl          (agent output: id, solution_code, explanation, hint)

Outputs (in repo so Colab can load from GitHub main):
  arena-procedural-drills/solutions/<topic-slug>/q<ID>-<subtopic-slug>.ipynb
  This-Directory-Only/backend/data/question_solution_notebooks.jsonl  (id -> repo-relative path)
  This-Directory-Only/backend/data/question_hints.jsonl               (id -> hint)

The notebook path is rooted at the "arena-procedural-drills/" prefix that
stats/predicted-links.js colabUpstreamHref() already routes to GitHub
AkiraTheSquid/Delta-Drills main — so the frontend needs zero new routing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
QUESTIONS = {q["id"]: q for q in json.loads((HERE / "dd_questions.json").read_text())}
AUTHORED_DIR = HERE / "authored"
SOLUTIONS_DIR = REPO / "arena-procedural-drills" / "solutions"
DATA = REPO / "This-Directory-Only" / "backend" / "app" / "data"

PIP = {
    "numpy": "%pip install -q numpy",
    "einops": "%pip install -q numpy einops",
    "einops.einsum": "%pip install -q numpy einops",
    "torch": "%pip install -q numpy torch --index-url https://download.pytorch.org/whl/cpu",
    "python": "",
}


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return s[:48] or "x"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def build_notebook(q: dict, sol: dict) -> dict:
    lib = q.get("primary_library", "python")
    pip = PIP.get(lib, "%pip install -q numpy")
    sub = q["subtopic"] or ""
    title = sub if sub.lower().startswith((q["topic"] or "").lower()) else f"{q['topic']}: {sub}"
    cells = [
        md(f"# {title}\n\n"
           f"**Solution notebook — Delta Drills #{q['id']}**\n\n"
           "Run the cells top-to-bottom to see the reference answer execute.\n"),
        md("## Problem\n\n" + (q.get("question_text") or "").strip() + "\n"),
        md("<details><summary>💡 Hint (click to reveal)</summary>\n\n"
           + (sol.get("hint") or "_No hint._").strip() + "\n\n</details>\n"),
    ]
    if pip:
        cells.append(code(pip))
    cells.append(md("## Reference solution\n"))
    cells.append(code(sol["solution_code"].rstrip() + "\n"))
    cells.append(md("## Why this works\n\n" + (sol.get("explanation") or "").strip() + "\n"))
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
    authored: dict[int, dict] = {}
    for f in sorted(AUTHORED_DIR.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                authored[obj["id"]] = obj

    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    manifest, hints, missing = [], [], []

    for qid, q in QUESTIONS.items():
        sol = authored.get(qid)
        if not sol or not sol.get("solution_code"):
            missing.append(qid)
            continue
        topic_dir = SOLUTIONS_DIR / slug(q["topic"])
        topic_dir.mkdir(parents=True, exist_ok=True)
        fname = f"q{qid}-{slug(q['subtopic'])}.ipynb"
        nb_path = topic_dir / fname
        nb_path.write_text(json.dumps(build_notebook(q, sol), indent=1))
        rel = nb_path.relative_to(REPO).as_posix()
        manifest.append({"id": qid, "path": rel})
        if sol.get("hint"):
            hints.append({"id": qid, "hint": sol["hint"]})

    (DATA / "question_solution_notebooks.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest) + "\n")
    (DATA / "question_hints.jsonl").write_text(
        "\n".join(json.dumps(h) for h in hints) + "\n")

    print(f"notebooks: {len(manifest)}  hints: {len(hints)}  missing: {len(missing)}")
    if missing:
        print("missing ids:", missing)


if __name__ == "__main__":
    main()
