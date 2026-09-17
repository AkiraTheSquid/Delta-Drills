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
# The re-dialected layer from retorch_authored.py, NOT the May-2026 numpy
# authoring in authored/. Run that script first; this one refuses without it.
AUTHORED_LAYER = HERE / "authored_torch" / "layer.jsonl"
SOLUTIONS_DIR = REPO / "arena-procedural-drills" / "solutions"
DATA = REPO / "This-Directory-Only" / "backend" / "app" / "data"

TORCH_PIP = "%pip install -q numpy torch --index-url https://download.pytorch.org/whl/cpu"
EINOPS_PIP = "%pip install -q numpy einops torch --index-url https://download.pytorch.org/whl/cpu"

EINOPS_IMPORT = re.compile(r"^\s*(?:import\s+einops\b|from\s+einops[\s.])", re.M)

# `numpy` is still installed alongside torch on purpose: the graders and the
# `np.load('/delta_numbers.npy')` fixture loader use it. The learner never
# writes it.

TOPIC_DISPLAY_LABELS = {"Numpy": "PyTorch tensors"}
"""Mirror of practice/config.js. The bank files every question's mastery under
the subtopic key `f"{topic}: {subtopic}"`, so the stored topic must stay
"Numpy" — renaming it orphans every learner's BKT posterior and EWMA. The
rename is a LABEL applied at display, and a solution notebook IS a display
surface, which is why it needs its own copy of the map."""


def pip_for(*sources: str) -> str:
    """The install line, DERIVED from what the solution actually imports.

    This used to key on `primary_library`, which the backend infers before the
    torch-dialect overrides are layered on — it still reports "numpy" for 437
    of 499 rows whose code is torch. Keying on it is precisely how every one of
    these notebooks ended up telling the learner to `%pip install -q numpy` for
    a torch exercise.

    Torch is the floor rather than a third branch because the July conversion
    left the bank single-dialect: `retorch_authored.py` reports zero rows whose
    answer is not torch. Should a non-torch question ever be added, it gets a
    harmlessly large install, not a wrong one — and that report is where it
    would show up first.
    """
    blob = "\n".join(s or "" for s in sources)
    return EINOPS_PIP if EINOPS_IMPORT.search(blob) else TORCH_PIP


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return s[:48] or "x"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def display_title(q: dict) -> str:
    """The heading a learner reads — relabelled, never the stored topic key."""
    topic = q.get("topic") or ""
    sub = q.get("subtopic") or ""
    label = TOPIC_DISPLAY_LABELS.get(topic, topic)
    # The loader already prefixes the subtopic with its topic, so `sub` normally
    # arrives as "Numpy: Core array literacy". Swap that prefix for the label
    # rather than concatenating, or the heading reads "PyTorch tensors: Numpy:".
    if topic and sub.lower().startswith(topic.lower() + ":"):
        return label + sub[len(topic):]
    return f"{label}: {sub}" if label and sub else (sub or label)


def build_notebook(q: dict, sol: dict) -> dict:
    solution = sol.get("solution_code") or ""
    pip = pip_for(solution, q.get("starter_code"))
    cells = [
        md(f"# {display_title(q)}\n\n"
           f"**Solution notebook — Delta Drills #{q['id']}**\n\n"
           "Run the cells top-to-bottom to see the reference answer execute.\n"),
        md("## Problem\n\n" + (q.get("question_text") or "").strip() + "\n"),
    ]
    hint = (sol.get("hint") or "").strip()
    if hint:
        cells.append(md("<details><summary>💡 Hint (click to reveal)</summary>\n\n"
                        + hint + "\n\n</details>\n"))
    cells.append(code(pip))
    cells.append(md("## Reference solution\n"))
    cells.append(code(solution.rstrip() + "\n"))
    # An explanation is omitted rather than stubbed when retorch_authored.py
    # withheld it: the authored prose described the pre-conversion numpy
    # algorithm, and a renamed version of it would be confidently wrong. The
    # problem, hint and runnable reference answer still stand on their own.
    explanation = (sol.get("explanation") or "").strip()
    if explanation:
        cells.append(md("## Why this works\n\n" + explanation + "\n"))
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


NB_ID = re.compile(r"^q(\d+)-")


def prune_orphans(built: dict[int, Path]) -> tuple[list[str], list[str]]:
    """Delete solution notebooks that no live question can reach.

    Keyed on whether the QUESTION still exists, never on whether this run
    happened to build it. Those are different things, and conflating them is
    destructive: a question with no authored prose yet produces no notebook
    this run, and pruning on "not built" would delete the perfectly good
    notebook it already had. Coverage gaps must not cascade into deletions.

    Two things are removed:
      retired  the question is gone from the bank — the July conversion
               retired the whole `numpy.structured-dtypes` KC and q65
               (`ndarray.flags.writeable`; torch has no read-only tensor).
               Left on disk these stay reachable by URL, and stay numpy.
      renamed  the question survives but its subtopic slug changed, so this
               run wrote a different filename and the old one would linger
               as a stale duplicate.

    `*.problem.ipynb` siblings belong to build_problem_colabs.py — left alone.
    """
    retired, renamed = [], []
    for path in sorted(SOLUTIONS_DIR.rglob("q*.ipynb")):
        if path.name.endswith(".problem.ipynb"):
            continue
        m = NB_ID.match(path.name)
        if not m:
            continue
        qid = int(m.group(1))
        rel = path.relative_to(REPO).as_posix()
        if qid not in QUESTIONS:
            path.unlink()
            retired.append(rel)
        elif qid in built and path != built[qid]:
            path.unlink()
            renamed.append(rel)
    return retired, renamed


def main() -> None:
    if not AUTHORED_LAYER.exists():
        raise SystemExit(
            f"missing {AUTHORED_LAYER.relative_to(REPO)} — "
            "run scripts/solution_build/retorch_authored.py first"
        )
    authored: dict[int, dict] = {}
    for line in AUTHORED_LAYER.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            authored[obj["id"]] = obj

    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    manifest, hints, missing, unexplained = [], [], [], []
    built: dict[int, Path] = {}

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
        built[qid] = nb_path
        rel = nb_path.relative_to(REPO).as_posix()
        manifest.append({"id": qid, "path": rel})
        if sol.get("hint"):
            hints.append({"id": qid, "hint": sol["hint"]})
        if sol.get("needs_authoring"):
            unexplained.append(qid)

    retired, renamed = prune_orphans(built)

    (DATA / "question_solution_notebooks.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest) + "\n")
    (DATA / "question_hints.jsonl").write_text(
        "\n".join(json.dumps(h) for h in hints) + "\n")

    print(f"notebooks: {len(manifest)}  hints: {len(hints)}")
    print(f"pruned (question retired): {len(retired)}")
    for r in retired:
        print(f"   - {r}")
    if renamed:
        print(f"pruned (subtopic slug changed): {len(renamed)}")
        for r in renamed:
            print(f"   - {r}")
    print(f"no 'Why this works' yet (withheld, needs authoring): {len(unexplained)}")
    # Not an error and NOT a reason to delete anything: these questions are in
    # the bank but have no authored prose, so they get no notebook this run and
    # keep whatever they already had.
    print(f"no authoring layer row (left untouched): {len(missing)}")


if __name__ == "__main__":
    main()
