#!/usr/bin/env python3
"""compile_web_notebooks.py — the Colab notebooks, shaped for the web app.

WHY THIS EXISTS
---------------
The default (non-Colab) edition renders a lesson as a notebook now, against the
persistent kernel from `app/kernel_runner.py`. That notebook has to be the SAME
notebook the Colab edition publishes — same prose, same runnable fences, same
four-cell problems, same `dd_check` cases — or a learner who reads one and a
learner who reads the other are on two different courses.

The only way to guarantee that is to not compile it twice. This script calls
`generate_colab_notebooks.build_notebook`, the function that already decides
which cells a lesson turns into and in what order, and rewrites its output into
the shape a browser wants. Nothing about content, ordering or ids is decided
here; if a cell is wrong it is wrong in both editions, which is the point.

WHAT IS DIFFERENT FROM THE .ipynb
    - nbformat scaffolding is dropped: no `metadata`, no `execution_count`, no
      empty `outputs` list per cell. Those exist for Jupyter and cost ~30% of
      the bytes over the wire.
    - Each cell gains a `role` and, where it has one, the problem it belongs
      to. The extension derives both from the cell id in JS
      (`content/colab_focus.js` groups by the number in `dd-q<n>`); deriving
      them once here means the web renderer reads a field instead of keeping a
      second copy of the id grammar, and `scripts/watch.py` can check it.

INPUTS (read-only)
    Local_Deployed_Shared/lessons/lessons_structured.json
    Local_Deployed_Shared/questions_structured.json

OUTPUT
    Local_Deployed_Shared/lessons/notebooks/<lesson-id>.json   one per lesson
    Local_Deployed_Shared/lessons/notebooks/manifest.json      the index

Usage:
    python3 scripts/compile_web_notebooks.py [--out DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Importable from anywhere: the sibling modules are found relative to THIS
# file, not to whatever directory the deploy script happens to be in.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_colab_notebooks import build_notebook  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LESSONS = REPO / "Local_Deployed_Shared" / "lessons" / "lessons_structured.json"
QUESTIONS = REPO / "Local_Deployed_Shared" / "questions_structured.json"
DEFAULT_OUT = REPO / "Local_Deployed_Shared" / "lessons" / "notebooks"

# The id grammar `colab_cells.py` mints. Kept as one table so the roles the web
# renderer switches on are stated once, next to the compiler that mints them.
_PROBLEM = re.compile(r"^dd-q(\d+)")
_SOLUTION = re.compile(r"^dd-q\d+-solution$")
_CHECK = re.compile(r"^dd-q\d+-check$")
_HINTS = re.compile(r"^dd-q\d+-hints$")
_ANCHOR = re.compile(r"^dd-q\d+$")


def cell_role(cell: dict) -> str:
    """What the renderer has to do with this cell, from its id alone.

    `setup` and `checker` are the two cells the notebook opens with and neither
    is content: setup names the lesson for the extension and has nothing to run,
    and the checker is an 80 KB base64 payload that must run before any
    `dd_check` and must never be shown as if it were a lesson.

    `solution` is the one role with a rule attached rather than a look: it is
    the answer, and it stays unreadable until the learner asks. `check` is the
    cell whose printed verdict is the only thing that reaches the engine.

    `hints` is called out because its source is a raw `<details>` block. The
    web renderer escapes HTML — deliberately, it renders authored markdown from
    a repo — so a hints cell handed to it as prose would put the literal text
    `<details>` on screen. Naming the role here lets the renderer unwrap it
    into a real disclosure element and render only the markdown inside.
    """
    cid = cell["id"]
    if cid == "dd-setup":
        return "setup"
    if cid == "dd-checker":
        return "checker"
    if _SOLUTION.match(cid):
        return "solution"
    if _CHECK.match(cid):
        return "check"
    if _HINTS.match(cid):
        return "hints"
    if _ANCHOR.match(cid):
        return "problem"
    return "code" if cell["cell_type"] == "code" else "prose"


def web_cell(cell: dict) -> dict:
    """One notebook cell, as little of it as the browser needs.

    `q` is the problem this cell belongs to, not the problem it names. Every
    cell in a problem's group carries it — header, hints, editor, checker,
    solution, and the worked example anchored `dd-q<n>-worked` above it — which
    is what lets the renderer draw the group as one unit and hide a solution
    with its own problem rather than with a heuristic about position.
    """
    out = {
        "t": "code" if cell["cell_type"] == "code" else "md",
        "id": cell["id"],
        "role": cell_role(cell),
        "src": cell["source"],
    }
    m = _PROBLEM.match(cell["id"])
    if m:
        out["q"] = int(m.group(1))
    return out


def web_notebook(nb: dict, lesson: dict) -> dict:
    dd = nb["metadata"]["delta_drills"]
    return {
        "id": lesson["id"],
        "title": lesson["title"],
        "topic": lesson.get("topic") or "",
        "subtopic_key": dd.get("subtopic_key") or "",
        # `<kc>#<concept_id>` -> the anchor of the cell that opens that concept.
        # Read off the built notebook, so it describes anchors that exist.
        "segments": dict(dd.get("segments") or {}),
        "cells": [web_cell(c) for c in nb["cells"]],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    lessons = json.loads(LESSONS.read_text(encoding="utf-8"))["lessons"]
    bank = {q["id"]: q for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))}

    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    total_cells = 0
    total_problems = 0
    for lesson in lessons:
        web = web_notebook(build_notebook(lesson, bank), lesson)
        problems = sorted({c["q"] for c in web["cells"] if c["role"] == "problem"})
        total_cells += len(web["cells"])
        total_problems += len(problems)
        path = args.out / f"{lesson['id']}.json"
        if not args.dry_run:
            path.write_text(
                json.dumps(web, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        size = len(json.dumps(web, ensure_ascii=False, separators=(",", ":")))
        print(
            f"  {lesson['id']:<6} {len(web['cells']):>4} cells  "
            f"{len(problems):>3} problems  {size / 1024:>6.0f} KB  {path.name}"
        )
        manifest.append(
            {
                "id": lesson["id"],
                "title": lesson["title"],
                "topic": web["topic"],
                "subtopic_key": web["subtopic_key"],
                "file": path.name,
                "cells": len(web["cells"]),
                # The problems the learner can reach IN THIS NOTEBOOK, read off
                # the emitted anchors rather than off the lesson record — same
                # reason `generate_colab_notebooks.build_index` does it that way.
                "questions": problems,
                "kcs": [kp["kc"] for kp in lesson.get("kps") or []],
                "segments": web["segments"],
            }
        )

    if not args.dry_run:
        (args.out / "manifest.json").write_text(
            json.dumps({"lessons": manifest}, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        # A notebook that is renamed or retired must STOP being reachable. The
        # view opens `<id>.json` on a deep link without consulting the manifest,
        # and the deploy mirrors this folder wholesale, so a leftover file is a
        # retired lesson still being served -- and nothing would ever say so.
        keep = {entry["file"] for entry in manifest} | {"manifest.json"}
        # 🔴 THIS SWEEP OWNS THE LESSON NOTEBOOKS AND NOTHING ELSE. Since
        # 2026-09-01 the folder also holds `arena-*.json` — the ARENA
        # curriculum's notebooks, compiled from upstream .ipynb by
        # scripts/compile_arena_notebooks.py and listed in their own index.
        # They are not in this manifest and never will be, so a sweep over a
        # bare `*.json` glob would DELETE all 31 of them on the next deploy,
        # and the only symptom would be every Courses section saying "no
        # notebook for …". Each compiler sweeps its own prefix.
        for stale in sorted(args.out.glob("*.json")):
            if stale.name.startswith("arena-"):
                continue
            if stale.name not in keep:
                stale.unlink()
                print(f"  removed stale notebook: {stale.name}")
    print(f"\n{len(manifest)} notebooks · {total_cells} cells · {total_problems} problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
