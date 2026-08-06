#!/usr/bin/env python3
"""audit_colab_notebook_index — does every routed question exist in its notebook?

The Colab edition (delta-drills-colab.vercel.app) hides the editor, the aids and
the submit bar when it routes a drill to Colab. So a question routed to a
notebook that does not contain it is not a cosmetic miss — it is a dead end with
no controls and no way forward, and it looks like a working feature right up
until the learner scrolls the notebook looking for a problem that was never
there.

`Local_Deployed_Shared/lessons/colab_notebooks.json` claims a lesson for each
question id. This confirms the claim against the notebooks actually PUBLISHED on
GitHub — the file the learner's browser opens, not the one in the working tree,
because those are two different things the moment `publish_colab_notebooks.sh`
has not been re-run.

Why this can drift: `generate_colab_notebooks.py` writes the index and the
notebooks together, so they agree at generation time. Publishing is a separate
step. Regenerate without publishing and the index promises cells that are not
live yet.

Usage: python3 This-Directory-Only/scripts/audit_colab_notebook_index.py
Exit 0 = every mapped question has its `dd-q<id>` cell upstream.
"""
from __future__ import annotations

import collections
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
INDEX_PATH = REPO_DIR / "Local_Deployed_Shared" / "lessons" / "colab_notebooks.json"
# The owner the app falls back to when the student has set no GitHub username of
# their own (practice/colab_mode.js DEFAULT_OWNER). A fork of this repo is a copy
# of these notebooks, so upstream is the one that has to be right.
OWNER = "AkiraTheSquid"


def raw_url(repo: str, prefix: str, filename: str) -> str:
    path = "/".join(p for p in (prefix, filename) if p)
    return f"https://raw.githubusercontent.com/{OWNER}/{repo}/main/{path}"


def main() -> int:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    directory = index.get("dir", "")
    repo, _, prefix = directory.partition("/")
    lessons = {lesson["id"]: lesson for lesson in index.get("lessons", [])}

    by_lesson: dict[str, list[str]] = collections.defaultdict(list)
    for question_id, lesson_id in index.get("questions", {}).items():
        by_lesson[lesson_id].append(str(question_id))

    # The concept anchors, routed the same way. A KP that teaches several ideas
    # is taught one at a time and the teaching step opens `dd-seg-<kc>-<n>`;
    # published without that cell, Colab ignores the fragment in silence and
    # the learner lands at the top of a 650-cell notebook with no error to
    # explain it. Grouped by lesson through `kcs`, which is where the concept's
    # notebook is recorded.
    kc_lesson = index.get("kcs", {})
    segments_by_lesson: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for key, anchor in index.get("segments", {}).items():
        segments_by_lesson[kc_lesson.get(key.split("#", 1)[0], "")].append((key, anchor))

    problems: list[str] = []
    total = 0

    for lesson_id in sorted(by_lesson):
        question_ids = by_lesson[lesson_id]
        total += len(question_ids)
        lesson = lessons.get(lesson_id)
        if not lesson:
            problems.append(f"{lesson_id}: mapped by {len(question_ids)} questions but has no lesson entry")
            continue
        url = raw_url(repo, prefix, lesson["file"])
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                notebook = json.load(response)
        except (urllib.error.URLError, json.JSONDecodeError) as err:
            problems.append(f"{lesson_id}: {lesson['file']} unreachable upstream ({err})")
            continue
        cell_ids = {cell.get("id") for cell in notebook.get("cells", [])}
        missing = [q for q in question_ids if f"dd-q{q}" not in cell_ids]
        concepts = segments_by_lesson.get(lesson_id, [])
        lost = [key for key, anchor in concepts if anchor not in cell_ids]
        status = "ok" if not missing else f"MISSING {len(missing)}"
        print(f"{lesson_id}: {len(question_ids)} mapped, {len(concepts)} concepts, {status}")
        if missing:
            problems.append(f"{lesson_id}: no cell for question(s) {missing[:8]}")
        if lost:
            problems.append(f"{lesson_id}: no cell for concept(s) {lost[:8]}")

    print(f"--- {total} questions routed to {len(by_lesson)} notebooks")
    if problems:
        print("FAIL — the Colab edition would strand these:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print("Fix: rerun scripts/generate_colab_notebooks.py, then "
              "scripts/publish_colab_notebooks.sh.", file=sys.stderr)
        return 1
    print("PASS — every routed question has its cell in the published notebook")
    return 0


if __name__ == "__main__":
    sys.exit(main())
