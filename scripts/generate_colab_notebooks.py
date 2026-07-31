#!/usr/bin/env python3
"""generate_colab_notebooks.py — compile the authored lessons into Colab notebooks.

WHY THIS EXISTS
---------------
The tutor is moving its student-facing surface from the web app to Google Colab.
The engine does not move: prerequisite gating, BKT + FIRe, decay-driven
resurfacing and the expertise-reversal ladder all stay in the backend, and the
Chrome side panel becomes a second client of the same endpoints. What has to
move is the *content* — the 63 authored knowledge points and the 424 problems
attached to them have to exist as cells in a notebook the panel can navigate to.

INPUTS (read-only — this script never writes into `lessons/`)
    Local_Deployed_Shared/lessons/lessons_structured.json   compiled lessons
    Local_Deployed_Shared/questions_structured.json         the 499-question bank

OUTPUT
    One notebook per lesson, nbformat 4.5, into --out.
    A notebook index, into --index, that the side panel loads as a plain script.

WHY THE INDEX EXISTS
--------------------
The tutor selects weakest-first across ALL subtopics and each lesson is its own
subtopic, so two consecutive problems routinely live in two different notebooks.
`/next-question` names a subtopic and a KC but not a file, so something has to
map `question_id -> lesson -> notebook`. That map is a property of how these
notebooks were compiled, which means it belongs here, next to the compiler, and
not hand-maintained in the extension.

TWO THINGS THAT ARE EASY TO GET WRONG
-------------------------------------
1. **Stable cell ids.** All 458 ARENA_5.0 notebooks are nbformat 4.2 with no
   `metadata.id`, so Colab mints fresh DOM ids on every load and `#scrollTo=`
   anchors are worthless there. We emit 4.5 with a deterministic id per cell
   (`dd-q<question_id>`, `dd-kp-<kc>`), which is what lets the panel jump
   straight to a problem. The id is ALSO written into the markdown body as an
   HTML comment, so navigation still works if Colab ever drops the attribute.

2. **Which fences become runnable cells.** The authoring contract
   (`lessons/README.md`) is "a fence gets a Run button iff CI runs it": a plain
   ```python fence is executed by the validator and must therefore render as a
   real code cell, while ```python no-run is illustrative and must stay inside
   the prose. Collapsing that distinction would put un-runnable snippets in
   front of a student as if they were exercises.

Usage:
    python3 scripts/generate_colab_notebooks.py [--out DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parents[1]
LESSONS = REPO / "Local_Deployed_Shared" / "lessons" / "lessons_structured.json"
QUESTIONS = REPO / "Local_Deployed_Shared" / "questions_structured.json"
DEFAULT_OUT = REPO / "arena-book-colab" / "ARENA_5.0" / "ch-1-foundations"
DEFAULT_INDEX = REPO / "extension" / "panel" / "notebook-index.js"
# The web app's copy of the same map. Practice sends the learner to Colab to
# run each problem, so the app needs question -> notebook for every question it
# can serve; without it the "open the right notebook" step has no data.
DEFAULT_WEB_INDEX = (
    REPO / "Local_Deployed_Shared" / "lessons" / "colab_notebooks.json"
)

BACKEND = "https://delta-drills-backend.fly.dev"

# nbformat 4.5 cell ids: 1-64 chars of [a-zA-Z0-9-_].
_ID_SAFE = re.compile(r"[^a-zA-Z0-9\-_]")

# A fenced block. Group 1 is the info string, group 2 the body. Only a bare
# `python` info string becomes a code cell; `python no-run`, `text`, etc. stay
# in the markdown.
_FENCE = re.compile(r"^```([^\n]*)\n(.*?)^```[ \t]*$", re.S | re.M)


# Anchors the side panel navigates to, as opposed to the cells beneath them.
_ANCHOR = re.compile(r"^dd-q\d+$")

MAX_ID = 64


def slug(text: str) -> str:
    return _ID_SAFE.sub("-", text)[:MAX_ID].strip("-")


# ── cell constructors ────────────────────────────────────────────────────────


def md_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {"id": cell_id},
        "source": source.rstrip("\n") + "\n",
    }


def code_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {"id": cell_id},
        "execution_count": None,
        "outputs": [],
        "source": source.rstrip("\n") + "\n",
    }


class IdMinter:
    """Deterministic, collision-free cell ids.

    Deterministic because a regenerated notebook must keep the anchors the
    panel navigates to; collision-free because nbformat rejects duplicates and
    two segments can legitimately share a title.
    """

    def __init__(self) -> None:
        self._issued: set[str] = set()

    def __call__(self, base: str) -> str:
        base = slug(base) or "cell"
        if base not in self._issued:
            self._issued.add(base)
            return base
        # Counting per base is not enough. Ids are capped at 64 chars, so two
        # DIFFERENT long bases ("…-concept" and "…-concept-code") can truncate
        # onto the same string — the collision has to be checked against what
        # was actually issued, not against a per-base tally.
        n = 1
        while True:
            suffix = f"-{n}"
            cand = f"{base[: MAX_ID - len(suffix)].rstrip('-')}{suffix}"
            if cand not in self._issued:
                self._issued.add(cand)
                return cand
            n += 1


def split_markdown(source: str, mint: IdMinter, base: str) -> Iterator[dict]:
    """Yield cells for a markdown blob, promoting runnable fences to code cells.

    Everything that is not a bare ```python fence stays markdown, including
    ```python no-run — see the module docstring.
    """
    if not source or not source.strip():
        return
    pos = 0
    for m in _FENCE.finditer(source):
        info = m.group(1).strip()
        runnable = info == "python"
        if not runnable:
            continue  # leave the fence embedded in the surrounding prose
        before = source[pos : m.start()]
        if before.strip():
            yield md_cell(before, mint(base))
        yield code_cell(m.group(2), mint(f"{base}-code"))
        pos = m.end()
    tail = source[pos:]
    if tail.strip():
        yield md_cell(tail, mint(base))


# ── problem rendering ────────────────────────────────────────────────────────


def problem_cells(
    qid: int,
    rung: str,
    bank: dict,
    mint: IdMinter,
    *,
    prompt: str | None = None,
    starter: str | None = None,
    hints: str | None = None,
) -> list[dict]:
    """The cells for one problem, at one rung.

    `dd-q<id>` is the anchor the side panel jumps to, and it goes on the header
    cell rather than the code cell so the student lands on the question text
    with the editor just below it.
    """
    q = bank.get(qid, {})
    ex = q.get("exercise", {})
    text = prompt or ex.get("question_text") or f"Question {qid}"
    code = starter if starter is not None else (ex.get("starter_code") or "")

    header = (
        f"<!-- dd:dd-q{qid} -->\n\n"
        f"### Problem {qid} · {rung}\n\n"
        f"{text.strip()}\n"
    )
    cells = [md_cell(header, f"dd-q{qid}")]

    if hints:
        cells.append(
            md_cell(
                "<details>\n<summary>Hints</summary>\n\n"
                f"{hints.strip()}\n\n</details>\n",
                mint(f"dd-q{qid}-hints"),
            )
        )

    cells.append(code_cell(code or "# your code here\n", mint(f"dd-q{qid}-code")))
    return cells


# ── notebook assembly ────────────────────────────────────────────────────────


def setup_cell(lesson: dict, mint: IdMinter) -> dict:
    return code_cell(
        "# === Delta Drills ===\n"
        "# The side panel decides what you practise next. This cell only tells\n"
        "# the notebook who you are, for the completion beacon.\n"
        'DD_TOKEN = ""  # paste from the extension\'s Settings if you want beacons\n'
        f'DD_BACKEND_URL = "{BACKEND}"\n'
        f'DD_LESSON_ID = "{lesson["id"]}"\n',
        mint("dd-setup"),
    )


def build_notebook(lesson: dict, bank: dict) -> dict:
    mint = IdMinter()
    cells: list[dict] = [
        md_cell(
            # The panel asks an open tab which notebook it is (`dd:identify`)
            # before deciding whether the next problem needs a switch. It reads
            # this marker, so it has to survive Colab dropping the cell id.
            f"<!-- dd:dd-lesson-{lesson['id']} -->\n\n"
            f"# {lesson['title']}\n\n"
            f"*{lesson['topic']} · `{lesson['id']}`*\n\n"
            "Work through this with the **Delta Drills** side panel open. "
            "It picks what you practise, sends you to the cell, and records how "
            "it went — you do not need to read this notebook in order.\n",
            mint(f"dd-lesson-{lesson['id']}"),
        ),
        setup_cell(lesson, mint),
    ]

    for kp in lesson.get("kps") or []:
        kc = kp["kc"]
        cells.append(
            md_cell(
                f"<!-- dd:dd-kp-{slug(kc)} -->\n\n"
                f"## {kp['title']}\n\n"
                f"`{kc}`\n",
                f"dd-kp-{slug(kc)}",
            )
        )

        # KP-level concept prose comes first when the KP was authored without
        # segments; segmented KPs carry their prose per segment instead.
        segments = kp.get("segments") or []
        if not segments:
            cells += list(split_markdown(kp.get("concept_markdown"), mint, f"{slug(kc)}-concept"))
            cells += list(
                split_markdown(kp.get("worked_example_markdown"), mint, f"{slug(kc)}-worked")
            )

        for seg in segments:
            base = slug(f"{kc}-{seg.get('title') or 'segment'}")
            if seg.get("title"):
                cells.append(md_cell(f"### {seg['title']}\n", mint(base)))
            cells += list(split_markdown(seg.get("concept_markdown"), mint, f"{base}-concept"))
            if seg.get("watch_out_markdown"):
                cells += list(
                    split_markdown(
                        f"> **Watch out.** {seg['watch_out_markdown'].strip()}",
                        mint,
                        f"{base}-watch",
                    )
                )
            cells += list(
                split_markdown(seg.get("worked_example_markdown"), mint, f"{base}-worked")
            )
            if seg.get("worked_example_code"):
                cells.append(code_cell(seg["worked_example_code"], mint(f"{base}-worked-code")))

            # Faded rung: the authored starter carries the `_____` blanks, which
            # is a hand-cut scaffold for this concept and beats the mechanical
            # backward fade the server can generate for anything else.
            for item in seg.get("faded_items") or []:
                cells += problem_cells(
                    item["question_id"],
                    "faded",
                    bank,
                    mint,
                    prompt=item.get("prompt"),
                    starter=item.get("starter_code"),
                )

        for item in kp.get("guided_items") or []:
            cells += problem_cells(
                item["question_id"],
                "guided",
                bank,
                mint,
                hints=item.get("hints_markdown"),
            )

        for qid in kp.get("independent_items") or []:
            cells += problem_cells(qid, "independent", bank, mint)

        if kp.get("misconceptions_markdown"):
            cells += list(
                split_markdown(
                    f"#### Common mistakes\n\n{kp['misconceptions_markdown']}",
                    mint,
                    f"{slug(kc)}-misconceptions",
                )
            )

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "delta_drills": {"lesson_id": lesson["id"], "subtopic_key": lesson.get("subtopic_key")},
        },
        "cells": cells,
    }


# ── validation ───────────────────────────────────────────────────────────────


def validate(nb: dict, lesson_id: str) -> list[str]:
    """Structural checks that do not need nbformat installed.

    Duplicate or malformed ids are the failure mode that matters: nbformat
    rejects them outright, and a duplicate `dd-q<id>` would silently send the
    panel to the wrong problem.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for i, c in enumerate(nb["cells"]):
        cid = c.get("id", "")
        if not cid or _ID_SAFE.search(cid) or len(cid) > 64:
            problems.append(f"{lesson_id} cell {i}: bad id {cid!r}")
        if cid in seen:
            problems.append(f"{lesson_id} cell {i}: duplicate id {cid!r}")
        seen.add(cid)
        if c["cell_type"] not in ("markdown", "code"):
            problems.append(f"{lesson_id} cell {i}: bad type {c['cell_type']!r}")
    return problems


# ── the notebook index ───────────────────────────────────────────────────────


def build_index(records: list[dict], out_dir: Path) -> dict:
    """The `question_id -> notebook` map, as a plain dict.

    Two consumers serialize this differently, which is why the dict is built
    once here and written twice by `main`:

    * the Chrome side panel loads it with a `<script>` tag (`--index`), because
      MV3's CSP forbids remote script but a bundled file is fine, and a script
      tag avoids putting a second HTTP client next to `api.js`;
    * the web app fetches it as JSON (`--web-index`), because it has no bundler
      and every other data file it reads arrives the same way.

    Both must come from this function. The web app opens the Colab notebook the
    learner is expected to run the problem in, so a map that disagrees with the
    notebooks on disk sends them to the wrong cell.

    Three lookup keys, because `/next-question` does not always give the same
    one. `questions` is exact and covers every authored problem;
    `subtopics` catches a question the bank added after this ran; `kcs` is what
    the lesson gate resolves through, since a gate names a KC and not a
    question.
    """
    try:
        rel = out_dir.resolve().relative_to(REPO).as_posix()
    except ValueError:
        # --out pointed outside the repo; a repo-relative path is meaningless.
        rel = ""

    index = {
        "dir": rel,
        "lessons": [
            {
                "id": r["id"],
                "title": r["title"],
                "topic": r["topic"],
                "subtopic_key": r["subtopic_key"],
                "file": r["file"],
                "problems": len(r["questions"]),
            }
            for r in records
        ],
        "questions": {str(q): r["id"] for r in records for q in r["questions"]},
        "subtopics": {r["subtopic_key"]: r["id"] for r in records if r["subtopic_key"]},
        "kcs": {kc: r["id"] for r in records for kc in r["kcs"]},
        # The concept sections, by the anchor that actually reaches them.
        #
        # The panel's lesson gate sends the learner to the KP section instead
        # of rendering the concept itself, so it needs the same `dd-kp-<slug>`
        # id this script writes onto the header cell. The slug is computed
        # HERE and shipped, rather than re-derived in JavaScript, because a
        # second implementation of `slug()` that drifted by one character
        # would produce an anchor Colab silently ignores — the learner would
        # land at the top of a 500-cell notebook with no error anywhere.
        #
        # Anchor ONLY. Which notebook a concept is in is already `kcs` above,
        # keyed identically; carrying the lesson id here too would be the same
        # fact written twice, and the copy that went stale would be the one
        # nothing checks.
        "kps": {kc: f"dd-kp-{slug(kc)}" for r in records for kc in r["kcs"]},
    }
    return index


def index_as_script(index: dict) -> str:
    """The extension's copy — a classic script that assigns a global."""
    return (
        "/* GENERATED by scripts/generate_colab_notebooks.py — do not edit.\n"
        "   Regenerate after changing lessons; the panel navigates by this map. */\n"
        "window.DD_NOTEBOOKS = "
        + json.dumps(index, indent=2, ensure_ascii=False)
        + ";\n"
    )


def index_as_json(index: dict) -> str:
    """The web app's copy — fetched, so it carries its provenance inline."""
    return json.dumps(
        {
            "_generated_by": "scripts/generate_colab_notebooks.py — do not edit by hand",
            **index,
        },
        indent=2,
        ensure_ascii=False,
    ) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--web-index", type=Path, default=DEFAULT_WEB_INDEX)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    lessons = json.loads(LESSONS.read_text())["lessons"]
    bank = {q["id"]: q for q in json.loads(QUESTIONS.read_text())}

    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)

    total_cells = 0
    total_problems = 0
    errors: list[str] = []
    records: list[dict] = []

    for lesson in lessons:
        nb = build_notebook(lesson, bank)
        errors += validate(nb, lesson["id"])
        anchors = [c["id"] for c in nb["cells"] if _ANCHOR.match(c["id"])]
        total_cells += len(nb["cells"])
        total_problems += len(anchors)

        path = args.out / f"{lesson['id']}-{slug(lesson['title']).lower()}.ipynb"
        if not args.dry_run:
            path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"  {lesson['id']:<6} {len(nb['cells']):>4} cells  {len(anchors):>3} problems  {path.name}")

        records.append(
            {
                "id": lesson["id"],
                "title": lesson["title"],
                "topic": lesson.get("topic") or "",
                "subtopic_key": lesson.get("subtopic_key") or "",
                "file": path.name,
                # Read back off the anchors rather than re-walking the lesson:
                # the index then describes what is actually navigable, which is
                # the only claim the panel relies on.
                "questions": sorted(int(a[len("dd-q"):]) for a in anchors),
                "kcs": [kp["kc"] for kp in lesson.get("kps") or []],
            }
        )

    print(f"\n{len(lessons)} notebooks · {total_cells} cells · {total_problems} problems")

    index = build_index(records, args.out)
    if not args.dry_run:
        args.index.parent.mkdir(parents=True, exist_ok=True)
        args.index.write_text(index_as_script(index))
        args.web_index.parent.mkdir(parents=True, exist_ok=True)
        args.web_index.write_text(index_as_json(index))
    print(
        f"index · {len(records)} lessons · {total_problems} questions mapped · "
        f"{args.index.name} + {args.web_index.name}"
    )

    # Two notebooks claiming the same question would send the panel to whichever
    # the index happened to write last, silently.
    claimed: dict[int, str] = {}
    for r in records:
        for q in r["questions"]:
            if q in claimed:
                errors.append(f"question {q} claimed by both {claimed[q]} and {r['id']}")
            claimed[q] = r["id"]

    # Every authored problem must be reachable, exactly once.
    expected = sum(
        len({i["question_id"] for s in (kp.get("segments") or []) for i in (s.get("faded_items") or [])})
        + len(kp.get("guided_items") or [])
        + len(kp.get("independent_items") or [])
        for l in lessons
        for kp in l.get("kps") or []
    )
    if total_problems != expected:
        errors.append(f"problem count {total_problems} != expected {expected}")

    if errors:
        print("\nFAILED:", file=sys.stderr)
        for e in errors[:20]:
            print(f"  {e}", file=sys.stderr)
        return 1
    print("OK — ids unique, every authored problem anchored exactly once.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
