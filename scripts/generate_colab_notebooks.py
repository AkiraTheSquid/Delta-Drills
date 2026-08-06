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

WHAT IS NOT HERE
----------------
What a CELL is — id minting, the four-cell shape of a problem, the checker —
lives in `colab_cells.py`, together with the two invariants that bite (stable
ids, and which fences become runnable cells). This file is the compiler: it
decides which cells a lesson turns into, in what order, and writes them out.

Usage:
    python3 scripts/generate_colab_notebooks.py [--out DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from colab_cells import (
    IdMinter,
    checker_cell,
    code_cell,
    id_is_valid,
    md_cell,
    problem_cells,
    segment_anchor,
    setup_cell,
    slug,
    split_markdown,
)

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

# Anchors the side panel navigates to, as opposed to the cells beneath them.
_ANCHOR = re.compile(r"^dd-q\d+$")


# ── notebook assembly ────────────────────────────────────────────────────────


def build_notebook(lesson: dict, bank: dict) -> dict:
    mint = IdMinter()
    tests: dict[str, dict] = {}
    # "<kc>#<concept_id>" -> the anchor of the cell that opens that concept.
    # Recorded here and read back off the finished notebook by `main`, for the
    # same reason the question list is read off the anchors: the index must
    # describe what was actually emitted, not what the lesson record said.
    seg_anchors: dict[str, str] = {}
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

        for seg_index, seg in enumerate(segments):
            base = slug(f"{kc}-{seg.get('title') or 'segment'}")
            # ONE CONCEPT IS A PLACE, not just a heading.
            #
            # The gate teaches a segmented KP one concept at a time
            # (`app/lessons.py::_segment_step`), and until this cell had an
            # anchor there was nowhere to send the learner: the panel said
            # "Concept 2 of 3" and the notebook opened the whole KP, all three
            # concepts and every drill, with nothing marking which third was
            # the one being taught.
            #
            # Emitted only for a KP that really has more than one concept —
            # which is the same test the backend and `practice/lessons.js` use
            # to decide a KP is segmented at all. A one-concept KP is taught
            # whole and routes through `dd-kp-<slug>` exactly as before.
            concept_id = str(seg.get("concept_id") or "").strip()
            if concept_id and len(segments) > 1:
                anchor = segment_anchor(kc, seg_index)
                seg_anchors[f"{kc}#{concept_id}"] = anchor
                cells.append(
                    md_cell(
                        f"<!-- dd:{anchor} -->\n\n"
                        f"### {seg.get('title') or kp['title']}\n",
                        anchor,
                    )
                )
            elif seg.get("title"):
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
            # ── the worked example, anchored to the problem it scaffolds ──
            #
            # THIS IS WHY THE EXAMPLE WAS NEVER ON SCREEN. The cells below have
            # always been emitted here, directly above the segment's faded
            # problem, in the right order. But `colab_focus.js` decides what is
            # in focus by the NUMBER in `dd-q<n>`, and these were anchored to
            # the segment (`<kc>-<title>-worked`), which carries no number. So
            # the moment the panel routed a learner to the problem, focus hid
            # the example sitting immediately above it — for all 118 segments,
            # silently, leaving a bare problem that looks exactly like a
            # problem. "It's on the worked example one, and yet it doesn't have
            # a worked example."
            #
            # Anchoring them `dd-q<problem>-worked` puts them in that problem's
            # group, so scrolling up from the problem lands on the example that
            # was always meant to be there. It costs no content and no authoring
            # — the example is already written for every segment.
            #
            # ONE COPY PER PROBLEM, not one per segment. A cell has a single id
            # and a single id names a single group, so a segment with two faded
            # items cannot have both of them share one example — three segments
            # do author two (np-3 `numpy.rescaling`, es-1 `einsum.matvec-matmul`,
            # eo-1 `einops.merge-axes`), and anchoring to the first left the
            # second as the exact bare problem this change exists to abolish.
            # The build guard in `validate` caught it, which is the whole reason
            # it checks anchors rather than lesson records.
            #
            # Duplicating the cells is right rather than merely expedient: the
            # two problems are never in focus together, so no learner ever meets
            # the repeat. Reading the notebook unfocused gives
            # example → problem → example → problem, which is the correct shape
            # for a pair of completion items anyway.
            def _worked_cells(base_id):
                out = list(
                    split_markdown(seg.get("worked_example_markdown"), mint, base_id)
                )
                # `worked_example_code` is the SAME fence `split_markdown` just
                # turned into a cell — `compile_lessons.py` extracts "each
                # segment's sole Python worked fence" into that field, so
                # emitting both means emitting the example's code twice. It was
                # byte-for-byte duplicated in 122 segments and nobody had
                # noticed, because the pair was buried in a wall of unfocused
                # cells. Now that the example is in focus with its problem, the
                # learner would meet the same code block twice, back to back,
                # and reasonably wonder which one is the one that matters.
                # Emitted only when the prose carried no fence of its own, which
                # is what the field is actually for.
                if seg.get("worked_example_code") and not any(
                    c["cell_type"] == "code" for c in out
                ):
                    out.append(code_cell(seg["worked_example_code"], mint(f"{base_id}-code")))
                return out

            _faded = seg.get("faded_items") or []
            if not _faded:
                # No problem to anchor to. The example is still the segment's
                # teaching content and belongs in the notebook; it simply has no
                # focus group to join.
                cells += _worked_cells(f"{base}-worked")

            # Faded rung: the authored starter carries the `_____` blanks, which
            # is a hand-cut scaffold for this concept and beats the mechanical
            # backward fade the server can generate for anything else.
            for item in _faded:
                qid = item["question_id"]
                worked = _worked_cells(f"dd-q{qid}-worked")
                cells += worked
                cells += problem_cells(
                    qid,
                    "faded",
                    bank,
                    mint,
                    tests,
                    prompt=item.get("prompt"),
                    starter=item.get("starter_code"),
                    example=item.get("example"),
                    # The segment's own worked example is directly above and now
                    # shares this problem's anchor, so it IS on screen.
                    after_example=bool(worked),
                )

        for item in kp.get("guided_items") or []:
            # Guided sits on the supported rungs alongside faded (see
            # kc_graph._STAGE_TO_RANKS), so the learner meets it while the strip
            # still promises an example. Anchoring it `dd-q<qid>-worked` is what
            # puts it in the problem's focus group; without that the example is
            # generated and then immediately hidden, which is the bug this whole
            # anchoring scheme exists to prevent.
            gid = item["question_id"]
            guided_worked = []
            if item.get("worked_example_code"):
                guided_worked = [
                    code_cell(item["worked_example_code"], mint(f"dd-q{gid}-worked-code"))
                ]
                cells += guided_worked
            cells += problem_cells(
                gid,
                "guided",
                bank,
                mint,
                tests,
                hints=item.get("hints_markdown"),
                after_example=bool(guided_worked),
            )

        # Applied drills are independent-rung questions that the KP gave an
        # example to, which is what makes them the ladder's third rung rather
        # than its fourth. Emitted before the exampleless ones so an unfocused
        # read of the notebook still runs in ladder order.
        _applied = {i["question_id"]: i for i in (kp.get("applied_items") or [])}
        for qid, item in _applied.items():
            applied_worked = []
            if item.get("worked_example_code"):
                applied_worked = [
                    code_cell(item["worked_example_code"], mint(f"dd-q{qid}-worked-code"))
                ]
                cells += applied_worked
            if item.get("prompt"):
                cells += list(
                    split_markdown(item["prompt"], mint, f"dd-q{qid}-worked-intro")
                )
            cells += problem_cells(
                qid, "applied", bank, mint, tests, after_example=bool(applied_worked)
            )

        for qid in kp.get("independent_items") or []:
            if qid in _applied:
                continue
            cells += problem_cells(qid, "independent", bank, mint, tests)

        if kp.get("misconceptions_markdown"):
            cells += list(
                split_markdown(
                    f"#### Common mistakes\n\n{kp['misconceptions_markdown']}",
                    mint,
                    f"{slug(kc)}-misconceptions",
                )
            )

    # Third from the top, but built last: the payload is only complete once
    # every problem has contributed its cases. It belongs above the problems so
    # `Runtime ▸ Run before` on any check cell picks it up.
    cells.insert(2, checker_cell(tests, mint))

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "delta_drills": {
                "lesson_id": lesson["id"],
                "subtopic_key": lesson.get("subtopic_key"),
                "segments": seg_anchors,
            },
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
        if not id_is_valid(cid):
            problems.append(f"{lesson_id} cell {i}: bad id {cid!r}")
        if cid in seen:
            problems.append(f"{lesson_id} cell {i}: duplicate id {cid!r}")
        seen.add(cid)
        if c["cell_type"] not in ("markdown", "code"):
            problems.append(f"{lesson_id} cell {i}: bad type {c['cell_type']!r}")

    # EVERY FADED PROBLEM MUST HAVE ITS EXAMPLE IN ITS OWN GROUP.
    #
    # The faded rung is "here is the move, now do it" — a completion problem in
    # Renkl's sense, which stops being one the moment the example is not on
    # screen. `colab_focus.js` decides that by the number in `dd-q<n>`, so an
    # example anchored anywhere else is an example the learner never sees while
    # they are on the problem. That was the state of all 118 segments until
    # 2026-08-03, and it is invisible from the outside: the notebook renders,
    # every cell is present in the file, and the rung just quietly behaves like
    # a solo problem.
    #
    # Checked against the emitted ANCHORS rather than against the lesson record,
    # because the anchor is what focus reads. A segment can carry a beautifully
    # written worked example and still fail this, which is exactly the bug.
    ids = {c.get("id", "") for c in nb["cells"]}
    for c in nb["cells"]:
        cid = c.get("id", "")
        if not _ANCHOR.match(cid):
            continue
        if "· faded" not in "".join(c.get("source") or []):
            continue
        n = cid[len("dd-q"):]
        if not any(i.startswith(f"dd-q{n}-worked") or i.startswith(f"dd-q{n}-example")
                   for i in ids):
            problems.append(
                f"{lesson_id} {cid}: a faded problem with no example in its focus "
                f"group — focus will hide whatever scaffold it has and the rung "
                f"becomes a solo problem wearing a faded label"
            )

    # EVERY CONCEPT THE INDEX WILL POINT AT MUST BE A CELL HERE.
    #
    # The panel routes a teaching step to `#scrollTo=<anchor>` and Colab
    # ignores a fragment that names nothing — no error, no scroll, no way for
    # the learner to tell the link from a dead one. The map and the cells come
    # out of the same pass, so the only way they disagree is a bug in this
    # file, which is exactly the kind that ships quietly.
    for key, anchor in (nb["metadata"]["delta_drills"].get("segments") or {}).items():
        if anchor not in ids:
            problems.append(
                f"{lesson_id} {key}: the index would point at {anchor!r}, which is "
                f"not a cell in this notebook"
            )
    return problems


def check_examples(lessons: list[dict], bank: dict) -> tuple[list[str], dict[int, int]]:
    """Validate every authored stage-2 pair. Returns (errors, example -> problem).

    Two things can go wrong with a pair, and both are silent at runtime:

    1. **The example names a question that is not in the bank, or one with no
       canonical solution.** Either way there is nothing to show, and the
       learner meets a "worked example" heading with an empty cell under it.

    2. **The example question is ALSO served as a problem.** Then the course
       hands out that problem's full answer at stage 2 and asks for it again
       later — checked against the anchors in `main`, so it catches the leak
       across lessons and not just within one. This is the reason an example
       question is *spent*: promoting one to be the demonstration means taking
       it out of the rungs, which is a content edit and not a rendering one.
    """
    errors: list[str] = []
    examples: dict[int, int] = {}
    for lesson in lessons:
        for kp in lesson.get("kps") or []:
            for seg in kp.get("segments") or []:
                for item in seg.get("faded_items") or []:
                    example = item.get("example")
                    if not example:
                        continue
                    eid = example.get("question_id")
                    where = f"{lesson['id']} {kp['kc']} q{item.get('question_id')}"
                    ex = (bank.get(eid) or {}).get("exercise", {})
                    if not ex:
                        errors.append(f"{where}: example question {eid} is not in the bank")
                        continue
                    if not (ex.get("canonical_solution") or "").strip():
                        errors.append(f"{where}: example question {eid} has no canonical solution")
                    examples[int(eid)] = int(item["question_id"])
    return errors, examples


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
    segments: dict[str, str] = {}
    for r in records:
        segments.update(r["segments"])
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
        # One CONCEPT of a KP, by the anchor that reaches it —
        # `"<kc>#<concept_id>": "dd-seg-<kc>-<n>"`. Same argument as `kps` one
        # line up, one level finer: a segmented KP is taught a concept at a
        # time, so "take me to the lesson" has to name the concept and not the
        # KP, or the learner is dropped on all three at once with no marker
        # saying which is theirs.
        #
        # Keyed by the EXPOSURE KEY the gate hands back (`app/lessons.py::
        # _segment_step`), so the panel looks up what it was already given
        # rather than reconstructing it. Multi-concept KPs only — a KP with one
        # concept is taught whole and keeps routing through `kps`.
        "segments": segments,
    }
    return index


# How the extension's copy is split across files, in load order. One `<script>`
# per entry; the first assigns the global and the rest extend it.
#
# Split because it is generated data that only grows — 424 questions, 63
# concepts and now their segments — and one file of it was the largest thing in
# the extension by a wide margin. The seam is by QUESTION, so the reader lands
# in the map they are looking for instead of scrolling past four hundred lines
# of `"217": "np-1"` to reach the concept anchors.
INDEX_PARTS = (
    ("", ("dir", "lessons", "subtopics")),
    ("-questions", ("questions",)),
    ("-concepts", ("kcs", "kps", "segments")),
)


def index_scripts(index: dict, path: Path) -> list[tuple[Path, str]]:
    """The extension's copy — classic scripts that build up one global.

    `path` names the first file; the rest are its siblings, so `--index` still
    takes a single argument and the panel's load order is `notebook-index.js`,
    then the parts that extend it.
    """
    out: list[tuple[Path, str]] = []
    for i, (suffix, keys) in enumerate(INDEX_PARTS):
        part = {k: index[k] for k in keys}
        body = json.dumps(part, indent=2, ensure_ascii=False)
        out.append(
            (
                path.with_name(f"{path.stem}{suffix}{path.suffix}"),
                "/* GENERATED by scripts/generate_colab_notebooks.py — do not edit.\n"
                "   Regenerate after changing lessons; the panel navigates by this map. */\n"
                + (
                    f"window.DD_NOTEBOOKS = {body};\n"
                    if i == 0
                    else f"Object.assign(window.DD_NOTEBOOKS, {body});\n"
                ),
            )
        )
    return out


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

    example_errors, examples = check_examples(lessons, bank)
    errors += example_errors

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
                # Read off the notebook for the same reason as `questions`:
                # these are the anchors that were emitted, not the ones the
                # lesson record implies.
                "segments": dict(nb["metadata"]["delta_drills"]["segments"]),
            }
        )

    print(f"\n{len(lessons)} notebooks · {total_cells} cells · {total_problems} problems")

    index = build_index(records, args.out)
    parts = index_scripts(index, args.index)
    if not args.dry_run:
        args.index.parent.mkdir(parents=True, exist_ok=True)
        for path, text in parts:
            path.write_text(text)
        args.web_index.parent.mkdir(parents=True, exist_ok=True)
        args.web_index.write_text(index_as_json(index))
    print(
        f"index · {len(records)} lessons · {total_problems} questions mapped · "
        f"{len(index['segments'])} concept sections · "
        f"{', '.join(p.name for p, _ in parts)} + {args.web_index.name}"
    )

    # Two notebooks claiming the same question would send the panel to whichever
    # the index happened to write last, silently.
    claimed: dict[int, str] = {}
    for r in records:
        for q in r["questions"]:
            if q in claimed:
                errors.append(f"question {q} claimed by both {claimed[q]} and {r['id']}")
            claimed[q] = r["id"]

    # A question shown fully solved as a worked example must not also be served
    # as a problem — anywhere, not just in its own notebook. Checked against the
    # anchors rather than against the lesson records, so it sees what the
    # learner can actually reach.
    for eid, problem in examples.items():
        if eid in claimed:
            errors.append(
                f"question {eid} is the worked example for problem {problem} but is also "
                f"served as a problem in {claimed[eid]} — an example question is spent, "
                f"so remove it from that KP's rungs"
            )

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
