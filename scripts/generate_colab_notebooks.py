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
import base64
import json
import re
import sys
import zlib
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

# The ARENA digits image, as published alongside the notebooks by
# `publish_colab_notebooks.sh`. The einops drills load it by the absolute path
# the backend's grading preamble rewrites; in a notebook there is no preamble,
# so the checker downloads it on first use instead.
FIXTURE_URL = (
    "https://raw.githubusercontent.com/AkiraTheSquid/arena-book-colab/main/"
    "ARENA_5.0/ch-1-foundations/numbers.npy"
)

# nbformat 4.5 cell ids: 1-64 chars of [a-zA-Z0-9-_].
_ID_SAFE = re.compile(r"[^a-zA-Z0-9\-_]")

# A fenced block. Group 1 is the info string, group 2 the body. Only a bare
# `python` info string becomes a code cell; `python no-run`, `text`, etc. stay
# in the markdown.
_FENCE = re.compile(r"^```([^\n]*)\n(.*?)^```[ \t]*$", re.S | re.M)


# Anchors the side panel navigates to, as opposed to the cells beneath them.
_ANCHOR = re.compile(r"^dd-q\d+$")

MAX_ID = 64

# The in-notebook checker, kept as real Python in its own file so watch.py can
# exec it and grade something. Everything between the markers is copied verbatim
# into one cell per notebook.
GRADER = REPO / "scripts" / "colab_grader.py"

# An expected output is a paste of what the example run prints. Some of them are
# a 7 KB tensor repr (the ARENA image drills), and a wall of pixels above the
# starter code buries the problem it is supposed to describe.
MAX_EXPECTED_LINES = 24
MAX_EXPECTED_CHARS = 1200


def grader_source() -> str:
    text = GRADER.read_text()
    start = text.index("# --- embed:start")
    start = text.index("\n", start) + 1
    return text[start : text.index("# --- embed:end")].rstrip() + "\n"


def fence(body: str, info: str = "") -> str:
    """A fenced block whose fence is longer than any run of backticks inside it."""
    longest = max((len(m) for m in re.findall(r"`+", body)), default=0)
    bar = "`" * max(3, longest + 1)
    return f"{bar}{info}\n{body}\n{bar}"


def expected_output_block(exercise: dict) -> str:
    """The "here is what you should see" section of a problem's prose.

    The whole point of the Colab edition is that the learner runs the code
    themselves and decides whether it worked — which they cannot do without
    knowing what working looks like. Reported as: "it doesn't show you the
    expected output that you should see".
    """
    text = (exercise.get("expected_output") or "").rstrip()
    if not text:
        return ""
    lines = text.splitlines()
    clipped = len(lines) > MAX_EXPECTED_LINES or len(text) > MAX_EXPECTED_CHARS
    if clipped:
        lines = lines[:MAX_EXPECTED_LINES]
        text = "\n".join(lines)[:MAX_EXPECTED_CHARS].rstrip() + "\n… (truncated)"
    note = ""
    if exercise.get("expected_artifact_type") == "image":
        note = " This one draws an image; below is the tensor behind it."
    return (
        "**Expected output** — run the cell below once `solve` is right and it "
        f"should print this.{note}\n\n" + fence(text, "text")
    )


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


def example_cells(qid: int, example: dict, bank: dict, mint: IdMinter) -> list[dict]:
    """The worked half of a stage-2 pair: a problem shown already solved.

    A worked example is not a third kind of content — it is a PROBLEM plus its
    known answer, which is why this is derived from the bank rather than
    authored. The lesson names a `question_id` and nothing else that could go
    stale; the prompt and the canonical solution are read out of
    `questions_structured.json` at build time, so the example is a real problem
    the grader agrees about rather than a snippet that drifted away from one.
    All the author has to write is `note_markdown`: the sentence saying what
    carries across to the problem below and what does not. That is the whole
    per-pair authoring cost, which is what makes converting 118 segments a
    plausible piece of work rather than a rewrite of the course.

    This is a NOTEBOOK-ONLY construct. The side panel is a tutor rail and the
    notebook is where the learner reads and writes, so the example belongs on
    the Colab side of the screen; `practice/ladder.js` is deliberately not
    involved and still shows what it always showed.

    THE ANCHOR IS THE WHOLE POINT. These cells are minted as
    `dd-q<problem>-example`, naming the PROBLEM they scaffold and not the
    question they were built from. `colab_focus.js` groups cells by the number
    in `dd-q<n>`, so an example anchored this way is in focus exactly when its
    problem is, and never otherwise. The alternative — a DOM heuristic like
    "also keep whatever sits above the target" — would work today and would one
    day put an unrelated segment's prose on screen, silently, with no way for
    the learner to tell that it does not belong to the problem they are on.

    No `<!-- dd:… -->` marker in the body, unlike a problem header. That marker
    is the text fallback `colab.js` searches when Colab drops cell ids, and a
    substring search for `dd-q481` would find `dd:dd-q481-example` first — the
    panel would route to the example and the learner would land above their own
    problem. Nothing needs to navigate TO an example; it only has to be visible
    when its problem is.
    """
    ex = (bank.get(example["question_id"]) or {}).get("exercise", {})
    prompt = (ex.get("question_text") or "").strip()
    solution = (ex.get("canonical_solution") or "").strip()
    note = (example.get("note_markdown") or "").strip()

    body = (
        "#### Worked example — read this one, you are not solving it\n\n"
        f"{prompt}\n\n"
        "The full answer is in the cell below. Run it, read it, then do the "
        "problem underneath — it is the same move on different specifics.\n"
    )
    if note:
        body += f"\n{note}\n"
    cells = [md_cell(body, mint(f"dd-q{qid}-example"))]
    if solution:
        cells.append(
            code_cell(
                "# The worked example, solved. Running it binds `solve` to THIS answer —\n"
                "# re-run your own cell below before checking, or you are checking mine.\n"
                f"{solution}\n",
                mint(f"dd-q{qid}-example-code"),
            )
        )
    return cells


def problem_cells(
    qid: int,
    rung: str,
    bank: dict,
    mint: IdMinter,
    tests: dict,
    *,
    prompt: str | None = None,
    starter: str | None = None,
    hints: str | None = None,
    example: dict | None = None,
    after_example: bool = False,
) -> list[dict]:
    """The cells for one problem, at one rung.

    `dd-q<id>` is the anchor the side panel jumps to, and it goes on the header
    cell rather than the code cell so the student lands on the question text
    with the editor just below it.

    Four cells, in the order a learner meets them: the problem (with the output
    it should produce), the starter code, a checker, and the answer. `tests` is
    the notebook's payload dict and gets this problem's cases added to it.

    An `example` puts a fifth thing FIRST — a solved problem, above the header,
    so the panel's jump still lands on the problem and the example is what the
    learner finds by scrolling up. That order is the request, verbatim: "you
    should be able to scroll up and see a particular problem, and then when you
    scroll down you should see a problem that's kind of similar but meaningfully
    different from it."
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
    expected = expected_output_block(ex)
    if expected:
        header += f"\n{expected}\n"
    cells = example_cells(qid, example, bank, mint) if example else []
    # "Your turn" whenever something demonstrated the move first — either a
    # promoted bank question (`example`) or the segment's own worked example,
    # which is now anchored into this problem's group and therefore on screen
    # right above it. Without the label the learner meets two problems in a row
    # and has to work out for themselves that the first one was already solved.
    if example or after_example:
        header = header.replace(
            f"### Problem {qid} · {rung}\n\n",
            f"### Problem {qid} · {rung} — your turn\n\n",
            1,
        )
    cells.append(md_cell(header, f"dd-q{qid}"))

    if hints:
        cells.append(
            md_cell(
                "<details>\n<summary>Hints</summary>\n\n"
                f"{hints.strip()}\n\n</details>\n",
                mint(f"dd-q{qid}-hints"),
            )
        )

    cells.append(code_cell(code or "# your code here\n", mint(f"dd-q{qid}-code")))

    # The checker. Comparing your own printed output against the expected block
    # above catches the example input and nothing else — these are the same
    # cases the tutor grades with, including the edge cases the example does not
    # cover, which is why a drill can look right and still be wrong.
    cases = ex.get("test_cases") or []
    if cases:
        tests[str(qid)] = {"fn": ex.get("function_name") or "solve", "cases": cases}
        cells.append(
            code_cell(
                f"# Did it work? Run this. (NameError → run the checker cell at\n"
                f"# the top of the notebook first: Runtime ▸ Run before.)\n"
                f"dd_check({qid})\n",
                mint(f"dd-q{qid}-check"),
            )
        )

    # The answer, last — and NOT on screen until the learner has said how it
    # went. The extension hides `dd-q<n>-solution` cells outright
    # (`content/colab_dd.css`) and the panel's verdict click unhides this one:
    # "then and only then it shows you the solution … below what you typed".
    #
    # It was a collapsed `display-mode: "form"` cell for one pass, which still
    # put "💡 Solution — Problem 480" on screen under the code you were trying
    # to write, and still needed a second click after the reveal. The `#@title`
    # stays because it renders as a heading; the collapse does not.
    #
    # Without the extension nothing hides it — a plain reader of the published
    # repo sees the answers, the way ARENA's own notebooks do. The toggle's
    # "Show every solution" is the same escape hatch for anyone re-reading a
    # notebook they have already worked through.
    solution = ex.get("canonical_solution") or ""
    if solution.strip():
        cells.append(
            code_cell(
                f"#@title 💡 Solution — Problem {qid}\n"
                "# Running this rebinds `solve` to the reference answer. Re-run your\n"
                "# own cell before dd_check() again, or you are checking this one.\n"
                f"{solution.strip()}\n",
                mint(f"dd-q{qid}-solution"),
            )
        )
    return cells


# ── notebook assembly ────────────────────────────────────────────────────────


def setup_cell(lesson: dict, mint: IdMinter) -> dict:
    """One line naming the lesson, for the extension to read off the page.

    It used to carry `DD_TOKEN` and `DD_BACKEND_URL` for a completion beacon.
    That beacon does not exist on this route and cannot: Colab renders a cell's
    output in a sandboxed iframe, so the panel learns a result by reading the
    line `dd_check` PRINTS (`content/colab_focus.js`), which needs no token and
    no URL. Two dead variables would be harmless if they were invisible — but
    they were the first thing in the notebook, so every lesson opened on a
    backend URL and an instruction to paste a credential, for a feature that
    was never wired.

    `DD_LESSON_ID` stays because `content/colab.js`'s `identify` matches it as
    the id-independent route to "which notebook is this" — it is rendered text,
    so it survives Colab dropping cell ids. The extension hides this cell.
    """
    return code_cell(
        "# === Delta Drills ===\n"
        "# Which lesson this notebook is, for the side panel. Nothing to run.\n"
        f'DD_LESSON_ID = "{lesson["id"]}"\n',
        mint("dd-setup"),
    )


def checker_cell(tests: dict, mint: IdMinter) -> dict:
    """The one cell that defines `dd_check`, carrying this notebook's cases.

    Deflated and base64'd, in a form cell. Both are for the same reason and
    neither is a secret: an 80 KB JSON literal of expected values, expanded, is
    the answer key to every problem below it printed at the top of the notebook.
    """
    payload = base64.b64encode(
        zlib.compress(json.dumps(tests, ensure_ascii=False).encode("utf-8"), 9)
    ).decode("ascii")
    chunks = "\n".join(f'    "{payload[i : i + 96]}"' for i in range(0, len(payload), 96))
    return code_cell(
        '#@title 🔧 Delta Drills checker — run me first { display-mode: "form" }\n'
        + grader_source()
        + f'\n_DD_FIXTURE_URL = "{FIXTURE_URL}"\n'
        + "_dd_install_fixtures()\n"
        + "_DD_TESTS = _dd_load(\n"
        + chunks
        + "\n)\n"
        + f'print("Delta Drills checker ready — {len(tests)} problems. '
        'Run dd_check(<problem number>) under any of them.")\n',
        mint("dd-checker"),
    )


def build_notebook(lesson: dict, bank: dict) -> dict:
    mint = IdMinter()
    tests: dict[str, dict] = {}
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
