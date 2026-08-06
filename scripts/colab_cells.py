#!/usr/bin/env python3
"""colab_cells.py — the cells a Colab notebook is made of.

Split out of `generate_colab_notebooks.py`, which had grown to hold two jobs at
once: what a cell IS, and which cells a lesson turns into. This half is the
first — id minting, markdown/code cells, and the four-cell shape of a problem —
and it is the half with the invariants that bite, so it is worth reading on its
own rather than as the top third of a compiler.

TWO THINGS THAT ARE EASY TO GET WRONG
-------------------------------------
1. **Stable cell ids.** All 458 ARENA_5.0 notebooks are nbformat 4.2 with no
   `metadata.id`, so Colab mints fresh DOM ids on every load and `#scrollTo=`
   anchors are worthless there. We emit 4.5 with a deterministic id per cell
   (`dd-q<question_id>`, `dd-kp-<kc>`, `dd-seg-<kc>-<n>`), which is what lets
   the panel jump straight to a problem or to one concept of a KP. The id is
   ALSO written into the markdown body as an HTML comment, so navigation still
   works if Colab ever drops the attribute.

2. **Which fences become runnable cells.** The authoring contract
   (`lessons/README.md`) is "a fence gets a Run button iff CI runs it": a plain
   ```python fence is executed by the validator and must therefore render as a
   real code cell, while ```python no-run is illustrative and must stay inside
   the prose. Collapsing that distinction would put un-runnable snippets in
   front of a student as if they were exercises.

Nothing here reads a lesson or writes a file. `generate_colab_notebooks.py`
does both.
"""

from __future__ import annotations

import base64
import json
import re
import zlib
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parents[1]

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


def id_is_valid(cell_id: str) -> bool:
    """What nbformat 4.5 will accept as a cell id.

    Exported so the compiler's `validate` asks the module that mints ids what a
    legal one looks like, rather than keeping a second copy of the rule.
    """
    return bool(cell_id) and not _ID_SAFE.search(cell_id) and len(cell_id) <= MAX_ID


def segment_anchor(kc: str, index: int) -> str:
    """The anchor for one concept of a KP — `dd-seg-<kc>-<n>`.

    BY POSITION, NOT BY CONCEPT ID. The id is authored (or derived from a
    title) and is free to be long, unicode, or edited; the anchor has 64 safe
    characters and has to stay put across regenerations, so it is minted from
    the KC and the segment's ORDER, and the map from `<kc>#<concept_id>` to
    this string is shipped in the index. Nothing re-derives it.

    Truncation is on the KC, not on the whole string: a KC long enough to be
    cut still has to end in `-<n>`, or two concepts of one KP collide onto the
    same anchor and focus shows the wrong one.
    """
    tail = f"-{int(index)}"
    head = slug(kc)[: MAX_ID - len("dd-seg-") - len(tail)].strip("-")
    return f"dd-seg-{head}{tail}"


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


# ── the two cells every notebook opens with ──────────────────────────────────


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
