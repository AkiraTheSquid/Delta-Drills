#!/usr/bin/env python3
"""compile_arena_notebooks.py — the ARENA exercise notebooks, shaped for the app.

WHY THIS EXISTS
---------------
The Courses tab is the ARENA curriculum: five chapters, thirty-one sections,
every section a notebook Callum McDougall publishes. Until now clicking one
LEFT the app — the section rows were `<a target="_blank">` at Google Colab, and
the fork gate existed to point them at the student's own ARENA_3.0 fork.

Seth, 2026-09-01: "it won't actually take you to the Google Colab. It will
instead stay inside of the app, and it will have an app version of those arena
notebooks." So the section rows open `practice/arena-notebook.js`, and this is
what feeds it: the upstream `.ipynb` rewritten into the same cell JSON the
lesson notebooks already use, so the ARENA notebook is rendered by the same CSS
and reads like the notebook surface next door.

🔴 THIS IS NOT `compile_web_notebooks.py`, AND THE DIFFERENCE MATTERS.
That compiler emits DELTA DRILLS content: it calls `build_notebook`, the one
function that also mints the .ipynb, so the two editions cannot drift. There is
no such shared function here, because the source is not ours — it is upstream's
notebook, read as data. Nothing in this file may edit what a cell SAYS. It
decides only which cells exist, what kind each one is, and how HTML that the
app's markdown renderer would escape is turned back into markdown.

WHERE THE SECTION LIST COMES FROM
    `Local_Deployed_Shared/courses.js`. That file is the curriculum — chapter
    titles, section numbers, section descriptions, and the book URL each row
    points at — and it is what the learner reads. Parsing it here keeps ONE
    list: a section added to the Courses tab gets a notebook on the next
    compile, and a section that is not on the tab cannot be opened. The parse
    is strict (see `_sections`): a courses.js that stops matching fails the
    run rather than quietly compiling half a curriculum.

INPUTS (read-only)
    Local_Deployed_Shared/courses.js                      the section list
    Local_Deployed_Shared/content/ARENA_5.0-main/**.ipynb the notebooks

    🔴 `Local_Deployed_Shared/content/` is gitignored — 89 MB of upstream
    course repos that live on this machine and are not part of this repo. A
    clone without it cannot run this script, which is why the output is
    committed rather than regenerated on demand.

OUTPUT
    Local_Deployed_Shared/lessons/notebooks/arena-<slug>.json   one per section
    Local_Deployed_Shared/lessons/notebooks/arena-index.json    the index

    Written into the lessons/notebooks folder, next to the Delta Drills
    notebooks, because the view fetches them the same way and the deploy
    already mirrors that folder wholesale. `arena-` is the prefix that keeps
    the two sets apart — `compile_web_notebooks.py` deletes any *.json in
    there that its own manifest does not name, so its stale-file sweep skips
    this prefix (there is a check for that in scripts/watch.py).

Usage:
    python3 scripts/compile_arena_notebooks.py [--out DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHARED = REPO / "Local_Deployed_Shared"
COURSES_JS = SHARED / "courses.js"
DEFAULT_OUT = SHARED / "lessons" / "notebooks"
PREFIX = "arena-"

# The upstream checkout the notebooks are read from. ARENA_3.0-main and
# ARENA_4.0-main are also on disk and are OLDER cuts of the same curriculum:
# 3.0 has 15 of the 31 sections, 4.0 has 19, and 5.0 has all of them. Naming
# the edition here rather than searching all three keeps every section on one
# cut of the course — a notebook silently read from 3.0 because 5.0 renamed a
# folder would put the learner on a different version of the exercise than the
# one the Courses tab describes.
ARENA_EDITION = "ARENA_5.0-main"

# courses.js writes book URLs, not notebook paths: `/arena-book/<rest>.html` is
# the Jupyter Book page, and `<rest>.ipynb` is the notebook it was built from.
BOOK_PREFIX = "/arena-book/"


# --------------------------------------------------------------------------
# the section list
# --------------------------------------------------------------------------

# One `{ number, title, desc, url }` literal per section, in courses.js order.
# Deliberately anchored on all four keys in that order: a looser pattern would
# match a partially-edited literal and drop the fields it did not find.
_SECTION_RE = re.compile(
    r'\{\s*number:\s*"(?P<number>[^"]*)",\s*'
    r'title:\s*"(?P<title>[^"]*)",\s*'
    r'desc:\s*"(?P<desc>[^"]*)",\s*'
    r'url:\s*"(?P<url>/arena-book/[^"]+)"\s*\}'
)

# The chapter each section belongs to, so the index can group them the way the
# tab does. `title: "Chapter 0 — Fundamentals"` opens each chapter literal.
_CHAPTER_RE = re.compile(r'^\s{4}\{\s*$\n\s{6}title:\s*"([^"]+)",', re.M)

# A courses.js that stops matching must FAIL rather than compile a shorter
# curriculum: the symptom of a silent partial parse is a section row that opens
# "no notebook for …", which reads as a broken app rather than a stale build.
MIN_SECTIONS = 25


def _sections(js: str) -> list[dict]:
    """Every section literal in courses.js, tagged with its chapter."""
    chapters = [(m.start(), m.group(1)) for m in _CHAPTER_RE.finditer(js)]
    out = []
    for m in _SECTION_RE.finditer(js):
        chapter = ""
        for start, title in chapters:
            if start < m.start():
                chapter = title
            else:
                break
        out.append({
            "number": m.group("number"),
            "title": m.group("title"),
            "desc": m.group("desc"),
            "url": m.group("url"),
            "chapter": chapter,
        })
    if len(out) < MIN_SECTIONS:
        raise SystemExit(
            f"courses.js: parsed only {len(out)} sections (expected >= {MIN_SECTIONS}). "
            "The section literal shape changed — fix _SECTION_RE rather than lowering "
            "the floor, or the Courses tab will link to notebooks that were never built."
        )
    return out


def _notebook_path(url: str) -> str:
    """`/arena-book/<rest>.html` -> `<rest>.ipynb`, the path inside ARENA.

    Book URLs are percent-encoded per segment (`%26` for `&`, which is in four
    of the section filenames). Decoded here, because what is on the other end
    is a filename on disk and not a URL.
    """
    rel = url[len(BOOK_PREFIX):]
    rel = re.sub(r"\.html$", ".ipynb", rel)
    return "/".join(urllib.parse.unquote(seg) for seg in rel.split("/"))


def _slug(section: dict, path: str) -> str:
    """The id a notebook is opened by: `0-0`, `1-3-1`, `2-4`.

    The section NUMBER, because that is what the curriculum calls it and what
    the learner sees on the row they clicked. Sections without a number fall
    back to the notebook's own filename stem, which is unique by construction.
    """
    number = (section.get("number") or "").strip()
    if number:
        return number.replace(".", "-")
    return re.sub(r"[^a-z0-9]+", "-", Path(path).stem.lower()).strip("-")


# --------------------------------------------------------------------------
# markdown: HTML the app's renderer would escape, turned back into markdown
# --------------------------------------------------------------------------
#
# The app has ONE markdown renderer (`practice/lessons.js::md`, shared with the
# lesson page, the ladder's inline example and lessons/viewer.html) and it
# escapes HTML on purpose — lesson prose is authored here and a lesson has no
# business emitting tags. ARENA prose is not authored here: across the 31
# notebooks it carries 1182 <details>, 274 <img>, 121 <code>, 60 <br> and 11
# <a>. Left alone every one of those would PRINT as tags.
#
# So the tags are converted to their markdown equivalents ONCE, here, rather
# than by teaching the shared renderer to pass HTML through. That direction was
# considered and rejected: HTML passthrough in the renderer would apply to
# every surface, including the ones whose content comes from the question bank.

_TAG_SUBS = [
    # `<img src="U" width="350">` -> `![](U)`. Width is dropped; the view sizes
    # images to the column, which is what a 350px-wide header image inside a
    # 900px column wanted anyway.
    (re.compile(r'<img\b[^>]*?\bsrc="([^"]+)"[^>]*>', re.I), r"![](\1)"),
    (re.compile(r"<br\s*/?>", re.I), "\n"),
    (re.compile(r"<code>(.*?)</code>", re.I | re.S), r"`\1`"),
    (re.compile(r"<(?:b|strong)>(.*?)</(?:b|strong)>", re.I | re.S), r"**\1**"),
    (re.compile(r"<(?:i|em)>(.*?)</(?:i|em)>", re.I | re.S), r"*\1*"),
    (re.compile(r'<a\b[^>]*?\bhref="([^"]+)"[^>]*>(.*?)</a>', re.I | re.S), r"[\2](\1)"),
]

# Plotly figures and the like: upstream embeds them as an <iframe> pointing at
# a hosted HTML file. There is nothing to render in an app that does not run
# them, and dropping the tag silently would leave a paragraph referring to "the
# figure below" with no figure and no explanation.
_IFRAME_RE = re.compile(r'<iframe\b[^>]*?\bsrc="([^"]+)"[^>]*>.*?</iframe>', re.I | re.S)

_BLOCKQUOTE_RE = re.compile(r"<blockquote>(.*?)</blockquote>", re.I | re.S)

# What is left after the substitutions above: <span>, <div>, <p>, stray closing
# tags. Removed rather than escaped — an unmatched `<span style=…>` printed to
# the learner is noise either way, and the text inside it is the content.
_LEFTOVER_TAG_RE = re.compile(r"</?(?:span|div|p|ul|ol|li|table|thead|tbody|tr|td|th|font|u|hr)\b[^>]*>", re.I)


def _quote(inner: str) -> str:
    body = inner.strip("\n")
    return "\n".join(("> " + line) if line.strip() else ">" for line in body.split("\n"))


def _to_markdown(src: str) -> str:
    """Upstream markdown-with-HTML -> the subset the app's renderer reads."""
    text = _IFRAME_RE.sub(lambda m: f"[Interactive figure — opens upstream ↗]({m.group(1)})", src)
    text = _BLOCKQUOTE_RE.sub(lambda m: _quote(_to_markdown(m.group(1))), text)
    for pattern, repl in _TAG_SUBS:
        text = pattern.sub(repl, text)
    text = _LEFTOVER_TAG_RE.sub("", text)
    # `&amp;` etc. survive the tag strip and would otherwise be double-escaped
    # by the renderer, which escapes what it is handed.
    return html.unescape(text)


# A disclosure, which upstream uses for every hint and every solution. Captured
# as a cell of its own so the view can render a real <details> — the same shape
# the lesson notebooks' hints and solutions already have.
_DETAILS_RE = re.compile(r"<details>\s*(?:<summary>(.*?)</summary>)?(.*?)</details>", re.I | re.S)


def _split_markdown(src: str) -> list[tuple[str, str, str]]:
    """One markdown cell -> `(role, summary, markdown)` parts, in order.

    A cell is usually prose, or a disclosure, or prose FOLLOWED by one or more
    disclosures — 1182 of them across the corpus, and only 546 cells begin with
    one. Splitting rather than rendering the whole cell as prose is what makes
    a solution collapsible instead of a wall of text under the exercise.
    """
    parts = []
    at = 0
    for m in _DETAILS_RE.finditer(src):
        before = src[at:m.start()]
        if before.strip():
            parts.append(("prose", "", _to_markdown(before)))
        summary = _to_markdown(m.group(1) or "").strip() or "Show"
        parts.append(("details", summary, _to_markdown(m.group(2) or "")))
        at = m.end()
    rest = src[at:]
    if rest.strip() or not parts:
        parts.append(("prose", "", _to_markdown(rest)))
    return [p for p in parts if p[2].strip() or p[0] == "details"]


# --------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------

# IPython line/cell magics (`%pip install einops`, `!git clone …`). They are a
# SyntaxError to `exec`, which is what the app's kernel runs, so a Run button on
# one is a button that cannot work. There are 32 such cells and they are all
# Colab setup — marked so the view renders them read-only with a reason, rather
# than letting a learner press Run on the first cell of every notebook and read
# a SyntaxError as "the app is broken".
#
# `%%` is in the alternation for a reason even though ARENA_5.0 contains none
# today (every one of the 224 magic lines here is `%pip` or a `!` shell line).
# A cell magic is `%%capture` / `%%time` / `%%bash`, and `[%!]\w` does not match
# it — the second `%` is not a word character — so it would have compiled to a
# RUNNABLE cell whose Run button can only ever produce a SyntaxError. That is
# one upstream edit away, and it fails in the learner's face rather than here.
_MAGIC_RE = re.compile(r"^\s*(?:%%|[%!])\w", re.M)


def _cells(nb: dict, slug: str) -> list[dict]:
    out = []
    for index, cell in enumerate(nb.get("cells") or []):
        src = "".join(cell.get("source") or [])
        # Cell ids are positional and stable across a recompile of the same
        # upstream notebook: the view's jump list, the deep link and the
        # `scrollIntoView` all address cells by id.
        base = f"{slug}-c{index:03d}"
        if cell.get("cell_type") == "code":
            if not src.strip():
                continue
            out.append({
                "t": "code",
                "id": base,
                "role": "magic" if _MAGIC_RE.search(src) else "code",
                "src": src.rstrip(),
            })
            continue
        for part, (role, summary, text) in enumerate(_split_markdown(src)):
            entry = {
                "t": "md",
                "id": base if part == 0 else f"{base}-{part}",
                "role": role,
                "src": text.strip(),
            }
            if role == "details":
                entry["summary"] = summary
            out.append(entry)
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="where the compiled notebooks are written")
    parser.add_argument("--dry-run", action="store_true",
                        help="compile and report, write nothing")
    args = parser.parse_args()

    if not COURSES_JS.exists():
        raise SystemExit(f"missing {COURSES_JS}")
    arena = SHARED / "content" / ARENA_EDITION
    if not arena.is_dir():
        raise SystemExit(
            f"missing {arena}\n"
            "Local_Deployed_Shared/content/ is gitignored — the upstream ARENA checkout "
            "lives on this machine only. Restore it before recompiling."
        )

    sections = _sections(COURSES_JS.read_text(encoding="utf-8"))
    # --dry-run writes nothing, and that includes not creating the folder it
    # would have written into.
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)

    index = []
    total_cells = 0
    missing = []
    for section in sections:
        rel = _notebook_path(section["url"])
        source = arena / rel
        if not source.exists():
            print(f"  ⚠ {section['number'] or section['title']}: no notebook at {rel}")
            missing.append(section["number"] or section["title"])
            continue
        nb = json.loads(source.read_text(encoding="utf-8"))
        slug = _slug(section, rel)
        cells = _cells(nb, slug)
        total_cells += len(cells)
        name = f"{PREFIX}{slug}.json"
        payload = {
            "id": slug,
            "number": section["number"],
            "title": section["title"],
            "desc": section["desc"],
            "chapter": section["chapter"],
            # Where this came from, carried into the notebook itself so the
            # view can offer the upstream original without a second lookup.
            "edition": ARENA_EDITION,
            "notebook_path": rel,
            "book_url": section["url"],
            "cells": cells,
        }
        if not args.dry_run:
            (args.out / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
            )
        index.append({
            "id": slug,
            "number": section["number"],
            "title": section["title"],
            "desc": section["desc"],
            "chapter": section["chapter"],
            "file": name,
            "cells": len(cells),
            "code_cells": sum(1 for c in cells if c["t"] == "code"),
            "notebook_path": rel,
        })
        print(f"  {slug:<8} {len(cells):>4} cells  {section['title']}")

    if not args.dry_run:
        (args.out / f"{PREFIX}index.json").write_text(
            json.dumps({"edition": ARENA_EDITION, "sections": index},
                       ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        # A section removed from courses.js must STOP being reachable. The view
        # opens `arena-<slug>.json` on a deep link without consulting the
        # index, and the deploy mirrors this folder wholesale, so a leftover
        # file is a retired section still being served with nothing to say so.
        keep = {entry["file"] for entry in index} | {f"{PREFIX}index.json"}
        for stale in sorted(args.out.glob(f"{PREFIX}*.json")):
            if stale.name not in keep:
                stale.unlink()
                print(f"  removed stale notebook: {stale.name}")

    print(f"\n{len(index)} ARENA notebooks · {total_cells} cells")

    # 🔴 A PARTIAL BUILD MUST NOT EXIT 0. Every section above still wrote what
    # it could, which is the right thing for a local rebuild — but the rows for
    # the skipped sections stay in courses.js and open the "not compiled here"
    # error, and a caller that only reads the exit code (a deploy step, a CI
    # job, `&&` in a shell line) would call that a successful build. Say it in
    # the status, not only in a line of output nobody is reading.
    if missing:
        print(
            f"⚠ {len(missing)} section(s) had no upstream notebook and were "
            f"SKIPPED: {', '.join(missing)}\n"
            f"  Their rows in courses.js will open the 'not compiled here' "
            f"message. Check {ARENA_EDITION} is the edition those URLs belong to."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
