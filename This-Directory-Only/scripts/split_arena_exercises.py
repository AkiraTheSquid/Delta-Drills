#!/usr/bin/env python3
"""Split ARENA exercise notebooks into one Colab-ready notebook per exercise.

Walks every notebookPath referenced by Local_Deployed_Shared/arena/manifest.js,
splits each `*_exercises.ipynb` at every `### Exercise -` heading, and writes
self-contained per-exercise notebooks under `arena-book-colab/ARENA_5.0/...`
suitable for opening in Google Colab via the GitHub redirect URL.

Each split notebook contains:
  1. A Delta-Drills banner markdown cell with link back to the SPA.
  2. ARENA's original setup cells (preserved verbatim — they handle Colab
     bootstrap by downloading the ARENA repo into /content).
  3. A `DD_TOKEN` cell (placeholder for the user's Delta-Drills auth token).
  4. An auto-import of every prior-exercise solution from `solutions.py` so
     the user can jump straight into the exercise they care about without
     having implemented all the predecessors.
  5. The exercise heading + stub cell + any intervening context cells.
  6. A "completion beacon" cell that wraps `tests.test_<fn>` so a passing
     test posts to the Delta Drills backend.

Output layout:
  arena-book-colab/ARENA_5.0/<chapter>/<section_dir>/<id>_<slug>.ipynb

Where <id> is `0.1.0`, `0.1.1`, etc. (chapter.section.idx0).
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_DIR / "Local_Deployed_Shared"
MANIFEST_PATH = SHARED_DIR / "arena" / "manifest.js"
OUT_ROOT = REPO_DIR / "arena-book-colab" / "ARENA_5.0"

EXERCISE_HEADING_RE = re.compile(r"^(#{2,4})\s+(Exercise\b.*)$")
SECTION_BREAK_RE = re.compile(r"^#{1,2}(?!#)\s+\S")  # `# ` or `## ` but not `### `
NOTEBOOK_PATH_RE = re.compile(r'notebookPath:\s*"([^"]+)"')
SECTION_RE = re.compile(r'section:\s*"([^"]+)"')
ID_RE = re.compile(r'\bid:\s*"([^"]+)"')
DEF_RE = re.compile(r"^def\s+(\w+)\s*\(", re.M)
CLASS_RE = re.compile(r"^class\s+(\w+)\s*[(:]", re.M)
HEADING_BACKTICK_RE = re.compile(r"`([A-Za-z_][\w]*)`")
TEST_CALL_RE = re.compile(r"\btests\.test_(\w+)\s*\(")

DD_BACKEND_URL = "https://delta-drills-backend.fly.dev"
DD_SPA_URL = "https://delta-drills.vercel.app"


def slugify(text: str) -> str:
    text = text.lower().replace("`", "")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def cell_source(cell: dict) -> str:
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src or ""


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


@dataclass
class ManifestEntry:
    id: str
    section: str
    notebook_path: str


def parse_manifest() -> list[ManifestEntry]:
    """Pull (id, section, notebookPath) tuples out of manifest.js by parsing
    each `{` ... `}` block in ARENA_CURRICULUM."""
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    entries: list[ManifestEntry] = []
    # Greedy block split on `{ id: "..."` openers.
    blocks = re.split(r"\n\s*\{\s*id:\s*", text)
    for block in blocks[1:]:
        # Reattach the leading `id: ` so our regexes still hit.
        block = "id: " + block
        id_m = ID_RE.search(block)
        sec_m = SECTION_RE.search(block)
        nb_m = NOTEBOOK_PATH_RE.search(block)
        if not (id_m and sec_m and nb_m):
            continue
        entries.append(ManifestEntry(id_m.group(1), sec_m.group(1), nb_m.group(1)))
    return entries


def split_notebook(entry: ManifestEntry) -> tuple[int, list[str]]:
    """Split one ARENA notebook into per-exercise files. Returns
    (count_written, warnings)."""
    nb_path = SHARED_DIR / entry.notebook_path
    if not nb_path.exists():
        return 0, [f"missing notebook: {entry.notebook_path}"]

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    nb_meta = nb.get("metadata", {})

    # Locate every `### Exercise -` heading and its bound function name.
    exercise_indices: list[tuple[int, str, str]] = []  # (cell_idx, heading_text, fn_name)
    for i, c in enumerate(cells):
        if c.get("cell_type") != "markdown":
            continue
        src = cell_source(c)
        first = (src.splitlines() or [""])[0].strip()
        m = EXERCISE_HEADING_RE.match(first)
        if not m:
            continue
        # Walk forward for the first def/class in the next handful of code cells.
        fn_name = ""
        for j in range(i + 1, min(i + 8, len(cells))):
            cj = cells[j]
            if cj.get("cell_type") != "code":
                continue
            src_j = cell_source(cj)
            d = DEF_RE.search(src_j) or CLASS_RE.search(src_j)
            if d:
                fn_name = d.group(1)
                break
        # Fallback: pull the first backtick-quoted identifier out of the heading
        # itself (e.g., "Exercise - implement `ReLU`" → "ReLU"). Avoids
        # leaving the beacon manual for class-based exercises.
        if not fn_name:
            heading_text = m.group(2).strip()
            bt = HEADING_BACKTICK_RE.search(heading_text)
            if bt:
                fn_name = bt.group(1)
        exercise_indices.append((i, m.group(2).strip(), fn_name))

    if not exercise_indices:
        return 0, [f"no exercises found in {entry.notebook_path}"]

    first_ex_idx = exercise_indices[0][0]
    setup_cells = cells[:first_ex_idx]

    # Section number, e.g. "0.1" from "0.1 Ray Tracing".
    section_number = entry.section.split(maxsplit=1)[0]

    # Derive section_dir name from the notebook path
    # (.../exercises/<section_dir>/<file>.ipynb).
    section_dir_name = Path(entry.notebook_path).parent.name
    chapter_dir = Path(entry.notebook_path).parts[2]  # chapterN_xxx

    out_dir = OUT_ROOT / chapter_dir / section_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    warnings: list[str] = []

    for idx, (heading_idx, heading_text, fn_name) in enumerate(exercise_indices):
        next_ex_idx = (
            exercise_indices[idx + 1][0] if idx + 1 < len(exercise_indices) else len(cells)
        )
        # Stop earlier if a top-level / sub-section header appears before the
        # next Exercise — that's the start of new content unrelated to this
        # exercise (e.g., `# 2️⃣ Batched Operations`, `## Tensor Operations`).
        cut_idx = next_ex_idx
        for j in range(heading_idx + 1, next_ex_idx):
            cj = cells[j]
            if cj.get("cell_type") != "markdown":
                continue
            first = (cell_source(cj).splitlines() or [""])[0].strip()
            if SECTION_BREAK_RE.match(first):
                cut_idx = j
                break
        body_cells = cells[heading_idx:cut_idx]
        exercise_id = f"{section_number}.{idx}"
        slug = slugify(heading_text)
        out_name = f"{exercise_id.replace('.', '_')}_{slug}.ipynb"
        out_path = out_dir / out_name

        # Find the actual tests.test_X(...) call name inside this exercise's
        # body cells. This is more reliable than guessing from fn_name, since
        # ARENA uses snake_case test names even for PascalCase classes.
        test_name = ""
        for c in body_cells:
            if c.get("cell_type") != "code":
                continue
            tm = TEST_CALL_RE.search(cell_source(c))
            if tm:
                test_name = tm.group(1)
                break

        # Prior-exercise function names (only those actually resolved).
        prior_fns = [
            fn for (_, _, fn) in exercise_indices[:idx] if fn and fn != fn_name
        ]

        new_cells: list[dict] = []

        # 1. Banner.
        spa_link = f"{DD_SPA_URL}/?arena_exercise={exercise_id}"
        new_cells.append(
            md_cell(
                f"# Exercise {exercise_id} — {heading_text.removeprefix('Exercise').lstrip(' -')}\n\n"
                f"> Part of [Delta Drills]({DD_SPA_URL}) ARENA practice. "
                f"When the test cell passes, your completion is reported back to your account automatically.\n\n"
                f"**Section:** `{entry.section}`  \n"
                f"**Notebook:** `{Path(entry.notebook_path).name}`  \n"
                f"**Return to Delta Drills:** [{spa_link}]({spa_link})\n"
            )
        )

        # 2. ARENA setup cells (verbatim). Preserves the !pip install + git
        #    download of the ARENA repo on Colab.
        for c in setup_cells:
            new_cells.append(copy.deepcopy(c))

        # 3. DD_TOKEN cell.
        new_cells.append(
            md_cell(
                "## Connect to Delta Drills\n\n"
                "Paste your Delta Drills auth token below so this exercise can report its completion back to your account.\n"
                "You can copy the token from your Delta Drills account page.\n"
            )
        )
        new_cells.append(
            code_cell(
                "# === Delta Drills auth ===\n"
                'DD_TOKEN = ""  # paste your token here, then run this cell\n'
                f'DD_EXERCISE_ID = "{exercise_id}"\n'
                f'DD_BACKEND_URL = "{DD_BACKEND_URL}"\n'
            )
        )

        # 4. Auto-import prior solutions, if any.
        section_pkg = section_dir_name  # e.g. part1_ray_tracing
        if prior_fns:
            import_line = (
                f"from {section_pkg}.solutions import "
                + ", ".join(prior_fns)
            )
            new_cells.append(
                md_cell(
                    "### Prior-exercise solutions (auto-imported)\n\n"
                    "These were imported from ARENA's reference `solutions.py` so you can jump straight into this exercise without "
                    "having implemented every predecessor. Re-implement them yourself if you'd rather build top-to-bottom.\n"
                )
            )
            new_cells.append(code_cell(import_line + "\n"))

        # 5. Exercise body cells (verbatim).
        for c in body_cells:
            new_cells.append(copy.deepcopy(c))

        # 6. Completion beacon.
        beacon_lines = [
            "# === Delta Drills completion beacon ===",
            "import urllib.request as _dd_req, json as _dd_json",
            "",
            "def _dd_report_complete():",
            "    if not DD_TOKEN:",
            "        print('[Delta Drills] DD_TOKEN is empty — completion not reported.')",
            "        return",
            "    try:",
            "        body = _dd_json.dumps({",
            "            'exercise_id': DD_EXERCISE_ID,",
            "            'passed': True,",
            "        }).encode('utf-8')",
            "        req = _dd_req.Request(",
            "            f'{DD_BACKEND_URL}/api/arena/complete',",
            "            data=body,",
            "            headers={",
            "                'Content-Type': 'application/json',",
            "                'Authorization': f'Bearer {DD_TOKEN}',",
            "            },",
            "            method='POST',",
            "        )",
            "        with _dd_req.urlopen(req, timeout=3) as r:",
            "            r.read()",
            "        print(f'[Delta Drills] reported completion of {DD_EXERCISE_ID}')",
            "    except Exception as e:",
            "        print(f'[Delta Drills] beacon failed: {e}')",
            "",
        ]
        if test_name:
            beacon_lines += [
                f"# Wrap tests.test_{test_name} so a passing test fires the beacon.",
                "try:",
                f"    _dd_orig = tests.test_{test_name}",
                f"    def _dd_wrapped(*args, **kwargs):",
                f"        result = _dd_orig(*args, **kwargs)",
                f"        _dd_report_complete()",
                f"        return result",
                f"    tests.test_{test_name} = _dd_wrapped",
                "except AttributeError:",
                "    print('[Delta Drills] no matching test function — call _dd_report_complete() manually when done.')",
            ]
        else:
            beacon_lines += [
                "# This exercise has no automatic test — call `_dd_report_complete()`",
                "# in a new cell once you're satisfied with your answer.",
            ]
        new_cells.append(code_cell("\n".join(beacon_lines) + "\n"))

        out_nb = {
            "cells": new_cells,
            "metadata": nb_meta,
            "nbformat": nb.get("nbformat", 4),
            "nbformat_minor": nb.get("nbformat_minor", 5),
        }
        out_path.write_text(json.dumps(out_nb, indent=1, ensure_ascii=False), encoding="utf-8")
        written += 1

        if not test_name:
            warnings.append(
                f"  {entry.id} ex{idx}: no tests.test_X(...) call found in body cells of `{heading_text}` — beacon left manual."
            )

    return written, warnings


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    entries = parse_manifest()
    print(f"Parsed {len(entries)} curriculum entries from manifest.js")

    total = 0
    all_warnings: list[str] = []
    for entry in entries:
        count, warnings = split_notebook(entry)
        total += count
        all_warnings.extend(warnings)
        print(f"  {entry.id:<40} → {count:>3} files")

    print(f"\nTotal exercise notebooks written: {total} → {OUT_ROOT}")
    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(w)


if __name__ == "__main__":
    main()
