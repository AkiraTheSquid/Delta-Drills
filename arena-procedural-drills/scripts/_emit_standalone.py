"""Shared helper for emitting standalone per-exercise drill notebooks.

Used by author scripts under `scripts/author_<atom>.py` and by the splitter.

Each notebook = preamble (title + setup + imports + auth + atom recap)
              + exercise (header + stub/test + solution)
              + beacon (single-exercise).

The author scripts assemble a SPEC dict + call `emit_standalone(spec)`. The
helper does the cell layout, source-list normalization, stable cell ids,
and atomic write. No torch / no verify — that's the author's responsibility
(should be a one-line `assert` in the author script, or just careful test
authoring).

Spec shape:
    {
        "atom_id":         "einops-rearrange",      # parent atom slug
        "subtopic":        "Einops: Rearrange",     # EWMA subtopic key
        "topic_folder":    "prereqs_einops",        # under arena-procedural-drills/
        "atom_recap_md":   "## einops.rearrange — quick refresher\\n\\n...",
        "exercise_index":  6,                        # int >= 6 for new (1-5 are split-from-original)
        "exercise_title":  "shape heatmap visualizer",
        "slug":            "shape-heatmap-visualizer",
        "bloom_level":     "Apply",
        "difficulty_num":  3,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords":        ["visualization", "matplotlib"],
        "kcs":             ["rearrange-axis-swap", "rearrange-axis-composition"],
        "lo":              "Visualize how rearrange...",
        "prompt_body":     "Use rearrange + matplotlib to plot...",
        "stub":            "def ex6_*(x): ...",
        "test_body":       "x = t.arange(...); y = ex6_*(x); assert ...",
        "solution_body":   "def ex6_*(x): return ...",
        "solution_notes":  "",                        # may be empty
        "extra_imports":   ["import matplotlib.pyplot as plt"],  # appended to setup
    }
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRILL_ROOT = REPO / "arena-procedural-drills"

DEFAULT_IMPORTS = (
    "import numpy as np\n"
    "import torch as t\n"
    "from torch import Tensor\n"
    "import einops\n"
    "from einops import rearrange, reduce, repeat\n"
    "\n"
    "t.manual_seed(0)\n"
    "np.random.seed(0)"
)


def _md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def _code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": "\n".join(lines),
    }


def _normalize(nb: dict) -> None:
    for c in nb["cells"]:
        s = c.get("source", "")
        if isinstance(s, str):
            lines = [line + "\n" for line in s.split("\n")]
            if lines:
                lines[-1] = lines[-1].rstrip("\n")
            c["source"] = lines


def _stable_ids(nb: dict, prefix: str) -> None:
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"{prefix}-{i:02d}"


def _exercise_header_md(spec: dict) -> dict:
    keywords_str = ", ".join(spec.get("keywords", []))
    kcs_str = ", ".join(f"`{k}`" for k in spec.get("kcs", []))
    return _md(
        f"### Exercise {spec['exercise_index']} — {spec['exercise_title']}",
        "",
        "> ```yaml",
        f"> Difficulty: {spec['difficulty_dots']}",
        f"> Bloom level: {spec['bloom_level']}",
        f"> LO: {spec['lo']}",
        f"> Keywords: {keywords_str}",
        "> ```",
        "",
        f"**KCs targeted:** {kcs_str}",
        "",
        spec["prompt_body"],
    )


def _exercise_code(spec: dict) -> dict:
    ex_id = f"ex{spec['exercise_index']}"
    test_body = spec["test_body"]
    # Indent the test body by 4 spaces so it nests inside def _test_exN.
    indented = "\n".join("    " + line if line else "" for line in test_body.split("\n"))
    return _code(
        spec["stub"],
        "",
        "",
        f"def _test_{ex_id}():",
        indented,
        f"    _dd_passed.add('{ex_id}')",
        f'    print("{ex_id} ✓")',
        "",
        f"_test_{ex_id}()",
    )


def _exercise_solution_md(spec: dict) -> dict:
    lines = [
        "<details><summary>Solution</summary>",
        "",
        "```python",
        spec["solution_body"],
        "```",
    ]
    if spec.get("solution_notes"):
        lines.extend(["", spec["solution_notes"]])
    lines.append("</details>")
    return _md(*lines)


def _single_exercise_beacon(ex_id: str) -> dict:
    return _code(
        "# === Delta Drills completion beacon ===",
        "import urllib.request as _dd_req, json as _dd_json",
        "",
        f"_DD_REQUIRED = {{'{ex_id}'}}",
        "",
        "def report_completion():",
        "    missing = _DD_REQUIRED - _dd_passed",
        "    if missing:",
        '        print(f"[Delta Drills] {sorted(missing)} not yet passing — fix the cell above, then re-run this one.")',
        "        return",
        "    if not DD_TOKEN:",
        "        print('[Delta Drills] DD_TOKEN is empty — completion not reported.')",
        "        return",
        "    body = _dd_json.dumps({",
        f"        'exercise_title': f'procedural-drill:{{DD_ATOM_ID}}:{ex_id}',",
        "        'subtopics': [DD_SUBTOPIC],",
        "        'feedback': 'somewhat',",
        "        'correct': True,",
        "    }).encode('utf-8')",
        "    req = _dd_req.Request(",
        "        f'{DD_BACKEND_URL}/api/practice/arena-rating',",
        "        data=body,",
        "        headers={",
        "            'Content-Type': 'application/json',",
        "            'Authorization': f'Bearer {DD_TOKEN}',",
        "        },",
        "        method='POST',",
        "    )",
        "    try:",
        "        with _dd_req.urlopen(req, timeout=5) as r:",
        "            resp = _dd_json.loads(r.read())",
        "        print(f'[Delta Drills] reported {DD_ATOM_ID} (subtopic={DD_SUBTOPIC!r})')",
        "        print(f'[Delta Drills] EWMA updated: {resp}')",
        "    except Exception as e:",
        "        print(f'[Delta Drills] beacon failed: {e}')",
        "",
        "report_completion()",
    )


def emit_standalone(spec: dict) -> Path:
    """Assemble + write the standalone notebook described by `spec`.

    Returns the output Path (repo-relative is in the caller's responsibility
    to log if needed).
    """
    atom_id = spec["atom_id"]
    subtopic = spec["subtopic"]
    topic_folder = spec["topic_folder"]
    ex_idx = spec["exercise_index"]
    ex_title = spec["exercise_title"]
    slug = spec["slug"]
    ex_id = f"ex{ex_idx}"

    title = f"{atom_id} — ex{ex_idx}: {ex_title}"

    # Build cells
    cells = []

    # Title + intro
    cells.append(_md(
        f"# {title}",
        "",
        "> Procedural drill from [Delta Drills](https://delta-drills.vercel.app).",
        f"> Atom: `{atom_id}`. Running the final beacon cell reports progress against the `{subtopic}` subtopic.",
        "",
        "**Why this is a Colab exercise.** This standalone exercises material the Delta Drills flashcards can't deliver on their own — interactive tensor execution, visualization, or multi-step debugging. Read the prompt, fill in the function body, run the test cell, then run the beacon at the bottom.",
    ))

    # Setup
    cells.append(_md("## Setup"))
    imports_src = DEFAULT_IMPORTS
    for extra in spec.get("extra_imports", []) or []:
        imports_src += "\n" + extra
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": imports_src,
    })

    # Auth
    cells.append(_md(
        "## Connect to Delta Drills",
        "",
        f"Paste your Delta Drills auth token below so this drill can report progress on the `{subtopic}` subtopic. Copy it from your Delta Drills account page.",
        "",
        f"This standalone exercises the atom **`{atom_id}`** (exercise {ex_idx}). Completion fires the beacon at the bottom.",
    ))
    cells.append(_code(
        "# === Delta Drills auth ===",
        'DD_TOKEN = ""  # paste your token here, then run this cell',
        f'DD_ATOM_ID = "{atom_id}"',
        f'DD_SUBTOPIC = "{subtopic}"',
        'DD_BACKEND_URL = "https://delta-drills-backend.fly.dev"',
        "",
        "_dd_passed = set()",
    ))

    # Atom recap
    cells.append(_md(spec["atom_recap_md"]))

    # Exercise triple
    cells.append(_exercise_header_md(spec))
    cells.append(_exercise_code(spec))
    cells.append(_exercise_solution_md(spec))

    # Beacon
    cells.append(_md(
        "## Report completion",
        "",
        "Run the cell below to send your progress to Delta Drills. The beacon fires only if the test cell above passed.",
    ))
    cells.append(_single_exercise_beacon(ex_id))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "delta_drills": {
                "atom_id": atom_id,
                "subtopic": subtopic,
                "drill_kind": "procedural-standalone",
                "template_version": "v0.3-standalone",
                "parent_combined_notebook": None,  # authored fresh, not split
                "exercise_index": ex_idx,
                "exercise": {
                    "id": ex_id,
                    "title": ex_title,
                    "bloom_level": spec.get("bloom_level", ""),
                    "difficulty": spec.get("difficulty_num", 0),
                    "keywords": spec.get("keywords", []),
                    "kcs": spec.get("kcs", []),
                    "lo": spec.get("lo", ""),
                },
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    _normalize(nb)
    _stable_ids(nb, ex_id)

    out_dir = DRILL_ROOT / topic_folder / atom_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ex_idx:02d}-{slug}.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    return out_path
