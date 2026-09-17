"""Composite-drill emitter — exercises 2-3 drill atoms together.

Differs from _emit_standalone:
  - `atom_ids: list[str]` instead of single `atom_id`
  - `subtopics: list[str]` instead of single `subtopic`
  - Beacon reports ALL subtopics in one POST (backend already accepts a list)
  - Stored under arena-procedural-drills/composites/<part>/ — separate from per-atom folders

Spec shape (additions vs _emit_standalone):
    {
        "atom_ids":      ["multiply-back", "unbroadcast-pattern"],   # 2-3 atoms
        "subtopics":     ["Backprop: multiply_back", "Backprop: Unbroadcast pattern"],
        "primary_atom":  "multiply-back",                              # for catalog/folder key
        "part":          "part4",                                      # ARENA part for foldering
        "exercise_index": 1,
        "exercise_title": "multiply_back wired through unbroadcast",
        "slug":          "multiply-back-via-unbroadcast",
        "atom_recap_md": "## How these two atoms compose\\n\\n...",  # joint recap
        "prompt_body":   "...",
        "stub_body":     "def cx1_*(...): ...",      # cx prefix = composite-exercise
        "test_body":     "...; cx1_*(...); ...",
        "solution_body": "def cx1_*(...): return ...",
        ...
    }
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRILL_ROOT = REPO / "arena-procedural-drills"
COMPOSITE_ROOT = DRILL_ROOT / "composites"

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


def _stable_ids(nb: dict, ex_id: str) -> None:
    for i, c in enumerate(nb["cells"]):
        c.setdefault("id", f"{ex_id}-cell-{i:02d}")


def _exercise_header_md(spec: dict) -> dict:
    return _md(
        f"### Composite Exercise — {spec['exercise_title']}",
        "",
        f"**Atoms exercised together**: " + ", ".join(f"`{a}`" for a in spec["atom_ids"]),
        "",
        spec["prompt_body"],
    )


def _exercise_code(spec: dict) -> dict:
    test_body = spec["test_body"]
    indented = "\n".join("    " + line if line else "" for line in test_body.split("\n"))
    return _code(
        "# Fill in the function below, then run this cell. The test asserts the composition is correct.",
        "",
        spec["stub_body"],
        "",
        f"def _test_cx{spec['exercise_index']}():",
        indented,
        f"    _dd_passed.add('cx{spec['exercise_index']}')",
        "",
        f"_test_cx{spec['exercise_index']}()",
    )


def _exercise_solution_md(spec: dict) -> dict:
    ex_idx = spec["exercise_index"]
    lines = [
        f"<details><summary>Show solution — cx{ex_idx}</summary>",
        "",
        "```python",
        spec["solution_body"],
        "```",
    ]
    if spec.get("solution_notes"):
        lines.extend(["", spec["solution_notes"]])
    lines.append("</details>")
    return _md(*lines)


def _composite_beacon(spec: dict) -> dict:
    ex_id = f"cx{spec['exercise_index']}"
    subtopics_lit = json.dumps(spec["subtopics"])
    return _code(
        "# === Delta Drills completion beacon (composite — fires for ALL atoms) ===",
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
        f"        'exercise_title': f'composite-drill:{{DD_PRIMARY_ATOM}}:{ex_id}',",
        f"        'subtopics': {subtopics_lit},",
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
        f"        print(f'[Delta Drills] reported composite (atoms={{DD_ATOM_IDS}})')",
        "        print(f'[Delta Drills] EWMA updated: {resp}')",
        "    except Exception as e:",
        "        print(f'[Delta Drills] beacon failed: {e}')",
        "",
        "report_completion()",
    )


def emit_composite(spec: dict) -> Path:
    """Write a composite drill notebook. Returns the output Path."""
    atom_ids = spec["atom_ids"]
    subtopics = spec["subtopics"]
    assert len(atom_ids) == len(subtopics), "atom_ids and subtopics must align"
    assert 2 <= len(atom_ids) <= 4, f"composites should exercise 2-4 atoms, got {len(atom_ids)}"

    primary_atom = spec["primary_atom"]
    part = spec["part"]
    ex_idx = spec["exercise_index"]
    ex_title = spec["exercise_title"]
    slug = spec["slug"]
    ex_id = f"cx{ex_idx}"

    title = f"{primary_atom} composite — cx{ex_idx}: {ex_title}"

    cells = []
    cells.append(_md(
        f"# {title}",
        "",
        f"> Composite procedural drill from [Delta Drills](https://delta-drills.vercel.app).",
        f"> Exercises {len(atom_ids)} atoms together: " + ", ".join(f"`{a}`" for a in atom_ids),
        f"> Running the final beacon reports progress against all {len(subtopics)} subtopics.",
        "",
        "**Why composite drills.** Single-atom drills test atomic skills in isolation. Composite drills test the COMPOSITION — how atoms wire together in real ARENA code. Passing this drill demonstrates you can apply the atoms jointly, not just individually.",
    ))

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

    cells.append(_md(
        "## Connect to Delta Drills",
        "",
        "Paste your Delta Drills auth token below. Beacon will report progress against ALL atoms exercised by this composite.",
    ))
    cells.append(_code(
        "# === Delta Drills auth (composite) ===",
        'DD_TOKEN = ""  # paste token, then run',
        f'DD_PRIMARY_ATOM = "{primary_atom}"',
        f'DD_ATOM_IDS = {json.dumps(atom_ids)}',
        f'DD_SUBTOPICS = {json.dumps(subtopics)}',
        'DD_BACKEND_URL = "https://delta-drills-backend.fly.dev"',
        "",
        "_dd_passed = set()",
    ))

    cells.append(_md(spec["atom_recap_md"]))
    cells.append(_exercise_header_md(spec))
    cells.append(_exercise_code(spec))
    cells.append(_exercise_solution_md(spec))

    cells.append(_md(
        "## Report completion",
        "",
        f"Run the cell below to send progress to Delta Drills. The beacon fires once and reports all {len(subtopics)} subtopics together.",
    ))
    cells.append(_composite_beacon(spec))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "delta_drills": {
                "atom_ids": atom_ids,
                "primary_atom": primary_atom,
                "subtopics": subtopics,
                "drill_kind": "procedural-composite",
                "template_version": "v0.4-composite",
                "arena_part": part,
                "exercise_index": ex_idx,
                "exercise": {
                    "id": ex_id,
                    "title": ex_title,
                    "bloom_level": spec.get("bloom_level", "Apply"),
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

    out_dir = COMPOSITE_ROOT / part
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ex_idx:03d}-{slug}.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    return out_path
