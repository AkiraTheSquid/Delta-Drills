#!/usr/bin/env python3
"""Split each combined drill notebook into per-exercise standalone notebooks.

Reads `arena-procedural-drills/prereqs_*/<atom-slug>.ipynb`, walks its cells
to locate the shared preamble (setup + auth + atom recap) and each
exercise triple (header md → code stub+test → solution md). Emits one
standalone notebook per exercise into `prereqs_*/<atom-slug>/NN-<slug>.ipynb`
with a single-exercise completion beacon, then deletes the combined source
notebook.

Pure JSON manipulation. No torch exec, no verify gate — the source
notebooks already passed their builder-time verify. The split preserves
every cell verbatim except the beacon, which is rewritten to track only
the single exercise the standalone notebook contains.

Run from anywhere:
    python3 arena-procedural-drills/scripts/split_to_standalone.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRILL_ROOT = REPO / "arena-procedural-drills"

# Combined drill notebooks to split. (path-relative, atom-slug pairs.)
DRILLS = [
    ("prereqs_einops/einops-rearrange.ipynb",            "einops-rearrange"),
    ("prereqs_einops/einops-reduce.ipynb",               "einops-reduce"),
    ("prereqs_einops/einops-repeat.ipynb",               "einops-repeat"),
    ("prereqs_einops/einops-einsum.ipynb",               "einops-einsum"),
    ("prereqs_numpy/tensor-zeros-init.ipynb",            "tensor-zeros-init"),
    ("prereqs_numpy/tensor-item-scalar.ipynb",           "tensor-item-scalar"),
    ("prereqs_numpy/broadcasting-rules.ipynb",           "broadcasting-rules"),
    ("prereqs_numpy/boolean-mask-identity-replace.ipynb","boolean-mask-identity-replace"),
    ("prereqs_numpy/tensor-unbind.ipynb",                "tensor-unbind"),
    ("prereqs_numpy/rotation-matrix-3d-y-axis.ipynb",    "rotation-matrix-3d-y-axis"),
    ("prereqs_numpy/as-strided-noncontig-source.ipynb",  "as-strided-noncontig-source"),
]


def _src(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60]


def _split_cells(cells: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[tuple[int, str, list[dict]]]]:
    """Return (preamble, atom_recap, exercise_triples, beacon_unused_kept_for_signature).

    preamble: setup (title + ## Setup md + imports + ## Connect md + auth code)
    atom_recap: the explainer cell(s) between auth and first exercise
    exercises: list of (n, title, [header, code, solution]) tuples
    """
    preamble: list[dict] = []
    atom_recap: list[dict] = []
    exercises: list[tuple[int, str, list[dict]]] = []

    i = 0
    n = len(cells)

    # 1. Preamble: everything up to and including the auth code cell.
    #    Auth code is identified by its content containing `DD_TOKEN`.
    while i < n:
        s = _src(cells[i])
        preamble.append(cells[i])
        if cells[i].get("cell_type") == "code" and "DD_TOKEN" in s:
            i += 1
            break
        i += 1

    # 2. Atom recap: cells between auth and the first "### Exercise" heading.
    while i < n:
        s = _src(cells[i])
        if cells[i].get("cell_type") == "markdown" and re.search(r"^###\s+Exercise\s+\d+", s, re.M):
            break
        atom_recap.append(cells[i])
        i += 1

    # 3. Exercise triples: header md → code stub+test → solution md.
    while i < n:
        s = _src(cells[i])
        m = re.match(r"^###\s+Exercise\s+(\d+)\s+—\s+(.+?)$", s, re.M)
        if not m:
            # Once we stop matching Exercise headers, we've hit the beacon
            # preamble (the "## Done" markdown). Stop harvesting exercises.
            break
        ex_num = int(m.group(1))
        ex_title = m.group(2).strip()
        triple = [cells[i]]
        i += 1
        # Code cell next.
        if i < n and cells[i].get("cell_type") == "code":
            triple.append(cells[i])
            i += 1
        # Solution markdown next.
        if i < n and cells[i].get("cell_type") == "markdown" and "<details><summary>Solution</summary>" in _src(cells[i]):
            triple.append(cells[i])
            i += 1
        if len(triple) != 3:
            raise ValueError(f"exercise {ex_num} ({ex_title}) — expected header+code+solution, got {len(triple)} cells")
        exercises.append((ex_num, ex_title, triple))

    return preamble, atom_recap, exercises, []  # last slot unused now


def _single_exercise_beacon(ex_id: str) -> dict:
    """Beacon cell that only requires `ex_id` to fire."""
    lines = [
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
        "        'feedback': 'somewhat',  # single-exercise standalone — neutral signal",
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
    ]
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": "\n".join(lines),
    }


def _beacon_preamble_md() -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": "## Report completion\n\nRun the cell below to send your progress to Delta Drills. The beacon fires only if the test cell above passed.",
    }


def _normalize_sources(nb: dict) -> None:
    """Convert any string `source` into list-of-line form (nbformat strict)."""
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


def split_one(rel_path: str, atom_slug: str) -> list[Path]:
    src_path = DRILL_ROOT / rel_path
    if not src_path.exists():
        print(f"[skip] {rel_path} not found")
        return []

    nb = json.loads(src_path.read_text())
    cells = nb["cells"]
    src_meta = nb.get("metadata", {})
    dd_meta = src_meta.get("delta_drills", {})

    preamble, atom_recap, exercises, _ = _split_cells(cells)

    out_dir = src_path.parent / atom_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for n, ex_title, triple in exercises:
        slug = _slugify(ex_title)
        out_path = out_dir / f"{n:02d}-{slug}.ipynb"

        # Build the standalone notebook. Cells: preamble + atom_recap +
        # exercise triple + beacon preamble + single-ex beacon.
        out_cells = (
            [json.loads(json.dumps(c)) for c in preamble]
            + [json.loads(json.dumps(c)) for c in atom_recap]
            + [json.loads(json.dumps(c)) for c in triple]
            + [_beacon_preamble_md(), _single_exercise_beacon(f"ex{n}")]
        )

        # Restamp the title heading on the first cell so the in-notebook H1
        # matches the standalone exercise (good for the Ctrl+F target and
        # general readability).
        if out_cells and out_cells[0].get("cell_type") == "markdown":
            first = _src(out_cells[0])
            # Replace the top "# <stuff>" line with the per-exercise title.
            new_title = f"# {atom_slug} — ex{n}: {ex_title}"
            first_new = re.sub(r"^#\s+.+$", new_title, first, count=1, flags=re.M)
            out_cells[0]["source"] = first_new

        # Per-exercise metadata for downstream tooling. Keep the parent
        # atom_id + subtopic so EWMA still aggregates per atom.
        per_ex_meta = None
        for ex_md in dd_meta.get("exercises", []) or []:
            if ex_md.get("id") == f"ex{n}":
                per_ex_meta = ex_md
                break

        out_nb = {
            "cells": out_cells,
            "metadata": {
                "kernelspec": src_meta.get("kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"}),
                "language_info": src_meta.get("language_info", {"name": "python"}),
                "delta_drills": {
                    "atom_id": dd_meta.get("atom_id", atom_slug),
                    "subtopic": dd_meta.get("subtopic", ""),
                    "drill_kind": "procedural-standalone",
                    "template_version": "v0.3-standalone",
                    "parent_combined_notebook": rel_path,
                    "exercise_index": n,
                    "exercise": per_ex_meta or {"id": f"ex{n}", "title": ex_title},
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        _normalize_sources(out_nb)
        _stable_ids(out_nb, f"ex{n}")
        out_path.write_text(json.dumps(out_nb, indent=1))
        written.append(out_path)
        print(f"  wrote {out_path.relative_to(REPO)}")

    # Delete the original combined notebook now that all splits succeeded.
    src_path.unlink()
    print(f"  deleted source: {rel_path}")
    return written


def main() -> int:
    all_written: list[Path] = []
    for rel_path, atom_slug in DRILLS:
        print(f"\n[split] {rel_path}  →  {Path(rel_path).parent}/{atom_slug}/")
        try:
            all_written.extend(split_one(rel_path, atom_slug))
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            return 1
    print(f"\nwrote {len(all_written)} standalone notebooks across {len(DRILLS)} atoms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
