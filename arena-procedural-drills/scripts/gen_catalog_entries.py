#!/usr/bin/env python3
"""Walk all standalone .ipynb files under arena-procedural-drills/prereqs_* and
emit JS array literal entries for drills-catalog.js.

Usage:
    python3 arena-procedural-drills/scripts/gen_catalog_entries.py
        # prints the JS entries to stdout

Reads each notebook's metadata.delta_drills.{atom_id, subtopic, exercise_index,
exercise.title} + the top-level # markdown heading. Outputs one E(...) line
per notebook, grouped by atom. Drop the output between the `window.DRILLS_CATALOG = [`
and `];` lines in `Local_Deployed_Shared/practice/drills-catalog.js`.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "arena-procedural-drills"


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def _jsstr(s) -> str:
    return json.dumps(s, ensure_ascii=False)


def main() -> int:
    by_atom: dict[str, list[dict]] = {}
    for topic_dir in sorted(ROOT.glob("prereqs_*")):
        if not topic_dir.is_dir():
            continue
        for atom_dir in sorted(topic_dir.iterdir()):
            if not atom_dir.is_dir():
                continue
            for nb_path in sorted(atom_dir.glob("*.ipynb")):
                nb = json.loads(nb_path.read_text())
                dd = nb.get("metadata", {}).get("delta_drills", {})
                ex = dd.get("exercise", {}) or {}
                first = nb["cells"][0]
                src = _src(first)
                heading = src.splitlines()[0].lstrip("# ").strip()
                atom = dd.get("atom_id") or atom_dir.name
                by_atom.setdefault(atom, []).append({
                    "subtopic": dd.get("subtopic", ""),
                    "ex_idx": dd.get("exercise_index", 0),
                    "ex_title": ex.get("title", ""),
                    "rel_path": str(nb_path.relative_to(REPO)),
                    "heading": heading,
                })

    for atom in sorted(by_atom.keys()):
        print(f"    // {atom}")
        for e in sorted(by_atom[atom], key=lambda x: x["ex_idx"]):
            print(f"    E({_jsstr(atom)}, {e['ex_idx']}, {_jsstr(e['ex_title'])}, {_jsstr(e['subtopic'])}, {_jsstr(e['rel_path'])}, {_jsstr(e['heading'])}),")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
