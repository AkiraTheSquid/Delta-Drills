#!/usr/bin/env python3
"""Build per-atom authoring bundles for the ERE worked/faded tier layer.

For every single-drill atom in the catalog, gather:
  - atom id / label / definition / domain (from concept-graph/vocab/atoms.json)
  - subtopic + module dir + a template notebook path (boilerplate to clone)
  - each existing exercise's concept-refresher md, prompt md, and solution code

These bundles are the reference an opus author agent reads to produce 3 worked +
3 faded *new-but-similar* problems per atom (matching house style + difficulty).

Inputs : ere/catalog_dump.json (from dump_catalog.js), concept-graph/vocab/atoms.json
Output : ere/ere_atom_bundles.json  (regenerable; gitignored)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DUMP = HERE / "catalog_dump.json"
ATOMS = REPO / "concept-graph" / "vocab" / "atoms.json"
OUT = HERE / "ere_atom_bundles.json"

_DETAILS = re.compile(
    r"<details>\s*<summary>([^<]*?solution[^<]*?)</summary>(.*?)</details>",
    re.IGNORECASE | re.DOTALL,
)
_PYBLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _solution_code(cells: list) -> str | None:
    for c in cells:
        if c["cell_type"] != "markdown":
            continue
        m = _DETAILS.search("".join(c["source"]))
        if m:
            blk = _PYBLOCK.search(m.group(2))
            if blk:
                return blk.group(1).rstrip() + "\n"
    return None


def _refresher_md(cells: list) -> str:
    """The teaching/refresher markdown cell: a markdown cell after Setup that is
    NOT the exercise prompt (no 'Difficulty:' / 'LO:' yaml) and not a section header."""
    for c in cells:
        if c["cell_type"] != "markdown":
            continue
        src = "".join(c["source"])
        low = src.lower()
        if len(src) > 300 and "difficulty:" not in low and "<details>" not in low \
                and not src.lstrip().startswith("## Connect") \
                and not src.lstrip().startswith("## Report"):
            return src
    return ""


def _prompt_md(cells: list) -> str:
    for c in cells:
        if c["cell_type"] != "markdown":
            continue
        src = "".join(c["source"])
        if "difficulty:" in src.lower() or re.search(r"###\s+Exercise", src):
            return src
    return ""


def main() -> None:
    rows = json.loads(DUMP.read_text())
    atoms_raw = json.loads(ATOMS.read_text())
    atoms_list = atoms_raw["atoms"] if isinstance(atoms_raw, dict) else atoms_raw
    atom_meta = {a["id"]: a for a in atoms_list}

    # Group single-drill exercises by atom.
    by_atom: dict[str, list[dict]] = {}
    for r in rows:
        if r["isComposite"] or not r["atomId"] or not r["notebookPath"]:
            continue
        by_atom.setdefault(r["atomId"], []).append(r)

    bundles = []
    missing_def = []
    for atom_id, exs in sorted(by_atom.items()):
        exs.sort(key=lambda e: e["exerciseIndex"] or 0)
        meta = atom_meta.get(atom_id, {})
        if not meta:
            missing_def.append(atom_id)
        subtopic = (exs[0]["subtopics"] or [""])[0]
        nb_rel = exs[0]["notebookPath"]
        module_dir = str(Path(nb_rel).parent)  # arena-procedural-drills/<module>/<atom>

        exercises = []
        for e in exs:
            p = REPO / e["notebookPath"]
            if not p.exists():
                continue
            cells = json.loads(p.read_text())["cells"]
            exercises.append({
                "notebookPath": e["notebookPath"],
                "exerciseIndex": e["exerciseIndex"],
                "heading": e["heading"],
                "refresher_md": _refresher_md(cells),
                "prompt_md": _prompt_md(cells),
                "solution_code": _solution_code(cells) or "",
            })

        bundles.append({
            "atomId": atom_id,
            "label": meta.get("label", atom_id),
            "definition": meta.get("definition", ""),
            "domain": meta.get("domain", ""),
            "subtopic": subtopic,
            "moduleDir": module_dir,
            "templateNotebook": nb_rel,  # clone boilerplate cells from here
            "exercises": exercises,
        })

    OUT.write_text(json.dumps(bundles, indent=1))
    print(f"atom bundles: {len(bundles)} -> {OUT.relative_to(REPO)}")
    print(f"  exercises total: {sum(len(b['exercises']) for b in bundles)}")
    print(f"  atoms w/ no atoms.json def: {len(missing_def)}", missing_def[:8])
    print(f"  atoms w/ <3 existing exercises: {sum(1 for b in bundles if len(b['exercises'])<3)}")


if __name__ == "__main__":
    main()
