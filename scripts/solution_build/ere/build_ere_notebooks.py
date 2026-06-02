#!/usr/bin/env python3
"""Assemble ERE worked/faded tier notebooks from authored specs.

Reads authored specs (ere/authored/*.json — one per atom, schema below) and the
atom bundles (ere/ere_atom_bundles.json, for the boilerplate template notebook),
and emits per-atom tier notebooks under:

  <moduleDir>/ere/worked-<NN>-<slug>.ipynb   (study-only: no auth/beacon)
  <moduleDir>/ere/faded-<NN>-<slug>.ipynb    (completion problem: auth+test+beacon)
  <moduleDir>/ere/faded-<NN>-<slug>.solution.ipynb  (reference fill in the stub)

Boilerplate cells (imports/seeds, Connect, auth, Report, beacon) are CLONED from
the atom's existing template notebook so torch/numpy/einops + the Delta Drills
beacon plumbing match house style exactly. The assembler owns the beacon marker
wiring (worked tier omits it entirely); the author agent only supplies content.

Also writes the frontend tier manifest:
  Local_Deployed_Shared/practice/ere-tiers-manifest.js
  window.__ereTiers = { "<atomId>": { worked: [paths], faded: [paths] } }

Authored spec schema (per atom):
{
  "atomId": str,
  "worked": [ {slug, title, concept_md, walkthrough_md, solution_code}, x3 ],
  "faded":  [ {slug, title, concept_md, prompt_md, scaffold_code,
               reference_fill, test_code, blank_description}, x3 ]
}
The faded `test_code` MUST define `def _test():` with assertions (no call).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
AUTHORED = HERE / "authored"
BUNDLES = HERE / "ere_atom_bundles.json"
MANIFEST = REPO / "Local_Deployed_Shared" / "practice" / "ere-tiers-manifest.js"


# ---- cell helpers ---------------------------------------------------------
def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _lines(text)}


def _lines(text: str) -> list[str]:
    text = text.rstrip("\n")
    return [l + "\n" for l in text.split("\n")[:-1]] + [text.split("\n")[-1]] if text else [""]


def _clone(cells: list, predicate) -> dict | None:
    for c in cells:
        if predicate(c):
            # deep copy via json round-trip; reset code outputs
            cc = json.loads(json.dumps(c))
            if cc["cell_type"] == "code":
                cc["outputs"] = []
                cc["execution_count"] = None
            return cc
    return None


def _src(c: dict) -> str:
    return "".join(c["source"])


# Agents sometimes emit `lhs = raise NotImplementedError()  # cmt`, which is
# invalid python (raise is a statement, not an expression). Rewrite it to a
# parseable stub that still halts clearly and shows the learner the target.
_BAD_STUB = re.compile(
    r"^(?P<indent>[ \t]*)(?P<lhs>[\w\s,()\[\].]+?)\s*=\s*raise\s+NotImplementedError\([^)]*\)\s*(?P<cmt>#.*)?$",
    re.MULTILINE,
)


def _sanitize_scaffold(code: str) -> str:
    def repl(m: "re.Match") -> str:
        ind, lhs, cmt = m.group("indent"), m.group("lhs").strip(), (m.group("cmt") or "").strip()
        note = cmt[1:].strip() if cmt.startswith("#") else ""
        return (f"{ind}{lhs} = None  # TODO: {note}\n"
                f'{ind}raise NotImplementedError("Complete `{lhs}` above, then delete this line.")')
    return _BAD_STUB.sub(repl, code)


def _template_cells(bundle: dict) -> dict:
    """Pull the boilerplate cells from the atom's template notebook by role."""
    nb = json.loads((REPO / bundle["templateNotebook"]).read_text())
    cells = nb["cells"]
    return {
        "imports": _clone(cells, lambda c: c["cell_type"] == "code"
                          and "import" in _src(c) and "DD_TOKEN" not in _src(c)),
        "connect_md": _clone(cells, lambda c: c["cell_type"] == "markdown"
                             and _src(c).lstrip().startswith("## Connect")),
        "auth": _clone(cells, lambda c: c["cell_type"] == "code" and "DD_TOKEN" in _src(c)),
        "report_md": _clone(cells, lambda c: c["cell_type"] == "markdown"
                            and _src(c).lstrip().startswith("## Report")),
        "beacon": _clone(cells, lambda c: c["cell_type"] == "code" and "_DD_REQUIRED" in _src(c)),
    }


def _rewire_beacon(beacon: dict, marker: str, atom_id: str) -> dict:
    src = _src(beacon)
    src = re.sub(r"_DD_REQUIRED\s*=\s*\{[^}]*\}", f"_DD_REQUIRED = {{'{marker}'}}", src)
    src = re.sub(r"procedural-drill:\{DD_ATOM_ID\}:ex\d+",
                 f"procedural-drill:{{DD_ATOM_ID}}:{marker}", src)
    beacon["source"] = _lines(src)
    return beacon


# ---- notebook builders ----------------------------------------------------
def build_worked(bundle: dict, w: dict, idx: int, tpl: dict) -> dict:
    atom = bundle["atomId"]
    title = w.get("title") or f"{atom} — worked example {idx}"
    cells = [
        md(f"# {atom} — worked example {idx}: {title}\n\n"
           f"> Worked example from [Delta Drills](https://delta-drills.vercel.app). "
           f"Atom: `{atom}`.\n\n"
           f"**This is a worked example — read it, run it, follow the reasoning.** "
           f"It is study material, not a graded drill (no completion beacon). "
           f"When the steps feel obvious, move to the faded version, then the full drill."),
        md("## Setup"),
        tpl["imports"] or code("import numpy as np\nimport torch as t\nimport einops\n"
                               "t.manual_seed(0)\nnp.random.seed(0)"),
        md("## Concept\n\n" + (w.get("concept_md") or bundle.get("definition", ""))),
        md("## Worked solution\n\n" + (w.get("walkthrough_md") or "")),
        code(w["solution_code"]),
    ]
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def build_faded(bundle: dict, f: dict, idx: int, tpl: dict, filled: bool) -> dict:
    atom = bundle["atomId"]
    subtopic = bundle["subtopic"]
    marker = f"faded{idx}"
    title = f.get("title") or f"{atom} — faded example {idx}"
    work = f["reference_fill"] if filled else _sanitize_scaffold(f["scaffold_code"])
    test = f.get("test_code", "")
    runner = (
        f"\n\ntry:\n    _test()\n    _dd_passed.add('{marker}')\n"
        f"    print('[Delta Drills] {marker} passed.')\n"
        f"except AssertionError as _e:\n    print('Test failed:', _e)"
    )
    cells = [
        md(f"# {atom} — faded example {idx}: {title}\n\n"
           f"> Faded drill from [Delta Drills](https://delta-drills.vercel.app). "
           f"Atom: `{atom}`. Running the beacon reports progress on the "
           f"`{subtopic}` subtopic.\n\n"
           f"**Most of the solution is filled in — complete the one blanked step**, "
           f"run the test, then fire the beacon. Less scaffolding than the worked "
           f"example, more than the full drill."),
        md("## Setup"),
        tpl["imports"] or code("import numpy as np\nimport torch as t\nimport einops"),
        tpl["connect_md"] or md("## Connect to Delta Drills"),
        tpl["auth"] or code(f'DD_TOKEN = ""\nDD_ATOM_ID = "{atom}"\n'
                            f'DD_SUBTOPIC = "{subtopic}"\n'
                            f'DD_BACKEND_URL = "https://delta-drills-backend.fly.dev"\n_dd_passed = set()'),
        md("## Concept\n\n" + (f.get("concept_md") or "")),
        md(f"## Faded exercise {idx}\n\n" + (f.get("prompt_md") or "")
           + (f"\n\n**Fill in:** {f.get('blank_description','')}" if f.get("blank_description") else "")),
        code(work + ("\n\n" + test + runner if test else "")),
        md("## Report completion\n\nRun the cell below to report progress. "
           "The beacon fires only if the test above passed."),
        _rewire_beacon(tpl["beacon"], marker, atom) if tpl["beacon"]
        else code(f"# beacon unavailable for {atom}"),
        md("<details><summary>Solution</summary>\n\n```python\n"
           + f["reference_fill"].rstrip() + "\n```\n</details>"),
    ]
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


# ---- driver ---------------------------------------------------------------
def main() -> None:
    bundles = {b["atomId"]: b for b in json.loads(BUNDLES.read_text())}
    tiers: dict[str, dict] = {}
    written = {"worked": 0, "faded": 0, "faded_solution": 0}
    skipped = []

    for spec_path in sorted(AUTHORED.glob("*.json")):
        spec = json.loads(spec_path.read_text())
        atom = spec.get("atomId")
        bundle = bundles.get(atom)
        if not bundle:
            skipped.append((spec_path.name, "no bundle for atomId"))
            continue
        tpl = _template_cells(bundle)
        out_dir = REPO / bundle["moduleDir"] / "ere"
        out_dir.mkdir(parents=True, exist_ok=True)
        entry = {"worked": [], "faded": []}

        for i, w in enumerate(spec.get("worked", []), 1):
            if not w.get("solution_code"):
                continue
            slug = (w.get("slug") or f"w{i}")[:60]
            nb = build_worked(bundle, w, i, tpl)
            rel = f"{bundle['moduleDir']}/ere/worked-{i:02d}-{slug}.ipynb"
            (REPO / rel).write_text(json.dumps(nb, indent=1))
            entry["worked"].append(rel)
            written["worked"] += 1

        for i, f in enumerate(spec.get("faded", []), 1):
            if not (f.get("scaffold_code") and f.get("reference_fill")):
                continue
            slug = (f.get("slug") or f"f{i}")[:60]
            nb = build_faded(bundle, f, i, tpl, filled=False)
            rel = f"{bundle['moduleDir']}/ere/faded-{i:02d}-{slug}.ipynb"
            (REPO / rel).write_text(json.dumps(nb, indent=1))
            sol = build_faded(bundle, f, i, tpl, filled=True)
            (REPO / rel.replace(".ipynb", ".solution.ipynb")).write_text(json.dumps(sol, indent=1))
            entry["faded"].append(rel)
            written["faded"] += 1
            written["faded_solution"] += 1

        if entry["worked"] or entry["faded"]:
            tiers[atom] = entry

    MANIFEST.write_text(
        "// AUTO-GENERATED by scripts/solution_build/ere/build_ere_notebooks.py — do not edit.\n"
        "// Per-atom ERE tier notebooks, keyed by atomId. The drill card's adaptive\n"
        "// tier selector picks worked (low mastery) / faded (mid) / full (high).\n"
        "window.__ereTiers = " + json.dumps(tiers, indent=1) + ";\n"
    )
    print(f"atoms with tiers: {len(tiers)}")
    print(f"notebooks written: {written}")
    print(f"manifest: {MANIFEST.relative_to(REPO)}")
    if skipped:
        print("skipped:", skipped[:10])


if __name__ == "__main__":
    main()
