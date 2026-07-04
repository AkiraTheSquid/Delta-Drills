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
# parseable stub that still halts clearly.
_BAD_STUB = re.compile(
    r"^(?P<indent>[ \t]*)(?P<lhs>[\w\s,()\[\].]+?)\s*=\s*raise\s+NotImplementedError\([^)]*\)\s*(?P<cmt>#.*)?$",
    re.MULTILINE,
)

# Generic pointer that replaces any answer-bearing TODO comment in the FADED
# scaffold. The learner gets the operation/shape guidance from the prompt and
# concept markdown cells above the code cell — the comment must never carry the
# literal answer expression.
_GENERIC_TODO = "# TODO: fill in this step — read the prompt cell above"

# Matches a trailing `# TODO ...` / `#TODO ...` comment (everything from the `#`
# onward). Authored faded scaffolds put the literal answer here, e.g.
#   out = None  # TODO: grad_out * out * (1 - out)
#   raise NotImplementedError()  # TODO: OR over the batch axis to get (H, W)
# We strip the answer text and substitute a generic conceptual pointer, while
# keeping the code structure (the `= None` / `____` placeholder, the halting
# `raise NotImplementedError()`, dict-key blanks, signatures, control flow).
# Note: this is a heuristic that assumes `#` introduces a comment. Scaffolds in
# this corpus never put `#` inside a string literal on a TODO line, so this is
# safe in practice; revisit if that assumption changes.
_TODO_COMMENT = re.compile(r"#\s*TODO\b.*$", re.MULTILINE)


def _sanitize_scaffold(code: str) -> str:
    # FADED tier only (worked + reference_fill paths never call this). Drop any
    # answer-revealing TODO comment, but keep the scaffold's structure so the
    # learner still has signatures, variable names, control flow, and a clear
    # blank to fill.
    def repl(m: "re.Match") -> str:
        ind, lhs = m.group("indent"), m.group("lhs").strip()
        return (f"{ind}{lhs} = None  {_GENERIC_TODO}\n"
                f'{ind}raise NotImplementedError("Complete `{lhs}` above, then delete this line.")')
    code = _BAD_STUB.sub(repl, code)
    code = _TODO_COMMENT.sub(_GENERIC_TODO, code)
    return code


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


# Orientation line at the top of every Concept cell. Tier notebooks open cold
# (a tester landed on "The core BackwardFuncLookup raises KeyError…" with no
# idea what BackwardFuncLookup was) — every name the Concept text references is
# defined in the Setup cell, so point the learner there explicitly.
_CONTEXT_BRIDGE = (
    "_First time on this topic? Run the **Setup** cell above and skim it: every "
    "class and helper mentioned below is defined there. You don't need to have "
    "done any other drill first._"
)


# ---- notebook builders ----------------------------------------------------
def build_worked(bundle: dict, w: dict, idx: int, tpl: dict) -> dict:
    atom = bundle["atomId"]
    title = w.get("title") or f"{atom} — worked example {idx}"
    cells = [
        md(f"# {atom} — worked example {idx}: {title}\n\n"
           f"> Worked example from [Delta Drills](https://delta-drills.vercel.app). "
           f"Atom: `{atom}`.\n\n"
           f"**This is a worked example — read it, run each cell, and follow the reasoning.** "
           f"It's study material, so there's nothing to submit here. Delta Drills hands you "
           f"a hands-on version to complete yourself as you get comfortable with the idea."),
        md("## Setup"),
        tpl["imports"] or code("import numpy as np\nimport torch as t\nimport einops\n"
                               "t.manual_seed(0)\nnp.random.seed(0)"),
        md("## Concept\n\n" + _CONTEXT_BRIDGE + "\n\n"
           + (w.get("concept_md") or bundle.get("definition", ""))),
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
           f"> Practice drill from [Delta Drills](https://delta-drills.vercel.app). "
           f"Atom: `{atom}`. The last cell reports your progress on the "
           f"`{subtopic}` subtopic back to Delta Drills.\n\n"
           f"**Most of the code is already written — complete the one blanked step**, "
           f"run the test to check it, then run the last cell to record your progress."),
        md("## Setup"),
        tpl["imports"] or code("import numpy as np\nimport torch as t\nimport einops"),
        tpl["connect_md"] or md("## Connect to Delta Drills"),
        tpl["auth"] or code(f'DD_TOKEN = ""\nDD_ATOM_ID = "{atom}"\n'
                            f'DD_SUBTOPIC = "{subtopic}"\n'
                            f'DD_BACKEND_URL = "https://delta-drills-backend.fly.dev"\n_dd_passed = set()'),
        md("## Concept\n\n" + _CONTEXT_BRIDGE + "\n\n" + (f.get("concept_md") or "")),
        # NOTE: we intentionally do NOT echo the authored `blank_description`
        # here — those strings spell out the literal answer (e.g. "grad_out
        # times the local Jacobian expressed through out"), which made the
        # faded tier mindless. The learner gets the concept from the Concept +
        # prompt_md cells and the structure from the scaffold; the missing
        # expression they must work out themselves.
        md(f"## Faded exercise {idx}\n\n" + (f.get("prompt_md") or "")
           + "\n\n**Your task:** complete the one blanked step in the code cell below. "
             "The surrounding code, function signatures, and variable names are given — "
             "work out the missing expression yourself, then run the test."),
        code(work + ("\n\n" + test + runner if test else "")),
        md("## Report your progress\n\nRun the cell below to send your progress to "
           "Delta Drills. It only counts if the test above passed."),
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
