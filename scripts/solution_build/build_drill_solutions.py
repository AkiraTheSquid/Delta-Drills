#!/usr/bin/env python3
"""Generate a runnable solution notebook per procedural-drill exercise.

Each drill notebook embeds its answer in a collapsed
`<details><summary>…solution…</summary>` markdown cell and leaves the work
cell as a `raise NotImplementedError()` stub. This script lifts that embedded
solution into the stub cell so the notebook runs top-to-bottom with the answer
already typed in — exactly what the drill card's "Show answer" button opens.

Output: a sibling `<name>.solution.ipynb` next to every `<name>.ipynb` under
arena-procedural-drills/ (excluding already-generated solution notebooks and
the bank-question solutions/ tree). The frontend derives the path by replacing
`.ipynb` → `.solution.ipynb`, so no catalog change is needed.

Validation: each emitted notebook must (a) contain no remaining
NotImplementedError in any code cell and (b) have every code cell compile.
Failures are reported and NOT emitted.
"""
from __future__ import annotations

import ast
import glob
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRILLS = REPO / "arena-procedural-drills"

# ```python … ``` inside a <details> whose <summary> mentions "solution"
_DETAILS = re.compile(
    r"<details>\s*<summary>([^<]*?solution[^<]*?)</summary>(.*?)</details>",
    re.IGNORECASE | re.DOTALL,
)
_PYBLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_solution_code(cells: list) -> str | None:
    """Return the python code from the first solution <details> block, or None."""
    for c in cells:
        if c["cell_type"] != "markdown":
            continue
        src = "".join(c["source"])
        m = _DETAILS.search(src)
        if not m:
            continue
        blk = _PYBLOCK.search(m.group(2))
        if blk:
            return blk.group(1).rstrip() + "\n"
    return None


def code_cells_ok(cells: list, solved_idx: int) -> tuple[bool, str]:
    """Every code cell must compile. An *un-filled* stub (NotImplementedError
    in a cell we did NOT just fill) is a failure; NotImplementedError inside
    the authored solution cell is allowed — some answers raise it by design."""
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "NotImplementedError" in src and i != solved_idx:
            return False, "NotImplementedError remains in a non-solution cell"
        # strip Colab/IPython magics before compiling
        clean = "\n".join("" if ln.lstrip().startswith(("%", "!")) else ln
                          for ln in src.splitlines())
        try:
            compile(clean, "<cell>", "exec")
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"
    return True, ""


def build_one(path: Path) -> dict:
    nb = json.loads(path.read_text())
    cells = nb["cells"]
    sol = extract_solution_code(cells)
    if not sol:
        return {"path": str(path), "ok": False, "detail": "no solution block"}

    # Replace the first stub cell (NotImplementedError) with the solution code.
    solved_idx = -1
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "NotImplementedError" in "".join(c["source"]):
            c["source"] = sol.splitlines(keepends=True)
            c["outputs"] = []
            c["execution_count"] = None
            solved_idx = i
            break
    if solved_idx < 0:
        return {"path": str(path), "ok": False, "detail": "no stub cell to replace"}

    ok, detail = code_cells_ok(cells, solved_idx)
    if not ok:
        return {"path": str(path), "ok": False, "detail": detail}

    out = path.with_name(path.stem + ".solution.ipynb")
    out.write_text(json.dumps(nb, indent=1))
    return {"path": str(out.relative_to(REPO)), "ok": True, "detail": ""}


def main() -> None:
    nbs = [Path(f) for f in glob.glob(str(DRILLS / "**" / "*.ipynb"), recursive=True)
           if "/solutions/" not in f and not f.endswith(".solution.ipynb")]
    results = [build_one(p) for p in nbs]
    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    print(f"drill notebooks: {len(nbs)}  solutions written: {len(ok)}  failed: {len(bad)}")
    from collections import Counter
    print("failure reasons:", dict(Counter(r["detail"] for r in bad)))
    for r in bad[:20]:
        print("  FAIL", r["path"], "→", r["detail"])
    (Path(__file__).resolve().parent / "drill_solutions_report.json").write_text(
        json.dumps(results, indent=2))

    # Frontend manifest: the set of SOURCE drill notebookPaths that now have a
    # `.solution.ipynb` sibling. The drill card derives the solution path by
    # convention (.ipynb → .solution.ipynb) and consults this set to know
    # whether "Show answer" has a real target (else it falls back to the
    # problem notebook, which carries the collapsed solution).
    sources = sorted(
        Path(r["path"]).as_posix().replace(".solution.ipynb", ".ipynb")
        if r["path"].endswith(".solution.ipynb") else
        Path(r["path"]).relative_to(REPO).as_posix()
        for r in ok
    )
    manifest_js = (
        "// AUTO-GENERATED by scripts/solution_build/build_drill_solutions.py — do not edit.\n"
        "// Drill notebookPaths that have a generated `<name>.solution.ipynb` sibling.\n"
        "window.__drillSolutionPaths = new Set(\n"
        + json.dumps(sources, indent=2)
        + "\n);\n"
    )
    (REPO / "Local_Deployed_Shared" / "practice" / "drill-solutions-manifest.js").write_text(manifest_js)
    print(f"manifest: {len(sources)} drills with solution notebooks")


if __name__ == "__main__":
    main()
