#!/usr/bin/env python3
"""Execution verifier for authored ERE specs — runs every exercise through a
torch/numpy/einops kernel the way the assembled notebook would.

Per atom, per exercise:
  WORKED: exec(setup + solution_code) must complete with no exception.
  FADED:  exec(setup + reference_fill + test_code + "_test()") must PASS.
          exec(setup + scaffold_code  + test_code + "_test()") must FAIL
          (the blank must be load-bearing — if the scaffold already passes,
           the test under-checks the blank).

Crash isolation: the driver runs ONE subprocess per atom (`--atom <id>`), so a
segfault / out-of-bounds as_strided / hung thread in one atom is contained and
attributed, not fatal to the whole run. Each snippet runs in a fresh namespace
pre-seeded np.random.seed(0)+t.manual_seed(0). Subprocess wall-clock timeout
per atom. Writes verify_exec_report.json; non-zero exit on any worked-run
failure, reference-fill failure, atom crash, or scaffold-that-wrongly-passes.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
AUTHORED = HERE / "authored"
BUNDLES = HERE / "ere_atom_bundles.json"
REPORT = HERE / "verify_exec_report.json"
PY = str(HERE / ".verify_venv" / "bin" / "python")
ATOM_TIMEOUT = 90  # seconds per atom subprocess

# Fallback setup if an atom's template imports cell can't be located.
FALLBACK_SETUP = (
    "import numpy as np\nimport torch\nimport torch as t\nfrom torch import Tensor\n"
    "import einops\nfrom einops import rearrange, reduce, repeat\nimport math\n"
    "np.random.seed(0)\nt.manual_seed(0)\n"
)


def _atom_setup(atom_id: str) -> str:
    """Reproduce the assembled notebook's imports cell for this atom (cloned by
    build_ere_notebooks from the atom's template notebook), so verification sees
    exactly the names the learner's notebook defines — no more, no less."""
    try:
        bundles = {b["atomId"]: b for b in json.loads(BUNDLES.read_text())}
        nb = json.loads((REPO / bundles[atom_id]["templateNotebook"]).read_text())
        for c in nb["cells"]:
            src = "".join(c["source"])
            if c["cell_type"] == "code" and "import" in src and "DD_TOKEN" not in src:
                return src + "\nimport math\nnp.random.seed(0)\nt.manual_seed(0)\n"
    except Exception:  # noqa: BLE001
        pass
    return FALLBACK_SETUP


def _run(setup: str, body: str, call_test: bool) -> str:
    """Exec setup+body (optionally appending _test()). '' on success else err."""
    clean = "\n".join("" if l.lstrip().startswith(("%", "!")) else l
                      for l in body.splitlines())
    src = setup + "\n" + clean + ("\n_test()\n" if call_test else "\n")
    g: dict = {}
    import io, contextlib
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(src, "<ere>", "exec"), g, g)
        return ""
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def _verify_one(spec: dict) -> dict:
    atom = spec.get("atomId", "?")
    setup = _atom_setup(atom)
    fails = []
    for i, w in enumerate(spec.get("worked", []), 1):
        err = _run(setup, w.get("solution_code", ""), call_test=False)
        if err:
            fails.append(f"worked[{i}] run: {err}")
    for i, f in enumerate(spec.get("faded", []), 1):
        ref, scaf, test = f.get("reference_fill", ""), f.get("scaffold_code", ""), f.get("test_code", "")
        if "def _test()" not in test:
            fails.append(f"faded[{i}] no _test()"); continue
        ref_err = _run(setup, ref + "\n" + test, call_test=True)
        if ref_err:
            fails.append(f"faded[{i}] reference_fill FAILS test: {ref_err}")
        scaf_err = _run(setup, scaf + "\n" + test, call_test=True)
        if not scaf_err:
            fails.append(f"faded[{i}] scaffold PASSES test (blank not load-bearing / under-tested)")
    return {"atomId": atom, "ok": not fails, "fails": fails}


def _worker(atom_id: str) -> None:
    spec = json.loads((AUTHORED / f"{atom_id}.json").read_text())
    print("@@RESULT@@" + json.dumps(_verify_one(spec)))


def _driver() -> None:
    specs = sorted(AUTHORED.glob("*.json"))
    results = []
    for n, p in enumerate(specs, 1):
        atom = p.stem
        try:
            proc = subprocess.run([PY, __file__, "--atom", atom],
                                  capture_output=True, text=True, timeout=ATOM_TIMEOUT)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("@@RESULT@@")), None)
            if line:
                results.append(json.loads(line[len("@@RESULT@@"):]))
            else:
                sig = proc.returncode
                results.append({"atomId": atom, "ok": False,
                                "fails": [f"CRASH exit={sig} (segfault/abort during exec)"]})
        except subprocess.TimeoutExpired:
            results.append({"atomId": atom, "ok": False,
                            "fails": [f"TIMEOUT >{ATOM_TIMEOUT}s (likely hung thread/loop)"]})
        if n % 25 == 0:
            print(f"  ...{n}/{len(specs)}", file=sys.stderr)

    REPORT.write_text(json.dumps(results, indent=2))
    hard, soft = [], []
    for r in results:
        h = [f for f in r["fails"] if "scaffold PASSES" not in f]
        s = [f for f in r["fails"] if "scaffold PASSES" in f]
        if h:
            hard.append((r["atomId"], h))
        if s:
            soft.append((r["atomId"], s))
    clean = sum(1 for r in results if r["ok"])
    print(f"\nERE exec-verify: {len(results)} atoms | clean {clean} "
          f"| hard-fail {len(hard)} | scaffold-under-test {len(soft)}")
    for a, fs in hard:
        print(f"  HARD {a}:")
        for f in fs:
            print(f"      {f}")
    for a, fs in soft:
        print(f"  soft {a}: {fs[0]}")
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--atom":
        _worker(sys.argv[2])
    else:
        _driver()
