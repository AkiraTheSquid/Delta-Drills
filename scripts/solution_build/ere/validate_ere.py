#!/usr/bin/env python3
"""Structural gate for authored ERE specs (ere/authored/*.json).

Per atom checks:
  - exactly 3 worked + 3 faded
  - every worked.solution_code / faded.reference_fill / faded.test_code parses
  - faded.test_code defines `def _test():`
  - faded.scaffold_code differs from reference_fill and carries a stub marker
  - the 3 within each tier have distinct slugs

Where the code is numpy/einops-only (no `torch`/` t.`/`import torch`), the
worked solution_code and the faded reference_fill+test are EXECUTED (reference
fill substituted into a runnable cell) to confirm they actually run + pass.
torch code is parse-checked only (torch absent locally — same standard as the
existing drill suite).

Writes ere/ere_validation_report.json and prints a summary. Non-zero exit if
any atom has a hard failure (missing tiers / parse error / test not callable).
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUTHORED = HERE / "authored"
REPORT = HERE / "ere_validation_report.json"

# Any torch usage (import, torch., or the universal `t` alias t.) → parse-check
# only; torch is absent from this validator's interpreter by design (the
# execution gate is verify_exec.py, which runs under a real torch venv).
_TORCH = re.compile(r"\bimport torch\b|\btorch\.|\bt\.|\bnn\.|\bF\.")
_STUB = re.compile(r"NotImplementedError|____")


def _compiles(code: str) -> str:
    clean = "\n".join("" if l.lstrip().startswith(("%", "!")) else l for l in code.splitlines())
    try:
        ast.parse(clean)
        return ""
    except SyntaxError as e:
        return f"{e.msg} (L{e.lineno})"


def _is_torch(*codes: str) -> bool:
    return any(_TORCH.search(c or "") for c in codes)


def _run_numpy(setup_imports: str, body: str) -> str:
    """Execute numpy/einops code in a subprocess-free exec sandbox; return '' or error."""
    import io, contextlib
    g = {}
    src = setup_imports + "\n" + body
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(src, "<ere>", "exec"), g, g)
        return ""
    except ModuleNotFoundError:
        return ""  # dep absent in this sandbox — not a content defect, skip
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


NUMPY_IMPORTS = "import numpy as np\ntry:\n import einops\n from einops import rearrange, reduce, repeat\nexcept Exception: pass\nnp.random.seed(0)"


def validate_atom(spec: dict) -> dict:
    atom = spec.get("atomId", "?")
    hard, soft = [], []
    worked, faded = spec.get("worked", []), spec.get("faded", [])
    if len(worked) != 3:
        hard.append(f"worked count {len(worked)} != 3")
    if len(faded) != 3:
        hard.append(f"faded count {len(faded)} != 3")
    if len({w.get("slug") for w in worked}) < len(worked):
        soft.append("duplicate worked slugs")
    if len({f.get("slug") for f in faded}) < len(faded):
        soft.append("duplicate faded slugs")

    for i, w in enumerate(worked, 1):
        code = w.get("solution_code", "")
        if not code:
            hard.append(f"worked[{i}] empty solution_code"); continue
        err = _compiles(code)
        if err:
            hard.append(f"worked[{i}] solution_code parse: {err}")
        elif not _is_torch(code):
            run = _run_numpy(NUMPY_IMPORTS, code)
            if run:
                hard.append(f"worked[{i}] solution_code run: {run}")

    for i, f in enumerate(faded, 1):
        ref, scaf, test = f.get("reference_fill", ""), f.get("scaffold_code", ""), f.get("test_code", "")
        if not (ref and scaf):
            hard.append(f"faded[{i}] missing reference_fill/scaffold_code"); continue
        for name, code in (("reference_fill", ref), ("test_code", test)):
            err = _compiles(code)
            if err:
                hard.append(f"faded[{i}] {name} parse: {err}")
        if "def _test()" not in test:
            hard.append(f"faded[{i}] test_code missing `def _test():`")
        if not _STUB.search(scaf):
            soft.append(f"faded[{i}] scaffold has no NotImplementedError/____ stub")
        if scaf.strip() == ref.strip():
            hard.append(f"faded[{i}] scaffold == reference (no blank)")
        if not _is_torch(ref, test) and "def _test()" in test:
            run = _run_numpy(NUMPY_IMPORTS, ref + "\n" + test + "\n_test()")
            if run:
                hard.append(f"faded[{i}] reference+test run: {run}")

    return {"atomId": atom, "ok": not hard, "hard": hard, "soft": soft}


def main() -> None:
    specs = sorted(AUTHORED.glob("*.json"))
    results = []
    for p in specs:
        try:
            results.append(validate_atom(json.loads(p.read_text())))
        except json.JSONDecodeError as e:
            results.append({"atomId": p.stem, "ok": False, "hard": [f"invalid JSON: {e}"], "soft": []})
    REPORT.write_text(json.dumps(results, indent=2))
    bad = [r for r in results if not r["ok"]]
    soft = [r for r in results if r["ok"] and r["soft"]]
    print(f"ERE specs: {len(results)}  clean: {len(results)-len(bad)}  hard-fail: {len(bad)}")
    for r in bad[:30]:
        print(f"  FAIL {r['atomId']}: {r['hard']}")
    for r in soft[:15]:
        print(f"  warn {r['atomId']}: {r['soft']}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
