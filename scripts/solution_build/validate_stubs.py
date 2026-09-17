#!/usr/bin/env python3
"""Validate authored starter_code stubs for the stdout-mode bank questions.

A good stub: compiles, keeps a print(...) scaffold, leaves a TODO for the
student, and is NOT the full answer (must differ from answer_code and must not
reproduce the expected output on its own).

Usage: validate_stubs.py <stubs.jsonl>
Exit 0 iff every stub passes.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
Q = {r["id"]: r for r in json.loads((HERE / "stub_inputs.json").read_text())}


def check(qid: int, stub: str) -> tuple[bool, str]:
    q = Q.get(qid)
    if q is None:
        return False, "unknown id"
    if not stub or not stub.strip():
        return False, "empty"
    try:
        compile(stub, "<stub>", "exec")
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    if "print(" not in stub:
        return False, "no print() scaffold"
    if "TODO" not in stub and "your solution" not in stub.lower():
        return False, "no TODO marker"
    if stub.strip() == (q["answer_code"] or "").strip():
        return False, "stub equals answer"
    return True, ""


def main() -> None:
    src = Path(sys.argv[1])
    items = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    out = [(it["id"], *check(it["id"], it.get("starter_code", ""))) for it in items]
    bad = [(i, d) for i, ok, d in out if not ok]
    print(json.dumps({"total": len(out), "pass": len(out) - len(bad),
                      "fail": len(bad), "fails": dict(bad)}, indent=2))
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
