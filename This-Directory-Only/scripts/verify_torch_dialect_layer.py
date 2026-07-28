#!/usr/bin/env python3
"""Run the emitted layer through the real backend grader.

Two things must hold for every record: the canonical answer passes every test
case, and the shipped `return None` starter passes none of them.  A starter
that passes is how a precompute leak shows up.
"""
import json
import sys
from pathlib import Path

REPO = Path("/home/stellar-thread/Applications/Delta-Drills-Local")
sys.path.insert(0, str(REPO / "This-Directory-Only/backend"))

from app.code_runner import preload_torch, run_function_tests  # noqa: E402

assert preload_torch(), 'torch must be preloaded for the fork runner'

LAYER = REPO / ("This-Directory-Only/chatgpt/" + (sys.argv[1] if len(sys.argv) > 1 else "torch_dialect_overrides_einops_einsum.jsonl"))

records = [json.loads(line) for line in LAYER.read_text().splitlines() if line.strip()]

fails, stub_passes, total_cases = [], [], 0
for rec in records:
    cases = rec["test_cases"]
    total_cases += len(cases)

    results, _ = run_function_tests(rec["answer_code"], cases)
    bad = [i for i, r in enumerate(results) if not r.passed]
    if bad or len(results) != len(cases):
        detail = ""
        for i in bad[:1]:
            detail = f" case{i}: {results[i].error or ''} " \
                     f"actual={results[i].actual[:90]} expected={results[i].expected[:90]}"
        fails.append((rec["id"], len(bad), detail))

    stub, _ = run_function_tests(rec["starter_code"], cases)
    n_pass = sum(1 for r in stub if r.passed)
    if n_pass:
        stub_passes.append((rec["id"], n_pass))

print(f"{len(records)} questions / {total_cases} test cases")
print(f"answer failures: {len(fails)}")
for qid, n, detail in fails[:15]:
    print(f"  #{qid}: {n} failing case(s).{detail}")
print(f"starters that pass anything: {len(stub_passes)} {stub_passes[:10]}")
sys.exit(1 if (fails or stub_passes) else 0)
