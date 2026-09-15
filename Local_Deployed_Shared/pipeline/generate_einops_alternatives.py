#!/usr/bin/env python3
"""
Generate and verify alternative einops solutions for questions in the question bank.
Every generated solution is verified against the question's actual test_cases.
Outputs verified solutions to Local_Deployed_Shared/practice/einops_solutions.json.
"""

import os
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED_DIR = HERE.parent
QUESTIONS_PATH = SHARED_DIR / "questions.json"
OUT_PATH = SHARED_DIR / "practice" / "einops_solutions.json"
POOL_PATH = HERE / "candidate_einops_pool.json"

# Add backend to path
BACKEND_DIR = Path("/home/stellar-thread/Applications/Delta-Drills-Local/This-Directory-Only/backend")
if BACKEND_DIR.exists():
    sys.path.insert(0, str(BACKEND_DIR))
    from app import code_runner
    if not code_runner.preload_torch():
        raise RuntimeError(
            "PyTorch unavailable. Re-run with the backend virtualenv: "
            "/home/stellar-thread/Applications/Delta-Drills-Local/This-Directory-Only/backend/.venv/bin/python"
        )
else:
    raise RuntimeError("Backend directory not found")

with open(QUESTIONS_PATH, encoding="utf-8") as f:
    questions = json.load(f)

q_by_id = {q["id"]: q for q in questions}

# Load candidates from candidate_einops_pool.json
if POOL_PATH.exists():
    with open(POOL_PATH, encoding="utf-8") as f:
        pool_raw = json.load(f)
    CANDIDATES = {int(k): v for k, v in pool_raw.items()}
else:
    CANDIDATES = {}

# Now test every candidate
verified = {}
failed = []

print(f"Testing {len(CANDIDATES)} einops candidate solutions...")
for qid, code in sorted(CANDIDATES.items()):
    q = q_by_id.get(qid)
    if not q:
        print(f"Warning: Question {qid} not found in bank!")
        continue
    test_cases = q.get("test_cases", [])
    if not test_cases:
        continue
    
    results, exec_res = code_runner.run_function_tests(code, test_cases)
    passed = all(r.passed for r in results)
    if passed:
        verified[str(qid)] = code
    else:
        failed.append((qid, exec_res.stderr))
        print(f"  ✗ Q{qid}: FAILED - {exec_res.stderr}")

print(f"\nResults: {len(verified)} verified, {len(failed)} failed")
ag_count = sum(1 for c in verified.values() if "Solution created by Antigravity" in c)
baseline_count = len(verified) - ag_count
print(f"Provenance: {baseline_count} baseline original, {ag_count} created by Antigravity")

# Save verified solutions to einops_solutions.json
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(verified, f, indent=2, ensure_ascii=False)
    f.write("\n")

print(f"Saved {len(verified)} verified solutions to {OUT_PATH}")
