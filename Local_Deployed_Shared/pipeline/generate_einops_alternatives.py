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

# Add backend to path
BACKEND_DIR = Path("/home/stellar-thread/Applications/Delta-Drills-Local/This-Directory-Only/backend")
if BACKEND_DIR.exists():
    sys.path.insert(0, str(BACKEND_DIR))
    from app import code_runner
    # Never replace the shipped data with an empty file when this script runs
    # in an environment without torch (for example the lightweight web
    # development interpreter). The candidates must be verified, not merely
    # parsed, before they become learner-visible alternatives.
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

# Dictionary of candidate solutions: qid -> solution_code
CANDIDATES = {}

# 1. Matrix multiplication / dot products / contractions (Numpy / PyTorch)
CANDIDATES[95] = """import torch as t
import einops

def solve(img):
    return einops.einsum(img, t.tensor([0.299, 0.587, 0.114]), '... c, c -> ...')
"""

CANDIDATES[144] = """import torch as t
import einops

def solve(z, v):
    return einops.einsum(z, v, 'n m, m -> n')
"""

CANDIDATES[151] = """import torch as t
import einops

def solve(a, b):
    return a * einops.rearrange(b, 'h w -> h w 1')
"""

# 2. Transposing / flattening / swapping axes
CANDIDATES[23] = """import torch as t
import einops

def solve(z):
    return einops.rearrange(z, 'h w -> (w h)')
"""

CANDIDATES[490] = """import torch as t
import einops

def solve(x):
    \"\"\"Return x collapsed to a single axis.\"\"\"
    return einops.rearrange(x, '... -> (...)')
"""

CANDIDATES[510] = """import torch as t
import einops

def solve(a):
    \"\"\"Return the transpose of a 2-D tensor, and its shape.\"\"\"
    tr = einops.rearrange(a, 'i j -> j i')
    return (tr.tolist(), tuple(tr.shape))
"""

CANDIDATES[558] = """import torch as t
import einops

def solve(rows):
    \"\"\"Return a.T's contents as a plain nested list.\"\"\"
    a = t.tensor(rows)
    return einops.rearrange(a, 'i j -> j i').tolist()
"""

# 3. Reductions (sum, mean, pooling)
CANDIDATES[84] = """import torch as t
import einops

def solve(a, b):
    return einops.reduce([a, b], 'two ... -> ...', 'mean')
"""

CANDIDATES[135] = """import torch as t
import einops

def solve(x):
    return einops.reduce(x, '... h w -> ...', 'sum')
"""

CANDIDATES[220] = """import torch as t
import einops

def solve(x):
    return einops.reduce(x, 'h w -> w', 'sum')
"""

CANDIDATES[418] = """import torch
import einops

def solve(x):
    return einops.reduce(x, 'b c h w -> b c', 'mean')
"""

CANDIDATES[503] = """import torch as t
import einops

def solve(x):
    \"\"\"Return the sum of each ROW.\"\"\"
    return einops.reduce(x, 'h w -> h', 'sum')
"""

CANDIDATES[504] = """import torch as t
import einops

def solve(x):
    \"\"\"Return the row sums with the reduced axis KEPT.\"\"\"
    return einops.reduce(x, 'h w -> h 1', 'sum')
"""

CANDIDATES[505] = """import torch as t
import einops

def solve(x):
    \"\"\"Make every row of a float tensor sum to 1.\"\"\"
    return x / einops.reduce(x, 'h w -> h 1', 'sum')
"""

CANDIDATES[515] = """import torch as t
import einops

def solve(x):
    \"\"\"Scale every row to unit length.\"\"\"
    return x / einops.reduce(x.pow(2), 'h w -> h 1', 'sum').sqrt()
"""

# 4. Ray Tracing (PyTorch: ar-01) - 2D ray grids and ray coordinates
CANDIDATES[1122] = """import torch as t
import einops

def solve(ny, nz, yl, zl):
    y = t.linspace(-yl, yl, ny)
    z = t.linspace(-zl, zl, nz)
    yy = einops.repeat(y, 'y -> (y z)', z=nz)
    zz = einops.repeat(z, 'z -> (y z)', y=ny)
    return einops.rearrange([yy, zz], 'two n -> n two')
"""

CANDIDATES[1123] = """import torch as t
import einops

def solve(ny, nz, yl, zl):
    y = t.linspace(-yl, yl, ny)
    z = t.linspace(-zl, zl, nz)
    yy = einops.repeat(y, 'y -> (y z)', z=nz)
    zz = einops.repeat(z, 'z -> (y z)', y=ny)
    r = t.zeros(ny * nz, 2, 3)
    r[:, 1, 0] = 1
    r[:, 1, 1] = yy
    r[:, 1, 2] = zz
    return r
"""

CANDIDATES[1124] = """import torch as t
import einops

def solve(ny, nz, yl, zl):
    y = t.linspace(-yl, yl, ny)
    z = t.linspace(-zl, zl, nz)
    yy = einops.repeat(y, 'y -> (y z)', z=nz)
    zz = einops.repeat(z, 'z -> (y z)', y=ny)
    r = t.zeros(ny * nz, 2, 3)
    r[:, 1, 0] = 1
    r[:, 1, 1] = yy
    r[:, 1, 2] = zz
    return r[:, 1]
"""

CANDIDATES[1127] = """import torch as t
import einops

def solve(ny, nz, yl, zl):
    y = t.linspace(-yl, yl, ny)
    z = t.linspace(-zl, zl, nz)
    yy = einops.repeat(y, 'y -> (y z)', z=nz)
    zz = einops.repeat(z, 'z -> (y z)', y=ny)
    r = t.zeros(ny * nz, 2, 3)
    r[:, 1, 0] = 1
    r[:, 1, 1] = yy
    r[:, 1, 2] = zz
    return r[:, 0] + 2 * r[:, 1]
"""

CANDIDATES[1128] = """import torch as t
import einops

def solve(ny, nz, yl, zl):
    y = t.linspace(-yl, yl, ny)
    z = t.linspace(-zl, zl, nz)
    yy = einops.repeat(y, 'y -> (y z)', z=nz)
    zz = einops.repeat(z, 'z -> (y z)', y=ny)
    r = t.zeros(ny * nz, 2, 3)
    r[:, 1, 0] = 1
    r[:, 1, 1] = yy
    r[:, 1, 2] = zz
    return einops.rearrange(r, '(ny nz) pts d -> ny nz pts d', ny=ny, nz=nz)[:, 0]
"""

CANDIDATES[1129] = """import torch as t
import einops

def solve(ny, nz, yl, zl):
    y = t.linspace(-yl, yl, ny)
    z = t.linspace(-zl, zl, nz)
    yy = einops.repeat(y, 'y -> (y z)', z=nz)
    zz = einops.repeat(z, 'z -> (y z)', y=ny)
    r = t.zeros(ny * nz, 2, 3)
    r[:, 1, 0] = 1
    r[:, 1, 1] = yy
    r[:, 1, 2] = zz
    r[:, 0, 1] = 1
    r[:, 1, 1] = r[:, 1, 1] - 1
    return r
"""

CANDIDATES[1130] = """import torch as t
import einops

def solve(ny, nz, yl, zl):
    y = t.linspace(-yl, yl, ny)
    z = t.linspace(-zl, zl, nz)
    yy = einops.repeat(y, 'y -> (y z)', z=nz)
    zz = einops.repeat(z, 'z -> (y z)', y=ny)
    r = t.zeros(ny * nz, 2, 3)
    r[:, 1, 0] = 1
    r[:, 1, 1] = yy
    r[:, 1, 2] = zz
    r[:, 1, 0] = 2
    return r[:, 1]
"""

# 5. CNNs & MLPs (PyTorch: ar-02) - flattening, affine combinations, pooling, strided views
CANDIDATES[1192] = """import torch as t
import einops

def solve(x, w, b):
    y = x @ w.T + b
    return einops.rearrange(y, '... out_f -> (...) out_f')
"""

CANDIDATES[1193] = """import torch as t
import einops

def solve(x, w, b):
    y = x @ w.T + b
    return einops.reduce(y, '... out_f -> out_f', 'mean')
"""

CANDIDATES[1201] = """import torch as t
import einops

def solve(x):
    return einops.rearrange(x, 'b ... -> b (...)')
"""

CANDIDATES[1202] = """import torch as t
import einops

def solve(x, w1, b1):
    flat = einops.rearrange(x, 'b ... -> b (...)')
    return flat @ w1.T + b1
"""

CANDIDATES[1203] = """import torch as t
import einops

def solve(x, w1, b1):
    flat = einops.rearrange(x, 'b ... -> b (...)')
    h = (flat @ w1.T + b1).clamp(min=0)
    return h
"""

CANDIDATES[1204] = """import torch as t
import einops

def solve(x, w1, b1, w2, b2):
    flat = einops.rearrange(x, 'b ... -> b (...)')
    h = (flat @ w1.T + b1).clamp(min=0)
    y = h @ w2.T + b2
    return y
"""

CANDIDATES[1205] = """import torch as t
import einops

def solve(x, w1, b1, w2, b2):
    flat = einops.rearrange(x, 'b ... -> b (...)')
    h = (flat @ w1.T + b1).clamp(min=0)
    return einops.reduce(h @ w2.T + b2, 'b c -> c', 'mean')
"""

CANDIDATES[1206] = """import torch as t
import einops

def solve(x, w1, b1, w2, b2):
    flat = einops.rearrange(x, 'b ... -> b (...)')
    h = (flat @ w1.T + b1).clamp(min=0)
    y = h @ w2.T + b2
    return y.argmax(dim=1)
"""

CANDIDATES[1207] = """import torch as t
import einops

def solve(x, w1, b1):
    flat = einops.rearrange(x, 'b ... -> b (...)')
    h = (flat @ w1.T + b1).clamp(min=0)
    return einops.reduce(h, 'b hidden -> hidden', 'mean')
"""

CANDIDATES[1208] = """import torch as t
import einops

def solve(x, w1, b1):
    flat = einops.rearrange(x, 'b ... -> b (...)')
    h = (flat @ w1.T + b1).clamp(min=0)
    return (h > 0).sum(dim=1)
"""

CANDIDATES[1210] = """import torch as t
import einops

def solve(x, w1, b1, w2, b2):
    flat = einops.rearrange(x, 'b ... -> b (...)')
    h = (flat @ w1.T + b1).clamp(min=0)
    y = h @ w2.T + b2
    return y.softmax(dim=-1)
"""

CANDIDATES[1212] = """import torch as t
import einops

def solve(x, w1, b1, w2):
    flat = einops.rearrange(x, 'b ... -> b (...)')
    h0 = (flat @ w1.T + b1)[:, :1].clamp(min=0)
    return einops.einsum(h0, w2[:, :1], 'b one, classes one -> b classes')
"""

CANDIDATES[1219] = """import torch as t
import einops

def solve(x, v):
    return einops.repeat(v, 'n -> m n', m=x.shape[0])
"""

CANDIDATES[1222] = """import torch as t
import einops

def solve(v):
    return einops.einsum(v, v, 'n, m -> n m')
"""

CANDIDATES[1224] = """import torch as t
import einops

def solve(x):
    return einops.repeat(x, 'm n -> m n 2')
"""

# Now test every candidate
verified = {}
failed = []

print(f"Testing {len(CANDIDATES)} einops candidate solutions...")
for qid, code in CANDIDATES.items():
    q = q_by_id.get(qid)
    if not q:
        print(f"Warning: Question {qid} not found in bank!")
        continue
    test_cases = q.get("test_cases", [])
    if not test_cases:
        print(f"Warning: Question {qid} has no test cases!")
        continue
    
    results, exec_res = code_runner.run_function_tests(code, test_cases)
    passed = all(r.passed for r in results)
    if passed:
        verified[str(qid)] = code
        print(f"  ✓ Q{qid} ({q.get('topic')}: {q.get('subtopic')}): PASSED ({len(results)} tests)")
    else:
        failed.append((qid, exec_res.stderr))
        print(f"  ✗ Q{qid}: FAILED - {exec_res.stderr}")

print(f"\nResults: {len(verified)} verified, {len(failed)} failed")

# Save verified solutions to einops_solutions.json
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(verified, f, indent=2)

print(f"Saved {len(verified)} verified solutions to {OUT_PATH}")
