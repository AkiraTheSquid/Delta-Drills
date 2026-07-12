#!/usr/bin/env python3
"""Regression tests for torch support in the grading harness + mech gate
(2026-07-11 torch regen; cases from the codex cross-review).

Run:  This-Directory-Only/backend/.venv/bin/python Local_Deployed_Shared/pipeline/test_torch_grading.py
Exit 0 = all pass.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_sys_path_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_sys_path_root))

from delta_paths import THIS_DIR_ONLY  # noqa: E402

FAILED = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILED
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILED += 1


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


code_runner = load("delta_code_runner", THIS_DIR_ONLY / "backend" / "app" / "code_runner.py")
code_runner.preload_torch()
mech = load("delta_mech_gate", Path(__file__).parent / "mech_gate_candidate.py")
import torch  # noqa: E402  (preloaded above)


def run(user_code, cases):
    results, execution = code_runner.run_function_tests(user_code, cases)
    return results, execution


# 1. multi-element tensor equality: honest passes, wrong fails
r, _ = run("import torch\ndef solve(x):\n    return x * 2\n",
           [{"setup_code": "import torch\nx = torch.tensor([1.0, 2.0])",
             "call": "solve(x)", "expected_expr": "torch.tensor([2.0, 4.0])"}])
check("tensor equality: honest passes", r and r[0].passed, getattr(r[0], "error", ""))

r, _ = run("import torch\ndef solve(x):\n    return x\n",
           [{"setup_code": "import torch\nx = torch.tensor([1.0, 2.0])",
             "call": "solve(x)", "expected_expr": "torch.tensor([2.0, 4.0])"}])
check("tensor equality: wrong fails", r and not r[0].passed)

# 2. tuple containing tensors compares recursively
r, _ = run("import torch\ndef solve(x):\n    return (x.sum(), x + 1)\n",
           [{"setup_code": "import torch\nx = torch.tensor([1.0, 3.0])",
             "call": "solve(x)", "expected_expr": "(torch.tensor(4.0), torch.tensor([2.0, 4.0]))"}])
check("tuple-of-tensors compares", r and r[0].passed, getattr(r[0], "error", ""))

# 3. unseeded torch rng in setup: identity solve must pass (setup re-execs for
#    the expected side; both sides must see the same draws)
r, _ = run("import torch\ndef solve(x):\n    return x\n",
           [{"setup_code": "import torch\nx = torch.rand(3)",
             "call": "solve(x)", "expected_expr": "x"}])
check("torch rng seeded across setup re-exec", r and r[0].passed, getattr(r[0], "error", ""))

# 4. bfloat16: equal tensors must not grade unequal (.numpy() raises on bf16)
r, _ = run("import torch\ndef solve(x):\n    return x + 1\n",
           [{"setup_code": "import torch\nx = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)",
             "call": "solve(x)",
             "expected_expr": "torch.tensor([2.0, 3.0], dtype=torch.bfloat16)"}])
check("bfloat16 equal tensors pass", r and r[0].passed, getattr(r[0], "error", ""))

# 5. grad-carrying tensor return compares (detach path)
r, _ = run("import torch\ndef solve(x):\n    w = x.clone().requires_grad_(True)\n    return w * 2\n",
           [{"setup_code": "import torch\nx = torch.tensor([1.0, 2.0])",
             "call": "solve(x)", "expected_expr": "torch.tensor([2.0, 4.0])"}])
check("requires-grad tensor compares", r and r[0].passed, getattr(r[0], "error", ""))

# 6. numpy-only grading unaffected
r, _ = run("import numpy as np\ndef solve(x):\n    return np.sort(x)\n",
           [{"setup_code": "import numpy as np\nx = np.array([3, 1, 2])",
             "call": "solve(x)", "expected_expr": "np.array([1, 2, 3])"}])
check("numpy grading unaffected", r and r[0].passed, getattr(r[0], "error", ""))

# 7. mech gate _values_equal handles tensors -> non-degenerate flags a
#    constant-expected torch candidate
constant = {
    "id": 0, "question_text": "t", "function_name": "solve",
    "starter_code": "import torch\n\ndef solve(x):\n    return None\n\nx = torch.tensor([1.0])\nprint(solve(x))\n",
    "answer_code": "import torch\n\ndef solve(x):\n    return torch.tensor([1.0, 2.0])\n\nx = torch.tensor([1.0])\nprint(solve(x))\n",
    "test_cases": [
        {"setup_code": "import torch\nx = torch.tensor([9.0])", "call": "solve(x)", "expected_expr": "torch.tensor([1.0, 2.0])"},
        {"setup_code": "import torch\nx = torch.tensor([8.0])", "call": "solve(x)", "expected_expr": "torch.tensor([1.0, 2.0])"},
        {"setup_code": "import torch\nx = torch.tensor([7.0])", "call": "solve(x)", "expected_expr": "torch.tensor([1.0, 2.0])"},
    ],
}
failures = mech.gate_candidate(constant, code_runner)
check("mech gate flags constant torch expecteds",
      any("non_degenerate" in f for f in failures), str(failures)[:120])

check("_values_equal: identical tensors equal",
      mech._values_equal(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])))
check("_values_equal: different tensors differ",
      not mech._values_equal(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0])))

print()
sys.exit(1 if FAILED else 0)
