#!/usr/bin/env python3
"""generate_test_case_round2.py — synthesize a second varied test case per question.

For each function-mode question in questions_full.json, derive an alternate
fixture (different shape, seed, or value range), validate it against the
canonical solution via the existing harness, and write the accepted alt
test_cases to function_mode_test_cases_extra.jsonl.

Output schema (one JSON object per line):
  {"id": <int>, "test_cases_extra": [<one new test_case dict>]}

The override loader appends test_cases_extra to test_cases (see backend
_load_csv_into and exporter equivalent) so the validator runs both cases.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

# ── pipeline bootstrap ──
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from delta_paths import THIS_DIR_ONLY, get_backend_python, get_chatgpt_runtime_dir
from validate_function_bank import synthesize_solution_code

QUESTIONS_PATH = THIS_DIR_ONLY / "questions_full.json"
CHATGPT_RUNTIME_DIR = get_chatgpt_runtime_dir()
OUTPUT_PATH = CHATGPT_RUNTIME_DIR / "function_mode_test_cases_extra.jsonl"
BACKEND_PYTHON = get_backend_python()


def load_code_runner():
    path = THIS_DIR_ONLY / "backend" / "app" / "code_runner.py"
    spec = importlib.util.spec_from_file_location("delta_code_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if BACKEND_PYTHON.exists():
        module.sys.executable = str(BACKEND_PYTHON)
    return module


# Pattern-specific mutation regexes. Each returns a list of mutated
# setup_code strings to try. Order: first that yields a passing harness wins.

_ARANGE_RESHAPE_RX = re.compile(
    r"np\.arange\(\s*(-?\d+)\s*\)\.reshape\(\s*(\d+)\s*,\s*(\d+)\s*\)"
)
_ARANGE_RESHAPE3_RX = re.compile(
    r"np\.arange\(\s*(-?\d+)\s*\)\.reshape\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"
)
_INDICES_RX = re.compile(r"np\.indices\(\s*\(([^()]+)\)\s*\)")
_ARRAY_LITERAL_RX = re.compile(r"np\.array\(\s*\[([^\[\]]+)\]\s*\)")
_RANDINT_RX = re.compile(r"np\.random\.randint\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*\(([^()]+)\)\s*\)")
_RAND_TUPLE_RX = re.compile(r"np\.random\.(random|uniform|normal)\(([^()]*?)\(([^()]+)\)\s*\)")
_RANDN_RX = re.compile(r"np\.random\.randn\(([^()]+)\)")
_RAND_FUNCALL_RX = re.compile(r"np\.random\.rand\(([^()]+)\)")
_ARANGE_RX = re.compile(r"np\.arange\(([^()]+)\)")
_LINSPACE_RX = re.compile(r"np\.linspace\(([^()]+)\)")
_SHAPE_FUNCS = ("zeros", "ones", "full", "empty")
_ZEROSY_RX = re.compile(r"np\.(" + "|".join(_SHAPE_FUNCS) + r")\(\s*\(([^()]+)\)([^)]*)\)")
_EYE_RX = re.compile(r"np\.eye\(\s*(\d+)\s*\)")


def _bump_int(s: str, delta: int = 1, min_val: int = 1) -> str:
    """Bump the first int in `s` by `delta`, keeping >= min_val."""
    m = re.search(r"-?\d+", s)
    if not m:
        return s
    new_val = max(min_val, int(m.group()) + delta)
    return s[: m.start()] + str(new_val) + s[m.end() :]


def _bump_tuple_dims(tuple_inner: str, deltas: tuple[int, ...]) -> str:
    """Bump each comma-separated int in `tuple_inner` by corresponding delta."""
    parts = [p.strip() for p in tuple_inner.split(",")]
    out = []
    for i, part in enumerate(parts):
        delta = deltas[i] if i < len(deltas) else 0
        try:
            v = int(part)
            out.append(str(max(2, v + delta)))
        except ValueError:
            out.append(part)
    return ", ".join(out)


def _mutate_randint(setup: str) -> list[str]:
    out = []
    m = _RANDINT_RX.search(setup)
    if not m:
        return out
    low, high, dims = m.group(1), m.group(2), m.group(3)
    # Strategy 1: bump shape dims by +1
    new_dims = _bump_tuple_dims(dims, (1, 1, 1, 1))
    if new_dims != dims:
        out.append(setup[: m.start()] + f"np.random.randint({low},{high},({new_dims}))" + setup[m.end() :])
    # Strategy 2: bump high (changes value distribution at same shape)
    try:
        new_high = int(high) + 5
        out.append(setup[: m.start()] + f"np.random.randint({low},{new_high},({dims}))" + setup[m.end() :])
    except ValueError:
        pass
    return out


def _mutate_rand_tuple(setup: str) -> list[str]:
    out = []
    m = _RAND_TUPLE_RX.search(setup)
    if not m:
        return out
    func, prefix_args, dims = m.group(1), m.group(2), m.group(3)
    new_dims = _bump_tuple_dims(dims, (1, 1, 1, 1))
    if new_dims != dims:
        out.append(setup[: m.start()] + f"np.random.{func}({prefix_args}({new_dims}))" + setup[m.end() :])
    # Seed change: prepend np.random.seed(7)
    if "np.random.seed" not in setup:
        out.append("np.random.seed(7)\n" + setup)
    return out


def _mutate_randn_rand(setup: str) -> list[str]:
    out = []
    for rx, fname in ((_RANDN_RX, "randn"), (_RAND_FUNCALL_RX, "rand")):
        m = rx.search(setup)
        if not m:
            continue
        dims = m.group(1)
        new_dims = _bump_tuple_dims(dims, (1, 1, 1, 1))
        if new_dims != dims:
            out.append(setup[: m.start()] + f"np.random.{fname}({new_dims})" + setup[m.end() :])
        if "np.random.seed" not in setup:
            out.append("np.random.seed(7)\n" + setup)
    return out


def _mutate_arange_reshape(setup: str) -> list[str]:
    """Compound: np.arange(N).reshape(R,C) -> bump R,C and N=R*C consistently."""
    out = []
    m3 = _ARANGE_RESHAPE3_RX.search(setup)
    if m3:
        a, b, c = int(m3.group(2)), int(m3.group(3)), int(m3.group(4))
        a2, b2, c2 = a + 1, b + 1, c + 1
        n = a2 * b2 * c2
        repl = f"np.arange({n}).reshape({a2},{b2},{c2})"
        out.append(setup[: m3.start()] + repl + setup[m3.end() :])
        return out
    m = _ARANGE_RESHAPE_RX.search(setup)
    if not m:
        return out
    r, c = int(m.group(2)), int(m.group(3))
    r2, c2 = r + 1, c + 1
    repl = f"np.arange({r2 * c2}).reshape({r2},{c2})"
    out.append(setup[: m.start()] + repl + setup[m.end() :])
    return out


def _mutate_indices(setup: str) -> list[str]:
    m = _INDICES_RX.search(setup)
    if not m:
        return []
    dims = m.group(1)
    new_dims = _bump_tuple_dims(dims, (1, 1, 1))
    if new_dims == dims:
        return []
    return [setup[: m.start()] + f"np.indices(({new_dims}))" + setup[m.end() :]]


def _mutate_array_literal(setup: str) -> list[str]:
    """np.array([a, b, c]) -> shift each numeric element by +1 (preserves shape)."""
    m = _ARRAY_LITERAL_RX.search(setup)
    if not m:
        return []
    inner = m.group(1)
    parts = [p.strip() for p in inner.split(",")]
    new_parts = []
    changed = False
    for p in parts:
        try:
            iv = int(p)
            new_parts.append(str(iv + 1))
            changed = True
            continue
        except ValueError:
            pass
        try:
            fv = float(p)
            new_parts.append(repr(fv + 0.5))
            changed = True
            continue
        except ValueError:
            pass
        new_parts.append(p)
    if not changed:
        return []
    new_inner = ", ".join(new_parts)
    return [setup[: m.start()] + f"np.array([{new_inner}])" + setup[m.end() :]]


def _mutate_arange(setup: str) -> list[str]:
    out = []
    if _ARANGE_RESHAPE_RX.search(setup) or _ARANGE_RESHAPE3_RX.search(setup):
        # Handled by _mutate_arange_reshape; don't mutate arange alone here.
        return out
    m = _ARANGE_RX.search(setup)
    if not m:
        return out
    args = m.group(1)
    parts = [p.strip() for p in args.split(",")]
    if len(parts) == 1:
        new_args = _bump_int(args, delta=1)
    elif len(parts) == 2:
        # arange(start, stop) -> bump stop
        try:
            new_stop = int(parts[1]) + 1
            new_args = f"{parts[0]},{new_stop}"
        except ValueError:
            return out
    elif len(parts) == 3:
        try:
            new_stop = int(parts[1]) + int(parts[2])
            new_args = f"{parts[0]},{new_stop},{parts[2]}"
        except ValueError:
            return out
    else:
        return out
    out.append(setup[: m.start()] + f"np.arange({new_args})" + setup[m.end() :])
    return out


def _mutate_linspace(setup: str) -> list[str]:
    m = _LINSPACE_RX.search(setup)
    if not m:
        return []
    args = m.group(1)
    parts = [p.strip() for p in args.split(",")]
    if len(parts) < 3:
        return []
    try:
        new_n = int(parts[2]) + 2
    except ValueError:
        return []
    new_args = f"{parts[0]},{parts[1]},{new_n}"
    return [setup[: m.start()] + f"np.linspace({new_args})" + setup[m.end() :]]


def _mutate_zerosy(setup: str) -> list[str]:
    m = _ZEROSY_RX.search(setup)
    if not m:
        return []
    func, dims, tail = m.group(1), m.group(2), m.group(3)
    new_dims = _bump_tuple_dims(dims, (1, 1, 1, 1))
    if new_dims == dims:
        return []
    return [setup[: m.start()] + f"np.{func}(({new_dims}){tail})" + setup[m.end() :]]


def _mutate_eye(setup: str) -> list[str]:
    m = _EYE_RX.search(setup)
    if not m:
        return []
    n = int(m.group(1))
    return [setup[: m.start()] + f"np.eye({n + 1})" + setup[m.end() :]]


def _mutate_npload(setup: str) -> list[str]:
    """Image-based questions: synthesize a 4D float32 tensor of plausible shape."""
    if "np.load" not in setup:
        return []
    # Replace the load call with a synthetic tensor of similar shape but different
    # values. Original arr shape is (10, 1, 28, 28). Use seed for determinism.
    new_setup = re.sub(
        r"np\.load\([^)]+\)",
        "np.random.RandomState(7).rand(10, 1, 28, 28).astype(np.float32) * 255",
        setup,
    )
    return [new_setup]


def _mutate_seed_only(setup: str) -> list[str]:
    """Universal fallback for any setup that uses randomness."""
    if "np.random" not in setup or "np.random.seed" in setup:
        return []
    return ["np.random.seed(7)\n" + setup]


MUTATORS = (
    _mutate_randint,
    _mutate_rand_tuple,
    _mutate_randn_rand,
    _mutate_arange_reshape,
    _mutate_arange,
    _mutate_linspace,
    _mutate_zerosy,
    _mutate_indices,
    _mutate_eye,
    _mutate_array_literal,
    _mutate_npload,
    _mutate_seed_only,
)


def generate_candidates(setup_code: str) -> list[str]:
    seen = set()
    out = []
    for mut in MUTATORS:
        for cand in mut(setup_code):
            if cand and cand != setup_code and cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out


def validate_alt_tc(code_runner, starter_code: str, primary_tc: dict, alt_tc: dict) -> bool:
    """Return True iff canonical solution + alt setup + expected_expr all line up."""
    solution = synthesize_solution_code(starter_code, [primary_tc])
    results, execution = code_runner.run_function_tests(solution, [alt_tc])
    if not results:
        return False
    return all(r.passed for r in results)


def main() -> None:
    code_runner = load_code_runner()
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    accepted: list[dict] = []
    skipped: list[tuple[int, str]] = []

    for q in questions:
        if q.get("submission_mode") != "function":
            continue
        tcs = q.get("test_cases") or []
        if not tcs:
            skipped.append((q["id"], "no_test_cases"))
            continue
        primary = tcs[0]
        setup = primary.get("setup_code", "") or ""
        if not setup.strip():
            skipped.append((q["id"], "empty_setup"))
            continue

        candidates = generate_candidates(setup)
        if not candidates:
            skipped.append((q["id"], "no_candidate"))
            continue

        starter = q.get("starter_code", "") or ""
        accepted_alt = None
        for alt_setup in candidates:
            alt_tc = {
                "setup_code": alt_setup,
                "call": primary.get("call", "solve()"),
                "expected_expr": primary.get("expected_expr", ""),
            }
            try:
                if validate_alt_tc(code_runner, starter, primary, alt_tc):
                    accepted_alt = alt_tc
                    break
            except Exception:
                continue

        if accepted_alt is None:
            skipped.append((q["id"], "all_candidates_failed"))
            continue

        accepted.append({"id": q["id"], "test_cases_extra": [accepted_alt]})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in accepted),
        encoding="utf-8",
    )

    print(f"Accepted: {len(accepted)} alt test cases")
    print(f"Skipped:  {len(skipped)}")
    from collections import Counter
    skip_reasons = Counter(r for _, r in skipped)
    for reason, count in skip_reasons.most_common():
        print(f"  {count:4d}  {reason}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
