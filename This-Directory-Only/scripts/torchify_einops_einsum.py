#!/usr/bin/env python3
"""Translate the einops + einsum drills from the NumPy dialect to PyTorch.

Design rule, same as the np-1 conversion: *nothing* about an expected value is
authored.  Every ``expected_output`` and every ``expected_expr`` in the emitted
layer is produced by executing the translated answer.  On top of that, each
question is cross-checked by running the ORIGINAL numpy answer and the
TRANSLATED torch answer over the same input values and comparing — that is what
proves the translation preserved meaning, rather than merely producing
something self-consistent.

Kept in the repo because it is the provenance of a committed artifact
(``chatgpt/torch_dialect_overrides_einops_einsum.jsonl``): the layer is
regenerable rather than hand-maintained, and re-running it is how you check a
change to the translation rules did not disturb the other 158 questions.

Usage — snapshot the PRE-conversion bank first, or the script reads its own
output and the shadow-rename pass rewrites the torch alias it introduced:

    git show <pre-conversion-sha>:Local_Deployed_Shared/questions.json \\
        > This-Directory-Only/scripts/questions_base.json
    This-Directory-Only/backend/.venv/bin/python \\
        This-Directory-Only/scripts/torchify_einops_einsum.py

The backend venv is the only interpreter here with torch installed.
Then verify the emitted layer through the real grader:

    .../python This-Directory-Only/scripts/verify_torch_dialect_layer.py
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch as t
import einops  # noqa: F401  (used inside exec'd snippets)

REPO = Path("/home/stellar-thread/Applications/Delta-Drills-Local")
# The pre-conversion bank, snapshotted from git (`git show HEAD:...`).  Reading
# the live questions.json would feed this script its own output on a re-run —
# and the rename pass would then rewrite the torch alias `t` it had introduced.
BANK = Path(__file__).with_name("questions_base.json")  # see docstring: snapshot it first
CROSSCHECKED: dict = {}
MAX_LITERAL_CHARS = 3000
OUT = REPO / "This-Directory-Only/chatgpt/torch_dialect_overrides_einops_einsum.jsonl"
NUMBERS = REPO / "This-Directory-Only/backend/app/data/numbers.npy"

# --------------------------------------------------------------------------
# The sandbox always has numpy imported and np.load patched to serve the ARENA
# image fixture from '/delta_numbers.npy'.  Reproduce both here so setup code
# executes identically to the grader.
_real_load = np.load


def _patched_load(file, *a, **kw):
    if str(file) == "/delta_numbers.npy":
        file = str(NUMBERS)
    return _real_load(file, *a, **kw)


np.load = _patched_load


# --------------------------------------------------------------------------
# Translation rules.  Ordered; applied as plain regex substitutions.
#
# Only imports, the answer body, the starter body and each case's setup_code
# need translating: `call` is regenerated from the argument list and
# `expected_expr` is always replaced by an executed literal, so no numpy
# expression survives into the output by accident.

_RULES: list[tuple[str, str]] = [
    # imports
    (r"^import numpy as np$", "import torch as t"),
    (r"^import numpy as np\n", "import torch as t\n"),
    # constructors that keep their spelling
    (r"\bnp\.einsum\(", "t.einsum("),
    (r"\bnp\.array\(", "t.tensor("),
    (r"\bnp\.arange\(", "t.arange("),
    (r"\bnp\.ones\(", "t.ones("),
    (r"\bnp\.zeros\(", "t.zeros("),
    (r"\bnp\.full\(", "t.full("),
    (r"\bnp\.eye\(", "t.eye("),
    (r"\bnp\.diag\(", "t.diag("),
    (r"\bnp\.stack\(", "t.stack("),
    (r"\bnp\.linspace\(", "t.linspace("),
    # renamed
    (r"\bnp\.concatenate\(", "t.cat("),
    (r"\bnp\.moveaxis\(", "t.movedim("),
    (r"\bnp\.swapaxes\(", "t.transpose("),
    (r"\bnp\.expand_dims\(", "t.unsqueeze("),
    # np.repeat is torch's repeat_interleave (torch's own .repeat is np.tile)
    (r"\bnp\.repeat\(([^,]+),\s*([^,]+),\s*axis=", r"t.repeat_interleave(\1, \2, dim="),
    # t.diag needs a tensor where np.diag accepted a bare list
    (r"t\.diag\(\[([^\]]*)\]\)", r"t.diag(t.tensor([\1]))"),
    # torch rejects a negative slice step outright — .flip is the replacement,
    # and it already copies, so a trailing .copy() is redundant.
    (r"\[::-1\]\.copy\(\)", ".flip(0)"),
    (r"\[::-1\]", ".flip(0)"),
    # assertion helpers (these appear in the lesson fences, not in drill code)
    (r"\bnp\.array_equal\(", "t.equal("),
    (r"\bnp\.allclose\(", "t.allclose("),
    (r"\bnp\.outer\(", "t.outer("),
    (r"\bnp\.trace\(", "t.trace("),
    (r"\bnp\.dot\(", "t.dot("),
    # torch.cov treats ROWS as variables, so numpy's rowvar=False is a transpose
    (r"\bnp\.cov\(([^,]+), rowvar=False\)", r"t.cov(\1.T)"),
    # dtypes
    (r"\bnp\.uint8\b", "t.uint8"),
    (r"\bnp\.float32\b", "t.float32"),
    (r"\bnp\.float64\b", "t.float64"),
    (r"\bnp\.int64\b", "t.int64"),
    (r"\bdtype=float\b", "dtype=t.float32"),
    # method-name differences
    (r"\.astype\(", ".to("),
    (r"\.mean\(axis=", ".mean(dim="),
    (r"\.sum\(axis=", ".sum(dim="),
    (r"\.max\(axis=", ".amax(dim="),
    (r"\.min\(axis=", ".amin(dim="),
    # np.load stays: numpy is always present in the sandbox and it is the only
    # way to reach the ARENA fixture.  Wrap it so setup hands solve() a tensor.
    (r"np\.load\('/delta_numbers\.npy'\)", "t.tensor(np.load('/delta_numbers.npy'))"),
]

# `x.size` (element count) has no torch equivalent spelled that way.
_SIZE_RE = re.compile(r"\b(\w+)\.size\b(?!\()")


def translate(code: str) -> str:
    if not code:
        return code
    out = code
    for pat, rep in _RULES:
        out = re.sub(pat, rep, out, flags=re.M)
    out = _SIZE_RE.sub(r"\1.numel()", out)
    # t.tensor(np.load(...)) already produces a tensor; a following .to(t.float32)
    # is fine, but the doubled wrap below can appear when the rule order fires
    # twice on nested loads.
    out = out.replace("t.tensor(t.tensor(", "t.tensor((")
    return out


# --------------------------------------------------------------------------
# Deterministic stand-ins for the random fixtures.  Distinct, strictly
# increasing, dyadic values: distinct so a wrong permutation cannot pass by
# coincidence, dyadic so float32 arithmetic stays exact and the frozen literal
# is stable.  No RNG at all, so nothing here depends on torch's generator
# staying byte-identical across versions.
_RAND_RE = re.compile(r"np\.random\.rand\(([^)]*)\)")
_RANDN_RE = re.compile(r"np\.random\.randn\(([^)]*)\)")
_RANDINT_RE = re.compile(r"np\.random\.randint\(([^,]+),\s*([^,]+),\s*\(([^)]*)\)\)")
_SEED_RE = re.compile(r"^\s*np\.random\.seed\([^)]*\)\n?", re.M)


def _dims(s: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", s)]


def _float_fixture(shape: list[int]) -> str:
    n = 1
    for d in shape:
        n *= d
    body = f"t.arange({n}, dtype=t.float32) / 4"
    if len(shape) > 1 or True:
        body = f"({body}).reshape({', '.join(str(d) for d in shape)})"
    return body


def _int_fixture(shape: list[int], lo: int, hi: int) -> str:
    n = 1
    for d in shape:
        n *= d
    span = max(hi - lo, 1)
    body = f"(t.arange({n}) % {span} + {lo})"
    return f"{body}.reshape({', '.join(str(d) for d in shape)})"


def derandomize(code: str) -> str:
    code = _SEED_RE.sub("", code)
    code = _RAND_RE.sub(lambda m: _float_fixture(_dims(m.group(1))), code)
    code = _RANDN_RE.sub(lambda m: _float_fixture(_dims(m.group(1))), code)
    code = _RANDINT_RE.sub(
        lambda m: _int_fixture(_dims(m.group(3)), int(m.group(1)), int(m.group(2))), code
    )
    return code


# --------------------------------------------------------------------------
def call_args(call: str) -> str | None:
    """Pull the argument list out of an existing `call` expression."""
    m = re.search(r"solve\((.*?)\)(?=[,.\)]|$)", call)
    if not m:
        return None
    return m.group(1)


def run(code: str, env: dict) -> dict:
    # The grader's preamble always has numpy (and einops) in scope, and patches
    # np.load — mirror that so snippets execute exactly as they will in the box.
    env.setdefault("np", np)
    env.setdefault("t", t)
    env.setdefault("einops", einops)
    exec(compile(code, "<snippet>", "exec"), env)
    return env


# A parameter literally named `t` shadows `import torch as t` and makes the
# whole drill unrunnable in this dialect.  Rename it — in the code, in the
# fixtures and in the prompt — rather than emitting a broken question.
SHADOW_RENAMES = {265: ("t", "a"), 271: ("t", "x"), 285: ("t", "x"), 338: ("t", "steps")}


def unshadow(text: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, text)


def to_numpy(v):
    if isinstance(v, t.Tensor):
        return v.detach().cpu().numpy()
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], t.Tensor):
        return [to_numpy(x) for x in v]
    return np.asarray(v)


def close(a, b) -> bool:
    # Mirror the grader: tuples/lists compare elementwise, so a (shape, digest)
    # pair is checked piece by piece rather than forced into one array.
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) \
            and not (a and isinstance(a[0], t.Tensor)):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    try:
        a, b = to_numpy(a), to_numpy(b)
        if isinstance(a, list) or isinstance(b, list):
            return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
        if a.shape != b.shape:
            return False
        if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
            return bool(np.allclose(a.astype(np.float64), b.astype(np.float64),
                                    rtol=1e-5, atol=1e-6, equal_nan=True))
        return bool(np.array_equal(a, b))
    except Exception:
        return False


def literal(v) -> str:
    """Freeze a computed value as a stable Python literal."""
    if isinstance(v, t.Tensor):
        return repr(v.tolist())
    if isinstance(v, tuple):
        return "(" + ", ".join(literal(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(literal(x) for x in v) + "]"
    if isinstance(v, (np.generic,)):
        return repr(v.item())
    return repr(v)


TEXT_FIXES = [
    (r"\bnumpy array\b", "PyTorch tensor"),
    (r"\bnumpy arrays\b", "PyTorch tensors"),
    (r"\bnumpy\b", "PyTorch"),
    (r"\bNumPy\b", "PyTorch"),
    (r"\barrays\b", "tensors"),
    (r"\barray\b", "tensor"),
    (r"\bndarray\b", "tensor"),
]


def fix_text(s: str) -> str:
    for pat, rep in TEXT_FIXES:
        s = re.sub(pat, rep, s)
    return s


# --------------------------------------------------------------------------
def main() -> int:
    bank = {q["id"]: q for q in json.loads(BANK.read_text())}
    targets = [q for q in bank.values()
               if (q.get("subtopic_key") or "").startswith(("Einsum", "Einops"))]
    targets.sort(key=lambda q: q["id"])

    records, failures = [], []
    for q in targets:
        qid = q["id"]
        try:
            rec = convert(q)
        except Exception as exc:  # noqa: BLE001 — report, do not emit
            failures.append((qid, f"{type(exc).__name__}: {exc}"))
            continue
        records.append(rec)

    OUT.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"emitted {len(records)} / {len(targets)} -> {OUT.name}")
    tot = sum(c for c, _ in CROSSCHECKED.values())
    allc = sum(n for _, n in CROSSCHECKED.values())
    print(f"cross-checked numpy-vs-torch on {tot} / {allc} test cases")
    weak = sorted(q for q, (c, n) in CROSSCHECKED.items() if c < n)
    print(f"questions with any unchecked case: {len(weak)} -> {weak}")
    if failures:
        print(f"\n{len(failures)} FAILED:")
        for qid, msg in failures:
            print(f"  #{qid}: {msg[:160]}")
    return 1 if failures else 0


def convert(q: dict) -> dict:
    qid = q["id"]
    answer_np = q["answer_code"]
    # Rename before translating: afterwards `t` is the torch alias and a blind
    # word-boundary rename would rewrite `t.einsum` too.
    ren = SHADOW_RENAMES.get(qid)
    src_answer = unshadow(answer_np, *ren) if ren else answer_np
    src_starter = unshadow(q["starter_code"], *ren) if ren else q["starter_code"]
    answer = translate(src_answer)
    starter = translate(src_starter)

    if "np." in re.sub(r"np\.load\('/delta_numbers\.npy'\)", "", answer):
        raise ValueError(f"untranslated numpy left in answer: {answer!r}")

    # stdout of the translated answer -> expected_output
    buf = io.StringIO()
    env: dict = {}
    with contextlib.redirect_stdout(buf):
        run(answer, env)
    expected_output = buf.getvalue().rstrip("\n")
    solve_t = env["solve"]

    env_np: dict = {}
    run(answer_np, env_np)
    solve_np = env_np["solve"]

    cases = []
    for case in q.get("test_cases") or []:
        args = call_args(case["call"])
        if args is None:
            raise ValueError(f"cannot parse call: {case['call']!r}")
        if ren:
            args = unshadow(args, *ren)
        raw_setup = unshadow(case["setup_code"], *ren) if ren else case["setup_code"]
        setup = derandomize(translate(raw_setup))
        if setup.strip().startswith("import torch as t"):
            pass
        elif "import torch as t" not in setup:
            setup = "import torch as t\n" + setup

        senv: dict = {"t": t, "np": np, "einops": einops}
        run(setup, senv)
        senv["solve"] = solve_t
        value = eval(args and f"solve({args})" or "solve()", senv)

        call = f"solve({args})"
        lit = literal(value)

        if isinstance(value, t.Tensor):
            if len(lit) > MAX_LITERAL_CHARS:
                # A few drills run on the 150x150 ARENA image fixture, where
                # freezing the whole result would add megabytes to every
                # learner's download.  Compare shape plus a position-weighted
                # integer checksum: any value that moves changes the weight it
                # is multiplied by, so a wrong permutation is still caught, and
                # an int is compared exactly rather than through the grader's
                # float tolerance.  `.shape`/`.numel()` also fail loudly on a
                # non-tensor, so tensor-ness is asserted here too.
                call = (f"(tuple({call}.shape), "
                        f"int(t.round({call}.flatten().to(t.float64) * 16) "
                        f"@ t.arange({call}.numel(), dtype=t.float64)))")
            else:
                # Assert tensor-ness, not just the numbers.  Learners arriving
                # from the numpy dialect reach for an ndarray or a plain list,
                # and the grader compares those equal to a tensor — without this
                # an answer that never builds a tensor passes.  (#361 hands
                # solve() a LIST of images and asks for one stacked tensor;
                # `return imgs` graded as correct before this check existed.)
                call = f"(isinstance({call}, t.Tensor), {call})"
            senv["solve"] = solve_t
            value = eval(call, senv)
            lit = literal(value)

        # the frozen literal must round-trip to the same value it was made from
        if not close(eval(lit, {"t": t, "np": np}), value):
            raise ValueError(f"frozen literal unstable for call {call}")

        cases.append({
            "setup_code": setup,
            "call": call,
            "expected_expr": lit,
        })

    verify_equivalence(q, solve_np, solve_t)

    return {
        "id": qid,
        "question_text": fix_text(
            unshadow(q.get("question_text") or "", *ren) if ren
            else (q.get("question_text") or "")
        ),
        "function_name": q.get("function_name", "solve"),
        "submission_mode": "function",
        "starter_code": starter,
        "answer_code": answer,
        "test_cases": cases,
        "expected_output": expected_output,
    }


def verify_equivalence(q: dict, solve_np, solve_t) -> None:
    """Run both dialects over identical inputs and require agreement.

    The torch side is fed tensors built FROM the numpy fixtures, so the two
    functions genuinely see the same numbers — this is independent of whatever
    the emitted setup_code happens to construct.
    """
    checked = 0
    for case in q.get("test_cases") or []:
        args = call_args(case["call"])
        if args is None:
            continue
        env: dict = {"np": np}
        try:
            run(case["setup_code"], env)
        except Exception:
            continue
        names = [a.strip() for a in args.split(",")] if args.strip() else []
        if not all(re.fullmatch(r"\w+", n) and n in env for n in names):
            continue
        np_vals = [env[n] for n in names]
        try:
            got_np = solve_np(*np_vals)
        except Exception:
            continue
        t_vals = [_as_tensor(v) for v in np_vals]
        got_t = solve_t(*t_vals)
        if not close(got_np, got_t):
            raise ValueError(
                f"dialects disagree on args ({args}): numpy -> "
                f"{np.asarray(got_np).ravel()[:6]}, torch -> "
                f"{to_numpy(got_t).ravel()[:6]}"
            )
        checked += 1
    if checked == 0:
        raise ValueError("no case could be cross-checked against numpy")
    CROSSCHECKED[q["id"]] = (checked, len(q.get("test_cases") or []))


def _as_tensor(v):
    if isinstance(v, list):
        return [_as_tensor(x) for x in v]
    arr = np.ascontiguousarray(v)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    return t.from_numpy(arr)


if __name__ == "__main__":
    sys.exit(main())
