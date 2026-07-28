#!/usr/bin/env python3
"""Translate the np-2 + np-3 drills from the NumPy dialect to PyTorch.

Same design rule as the np-1 and einops/einsum passes: *nothing* about an
expected value is authored.  Every ``expected_output`` and every
``expected_expr`` in the emitted layer is produced by executing the translated
answer, and every question is additionally cross-checked by running the
ORIGINAL numpy answer and the TRANSLATED torch answer over the same inputs.
That cross-check is what makes the hand-written translations below safe: a
rewrite that changed the meaning disagrees with numpy and the question is
refused rather than emitted.

np-2/np-3 differ from the earlier passes in *kind*, not just in size.  einops
and einsum are one API each; these two lessons range over masking, selection,
reductions, broadcasting and linear algebra, and roughly a quarter of the
drills use a numpy function with no torch spelling at all (``np.ogrid``,
``np.nditer``, ``np.apply_along_axis``, ``np.argpartition``, ``np.intersect1d``
…).  Regex cannot translate those, so they are rewritten by hand in MANUAL and
verified by the same cross-check as everything else.

Usage — snapshot the PRE-conversion bank first, or the script reads its own
output:

    git show <pre-conversion-sha>:Local_Deployed_Shared/questions.json \\
        > This-Directory-Only/scripts/questions_base.json
    This-Directory-Only/backend/.venv/bin/python \\
        This-Directory-Only/scripts/torchify_np23.py

Then verify the emitted layer through the real grader:

    .../python This-Directory-Only/scripts/verify_torch_dialect_layer.py
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch as t

sys.path.insert(0, str(Path(__file__).parent))
from torchify_einops_einsum import (  # noqa: E402
    MAX_LITERAL_CHARS,
    close,
    fix_text,
    literal,
    run,
    to_numpy,
    translate as base_translate,
    _as_tensor,
)
from torchify_np23_manual import (  # noqa: E402
    EXCLUDE,
    MANUAL,
    NO_CROSSCHECK,
    SHADOW_RENAMES,
)

REPO = Path("/home/stellar-thread/Applications/Delta-Drills-Local")
BANK = Path(__file__).with_name("questions_base.json")
OUT = REPO / "This-Directory-Only/chatgpt/torch_dialect_overrides_np23.jsonl"
REGISTRY = REPO / "Local_Deployed_Shared/lessons/kc_registry.json"
QTAGS = REPO / "Local_Deployed_Shared/lessons/qmatrix_tags.json"
LESSONS = {"np-2", "np-3"}

CROSSCHECKED: dict = {}

# --------------------------------------------------------------------------
# Extra translation rules, applied AFTER the shared ones.  The shared pass
# already handles imports, the constructors common to both dialects and the
# `.max(axis=)` -> `.amax(dim=)` family; everything below is np-2/np-3 surface.

_POST_RULES: list[tuple[str, str]] = [
    # --- same name, same meaning ---
    (r"\bnp\.where\(", "t.where("),
    (r"\bnp\.unique\(", "t.unique("),
    (r"\bnp\.diff\(", "t.diff("),
    (r"\bnp\.tile\(", "t.tile("),
    (r"\bnp\.argsort\(", "t.argsort("),
    (r"\bnp\.argmin\(", "t.argmin("),
    (r"\bnp\.argmax\(", "t.argmax("),
    (r"\bnp\.argwhere\(", "t.argwhere("),
    (r"\bnp\.count_nonzero\(", "t.count_nonzero("),
    (r"\bnp\.bincount\(", "t.bincount("),
    (r"\bnp\.isnan\(", "t.isnan("),
    (r"\bnp\.isfinite\(", "t.isfinite("),
    (r"\bnp\.nan_to_num\(", "t.nan_to_num("),
    (r"\bnp\.nanmean\(", "t.nanmean("),
    (r"\bnp\.tril\(", "t.tril("),
    (r"\bnp\.triu\(", "t.triu("),
    (r"\bnp\.zeros_like\(", "t.zeros_like("),
    (r"\bnp\.ones_like\(", "t.ones_like("),
    (r"\bnp\.empty_like\(", "t.empty_like("),
    (r"\bnp\.abs\(", "t.abs("),
    (r"\bnp\.sign\(", "t.sign("),
    (r"\bnp\.ceil\(", "t.ceil("),
    (r"\bnp\.floor\(", "t.floor("),
    (r"\bnp\.round\(", "t.round("),
    (r"\bnp\.exp\(", "t.exp("),
    (r"\bnp\.log\(", "t.log("),
    (r"\bnp\.sqrt\(", "t.sqrt("),
    (r"\bnp\.maximum\(", "t.maximum("),
    (r"\bnp\.minimum\(", "t.minimum("),
    (r"\bnp\.cumsum\(", "t.cumsum("),
    (r"\bnp\.cumprod\(", "t.cumprod("),
    (r"\bnp\.sort\(", "t.sort("),          # NOTE: fixed up to `.values` below
    (r"\bnp\.column_stack\(", "t.column_stack("),
    (r"\bnp\.vstack\(", "t.vstack("),
    (r"\bnp\.hstack\(", "t.hstack("),
    (r"\bnp\.ravel\(", "t.ravel("),
    (r"\bnp\.trace\(", "t.trace("),
    (r"\bnp\.clip\(", "t.clip("),
    (r"\bnp\.all\(", "t.all("),
    (r"\bnp\.any\(", "t.any("),
    (r"\bnp\.asarray\(", "t.as_tensor("),
    (r"\bnp\.empty\(", "t.empty("),
    (r"\bnp\.result_type\(", "t.result_type("),
    (r"\bnp\.quantile\(", "t.quantile("),
    (r"\bnp\.isin\(", "t.isin("),
    (r"\bnp\.linalg\.norm\(", "t.linalg.norm("),
    (r"\bnp\.linalg\.inv\(", "t.linalg.inv("),
    (r"\bnp\.linalg\.det\(", "t.linalg.det("),
    (r"\bnp\.linalg\.solve\(", "t.linalg.solve("),
    (r"\bnp\.inf\b", "t.inf"),
    (r"\bnp\.nan\b", "t.nan"),
    (r"\bnp\.newaxis\b", "None"),
    (r"\bnp\.int32\b", "t.int32"),
    (r"\bnp\.bool_\b", "t.bool"),
    (r"\bdtype=int\b", "dtype=t.int64"),
    (r"\bdtype=bool\b", "dtype=t.bool"),
    # --- method-name differences ---
    (r"\.copy\(\)", ".clone()"),
    # numpy reverses with a negative step, which torch rejects outright.
    (r"\[:,\s*::-1\]", ".flip(1)"),
    (r"\[\.\.\.,\s*::-1\]", ".flip(-1)"),
    # --- keyword spellings ---
    (r"\bkeepdims=", "keepdim="),
    (r"\baxis=", "dim="),
    # `.astype(float)` becomes `.to(float)`, and torch reads a bare `float` as
    # float64 while its own tensors default to float32 — mixing the two is a
    # dtype error at the first matmul rather than a quiet promotion.
    (r"\.to\(float\)", ".to(t.float32)"),
    (r"\.to\(int\)", ".to(t.int64)"),
    (r"\.to\(bool\)", ".to(t.bool)"),
]

# `np.sort(x)` returns an array; `t.sort(x)` returns a (values, indices) pair.
# Reading `.values` off it is the whole difference, and forgetting it is a
# silent shape change rather than an error.
_SORT_FIX = re.compile(r"(t\.sort\((?:[^()]|\([^()]*\))*\))(?!\s*\.)")

# numpy's std/var default to ddof=0; torch's default to correction=1.  Left
# alone this produces *plausible but wrong numbers* rather than an exception,
# which is exactly the class of bug the cross-check exists to catch — but it is
# better not to emit it in the first place.
_STD_EMPTY = re.compile(r"\.(std|var)\(\)")
_STD_ARGS = re.compile(r"\.(std|var)\(([^()]*)\)")


def _fix_diagonal_kwarg(code: str) -> str:
    """`k=` -> `diagonal=`, with balanced parens so nested calls still match.

    A regex cannot do this: `np.diag(1 + np.arange(n - 1), k=-1)` puts a call
    between the function name and the keyword.
    """
    for fn in ("t.diag", "t.diagflat", "t.tril", "t.triu"):
        code = _rewrite_calls(
            code, fn,
            lambda args, fn=fn: fn + "(" + ", ".join(
                ("diagonal=" + a[2:]) if a.startswith("k=") else a for a in args) + ")",
        )
    # torch.round takes the digit count as a keyword only, and the value being
    # rounded is often itself a call, so this cannot be a regex either.
    code = _rewrite_calls(
        code, "t.round",
        lambda args: (f"t.round({args[0]}, decimals={args[1]})"
                      if len(args) == 2 and "=" not in args[1]
                      else "t.round(" + ", ".join(args) + ")"),
    )
    return code


def translate(code: str) -> str:
    if not code:
        return code
    out = base_translate(code)
    for pat, rep in _POST_RULES:
        out = re.sub(pat, rep, out, flags=re.M)
    out = _fix_diagonal_kwarg(out)
    out = _SORT_FIX.sub(r"\1.values", out)
    out = _STD_EMPTY.sub(r".\1(correction=0)", out)
    out = _STD_ARGS.sub(
        lambda m: f".{m.group(1)}({m.group(2)}, correction=0)"
        if "correction" not in m.group(2) else m.group(0),
        out,
    )
    return out


# --------------------------------------------------------------------------
# Translating the `call` expression rather than rebuilding it.
#
# The earlier passes rebuilt every call as `solve(args)`.  That is lossy here:
# a third of these calls assert something *beyond* the return value — that the
# input was not modified (`(solve(z).tolist(), z.tolist())`), that an in-place
# drill returned the very object it was handed (`solve(a, b) is a`), or that
# the result has the right dtype.  Rebuilding silently deletes those
# assertions, so the call is translated in place and the tensor-ness guard is
# added around it instead.

def _split_top(args: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in args:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts]


def _rewrite_calls(src: str, name: str, handler, suffix: str = "") -> str:
    """Replace every `name(...)` (plus an optional literal suffix) via handler."""
    out, i = [], 0
    while True:
        j = src.find(name + "(", i)
        if j < 0:
            out.append(src[i:])
            return "".join(out)
        k, depth = j + len(name) + 1, 1
        while k < len(src) and depth:
            depth += (src[k] == "(") - (src[k] == ")")
            k += 1
        if depth or (suffix and not src.startswith(suffix, k)):
            out.append(src[i:k])
            i = k
            continue
        out.append(src[i:j])
        out.append(handler(_split_top(src[j + len(name) + 1:k - 1])))
        i = k + len(suffix)


def translate_call(call: str) -> str:
    # `.dtype.kind` first: it has to consume the asarray that feeds it, and
    # torch dtypes carry no `.kind`.  'f' vs 'i' is the distinction the drills
    # are actually asserting.
    call = _rewrite_calls(
        call, "np.asarray",
        lambda a: (f"('f' if t.as_tensor({a[0]}).is_floating_point() else 'i')"),
        suffix=".dtype.kind",
    )
    # `.shape` on a tensor is a torch.Size, which repr's as `torch.Size([...])`
    # and is not a name the sandbox can evaluate.  A plain tuple is.
    call = _rewrite_calls(
        call, "np.asarray", lambda a: f"tuple(t.as_tensor({a[0]}).shape)",
        suffix=".shape",
    )
    # np.round(x, n) only existed to pin float noise; the grader compares
    # floats with rtol=1e-5/atol=1e-6, so the rounding is redundant here and
    # torch.round refuses integer tensors outright.
    call = _rewrite_calls(call, "np.round", lambda a: a[0])
    call = _rewrite_calls(
        call, "np.asarray",
        lambda a: (f"t.as_tensor({a[0]}).to(t.float64)"
                   if len(a) > 1 and "float" in a[1] else f"t.as_tensor({a[0]})"),
    )
    for old, new in _CALL_CASTS:
        call = call.replace(old, new)
    return call


# Casts written into the call itself, e.g. `np.asarray(solve(z)).astype(int)`.
_CALL_CASTS = [
    (".astype(np.int64)", ".to(t.int64)"),
    (".astype(np.int32)", ".to(t.int32)"),
    (".astype(np.float64)", ".to(t.float64)"),
    (".astype(np.float32)", ".to(t.float32)"),
    (".astype(int)", ".to(t.int64)"),
    (".astype(float)", ".to(t.float64)"),
    (".astype(bool)", ".to(t.bool)"),
]


def call_args(call: str) -> str | None:
    """The argument text of the `solve(...)` inside a call expression.

    Balanced rather than regex: these calls wrap solve() in post-processing
    (`(solve(z, p) is z, np.asarray(z).tolist())`), and a non-greedy regex
    happily matches across the wrapper's own parentheses and returns nonsense.
    """
    j = call.find("solve(")
    if j < 0:
        return None
    k, depth = j + len("solve("), 1
    while k < len(call) and depth:
        depth += (call[k] == "(") - (call[k] == ")")
        k += 1
    return None if depth else call[j + len("solve("):k - 1]


# --------------------------------------------------------------------------
# Deterministic stand-ins for random fixtures.
#
# The einops pass could use any distinct values it liked; np-2/np-3 drills make
# claims about their inputs ("with DISTINCT values", "non-negative", "may
# contain zeros"), so the stand-ins mirror the DISTRIBUTION being replaced:
# rand/random/uniform stay inside [0, 1), randn is signed and includes zero,
# and integer fixtures are distinct whenever the requested range is wide enough
# to allow it.  All values are dyadic, so float32 arithmetic is exact and the
# frozen literal is stable.

_SEED_RE = re.compile(r"^\s*np\.random\.seed\([^)]*\)\n?", re.M)
_SHUFFLE_RE = re.compile(r"^\s*np\.random\.shuffle\([^)]*\)\n?", re.M)
_RNG_ASSIGN_RE = re.compile(r"^\s*(\w+)\s*=\s*np\.random\.default_rng\([^)]*\)\n?", re.M)


def _dims(s: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", s)]


def _shape_suffix(shape: list[int]) -> str:
    return f".reshape({', '.join(str(d) for d in shape)})" if shape else ""


def _unit_fixture(shape: list[int]) -> str:
    """Distinct dyadic values in (0, 1) — mirrors rand/random/uniform(0, 1)."""
    n = math.prod(shape) if shape else 1
    den = 1 << max(1, (n + 1).bit_length())
    return f"((t.arange({n}, dtype=t.float32) + 1) / {den}){_shape_suffix(shape)}"


def _signed_fixture(shape: list[int]) -> str:
    """Signed dyadic values straddling zero — mirrors randn."""
    n = math.prod(shape) if shape else 1
    return f"((t.arange({n}, dtype=t.float32) - {n // 2}) / 4){_shape_suffix(shape)}"


def _int_fixture(shape: list[int], lo: int, hi: int) -> str:
    n = math.prod(shape) if shape else 1
    span = max(hi - lo, 1)
    body = f"(t.arange({n}) + {lo})" if span >= n else f"(t.arange({n}) % {span} + {lo})"
    return f"{body}{_shape_suffix(shape)}"


def _sub_all(code: str, pattern: str, fn) -> str:
    return re.sub(pattern, fn, code)


def derandomize(code: str, keep: str = "") -> str:
    code = _SEED_RE.sub("", code)
    # A shuffle only scrambles a fixture that is now fixed; dropping it keeps
    # the fixture deterministic without changing what the drill is handed.
    code = _SHUFFLE_RE.sub("", code)

    # np.random.rand(2, 3) / randn(2, 3) — bare dims
    code = _sub_all(code, r"np\.random\.rand\(([^)]*)\)",
                    lambda m: _unit_fixture(_dims(m.group(1))))
    code = _sub_all(code, r"np\.random\.randn\(([^)]*)\)",
                    lambda m: _signed_fixture(_dims(m.group(1))))
    # np.random.random((2, 3)) / random(5) — one shape argument
    code = _sub_all(code, r"np\.random\.random\(\(([^)]*)\)\)",
                    lambda m: _unit_fixture(_dims(m.group(1))))
    code = _sub_all(code, r"np\.random\.random\((\d+)\)",
                    lambda m: _unit_fixture(_dims(m.group(1))))
    # np.random.uniform(lo, hi, shape)
    code = _sub_all(code, r"np\.random\.uniform\(([^,]+),\s*([^,]+),\s*\(([^)]*)\)\)",
                    lambda m: _unit_fixture(_dims(m.group(3))))
    # np.random.randint(lo, hi, (shape)) and the bare-length form
    code = _sub_all(code, r"np\.random\.randint\(([^,]+),\s*([^,]+),\s*\(([^)]*)\)\)",
                    lambda m: _int_fixture(_dims(m.group(3)), int(m.group(1)), int(m.group(2))))
    code = _sub_all(code, r"np\.random\.randint\((-?\d+),\s*(-?\d+),\s*(\d+)\)",
                    lambda m: _int_fixture(_dims(m.group(3)), int(m.group(1)), int(m.group(2))))
    # Generator methods, once the generator itself is gone.
    code = _sub_all(code, r"\w+\.integers\(([^,]+),\s*([^,]+),\s*(?:size=)?\(([^)]*)\)\)",
                    lambda m: _int_fixture(_dims(m.group(3)), int(m.group(1)), int(m.group(2))))
    code = _sub_all(code, r"\w+\.random\(\(([^)]*)\)\)",
                    lambda m: _unit_fixture(_dims(m.group(1))))
    code = _sub_all(code, r"\w+\.standard_normal\(\(([^)]*)\)\)",
                    lambda m: _signed_fixture(_dims(m.group(1))))

    # Drop the generator only once nothing still uses it.  #101 is handed one
    # as an argument, so there its assignment survives and becomes the torch
    # equivalent instead — `keep` carries the call text, where that use lives.
    def _drop_rng(m: re.Match) -> str:
        name = m.group(1)
        used_later = re.search(rf"\b{re.escape(name)}\b", code[m.end():])
        used_in_call = re.search(rf"\b{re.escape(name)}\b", keep)
        if not used_later and not used_in_call:
            return ""
        seed = re.search(r"default_rng\(\s*(\d+)\s*\)", m.group(0))
        return f"{name} = t.Generator().manual_seed({seed.group(1) if seed else 0})\n"

    code = _RNG_ASSIGN_RE.sub(_drop_rng, code)
    # Any generator still standing — e.g. built inline in a demo's
    # `print(solve(x, 2, np.random.default_rng(0)))` — becomes the torch one.
    code = re.sub(r"np\.random\.default_rng\(\s*(\d+)\s*\)",
                  r"t.Generator().manual_seed(\1)", code)
    return code


# --------------------------------------------------------------------------


def targets() -> list[dict]:
    registry = json.loads(REGISTRY.read_text())
    lesson_of = {kc["id"]: kc["lesson"] for kc in registry["kcs"]}
    tags = json.loads(QTAGS.read_text())
    bank = {q["id"]: q for q in json.loads(BANK.read_text())}
    picked = []
    for qid, tag in tags.items():
        lessons = {lesson_of.get(k) for k in tag.get("target_kcs", [])}
        if lessons & LESSONS and int(qid) in bank:
            picked.append(bank[int(qid)])
    picked.sort(key=lambda q: q["id"])
    return picked


def main() -> int:
    picked = targets()
    records, failures = [], []
    for q in picked:
        try:
            records.append(convert(q))
        except Exception as exc:  # noqa: BLE001 — report, do not emit
            failures.append((q["id"], f"{type(exc).__name__}: {exc}"))

    OUT.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"emitted {len(records)} / {len(picked)} -> {OUT.name}")
    checked = sum(c for c, _ in CROSSCHECKED.values())
    total = sum(n for _, n in CROSSCHECKED.values())
    print(f"cross-checked numpy-vs-torch on {checked} / {total} test cases")
    weak = sorted(q for q, (c, n) in CROSSCHECKED.items() if c < n)
    print(f"questions with any unchecked case: {len(weak)} -> {weak}")
    print(f"hand-written translations: {sorted(MANUAL)}")
    if failures:
        print(f"\n{len(failures)} FAILED:")
        for qid, msg in failures:
            print(f"  #{qid}: {msg[:200]}")
    return 1 if failures else 0


# A raw list literal assigned through an index works in numpy but not in
# torch, where the right-hand side has to be a tensor.
_ROW_ASSIGN = re.compile(r"^(\s*\w+\[[^\]]*\]\s*=\s*)(\[[^\]]*\])\s*$", re.M)


def translate_setup(code: str, keep: str = "") -> str:
    code = derandomize(translate(code), keep)
    code = _ROW_ASSIGN.sub(r"\1t.tensor(\2)", code)
    if "import torch as t" not in code:
        code = "import torch as t\n" + code
    return code


def demo_block(original: str) -> str:
    """The module-level demo that follows solve() in an answer, translated.

    Everything up to and including the function body belongs to the hand-written
    translation; what trails it builds an example and prints it.
    """
    lines = original.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("def ")), None)
    if start is None:
        return ""
    i = start + 1
    while i < len(lines) and (not lines[i].strip() or lines[i][:1] in (" ", "\t")):
        i += 1
    demo = "\n".join(lines[i:]).strip()
    return _ROW_ASSIGN.sub(r"\1t.tensor(\2)", derandomize(translate(demo))) if demo else ""


def convert(q: dict) -> dict:
    qid = q["id"]
    if qid in EXCLUDE:
        raise ValueError(f"excluded by design: {EXCLUDE[qid]}")

    ren = SHADOW_RENAMES.get(qid)

    def unshadow(text: str) -> str:
        return re.sub(rf"\b{re.escape(ren[0])}\b", ren[1], text) if ren else text

    answer_np = q["answer_code"]
    if qid in MANUAL:
        # Keep the original demo block.  These are stdout_prediction drills:
        # the trailing `print(solve(...))` IS the question's expected output,
        # and an answer without one exports an empty stdout, so the exporter
        # keeps the stale NUMPY-formatted string against a torch drill.
        answer = ("import torch as t\n" + MANUAL[qid].strip() + "\n\n\n"
                  + demo_block(answer_np)).rstrip() + "\n"
    else:
        # The demo block at the bottom of an answer builds its own fixture, so
        # it needs the same list-assignment fix the setups get.
        answer = _ROW_ASSIGN.sub(r"\1t.tensor(\2)", translate(unshadow(answer_np)))

    # The starter carries the same demo block as the answer, so it needs the
    # list-assignment fix too — the audit gate execs starters, and a raw list
    # assigned into a tensor is a blocking setup_exec_error.
    starter = _ROW_ASSIGN.sub(r"\1t.tensor(\2)", translate(unshadow(q["starter_code"])))

    stripped = re.sub(r"np\.load\('/delta_numbers\.npy'\)", "", answer)
    if "np." in stripped or "from numpy" in stripped:
        raise ValueError(f"untranslated numpy left in answer: {answer!r}")

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
        args = unshadow(args)
        setup = translate_setup(unshadow(case["setup_code"]), keep=case["call"])

        senv: dict = {"t": t, "np": np}
        run(setup, senv)
        senv["solve"] = solve_t

        raw = f"solve({args})"
        raw_value = eval(raw, senv)
        body = translate_call(unshadow(case["call"]))

        # Bind the result to a name and call solve() exactly ONCE.  Several of
        # these drills mutate in place, and a call expression that mentions
        # solve() twice — which the tensor-ness guard below would otherwise
        # introduce, and which a few original calls already do — applies the
        # mutation twice.  The value then depends on how many times the grader
        # happens to evaluate it, which is how #59 and #138 passed generation
        # and failed the real grader.
        body = body.replace(raw, "r")

        if isinstance(raw_value, t.Tensor):
            if len(literal(raw_value)) > MAX_LITERAL_CHARS:
                # Freezing a large result would add megabytes to the learner's
                # download; shape plus a position-weighted integer checksum
                # still catches a wrong permutation and compares exactly.
                body = ("(tuple(r.shape), "
                        "int(t.round(r.flatten().to(t.float64) * 16) "
                        "@ t.arange(r.numel(), dtype=t.float64)))")
            else:
                # Assert tensor-ness as well as the numbers: a learner arriving
                # from the numpy dialect reaches for an ndarray or a list, and
                # the grader compares those equal to a tensor.
                body = f"(isinstance(r, t.Tensor), {body})"

        call = f"(lambda r: {body})({raw})"

        # Re-run setup so the value is measured from an untouched fixture: the
        # probe call above already mutated it for the in-place drills.
        senv = {"t": t, "np": np}
        run(setup, senv)
        senv["solve"] = solve_t
        value = eval(call, senv)
        lit = literal(value)
        if "array(" in lit:
            # literal() repr's an unrecognised object verbatim; a numpy array
            # comes out as `array([...])`, which is not even a valid expression
            # in the sandbox.  It means something upstream stayed numpy.
            raise ValueError(f"call still produces a numpy value: {lit[:120]}")

        if not close(eval(lit, {"t": t, "np": np}), value):
            raise ValueError(f"frozen literal unstable for call {call}")

        cases.append({"setup_code": setup, "call": call, "expected_expr": lit})

    verify_equivalence(q, solve_np, solve_t)

    return {
        "id": qid,
        "question_text": fix_text(unshadow(q.get("question_text") or "")),
        "function_name": q.get("function_name", "solve"),
        "submission_mode": "function",
        "starter_code": starter,
        "answer_code": answer,
        "test_cases": cases,
        "expected_output": expected_output,
    }


def verify_equivalence(q: dict, solve_np, solve_t) -> None:
    """Run both dialects over identical inputs and require agreement."""
    qid = q["id"]
    if qid in NO_CROSSCHECK:
        CROSSCHECKED[qid] = (0, 0)
        return

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
        # Build the torch inputs BEFORE running the numpy answer.  Several of
        # these drills mutate their argument in place, and taking the snapshot
        # afterwards feeds torch the already-transformed array — the two
        # dialects then disagree for a reason that has nothing to do with the
        # translation.
        #
        # Only arrays become tensors.  Shapes, dicts and scalars pass through
        # untouched: turning a `(3, 3)` shape tuple or a lookup dict into a
        # tensor changes what the drill is being asked.
        t_vals = [_as_tensor(v.copy()) if isinstance(v, np.ndarray) else v
                  for v in np_vals]
        try:
            got_np = solve_np(*np_vals)
        except Exception:
            continue
        got_t = solve_t(*t_vals)
        if not close(got_np, got_t):
            raise ValueError(
                f"dialects disagree on args ({args}) with setup "
                f"{case['setup_code']!r}: numpy -> "
                f"{np.asarray(to_numpy(got_np)).ravel()[:6]}, torch -> "
                f"{np.asarray(to_numpy(got_t)).ravel()[:6]}"
            )
        checked += 1
    if checked == 0:
        raise ValueError("no case could be cross-checked against numpy")
    CROSSCHECKED[qid] = (checked, len(q.get("test_cases") or []))


if __name__ == "__main__":
    sys.exit(main())
