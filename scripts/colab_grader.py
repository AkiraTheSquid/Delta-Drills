#!/usr/bin/env python3
"""colab_grader.py — the checker that runs INSIDE a generated Colab notebook.

This file is not imported by the notebook. Everything after the `embed:start`
marker below is copied verbatim into one cell by
`generate_colab_notebooks.py`, because a Colab notebook opened from GitHub has
no way to import a module that lives in this repo.

WHY IT IS A FILE AND NOT A STRING LITERAL
    It is a fourth implementation of the same grading rule — the backend's
    `code_runner.py` harness, the guest Pyodide harness in `practice/api.js`,
    and `validate_lessons.py` are the other three. Drift between them means a
    learner is told "wrong" by one and "right" by another for the same code,
    which is the single most corrosive bug this tutor can have. Keeping it as
    real Python means `scripts/watch.py` can exec it and GRADE something,
    rather than grep the generator for a comment that may no longer describe
    what the string contains.

THE RULE, in one line: compare with a float tolerance only when a float is
involved, elementwise for arrays and tensors, and re-run the fixture setup
before evaluating the expectation so a solution that mutates its input cannot
poison its own expected value.

Run this file directly for a smoke test of the comparison rule.
"""

from __future__ import annotations

# --- embed:start -------------------------------------------------------------
# Delta Drills — problem checker. Generated; see scripts/colab_grader.py.
#
# `dd_check(<problem id>)` runs your `solve` against the same cases the tutor
# grades with, and tells you which ones failed. It reads `solve` out of the
# notebook, so define it (run your cell) before you check.
import base64
import json
import sys
import zlib

import numpy as np

# Filled in by the generated cell that follows this source: {qid: {fn, cases}}.
_DD_TESTS = {}

# Where the ARENA digits fixture is fetched from, also filled in by that cell.
_DD_FIXTURE_URL = ""
_DD_FIXTURE_PATH = "/delta_numbers.npy"

_DD_RTOL = 1e-5
_DD_ATOL = 1e-6


def _dd_install_fixtures():
    """Make `np.load('/delta_numbers.npy')` work here the way it does in the app.

    24 of the einops drills are written against the ARENA digits image, and the
    bank refers to it by an absolute path the backend rewrites at grade time
    (`code_runner.CODE_PREAMBLE`). Nothing rewrote it in a notebook, so those
    problems could not run at all in Colab — not the checker, not the starter
    code the learner was sent there to fill in. Downloaded on first use, so the
    six notebooks that never touch it never pay for it.
    """
    import os
    import urllib.request

    original = np.load
    if getattr(original, "_dd_patched", False):
        return

    def _load(file, *args, **kwargs):
        if str(file) == _DD_FIXTURE_PATH and not os.path.exists(_DD_FIXTURE_PATH):
            if not _DD_FIXTURE_URL:
                raise FileNotFoundError(
                    "This drill needs the ARENA digits fixture and no source was "
                    "compiled into this notebook — regenerate it."
                )
            urllib.request.urlretrieve(_DD_FIXTURE_URL, _DD_FIXTURE_PATH)
        return original(file, *args, **kwargs)

    _load._dd_patched = True
    np.load = _load


def _dd_load(blob):
    """The test payload, deflated and base64'd.

    Not encryption and not pretending to be — it is one `zlib.decompress` away.
    It is compressed because the payload for a 84-problem notebook is ~80 KB of
    JSON, and out of sight because an expanded grader cell would otherwise sit
    in the notebook spelling out the expected answer to every problem below it.
    """
    return json.loads(zlib.decompress(base64.b64decode(blob)).decode("utf-8"))


def _dd_preflight_torch():
    """Import torch once, here, where a failure can still be explained.

    Every drill cell opens with `import torch as t`, so the learner meets a
    broken torch install as a traceback through torch's own internals — the one
    reported was `AttributeError: partially initialized module 'torch' has no
    attribute 'fx'` from `torch/_export/utils.py`, raised while evaluating a
    function's annotations. That message names neither the cause nor the cure,
    and it is not even the real error: it is what a LATER import sees after an
    earlier one died partway and left the half-built module in `sys.modules`.
    Python does unwind a failed import normally, but a torch that was swapped
    on disk under a running kernel (a `pip install` mid-session) or shadowed by
    a stray `torch.py` gets far enough in to be cached before it falls over.

    So: purge the wreckage and retry ONCE, which is the whole fix whenever the
    first failure was transient, and report what actually broke when it is not.
    Importing torch in this cell rather than lazily is safe now in a way the
    `_dd_tensor` comment below still guards against for the per-comparison
    path — the bank is 448/448 torch and every notebook imports it a few cells
    down, so there is no numpy-only notebook left to charge for it.

    Never raises: a checker that refuses to load over this would take the
    lesson down with the runtime.
    """

    def _purge():
        # Submodules too, and that is the whole point. Python drops only the
        # module that raised, so `torch` goes and a `torch._export` imported
        # seconds earlier STAYS — and the next `import torch` re-runs
        # `torch/__init__.py` straight back into that stale submodule, which
        # reaches for a `torch.fx` the half-built parent has not bound yet.
        # Leaving one behind reproduces the bug instead of clearing it.
        for name in [n for n in sys.modules if n == "torch" or n.startswith("torch.")]:
            del sys.modules[name]

    def _usable(mod):
        # `import torch` does NOT re-execute a module already in sys.modules,
        # so a corpse left by a failed import is imported "successfully" and
        # the error surfaces later, from the learner's own cell. Judge the
        # object, not the statement: a torch that finished has both of these.
        return hasattr(mod, "fx") and hasattr(mod, "__version__")

    cached = sys.modules.get("torch")
    if cached is not None and not _usable(cached):
        _purge()

    for attempt in (1, 2):
        try:
            import torch
            if not _usable(torch):
                raise ImportError(
                    "torch imported but is only partially initialised "
                    "(no .fx) — an earlier import in this session died partway"
                )
            return True
        except Exception as exc:
            if attempt == 1:
                _purge()
                continue
            print(
                "⚠️  This runtime cannot import PyTorch, so no drill in this "
                "notebook will run.\n"
                "    %s: %s\n"
                "    Fix: Runtime ▸ Disconnect and delete runtime, then reopen "
                "this notebook and run\n"
                "    this cell first. If it comes back, check for a file named "
                "torch.py in /content,\n"
                "    and re-run any pip install BEFORE anything imports torch."
                % (type(exc).__name__, exc)
            )
    return False


def _dd_tensor(value):
    # torch only if something already imported it. numpy-only notebooks must
    # not pay a torch import to compare two lists of ints.
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def _dd_close(a, b):
    """Tolerance compare, but ONLY when a float or complex is involved.

    torch defaults to float32 where numpy defaults to float64 and honest
    answers differ in reduction order, so exact equality fails correct work.
    Integer and boolean results stay exact — an index answer (argmax, nonzero,
    searchsorted) must never be fudged by a tolerance. Returns None to mean
    "not a float comparison, use exact equality".
    """
    try:
        floaty = any(
            np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.complexfloating)
            for x in (a, b)
        )
        if not floaty:
            return None
        if a.shape != b.shape:
            return False
        return bool(np.allclose(a, b, rtol=_DD_RTOL, atol=_DD_ATOL, equal_nan=True))
    except Exception:
        return None


def _dd_array_equal(a, b):
    close = _dd_close(a, b)
    if close is not None:
        return close
    return bool(np.array_equal(a, b))


def _dd_equal(a, b):
    if _dd_tensor(a) or _dd_tensor(b):
        try:
            a2 = a.detach().cpu().numpy() if _dd_tensor(a) else np.asarray(a)
            b2 = b.detach().cpu().numpy() if _dd_tensor(b) else np.asarray(b)
            return _dd_array_equal(a2, b2)
        except Exception:
            # dtypes numpy cannot hold (bfloat16, conj views): equal tensors
            # must not grade as unequal — ask torch itself.
            torch = sys.modules.get("torch")
            if torch is not None and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                try:
                    return bool(torch.equal(a.detach().cpu().resolve_conj(),
                                            b.detach().cpu().resolve_conj()))
                except Exception:
                    return False
            return False
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return _dd_array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_dd_equal(x, y) for x, y in zip(a, b))
    close = _dd_close(np.asarray(a), np.asarray(b))
    if close is not None:
        return close
    return bool(a == b)


def _dd_seed():
    # The same seed before the actual-side and the expected-side setup runs, for
    # BOTH rngs: setup executes twice, so an unseeded torch.rand in a fixture
    # would hand the two sides different data and fail an honest answer.
    np.random.seed(0)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(0)


def _dd_show(value, limit=320):
    try:
        text = repr(value)
    except Exception as exc:
        text = "<unrepresentable: %s>" % type(exc).__name__
    text = " ".join(text.split()) if len(text) > limit else text
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def dd_check(question_id, verbose=True):
    """Grade the `solve` you just defined against this problem's cases.

    Returns True when every case passes. Prints which ones did not, with the
    fixture, what was expected and what came back — a failing grade should be
    evidence you can act on, not a verdict.
    """
    qid = str(question_id)
    entry = _DD_TESTS.get(qid)
    if entry is None:
        print("No checker for problem %s in this notebook." % qid)
        return False

    # The learner's namespace, not this function's: `solve` lives in the cell
    # they ran, and in Colab that is the caller's globals.
    try:
        env = sys._getframe(1).f_globals
    except Exception:
        env = globals()

    fn_name = entry.get("fn") or "solve"
    if fn_name not in env:
        print("❌ `%s` is not defined yet — run your solution cell first." % fn_name)
        return False

    cases = entry.get("cases") or []
    failures = []
    for i, case in enumerate(cases, 1):
        # A fresh copy per case: fixtures are exec'd, and exec'ing them into the
        # notebook's own globals would quietly overwrite whatever the learner
        # named `x` two cells ago.
        ns = dict(env)
        try:
            if case.get("setup_code"):
                _dd_seed()
                exec(case["setup_code"], ns)
            actual = eval(case["call"], ns)
            expected_setup = case.get("expected_setup_code") or case.get("setup_code")
            if expected_setup:
                _dd_seed()
                exec(expected_setup, ns)
            expected = eval(case["expected_expr"], ns)
            if not _dd_equal(actual, expected):
                failures.append((i, case, _dd_show(expected), _dd_show(actual), ""))
        except Exception as exc:
            failures.append((i, case, "", "", "%s: %s" % (type(exc).__name__, exc)))

    total = len(cases)
    if not failures:
        print("✅ Problem %s — %d/%d cases passed." % (qid, total, total))
        return True

    print("❌ Problem %s — %d of %d cases failed." % (qid, len(failures), total))
    if verbose:
        for i, case, expected, actual, error in failures:
            print("\n  case %d" % i)
            if case.get("setup_code"):
                for line in case["setup_code"].strip().splitlines():
                    print("    given     %s" % line)
            print("    called    %s" % _dd_show_source(case.get("call", "")))
            if error:
                print("    raised    %s" % error)
            else:
                print("    expected  %s" % expected)
                print("    you got   %s" % actual)
    return False


def _dd_show_source(text, limit=160):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
# --- embed:end ---------------------------------------------------------------


if __name__ == "__main__":
    # Smoke test of the rule this file exists to hold still.
    _DD_TESTS = {
        "1": {"fn": "solve", "cases": [
            {"setup_code": "xs = [1, 2, 3]", "call": "solve(xs)", "expected_expr": "[2, 4, 6]"},
        ]},
        "2": {"fn": "solve", "cases": [
            # A solution that empties its input: only passes if the fixture is
            # re-run before the expectation is evaluated.
            {"setup_code": "xs = [1, 2, 3]", "call": "solve(xs)", "expected_expr": "xs"},
        ]},
    }

    def solve(xs):
        return [x * 2 for x in xs]

    ok_a = dd_check(1)

    def solve(xs):  # noqa: F811 — second scenario
        out = list(xs)
        xs.clear()
        return out

    ok_b = dd_check(2)

    def solve(xs):  # noqa: F811 — a wrong answer must fail
        return [x * 3 for x in xs]

    _DD_TESTS["1"]["cases"][0]["expected_expr"] = "[2, 4, 6]"
    ok_c = dd_check(1)

    print()
    print("PASS" if (ok_a and ok_b and not ok_c) else "FAIL")
    raise SystemExit(0 if (ok_a and ok_b and not ok_c) else 1)
