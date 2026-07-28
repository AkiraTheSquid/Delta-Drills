"""watch.py — health checks for the lesson toolchain

`validate_lessons.py` is the release gate for lesson content and
`compile_lessons.py` writes the JSON the app serves, so a break here is
silent: the lessons keep rendering from the last compile while the gate that
would have caught bad content no longer runs.

The invariant with teeth is that this validator grades the way the RUNTIME
grades. It re-implements a slice of `backend/app/code_runner.py`, and every
time the two drift, a lesson either fails here for a reason that cannot
happen in the sandbox, or passes here and breaks for a learner. Both have
happened (see README: expected-setup fallback, numpy preamble), so the drift
is checked rather than remembered.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, '..'))
sys.path.insert(0, _DIR)


# ── Import checks ──────────────────────────────
def check_imports():
    """The lesson toolchain must import — these are run by hand and by deploy."""
    import lesson_lib  # noqa: F401
    import validate_lessons  # noqa: F401
    import compile_lessons  # noqa: F401


# ── Public API checks ─────────────────────────
def check_public_api():
    """Names validate_lessons.py and compile_lessons.py import from lesson_lib."""
    import lesson_lib
    for name in ('LESSONS_DIR', 'REPO', 'all_kp_paths', 'code_fences',
                 'load_bank', 'load_registry', 'parse_kp', 'split_items'):
        assert hasattr(lesson_lib, name), f'lesson_lib lost {name}'

    import validate_lessons
    for name in ('grade_against_bank', 'run_code', 'values_equal'):
        assert callable(getattr(validate_lessons, name, None)), \
            f'validate_lessons.{name} is not callable'


# ── Invariant checks ──────────────────────────
def check_invariants():
    import validate_lessons

    # Both mirrorings are checked by GRADING something, not by reading the
    # source: a check that greps for `import numpy as np` still passes if the
    # call is deleted and the comment stays, which is precisely the drift it
    # exists to catch.
    def graded(solution, setup, call, expected_expr, expected_setup=None):
        case = {'setup_code': setup, 'call': call, 'expected_expr': expected_expr}
        if expected_setup is not None:
            case['expected_setup_code'] = expected_setup
        return validate_lessons.grade_against_bank(
            solution, {'exercise': {'test_cases': [case]}})

    # Mirror 1: the runtime preamble always provides numpy, so a fixture may
    # use it even in a torch drill — np.load is the only route to the ARENA
    # image. A torch solution imports torch, never numpy, so if the validator
    # does not seed numpy itself this setup dies with NameError.
    failures = graded('def solve(x):\n    return x * 2\n',
                      'v = np.array([1, 2, 3])', 'list(solve(v))', '[2, 4, 6]')
    assert not failures, (
        'grade_against_bank no longer seeds numpy — torch-dialect lessons whose '
        f'fixtures load /delta_numbers.npy will fail here but pass in the sandbox: {failures}'
    )

    # Mirror 2: expected_expr is evaluated against a FRESH setup run, so a
    # solution that mutates its input cannot poison its own expectation. This
    # solution empties the list it is handed; the expectation only holds if
    # setup is re-run before expected_expr is evaluated.
    failures = graded('def solve(xs):\n    out = list(xs)\n    xs.clear()\n    return out\n',
                      'xs = [1, 2, 3]', 'solve(xs)', 'xs')
    assert not failures, (
        'grade_against_bank stopped re-running setup before expected_expr — '
        f'in-place drills will grade differently here than in prod: {failures}'
    )

    # The bank fixture path rewrite has to point at a file that exists, or every
    # ARENA-image lesson fails with a confusing FileNotFoundError.
    assert os.path.exists(validate_lessons.NUMBERS_NPY), (
        f'NUMBERS_NPY missing: {validate_lessons.NUMBERS_NPY}'
    )


# ── Run all checks ────────────────────────────
if __name__ == '__main__':
    # These checks are written as asserts, which `python -O` strips entirely —
    # a stripped run reports success having verified nothing. Refuse instead.
    if not __debug__:
        print('FAIL: watch.py needs assertions enabled (do not run under -O)',
              file=sys.stderr)
        sys.exit(1)
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
