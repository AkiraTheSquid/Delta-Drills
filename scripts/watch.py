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
import contextlib
import io
import re
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


# ── The in-notebook checker ───────────────────
def check_colab_grader():
    """The grader compiled into every Colab notebook mirrors the runtime too.

    `dd_check` is what a learner is told to trust when the tutor is not
    watching, and it is the one grader that ships to a machine we cannot
    inspect. Graded here, not grepped, for the same reason as the two mirrors
    above — and against the same two cases, which are the drifts that have
    actually happened.
    """
    import colab_cells

    src = colab_cells.grader_source()
    assert 'def dd_check(' in src, 'grader_source no longer carries dd_check'

    printed = []

    def graded(solution, cases):
        ns = {'__name__': '__main__'}
        exec(src, ns)
        ns['_DD_TESTS'] = {'1': {'fn': 'solve', 'cases': cases}}
        exec(solution, ns)
        # dd_check talks to the learner; a watch run is not the audience. The
        # text is kept, though — it is also the reporting channel (below).
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok = eval('dd_check(1, verbose=False)', ns)
        printed.append(buf.getvalue())
        return ok

    # Mirror 1: numpy is always available, so a fixture may use it even in a
    # torch drill — np.load is the only route to the ARENA image.
    assert graded(
        'def solve(x):\n    return x * 2\n',
        [{'setup_code': 'v = np.array([1, 2, 3])', 'call': 'list(solve(v))',
          'expected_expr': '[2, 4, 6]'}],
    ), 'dd_check no longer provides numpy — ARENA-fixture drills would fail in Colab'

    # Mirror 2: expected_expr is evaluated against a FRESH setup run, so a
    # solution that mutates its input cannot poison its own expectation.
    assert graded(
        'def solve(xs):\n    out = list(xs)\n    xs.clear()\n    return out\n',
        [{'setup_code': 'xs = [1, 2, 3]', 'call': 'solve(xs)', 'expected_expr': 'xs'}],
    ), 'dd_check stopped re-running setup before expected_expr — in-place drills misgrade'

    # And it must still fail a wrong answer: a checker that says yes to
    # everything is worse than no checker at all.
    assert not graded(
        'def solve(xs):\n    return [x * 3 for x in xs]\n',
        [{'setup_code': 'xs = [1, 2, 3]', 'call': 'solve(xs)', 'expected_expr': '[2, 4, 6]'}],
    ), 'dd_check passes a wrong answer'

    # ── The line is the wire ──────────────────────────────────────────
    # `dd_check`'s summary is not just for the learner: it is the ONLY way a
    # notebook can tell the app how a problem went. A cell's rich output is
    # sandboxed away from the Colab page and a beacon would need a token pasted
    # into the notebook, so `extension/content/colab_focus.js` reads this
    # printed text off the DOM. Reword it without telling that file and the app
    # silently stops recording anything a learner does in Colab.
    pattern = r"(✅|❌) Problem (\d+) — "
    focus = os.path.join(_DIR, "..", "extension", "content", "colab_focus.js")
    with open(focus, encoding="utf-8") as fh:
        reader = fh.read()
    assert pattern in reader, (
        f"extension/content/colab_focus.js no longer looks for {pattern!r} — "
        f"it is the only channel from the notebook back to the app"
    )
    assert len(printed) >= 2, "expected dd_check output to inspect"
    for text in printed:
        assert re.search(pattern, text), (
            f"dd_check printed {text.strip()[:80]!r}, which the extension's "
            f"reader ({pattern!r}) cannot parse — the app would record nothing"
        )

    # The fixture the einops drills load has to be somewhere the notebook can
    # reach, and that URL is compiled into every one of them.
    assert colab_cells.FIXTURE_URL.startswith('https://'), f'bad FIXTURE_URL: {colab_cells.FIXTURE_URL}'
    assert 'numbers.npy' in colab_cells.FIXTURE_URL, (
        f'FIXTURE_URL no longer points at the digits fixture: {colab_cells.FIXTURE_URL}'
    )
    publish = open(os.path.join(_DIR, 'publish_colab_notebooks.sh'), encoding='utf-8').read()
    assert 'numbers.npy' in publish, (
        'publish_colab_notebooks.sh stopped shipping numbers.npy — the URL '
        'compiled into every notebook would 404 and the einops drills die'
    )


# ── Solution / prerequisite ratchet ───────────
def check_solution_prereq_ratchet():
    """No drill may require syntax its own concept has not been taught yet.

    `audit_solution_prereqs.py` reads every solution in the bank, collects
    every function, method, attribute and language construct it uses, and asks
    the prerequisite lattice whether a lesson for it exists AT OR BEFORE that
    drill's concept. `a.T` on the first concept of the course is the case this
    was built for: eight faded drills asked for it, the page that teaches
    transposition sat four lessons later, and nothing failed.

    Widened 2026-08-29 from the solution alone to every surface — the solution
    AND the problem the learner is handed. A starter is not just a faded
    solution: at the worked rung it IS the code, and it carries scaffold lines
    the solution never shows. That widening is what took the known backlog from
    646 to 1464.

    The corpus carries a large backlog, so this is a RATCHET, not a gate:
    `solution_prereq_baseline.json` records what is already broken and this
    fails only on something NEW. Fixing content leaves stale entries behind —
    those are reported by the audit, not failed on, because a shrinking
    baseline should never turn the build red.
    """
    import audit_solution_prereqs as A

    violations = A.find(A.SURFACES)
    known = A.load_baseline()
    assert known is not None, (
        'solution_prereq_baseline.json is missing — the ratchet cannot tell '
        'new debt from old; re-record it with '
        'audit_solution_prereqs.py --write-baseline')
    new = sorted({A.key(v) for v in violations} - known)
    assert not new, (
        f'{len(new)} drill(s) require syntax their concept has not reached: '
        + '; '.join(new[:6])
        + '  — teach it earlier, retag the drill, or rewrite the solution. '
          'Re-recording the baseline is admitting the debt, not fixing it.')

    # The check has to actually be able to see the case it exists for. This is
    # the shape of the original bug, asserted against live data rather than
    # remembered: transposition is taught somewhere, and somewhere LATER than
    # the course's first concept.
    declared, kc_of_page = A.declaring_kcs()
    rank = A.lesson_order(kc_of_page)
    owner = A.owner_of('Tensor.T', declared, rank)
    assert owner is not None, 'no lesson declares Tensor.T — the audit is blind to it'
    first = 'numpy.ndarray-model'
    assert rank.get(owner, -1) > rank.get(first, 0), (
        f'Tensor.T is owned by {owner}, which no longer sits after {first}; '
        'if that is deliberate, this assertion is the thing to update')


def check_solution_symbol_coverage():
    """The collector must not walk past a construct without naming it.

    "Every function whatsoever needs a prerequisite" is only a guarantee while
    nothing is invisible to the pass that collects them. A drill written with a
    construct the collector has no visitor for would sail through the ratchet
    reporting nothing, which is the worst possible failure for a guard: green,
    and blind.
    """
    import json
    import solution_symbols as S
    import audit_solution_prereqs as A

    bank = json.loads(A.QUESTIONS.read_text(encoding='utf-8'))
    questions = bank if isinstance(bank, list) else bank.get('questions', bank)
    sources = [q.get('answer_code') or '' for q in questions]
    sources += [q.get('starter_code') or '' for q in questions]
    missed = S.unhandled_node_types(sources)
    assert not missed, (
        f'solution_symbols has no visitor for {sorted(missed)} — a drill uses '
        'it and the prerequisite audit cannot see it. Add a visitor, or list '
        'the node in STRUCTURAL with the reason it teaches nothing.')

    # And the collector must still see the thing it was built for.
    assert 'Tensor.T' in S.collect('def solve(a):\n    return a.T\n'), \
        'attribute access stopped being collected — a.T is the case this exists for'


def check_arena_grounding_ratchet():
    """No drill may teach a function ARENA never uses.

    ARENA is what the course exists to prepare people for, so it is the source
    of truth for what is worth teaching, and the test is empirical: a library
    symbol appearing in ZERO of the 458 notebooks is attention spent on
    something no learner will ever need. `torch.einsum` is the case this was
    built for — 69 of our drill solutions are written in it and the corpus does
    not contain one notebook that uses it; ARENA writes `einops.einsum`, whose
    arguments come in a different order.

    A RATCHET, like the prerequisite check above: `arena_grounding_baseline.json`
    holds the existing backlog and this fails only on something NEW.
    """
    import guard_checks

    guard_checks.check_arena_grounding()


def check_declared_symbols_are_drilled():
    """Every symbol a concept declares must be drilled twice ON that concept.

    A KP page's `new_syntax:` is the graph claiming that mastery of the
    concept includes mastery of the symbol, and both mastery models take that
    literally: one estimate per concept, and the lattice gates on it. So a
    declared symbol with no drills of its own is marked learned on evidence
    collected about something else. `numpy.random-generator` declares ten and
    drills five, across three questions.

    A RATCHET like the two above, with one addition: the baseline records the
    drill COUNT, so deleting the single drill that was holding a symbol at one
    fails even though the symbol was already known debt.
    """
    import guard_checks

    guard_checks.check_symbol_coverage()


def check_arena_index_is_current():
    """The frozen corpus summary must still describe the corpus on disk.

    The grounding guard answers from `arena_symbol_index.json` rather than
    rescanning 458 notebooks, which is what makes it fast enough for a
    watcher. The cost of that is an artifact that can go stale in silence, and
    a stale guard is worse than none — it keeps answering, confidently, about
    a corpus that is no longer there. Only the notebook COUNT is verified
    here; recomputing the symbols needs torch, and this is the failure that
    actually happens (the corpus is updated, or moved, and nobody re-ran the
    scan).
    """
    import guard_checks

    guard_checks.check_arena_index_is_current()


def check_graph_structure_ratchet():
    """The lattice must record the dependencies the content actually has.

    `audit_graph_structure.py` asks four structural questions the other audits
    cannot: registry/atom-graph sanity (cycles, dangling ids, encompassing
    weights the runtime silently drops), ORDER-ONLY dependencies (a drill uses
    a symbol taught earlier in course order with NO prerequisite path behind
    it — the prereq ratchet is blind to this by design, it ranks by order),
    duplicate moves on one rung (two solutions identical after normalization),
    and rung-difficulty inversions. Backlog exists, so: RATCHET —
    `graph_structure_baseline.json` records it, this fails only on new debt.
    """
    import audit_graph_structure as G

    findings = G.find()
    known = G.load_baseline()
    assert known is not None, (
        'graph_structure_baseline.json is missing — re-record it with '
        'audit_graph_structure.py --write-baseline')
    new = sorted({f["key"] for f in findings} - known)
    assert not new, (
        f'{len(new)} new graph-structure finding(s): ' + '; '.join(new[:6])
        + '  — add the edge, differentiate the drill, or argue the baseline.')

    # Detector health, corpus-independent (codex round 1: asserting the
    # CORPUS still has edge-missing debt meant the ratchet could never
    # reach clean). Synthetic cases instead: the duplicate detector must
    # collapse a rename+numeric change, refuse the empty program, and
    # the ancestor walk must be transitive and directional.
    same = 'def f(x):\n    return x + 1\n'
    also = 'def g(y):\n    return y + 2\n'
    assert G._move_hash(same) == G._move_hash(also), 'move-hash went blind'
    assert G._move_hash('') is None, 'empty program must not hash'
    anc = G._ancestors({'a': [], 'b': ['a'], 'c': ['b']})
    assert anc['c'] == {'a', 'b'} and anc['a'] == set(), 'ancestor walk broken'


def check_prose_prereq_ratchet():
    """No page may USE a word before the page that defines it — prose twin of
    the solution-prereq ratchet. Vocabulary comes from lessons/glossary.js
    (already term->KC mapped, guarded by watch_jargon.py); this only asks the
    ordering question, over prose with all code stripped. RATCHET against
    `prose_prereq_baseline.json`; fails on new debt only.
    """
    import audit_prose_prereqs as P

    findings = P.find()
    known = P.load_baseline()
    assert known is not None, (
        'prose_prereq_baseline.json is missing — re-record it with '
        'audit_prose_prereqs.py --write-baseline')
    new = sorted({f["key"] for f in findings} - known)
    assert not new, (
        f'{len(new)} page(s) use a term before its defining page: '
        + '; '.join(new[:6])
        + '  — reword, define earlier, or argue the baseline.')

    # Detector health: the audit exists because kp-dots-and-imports (py-0)
    # says "tensor" one lesson before numpy.ndarray-model defines it. That is
    # baselined debt; assert the detector still SEES it while it exists, and
    # allow a clean corpus once the page is reworded and the baseline shrunk.
    if known:
        assert findings, (
            'baseline lists prose debt but the audit found NOTHING — the '
            'detector went blind (glossary parse, prose stripping, or rank '
            'lookup broke), it did not heal 41 pages at once')


# ── Run all checks ────────────────────────────
if __name__ == '__main__':
    # These checks are written as asserts, which `python -O` strips entirely —
    # a stripped run reports success having verified nothing. Refuse instead.
    if not __debug__:
        print('FAIL: watch.py needs assertions enabled (do not run under -O)',
              file=sys.stderr)
        sys.exit(1)
    checks = [check_imports, check_public_api, check_invariants, check_colab_grader,
              check_solution_prereq_ratchet, check_solution_symbol_coverage,
              check_arena_grounding_ratchet, check_declared_symbols_are_drilled,
              check_arena_index_is_current, check_graph_structure_ratchet,
              check_prose_prereq_ratchet]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
