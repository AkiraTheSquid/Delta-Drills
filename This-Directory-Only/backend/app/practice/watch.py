"""watch.py — health checks for practice

The routers here are wired to the app by string (a path in `EXPECTED_PATHS`, a
`router` symbol `__init__.py` mounts), so a rename breaks them at runtime as a
404 rather than at import as an error. These checks pin the wiring, and pin the
one invariant whose violation is not an error at all: that every graded attempt
reaches `finalize_attempt`.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import ast
import os
import re
import sys

THIS = os.path.dirname(__file__)
# practice/ lives at backend/app/practice/, so backend/ is two levels up.
sys.path.insert(0, os.path.join(THIS, '..', '..'))

SUB_ROUTERS = [
    'ai_router.py',
    'diagnostic_router.py',
    'feedback_router.py',
    'problem_feedback_router.py',
    'questions_router.py',
    'subtopic_router.py',
]
EXPECTED_PATHS = {
    '/api/practice/next-question',
    '/api/practice/submit',
    '/api/practice/submit-local-eval',
    '/api/practice/override',
    '/api/practice/feedback',
    '/api/practice/problem-feedback',
    '/api/practice/visual-debug',
    '/api/practice/subtopics',
    '/api/practice/weights',
    '/api/practice/run-code',
    '/api/practice/ai-explanation',
    '/api/practice/ai-judge',
    '/api/practice/diagnostic/status',
    '/api/practice/diagnostic/start',
    '/api/practice/diagnostic/answer',
    '/api/practice/diagnostic/finish',
    '/api/practice/diagnostic/decline',
}


def _fastapi_available():
    try:
        import fastapi  # noqa: F401
        return True
    except Exception:
        return False


def _scoring_module():
    """`attempt_scoring`, or None when the minimal env cannot import it.

    A missing THIRD-PARTY dependency (fastapi, pydantic_settings) is a fact
    about where `mod watch` is running, not a health failure — the same reason
    check_imports bails out. A missing `app.*` module is a real failure and is
    re-raised, so this cannot quietly swallow the module going away.
    """
    try:
        from app.practice import attempt_scoring
        return attempt_scoring
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] == "app":
            raise
        return None


def _calls_in(src, func_name):
    """Every function called by name inside the def `func_name`.

    Parsed, not grepped. The previous generation of this check matched the
    plain text of the file, which passed on a router whose only mention of the
    functions it was supposed to call was the PROSE COMMENT above them — the
    bug it existed to catch was live the whole time. A comment, a docstring and
    a name in a string literal are not ast.Call nodes, so none of them can
    satisfy this.
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            calls = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    if isinstance(fn, ast.Name):
                        calls.add(fn.id)
                    elif isinstance(fn, ast.Attribute):
                        calls.add(fn.attr)
            return calls
    raise AssertionError(f"no function named {func_name!r} found")


# ── Import checks ──────────────────────────────
def check_imports():
    if not _fastapi_available():
        return  # `mod watch` runs in a minimal env; skip when fastapi missing
    from app.practice import (  # noqa: F401
        ai_router,
        chatgpt_helpers,
        diagnostic_router,
        feedback_router,
        grading,
        prompts,
        questions_router,
        subtopic_router,
    )
    from app.practice import router  # noqa: F401


# ── Public API checks ─────────────────────────
def check_public_api():
    # Always do the cheap text check so this works even without fastapi installed.
    for fname in SUB_ROUTERS + ['__init__.py']:
        path = os.path.join(THIS, fname)
        with open(path) as f:
            src = f.read()
        assert 'router' in src, f"{fname}: missing 'router' symbol"

    if not _fastapi_available():
        return
    from app.practice import router
    paths = {r.path for r in router.routes}
    missing = EXPECTED_PATHS - paths
    assert not missing, f"router missing endpoints: {sorted(missing)}"


# ── Invariant checks ──────────────────────────
def check_invariants():
    # Each sub-router declares a top-level `router = APIRouter(...)`.
    for fname in SUB_ROUTERS:
        path = os.path.join(THIS, fname)
        with open(path) as f:
            src = f.read()
        assert re.search(r'^router\s*=\s*APIRouter\(', src, re.MULTILINE), \
            f"{fname}: missing top-level `router = APIRouter(...)`"

    # __init__.py mounts every sub-router.
    with open(os.path.join(THIS, '__init__.py')) as f:
        init_src = f.read()
    for fname in SUB_ROUTERS:
        mod = fname[:-3]
        assert mod in init_src, f"__init__.py does not include {mod}"
    # And exposes a single aggregated `router` with the /api/practice prefix.
    assert "prefix=\"/api/practice\"" in init_src or "prefix='/api/practice'" in init_src, \
        "__init__.py must declare prefix='/api/practice' on the aggregated router"


# ── Every graded attempt gets finalized ───────
def check_attempts_are_finalized():
    """A parked attempt that nothing closes out is a silent, total data loss.

    `record_attempt` scores nothing — it parks the attempt in
    `pending_attempt`. Everything that MOVES (`sub_state.n`, history, the
    per-atom BKT posteriors, the mastery snapshot, the target difficulty)
    happens in `finalize_attempt`. `/submit-local-eval` used to do the first
    and never the second, so on the Colab edition — where the notebook's
    checker IS the submit and no felt-difficulty step follows — every attempt
    sat pending until the next one overwrote it. Nothing errored: the rail
    advanced, the grade showed, and the learner's practice existed nowhere.

    So: both exits must reach `finalize_attempt`, and neither may grow its own
    private copy of the scoring tail.
    """
    def src(name):
        with open(os.path.join(THIS, name)) as f:
            return f.read()

    # Every route that records a NEW attempt must first close out one already
    # parked. `record_attempt` overwrites the pending slot outright, so without
    # this a graded answer that never reached /feedback — a Skip, a closed tab,
    # a client running half a deploy behind — is not un-rated, it is gone.
    for route in ('submit_answer', 'submit_local_eval'):
        calls = _calls_in(src('questions_router.py'), route)
        assert 'record_attempt' in calls, f"{route} no longer records an attempt"
        assert 'flush_stale_attempt' in calls, (
            f"{route} records an attempt without flushing the one it is about "
            "to overwrite — that silently loses a real answer"
        )
    local_eval = _calls_in(src('questions_router.py'), 'submit_local_eval')
    assert 'record_attempt' in local_eval, "submit_local_eval must record the attempt"
    assert 'finalize_attempt' in local_eval, (
        "submit_local_eval records an attempt but never finalizes it — that is "
        "the Colab edition losing every answer it takes"
    )

    feedback = _calls_in(src('feedback_router.py'), 'submit_feedback')
    assert 'finalize_attempt' in feedback, "submit_feedback must finalize through the shared path"
    # A second copy of the tail is how the two exits drift apart, one commit at
    # a time, until an answer is worth different amounts depending on which
    # button recorded it.
    for banned in ('apply_feedback', 'apply_attempt', 'subtopic_mastery', 'target_difficulty'):
        assert banned not in feedback, (
            f"submit_feedback calls {banned} directly — the scoring tail belongs "
            f"to attempt_scoring.finalize_attempt, and two copies will drift"
        )

    # The aim the learner is SHOWN and the aim the submit recomputes have to be
    # read the same way, or the bar jumps on submit and jumps back on the next
    # load. Both go through `question_target_difficulty`, keyed to the concept
    # of the question on screen.
    assert 'question_target_difficulty' in _calls_in(
        src('questions_router.py'), 'next_question'
    ), (
        "next_question reports an aim it did not measure on the served "
        "question's concept — finalize_attempt does, and the two will disagree"
    )

    # ORDER inside the shared path. The mastery snapshot has to read POST-attempt
    # BKT, or every attempt records the previous attempt's mastery and the
    # learning-rate chart plots a lag-one copy of itself.
    scoring = src('attempt_scoring.py')

    # The logistic engine has to SEE every graded answer. It was written, tested
    # and left unwired for months — imported by nothing but its own test, with
    # `attempt_log` never written — and nothing failed, because an unwired model
    # is silent rather than wrong.
    assert 'record_attempt_across_kcs' in _calls_in(scoring, 'finalize_attempt'), (
        "finalize_attempt no longer feeds the logistic engine — the posteriors "
        "freeze and attempt_log stops recording, which is the model going quiet "
        "rather than the model going away"
    )
    assert scoring.index('bkt_mastery.apply_attempt') < scoring.index('record_attempt_across_kcs'), (
        "the engine update must run AFTER the BKT update: its `encompassing` "
        "feature is a mean over this concept's atoms, so reading it first feeds "
        "the model the atom posteriors from before the answer it is being told about"
    )

    # `question_target_difficulty`, not the bare `target_difficulty`: the aim has
    # to be measured on the concept the answered question belongs to. The
    # subtopic-wide one averages in every atom of a thirty-atom subtopic the
    # learner has never met, which pinned a real account's aim at 24.5/100
    # through a session that took the concept itself to 0.92.
    for needle in ('apply_feedback', 'apply_attempt', 'subtopic_mastery',
                   'question_target_difficulty'):
        assert needle in _calls_in(scoring, 'finalize_attempt'), \
            f"finalize_attempt no longer calls {needle}"
    assert scoring.index('bkt_mastery.apply_attempt') < scoring.index('subtopic_mastery('), \
        "the BKT update must run BEFORE the mastery snapshot, not after"


def check_finalize_actually_moves_state():
    """The behavioural half: drive one attempt through and watch `n` move.

    String checks above prove the call is written down. This proves it does
    something — and that an unrated attempt is recorded as unrated rather than
    silently borrowing one of the three real feedback levels' alphas.
    """
    scoring = _scoring_module()
    if scoring is None:
        return  # minimal env, third-party deps absent — see _scoring_module
    from app.adaptive import UNRATED, UserPracticeState, record_attempt

    state = UserPracticeState(user_id="watch-check")
    sub = "Numpy: Core array literacy"
    record_attempt(
        user_state=state, question_id=-1, subtopic=sub,
        difficulty_score=50, correct=True,
    )
    assert state.pending_attempt is not None, "record_attempt must park the attempt"
    assert state.get_subtopic_state(sub).n == 0, (
        "record_attempt must NOT count the attempt — that is what finalize is for"
    )

    attempt = scoring.finalize_attempt(state, UNRATED)
    assert attempt is not None, "finalize_attempt returned nothing for a pending attempt"
    after = state.get_subtopic_state(sub)
    assert after.n == 1, f"attempt was not counted (n={after.n})"
    assert state.pending_attempt is None, "finalize_attempt must clear the pending slot"
    assert len(after.history) == 1, "the attempt never reached history"
    assert attempt.feedback == "unrated", f"recorded as {attempt.feedback!r}, not unrated"
    assert attempt.alpha is None, (
        "an unrated attempt must carry no alpha — the learner was never asked, "
        "and borrowing a level's alpha invents an opinion"
    )
    # Finalizing twice must be a no-op, not a second count.
    assert scoring.finalize_attempt(state, UNRATED) is None
    assert state.get_subtopic_state(sub).n == 1, "an attempt was counted twice"
    assert after.difficulty_offset == 0.0, (
        "an UNRATED attempt moved the difficulty offset — nobody was asked, so "
        "nothing was said, and eroding a real correction on every Skip makes "
        "the rating quietly worthless"
    )


def check_felt_difficulty_reaches_the_next_question():
    """The rating has to MOVE something, or the three buttons are decoration.

    They were, for a while: `alpha` is still written onto the attempt record and
    read by nothing, because BKT replaced the EWMA that used to consume it. The
    live path is the per-subtopic offset — the learner's own correction to where
    the queue aims — so this drives it end to end rather than trusting that
    `nudge_difficulty_offset` is called somewhere.
    """
    scoring = _scoring_module()
    if scoring is None:
        return  # minimal env, third-party deps absent — see _scoring_module
    from app.adaptive import DIFFICULTY_OFFSET_LIMIT, UserPracticeState, record_attempt
    from app.prioritization import target_difficulty

    state = UserPracticeState(user_id="watch-check-felt")
    sub = "Numpy: Core array literacy"

    # Returns copies, not the live row: `get_subtopic_state` hands back the same
    # object every time, so holding onto it compares a value with itself.
    def answer(correct, feedback):
        record_attempt(
            user_state=state, question_id=-1, subtopic=sub,
            difficulty_score=50, correct=correct,
        )
        scoring.finalize_attempt(state, feedback)
        row = state.get_subtopic_state(sub)
        return row.difficulty_offset, row.target_difficulty

    base = target_difficulty(state, sub)
    assert not state.subtopic_states, (
        "target_difficulty created a subtopic row just by being asked — it runs "
        "over every subtopic there is, including ones never practised"
    )
    up_off, up_target = answer(True, "a_lot")
    assert up_off > 0 and up_target > base, (
        f"'way too easy' did not raise the aim ({base} -> {up_target})"
    )
    down_off, _ = answer(False, "a_lot")
    assert down_off < up_off, "'way too hard' after a miss did not lower the aim"

    answer(True, "a_lot")
    for _ in range(6):
        settled_off, _ = answer(True, "not_much")
    assert abs(settled_off) < abs(up_off), (
        "'about right' does not decay the correction — one stale click would "
        "outlive every problem that answered it"
    )
    for _ in range(12):
        capped_off, _ = answer(True, "a_lot")
    assert capped_off <= DIFFICULTY_OFFSET_LIMIT, (
        "the offset is not capped — a learner could talk the queue into serving "
        "problems their mastery says they cannot read"
    )


# ── Run all checks ────────────────────────────
if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants,
              check_attempts_are_finalized, check_finalize_actually_moves_state,
              check_felt_difficulty_reaches_the_next_question]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
