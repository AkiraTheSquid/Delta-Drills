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
    '/api/practice/lesson-feedback',
    '/api/practice/problem-feedback/revisions',
    '/api/practice/problem-feedback/rollback',
    '/api/practice/problem-feedback/repair-queue',
    '/api/practice/problem-feedback/repair-queue/claim',
    '/api/practice/problem-feedback/repair-queue/complete',
    '/api/practice/visual-debug',
    '/api/practice/subtopics',
    '/api/practice/weights',
    '/api/practice/run-code',
    '/api/practice/ai-explanation',
    '/api/practice/ai-judge',
    '/api/practice/ai-tutor',
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
    assert 'ladder_fields' in _calls_in(src('questions_router.py'), 'submit_answer'), (
        "submit response lost fresh ladder estimate — progress would wait for "
        "felt-difficulty feedback instead of moving on answer"
    )
    local_eval = _calls_in(src('questions_router.py'), 'submit_local_eval')
    assert 'record_attempt' in local_eval, "submit_local_eval must record the attempt"
    assert 'finalize_attempt' in local_eval, (
        "submit_local_eval records an attempt but never finalizes it — that is "
        "the Colab edition losing every answer it takes"
    )

    feedback = _calls_in(src('feedback_router.py'), 'submit_feedback')
    assert 'finalize_attempt' in feedback, "submit_feedback must finalize through the shared path"
    assert 'kc_estimate' in feedback, (
        "feedback response lost fresh KC ladder estimate — current rung would "
        "stay frozen until next question"
    )
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
    # 🔴 The engine scores BEFORE the BKT block, and this assertion is the only
    # thing standing between the attempt log and a flattering lie. `encompassing`
    # is a mean over the concept's atoms — the atoms BKT is about to move with
    # this same answer — so scoring afterwards computes `predicted_p` from a
    # feature that already knows the outcome it claims to precede. Nothing would
    # fail; the Brier score would simply come out good and mean nothing. This
    # file previously asserted the OPPOSITE ordering, for a reason that sounded
    # right (feed the model current state) and was exactly backwards: a
    # prediction has to precede the outcome in INFORMATION, not in line number.
    assert scoring.index('record_attempt_across_kcs') < scoring.index('bkt_mastery.apply_attempt'), (
        "the engine now scores AFTER the BKT update, so its `encompassing` "
        "feature already encodes the answer being predicted — every logged "
        "predicted_p is contaminated by its own label and calibration is void"
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

    # 🔴 THE SMALLEST ANSWER IS STILL AN ANSWER (2026-08-28). The three
    # choices used to be "About right / A bit off / Way off", where the first
    # meant "stop correcting" and moved the offset by nothing. They are now
    # "Slightly / Somewhat / Significantly harder-or-easier": every one of
    # them asks for a step, and a `not_much` that moved nothing would push the
    # aim the OPPOSITE way from the words on the button, because the decay
    # pulls the offset back toward the model's own number.
    fresh = UserPracticeState(user_id="watch-check-felt-slight")

    def answer_on(st, correct, feedback):
        record_attempt(
            user_state=st, question_id=-1, subtopic=sub,
            difficulty_score=50, correct=correct,
        )
        scoring.finalize_attempt(st, feedback)
        return st.get_subtopic_state(sub).difficulty_offset

    slight_off = 0.0
    for _ in range(8):
        slight_off = answer_on(fresh, True, "not_much")
    assert slight_off > 0, (
        "'slightly harder' does not raise the aim at all — the smallest of "
        "the three choices is behaving like the neutral answer it replaced, so "
        "a learner asking for a harder problem gets one pitched no higher"
    )

    # ...and it must stay the SMALLEST answer. The decay still lives in the
    # nudge (it runs before the step now), so a sustained request converges
    # instead of running away, and a sustained small request has to converge
    # somewhere below a sustained large one or the sizes mean nothing.
    loud = UserPracticeState(user_id="watch-check-felt-loud")
    loud_off = 0.0
    for _ in range(8):
        loud_off = answer_on(loud, True, "a_lot")
    assert slight_off < loud_off, (
        f"'slightly harder' settles at {slight_off}, not below 'significantly "
        f"harder' at {loud_off} — the three choices no longer size the step"
    )

    # A correction still fades when the learner stops asking for it: eight
    # small requests must not preserve a big old one at full size.
    faded = answer_on(loud, True, "not_much")
    assert faded < loud_off, (
        "a standing correction never decays — one 'significantly harder' would "
        "outlive every problem that answered it"
    )
    for _ in range(12):
        capped_off, _ = answer(True, "a_lot")
    assert capped_off <= DIFFICULTY_OFFSET_LIMIT, (
        "the offset is not capped — a learner could talk the queue into serving "
        "problems their mastery says they cannot read"
    )


def check_ai_repairs_are_gated():
    """The AI question repair writes to the LIVE bank, so its gates are the
    only thing between one bad model response and every learner seeing it.

    None of these gates fails loudly when removed — a question just quietly
    becomes something other than what it was, which is exactly the class of
    change this file exists to pin.
    """
    # Importing this module goes through app/practice/__init__.py, which pulls
    # in the routers — same bail-out the other checks take outside the venv.
    if not _fastapi_available():
        return

    from app.practice.feedback_ai_improver import (
        is_actionable_tag, validated_changes,
    )

    snapshot = {
        "id": -1, "question_text": "original prompt",
        "starter_code": "def solve():\n    pass", "answer_code": "print(1)",
    }

    def repair(**kw):
        base = dict(verdict="rewrite", rationale="r", question_text="",
                    starter_code="", answer_code="")
        base.update(kw)
        return base

    assert not is_actionable_tag("good"), (
        "'good' is praise, not a defect report — rewriting on it churns "
        "questions that are already working"
    )

    assert "answer_code" not in validated_changes(
        repair(answer_code="print(2)"), snapshot, "unclear"), (
        "answer_code is reachable without a 'broken' flag — the reference "
        "answer decides whether every future attempt is graded right or wrong"
    )
    assert "answer_code" in validated_changes(
        repair(answer_code="print(2)"), snapshot, "broken"), (
        "a 'broken' flag can no longer repair the reference answer"
    )

    for field in ("starter_code", "answer_code"):
        assert field not in validated_changes(
            repair(**{field: "def solve(:\n  bad"}), snapshot, "broken"), (
            f"{field} is written without compiling — a SyntaxError here breaks "
            f"the grader for that question on every future attempt"
        )

    assert not validated_changes(repair(question_text="original prompt"), snapshot, "unclear"), (
        "an identical rewrite is still applied — that writes a revision-log "
        "entry claiming a change that never happened"
    )
    assert not validated_changes(repair(), snapshot, "broken"), (
        "empty fields count as a rewrite — the model uses '' to mean 'leave "
        "this alone', so this would blank the question"
    )
    assert not validated_changes(
        repair(verdict="no_change", question_text="clearer prompt"), snapshot, "unclear"), (
        "a no_change verdict still writes its fields — the runner and the "
        "endpoint both pass the whole payload through, so the verdict is the "
        "only thing saying 'I decided not to'"
    )
    assert validated_changes(repair(question_text="clearer prompt"), snapshot, "unclear") == {
        "question_text": "clearer prompt"}, "a genuine prompt fix no longer applies"

    # A compiling answer can still be WRONG, and a wrong reference answer marks
    # every future learner wrong. The runner checks this, but the completion
    # endpoint takes a rewrite from anything with an allowlisted token, so the
    # server has to check it too.
    from app.practice.feedback_ai_improver import verify_answer_code

    cases = [{
        "setup_code": "z = [[1, 9, 3], [7, 2, 5]]",
        "call": "solve(z)",
        "expected_expr": "[1, 0]",
    }]
    ok, _ = verify_answer_code(
        "def solve(z):\n    return [max(range(len(r)), key=lambda i: r[i]) for r in z]\n", cases)
    assert ok, "verify_answer_code rejects a correct answer — every repair would be dropped"
    ok, detail = verify_answer_code("def solve(z):\n    return [0 for _ in z]\n", cases)
    assert not ok, (
        "verify_answer_code accepts an answer that fails the question's own "
        "test cases — that is the one gate a compile check cannot stand in for"
    )
    ok, _ = verify_answer_code("def solve(z):\n    return z\n", [])
    assert ok, "a question with no test cases must stay repairable"


def check_repair_runs_off_the_local_cli():
    """The repair loop must not grow a server-side model credential again.

    The whole point of the redesign is that the model runs on Seth's machine,
    under his login, in a read-only sandbox — a server that can call a model
    directly is a server that can rewrite the bank with nobody watching. Two
    things pin that: no API-key path in the backend, and a sandbox that denies
    by default rather than by list.
    """
    improver = os.path.join(THIS, 'feedback_ai_improver.py')
    with open(improver, encoding='utf-8') as fh:
        source = fh.read()
    for needle in ('import anthropic', 'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN'):
        assert needle not in source, (
            f"{needle} is back in feedback_ai_improver.py — repairs are supposed "
            "to run through the local claude CLI, not a server-side credential"
        )

    guard = os.path.abspath(os.path.join(
        THIS, '..', '..', '..', '..', 'ops', 'question_repair', 'sandbox_guard.py'))
    assert os.path.exists(guard), (
        f"the repair sandbox hook is missing ({guard}) — without it the local "
        "session runs with --dangerously-skip-permissions and nothing else"
    )
    with open(guard, encoding='utf-8') as fh:
        guard_source = fh.read()
    tree = ast.parse(guard_source)
    allowed = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == 'ALLOWED_TOOLS' for t in node.targets
        ):
            allowed = {c.value for c in ast.walk(node.value) if isinstance(c, ast.Constant)}
    assert allowed is not None, "sandbox_guard.py no longer defines ALLOWED_TOOLS"
    for writer in ('Write', 'Edit', 'NotebookEdit', 'Bash', 'Task', 'WebFetch'):
        assert writer not in allowed, (
            f"the repair sandbox now allows {writer} — that session runs with "
            "permissions bypassed, so this is the only thing stopping it"
        )


def check_repair_queue_never_loses_an_open_job():
    """Pruning the queue must not drop work that is still waiting.

    The queue file is rewritten whole on every status change, which makes an
    over-eager prune a silent way to lose a learner's report — there is no
    second copy of a pending job anywhere.
    """
    if not _fastapi_available():
        return
    import tempfile

    from app import feedback_repair_queue as q

    with tempfile.TemporaryDirectory() as tmp:
        previous = os.environ.get('DELTA_FEEDBACK_AI_DIR')
        os.environ['DELTA_FEEDBACK_AI_DIR'] = tmp
        try:
            for i in range(5):
                job = q.enqueue(question_id=100 + i, tag='unclear', note='n',
                                correct=None, user_email='t@example.com')
                if i < 3:
                    q.finish(job['job_id'], status=q.DONE)
            q.prune(keep=1)
            statuses = [j['status'] for j in q.load_jobs()]
            assert statuses.count(q.PENDING) == 2, (
                "prune() dropped a pending repair job — an open job is the only "
                f"record that a learner flagged that question (left: {statuses})"
            )

            first = q.enqueue(question_id=777, tag='broken', note='first',
                              correct=None, user_email='t@example.com')
            second = q.enqueue(question_id=777, tag='broken', note='second',
                               correct=None, user_email='t@example.com')
            open_777 = [j for j in q.load_jobs()
                        if j['question_id'] == 777 and j['status'] in q.OPEN_STATUSES]
            assert len(open_777) == 1 and open_777[0]['job_id'] == second['job_id'], (
                "re-flagging a question leaves two open jobs — two runners would "
                "each repair it and each overwrite the other's override"
            )
            assert first['job_id'] != second['job_id']

            # A claim is exclusive, or two runners repair the same question and
            # each silently overwrites the other's override. Read-then-write
            # would let both of these succeed.
            assert q.claim(second['job_id'], runner='a') is not None
            assert q.claim(second['job_id'], runner='b') is None, (
                "a second runner can claim a job that is already RUNNING — both "
                "would repair the same question and the later write wins silently"
            )

            # A flag arriving mid-repair must not delete the job under the
            # runner holding it; the runner closes by id and would 404.
            q.enqueue(question_id=777, tag='broken', note='third',
                      correct=None, user_email='t@example.com')
            assert q.get_job(second['job_id'])['status'] == q.RUNNING, (
                "re-flagging deleted the job a runner is mid-session on — that "
                "throws away a repair that was already paid for"
            )
        finally:
            if previous is None:
                os.environ.pop('DELTA_FEEDBACK_AI_DIR', None)
            else:
                os.environ['DELTA_FEEDBACK_AI_DIR'] = previous


def check_the_two_nudge_tables_agree():
    """The learner's step sizes exist TWICE, and drift between them is silent.

    `adaptive.DIFFICULTY_NUDGE` is what the live backend applies when a learner
    answers the post-submit question; `practice_engine.STAIRCASE_FEELING_BONUS`
    is the offline twin used to simulate the same staircase. They sit on
    opposite sides of a deployment boundary and cannot share an import, and
    nothing fails when one is edited alone — the offline model just starts
    recommending a schedule the live engine does not follow, which surfaces as
    a tuning result that will not reproduce.

    🔴 READ BY AST, NOT BY IMPORT. This watcher also runs under a bare
    interpreter with none of the backend's third-party deps installed;
    importing `app.adaptive` there raises ModuleNotFoundError and the check
    reads as a failure of the thing it is checking.
    """
    import ast

    def table(path, name):
        tree = ast.parse(open(path, encoding='utf-8').read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets
            ):
                return ast.literal_eval(node.value)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                    and node.target.id == name and node.value is not None:
                return ast.literal_eval(node.value)
        return None

    repo = os.path.abspath(os.path.join(THIS, '..', '..', '..', '..'))
    twin = os.path.join(repo, 'Local_Deployed_Shared', 'practice_engine.py')
    assert os.path.isfile(twin), f"the offline staircase twin is gone ({twin})"

    live = table(os.path.join(THIS, '..', 'adaptive.py'), 'DIFFICULTY_NUDGE')
    offline = table(twin, 'STAIRCASE_FEELING_BONUS')
    assert live, "adaptive.py no longer defines DIFFICULTY_NUDGE"
    assert offline, "practice_engine.py no longer defines STAIRCASE_FEELING_BONUS"
    assert live == offline, (
        f"the difficulty step-size tables have drifted.\n  live    {live}\n"
        f"  offline {offline}\nEdit both, or the offline runs describe a "
        "staircase the backend does not climb."
    )
    # All three answers now name a DIRECTION, so a zero is a button whose words
    # promise a move it never makes. "About right" was retired on 2026-08-28.
    assert all(live.get(k, 0) > 0 for k in ('not_much', 'somewhat', 'a_lot')), \
        f"a step size is missing or zero: {live}"
    assert live['not_much'] < live['somewhat'] < live['a_lot'], (
        f"the steps are not increasing: {live}. The buttons are ordered "
        "smallest-to-largest on screen and the learner is choosing a SIZE"
    )


# ── Run all checks ────────────────────────────
if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants,
              check_attempts_are_finalized, check_finalize_actually_moves_state,
              check_felt_difficulty_reaches_the_next_question,
              check_ai_repairs_are_gated,
              check_repair_runs_off_the_local_cli,
              check_repair_queue_never_loses_an_open_job,
              check_the_two_nudge_tables_agree]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
