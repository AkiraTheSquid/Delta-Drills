"""watch_feedback.py — the learner must always be able to say what is wrong.

Two checks, one for each surface. Both exist because the feature they guard was
NOT missing on 2026-08-27 — it was present, wired, and unreachable:
`#problem-feedback-row` was `display:none` under `body.dd-basic-mode`, which is
the default mode, so every learner saw a working feedback channel exactly never.
An assertion that the code exists would have passed the whole time; these assert
that it is REACHABLE.
"""
import os
import re

from watch_common import HERE, SHARED, read

_REPO = os.path.dirname(SHARED)
_BACKEND_ROUTER = os.path.join(
    _REPO, "This-Directory-Only", "backend", "app", "practice", "problem_feedback_router.py"
)
_BACKEND_WATCH = os.path.join(
    _REPO, "This-Directory-Only", "backend", "app", "practice", "watch.py"
)


def _slice(text, start_marker, end_marker):
    """The text between two markers, both of which must be present and in order."""
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def _css_rules(css):
    """(selector, body) for every top-level rule, selectors joined across lines.

    Not a parser — nested at-rules keep their inner braces in `body`, which is
    fine here: every caller asks what a selector matches and whether its body
    hides something.
    """
    out = []
    depth = 0
    sel = []
    body = []
    # Comments first: a block comment sitting above a rule would otherwise be
    # accumulated INTO that rule's selector, and every word in it would read as
    # part of the match.
    css = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
    for ch in css:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out.append((" ".join("".join(sel).split()), "".join(body)))
                sel, body = [], []
                continue
        (body if depth else sel).append(ch)
    return out


def _fn_body(js, name):
    """Everything from a declaration to the start of the next top-level one.

    Crude on purpose: the checks below only ask what a function does NOT
    mention, and over-reading makes that answer safer, never falser.
    """
    start = js.index(name)
    rest = js[start + len(name):]
    nxt = re.search(r"\n(?:  (?:const|function|// ──)|@router\.|def |class )", rest)
    return rest[: nxt.start()] if nxt else rest


def check_a_learner_can_always_report_a_broken_problem():
    index_html = read(os.path.join(SHARED, "index.html"))
    basic = read(os.path.join(SHARED, "styles", "practice", "basic-mode.css"))
    panel_js = read(os.path.join(HERE, "feedback-panel.js"))
    events = read(os.path.join(HERE, "events.js"))

    # 1. THE bug. Basic mode is the default, so anything it hides is hidden for
    #    everyone — and the report-a-problem channel is not a difficulty dial.
    # 🔴 Read RULES, not lines. The line-wise version passed a selector list
    #    broken across two lines, which is how anyone would write a long one —
    #    it would have missed the exact regression it was written for (codex,
    #    round 3).
    hidden_in_basic = [
        sel.strip() for sel, body in _css_rules(basic)
        if "dd-basic-mode" in sel and "problem-feedback" in sel
        and ("display: none" in body or "display:none" in body)
    ]
    assert not hidden_in_basic, (
        "basic-mode.css hides the problem-feedback surface again: "
        f"{hidden_in_basic} — basic mode is the DEFAULT, so this is 'no learner "
        "can report a broken problem', which is what this file exists to stop"
    )

    # 2. The way in sits beside Submit, where the learner already is.
    submit_area = _slice(index_html, 'id="practice-submit-area"', 'id="practice-feedback-area"')
    assert 'data-fb-target="problem"' in submit_area, (
        "no feedback trigger inside #practice-submit-area — the button has to "
        "be next to Submit or it is a feature nobody finds"
    )

    # 3. The panel expands at the BOTTOM of the left column, not inside the
    #    post-submit area: it has to be openable before a submit as well.
    panel_at = index_html.index('id="problem-feedback-panel"')
    assert panel_at > index_html.index('id="practice-feedback-area"'), (
        "#problem-feedback-panel moved back above #practice-feedback-area"
    )
    assert panel_at < index_html.index('class="practice-right"'), (
        "#problem-feedback-panel left .practice-left — it expands the LEFT "
        "column, and the right one is the notebook"
    )
    assert 'id="problem-feedback-send"' in index_html, (
        "the panel lost its Send button, so the note can only be sent by "
        "clicking a chip — which is the bug the panel replaced"
    )

    # 4. Choosing a kind is not sending. A chip that posts on click makes the
    #    note box decorative: whatever is typed after it is never transmitted.
    # Scoped to the CHIPS, not to the channel: #missed-fact-btn legitimately
    # posts on the same endpoint, and `dataset.flag` is the tag only a chip has.
    assert "dataset.flag" not in events, (
        "events.js reads a chip's tag again — the chips are a selection, and "
        "feedback-panel.js owns the one send"
    )
    assert "reportProblem" not in _fn_body(panel_js, "const _bindChips"), (
        "_bindChips sends on click again — selection and submission are "
        "separate acts, or the note under the chips is never read"
    )
    assert panel_js.count("PracticeAPI.reportProblem") == 1, (
        "more than one place in feedback-panel.js posts problem feedback"
    )

    # 5. Opening it has to SHOW it: the panel is at the bottom of a scroller
    #    and a freshly-unhidden element has no height for a frame, so the
    #    scroll is clamped unless it re-aims. Never scrollIntoView — that walks
    #    every scrollable ancestor and moves the page under the notebook too.
    assert "scrollIntoView" not in panel_js, (
        "feedback-panel.js uses scrollIntoView — it scrolls the document and "
        "the right pane along with the left column"
    )
    assert "_isRevealed(panel, pane)) return" in _fn_body(panel_js, "const _reveal"), (
        "_reveal no longer checks whether the panel is ALREADY on screen — it "
        "then runs its window fallback for a panel that needed nothing, which "
        "scrolls the document and drags the notebook pane with it"
    )
    assert "_reveal(panel, retries - 1)" in _fn_body(panel_js, "const _reveal"), (
        "_reveal no longer re-aims while the panel settles — it lands at the "
        "old scroll height and the panel stays below the fold"
    )
    assert "_reveal(problem.panel)" in _fn_body(panel_js, "const openProblem"), (
        "openProblem no longer scrolls the panel into view"
    )


def check_a_lesson_can_be_reported_without_touching_the_question():
    api = read(os.path.join(HERE, "api.js"))
    lessons = read(os.path.join(HERE, "lessons.js"))
    panel_css = read(os.path.join(SHARED, "styles", "practice", "feedback-panel.css"))
    index_html = read(os.path.join(SHARED, "index.html"))

    # 1. Its own endpoint. /problem-feedback takes an integer question id and
    #    queues an AI REWRITE OF THAT QUESTION for an actionable tag, so lesson
    #    prose routed through it would rewrite the drill the lesson gates.
    lesson_call = _fn_body(api, "async reportLesson")
    assert "/api/practice/lesson-feedback" in lesson_call, (
        "reportLesson no longer posts to /api/practice/lesson-feedback"
    )
    assert "problem-feedback" not in lesson_call, (
        "reportLesson posts to the problem channel — that files the note "
        "against a question id and can queue a rewrite of it"
    )

    # 2. The panel knows which concept it is about, and stops knowing when the
    #    lesson ends — the open class is a LAYOUT switch on .practice-left.
    assert "setLessonContext" in lessons, "lessons.js never names the lesson being reported"
    assert "closeLesson" in _fn_body(lessons, "const _cleanup"), (
        "_cleanup leaves the lesson feedback panel open, so the question "
        "screen inherits the two-column lesson layout"
    )

    # 3. The window opens to the LEFT of the lesson, with the lesson still
    #    readable on the right.
    assert re.search(
        r"^body\.lesson-mode\.dd-lesson-feedback-open \.practice-left \{[^}]*flex-direction: row",
        panel_css, re.M,
    ), (
        "feedback-panel.css lost the open-state two-column rule — the panel "
        "would stack above the lesson instead of docking beside it. 🔴 It is "
        "`flex-direction: row` that is pinned, not `display: flex`: layout.css "
        "already makes .practice-left a flex COLUMN, so restating display "
        "changes nothing — that exact rule shipped once and did nothing. And "
        "the selector alone is not enough either: the <=900px block repeats it "
        "to go back to a column."
    )
    assert "body.lesson-mode.dd-lesson-feedback-open #question-text" in panel_css, (
        "the lesson column has no open-state rule, so notebook.css keeps it "
        "centred and the docked panel overlaps or squeezes it"
    )

    # 4. One dock, holding the button and the panel, or opening it turns the
    #    reading column into a three-column layout.
    dock_at = index_html.index('id="lesson-feedback-dock"')
    assert dock_at < index_html.index('id="lesson-feedback-toggle"') < index_html.index(
        'id="lesson-feedback-panel"'
    ), "the lesson feedback button and panel are no longer inside #lesson-feedback-dock"
    assert dock_at < index_html.index('class="practice-right"'), (
        "#lesson-feedback-dock left .practice-left, which is the column the "
        "lesson is rendered into"
    )

    # 5. The backend half. Skipped where it does not ship: the deploy worktree
    #    holds Local_Deployed_Shared and nothing else.
    if os.path.isfile(_BACKEND_ROUTER):
        router = read(_BACKEND_ROUTER)
        assert '@router.post("/lesson-feedback"' in router, (
            "the backend lost POST /lesson-feedback; the frontend falls back "
            "to a localStorage queue, so this fails silently in the browser"
        )
        assert "enqueue_repair" not in _fn_body(router, "def submit_lesson_feedback"), (
            "lesson feedback queues an AI repair — the only thing it has a "
            "question id for is triage context, and repairing that question "
            "off a note about the lesson is the failure this split prevents"
        )
    if os.path.isfile(_BACKEND_WATCH):
        assert "'/api/practice/lesson-feedback'" in read(_BACKEND_WATCH), (
            "the backend route census does not list /lesson-feedback, so the "
            "route can be dropped without failing anything"
        )


def check_feedback_that_never_left_the_browser_is_not_called_logged():
    """An offline report must be queued AND drained AND described honestly.

    codex (/critic, 2026-08-27) found the trio: `reportProblem`/`reportLesson`
    fall back to a localStorage queue, return `{success: true}`, and the panel
    said "Thanks — logged ✓". Nothing ever read those queues back, so a guest's
    feedback was written to their own browser, declared delivered, and lost.
    """
    api = read(os.path.join(HERE, "api.js"))
    panel = read(os.path.join(HERE, "feedback-panel.js"))

    # 1. Both queues are DRAINED on the next report that reaches the server.
    for key, path in (
        ("problem_feedback_queue", "/api/practice/problem-feedback"),
        ("lesson_feedback_queue", "/api/practice/lesson-feedback"),
    ):
        assert '_flushFeedbackQueue("%s", "%s")' % (key, path) in api, (
            "nothing drains %s — feedback written offline stays in that one "
            "browser forever" % key
        )
    flush = _slice(api, "const _flushFeedbackQueue", "const FEEDBACK_QUEUES")
    assert "localStorage.setItem(key, JSON.stringify(remaining))" in flush, (
        "the flush never writes the queue back, so a delivered entry is re-sent forever"
    )
    # 🔴 The write-back must be a MERGE. Reading the queue once, awaiting N
    #    posts and then overwriting storage erases everything the learner
    #    queued during those posts (codex, round 3).
    assert "_readQueue(key)" in flush.split("delivered.length")[-1], (
        "the flush writes back the queue it read BEFORE the network round "
        "trips — anything queued during the drain is erased. Re-read and "
        "remove only what was delivered"
    )
    assert "_flushing.has(key)" in flush and "_flushing.delete(key)" in flush, (
        "the per-queue in-flight latch is gone, so two overlapping drains "
        "send the same entries twice"
    )

    # 1b. Signing in reloads the app, so a boot drain is what makes the
    #     panel's promise ('it sends when you are signed in') true. Without
    #     it a queued report only leaves once a SECOND report is filed.
    assert "flushFeedbackQueues" in api, (
        "nothing drains the queues on load — a learner who queues one report "
        "and signs in never sends it, which the status line promises they do"
    )

    # 1c. A queue write that FAILED is not a save. BOTH surfaces — asserting
    #     that the pattern exists somewhere in the file passes a tree where
    #     only one of them checks (it did; the mutation survived).
    for key in ("problem_feedback_queue", "lesson_feedback_queue"):
        fallback = _slice(
            api, 'const stored = _queueFeedback("%s"' % key, "},",
        )
        assert "stored ?" in fallback and "{ success: false }" in fallback, (
            "the %s fallback claims queuedLocally without checking that "
            "localStorage accepted the write — a full or disabled store then "
            "reads back to the learner as 'saved on this device'" % key
        )

    # 2. The panel distinguishes queued-here from logged-on-the-server.
    assert panel.count("result.queuedLocally") == 2, (
        "a send path stopped checking queuedLocally, so it claims 'logged ✓' "
        "for a report that never left the device"
    )

    # 3. Clearing a note tells DDAutoGrow, or the box keeps its grown height.
    assert '.note.value = ""' not in panel, (
        "a note is cleared by assigning .value, which fires no input event — "
        "route it through _clearNote"
    )
    assert 'note.dispatchEvent(new Event("input"' in panel, (
        "_clearNote no longer announces the change, so an autogrown textarea "
        "stays tall after it is emptied"
    )

    # 4. A lesson draft is keyed by the CONCEPT, not the displayed title.
    assert "lesson.context.title !== next.title" not in panel, (
        "walking between segments of one concept clears a half-written note "
        "again; key the draft on kc"
    )

    # 5. A completing send may not touch a panel that has moved on. The pool
    #    advances while the post is in flight; clearing then wipes what the
    #    learner has typed about the NEXT question (codex, round 3).
    assert "_currentQuestionId() !== subject" in panel, (
        "_sendProblem clears the panel without checking the question is still "
        "the one it reported — a slow send erases the next question's draft"
    )
    assert "_lessonKey(lesson.context) !== subject" in panel, (
        "_sendLesson clears the panel without checking the concept is still "
        "the one it reported"
    )
    assert panel.count("const _lessonKey") == 1, (
        "the lesson key is defined more than once — the draft-clearing rule "
        "and the in-flight-send rule have to agree on what one lesson is"
    )
