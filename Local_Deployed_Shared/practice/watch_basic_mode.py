"""watch_basic_mode.py — health checks for Basic mode on the practice tab

Split out of watch.py, which is RED on LOC: this folder's answer to that has
been a sibling module per subject since watch_invariants / watch_lessons /
watch_notebook. 🔴 Every `check_*` defined here is scanned by watch.py's
`_every_check_is_registered`, so adding one WITHOUT adding it to watch.py's
runner list fails the suite rather than silently not running.
"""
import os
import re

from watch_common import HERE, SHARED, read

def check_a_hidden_rating_still_commits_the_attempt():
    """🔴 2026-08-28: BASIC MODE NO LONGER HIDES THE RATING, so most of this
    check is dormant — it returns early at `hides_rating`. Kept whole, and not
    deleted, because it is the contract that has to come back INTACT if the
    hiding ever does: the stand-in click, its two call sites, the load order
    and the ⓘ. Seth asked for the three choices to be the post-submit
    interface itself (docked to the bottom of the viewport, "how much harder /
    easier do you want the next problem to be?", one click to rate AND
    advance), which makes the default mode exactly the mode that must show
    them. See practice/difficulty-dock.js and the header of
    styles/practice/basic-mode.css.

    What the dormant half asserted, and why:

    Basic mode hid the felt-difficulty rating. That row is not decoration:
    `POST /api/practice/feedback` is the ONLY backend mutation that moves a
    subtopic baseline (submit and override write `pending_attempt` and stop),
    and `#next-problem-btn` is revealed inside that button's own click handler.

    Hide it with nothing standing in for it and the failure is silent in both
    directions at once — the student model stops moving while the learner keeps
    answering, and the only way to the next question is waiting out the 02:00
    review clock. This check fails the run if the three pieces stop agreeing.
    """
    css_path = os.path.join(SHARED, "styles", "practice", "basic-mode.css")
    assert os.path.isfile(css_path), "styles/practice/basic-mode.css is gone"
    css = read(css_path)
    index = read(os.path.join(SHARED, "index.html"))
    basic = read(os.path.join(HERE, "basic-mode.js"))
    events = read(os.path.join(HERE, "events.js"))

    hides_rating = "body.dd-basic-mode .feedback-btn" in css

    # 1. The sheet has to be linked, or none of it is true on screen.
    assert 'href="styles/practice/basic-mode.css' in index, (
        "index.html no longer links styles/practice/basic-mode.css — the practice "
        "screen is back to showing every rail to every learner"
    )
    # 2. It rides on the toggle that already existed, not on a second flag.
    assert "body.dd-basic-mode" in css, (
        "basic-mode.css stopped keying on body.dd-basic-mode, the class app.js "
        "writes from the Account tab's Advanced mode checkbox — the Advanced "
        "toggle no longer reaches the practice screen"
    )
    # 3. Next problem must survive the cull. It is a SIBLING of the three
    #    rating buttons inside .feedback-buttons and carries no .feedback-btn
    #    class; hiding the row instead of the buttons takes it with them.
    assert "body.dd-basic-mode .feedback-buttons" not in css, (
        "basic-mode.css hides .feedback-buttons wholesale, which hides "
        "#next-problem-btn with it — there is now no way off a graded question "
        "but the review clock running out"
    )
    if not hides_rating:
        return  # the rating is visible again; the stand-in below is moot

    # 3b. Hiding a data-dd-info host orphans its ⓘ: infotips.js mints the dot
    #     as a SIBLING, so it survives its subject and reads as a stray "i".
    assert '.dd-info[data-dd-info="feedback-rating"]' in css, (
        "basic-mode.css hides #feedback-prompt but not the ⓘ infotips.js mints "
        "beside it — a bare dot is left under the result badge explaining a "
        "rating that is no longer on screen"
    )

    # 3c. The Colab edition keeps the two things this sheet would take: the
    #     reference solution (its whole review step) and the rating row it
    #     restyles into a stacked column. Both halves must state the SAME
    #     guard — a JS predicate true where the CSS is false auto-rates a
    #     row the learner can still see.
    assert "html:not(.dd-colab-edition)" in css, (
        "basic-mode.css no longer excludes the Colab edition. Its rules "
        "out-specify colab-edition.css, which only sets border and padding on "
        ".solution-section — so Basic mode deletes the reference solution that "
        "deploy's review step is built around"
    )
    rules = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    assert rules.count("body.dd-basic-mode") == rules.count(
        "html:not(.dd-colab-edition) body.dd-basic-mode"), (
        "a body.dd-basic-mode rule in basic-mode.css is missing the "
        "html:not(.dd-colab-edition) guard — it will fire on the Colab edition"
    )
    assert 'classList.contains("dd-colab-edition")' in basic, (
        "practice/basic-mode.js dropped the Colab guard from active(), so "
        "settleRating fires on a deploy where the rating buttons are still "
        "visible — it answers the question the learner is being asked"
    )

    # 4. Something has to send the rating the learner can no longer send.
    assert "settleRating" in basic and ".feedback-btn--default" in basic, (
        "practice/basic-mode.js no longer settles the hidden rating by clicking "
        "the real default button — in basic mode nothing commits the attempt"
    )
    assert "PracticeAPI.sendFeedback" not in basic, (
        "practice/basic-mode.js is calling sendFeedback directly. It must click "
        "the button: the handler in events.js also records the completed "
        "question, animates the target difficulty, writes the concept "
        "understanding, pushes the ladder estimate and parks pendingFeedback "
        "for resume. A second copy of that list is how the two drift"
    )
    # 5. ...and it has to actually be reached from the graded paths.
    assert events.count("PracticeBasicMode?.settleRating()") >= 2, (
        "events.js no longer settles the rating on both graded paths (the "
        "Submit handler and the Colab verdict with a pending attempt) — one of "
        "them now dead-ends in basic mode"
    )
    assert index.find('src="practice/basic-mode.js') < index.find(
        'src="practice/events.js'), (
        "basic-mode.js must load before events.js, which calls into it"
    )
    # 6. The refusals that keep it from firing twice, or on a path that already
    #    sent its rating (a resumed review, a placement probe).
    for guard in ("next-problem-btn", "practice-feedback-area", "def.disabled"):
        assert guard in basic, (
            f"basic-mode.js dropped its {guard} guard — settleRating can now fire "
            "on a question whose rating was already sent, logging a second "
            "attempt against the same problem"
        )



def check_the_difficulty_question_is_one_row_docked_to_the_bottom():
    """The post-submit interface is three buttons, docked, and they advance.

    Seth, 2026-08-28: after Submit the button is replaced by a bar frozen to
    the bottom of the viewport asking how much harder (correct) or easier
    (miss) the next problem should be, and clicking one is the Next button.
    Four ways that regresses without anyone noticing, so four assertions.

    1. The dock is MOVED markup, never a second copy. `difficulty-dock.js`
       re-parents `#feedback-prompt`, `.feedback-buttons`, and `#override-row`, so the buttons
       on screen are the ones `events.js` bound and `ui.js` relabels. A dock
       that minted its own `data-feedback` buttons and forwarded the clicks
       would be a second copy of a list this folder has already watched drift
       once (see `settleRating` in practice/README.md), and the copy would go
       stale the first time the labels changed.

    2. Its two files are actually loaded. A dock that never builds leaves the
       rating where it was — visible, but back in the left rail and no longer
       the obvious thing on screen, which is the whole of the request.

    3. Rating advances. `nextProblemBtn.click()` is in the feedback handler
       and it is a synthetic click on the REAL button, not a direct call to
       the loader, because that handler is also where ArenaUnlock gets its
       chance to show the unlock interstitial.

    4. Nothing answers the question on the learner's behalf while it is
       visible. Basic mode's stand-in click is gone; if it ever comes back
       while the buttons are on screen, it both invents an opinion and — now
       that the same click advances — skips the problem before the learner
       has answered.
    """
    dock_js = os.path.join(HERE, "difficulty-dock.js")
    dock_css = os.path.join(SHARED, "styles", "practice", "difficulty-dock.css")
    assert os.path.isfile(dock_js), "practice/difficulty-dock.js is gone"
    assert os.path.isfile(dock_css), "styles/practice/difficulty-dock.css is gone"

    dock = read(dock_js)
    css = read(os.path.join(SHARED, "styles", "practice", "basic-mode.css"))
    index = read(os.path.join(SHARED, "index.html"))
    events = read(os.path.join(HERE, "events.js"))
    basic = read(os.path.join(HERE, "basic-mode.js"))

    # 1. Moved, not copied.
    assert "data-feedback" not in dock, (
        "difficulty-dock.js is minting its own rating buttons. It must MOVE "
        "#feedback-prompt and .feedback-buttons into the dock — the handler in "
        "events.js and the labels in ui.js::applyResult are bound to those "
        "exact nodes, and a forwarded copy drifts from them silently"
    )
    for node in ("feedback-prompt", ".feedback-buttons", "override-row"):
        assert node in dock, (
            f"difficulty-dock.js no longer re-parents {node} — the dock is "
            "empty and the rating is back in the left rail"
        )

    # 1a. On an incorrect grade, the correctness override is a fourth dock
    #     option. It must use the existing handler: that POSTs the corrected
    #     verdict, hides the override, and repaints the remaining choices as
    #     harder without advancing before the learner chooses a step size.
    assert "buttons.appendChild(override)" in dock, (
        "#override-row is looked up but not moved into .feedback-buttons — "
        "'I actually got it right' remains hidden back in the left rail"
    )
    assert "I actually got it right" in index, (
        "the restored dock option lost its explicit correctness wording"
    )
    override_start = events.find('overrideCorrectBtn.addEventListener("click"')
    assert override_start != -1, "the correctness override click handler is gone"
    override_handler = events[override_start:events.find("feedbackButtons.forEach", override_start)]
    for behavior in ("PracticeAPI.overrideCorrect", "paintDifficultyQuestion(true)", "overrideRow.classList.add"):
        assert behavior in override_handler, (
            f"the correctness override no longer performs {behavior}; clicking "
            "it must flip the grade, remove itself, and turn easier choices into harder ones"
        )

    # 1a. The question and its ⓘ share one row. `#feedback-prompt` is a block
    #     and infotips' icon is inline-flex, so as bare siblings the icon wraps
    #     onto its own line and renders as a stray "i" between the question and
    #     the answers — found on the first build, in the browser.
    assert "difficulty-dock-question" in dock and "difficulty-dock-question" in read(dock_css), (
        "the dock's question row is gone. #feedback-prompt and the ⓘ that "
        "infotips mints next to it must share one flex row, or the icon wraps "
        "below the question as a stray \"i\""
    )

    # 1a2. ONE OWNER OF THE ADVANCE. The rating click is the navigation now, so
    #      `timer.js::_forceAdvance`'s watchdog must not click Next as well. It
    #      used to — before the dock, the feedback handler only REVEALED the
    #      button and the review-clock timeout polled for it and clicked. Both
    #      clicking is a double advance in a wide window: Next stays visible
    #      from the handler's click until the next question renders, a network
    #      fetch away, and the watchdog ticks every 250ms. The learner silently
    #      loses a question and ArenaUnlock is asked twice.
    timer = read(os.path.join(HERE, "timer.js"))
    poll_start = timer.find("advancePoll = setInterval(")
    assert poll_start != -1, "timer.js no longer has the _forceAdvance watchdog"
    poll = timer[poll_start:timer.find("}, 250);", poll_start)]
    assert "nextProblemBtn.click()" not in poll, (
        "timer.js's _forceAdvance watchdog clicks #next-problem-btn again. The "
        "feedback handler in events.js already clicks it, in the same "
        "synchronous run as showNextProblemButton() — two owners of one "
        "advance skips a question. The watchdog may only act when Next is "
        "still HIDDEN, which is the rating-POST-failed case"
    )

    # 1b. It closes when the practice tab does. The dock hangs off <body> so
    #     that it can sit over the viewport, which also means leaving the tab
    #     does not take it away the way it takes the rest of the practice UI.
    assert 'getElementById("page-practice")' in dock, (
        "difficulty-dock.js no longer checks whether the practice page itself "
        "is visible — the dock hangs off <body>, so a difficulty question "
        "stays pinned to the bottom of the window on Account, Concepts and "
        "every other tab"
    )

    # 2. Both halves linked.
    assert 'src="practice/difficulty-dock.js' in index, (
        "index.html no longer loads practice/difficulty-dock.js — the rating "
        "never leaves the rail and the bottom dock does not exist"
    )
    assert 'href="styles/practice/difficulty-dock.css' in index, (
        "index.html no longer links styles/practice/difficulty-dock.css — the "
        "dock builds but is not fixed to the bottom of the viewport"
    )

    # 3. One click rates AND advances. 🔴 Matched against the handler with its
    #    comments STRIPPED: the block above the call explains it at length and
    #    names it, so a plain substring search passes on a commented-out call —
    #    which is exactly how this regresses.
    rate_start = events.find("feedbackButtons.forEach((btn) => {")
    assert rate_start != -1, "events.js no longer binds the rating buttons"
    handler = events[rate_start:]
    live = "\n".join(
        ln for ln in handler.splitlines()
        if not ln.lstrip().startswith(("//", "*", "/*"))
    )
    assert "nextProblemBtn.click()" in live, (
        "events.js no longer advances on the rating click — the learner is "
        "back to answering the difficulty question and then pressing Next"
    )

    # 3b. ...but it gives the topbar its half second first. Measured
    #     2026-08-28 with the bar sampled every 40ms: the next question
    #     rendered 43ms after the click and published its own reading at 35ms,
    #     so the progress the learner had just earned was on screen for about
    #     one frame and never animated. Seth: "I didn't see the top bar show
    #     the update for the progress like it usually does with the
    #     animation." An unwaited click is that bug, exactly, again.
    deferred = re.search(
        r"setTimeout\(\s*\(\)\s*=>\s*\{(.*?)\}\s*,\s*([A-Za-z_0-9]+)\s*\)", live, re.S
    )
    assert deferred and "nextProblemBtn.click()" in deferred.group(1), (
        "the rating's advance must be deferred (setTimeout) so the topbar's "
        "0.55s fill can run. Called straight through, the next question "
        "overwrites the reading within a frame and the click shows nothing"
    )
    assert deferred.group(2) == "TOPBAR_SETTLE_MS" and "const TOPBAR_SETTLE_MS" in events, (
        "the advance delay must be the named TOPBAR_SETTLE_MS constant, "
        "declared in events.js beside the note on what it is paired with — a "
        "bare number here drifts away from the transition it exists to cover"
    )
    # 🔴 THE FLOOR IS READ OUT OF THE STYLESHEET, not written here as a second
    #    copy of it. The delay and the transition it covers live in different
    #    files and neither can see the other at runtime (a computed-style read
    #    of a transition that has not started returns the wrong element's
    #    duration), so the only place they can be held together is a check.
    #    codex, 2026-08-28, raised the duplication; this is the answer to it —
    #    shortening the wait or lengthening the fill now fails here instead of
    #    silently cutting the bar off mid-slide again.
    pill_css = read(os.path.join(SHARED, "styles", "concept-pill.css"))
    fill_ms = re.search(r"transition:\s*width\s+([\d.]+)s", pill_css)
    assert fill_ms, (
        "styles/concept-pill.css no longer states the fill's transition "
        "duration as `transition: width <n>s` — the settle delay below has "
        "nothing to be checked against"
    )
    need_ms = float(fill_ms.group(1)) * 1000
    settle = re.search(r"const TOPBAR_SETTLE_MS\s*=\s*(\d+)", events)
    assert settle and int(settle.group(1)) >= need_ms, (
        "TOPBAR_SETTLE_MS (%s) is shorter than the concept pill's own fill "
        "(%dms, styles/concept-pill.css) — the bar is cut off mid-slide, "
        "which is the state this was written to fix"
        % (settle.group(1) if settle else "missing", need_ms)
    )
    assert "PracticeAPI.currentQuestion !== q" in deferred.group(1), (
        "the deferred advance must re-check the question before firing. "
        "TOPBAR_SETTLE_MS is long enough for the learner to skip, end the "
        "session or switch tabs, and a click landing after that navigates "
        "away from whatever replaced it"
    )

    # 3c. The post-attempt ladder reading belongs to the RATING, not to
    #     submit. The backend does not score the attempt until /feedback
    #     (finalize_attempt), and painting at submit spends the one update the
    #     rating click had to show — the concept pill measured 33.33% before
    #     the click and 33.33% for the whole 2.2s after it.
    submit_paint = re.search(
        r"if \(result\.ladder_estimate && window\.StageLadder([^)]*)\)", events
    )
    assert submit_paint and "!" in submit_paint.group(1), (
        "the submit handler paints result.ladder_estimate unconditionally "
        "again. It must be guarded on there being no rating step to follow "
        "(placement probes only) or the difficulty answer has nothing left "
        "to show the learner"
    )

    # 4. Nobody answers it for them while it is visible.
    hides_rating = "body.dd-basic-mode .feedback-btn" in css
    if not hides_rating:
        assert "settleRating" not in events, (
            "the rating buttons are visible to everyone, but events.js still "
            "calls settleRating — that answers the question the learner is "
            "being asked and, because the same click advances, skips the "
            "problem out from under them"
        )
        assert ".feedback-btn--default\").click()" not in basic.replace(" ", ""), (
            "basic-mode.js is clicking the default rating button while the "
            "rating is on screen for everyone"
        )
