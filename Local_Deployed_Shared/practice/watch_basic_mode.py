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
    """Basic mode hides the felt-difficulty rating. That row is not decoration:
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

