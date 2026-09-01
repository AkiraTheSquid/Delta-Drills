"""Placement/diagnostic health checks re-exported by practice/watch.py."""
import os
import re

from watch_common import HERE, SHARED, read


def check_every_placement_question_gets_the_same_clock():
    """One fixed allowance per probe, and every advance path kills it.

    The placement test runs OUTSIDE a practice session — starting it calls
    PracticeSession.finish("placement") — so none of the session timers apply
    and a probe had no limit at all until placement-timer.js. Three ways that
    silently regresses, so three assertions:

      * the allowance stops being ONE constant (per-question or
        difficulty-scaled time would make the probes incomparable, which is
        the whole point of a placement);
      * the module stops being loaded, or loads before the hooks that call it;
      * an advance path forgets to stop the clock, which is how a countdown
        expires onto the NEXT question and answers it for the learner.
    """
    timer = read(os.path.join(HERE, "placement-timer.js"))
    assert "const PLACEMENT_ANSWER_SECS = 120;" in timer, (
        "the placement allowance must stay one named constant — every probe "
        "gets the same time or the estimates are not comparable"
    )
    # No second source of truth: the only other number the clock may hold is
    # the resume floor/grace, never a per-question or per-difficulty value.
    assert "q().answer_secs" not in timer and "difficulty" not in timer, (
        "placement timing must not vary by question or difficulty"
    )
    # `PracticeAPI` is a top-level const in api.js — NOT a window property.
    # Reading it off window is undefined at runtime and silent at review time:
    # the clock simply never sees a probe. Same trap notebook-view.js documents.
    assert "window.PracticeAPI?.currentQuestion" not in timer, (
        "read PracticeAPI from script scope, not window — a top-level const is "
        "not a window property and the clock would never start"
    )
    for token in ("onQuestionRendered", "pauseForGrading", "resumeAfterFailedSubmit", "stop"):
        assert f"{token}," in timer or f"{token}:" in timer, (
            f"placement-timer.js no longer exports {token}"
        )

    index = read(os.path.join(SHARED, "index.html"))
    assert "practice/placement-timer.js" in index, "placement-timer.js is not loaded"
    # Match the SCRIPT TAG, not the bare name: both files are named in prose
    # comments elsewhere in the page, and a comment that happens to sit higher
    # would satisfy a plain substring search while the load order was wrong.
    assert index.find('src="practice/ui.js') < index.find('src="practice/placement-timer.js'), (
        "placement-timer.js must load after the modules whose hooks call it"
    )
    assert 'id="placement-timer"' in index, (
        "the countdown needs a static element: it is the anchor the info dot "
        "attaches to, and check_infotips fails without it"
    )

    ui = read(os.path.join(HERE, "ui.js"))
    assert "PlacementTimer?.onQuestionRendered()" in ui, (
        "no probe starts a clock — renderQuestion must arm the placement timer"
    )
    events = read(os.path.join(HERE, "events.js"))
    assert "PlacementTimer?.stop()" in events, (
        "_loadNextPracticeQuestion must kill the placement clock, or an expiry "
        "at 00:01 lands on the question that just loaded"
    )
    assert "PlacementTimer?.pauseForGrading()" in events, (
        "submitting must stop the placement clock while the grade is in flight"
    )
    # Pyodide cannot import torch, and the einops questions in the bank ARE
    # torch questions — routing them local made every Submit unanswerable.
    api = read(os.path.join(HERE, "api.js"))
    assert "!needsTorchRuntime(this.currentQuestion, userCode)" in api, (
        "einops questions that touch torch must grade on the backend — local "
        "Pyodide refuses them and the learner gets no verdict at all"
    )
    # A question that CANNOT run here must not re-arm a countdown: expiry
    # force-submits, the submit is refused again, and the clock loops at 00:30.
    timer = read(os.path.join(HERE, "timer.js"))
    assert "blockOnUnrunnableQuestion" in timer and "blockOnUnrunnableQuestion" in events, (
        "a blocked submit must stop the session clock, not resume it"
    )
    # A probe is timed by the placement's rule even inside a learner's session.
    assert "PlacementTimer.secondsPerQuestion()" in timer, (
        "a placement probe inside a session must use the placement allowance, "
        "not whatever answer time that session was set up with"
    )
    # One writer for the start button: two copies of the label is how it
    # flickers between two names when the page refreshes its status.
    page = read(os.path.join(HERE, "diagnostic-page.js"))
    assert "renderStartButton" in page and "renderStartButton" in events, (
        "events.js must delegate the placement start button to diagnostic-page.js"
    )
    assert "Take the placement test" not in events, (
        "the start-button label lives in diagnostic-page.js only"
    )


def check_a_taken_placement_leaves_the_learner_home():
    """The placement lives on its own page, and the account menu is the way to it.

    Seth, 2026-08-31: "don't put it on the learner home tab ... remove it from
    there" — and, in the same breath, "keep the diagnostic in the drop-down ...
    that way you can always retake the placement diagnostic". Then 2026-09-01:
    "we can probably just keep the interface for the diagnostic ... separate,
    and then it only gets displayed whenever you click on the drop-down one and
    you go to it specifically".

    The second ask subsumes the first and is enforced structurally instead of by
    a visibility class: the card is not on the Learner Home in ANY state, so
    there is no longer a state in which it can come back. What still has to hold
    as a pair is the removal and the ROUTE — a card nothing points at is a
    feature that has been deleted, whatever the markup says.

    🔴 THE OLD RULE IS ASSERTED GONE, not just replaced. `.placement-taken` hid
    the card on `!!status?.completed_at && !status.active` and `forceShow` lifted
    that for one visit. Both would now be actively wrong: they would hide, on
    arrival, the only thing on a page the learner navigated to on purpose.
    """
    page = read(os.path.join(HERE, "diagnostic-page.js"))
    css = read(os.path.join(SHARED, "styles", "practice", "diagnostic.css"))
    menu_js = read(os.path.join(SHARED, "account-menu.js"))
    index = read(os.path.join(SHARED, "index.html"))

    home = index.split('id="page-practice"')[1].split("\n  </main>")[0]
    assert 'id="diagnostic-overview"' not in home, (
        "the placement card is back on the Learner Home. Seth, 2026-09-01: it "
        "is its own page and shows only when you go to it specifically"
    )
    # 🔴 THE PAGE ALWAYS SHOWS ITS CARD. The visibility machinery is gone, and a
    # regrown copy of it would hide the page's only content on arrival.
    # Matched on the RULE and on the WRITE, not on the word: both files explain
    # at length what .placement-taken was and why it went, and an assertion that
    # a name is gone must not be tripped by the note explaining it.
    assert ".placement-taken " not in css, (
        "the .placement-taken hide is back in the stylesheet. It belonged to a "
        "card parked on the daily screen; on a page reached deliberately it is "
        "a dead end"
    )
    assert 'toggle("placement-taken"' not in page, (
        "diagnostic-page.js is writing .placement-taken again — the card it "
        "hides is the only content of the page it is on"
    )
    # The ASSIGNMENT, for the same reason: the file explains what forceShow was.
    assert "forceShow =" not in page, (
        "the reveal override is back. It existed only to lift .placement-taken "
        "for one visit, and there is nothing left to lift"
    )
    # The route. The row exists, it points at the page, and the page has a way
    # back out — a page with no tab in the strip that a learner can be dropped
    # on by the welcome fork MUST have a door.
    assert "data-placement-retake" in index, (
        "the account menu lost the placement row — with no tab in the strip, "
        "that row is the only standing route to the placement page"
    )
    assert 'data-goto-tab="placement"' in index, (
        "nothing routes to #page-placement any more; the page is reachable "
        "only by typing /diagnostic"
    )
    assert 'id="placement-skip-btn"' in index, (
        "the placement page lost its way out. The welcome fork's right arm "
        "lands a first-time learner here, and a learner who does not want the "
        "test would be stranded on it"
    )
    # account-menu.js may no longer call reveal() — the export is gone — but it
    # must still do the part a bare jump cannot: point at the button that acts.
    assert "placement-cta-flash" in menu_js, (
        "the placement row stopped flashing the button it routes to, so the "
        "row is a bare page switch onto a card with two controls on it"
    )
    assert "reveal" not in page.split("return {")[-1], (
        "DiagnosticPage is exporting reveal() again — there is no visibility "
        "override left for it to set, so it can only mislead its callers"
    )


def check_the_placement_result_is_the_number_the_backend_seeded():
    """The results card must report the placement, and say what it doesn't know.

    Three separate failures live in this one check, all of them from
    2026-08-23.

    1. THE READINESS FIGURE IS A COPY. `placement-results.js` turns the
       backend's theta into a percentage with the same affine map that
       `diagnostic.py::_mastery_from_theta` uses to SEED per-atom BKT mastery
       at finish(). If those four constants drift apart, the card tells the
       learner a readiness the rest of the app does not act on — the quiet
       kind of wrong, because both halves keep working. Same reasoning as
       check_promotion_threshold_matches_the_backend, same remedy.

    2. `.primary` MUST NOT BE ON THE CTA. `.primary` sets `width: 100%` and
       12px/28px padding; `.placement-start-btn` overrode the padding and not
       the width, so the button rendered as a card-wide 4px-tall strip. It is
       a two-class collision, so nothing in either rule looks wrong on its
       own — this is the only place it can be caught.

    3. AN UNPROBED AREA IS NOT A MEASUREMENT. The backend returns a theta for
       every area whether or not the test ever probed it; the unprobed ones
       are the prior, propagated. Rendering those as bare percentages invents
       confidence the test never earned, and the learner plans around it. The
       renderer must keep the label and the dimmed style that separate them.
    """
    results_js = read(os.path.join(HERE, "placement-results.js"))
    index = read(os.path.join(SHARED, "index.html"))
    page = read(os.path.join(HERE, "diagnostic-page.js"))
    css = read(os.path.join(SHARED, "styles", "practice", "diagnostic.css"))

    # 1. the readiness map is a copy of the backend's seeding map
    diag = os.path.join(
        HERE, "..", "..", "This-Directory-Only", "backend", "app", "diagnostic.py")
    if os.path.exists(diag):
        backend = read(diag)
        for js_name, py_name in (("DIFF_FLOOR", "_DIFF_FLOOR"),
                                 ("DIFF_SPAN", "_DIFF_SPAN"),
                                 ("SEED_MASTERY_FLOOR", "SEED_MASTERY_FLOOR"),
                                 ("SEED_MASTERY_CAP", "SEED_MASTERY_CAP")):
            m_js = re.search(rf"^\s*const {js_name} = ([0-9.]+)\s*;", results_js, re.M)
            m_py = re.search(rf"^{py_name} = ([0-9.]+)", backend, re.M)
            assert m_js, f"placement-results.js lost its {js_name} constant"
            assert m_py, f"diagnostic.py lost {py_name}"
            assert float(m_js.group(1)) == float(m_py.group(1)), (
                f"readiness map drifted: placement-results.js {js_name}="
                f"{m_js.group(1)} but diagnostic.py {py_name}={m_py.group(1)} — "
                "the card would report a readiness the seeding does not use"
            )

    # 2. neither placement button may carry .primary, and the CTA keeps its
    #    wrapper (without it the infotip dot is a sibling flex item and drops
    #    onto its own line under the button).
    for btn_id in ("placement-start-btn", "diagnostic-practice-btn"):
        tag = re.search(rf'<button[^>]*id="{btn_id}"[^>]*>', index)
        assert tag, f"index.html lost #{btn_id}"
        classes = re.search(r'class="([^"]*)"', tag.group(0))
        assert classes and "primary" not in classes.group(1).split(), (
            f"#{btn_id} carries .primary again — width:100% plus the placement "
            "button's own padding is the full-bleed 4px-tall strip"
        )
    assert re.search(
        r'<span class="placement-cta">\s*<button[^>]*id="placement-start-btn"', index), (
        "the placement CTA lost its .placement-cta wrapper — infotips.js inserts "
        "the dot into the anchor's parentNode, so unwrapped it orphans below the button"
    )
    assert ".placement-cta" in css, "styles/practice/diagnostic.css lost .placement-cta"
    assert ".placement-cta.hidden" in css and "el.parentElement?.classList.toggle" in page, (
        "the CTA wrapper must hide WITH its button, and .hidden must be re-asserted "
        "at this file's specificity — diagnostic.css loads after components.css, so "
        "the tie goes to .placement-cta and an empty flex item keeps its gap"
    )

    # 3. one writer for the card body, and it must be reached
    for anchor in ("placement-results-meta", "placement-readiness",
                   "placement-areas", "placement-results-empty"):
        assert f'id="{anchor}"' in index, f"index.html lost the #{anchor} results anchor"
    assert "window.PlacementResults?.render(status)" in page, (
        "diagnostic-page.js must hand the status to placement-results.js — "
        "without it the card is empty for someone who finished the test"
    )
    assert "!!status.completed_at && !status.active" in page, (
        "the results card must render only for a FINISHED placement; a mid-test "
        "status carries live estimates that would read as a final result"
    )
    # 🔴 SCOPED TO THE RESULTS CARD, not to the whole file. This read
    # `"will appear here" not in index` until 2026-08-24, when e4a65d2b gave the
    # Account tab's instructor-mode hint the sentence "Review tools will appear
    # here as they are built" — copy in another tab, about another feature, that
    # turned this into a standing false failure for every session in the repo.
    # The invariant was always about ONE element: #diagnostic-results, which
    # placement-results.js fills from the status payload. A placeholder anywhere
    # else is not this bug, and a substring test over 1,700 lines of markup will
    # keep colliding with unrelated copy.
    #
    # 🔴 `\sid=` AND THE NESTING ASSERT ARE BOTH LOAD-BEARING (codex, 2026-08-24).
    # `[^>]*id="` also matches `aria-labelledby="diagnostic-results"`, which would
    # scope this to the wrong element; and a non-greedy `.*?</section>` stops at
    # the FIRST close tag, so nesting a <section> in the card would silently slice
    # off the part the placeholder is most likely to be in. A check that quietly
    # covers less than it claims is worse than one that fails — hence the assert
    # rather than a balanced-scan parser.
    start = re.search(r'<section\b[^>]*\sid="diagnostic-results"[^>]*>', index)
    assert start, "index.html lost the #diagnostic-results card"
    rest = index[start.end():]
    end = rest.find("</section>")
    assert end != -1, "#diagnostic-results is never closed"
    results_card = rest[:end]
    assert "<section" not in results_card, (
        "#diagnostic-results now nests a <section>, so this slice stops at the "
        "inner close tag and the assertion below would cover only part of the card"
    )
    assert "will appear here" not in results_card, (
        "the results placeholder is back — it showed to learners who had already "
        "finished the test, while the real numbers sat unrendered in the payload"
    )
    assert "placement-areas" not in page, (
        "placement-results.js is the only writer of the card body; a second "
        "writer is how the two halves disagree about the same placement"
    )
    assert index.find('src="practice/placement-results.js') < index.find(
        'src="practice/diagnostic-page.js'), (
        "placement-results.js must load BEFORE diagnostic-page.js, which calls it "
        "from a refresh it starts at load"
    )

    # 4. the honesty rule: an unprobed area is visibly not a measurement
    assert "not probed" in results_js and "placement-area--unprobed" in results_js, (
        "an area with zero probes must be labelled and dimmed — its theta is the "
        "prior propagated, and printing it bare invents confidence the test never earned"
    )
    assert ".placement-area--unprobed" in css, (
        "diagnostic.css lost the unprobed styling, so a prior renders identically "
        "to a measured area"
    )


