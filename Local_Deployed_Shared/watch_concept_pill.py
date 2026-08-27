"""watch_concept_pill.py — the topbar concept pill, and the timer notch.

Same contract as every check in watch.py: raise AssertionError to fail. Kept in
its own file the way watch_instructor_graph.py and watch_front_door.py are —
watch.py is already past Modulario's line. watch.py must both IMPORT
`check_concept_pill` and list it in __main__: a runner list has dropped defined
checks SILENTLY twice in this repo, so the import and the list entry are each
load-bearing on their own.

What this guards, 2026-08-27. Seth moved two things at once:

  - the session clock went back to being a NOTCH, this time hanging off the
    TOPBAR's bottom edge rather than off `.practice-container`;
  - the concept under test moved OUT of the left panel's heading card and ONTO
    the topbar as a chip that fills, dropping the four-rung stage ladder.

Every way either of those breaks is quiet:

  - `#practice-notch` is absolutely positioned against `.topbar` only because
    nothing between the two is positioned. Anyone adding `position: relative`
    to `.topbar-mid` re-anchors it and the clock snaps back INTO the bar — a
    layout that looks intentional and throws nothing;
  - the pill draws a number it does not compute. A second implementation of the
    ladder arithmetic up here would be a second answer to "how far in am I",
    which is the exact thing the ladder's header is a long note about;
  - `pct: null` means NO READING. Collapsing it to 0 reports no progress, which
    is a claim about the learner, and it renders identically to a real 0;
  - the chip and the heading card are complementary at one breakpoint. Move
    either edge and the concept is on screen twice, or nowhere;
  - concept-pill.css hides `.question-number-row` at the SAME specificity
    question.css declares it, so the <link> order is the whole rule. Reorder
    the two tags and the name is drawn twice, in two type sizes;
  - the three index.html tags are what load the feature at all. Two of three
    loads a stylesheet with no behaviour, or an engine with no chip to write.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(*parts):
    with open(os.path.join(HERE, *parts), "r", encoding="utf-8") as f:
        return f.read()


def _code(js):
    """`js` with its comments removed.

    The banned-token scan below is about what the file DOES, and these files
    are 40% prose by design — concept-pill.js's own header names the Wilson
    bound and the promotion streak while explaining that it must never compute
    them. Scanning the raw text makes the guard fail on the sentence that
    documents the rule it is enforcing, which is the fastest way to get a check
    deleted. Crude on purpose: no string-literal awareness, because nothing in
    these files puts `/*` inside a string.
    """
    js = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    return re.sub(r"^\s*//.*$", "", js, flags=re.M)


def _rule(css, selector):
    r"""The declaration block for `selector`, WITHOUT its comments, or None.

    Deliberately crude — these sheets are hand-authored and one selector per
    rule is the house style. `\s*\{` after the escaped selector is what stops
    `.dd-concept` returning `.dd-concept-fill`'s body.

    🔴 THE COMMENTS COME OUT FIRST, and that is not tidiness. Every rule in
    these sheets carries a paragraph explaining itself, and those paragraphs
    NAME the declarations they are about — `.topbar`'s note says the word
    "position: relative" while explaining what would break if anyone added it.
    A scan over the raw block is then satisfied by the prose warning against
    the very thing it is checking for: deleting `position: sticky` from
    `.topbar` left this check passing until the comment was stripped. Every
    substring assertion below depends on this.
    """
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    m = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    return m.group(1) if m else None


def _parent_of(html, element_id):
    """(tag, class) of the element that CONTAINS `#element_id`, or None.

    Parsed with the stdlib parser rather than matched with a regex: the
    question this answers — "is the notch a direct child of the header" — is
    about the tree, and every regex that looks like it answers that is really
    answering "do these two strings appear in this order".
    """
    from html.parser import HTMLParser

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
            "meta", "param", "source", "track", "wbr"}

    class _Finder(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.stack = []
            self.found = None

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if attrs.get("id") == element_id and self.found is None:
                self.found = self.stack[-1] if self.stack else None
            if tag not in VOID and not tag.startswith("!"):
                self.stack.append((tag, attrs.get("class")))

        def handle_startendtag(self, tag, attrs):
            self.handle_starttag(tag, attrs)
            if self.stack and self.stack[-1][0] == tag:
                self.stack.pop()

        def handle_endtag(self, tag):
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i][0] == tag:
                    del self.stack[i:]
                    return

    f = _Finder()
    f.feed(html)
    return f.found


def check_concept_pill():
    pill_js = _read("concept-pill.js")
    pill_css = _read("styles", "concept-pill.css")
    ladder_js = _read("practice", "stage-ladder.js")
    notch_css = _read("styles", "practice", "notch-menu.css")
    layout_css = _read("styles", "layout.css")
    index_html = _read("index.html")

    # ── The pill computes NOTHING ──────────────────────────────────
    # It is handed `pct` and draws it. The moment it grows a threshold, a
    # stage list or a bound of its own, this screen has two answers to one
    # question again — see the header of practice/stage-ladder.js.
    pill_code = _code(pill_js)
    for banned in ("PROMOTE_AT", "STAGES", "wilson", "Wilson", "streak"):
        assert banned not in pill_code, (
            "concept-pill.js must not carry ladder arithmetic (%r found). The "
            "fraction is computed once, in practice/stage-ladder.js, and "
            "handed over as the dd-concept-progress event" % banned
        )
    assert "dd-concept-progress" in pill_js, (
        "concept-pill.js has no dd-concept-progress listener — the chip has no "
        "input at all and will sit empty through every question"
    )

    # ── The ladder PUBLISHES, from both ends ───────────────────────
    # `_render` is every reading; `hide` is the KC-less question that must put
    # the chip away. Losing either one is silent: the chip freezes on the last
    # concept it was told about and keeps naming it over a different problem.
    assert "dd-concept-progress" in ladder_js, (
        "practice/stage-ladder.js no longer fires dd-concept-progress — the "
        "topbar concept pill has nothing to draw"
    )
    render_body = re.search(r"const _render = \(\) => \{(.*?)\n  \};", ladder_js, re.S)
    assert render_body and "_publish()" in render_body.group(1), (
        "_render() must call _publish(): it is the one path every reading goes "
        "through, so a chip that is not updated there is a chip showing the "
        "PREVIOUS question's progress"
    )
    hide_body = re.search(r"const hide = \(\) => \{(.*?)\n  \};", ladder_js, re.S)
    assert hide_body and "_publish()" in hide_body.group(1), (
        "hide() must call _publish() AFTER clearing `current` — a KC-less "
        "question otherwise leaves the previous concept named in the topbar "
        "above a problem it has nothing to do with"
    )

    # ── The 75% ceiling is inherited, not rescaled ─────────────────
    # `_overall()` tops out at (rungs - 1) / rungs because arriving at Solo is
    # not being done with the concept. Rescaling to 100 anywhere on the way to
    # the chip fills it to the brim while the queue is still going to serve
    # this concept.
    publish = re.search(r"const _publish = \(\) => \{(.*?)\n  \};", ladder_js, re.S)
    assert publish, "practice/stage-ladder.js has no _publish() — see above"
    assert "_overall()" in publish.group(1), (
        "_publish() must send _overall() itself. Any other number is a second "
        "reading of the same thing"
    )
    assert "0.75" not in publish.group(1) and "/ 0.75" not in pill_js, (
        "the ladder's 75% ceiling must reach the chip UNCHANGED — rescaling it "
        "to 100 tells the learner they are finished with a concept the queue "
        "is still going to serve them"
    )

    # ── null is not zero ───────────────────────────────────────────
    assert "is-unmeasured" in pill_js and "is-unmeasured" in pill_css, (
        "a missing reading (pct: null) must be DRAWN as missing — the "
        "is-unmeasured state exists so an unknown rung does not render as a "
        "0% bar, which reports no progress instead of no reading"
    )
    # 🔴 THE READING IS WRITTEN UNCONDITIONALLY. A "skip the write if nothing
    # changed" cache on `pct` cannot work, because `pct` has three states and
    # the cache's empty value is one of them: `hide()` resets it to null, so
    # `pct !== cached` is FALSE for the first UNMEASURED concept after a hide
    # and the whole block — fill width, is-unmeasured, aria-valuenow — is
    # skipped. Caught live: an unknown rung rendered as 33% of the previous
    # concept while the tooltip correctly said there was no reading.
    assert not re.search(r"pct\s*!==\s*\w", pill_code), (
        "concept-pill.js must not gate the reading behind a cached previous "
        "value. `pct` is number-or-null and null is also the cache's reset "
        "state, so the guard collapses exactly on the case it must not skip — "
        "the unmeasured concept that follows a hide()"
    )
    assert "aria-valuetext" in pill_js, (
        "a progressbar with no value states it in aria-valuetext; a bare "
        "aria-valuenow of 0 says the same wrong thing to a screen reader that "
        "an empty bar says to everyone else"
    )

    # ── Both label layers, one write ───────────────────────────────
    # The clip is a percentage of the layer's own box, so two layers holding
    # different strings put the two-colour seam in the middle of a glyph that
    # is not on the other layer.
    assert "dd-concept-label" in pill_js and "dd-concept-label-on" in pill_js, (
        "concept-pill.js must write BOTH label layers. One of them alone is a "
        "label that is legible on the fill or off it, never both"
    )
    assert "--dd-concept-pct" in pill_js, (
        "one custom property drives the fill's width AND the clip on the "
        "on-accent layer; two separate writes are how a label ends up painted "
        "for a ground the fill has not reached"
    )
    # ── IT IS A METER, NOT A LABEL ─────────────────────────────────
    # Seth asked for this twice over: the first build sized the chip to its
    # text, exactly like the level pill, and the note back was "much bigger ...
    # probably one-third of the screen". A width that goes back to `auto` or to
    # a `max-content` cap is the same regression, and it looks tidy in a diff.
    pill_rule = _rule(pill_css, ".dd-concept")
    assert pill_rule and re.search(r"width:\s*\d+vw", pill_rule), (
        ".dd-concept must claim a FIXED share of the viewport width (vw). "
        "Sized to its own text it is a label with a fill behind it, and a "
        "track whose length changes with the concept's name makes two "
        "different fractions look like the same progress"
    )
    height = re.search(r"height:\s*(\d+)px", pill_rule or "")
    assert height and 28 <= int(height.group(1)) <= 40, (
        "the pill must fill the bar's height without changing it — "
        "`--dd-topbar-h` is 44px, so anything under ~28px is the small chip "
        "again and anything over ~40px starts pushing the bar"
    )
    assert "flex: 0 1" in pill_rule and "min-width: 0" in pill_rule, (
        "the pill must be SHRINKABLE (`flex: 0 1 …` plus `min-width: 0`). In "
        "advanced mode this cell also holds the ten-button tab strip, and an "
        "unshrinkable third of the viewport pushes the topbar's side columns "
        "out of it"
    )

    fill = _rule(pill_css, ".dd-concept-fill")
    on = _rule(pill_css, ".dd-concept-text--on")
    assert fill and "--dd-concept-pct" in fill, (
        ".dd-concept-fill must take its width from --dd-concept-pct"
    )
    assert on and "--dd-concept-pct" in on, (
        ".dd-concept-text--on must clip to --dd-concept-pct — the same property "
        "the fill reads, or the two edges drift"
    )

    # ── The name is on screen exactly ONCE, at every width ─────────
    # The chip is dropped under 620px and the heading card comes back at 621px.
    # Move either edge independently and there is a width where the concept is
    # named twice, or a width where it is named nowhere.
    chip_off = re.search(r"@media \(max-width: (\d+)px\)\s*\{\s*\.dd-concept \{\s*display: none", pill_css)
    card_on = re.search(r"@media \(min-width: (\d+)px\)\s*\{\s*body \.question-number-row \{\s*display: none", pill_css)
    assert chip_off, "concept-pill.css must drop the chip at a narrow breakpoint"
    assert card_on, (
        "concept-pill.css must hide .question-number-row where the chip is "
        "shown — otherwise the concept is named twice, in two type sizes"
    )
    assert int(card_on.group(1)) == int(chip_off.group(1)) + 1, (
        "the two breakpoints must be complementary: the chip goes at <=%spx "
        "and the heading card must come back at %spx, not %spx"
        % (chip_off.group(1), int(chip_off.group(1)) + 1, card_on.group(1))
    )

    # ── The notch hangs off the BAR ────────────────────────────────
    notch = _rule(notch_css, ".practice-notch")
    assert notch and "position: absolute" in notch, (
        ".practice-notch must be absolutely positioned — in flow it is a "
        "toolbar item inside the bar, which is the layout it was moved out of"
    )
    assert "top: 100%" in notch, (
        ".practice-notch must sit at top: 100% of the bar's padding box, which "
        "is ON its bottom border. Any other offset floats it below the bar and "
        "it stops reading as a notch"
    )
    tab = _rule(notch_css, ".practice-notch-tab")
    assert tab and "border-top: 0" in tab, (
        "the notch shares the bar's bottom border; a second one across its top "
        "draws a double rule"
    )
    assert tab and re.search(r"border-radius:\s*0 0 ", tab), (
        "bottom-only corners are what make it a notch — a fully rounded pill "
        "under the bar reads as a detached toolbar"
    )

    # 🔴 The anchor. `.practice-notch` reaches `.topbar` only because nothing
    # between them is positioned.
    topbar = _rule(layout_css, ".topbar")
    assert topbar and ("position: sticky" in topbar or "position: relative" in topbar), (
        ".topbar must stay a POSITIONED box — it is the containing block the "
        "notch hangs from. Drop `position: sticky` and the notch resolves "
        "against the viewport instead"
    )
    # 🔴 THE SIDE TRACKS MUST KEEP THEIR AUTOMATIC MINIMUM. `min-width: 0` on
    # `.topbar-side` waives it, and two `1fr` tracks that may size below their
    # contents do exactly that once the middle cell asks for a third of the
    # viewport: measured at 1440px in advanced mode, the left track collapsed
    # to 62px with the level pill hanging past its edge and under the tab
    # strip. Nothing overflows the document, so there is no scrollbar to notice
    # and no error — just two overlapping controls.
    side = _rule(layout_css, ".topbar-side")
    assert side is not None and "min-width: 0" not in side, (
        ".topbar-side must not waive its automatic minimum (`min-width: 0`). "
        "Both side tracks are 1fr; below their content the level pill and the "
        "account control overlap the middle cell instead of shrinking — "
        "neither can shrink, both are nowrap"
    )

    # 🔴 AND IT IS A DIRECT CHILD OF THE HEADER. It worked inside `.topbar-mid`
    # — an absolutely-positioned box resolves against the nearest POSITIONED
    # ancestor and that cell is not one — but that is a NEGATIVE invariant held
    # by every element in between: `position: relative` on any of them
    # re-anchors the clock back into the bar, silently. Codex flagged the
    # fragility; the markup owns the relationship now, and this is what keeps
    # it owned. Parsed rather than pattern-matched: "is X inside Y" is a tree
    # question and a regex answers a different one.
    parent = _parent_of(index_html, "practice-notch")
    assert parent == ("header", "topbar"), (
        "#practice-notch must be a DIRECT child of <header class=\"topbar\"> "
        "(found parent: %r). It is positioned against that box; any element "
        "in between is one `position: relative` away from pulling the clock "
        "back up into the bar with no error anywhere" % (parent,)
    )

    # ── Nothing was renamed on the way ─────────────────────────────
    # The whole point of moving the notch with CSS is that practice/timer.js,
    # practice/placement-timer.js and practice/notch-menu.js keep working
    # untouched. They address it by id.
    for el_id in ("practice-notch", "practice-notch-tab", "practice-notch-clock",
                  "practice-notch-stop", "practice-notch-btn", "placement-timer"):
        assert 'id="%s"' % el_id in index_html, (
            "#%s is gone from index.html. The notch moved by CSS precisely so "
            "the three scripts that write to it did not have to change" % el_id
        )

    # ── THE CHIP IS DOWN WHEN THE QUESTION IS NOT ON SCREEN ────────
    # The ladder publishes per QUESTION and renders once in the background at
    # load, so on the event alone the chip named a concept on the idle screen
    # (a second after a cold load, for a question nobody had asked for) and on
    # every other tab. It reads the same two facts styles/practice/timer.css
    # uses to hide the ladder card itself.
    # 🔴 SCOPED TO THE TWO PLACES THAT DO THE WORK, not the file. Whole-file
    # substring checks pass on any other line that happens to contain the same
    # token — `"hidden"` is the class this file puts on the chip ITSELF, twice
    # — so a gate that had been disconnected from the page would still satisfy
    # them. Codex flagged the shape; these two slices are the fix.
    onscreen = re.search(r"const _onScreen\s*=.*?\n  \};", pill_code, re.S)
    assert onscreen, (
        "concept-pill.js has no _onScreen — the chip has no way to tell whether "
        "the question it names is on the screen, so it draws on the idle dial "
        "and on every other tab, naming a concept nothing on screen is about"
    )
    onscreen = onscreen.group(0)
    assert "page-practice" in onscreen, (
        "_onScreen must read #page-practice. It is the element whose state "
        "says whether there is a question on the screen at all"
    )
    for cls in ("hidden", "session-idle"):
        assert '.contains("%s")' % cls in onscreen, (
            "_onScreen must consult the %r class on #page-practice. `hidden` is "
            "another tab being up; `session-idle` is the practice page being up "
            "with no question on it - both mean the chip has nothing to "
            "describe. (`.contains`, not the bare name: this file writes the "
            "class `hidden` onto the chip itself.)" % cls
        )
    # The gate has to be ON the hide branch, not merely defined. A helper that
    # is written and never consulted is what this would otherwise miss.
    assert re.search(r"!\s*title\s*\|\|\s*!\s*_onScreen\(\)", pill_code), (
        "the chip's hide branch must test BOTH: no concept published, OR no "
        "question on screen to have one. Either alone leaves one of the two "
        "ways it goes stale"
    )
    # And the other half: a tab switch and a pause change nothing about the
    # READING, so the ladder never fires for either. Without an observer the
    # chip keeps whatever it had when the last question rendered.
    obs = re.search(r"if \(typeof MutationObserver.*?\n  \}", pill_code, re.S)
    assert obs, "concept-pill.js has no MutationObserver block"
    obs = obs.group(0)
    assert re.search(r"new MutationObserver\([^)]*\)\.observe\(", obs), (
        "the MutationObserver must actually observe something — constructing "
        "one and never calling .observe is the same as not having it"
    )
    assert "page-practice" in obs and 'attributeFilter: ["class"]' in obs, (
        "the observer must watch #page-practice's CLASS. Watching anything "
        "else means the chip survives a pause and a tab switch unchanged"
    )

    # ── A LESSON ON SCREEN IS A SCREEN ─────────────────────────────
    # The gate draws into #question-text, which is inside .practice-split, and
    # styles/practice/timer.css sets
    # `#page-practice.session-idle .practice-split { display: none }`. A gate
    # that fires while the page is idle therefore renders a whole lesson into a
    # display:none box and returns TRUE, telling its caller the learner is
    # reading. timer.js `resume()` is exactly that caller: it hands _resumeCore
    # over as onDone, and _resumeCore is the only thing that takes
    # `session-idle` off — so Continue led to a screen that never changed, with
    # an invisible lesson behind it and no way back but a reload.
    lessons_code = _code(_read("practice", "lessons.js"))
    gate = re.search(r'classList\.add\("lesson-mode"\).*?showPage\(\);', lessons_code, re.S)
    assert gate, (
        "practice/lessons.js: could not find the gate body between "
        "`lesson-mode` and its first showPage() — this guard can no longer see "
        "the screen it opens"
    )
    assert re.search(r'"page-practice"\)\??\.classList\.remove\("session-idle"\)', gate.group(0)), (
        "the lesson gate must clear `session-idle` on #page-practice before it "
        "draws. Its own screen lives inside .practice-split, which that class "
        "hides, so without this the gate renders a lesson nobody can see and "
        "tells its caller the learner is reading it"
    )
    # And ONLY that: unhiding the page would yank a learner out of whatever tab
    # they are actually reading.
    assert not re.search(r'"page-practice"\)\??\.classList\.remove\("hidden"\)', gate.group(0)), (
        "the lesson gate must NOT unhide #page-practice. `hidden` means the "
        "learner is on another tab, and pulling them off it is not the gate's "
        "call"
    )

    # ── AND DOES NOT RE-TEACH WHAT WAS ALREADY READ ────────────────
    # In backend mode `_pendingSteps` takes its steps from
    # `question.lesson_gate`, which the SERVER attached when the question was
    # served — a snapshot. Nothing the learner does afterwards changes the copy
    # riding on a question object already in this tab, and the exposure posts
    # made when a page is read are about the NEXT question the server picks.
    # resume() asks about exactly that object, rehydrated from what was
    # persisted at pause, so without a local check, pausing on a drill and
    # pressing Continue re-taught the page the learner had just read on the way
    # to it. Measured on prod: the exposure map held the concept's key while
    # the question's own gate still listed it as pending.
    backend_branch = re.search(
        r'if \(practiceMode === "backend"\).*?\n    \}', lessons_code, re.S
    )
    assert backend_branch, (
        "practice/lessons.js: could not find _pendingSteps' backend branch"
    )
    backend_branch = backend_branch.group(0)
    assert "_localExposure()" in backend_branch, (
        "the backend branch of _pendingSteps must drop gate entries this "
        "browser has already shown. Without it a resumed question re-teaches "
        "from its own stale snapshot of the gate"
    )
    assert re.search(r"exposed\[entry\.exposure_key \|\| entry\.kc\]", backend_branch), (
        "the suppression must key on the entry's OWN `exposure_key` (falling "
        "back to the kc), which is what `_markLocalExposure` writes when the "
        "page is read. Keying on anything else either never matches or "
        "suppresses concepts of the same KP that have not been taught yet"
    )
    # 🔴 ORDER. Dedupe-by-kc first would let a suppressed entry take a later,
    # UNREAD entry for the same KC down with it — the concept is then never
    # taught, which this file's own comments call the unrecoverable mistake.
    assert backend_branch.index("exposed[") < backend_branch.index("seen.has("), (
        "filter the gate by exposure BEFORE the dedupe-by-kc, or a dropped "
        "entry silently takes a later unread entry for the same KC with it"
    )
    # And it stays a SUPPRESSION, not a replacement: the server's per-account
    # map still decides what to teach next.
    assert "_stepFromGate" in backend_branch, (
        "backend mode must still build its steps from the server's gate "
        "entries — the exposure map is per BROWSER and cannot be the source of "
        "truth for an account that practises on two machines"
    )

    # ── RESUMING A SESSION KEEPS THE CONCEPT ───────────────────────
    # `buildPracticeQuestionFromBank` maps a BANK record to the render shape,
    # and the bank has no ladder_kc / ladder_stage / ladder_kc_title - those
    # are per-served-question, from the backend queue. Rebuilding a paused
    # question from the bank alone handed renderQuestion a question with no
    # concept on it, LadderUI.decorate found no kc and called
    # StageLadder.hide(), and the concept left the screen for the whole of the
    # resumed question - heading, ladder card and topbar chip together. It came
    # back only at the next question, which is served by the queue.
    timer_code = _code(_read("practice", "timer.js"))
    restore = re.search(r"const _restoreSavedQuestion\s*=.*?\n  \};", timer_code, re.S)
    assert restore, (
        "practice/timer.js has no _restoreSavedQuestion - the resume path was "
        "renamed and this guard can no longer see it"
    )
    restore = restore.group(0)
    assert re.search(r"\?\s*hydrateSavedPracticeQuestionFromBank", restore), (
        "resume must PREFER hydrateSavedPracticeQuestionFromBank over a plain "
        "buildPracticeQuestionFromBank: the saved question is the only place "
        "the ladder fields still exist, and the hydrate overwrites every "
        "artifact field from the bank afterwards, so the bank stays "
        "authoritative for the question itself"
    )
    # The comparison itself, not the identifier: `pausedState.questionId` is
    # already read one line above, by the early return.
    assert re.search(
        r"String\(\s*saved\.question_id[^)]*\)\s*===\s*String\(\s*pausedState\.questionId",
        restore,
    ), (
        "the hydrate must be gated on the saved question being THIS question. "
        "practiceProgress.currentQuestion is whatever was served last, and "
        "hydrating a different one would put another concept's name over a "
        "resumed question"
    )

    # ── All three tags, or none ────────────────────────────────────
    css_tag = "styles/concept-pill.css" in index_html
    js_tag = re.search(r'src="concept-pill\.js', index_html) is not None
    markup = 'id="dd-concept"' in index_html
    # 🔴 ALL THREE, NOT "all or none". The lenient form is the shape the other
    # runners in this tree use, and it was right for exactly as long as the
    # markup had not landed yet — but it also passes when the whole feature is
    # deleted, which is the regression this file exists to catch. Codex flagged
    # it; the markup is in, so the check is strict now.
    assert all([css_tag, js_tag, markup]), (
        "index.html is missing part of the concept-pill wiring (css=%s js=%s "
        "markup=%s). All three make the feature; any two of them make nothing "
        "and throw nothing" % (css_tag, js_tag, markup)
    )

    if True:
        # 🔴 SPECIFICITY, NOT SOURCE ORDER. The override used to be a bare
        # `.question-number-row`, which ties the whole behaviour to which of
        # two <link> tags comes first in a 60-tag <head> — an invariant nobody
        # reading either sheet can see. `body ` in front of it wins wherever
        # this sheet is linked.
        assert re.search(r"body\s+\.question-number-row\s*\{", pill_css), (
            "the heading-card override must out-SPECIFY question.css "
            "(`body .question-number-row`), not merely out-order it. At equal "
            "specificity, reordering two <link> tags brings the card back and "
            "the concept is named twice with no error anywhere"
        )
        # The chip lives in the bar. Anywhere else and it scrolls away with a
        # page, which is what the 08-24 move fixed for the clock.
        head = index_html[index_html.index('<header class="topbar"'):]
        head = head[:head.index("</header>")]
        assert 'id="dd-concept"' in head, (
            "#dd-concept must be inside <header class=\"topbar\">. Outside it, "
            "the concept scrolls away with whatever page it landed in"
        )
        assert 'id="practice-notch"' in head, (
            "#practice-notch must stay inside the topbar — it is positioned "
            "against it"
        )
        # Both label layers exist to be written. One of them missing is the
        # contrast bug the two-layer trick exists to avoid, and it is invisible
        # until the fill crosses the text.
        assert 'id="dd-concept-label"' in head and 'id="dd-concept-label-on"' in head, (
            "the concept pill needs BOTH label layers in the markup — the base "
            "one for the empty track and the clipped copy for the fill"
        )


if __name__ == "__main__":
    check_concept_pill()
    print("watch_concept_pill: ok")
