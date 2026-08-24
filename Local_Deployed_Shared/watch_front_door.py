"""watch_front_door.py — the front-door checks, split out of watch.py.

Extracted 2026-08-24 when watch.py crossed Modulario's 700-LOC line. Same
contract as every check in watch.py: raise AssertionError to fail. watch.py
imports check_front_door back into its own namespace and keeps it in the
__main__ checks list, so `mod watch` and the explicit runner both still see
it — the split must never change WHICH checks run (a runner list has dropped
checks silently before; see delta-note-calendar-style-inheritance).
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── Front door: the welcome fork + "Learn about the App" ──────────
def check_front_door():
    """The 2026-08-23 merge (Seth). Two tabs that both answered "what is this"
    became one, and the landing page became a two-arrow fork with nothing else
    on it. Every assertion here is a thing that fails SILENTLY — a page with no
    route in, a tab strip that comes back, a map that never draws — rather than
    raising anything at runtime."""
    index_html = _read(os.path.join(HERE, "index.html"))
    app_js = _read(os.path.join(HERE, "app.js"))
    learn_css = _read(os.path.join(HERE, "styles", "learn-about.css"))
    why_graph = _read(os.path.join(HERE, "concept-graph", "why-graph.js"))
    diagnostic = _read(os.path.join(HERE, "practice", "diagnostic-page.js"))

    # ONE tab, and the two it replaced are gone from the markup entirely.
    assert index_html.count('data-tab="learn-about-app"') == 1, (
        "there must be exactly one Learn about the App tab"
    )
    assert 'id="page-learn-about-app"' in index_html, "merged page missing"
    for dead in ('data-tab="why-this-app"', 'data-tab="how-to-use"',
                 'id="page-why-this-app"', 'id="page-how-to-use"'):
        assert dead not in index_html, f"{dead} came back; the merge is one page"

    # The disclosures. The lead paragraph stays OUTSIDE them on purpose: what
    # is unconditionally on screen is what the app is.
    assert index_html.count('class="lab-disclosure"') == 2, (
        "Learn about the App is two disclosures: the three markers, and How "
        "the app works"
    )
    hero = index_html.split('id="page-learn-about-app"')[1].split("<details")[0]
    assert "<h1>Why this app exists</h1>" in hero and 'class="hiw-lead"' in hero, (
        "the heading and the lead paragraph must sit ABOVE the first "
        "disclosure — everything else on the page is opt-in reading"
    )
    # Comment-stripped: this file's own comments narrate the merge and name
    # the old tab, and an assertion that a phrase is gone must not be fooled
    # by prose that is explaining why it is gone.
    markup = re.sub(r"<!--.*?-->", "", index_html, flags=re.S)
    assert "How the app works" in markup and "How to use it" not in markup, (
        'the second disclosure is titled "How the app works" (Seth), not '
        '"How to use it"'
    )
    # The map moved under "How the app works" and must have stayed there.
    how_block = index_html.split('id="lab-how"')[1].split("</details>")[0]
    assert 'id="wta-graph-cy"' in how_block, (
        "the concept map belongs inside the How-the-app-works disclosure"
    )
    assert 'container.closest("details")' in why_graph and '"toggle"' in why_graph, (
        "why-graph.js must draw on the <details> toggle as well as on the "
        "page's class: inside a closed disclosure offsetParent is null, so "
        "the page-class observer alone leaves the map on 'Loading the map...'"
    )

    # The fork. Two learner arms plus the quiet instructor arm below them
    # (Seth, 2026-08-24: the expert's workflow parts from the learner's here),
    # all [data-goto-tab], and NOT a tab.
    assert 'id="page-welcome"' in index_html, "the welcome fork is missing"
    # Comment-stripped for the same reason as `markup` above: the arm comments
    # narrate the data-goto-tab idiom by name.
    fork = markup.split('id="page-welcome"')[1].split("</main>")[0]
    assert fork.count("data-goto-tab") == 3, (
        "the fork is the two learner choices plus the instructor arm, and "
        "nothing else (Seth)"
    )
    assert 'id="welcome-arm-instructor"' in fork, (
        "the instructor arm lost the id instructor-mode.js listens on — the "
        "button would still navigate but never flip the flag"
    )
    assert 'data-goto-tab="instructor-review"' in fork, (
        "the instructor arm lands on the REVIEW SURFACE (2026-08-24), not on "
        "Practice — retargeting it to a learner page makes entering the mode "
        "drop an expert into drills with nothing to review"
    )
    assert fork.find("welcome-arm--right") < fork.find("welcome-arm--instructor"), (
        "the instructor arm sits BELOW the learner pair, not among them"
    )
    # The right arm points at "practice" since the merge: the placement test
    # is a card ON the Learner Home now, not a page of its own.
    assert 'data-goto-tab="learn-about-app"' in fork and 'data-goto-tab="practice"' in fork, (
        "left arm reads about the app, right arm goes to the Learner Home "
        "where the placement test is"
    )
    assert "optional" in fork, (
        "the reading path must say out loud that it is optional, or a fork "
        "reads as a prerequisite"
    )
    assert 'data-tab="welcome"' not in markup, (
        "#page-welcome is a fork, not a tab"
    )
    assert '"welcome")' in app_js and 'dd-welcome' in app_js, (
        "app.js must land a first-time visitor on the fork and stamp "
        "body.dd-welcome so the guest banner comes off that one screen"
    )
    assert "body.dd-welcome .guest-banner" in learn_css, (
        "the guest banner lives outside every .page, so only a body-class "
        "rule can take it off the fork"
    )

    # The strip. Basic mode has none; advanced mode is the whole way back.
    for selector in ("body.dd-basic-mode .tabs",
                     "body.dd-basic-mode .nav-drawer .tabs",
                     "body.dd-basic-mode .nav-toggle"):
        assert selector in learn_css, (
            f"{selector} missing: basic mode must have no tab strip, in the "
            "topbar OR parked in the drawer, and no hamburger to open an "
            "empty drawer with"
        )
    css_pos = index_html.find('href="styles/learn-about.css')
    drawer_pos = index_html.find('href="styles/nav-drawer.css')
    assert css_pos > drawer_pos >= 0, (
        "learn-about.css must load AFTER nav-drawer.css — its rule for the "
        "strip parked in the drawer ties on specificity with nav-drawer.css's "
        "own, and source order is the only thing that settles it"
    )
    assert 'class="dd-basic-nav"' in index_html and ".dd-basic-nav" in learn_css, (
        "the Account page needs the basic-mode escape row: with no strip, the "
        "cog is the only way off Practice and this row is the only way back"
    )
    assert 'data-goto-tab="practice"' in index_html.split('class="dd-basic-nav"')[1][:600], (
        "the escape row must lead back to Practice"
    )

    # 🔴 A NAME THAT MATCHES NO PAGE BLANKS THE APP. switchTab hides every
    # `.page` whose id is not `page-<name>`, so an unknown name leaves a topbar
    # over nothing, silently. The rename is the case we know about — a stale
    # `dd_recovered_tab` written by guest-session.js before a reload that
    # crosses a deploy — but the guard has to be general, because the next
    # rename will not come with a note.
    assert "renamedTabs" in app_js and '"why-this-app": "learn-about-app"' in app_js, (
        "switchTab must map the two retired tab names onto the page they "
        "merged into: a URL alias in solo-route.js cannot reach a name that "
        "arrives from sessionStorage"
    )
    assert "if (!document.getElementById(`page-${tabName}`))" in app_js, (
        "switchTab must fall back when the requested page does not exist — "
        "without it an unknown tab name hides every page and shows nothing"
    )

    # ---- ONE TAB: the Learner Home ---------------------------------------
    # Seth, 2026-08-24: "the diagnostic and practice should be combined into
    # one tab, with it being called Learner Home". #page-diagnostic is deleted
    # and everything that was on it lives inside #page-practice.
    assert 'id="page-diagnostic"' not in index_html, (
        "#page-diagnostic is back. The placement test is a card on the Learner "
        "Home; two pages sharing one editor is what the tab lock existed for"
    )
    assert 'data-tab="diagnostic"' not in markup, (
        "the Placement test tab is back in the strip"
    )
    assert ">Learner Home<" in index_html, (
        "the Practice tab is called Learner Home now"
    )
    home = index_html.split('id="page-practice"')[1]
    for needle in ('id="diagnostic-overview"', 'id="diagnostic-results"',
                   'id="diagnostic-workspace-host"', 'id="learner-areas"',
                   'id="placement-areas"', 'id="readiness-dial"'):
        assert needle in home, (
            f"{needle} is not on the Learner Home — the merge moved the whole "
            "placement surface onto #page-practice"
        )
    # 🔴 THE STATUS READ IS WIRED TO THE PAGE THE PLACEMENT IS ON. app.js calls
    # DiagnosticPage.refresh() on entry to that tab and leave() on entry to any
    # other; pointed at the retired name it silently took the leave branch every
    # time, and the Learner Home sat on "Loading placement status…" with no area
    # bars. Nothing throws when this is wrong.
    assert 'if (tabName === "practice") window.DiagnosticPage.refresh();' in app_js, (
        "app.js must refresh the placement status when the Learner Home opens"
    )
    assert 'diagnostic: "practice"' in app_js, (
        "switchTab must map the retired `diagnostic` name onto the Learner "
        "Home: it still arrives from /diagnostic, from a stale "
        "dd_recovered_tab and from practice/events.js"
    )
    # The lock, and its redirect, are gone with the second tab.
    # `setPracticeLock(` — the call or the definition, not the WORD: the file
    # documents at length what the lock was and why it went, and an assertion
    # that a name is gone must not be tripped by the note explaining it.
    assert "setPracticeLock(" not in diagnostic, (
        "the Practice tab lock is back. One tab cannot be locked against "
        "itself, and a disabled tab has no :disabled style — it looks live "
        "and eats the click"
    )
    assert 'byId("page-diagnostic")' not in diagnostic, (
        "diagnostic-page.js is still looking for the deleted placement page; "
        "every one of those reads is silently undefined"
    )

    # 🔴 THE AREA BARS ARE ON THE IDLE SCREEN, not locked inside the results
    # card. Seth, 2026-08-24: "it should display the information about einops,
    # numpy, and einsum to be learned". /diagnostic/status returns all three
    # areas from the first call on a new account, so there is always something
    # to draw; before this they appeared only after a COMPLETED placement.
    setup = home.split('id="practice-session-setup"')[1].split('class="practice-split"')[0]
    assert 'id="learner-areas"' in setup and 'id="placement-areas"' in setup, (
        "the area bars must sit on the idle surface, which is what a learner "
        "opens every day — inside #diagnostic-results they show only after a "
        "placement is complete"
    )
    results = index_html.split('id="diagnostic-results"')[1].split("</section>")[0]
    assert 'id="placement-areas"' not in results, (
        "there are two area lists again. One writer, one host: "
        "placement-results.js renderAreas fills #placement-areas and it is on "
        "the idle surface"
    )
    placement_results = _read(os.path.join(HERE, "practice", "placement-results.js"))
    assert "renderAreas," in placement_results or "renderAreas }" in placement_results, (
        "renderAreas must be public — diagnostic-page.js calls it on every "
        "status read, not only on a finished placement"
    )
    assert "PlacementResults?.renderAreas(" in diagnostic, (
        "the area bars are only drawn when a placement COMPLETES again"
    )

    # 🔴 THE SPECIFICITY GUARD. With the host on the same page as the idle
    # screen, `#page-practice.session-idle .practice-split { display: none }`
    # (styles/practice/timer.css) outranks `.diagnostic-workspace-host
    # .practice-split { display: flex }` — a probe would render as a blank idle
    # screen. Two IDs settle it without depending on stylesheet order.
    diag_css = _read(os.path.join(HERE, "styles", "practice", "diagnostic.css"))
    assert "#page-practice #diagnostic-workspace-host .practice-split" in diag_css, (
        "the hosted workspace has no rule that outranks #page-practice."
        "session-idle, so a placement probe renders as a blank idle screen"
    )
    assert "#page-practice.diagnostic-running #diagnostic-overview" in diag_css, (
        ".diagnostic-running is written on #page-practice now; the old "
        "#page-diagnostic selector matches nothing"
    )

    # 🔴 THE BOOT REFRESH MUST WAIT FOR THE MODE. diagnostic-page.js parses
    # before practice/init.js, and `PracticeAPI.diagnosticStatus()` returns null
    # — not a failure, NULL — while `practiceMode` is still its "local" default.
    # `render(null)` paints "Sign in to take the placement test." with the area
    # bars hidden. Harmless while the placement had its own (hidden-at-load)
    # page; since the merge the guard reads #page-practice, which every visitor
    # lands on, so a parse-time refresh is the first thing they see.
    init_js = _read(os.path.join(HERE, "practice", "init.js"))
    assert "delta:practice-mode-ready" in init_js and "detectPracticeMode();" in init_js, (
        "practice/init.js no longer announces the decided mode; "
        "diagnostic-page.js is waiting on an event that never fires"
    )
    assert "delta:practice-mode-ready" in diagnostic, (
        "diagnostic-page.js refreshes at parse time again — before "
        "detectPracticeMode() has run, which renders the signed-out copy at "
        "signed-in learners"
    )
    boot = diagnostic.split("window.DiagnosticPage = DiagnosticPage;")[-1]
    assert "DiagnosticPage.refresh();" in boot and "addEventListener" in boot, (
        "the boot block must refresh THROUGH the mode-ready event, not at "
        "parse time"
    )

    # 🔴 apiFetch MUST BE ON `window`. It is a top-level const in app.js, so it
    # is not one by default, and concept-graph/kc_lattice_read.js and
    # concept-graph/lesson-graph.js both guard `window.apiFetch || fetch` — the
    # fallback being a RELATIVE url that never reaches the backend. Locally a
    # 404; on Vercel a 200 text/html from the SPA rewrite, which reads as
    # "guest" on both the knowledge graph and the "Why this app exists" map.
    app_js = _read(os.path.join(HERE, "app.js"))
    assert "window.apiFetch = apiFetch;" in app_js, (
        "apiFetch is no longer published on window; concept-graph's two "
        "readers are silently fetching relative /api urls again"
    )


