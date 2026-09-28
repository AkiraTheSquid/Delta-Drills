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

    # ONE tab, and the two it replaced are gone from the markup entirely.
    assert index_html.count('data-tab="learn-about-app"') == 1, (
        "there must be exactly one Learn about the App tab"
    )
    assert 'id="page-learn-about-app"' in index_html, "merged page missing"
    for dead in ('data-tab="why-this-app"', 'data-tab="how-to-use"',
                 'id="page-why-this-app"', 'id="page-how-to-use"'):
        assert dead not in index_html, f"{dead} came back; the merge is one page"

    # Since 2026-09-24 the page body is the AISC write-up (Seth: replace "Why
    # this app exists" and "How this app works" with the delta-drills-aisc
    # site). Each assertion is a way it fails SILENTLY.
    markup = re.sub(r"<!--.*?-->", "", index_html, flags=re.S)
    page = markup.split('id="page-learn-about-app"')[1].split("\n  </main>")[0]
    assert 'id="aisc-root"' in page and 'id="aisc-how"' in page, (
        "the AISC write-up (#aisc-root, with its #aisc-how section) must be "
        "the body of #page-learn-about-app"
    )
    assert 'class="lab-disclosure"' not in page, (
        "the old disclosures came back beside the write-up"
    )
    # The app's concept map (concept-graph/why-graph.js) is Fig. 2 of the
    # write-up since 2026-09-24 (Seth: "use the old embedded graph"). Exactly
    # one, inside that figure, or why-graph.js binds to the wrong frame.
    assert page.count('id="wta-graph-cy"') == 1, "Fig. 2 must hold ONE #wta-graph-cy"
    fig2 = page.split('id="aisc-fig-graph"')[1].split("</figure>")[0]
    has_frame = re.search(r'class="(?:[^"]*\s)?wta-graph(?:\s[^"]*)?"', fig2)
    assert 'id="wta-graph-cy"' in fig2 and has_frame, (
        "the concept map belongs inside Fig. 2 (#aisc-fig-graph)"
    )
    # Every id is prefixed, and every in-page link names a prefixed id: an
    # unprefixed `href="#how"` sets location.hash and scrolls nowhere.
    # wta-graph-* are why-graph.js's own ids, kept as that file expects them.
    aisc = page.split('id="aisc-root"')[1]
    for bad in re.findall(r'\bid="([^"]+)"', aisc):
        assert bad.startswith(("aisc-", "about-", "wta-graph-")), (
            f"#{bad} inside the write-up is not aisc- prefixed"
        )
    for ref in re.findall(r'href="#([^"]+)"', aisc):
        assert ref.startswith("aisc-") and f'id="{ref}"' in aisc, (
            f'href="#{ref}" in the write-up points at no aisc- section'
        )
    # The styles are scoped and loaded after the app's own, and the figures are
    # booted lazily AFTER cytoscape + dagre (vendor/graph/, deferred).
    assert 'href="aisc/aisc.css' in index_html, "aisc/aisc.css is not linked"
    boot = index_html.find('src="aisc/boot.js')
    dagre = index_html.find('src="vendor/graph/cytoscape-dagre.min.js')
    assert boot > dagre >= 0, (
        "aisc/boot.js must load after vendor/graph/cytoscape-dagre.min.js: its "
        "graph figures use the app's cytoscape with the dagre layout"
    )
    boot_js = _read(os.path.join(HERE, "aisc", "boot.js"))
    assert "DDAboutContentReady" in boot_js, (
        "aisc/boot.js must wait for about-page-editor.js to swap in any saved "
        "copy, or the figures bind to nodes that are about to be replaced"
    )
    for rel in ("aisc/data/kc_graph.json",
                "aisc/data/seth_progress.json"):
        assert os.path.exists(os.path.join(HERE, rel)), f"{rel} is missing"

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
    # THE RIGHT ARM ASKS WHICH COURSES, THEN OPENS PRACTICE. Since 2026-09-25
    # by way of ONE question first (Seth: "a single prompt question that asks
    # them which courses they want to include for study"): the arm opens
    # #page-course-pick, whose Continue opens Practice — where the "what have
    # you done before?" survey waits (it replaced the placement test the arm
    # led to until 2026-09-26).
    assert 'data-goto-tab="learn-about-app"' in fork and 'data-goto-tab="course-pick"' in fork, (
        "left arm reads about the app, right arm asks which courses — "
        "#page-course-pick — before Practice"
    )
    assert 'id="page-course-pick"' in markup and 'data-tab="course-pick"' not in markup, (
        "#page-course-pick is a page with no tab"
    )
    with open(os.path.join(HERE, "practice", "course-pick.js")) as f:
        pick_js = f.read()
    assert '"/api/practice/study-courses"' in pick_js and 'switchTab("practice")' in pick_js, (
        "the course question saves the answer and then opens Practice — "
        "it is a step on the way there, not a dead end"
    )
    assert 'src="practice/course-pick.js' in index_html, "course-pick.js is not loaded"
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

    # ---- ONE PAGE. THE PLACEMENT TEST IS RETIRED ------------------------
    # Seth, 2026-08-24: "the diagnostic and practice should be combined into one
    # tab, with it being called Learner Home". The placement page that split
    # back out on 2026-09-01 was retired with the test on 2026-09-26 ("just
    # removing the diagnostic test distinction"); the survey on the Learner
    # Home sets the starting point instead.
    for gone in ('id="page-diagnostic"', 'id="page-placement"'):
        assert gone not in index_html, (
            f"{gone} is back. The placement test was retired on 2026-09-26 — "
            "its routes are gone, so its page would render and never load"
        )
    assert 'data-tab="diagnostic"' not in markup and 'data-tab="placement"' not in markup, (
        "a placement tab is in the strip, for a test that no longer exists"
    )
    assert ">Learner Home<" in index_html, (
        "the Practice tab is called Learner Home now"
    )
    # 🔴 OLD NAMES STILL ARRIVE. `/diagnostic` (solo-route.js), a stale
    # `dd_recovered_tab` and old links all name the retired page; each must
    # land on Practice, not on the "unknown tab" fallback.
    assert 'diagnostic: "practice"' in app_js and 'placement: "practice"' in app_js, (
        "switchTab must map the retired `diagnostic` / `placement` names onto "
        "Practice"
    )
    # 🔴 SLICE THE PAGE, NOT THE REST OF THE DOCUMENT. `split(...)[1]` alone
    # runs to the end of the file. The pages are top-level <main>s, closed at
    # two spaces.
    home = index_html.split('id="page-practice"')[1].split("\n  </main>")[0]
    for needle in ('id="learner-survey"', 'id="learner-xp"', 'id="learner-xp-graphs"'):
        assert needle in home, (
            f"{needle} is not on the Learner Home — the daily surface is the "
            "survey that sets the start, measured XP and its two graphs"
        )
    # Seth, 2026-09-26: the page has what he described and nothing else.
    for gone in ('id="readiness-dial"', 'id="learner-group"', 'id="learner-activity"'):
        assert gone not in home, f"{gone} is back on the Learner Home; it was removed"
    # Seth, 2026-09-27: the Groups TAB is back (account menu → Groups), and the
    # board stays off the Learner Home — a group changes the Home's graphs
    # (practice/xp-group-view.js), nothing else.
    assert 'id="page-groups"' in index_html and 'data-goto-tab="groups"' in index_html, (
        "the Groups tab and its account-menu row are gone again"
    )
    for word in ("ARENA", "LeetCode"):
        assert word not in index_html.split('id="practice-session-setup"')[1].split('id="session-resume-panel"')[0], (
            f"the idle card names {word}; its copy is course-agnostic"
        )

    # 🔴 THE MODE IS ANNOUNCED. Every surface that reads the backend on load
    # (xp.js, survey.js, course-pick.js, courses.js)
    # parses before practice/init.js, and a read made while `practiceMode` is
    # still its "local" default answers null.
    init_js = _read(os.path.join(HERE, "practice", "init.js"))
    assert "delta:practice-mode-ready" in init_js and "detectPracticeMode();" in init_js, (
        "practice/init.js no longer announces the decided mode; the Learner "
        "Home's surfaces wait on an event that never fires"
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

    # 🔴 THE FIRST-RUN LANDING IS A TWO-FILE HANDSHAKE ON A STRING. test-users.js
    # writes sessionStorage["dd_first_run_tab"] = "welcome" just before the
    # reload that commits a switch INTO a never-used test user; app.js reads it
    # once at boot and prefers it over its `authToken ? "practice"` default. A
    # test user always has a token, so with either half missing the demo opens
    # on the practice surface — the exact screen the fork exists to come before,
    # and nothing anywhere says the handover was skipped.
    test_users = _read(os.path.join(HERE, "test-users.js"))
    assert '"dd_first_run_tab"' in test_users and '"dd_first_run_tab"' in app_js, (
        "the first-run landing key must be spelled the same in test-users.js "
        "(writer) and app.js (reader), or a new test user silently opens on "
        "the practice surface instead of the welcome fork"
    )
    # 🔴 THE GUARD, NOT JUST THE CALL. A bare "is requestOnboardingLanding()
    # mentioned" check passes just as happily on an unconditional call or an
    # inverted test — and both of those send a test user who has been demoed a
    # dozen times back to the fork, which is a wrong answer that looks like a
    # working feature. Match the guarded line itself.
    assert re.search(r"if\s*\(\s*!user\.lastUsedAt\s*\)\s*requestOnboardingLanding\(\);", test_users), (
        "test-users.js must ask for the onboarding landing ONLY for a test "
        "user that has never been used — `if (!user.lastUsedAt) "
        "requestOnboardingLanding();`"
    )
    assert "firstRunTab ||" in app_js, (
        "app.js's boot switchTab no longer consults the first-run landing key"
    )
    assert "takeSessionTab(FIRST_RUN_TAB_KEY)" in app_js, (
        "the first-run key must be read THROUGH the read-once-and-clear "
        "helper; a value left behind lands a later reload on the fork"
    )
