"""watch.py — health checks for content

This script's correctness depends on Colab internals that no test can pin down
from the shell, so these checks guard the parts that CAN be checked: that the
message contract still matches what the panel sends, that the selectors are
still referenced, and that the two rules whose violation produces silent wrong
behaviour are still in place.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "..", "panel")

# Undocumented Colab internals. Losing a reference means every jump silently
# fails, so assert each is still mentioned.
SELECTORS = [
    "colab-scroller#notebook-main",
    "div.cell",
    "md-icon-button.header-section-toggle",
]


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def check_imports():
    script = _read(os.path.join(HERE, "colab.js"))
    assert "chrome.runtime.onMessage.addListener" in script, (
        "colab.js must register a message listener or the panel cannot reach it"
    )


def check_public_api():
    """Every `dd:` message the panel sends has a case here.

    The two files are wired only by string, so a rename on one side fails at
    runtime as an unanswered message rather than as an error.
    """
    script = _read(os.path.join(HERE, "colab.js"))
    handled = set(re.findall(r'case "(dd:[a-z-]+)"', script))
    sent = set()
    for name in ("panel.js", "api.js", "navigate.js"):
        path = os.path.join(PANEL, name)
        if os.path.exists(path):
            sent |= set(re.findall(r'type: "(dd:[a-z-]+)"', _read(path)))
    unhandled = sent - handled
    assert not unhandled, f"panel sends messages colab.js does not handle: {sorted(unhandled)}"


def check_invariants():
    script = _read(os.path.join(HERE, "colab.js"))

    for sel in SELECTORS:
        assert sel in script, f"lost the Colab selector {sel!r}"

    # A cell in a collapsed section has zero height: scrollIntoView succeeds and
    # nothing moves. Expanding first is the whole fix.
    assert "expandAbove" in script, "goto must expand collapsed sections before scrolling"
    assert script.index("expandAbove(cell)") < script.index("await scrollToCell"), (
        "expandAbove must run BEFORE the scroll, not after"
    )

    # The text fallback is the only thing that works on upstream ARENA
    # notebooks, which are nbformat 4.2 with no cell ids.
    assert "if (text)" in script, "findCell must keep its rendered-text fallback"

    # Notebook identity decides whether the panel switches notebooks before a
    # jump. Cell ids are the fragile route (Colab drops them on 4.2 files), so
    # the rendered-text routes have to stay.
    assert "DD_LESSON_ID" in script, (
        "identify must keep reading DD_LESSON_ID — it is the id-independent route"
    )
    assert "dd:dd-lesson-" in script, "identify must keep the lesson comment-marker fallback"

    # A URL carrying #scrollTo= gets stored as the notebook's address and
    # reopens mid-notebook forever after.
    assert 'location.href.split("#")' in script, (
        "identify must report a fragmentless URL — the panel stores it to reopen the notebook"
    )

    # A modal blocks the page and the extension stops receiving messages.
    for banned in ("alert(", "confirm(", "prompt("):
        assert banned not in script, f"{banned} blocks the page and kills the message channel"


def check_focus_cannot_blank_the_notebook():
    """Focus mode hides cells. The ways that goes wrong are all silent.

    * **Nothing is hidden unless a target RESOLVED to real cells.** With no
      match every cell would be tagged out-of-focus and the notebook would go
      blank — and blank is exactly what a failed load looks like. On upstream
      ARENA notebooks (nbformat 4.2, no cell ids at all) no match is the normal
      case, so this is not an edge condition.
    * **The setup cell is never hidden.** It holds the imports and
      `DD_LESSON_ID`; hidden, the answer cell dies on NameError and reads as
      broken starter code rather than as a missing prerequisite.
    * **Group membership needs a digit boundary.** A bare prefix test puts
      `dd-q12` and `dd-q123` in one group, so one problem drags another's cells
      on screen.
    * **The fragment changes without a navigation.** Opening the next problem
      only rewrites `#scrollTo=`, so without a hashchange listener the notebook
      keeps showing the previous problem.
    """
    script = _read(os.path.join(HERE, "colab_focus.js"))

    assert "inFocus > 0" in script, (
        "colab_focus.js must require at least one matching cell before hiding "
        "anything — otherwise an unmatched target blanks the notebook"
    )
    assert "dd-always-visible" in script and "dd-setup" in script, (
        "the setup cell must stay visible in focus mode, or the problem cannot run"
    )
    assert re.search(r"dd-q\(\\d\+\)\(\?:\$\|\[\^0-9\]\)", script), (
        "problemOf must match dd-q<n> with a trailing boundary, or dd-q12 and "
        "dd-q123 land in the same group"
    )

    # The worked-example pair rides on that same boundary — see
    # `check_stage_two_pair_survives_focus`, which runs the shipped pattern
    # against real anchors rather than asserting its text a second time here.
    assert 'addEventListener("hashchange"' in script, (
        "focus must re-apply on hashchange — opening the next problem changes "
        "only the fragment, which is not a navigation"
    )

    # Colab rewrites #scrollTo= to whatever cell is at the top of the viewport,
    # so the fragment is a moving target and reading it straight means scrolling
    # off the problem turns focus off. It has to be a way to CHANGE the target,
    # never a way to clear it. `apply` reads the sticky value, not the raw one.
    assert "const seen = focusTarget();" in script, (
        "apply must read the sticky focusTarget, not targetProblem — Colab "
        "rewrites the fragment while the learner scrolls, so reading it "
        "directly drops focus the moment they scroll past the problem"
    )
    # ...and unconditionally. Observing only when focus is ON lets the sticky
    # value go stale: route with focus off, scroll, re-enable, and the LAST
    # problem comes back instead of the current one.
    assert script.index("const seen = focusTarget();") < script.index("settings.focus ? seen"), (
        "focusTarget must be called before the settings.focus test, not inside "
        "it — the fragment has to be observed even while focus is disabled"
    )
    for banned in ("alert(", "confirm(", "prompt("):
        assert banned not in script, f"{banned} blocks the page and kills the message channel"


def check_stage_two_pair_survives_focus():
    """A stage-2 problem's worked example must be in focus with its problem.

    Stage 2 is a PAIR — a solved example, then the same move on different
    specifics for the learner to do — and half a pair is worse than neither
    half: the learner is routed to a problem the course believes it has just
    demonstrated, and the demonstration is not on screen. That was the state
    before the pair existed, and it is a silent one, because a bare problem
    looks exactly like a problem.

    The pairing is carried by the ANCHOR: the generator mints the example as
    `dd-q<problem>-example`, naming the problem it scaffolds rather than the
    bank question it was built from, so `problemOf` groups it with its problem
    for free. Two things therefore have to hold, and both are checked here
    rather than in the generator, because this is the file that would break
    them.

    1. `problemOf` must keep accepting a NON-DIGIT suffix. It is the same regex
       that stops `dd-q12` from dragging in `dd-q123`, so the boundary can only
       be `[^0-9]` — tighten it to `$` and every scaffold cell silently leaves
       the group.
    2. Nothing may hide an example cell. The one mechanism that hides a cell
       inside a focused group is `dd-hide-solutions`, and it keys on
       `dd-q<n>-solution`; an example is not a solution and must never be
       tagged as one, or the thing the learner is told to read disappears until
       they answer.

    Rule 1 is checked by LIFTING the pattern out of the file and running it,
    rather than by matching its text. The two are not the same check. Someone
    tidying `dd-q(\\d+)(?:$|[^0-9])` into an explicit
    `dd-q(\\d+)(?:$|-(?:hints|code|check|solution))` writes something that reads
    more careful, passes any text-shaped assertion that was updated alongside
    it, and silently drops every example out of focus — leaving a bare problem
    with no way for the learner to tell that something was meant to be above it.
    Executing the real thing against real anchors is the only version of this
    check that cannot be satisfied by editing it.
    """
    script = _read(os.path.join(HERE, "colab_focus.js"))

    found = re.search(r"const m = /(\^dd-q\(\\d\+\).*?)/\.exec", script)
    assert found, "could not find problemOf's regex in colab_focus.js to test it"
    problem_of = re.compile(found.group(1))
    # `-worked*` is the general case and by far the more load-bearing of the
    # two: it is the segment's own authored worked example, re-anchored so that
    # focus keeps it with the problem it demonstrates. 292 cells across the nine
    # notebooks ride on this matching. `-example` is the promoted-bank-question
    # variant, which so far exists for one pair.
    for anchor in ("dd-q481", "dd-q481-example", "dd-q481-example-code",
                   "dd-q481-worked", "dd-q481-worked-code", "dd-q481-worked-1"):
        m = problem_of.match(anchor)
        assert m and m.group(1) == "481", (
            f"{anchor} must resolve to problem 481 — a worked example is on screen "
            f"only because it shares its problem's number"
        )
    # ...and the boundary still has to hold the other way, or one problem's
    # example would surface on a different problem entirely.
    assert problem_of.match("dd-q4811-example").group(1) == "4811"
    assert re.search(r"SOLUTION\s*=\s*/\^dd-q\(\\d\+\)-solution\$/", script), (
        "the solution pattern must stay anchored to `-solution$` — a looser one "
        "would also match dd-q<n>-example and hide the worked example until the "
        "learner had already answered"
    )
    assert "dd-example" in script, (
        "the example cells must still be tagged, or nothing on screen tells the "
        "learner which of two adjacent code cells is the one to read"
    )


def check_a_lesson_is_a_focus_target():
    """A concept anchor must focus the concept, not fall through to a problem.

    The ladder teaches before it drills, and on the Colab edition the teaching
    step routes to `#scrollTo=dd-kp-<slug>` (practice/colab_mode.js's
    `hrefForKc`). When only `dd-q<n>` counted as a target, that fragment
    resolved to nothing, `sticky` kept the PREVIOUS problem, and the lesson the
    learner had just been sent to stayed `display:none` — scrolled to and
    invisible. The symptom is the worst kind: the notebook does not move, which
    reads as the link being dead rather than as focus being wrong.

    Both halves are checked by lifting the shipped patterns and running them,
    for the reason spelled out in `check_stage_two_pair_survives_focus`: a
    text-shaped assertion is satisfied by editing it.

    The ORDER inside `groupOf` is the other half. A segment's problems sit
    inside its KP section, so a `dd-q…` cell would match the concept run too if
    the concept were tested first — and then routing to a problem would unfold
    the entire lesson around it.
    """
    script = _read(os.path.join(HERE, "colab_focus.js"))

    found = re.search(r"const KP = /(\^dd-kp-.*?)/i;", script)
    assert found, "could not find the KP anchor pattern in colab_focus.js"
    kp = re.compile(found.group(1), re.I)
    assert kp.match("dd-kp-numpy-ndarray-model"), "a concept header must be a target"
    assert not kp.match("dd-q49"), "a problem must not be read as a concept"
    assert not kp.match("dd-kp"), "a slugless concept anchor names no section"

    assert "function targetGroup()" in script, (
        "the fragment reader must resolve BOTH kinds of target — a "
        "problem-only reader leaves a lesson pinned to the last problem"
    )
    assert "targetProblem" not in script, (
        "targetProblem is gone; leaving a second reader around invites one "
        "caller to keep using the problem-only one"
    )
    assert "groupsOf(anchor, currentKp, currentSeg)" in script, (
        "membership in a concept section is positional — the tagging walk has "
        "to carry the current KP and the current segment, since the prose "
        "cells' ids name nothing"
    )
    # The body only — the signature names `currentKp` before anything runs.
    group_of = script[script.index("function groupsOf("):]
    group_of = group_of[group_of.index("{"):group_of.index("\n  }")]
    assert group_of.index("problemOf(anchor)") < group_of.index("currentKp"), (
        "groupsOf must test the problem anchor FIRST — every problem sits inside "
        "a KP section, so testing the section first would put a problem's cells "
        "in the lesson's group and unfold the whole concept around the drill"
    )


def check_one_concept_of_a_lesson_is_a_focus_target():
    """A segmented KP must focus ONE concept, not all of them.

    A KP is not one idea — `numpy.ndarray-model` teaches three — and the tutor
    hands them out one at a time. The panel says "Concept 2 of 3"; the notebook
    has to agree, or the learner is told they are on the second of three things
    and shown all three at once. That is not a crash and not even visibly
    wrong: it is the pre-segment behaviour, still rendering, still scrolled to
    roughly the right place.

    The two halves are separate failures and both are checked. The ANCHOR has
    to be recognised as a target (otherwise `sticky` keeps the previous
    problem and the concept stays `display:none`), and a concept's cells have
    to stay in their KP's run as well as their own (otherwise segmenting a KP
    silently breaks the `dd-kp-…` link the knowledge graph routes through).
    """
    script = _read(os.path.join(HERE, "colab_focus.js"))

    found = re.search(r"const SEG = /(\^dd-seg-.*?)/i;", script)
    assert found, "could not find the segment anchor pattern in colab_focus.js"
    seg = re.compile(found.group(1), re.I)
    assert seg.match("dd-seg-numpy-ndarray-model-1"), "a concept header must be a target"
    assert not seg.match("dd-q49"), "a problem must not be read as a concept"
    assert not seg.match("dd-kp-numpy-ndarray-model"), (
        "a whole-KP anchor must not be read as one concept of it"
    )

    body = script[script.index("function groupsOf("):]
    body = body[body.index("{"):body.index("\n  }")]
    assert "kp:" in body and "seg:" in body, (
        "a concept's prose must belong to BOTH its concept and its KP — one or "
        "the other means segmenting a KP breaks the whole-lesson link"
    )
    assert "return groups" in body, (
        "groupsOf must return every group a cell is in, not the first one it "
        "matches"
    )

    walk = script[script.index("cells.forEach((cell) => {"):]
    assert "currentSeg = null" in walk[:walk.index("cell.classList.toggle")], (
        "the walk must close the open concept — at the next KP header, and at "
        "the first problem, or the KP's guided/applied/independent drills and "
        "its Common mistakes trail into whichever concept came last"
    )


def check_css_is_opt_in():
    """Every styling rule is scoped to a class this extension adds.

    An unscoped rule restyles Colab for the student even with both toggles off,
    on every notebook they ever open — including ones that have nothing to do
    with Delta Drills. The toggle would then be a lie, and the only way back
    would be uninstalling.

    The toggle panel itself (`#dd-colab-toggle`) is the deliberate exception: it
    is the way OUT of the theme, so scoping it under the theme would make it
    disappear along with what it disables.

    `html.dd-hide-solutions` is a third scope, and the only one that is ON by
    default — solutions stay hidden until the learner has answered. That makes
    it the one class that could restyle an unrelated notebook, so every rule
    under it must ALSO name `.dd-solution`, which colab_focus.js only ever puts
    on a `dd-q<n>-solution` cell. No generated notebook, no match, no change.

    `html.dd-no-ai` is also default-on, and its rules name Monaco's classes
    rather than any of ours — so the gate cannot live in the selector, and lives
    in the class instead: `apply` only adds it to a notebook carrying `dd-`
    anchors. That is asserted separately, in
    `check_gemini_suppression_is_ours_only`.
    """
    css = _read(os.path.join(HERE, "colab_dd.css"))
    stripped = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    for block in re.finditer(r"([^{}]+)\{[^{}]*\}", stripped):
        selectors = [s.strip() for s in block.group(1).split(",") if s.strip()]
        for selector in selectors:
            scoped = selector.startswith(
                ("html.dd-theme", "html.dd-focus", "html.dd-hide-solutions",
                 "html.dd-no-ai", "#dd-colab-toggle"))
            assert scoped, (
                f"unscoped CSS rule {selector!r} — it would restyle every Colab "
                f"page even with the toggles off. Scope it under html.dd-theme "
                f"or html.dd-focus"
            )
            if selector.startswith("html.dd-hide-solutions"):
                assert ".dd-solution" in selector, (
                    f"{selector!r} is under the one default-ON scope but does not "
                    f"name .dd-solution — it would restyle notebooks that have "
                    f"nothing to do with Delta Drills"
                )


def check_gemini_suppression_is_ours_only():
    """Gemini's shadow text is switched off at the editor, and only on our pages.

    Colab ships "Show AI-powered inline completions" ON. On a Delta Drills
    notebook the thing it completes is the answer, so it has to go — but every
    way of doing that has a failure mode that is silent:

    * **CSS alone is worse than nothing.** Monaco's Tab handler accepts the
      suggestion held in the model, not the one on screen. Hide the ghost text
      and Tab still pastes in an answer the learner never saw. So the
      `dd-no-ai` rules in colab_dd.css are only ever a backstop, and the real
      suppression must live in colab_no_ai.js.
    * **It cannot run in the isolated world.** A content script does not see
      `window.monaco`, so the suppressor needs `"world": "MAIN"` in the
      manifest. Drop that key and the file loads, throws nothing, and does
      nothing — inline completions come back with no symptom but the shadow
      text itself.
    * **The two files are wired only by event name.** Rename one side and the
      MAIN-world half never hears the policy, so it stays on its default.
    * **The suppression must not spread.** The browser this is installed in
      opens other people's notebooks too, and disabling a Google feature on
      those is not the extension's business. `dd-no-ai` is therefore gated on
      the page carrying `dd-` anchors.
    """
    focus = _read(os.path.join(HERE, "colab_focus.js"))
    main = _read(os.path.join(HERE, "colab_no_ai.js"))
    manifest = _read(os.path.join(HERE, "..", "manifest.json"))

    assert '"content/colab_no_ai.js"' in manifest, (
        "colab_no_ai.js must be declared in the manifest"
    )
    entry = manifest[manifest.index('"content/colab_no_ai.js"'):]
    entry = entry[:entry.index("]", entry.index("}"))] if "}" in entry else entry
    assert '"world": "MAIN"' in entry, (
        'colab_no_ai.js must be injected with "world": "MAIN" — an isolated-world '
        "content script cannot see window.monaco, so it would silently do nothing"
    )

    # The actual off switch. `inlineSuggest.enabled: false` is the editor option
    # behind Colab's "Show AI-powered inline completions".
    assert "inlineSuggest" in main and "enabled: false" in main, (
        "colab_no_ai.js must turn inlineSuggest off on the editor — hiding the "
        "ghost text in CSS leaves Tab accepting a suggestion nobody can see"
    )
    # New cells mount as the learner scrolls, and Colab re-applies its own option
    # set to editors we have already dealt with. Either hook alone leaves most of
    # a notebook completing.
    for hook in ("onDidCreateEditor", "onDidChangeConfiguration"):
        assert hook in main, f"colab_no_ai.js must hook {hook} or later cells keep completing"
    # ...and re-aligning from a configuration event is only safe because it reads
    # before it writes. An unconditional updateOptions re-fires the event.
    assert "if (on === null) return;" in main, (
        "align must bail on an unreadable option, and must read before it writes "
        "— an unconditional updateOptions loops through onDidChangeConfiguration"
    )
    # MAIN world has no chrome.* at all; a reference would throw on load. Read
    # past the comments, which say so in prose and would match themselves.
    code = re.sub(r"/\*.*?\*/", "", main, flags=re.S)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
    assert "chrome." not in code, (
        "colab_no_ai.js runs in the MAIN world, where chrome.* does not exist"
    )

    for event in ("dd:gemini-off", "dd:gemini-on"):
        assert event in focus and event in main, (
            f"{event} must be spelled the same in both files — they are wired by "
            f"string, so a rename fails as silence"
        )

    assert 'root.classList.toggle("dd-no-ai", suppressAi)' in focus, (
        "the CSS backstop and the editor policy must move together, off one value"
    )
    assert "const suppressAi = ourCells > 0 && !settings.gemini;" in focus, (
        "Gemini may only be suppressed on a notebook carrying dd- anchors — "
        "otherwise the extension silently disables a Google feature on every "
        "Colab page the user ever opens"
    )
    assert "gemini: false" in focus, (
        "Gemini autocomplete must default to OFF; a default that leaks the answer "
        "is not worth having a toggle for"
    )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants,
              check_focus_cannot_blank_the_notebook,
              check_stage_two_pair_survives_focus,
              check_a_lesson_is_a_focus_target,
              check_one_concept_of_a_lesson_is_a_focus_target,
              check_css_is_opt_in,
              check_gemini_suppression_is_ours_only]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
