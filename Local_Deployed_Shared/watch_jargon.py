"""watch_jargon.py — the jargon-link layer's checks.

Same contract as every check in watch.py: raise AssertionError to fail. Kept
in its own file the way watch_instructor_graph.py and watch_front_door.py are
(watch.py is already past Modulario's 700-LOC line). watch.py should import
`check_jargon_links` into its namespace and keep it in the __main__ list — a
runner list has dropped checks SILENTLY twice in this repo, so the import and
the list entry are both load-bearing.

What this guards, 2026-08-27: jargon.js underlines ~73 course terms in lesson
prose, and each one opens a panel whose button sends the learner to the
concept that teaches it, maximized, in a new tab. Every way that breaks is
quiet:

  - a glossary entry pointing at a KC id that does not exist focuses NOTHING;
    deltaFocusConceptGraphKc returns on an unknown id by design, so the new
    tab opens on the graph and simply sits there;
  - decorating inside <code> would underline Python identifiers, which reads
    as "you can click your own program";
  - a term linked inside the lesson that teaches it is a link to the page you
    are already reading;
  - the kcLesson map in glossary.js is a COPY of the registry, so it can drift
    without anything failing;
  - the three tags in index.html are what load the feature at all, and two of
    three loads a stylesheet with no behaviour, or an engine with no glossary.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _glossary_terms(src):
    """Pull (term, kc, aliases) out of glossary.js without a JS engine.

    The file is authored by hand as object literals; parsing the shapes we
    assert on is enough, and it keeps this runner dependency-free like every
    other check here.
    """
    terms = []
    for block in re.findall(r"\{\s*\n\s*term:(.*?)\n\s*\},", src, re.S):
        term = re.search(r'^\s*"(.*?)"', block)
        kc = re.search(r'kc:\s*"(.*?)"', block)
        aliases = re.search(r"aliases:\s*\[(.*?)\]", block, re.S)
        if not term or not kc:
            continue
        alias_list = re.findall(r'"(.*?)"', aliases.group(1)) if aliases else []
        terms.append((term.group(1), kc.group(1), alias_list))
    return terms


def check_jargon_links():
    glossary = _read(os.path.join(HERE, "lessons", "glossary.js"))
    engine = _read(os.path.join(HERE, "jargon.js"))
    css = _read(os.path.join(HERE, "styles", "jargon.css"))
    index_html = _read(os.path.join(HERE, "index.html"))
    registry = json.loads(_read(os.path.join(HERE, "lessons", "kc_registry.json")))

    terms = _glossary_terms(glossary)
    assert len(terms) >= 60, (
        "parsed only %d glossary terms — the entry shape this check reads must "
        "have changed, and a parser that silently finds nothing asserts "
        "nothing" % len(terms)
    )

    # ── every destination exists ───────────────────────────────────
    known = {kc["id"] for kc in registry["kcs"]}
    dead = sorted({kc for _, kc, _ in terms if kc not in known})
    assert not dead, (
        "glossary terms point at KC ids that are not in kc_registry.json: %s. "
        "deltaFocusConceptGraphKc returns quietly on an unknown id, so the new "
        "tab would open on the graph and focus nothing at all" % ", ".join(dead)
    )

    # ── one surface form, one meaning ──────────────────────────────
    seen = {}
    for term, kc, aliases in terms:
        for form in [term] + aliases:
            key = form.lower()
            assert key not in seen, (
                'the surface form "%s" is claimed by both %s and %s. Matching '
                "is first-match-wins over one alternation, so the loser would "
                "never light up anywhere" % (form, seen.get(key), kc)
            )
            seen[key] = kc

    # ── the kcLesson copy cannot drift ─────────────────────────────
    # It exists so a hover costs no network request (a runtime-fetched file can
    # be dropped by .vercelignore and the SPA rewrite answers 404 with 200
    # text/html — silent in production). The price of the copy is this check.
    lesson_title = {les["id"]: les["title"] for les in registry["lessons"]}
    kc_lesson = {kc["id"]: lesson_title[kc["lesson"]] for kc in registry["kcs"]}
    mapped = dict(re.findall(r'"([\w.-]+)":\s*\["(.*?)",', glossary))
    assert mapped, "glossary.js no longer carries a kcLesson map"
    for kc_id, title in mapped.items():
        assert kc_id in kc_lesson, (
            "glossary.js's kcLesson map holds %s, which kc_registry.json does "
            "not" % kc_id
        )
        assert kc_lesson[kc_id] == title, (
            'kcLesson says %s is taught in "%s"; the registry says "%s". The '
            "map is a copy of the registry and has drifted"
            % (kc_id, title, kc_lesson[kc_id])
        )
    missing = sorted(set(kc_lesson) - set(mapped))
    assert not missing, (
        "kcLesson is missing %d KCs (%s…) — their popups lose the 'Taught in' "
        "line" % (len(missing), ", ".join(missing[:3]))
    )

    # ── it never decorates code ────────────────────────────────────
    for tag in ('"code"', '"pre"', '"a"', '"button"', '"h2"', '".nb-cell"'):
        assert tag in engine, (
            "%s must stay in jargon.js's SKIP list. A term underlined inside a "
            "code fence is a Python identifier, and .nb-cell is a RUNNABLE "
            "cell — notebook.js builds those out of the same prose region"
            % tag
        )
    assert "NodeFilter.SHOW_TEXT" in engine, (
        "decoration must be text-node only. Rewriting innerHTML would re-parse "
        "the region and hand notebook.js different `pre > code` nodes than the "
        "ones it already turned into cells"
    )

    # ── the two rules that keep it a signpost, not wallpaper ───────
    assert "used.has(rec.kc)" in engine, (
        "only the FIRST mention of a concept per section may be linked — ten "
        "underlined 'tensor's in one lesson is wallpaper"
    )
    assert "rec.kc === selfKc" in engine and "_selfKc" in engine, (
        "a term must not be linked inside the lesson that teaches it: the "
        "button would offer the page the learner is already reading"
    )
    assert "_undecorate" in engine and "root.normalize()" in engine, (
        "re-decorating prose that is already decorated is NOT idempotent: the "
        "existing spans sit in SKIP, so `used` starts empty and the SECOND "
        "mention of a concept gets linked beside the first. Measured at 18 -> "
        "20 -> 21 -> 22 links over three rescans, 7 of them repeats. Any path "
        "back into a decorated region must unwrap it first, and normalize() is "
        "what re-joins the text so a term spanning the seam still matches"
    )
    assert re.search(r"const used = new Set\(\s*\[\.\.\.root\.querySelectorAll", engine), (
        "_decorateScope must seed `used` from the links already in the region, "
        "so the one-link-per-concept rule survives being handed a partly "
        "decorated scope"
    )
    assert 'getElementById("kg-maximize")' in engine, (
        "_selfKc reads the selected node off #kg-maximize's dataset.kc — that "
        "is the only marker lesson-graph.js leaves for which concept the "
        "right-hand pane is showing"
    )
    assert "__ddJargonWrapped" in engine, (
        "the practice screen leaves NO kc marker in the DOM, so LessonGate's "
        "entry points are wrapped to record it. The latch is what stops a "
        "second wrap stacking on the first"
    )

    # ── the boundary class, not \\b ────────────────────────────────
    assert r"(?<![\\w-])" in engine and r"(?![\\w-])" in engine, (
        r"matching must bound on [\w-], not \b. \b treats the hyphen in "
        r'"top-k" and the dot in "einops.reduce" as boundaries, so the match '
        "would land on a fragment and leave the rest of the term as loose text"
    )
    assert "sort((a, b) => b.length - a.length)" in engine, (
        "surface forms must be tried LONGEST FIRST — alternation is "
        'first-match-wins, so "mask" would otherwise beat "boolean mask"'
    )

    # ── the route out, and the route back in ───────────────────────
    assert '"index.html?kc=" + encodeURIComponent(kc) + "&maximize=1"' in engine, (
        "the button's destination is the shareable URL the new tab lands on; "
        "a pre-built view handed across tabs would not survive a reload"
    )
    assert "window.open(" in engine and "location.href = url" in engine, (
        "a blocked popup must still take the learner somewhere — otherwise "
        "the button silently does nothing"
    )
    assert "deltaFocusConceptGraphKc" in engine and "switchTab(" in engine, (
        "the ?kc= landing drives the EXISTING exports. Reimplementing node "
        "selection here would give the app a second opinion about which "
        "concept is on screen"
    )
    assert 'btn.dataset.kc === kc' in engine, (
        "#kg-maximize carrying THIS kc is the ready signal — lesson-graph.js "
        "sets it in renderContent, i.e. only once the node is selected and "
        "its lesson is drawn. Clicking sooner maximizes the wrong concept"
    )

    # ── the panel escapes its overflow ancestors ───────────────────
    assert re.search(r"\.dd-jargon-pop\s*\{[^}]*position:\s*fixed", css), (
        "the panel is parked on <body> and must be position: fixed to escape "
        "the overflow:hidden ancestors between it and the word (the practice "
        "split, the notebook column, the graph's right-hand pane)"
    )
    assert "window.scrollY" not in engine and "window.scrollX" not in engine, (
        "place() positions in VIEWPORT coordinates to match position: fixed. "
        "Adding the scroll offset pushes the panel off screen by exactly the "
        "scroll distance on any scrolled lesson"
    )

    # ── all three tags, in order ───────────────────────────────────
    # This started as all-or-none, so that a peer session mid-flight in
    # index.html could not be handed a red guard for a feature they had not
    # wired yet. That window is closed (the tags landed 2026-08-27), and the
    # loophole was itself the risk: a check that passes when the feature is
    # entirely absent cannot tell "not wired" from "wired correctly".
    css_tag = "styles/jargon.css" in index_html
    glossary_tag = "lessons/glossary.js" in index_html
    engine_tag = re.search(r'src="jargon\.js', index_html) is not None
    assert css_tag and glossary_tag and engine_tag, (
        "index.html must load all three jargon assets (css=%s glossary=%s "
        "engine=%s). Any missing one is silent: no error, no underlines, and "
        "every term stays a plain word"
        % (css_tag, glossary_tag, engine_tag)
    )
    assert index_html.index("lessons/glossary.js") < index_html.index('src="jargon.js'), (
        "lessons/glossary.js must be tagged BEFORE jargon.js: the engine reads "
        "window.DD_GLOSSARY at eval time and returns immediately if it is not "
        "there yet — no error, no underlines"
    )

    # ── the new tab must not take the old one with it ──────────────
    # Matched on the CALL SHAPE, not the word: the fix's own comment has to be
    # able to name "noopener" to explain why it is absent. A guard a doc
    # comment can trip (or satisfy) is guarding prose, not behaviour.
    assert re.search(r'window\.open\([^)]*,\s*"_blank"\s*,', engine) is None, (
        'window.open(url, "_blank", "noopener") returns null from a call that '
        "SUCCEEDED in every browser that honours the feature — indistinguishable "
        "from the null a blocked popup returns. The popup fallback then fired on "
        "the happy path and navigated THIS tab too, losing the page and the "
        "draft the new tab existed to protect. Open without the feature and "
        "sever win.opener instead"
    )
    assert "win.opener = null" in engine, (
        "opening without the noopener feature keeps the new tab's handle on "
        "this window; sever it explicitly"
    )

    # ── in-place mutation of a decorated region ────────────────────
    assert "delete scope.dataset.ddJargon" in engine, (
        "a scope mutated IN PLACE keeps its done-marker and becomes permanently "
        "invisible to decoration — practice/notebook.js rewrites blocks inside "
        ".lesson-body long after it was first decorated. The observer must "
        "clear the marker on the scope the mutation landed in"
    )
    assert "observer.takeRecords()" in engine, (
        "with the observer invalidating scopes, decoration feeds itself: "
        "observer callbacks are MICROTASKS, so the `mutating` latch is always "
        "back to false by the time one runs. Our own records must be dropped "
        "inside the same synchronous block that made them"
    )

    # ── the panel is reachable from the keyboard ───────────────────
    assert "fromKeyboard" in engine and "popGo.focus()" in engine, (
        "the panel is appended to <body>, so Tab from the word walks the rest "
        "of the document before reaching the one button it offers. A keyboard "
        "open must move focus onto that button"
    )
    assert 'setAttribute("aria-expanded"' in engine, (
        "the word is the control for the panel and has to say whether it is open"
    )


if __name__ == "__main__":
    check_jargon_links()
    print("watch_jargon: ok")
