"""watch.py — health checks for concept-graph

Guards the route from a graded answer to a coloured bubble. Every failure this
folder has had was silent: the graph kept drawing 63 bubbles, in plausible
colours, that no amount of practice moved. Nothing threw, nothing 500'd, and
the numbers were priors.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import json
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def check_crosswalk_is_present_and_tiered():
    """The join between concept ids and the atoms belief is held under.

    Everything downstream reads as "this learner has done nothing" without it,
    on both sides: `kc_graph.kc_mastery` returns the bare prior, and the
    browser's own read declines every concept.
    """
    path = os.path.join(HERE, 'kc_atom_crosswalk.json')
    assert os.path.exists(path), (
        'kc_atom_crosswalk.json is missing — every concept will report the '
        'prior. Regenerate: This-Directory-Only/scripts/export_kc_atom_crosswalk.py'
    )
    kcs = (json.loads(_read(path)) or {}).get('kcs') or {}
    # Counted against the registry rather than a literal since 2026-08-30, when
    # the ARENA cut took the graph from 63 concepts to 37 and this check failed
    # on a crosswalk that was in fact complete. What it is really asking is
    # whether the export covered EVERY concept, and only the registry knows how
    # many that is.
    registry = json.loads(_read(os.path.join(HERE, '..', 'lessons', 'kc_registry.json')))
    missing = sorted({kc['id'] for kc in registry['kcs']} - set(kcs))
    # By identity, not by count: a crosswalk carrying enough STALE keys — the
    # retired concepts, say — passes a count test while the concepts actually
    # being served are the ones absent from it.
    assert not missing, (
        f'crosswalk is missing {len(missing)} registry KC(s): '
        + ', '.join(missing[:6]) + (' …' if len(missing) > 6 else '')
        + '. Regenerate: This-Directory-Only/scripts/export_kc_atom_crosswalk.py'
    )
    tiers = {row.get('tier') for row in kcs.values()}
    assert 'measured' in tiers, (
        'no KC is in the `measured` tier — with every concept a topic proxy, '
        'nothing in the graph is ever a per-concept measurement'
    )


def check_the_backend_image_carries_the_crosswalk():
    """The Fly image must COPY it, and .dockerignore must let it through.

    This is the 2026-08-06 bug, verbatim: the folder is gitignored, the
    Dockerfile copied `lessons/` but not this, and production served
    `tier: unmapped` for all 63 concepts for a month. The only symptom was one
    WARNING line. Both halves are asserted because either one alone silently
    drops the file back out of the image.
    """
    dockerfile = _read(os.path.join(REPO, 'This-Directory-Only', 'Dockerfile'))
    assert 'concept-graph/kc_atom_crosswalk.json' in dockerfile, (
        'the Dockerfile no longer COPYs kc_atom_crosswalk.json — the backend '
        'lattice will report every concept unmapped in production while '
        'passing every test locally'
    )
    ignore = _read(os.path.join(REPO, '.dockerignore'))
    assert '!Local_Deployed_Shared/concept-graph/kc_atom_crosswalk.json' in ignore, (
        '.dockerignore excludes everything by default; without the ! line the '
        'COPY above fails the build or ships nothing'
    )


def check_the_graph_reads_the_live_learner_model():
    """Not `localStorage`, which backend mode never writes.

    `saveAdaptiveState` is reached only by the Pyodide paths, so for a
    signed-in learner the persisted key is absent — or holds a stale
    `adaptive_state_guest` blob from before they signed in, which is worse: it
    reports month-old numbers as current.
    """
    src = _read(os.path.join(HERE, 'lesson-graph.js'))
    readiness = src[src.index('const kcReadinessInfo'):src.index('const kcReadiness =')]
    assert '_learnerState()' in readiness, (
        'kcReadinessInfo must read the freshest state (_learnerState), not the '
        'persisted copy — backend mode never writes localStorage'
    )
    assert '_persistedState()' not in readiness, (
        'kcReadinessInfo is back on the persisted state; a signed-in learner '
        "will see a stale guest snapshot or nothing"
    )


def check_the_server_reading_is_preferred():
    """The colour and the gate must come from the same number.

    `/api/practice/kc-lattice` is `kc_graph.kc_report` — the code that decides
    what practice serves next. Recomputing mastery in the browser instead lets
    a bubble disagree with the queue that draws its ring.
    """
    src = _read(os.path.join(HERE, 'lesson-graph.js'))
    assert 'window.kcLatticeReadiness(kc, lattice, s, _decay)' in src, (
        'lesson-graph.js no longer routes the concept read through '
        'kc_lattice_read.js — the server lattice would go unread'
    )
    assert 'window.kcLatticeNote(data)' in src, (
        'the lattice response must pass through kcLatticeNote, or a lattice '
        'that can measure nothing will be drawn as if it could'
    )
    index = _read(os.path.join(REPO, 'Local_Deployed_Shared', 'index.html'))
    # Comments first: the block above these tags SAYS "Before lesson-graph.js",
    # so a raw substring search finds the prose and reports the two script tags
    # as being in the wrong order while they are in the right one. This check
    # had been failing on its own explanation.
    index = re.sub(r'<!--.*?-->', '', index, flags=re.S)
    assert index.index('kc_lattice_read.js') < index.index('lesson-graph.js'), (
        'kc_lattice_read.js must load before lesson-graph.js'
    )


def check_the_measured_tier_is_never_widened():
    """A topic proxy must not be shown as a concept's own measurement.

    40 of the 63 concepts sit on atoms coarser than themselves. Reading one is
    reporting the topic's number under the concept's name, for a dozen
    siblings at once — the exact overclaim the tiering exists to prevent.
    """
    src = _read(os.path.join(HERE, 'kc_lattice_read.js'))
    assert 'row.tier === "measured"' in src, (
        'kc_lattice_read.js must gate the server reading on the measured tier'
    )
    assert 'row.evidenced' in src, (
        'the server\'s covered-weight test must gate the reading too, or one '
        'lightly-weighted atom stands in for a whole concept'
    )


def check_the_topic_reading_cannot_claim_a_measurement():
    """The floor under every reading must stay labelled as a floor.

    `kcTopicReadiness` exists so a concept the server holds a number for is not
    drawn "no estimate" — the case a finished placement test creates for 40 of
    the 63 concepts at once. It is a TOPIC-grain number, though, so three things
    have to hold or it becomes the overclaim the tiering exists to prevent: it
    reports its own source, it reports no concept-specific evidence, and it is
    gated on the server saying it has evidence at all. Drop the last one and a
    learner who has answered nothing gets 63 bubbles at the starting prior,
    which is the 2026-08-06 production bug wearing new clothes.
    """
    src = _read(os.path.join(HERE, 'kc_lattice_read.js'))
    assert 'function kcTopicReadiness' in src, (
        'kcTopicReadiness is gone — concepts the server can measure at topic '
        'grain will fall through to "no estimate" again'
    )
    body = src[src.index('function kcTopicReadiness'):]
    body = body[:body.index('\n  }') + 4]
    assert '!row.evidenced' in body, (
        'kcTopicReadiness must decline a row the server reports as unevidenced, '
        'or a learner who has answered nothing is shown 63 bubbles at the prior'
    )
    assert 'source: "topic"' in body, (
        'the topic-grain reading must name its own source, or the dock will '
        'present a number shared across a topic as this concept\'s own'
    )
    assert 'coveredW: 0' in body, (
        'a topic-grain reading has no concept-specific evidence; a non-zero '
        'coveredW would narrow the confidence band on nothing'
    )


def check_the_topic_reading_sits_below_the_lesson_average():
    """Source precedence, which is the whole claim this file makes.

    A subtopic number rests on the learner's own graded attempts at LESSON
    grain. A topic number can rest on nothing but a placement seed. Reordering
    these swaps a measurement for an estimate without any visible symptom —
    the bubble keeps its colour and only the label underneath changes.
    """
    src = _read(os.path.join(HERE, 'lesson-graph.js'))
    for needle in ('source: "subtopic"', 'window.kcTopicReadiness', '_extrapolated(kc)'):
        assert needle in src, f'lesson-graph.js no longer contains {needle}'
    assert (src.index('source: "subtopic"')
            < src.index('window.kcTopicReadiness')
            < src.index('const ex = _extrapolated(kc)')), (
        'kcReadinessInfo must try the topic reading AFTER the subtopic average '
        'and BEFORE the extrapolation — see kc_lattice_read.js for why'
    )


def check_a_topic_reading_contributes_no_direct_evidence():
    """A topic-grain number must never size a confidence band.

    `kc_graph.kc_mastery` counts an atom as covered once it has a POSTERIOR, and
    placement seeds a posterior onto every atom without one attempt behind it —
    so `covered_w` is 1.0 on all 63 concepts for a learner who has answered
    nothing per concept. Feed that to `directEvidenceN` and nDirect becomes
    1 x 1.0 x specificity, where specificity degrades to 1/siblings whenever the
    browser's crosswalk fetch failed. One single-KC lesson in kc_registry.json
    and a placement seed draws as a measured concept with a tight band.
    """
    src = _read(os.path.join(HERE, 'lesson-graph.js'))
    assert 'info && info.source === "topic"' in src, (
        "_evidence must take a topic reading's own coveredW (0) rather than the "
        "server's covered_w, or a placement seed can be sized like observation"
    )


def check_a_placed_learner_is_not_told_they_answered_nothing():
    """The notice has to be able to say something other than one sentence.

    "No problems answered yet" was hard-coded, and it was false for every
    learner who had just finished the placement test — they answered six to
    fourteen probes and the engine placed and locked the whole lattice on the
    result. It was also permanent: placement produces no per-concept evidence,
    so the condition that hides the notice can never go true from placement
    alone. Both maps must read the text from one place.
    """
    lesson = _read(os.path.join(HERE, 'lesson-graph.js'))
    assert 'const noDataText = ()' in lesson, (
        'lesson-graph.js must choose the cold-start notice text at call time; '
        'a hard-coded string tells a placed learner they have answered nothing'
    )
    assert 'window.kcPlacementStatus' in lesson, (
        'the notice must consult the placement status, or it cannot tell a '
        'placed learner from an untouched one'
    )
    assert 'el.textContent = noDataText();' in lesson, (
        'the notice text must be set on every refresh — the placement status '
        'arrives after the first paint'
    )
    why = _read(os.path.join(HERE, 'why-graph.js'))
    assert 'global.deltaKcNoDataText' in why, (
        'why-graph.js must borrow the notice text from lesson-graph.js, or the '
        'landing map and the Knowledge Graph will disagree about one learner'
    )
    assert 'global.deltaRefreshKcLattice' in why, (
        '"Your mastery" on the landing map needs the server report fetched '
        'through lesson-graph.js; without it that map reads the offline answer '
        'for a signed-in learner'
    )


def check_disabled_concepts_have_an_optional_graph_treatment():
    """Disabled state must remain visible across every colour reading.

    This is display-only: the per-concept API preference owns whether a concept
    is disabled, while one browser preference owns whether that state is faded.
    Keeping the state check outside the mastery-only branch prevents Sections
    and Categories from repainting a disabled concept as active.
    """
    src = _read(os.path.join(HERE, 'lesson-graph.js'))
    assert 'id="kg-set-dim-disabled"' in src, (
        'Settings must expose the graph-wide disabled-concept fade toggle'
    )
    assert 'localStorage.setItem(DIM_DISABLED_KEY, String(dimDisabled))' in src, (
        'the disabled-concept display choice must persist in this browser'
    )
    recolor = src[src.index('const recolor = () =>'):src.index('const _announceReadiness')]
    assert 'disabled ? DISABLED_COLOR : nodeColor(n.id())' in recolor, (
        'disabled concepts must replace every semantic colour with neutral gray'
    )
    gate = src[src.index('const markNextUp = () =>'):src.index('/* ---------------- build')]
    assert 'if (dimDisabled && row.state === "disabled")' in gate, (
        'disabled styling must not be limited to mastery colour mode'
    )
    assert '"opacity": 0.18' in src, (
        'disabled concepts must stay mostly transparent relative to active ones'
    )



# ---------------------------------------------------------------------------
# LESSON DEEP LINKS — /einops/repeat-model. concept-graph/lesson-links.js.
#
# Every failure this feature can have is silent in the same way the colouring
# ones were: the graph still draws, the pane still opens, and the only thing
# that breaks is a URL somebody already sent to somebody else.
# ---------------------------------------------------------------------------

LESSON_LINKS = os.path.join(HERE, 'lesson-links.js')


def _link_families():
    """The FAMILIES allowlist as lesson-links.js actually declares it."""
    src = _read(LESSON_LINKS)
    m = re.search(r'const FAMILIES = Object\.freeze\(\[(.*?)\]\)', src, re.S)
    assert m, 'lesson-links.js no longer declares `const FAMILIES = Object.freeze([...])`'
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def _registry_ids():
    path = os.path.join(HERE, '..', 'lessons', 'kc_registry.json')
    return [kc['id'] for kc in json.loads(_read(path))['kcs']]


def check_every_lesson_family_is_routable():
    """A concept in an unlisted family has no link, and nothing says so.

    FAMILIES is an allowlist because it shares the pathname space with
    solo-route.js, so it cannot be inferred at read time. That makes it the one
    thing here that goes stale: add a concept in a new family and its bubble
    opens, its lesson renders, and the permalink simply does not appear.
    """
    families = _link_families()
    needed = {kc.split('.')[0] for kc in _registry_ids()}
    missing = sorted(needed - families)
    assert not missing, (
        f'lessons/kc_registry.json has {len(missing)} concept family/families that '
        f'lesson-links.js cannot route: {missing}. Add them to FAMILIES in '
        'concept-graph/lesson-links.js, or those lessons get no deep link.'
    )
    stale = sorted(families - needed)
    assert not stale, (
        f'lesson-links.js claims path prefixes no concept uses: {stale}. They take '
        'those pathnames away from every other route for nothing.'
    )


def check_lesson_links_do_not_collide_with_page_slugs():
    """Two route readers, one pathname.

    solo-route.js owns the single-segment page slugs (/notebooks,
    /knowledge-graph). A family named the same as one of them would make
    `/practice/anything` read as a lesson link, and the loser would be whichever
    reader app.js consults second.
    """
    src = _read(os.path.join(HERE, '..', 'solo-route.js'))
    m = re.search(r'const ROUTES = Object\.freeze\(\{(.*?)\}\)', src, re.S)
    assert m, 'solo-route.js no longer declares `const ROUTES = Object.freeze({...})`'
    slugs = set(re.findall(r'^\s*"?([a-z0-9-]+)"?\s*:', m.group(1), re.M))
    clash = sorted(_link_families() & slugs)
    assert not clash, (
        f'these names are BOTH a lesson family and a solo page slug: {clash} — '
        'one of the two routes is unreachable'
    )


def check_every_concept_id_makes_one_clean_path():
    """The URL is derived from the id, so the id has to be URL-shaped.

    Exactly one dot, and nothing on either side of it that a browser would
    escape — a `%20` or a second dot in a concept id would mint a link that
    does not round-trip back through kcFromPath.
    """
    bad = [kc for kc in _registry_ids()
           if not re.fullmatch(r'[a-z0-9]+\.[a-z0-9][a-z0-9-]*', kc)]
    assert not bad, (
        f'these concept ids do not map to a clean /family/concept path: {bad} — '
        'lesson-links.js derives the URL from the id, so the id is the URL'
    )


def check_the_graph_tells_the_router_what_is_selected():
    """Without this call every lesson link points at the same page.

    The address bar following the selection is the whole feature: it is what
    makes the URL in front of a learner the link to what they are reading. The
    hook is one line in a RED file and is exactly the kind of line a refactor
    drops.
    """
    src = _read(os.path.join(HERE, 'lesson-graph.js'))
    assert 'DDLessonLinks?.onSelect?.(' in src, (
        'lesson-graph.js no longer calls DDLessonLinks.onSelect — selecting a '
        'concept stops updating the URL and the permalink stops appearing'
    )
    assert 'DDLessonLinks?.onDeselect?.(' in src, (
        'lesson-graph.js no longer calls DDLessonLinks.onDeselect — dropping the '
        'selection leaves the last concept in the address bar'
    )
    assert 'window.deltaClearConceptGraphSelection' in src, (
        'lesson-graph.js no longer exports deltaClearConceptGraphSelection — Back '
        'off the last lesson URL leaves the concept selected under a bare "/"'
    )


def check_a_two_segment_url_can_still_load_the_app():
    """🔴 THE ONE THAT BREAKS EVERYTHING, AND SILENTLY.

    index.html asks for its 181 scripts and stylesheets by RELATIVE path
    ("app.js", "styles/base.css"), and every route the app had before this one
    was a single segment — so "app.js" resolved to "/app.js" by accident of
    depth. A lesson link has two segments. On /einops/repeat-model every one of
    those relative URLs resolves under /einops/, misses, and comes back as
    index.html through the SPA rewrite: 200 OK, text/html, and a page whose
    console is nothing but `Unexpected token '<'`. The learner sees a blank app
    from a link that looks perfect.

    `<base href="/">` is what pins them, and it has to sit ABOVE the first
    stylesheet and the first script or the ones above it resolve the old way.
    It also fixes the runtime fetches — lessons/kc_registry.json is fetched
    relatively by four files, and those 404 to index.html too.
    """
    html = _read(os.path.join(HERE, '..', 'index.html'))
    m_base = re.search(r'<base\s[^>]*href="/"[^>]*>', html)
    assert m_base, (
        'index.html has no <base href="/"> — every relative script, stylesheet '
        'and fetch breaks on a two-segment lesson URL like /einops/repeat-model, '
        'and the SPA rewrite hands back index.html instead of a 404 so nothing '
        'reports it'
    )
    first_asset = min(
        [m.start() for m in re.finditer(r'<(?:script|link)\s[^>]*(?:src|href)="[a-zA-Z]', html)]
        or [len(html)]
    )
    assert m_base.start() < first_asset, (
        '<base href="/"> sits BELOW the first relative <script>/<link> in '
        'index.html — everything above it still resolves against the URL path'
    )


def check_a_lesson_url_is_not_stamped_as_a_solo_page():
    """The pre-paint stamp in index.html is single-segment only.

    It adds `dd-solo` (chrome hidden, one page, no tabs) for ANY non-empty
    pathname before solo-route.js has had a chance to disagree. Every real solo
    slug is one segment; a lesson link is two and opens the FULL app, so
    without the guard it paints chromeless and then grows a topbar.
    """
    html = _read(os.path.join(HERE, '..', 'index.html'))
    assert 'path.indexOf("/") === -1' in html, (
        'the pre-paint dd-solo stamp in index.html no longer restricts itself to '
        'single-segment paths — a lesson deep link will flash as a chromeless page'
    )


def check_the_router_loads_before_the_app_reads_it():
    """app.js calls DDLessonLinks.read() at boot, not on an event.

    `defer` or a tag below app.js and the read returns undefined: the deep link
    is ignored, the learner lands on Practice, and the URL still says the
    lesson. Silent, and only on a first load from a link — never in the tab
    you are testing in.
    """
    html = _read(os.path.join(HERE, '..', 'index.html'))
    m_links = re.search(r'<script src="concept-graph/lesson-links\.js[^"]*"([^>]*)>', html)
    assert m_links, 'index.html does not load concept-graph/lesson-links.js'
    assert 'defer' not in m_links.group(1) and 'async' not in m_links.group(1), (
        'concept-graph/lesson-links.js is deferred — app.js reads DDLessonLinks '
        'while it parses, so a deferred router is not there yet'
    )
    m_app = re.search(r'<script src="app\.js', html)
    assert m_app and m_links.start() < m_app.start(), (
        'concept-graph/lesson-links.js must be loaded BEFORE app.js'
    )

def _assert_every_check_is_listed(checks):
    """A check this module defines but never calls is worse than no check.

    Twice now a guard has been written, reviewed and left out of the runner
    list, so it passed by never executing. This is NOT itself one of the listed
    checks — dropping it from the list is precisely the mistake it catches, and
    a dropped check does not run to complain about being dropped. It runs from
    the runner, before anything else, whatever the list says.
    """
    defined = {n for n, v in globals().items()
               if n.startswith('check_') and callable(v)}
    missing = sorted(defined - {fn.__name__ for fn in checks})
    assert not missing, (
        f'these checks are defined but never run: {missing} — add them to the '
        '`checks` list at the bottom of this file'
    )


if __name__ == '__main__':
    checks = [
        check_crosswalk_is_present_and_tiered,
        check_the_backend_image_carries_the_crosswalk,
        check_the_graph_reads_the_live_learner_model,
        check_the_server_reading_is_preferred,
        check_the_measured_tier_is_never_widened,
        check_the_topic_reading_cannot_claim_a_measurement,
        check_the_topic_reading_sits_below_the_lesson_average,
        check_a_topic_reading_contributes_no_direct_evidence,
        check_a_placed_learner_is_not_told_they_answered_nothing,
        check_disabled_concepts_have_an_optional_graph_treatment,
        check_every_lesson_family_is_routable,
        check_lesson_links_do_not_collide_with_page_slugs,
        check_every_concept_id_makes_one_clean_path,
        check_a_two_segment_url_can_still_load_the_app,
        check_a_lesson_url_is_not_stamped_as_a_solo_page,
        check_the_graph_tells_the_router_what_is_selected,
        check_the_router_loads_before_the_app_reads_it,
    ]
    try:
        _assert_every_check_is_listed(checks)
    except AssertionError as e:
        print(f"FAIL _assert_every_check_is_listed: {e}", file=sys.stderr)
        sys.exit(1)
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
