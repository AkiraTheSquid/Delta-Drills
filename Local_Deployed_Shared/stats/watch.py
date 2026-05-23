"""watch.py — health checks for stats

This folder is JavaScript (loaded as plain <script> tags, not ES modules),
so we use file-existence + text-grep checks rather than Python imports.
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.abspath(os.path.join(HERE, '..'))
INDEX_HTML = os.path.join(SHARED, 'index.html')

JS_FILES = [
    'dom.js', 'data.js', 'weights.js', 'render.js', 'graph.js',
    'predicted-links.js', 'predicted-data.js',
    'predicted-prereqs-temp.js',  # TEMP scaffold — delete with concept graph
    'predicted.js',
    'init.js',
]
SUB_TABS = ['areas', 'graph', 'advanced', 'predicted']


def _read(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ── Import checks ──────────────────────────────
# Verify all expected JS files in this folder exist and are non-empty.
def check_imports():
    for name in JS_FILES:
        path = os.path.join(HERE, name)
        assert os.path.isfile(path), f"missing JS file: {name}"
        assert os.path.getsize(path) > 0, f"empty JS file: {name}"


# ── Public API checks ─────────────────────────
# Verify the functions/symbols other parts of the page rely on are still defined.
def check_public_api():
    expected = {
        'dom.js': ['statsTabs', 'statsPanels', 'statsTableBody'],
        'data.js': ['buildAreas', 'calcDiffMult'],
        'weights.js': ['defaultWeights', 'normalizeWeights'],
        'render.js': ['renderStatsTable', 'renderAdvancedTable'],
        'graph.js': ['renderGraph', 'initGraphControls'],
        'predicted-links.js': [
            'bookHrefForNotebook', 'colabUpstreamHref', 'vsCodeHrefFor',
            'openLinkCell', 'arenaColabOwner', 'ARENA_UPSTREAM_OWNER',
            'ARENA_FORK_REPO', 'VSCODE_LOCAL_ABS_ROOT',
            'encodePathSegments',
        ],
        'predicted-data.js': [
            'computeProblemScore', 'exercisesForProblem', 'compareSectionLabels',
            'subsectionKeyForProblem', 'sectionLabelForProblem',
            'aggregateTopSkill', 'topSkillLabel',
        ],
        'predicted-prereqs-temp.js': [
            'ARENA_PREREQS_TEMP_ENABLED', 'ARENA_PREREQS_TEMP_BY_EXERCISE',
            'ARENA_PREREQS_TEMP_EXERCISES', 'ARENA_PREREQS_TEMP_NOTEBOOK_PATH',
            'getArenaPrereqSubtopicScore', 'isArenaExerciseUnlocked',
            'getNextUnshownUnlockedArenaExercise', 'markArenaExerciseShown',
            'getArenaPrereqsForExercise',
        ],
        'predicted.js': [
            'renderPredictedTable', 'buildPredictedAreas',
            'data-pred-problem-toggle', 'data-pred-exercise-for',
            'data-copy-key', 'stats-col-open', 'copyKeyAttr',
            'data-pred-prereq-toggle', 'data-pred-prereq-for',
            'getExercisesWithTempFallback', 'ARENA_PREREQS_TEMP_ENABLED',
        ],
        'init.js': ['showStatsPanel', 'initStats', 'loadAndRenderStats'],
    }
    for fname, symbols in expected.items():
        src = _read(os.path.join(HERE, fname))
        for sym in symbols:
            assert sym in src, f"{fname}: expected symbol `{sym}` not found"


# ── Invariant checks ──────────────────────────
# Every data-stats-tab button must have a matching data-stats-panel section.
# The Statistics page DOM was extracted out of index.html and is now injected
# at runtime by stats/stats-dom.js, so this check reads that file directly.
def check_invariants():
    dom_path = os.path.join(HERE, 'stats-dom.js')
    assert os.path.isfile(dom_path), f"stats-dom.js missing at {dom_path}"
    html = _read(dom_path)
    # Sanity: the DOM module must still mount the page-statistics container.
    assert 'id="page-statistics"' in html, "stats-dom.js no longer mounts #page-statistics"
    tabs = set(re.findall(r'data-stats-tab="([^"]+)"', html))
    panels = set(re.findall(r'data-stats-panel="([^"]+)"', html))
    assert tabs == panels, (
        f"data-stats-tab / data-stats-panel mismatch — "
        f"tabs only: {tabs - panels}, panels only: {panels - tabs}"
    )
    for name in SUB_TABS:
        assert name in tabs, f"expected sub-tab `{name}` missing from stats-dom.js"

    # The index.html load order MUST place stats-dom.js BEFORE any other
    # stats/ script — otherwise the controllers' getElementById calls
    # return null at IIFE-eval time and the panels silently never render.
    index_html = _read(INDEX_HTML)
    dom_pos = index_html.find('src="stats/stats-dom.js')
    other_pos = index_html.find('src="stats/dom.js')
    assert dom_pos != -1, 'index.html missing <script src="stats/stats-dom.js">'
    assert other_pos != -1, 'index.html missing <script src="stats/dom.js">'
    assert dom_pos < other_pos, (
        "stats/stats-dom.js must load BEFORE stats/dom.js (and all other "
        "stats/ controllers) — they query DOM ids injected by stats-dom.js"
    )


# ── Run all checks ────────────────────────────
if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
