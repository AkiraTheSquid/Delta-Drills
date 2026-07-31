"""watch.py — health checks for stats

This folder no longer backs a Statistics tab (removed 2026-07-31). What
remains are three files the REST of the app depends on; they keep the
`stats/` path only so their consumers' <script src=...> tags stay valid.

This folder is JavaScript (loaded as plain <script> tags, not ES modules),
so we use file-existence + text-grep checks rather than Python imports.
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.abspath(os.path.join(HERE, '..'))
INDEX_HTML = os.path.join(SHARED, 'index.html')

JS_FILES = [
    'weights.js',
    'predicted-links.js',
    'predicted-prereqs-temp.js',  # TEMP scaffold — delete with concept graph
]

# Files deleted with the Statistics tab. If one reappears, either the tab is
# being rebuilt (fine — update this list and the guards in ../watch.py) or a
# revert half-landed (not fine — the index.html <script> tags are gone, so the
# file would sit dead on disk).
REMOVED_FILES = [
    'dom.js', 'data.js', 'render.js', 'graph.js',
    'predicted-data.js', 'predicted.js', 'init.js', 'stats-dom.js',
]


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
    for name in REMOVED_FILES:
        assert not os.path.isfile(os.path.join(HERE, name)), (
            f"{name} is back but index.html no longer loads it — it belonged "
            "to the Statistics tab, removed 2026-07-31"
        )


# ── Public API checks ─────────────────────────
# Verify the functions/symbols other parts of the page rely on are still defined.
def check_public_api():
    expected = {
        # practice/adaptive.js reads buildEffectiveWeightsFromSubtopics;
        # practice/questions.js reads isSubtopicEnabled.
        'weights.js': [
            'defaultWeights', 'normalizeWeights', 'loadWeights', 'saveWeights',
            'buildEffectiveWeightsFromSubtopics', 'isSubtopicEnabled',
        ],
        # colabUpstreamHref is consumed by courses.js, courses-fork-gate.js,
        # practice/ui.js, practice/drills-catalog.js, practice/arena-unlock.js
        # and targeted-practice/targeted-practice.js.
        'predicted-links.js': [
            'colabUpstreamHref', 'arenaColabOwner', 'drillsColabOwner',
            'ARENA_UPSTREAM_OWNER', 'ARENA_FORK_REPO', 'encodePathSegments',
        ],
        # Read by practice/arena-unlock.js, practice/drills-catalog.js and
        # targeted-practice/targeted-practice.js.
        'predicted-prereqs-temp.js': [
            'ARENA_PREREQS_TEMP_ENABLED', 'ARENA_PREREQS_TEMP_BY_EXERCISE',
            'ARENA_PREREQS_TEMP_EXERCISES', 'ARENA_PREREQS_TEMP_NOTEBOOK_PATH',
            'getArenaPrereqSubtopicScore', 'isArenaExerciseUnlocked',
            'getNextUnshownUnlockedArenaExercise', 'markArenaExerciseShown',
            'getArenaPrereqsForExercise',
        ],
    }
    for fname, symbols in expected.items():
        src = _read(os.path.join(HERE, fname))
        for sym in symbols:
            assert sym in src, f"{fname}: expected symbol `{sym}` not found"


# ── Invariant checks ──────────────────────────
# The Statistics page is gone: nothing in this folder may mount it, and
# index.html must still load the three survivors.
def check_invariants():
    for name in JS_FILES:
        src = _read(os.path.join(HERE, name))
        assert 'id="page-statistics"' not in src, (
            f"{name} mounts #page-statistics — the Statistics tab was removed"
        )

    index_html = _read(INDEX_HTML)
    for name in JS_FILES:
        assert f'src="stats/{name}' in index_html, (
            f'index.html missing <script src="stats/{name}"> — its consumers '
            "(courses / practice / targeted-practice) read its globals"
        )
    for name in REMOVED_FILES:
        assert f'src="stats/{name}' not in index_html, (
            f'index.html still loads stats/{name}, which no longer exists'
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
