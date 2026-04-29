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

JS_FILES = ['dom.js', 'data.js', 'weights.js', 'render.js', 'graph.js', 'predicted.js', 'init.js']
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
        'predicted.js': ['renderPredictedTable', 'buildPredictedAreas'],
        'init.js': ['showStatsPanel', 'initStats', 'loadAndRenderStats'],
    }
    for fname, symbols in expected.items():
        src = _read(os.path.join(HERE, fname))
        for sym in symbols:
            assert sym in src, f"{fname}: expected symbol `{sym}` not found"


# ── Invariant checks ──────────────────────────
# Every data-stats-tab button must have a matching data-stats-panel section.
# This is the invariant called out in README.md.
def check_invariants():
    assert os.path.isfile(INDEX_HTML), f"index.html not found at {INDEX_HTML}"
    html = _read(INDEX_HTML)
    tabs = set(re.findall(r'data-stats-tab="([^"]+)"', html))
    panels = set(re.findall(r'data-stats-panel="([^"]+)"', html))
    assert tabs == panels, (
        f"data-stats-tab / data-stats-panel mismatch — "
        f"tabs only: {tabs - panels}, panels only: {panels - tabs}"
    )
    for name in SUB_TABS:
        assert name in tabs, f"expected sub-tab `{name}` missing from index.html"


# ── Run all checks ────────────────────────────
if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
