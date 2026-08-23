"""watch.py — health checks for targeted-practice

Frontend-only module (vanilla JS + CSS), so the checks are text-grep over
the source files rather than Python imports. Verifies:
  - Both source files exist and aren't empty.
  - The JS module references the DOM ids it queries (so an HTML rename
    breaks the check before it breaks the page).
  - The CSS keeps the `.hidden` override that prevents `.tp-card`'s
    `display: flex` from defeating the global `.hidden { display: none }`.
  - The global banner DOM lives in index.html, outside <main>, with the
    ids the JS controller wires.
  - The script tag for this module loads AFTER its hard dependencies
    (app.js, arena/manifest.js, arena/exercises.js, stats/predicted-links.js).
"""
import sys
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.abspath(os.path.join(HERE, '..'))


def _read(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ── Import checks ──────────────────────────────
def check_imports():
    js = os.path.join(HERE, 'targeted-practice.js')
    css = os.path.join(HERE, 'targeted-practice.css')
    readme = os.path.join(HERE, 'README.md')
    for p in (js, css, readme):
        assert os.path.exists(p), f"missing file: {p}"
        assert os.path.getsize(p) > 200, f"file unexpectedly small: {p}"


# ── Public API checks ─────────────────────────
def check_public_api():
    js = _read(os.path.join(HERE, 'targeted-practice.js'))
    # IDs the controller queries — if any of these disappear, the module
    # silently no-ops because the top-of-file `if (!root) return` guard
    # bails on a missing #page-targeted-practice.
    required_ids = [
        'page-targeted-practice',
        'tp-search-input',
        'tp-search-clear',
        'tp-results-list',
        'tp-results-hint',
        'tp-selected-list',
        'tp-selected-count',
        'tp-selected-empty',
        'tp-selected-title',
        'tp-submit-btn',
        'tp-back-btn',
        'tp-banner',
        'tp-banner-meta',
        'tp-banner-end',
    ]
    for el_id in required_ids:
        assert el_id in js, f"controller no longer references id #{el_id}"

    # Core functions the controller relies on (renamed = silent breakage).
    for fn in ('startSession', 'endSession', 'enterReviewMode', 'resetToSearch', 'simulateAfter'):
        assert re.search(rf'\b{fn}\b', js), f"missing function: {fn}"


# ── Invariant checks ──────────────────────────
def check_invariants():
    css = _read(os.path.join(HERE, 'targeted-practice.css'))
    js = _read(os.path.join(HERE, 'targeted-practice.js'))
    index_html = _read(os.path.join(SHARED, 'index.html'))

    # The .hidden override — must include all card-style classes that set
    # display: flex (otherwise the global .hidden rule is overridden).
    for sel in ('.tp-card.hidden', '.tp-submit-btn.hidden', '.tp-back-btn.hidden', '.tp-banner.hidden'):
        assert sel in css, f"missing hidden-override rule for {sel}"

    # Global banner DOM must live OUTSIDE <main> in index.html so it
    # persists across tab switches. Cheap proxy: banner element must appear
    # before the first real <main ...> tag (skip HTML comments — they
    # contain a literal "<main>" in the banner doc-string).
    banner_pos = index_html.find('id="tp-banner"')
    m = re.search(r'<main\s', index_html)
    main_pos = m.start() if m else -1
    assert banner_pos != -1, "tp-banner is missing from index.html"
    assert main_pos != -1, "no <main ...> in index.html (unexpected)"
    assert banner_pos < main_pos, "tp-banner moved inside/after <main> — must stay above all pages"

    # Tab visibility contract (guest-first rework replaced authRequiredTabs):
    # the tab button carries .auth-only, and app.js#guestVisibleTabs whitelists
    # "targeted-practice" so guests can still reach it. Both halves must hold —
    # dropping the whitelist entry hides the tab from guests; dropping the
    # class breaks the logged-in visibility toggle.
    app_js = _read(os.path.join(SHARED, 'app.js'))
    assert 'guestVisibleTabs' in app_js and '"targeted-practice"' in app_js, (
        "app.js#guestVisibleTabs no longer includes targeted-practice"
    )
    # Matched on class TOKENS, not on the literal attribute string: the tab
    # picked up a `has-info` class when the ⓘ dots landed (2026-08-07) and lost
    # it again when they were deleted (2026-08-23), and an `class="tab
    # auth-only"` exact match broke on both. The contract is what matters:
    # every element tagged data-tab="targeted-practice" carries auth-only, or
    # app.js's .auth-only NodeList leaves a guest looking at a tab that 403s.
    tp_tags = [
        m.group(0)
        for m in re.finditer(r'<button[^>]*>', index_html)
        if 'data-tab="targeted-practice"' in m.group(0)
    ]
    def _class_tokens(tag):
        m = re.search(r'class="([^"]*)"', tag)
        return m.group(1).split() if m else []

    tp_classes = [_class_tokens(tag) for tag in tp_tags]
    assert any('tab' in c for c in tp_classes), (
        "targeted-practice tab button is missing from index.html"
    )
    assert all('auth-only' in c for c in tp_classes), (
        "targeted-practice tab button lost its auth-only class"
    )

    # Script load order — controller must load AFTER all hard deps.
    def script_offset(needle):
        m = re.search(rf'<script\s+src="{re.escape(needle)}', index_html)
        return m.start() if m else -1

    self_pos = script_offset('targeted-practice/targeted-practice.js')
    assert self_pos != -1, "targeted-practice.js not loaded in index.html"
    for dep in ('app.js', 'arena/manifest.js', 'arena/exercises.js', 'stats/predicted-links.js'):
        dep_pos = script_offset(dep)
        assert dep_pos != -1, f"hard dep not loaded in index.html: {dep}"
        assert dep_pos < self_pos, f"load-order broken: {dep} must come before targeted-practice.js"

    # README must be filled (no leftover template marker).
    readme = _read(os.path.join(HERE, 'README.md'))
    assert 'modulario:template' not in readme, "README.md still has the template marker"


# ── Run all checks ────────────────────────────
if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
