"""watch.py — health checks for styles

Pure-CSS folder; no Python imports are meaningful here. Checks verify that
the stylesheets the root UI depends on exist, parse minimally, and that
`variables.css` still defines the design tokens feature stylesheets reference.
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.normpath(os.path.join(HERE, "..", "index.html"))

REQUIRED_CSS = [
    "variables.css",
    "base.css",
    # Part of the base layer: sets color-scheme on :root and skins every
    # scrollbar in the app, so it must load with base.css, not with the
    # feature stylesheets.
    "scrollbars.css",
    "layout.css",
    "components.css",
    "responsive.css",
    "stats.css",
    "arena.css",
    "nav-drawer.css",
    # courses tab is split into a sub-folder so each fragment stays small.
    # Order here mirrors the link order in index.html: page → forkgate →
    # detail → modal → responsive (responsive must be last so its
    # ≤720px overrides win).
    "courses/page.css",
    "courses/forkgate.css",
    "courses/detail.css",
    "courses/modal.css",
    "courses/responsive.css",
]
REQUIRED_TOKENS = (
    "--bg", "--surface", "--card", "--text",
    "--muted", "--accent", "--accent-dark", "--border", "--white",
    # scrollbars.css and nav-drawer.css are token-first, so dropping one of
    # these does not fail loudly — the scrollbar thumb just goes transparent
    # and the drawer collapses to zero width.
    "--scroll-thumb", "--scroll-thumb-hover", "--drawer-width", "--scrim",
)


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── Import checks ──────────────────────────────
# Verify every stylesheet the root UI links from this folder is present and
# non-empty. CSS has no module system, so "import" here means "file exists".
def check_imports():
    missing = [f for f in REQUIRED_CSS if not os.path.isfile(os.path.join(HERE, f))]
    assert not missing, f"missing required stylesheets: {missing}"
    empty = [f for f in REQUIRED_CSS if os.path.getsize(os.path.join(HERE, f)) == 0]
    assert not empty, f"empty stylesheets: {empty}"


# ── Public API checks ─────────────────────────
# The "public API" of this folder is the design-token contract in variables.css
# that every other stylesheet references. If a token is removed, feature CSS
# breaks silently (rules just stop applying). Verify each token is declared.
def check_public_api():
    variables = _read(os.path.join(HERE, "variables.css"))
    missing_tokens = [t for t in REQUIRED_TOKENS if not re.search(rf"{re.escape(t)}\s*:", variables)]
    assert not missing_tokens, f"variables.css missing tokens: {missing_tokens}"


# ── Invariant checks ──────────────────────────
# Structural rules: every CSS file balances braces, index.html links the
# stylesheets in the correct order (variables first, responsive after feature
# stylesheets), and feature CSS does not hardcode hex colors that should use
# the token palette.
def check_invariants():
    for fname in REQUIRED_CSS:
        text = _read(os.path.join(HERE, fname))
        opens, closes = text.count("{"), text.count("}")
        assert opens == closes, f"{fname} has unbalanced braces ({opens} '{{' vs {closes} '}}')"

    if os.path.isfile(INDEX_HTML):
        html = _read(INDEX_HTML)
        order = []
        for fname in REQUIRED_CSS:
            m = re.search(rf'href="styles/{re.escape(fname)}', html)
            if m:
                order.append((m.start(), fname))
        order.sort()
        names = [n for _, n in order]
        if "variables.css" in names:
            assert names[0] == "variables.css", (
                f"variables.css must be the first styles/* link in index.html; got {names}"
            )
        if "responsive.css" in names and "components.css" in names:
            assert names.index("responsive.css") > names.index("components.css"), (
                "responsive.css must be linked after components.css so its rules win"
            )
        # Every courses/* fragment must be linked before responsive.css so the
        # global narrow-viewport overrides win over the courses-specific ones.
        # The courses/responsive.css fragment scopes its overrides to courses
        # selectors only, so the relative order between courses/responsive.css
        # and the global responsive.css does not matter — both must come after
        # the four non-responsive courses fragments.
        if "responsive.css" in names:
            for fname in [n for n in names if n.startswith("courses/")]:
                if fname == "courses/responsive.css":
                    continue
                assert names.index(fname) < names.index("responsive.css"), (
                    f"{fname} must be linked before responsive.css"
                )
            for fname in ("courses/page.css", "courses/forkgate.css", "courses/detail.css", "courses/modal.css"):
                if fname in names and "courses/responsive.css" in names:
                    assert names.index(fname) < names.index("courses/responsive.css"), (
                        f"{fname} must be linked before courses/responsive.css so its overrides win"
                    )

        # scrollbars.css is base-layer, not feature-layer: it sets
        # `color-scheme` on :root, which the engine reads for the whole
        # document, and skins every scroller in the app. Linked late it would
        # sit after stylesheets that assume the dark chrome is already there.
        if "scrollbars.css" in names and "layout.css" in names:
            assert names.index("scrollbars.css") < names.index("layout.css"), (
                "scrollbars.css belongs with the base layer — link it before layout.css"
            )
        if "nav-drawer.css" in names and "responsive.css" in names:
            assert names.index("nav-drawer.css") < names.index("responsive.css"), (
                "nav-drawer.css is a feature stylesheet — link it before responsive.css"
            )

        # The three hooks nav-drawer.css styles and nav-drawer.js drives. Every
        # rule in that stylesheet is scoped to one of them, so losing a hook
        # does not throw — the menu simply stops existing below 900px, and the
        # tab strip goes back to being a sideways-scrolling sliver in the
        # extension's side panel, which is the bug the drawer was built for.
        for hook in ('id="nav-toggle"', 'id="nav-drawer"', 'id="nav-scrim"'):
            assert hook in html, f"index.html is missing the nav drawer hook {hook}"
        # #nav-drawer must stay EMPTY in the markup. The strip it holds below
        # 900px is the live <nav class="tabs">, moved there by nav-drawer.js —
        # a second copy written into the aside would be outside the NodeLists
        # app.js captured, so its tabs would not switch pages and would show
        # auth-only tabs to a guest. See ../nav-drawer.js.
        drawer_markup = re.search(r'<aside[^>]*id="nav-drawer".*?</aside>', html, re.DOTALL)
        assert drawer_markup, 'index.html: #nav-drawer must be an <aside>…</aside>'
        assert 'class="tabs"' not in drawer_markup.group(0), (
            "#nav-drawer must not contain its own .tabs — nav-drawer.js moves the "
            "real strip in, and a copy is invisible to app.js's static NodeLists"
        )

    # Newly-added feature stylesheets must reach for design tokens, not raw hex
    # colors. Legacy files (arena.css, stats.css) predate this rule and are not
    # enforced here — only files added under the token-first convention.
    # rgba() neutral overlays in modal.css (backdrop scrim, drop shadow) are
    # not brand surfaces and have no token equivalent; the regex matches '#'
    # literals only, so rgba/hsla pass through.
    hex_re = re.compile(r"#[0-9a-fA-F]{3,8}\b")
    token_first_files = (
        "scrollbars.css",
        "nav-drawer.css",
        "courses/page.css",
        "courses/forkgate.css",
        "courses/detail.css",
        "courses/modal.css",
        "courses/responsive.css",
    )
    for fname in token_first_files:
        text = _read(os.path.join(HERE, fname))
        stripped = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        offenders = hex_re.findall(stripped)
        assert not offenders, (
            f"{fname} hardcodes hex color(s) {offenders}; use a var(--token) from variables.css"
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
