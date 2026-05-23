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
    "layout.css",
    "components.css",
    "responsive.css",
    "stats.css",
    "arena.css",
    # courses tab is split into a sub-folder so each fragment stays small.
    # Order here mirrors the link order in index.html: list → include →
    # detail → modal → responsive (responsive must be last so its
    # ≤720px overrides win).
    "courses/list.css",
    "courses/include.css",
    "courses/detail.css",
    "courses/modal.css",
    "courses/responsive.css",
]
REQUIRED_TOKENS = (
    "--bg", "--surface", "--card", "--text",
    "--muted", "--accent", "--accent-dark", "--border", "--white",
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
            for fname in ("courses/list.css", "courses/include.css", "courses/detail.css", "courses/modal.css"):
                if fname in names and "courses/responsive.css" in names:
                    assert names.index(fname) < names.index("courses/responsive.css"), (
                        f"{fname} must be linked before courses/responsive.css so its overrides win"
                    )

    # Newly-added feature stylesheets must reach for design tokens, not raw hex
    # colors. Legacy files (arena.css, stats.css) predate this rule and are not
    # enforced here — only files added under the token-first convention.
    # rgba() neutral overlays in modal.css (backdrop scrim, drop shadow) are
    # not brand surfaces and have no token equivalent; the regex matches '#'
    # literals only, so rgba/hsla pass through.
    hex_re = re.compile(r"#[0-9a-fA-F]{3,8}\b")
    token_first_files = (
        "courses/list.css",
        "courses/include.css",
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
