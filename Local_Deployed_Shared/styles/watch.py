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
    # The level chip + the topbar-seam progress bar (../xp.js).
    "xp.css",
    # The Account-tab test-user roster + the floating switch pill
    # (../test-users-ui.js). Both surfaces are INJECTED by that script, so
    # these class names are the only contract between the two files — nothing
    # in index.html carries them and a missing link fails silently as unstyled
    # markup at the bottom of the Account card.
    "test-users.css",
    # The front door (2026-08-23): the two-arrow welcome fork, the disclosures
    # on "Learn about the App", and the rule that takes the tab strip off the
    # screen in basic mode. 🔴 Its LINK POSITION is asserted in
    # ../watch.py::check_front_door, not here: it has to come after
    # nav-drawer.css or the strip reappears in the drawer on a narrow screen.
    "learn-about.css",
    "account-menu.css",
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
    # Added with the three themes (2026-08-22). --on-accent and the two
    # ladders are the ones that keep the light theme readable: --white means
    # STRONGEST TEXT (near-black in light), so text on an --accent fill has to
    # say --on-accent, and translucent film has to come off a ladder rather
    # than out of a rgba(255,255,255,…) literal.
    "--on-accent", "--accent-text", "--muted-dim", "--border-strong",
    "--surface-2", "--panel", "--panel-deep",
    "--tint-rgb", "--tint-k", "--well-rgb", "--well-k", "--wash-k",
    "--shadow-soft", "--shadow-strong", "--scrim-modal", "--input-bg",
    "--img-mat", "--canvas-bg", "--gold",
    "--ok", "--ok-rgb", "--danger", "--danger-rgb",
    "--warn", "--warn-rgb", "--info", "--info-rgb",
    "--accent-2", "--accent-2-rgb",
    "--code-key", "--code-str", "--code-err",
    # The XP seam gradient (xp.css). Dropping one of these in a single
    # theme paints the progress bar in the previous theme's colours, or in
    # nothing at all — the same silent-drop failure as the rest.
    "--xp-from", "--xp-to", "--xp-glow-rgb",
)

# The three theme blocks in variables.css, by the FULL selector list that opens
# each — a block is matched by its whole list, not by one selector in it, which
# is why two of these carry a comma.
# ":root," is the blue block's first selector — blue is also the unstamped
# default, which is why it and ':root[data-theme="blue"]' share a rule. The
# light block is shared the same way with '#page-arena-notebook', because the
# ARENA notebook page is LessWrong's light reading surface in every theme and
# takes this palette rather than declaring a second copy of it.
THEME_BLOCKS = (
    ':root,\n:root[data-theme="blue"]',
    ':root[data-theme="dark"]',
    ':root[data-theme="light"],\n#page-arena-notebook',
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
def _theme_blocks(variables):
    """Return {theme selector: {token: value}} for the three theme rules."""
    blocks = {}
    for sel in THEME_BLOCKS:
        m = re.search(re.escape(sel) + r"\s*\{(.*?)\n\}", variables, re.DOTALL)
        assert m, f"variables.css is missing the theme block for `{sel}`"
        body = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.DOTALL)
        blocks[sel] = dict(re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", body))
    return blocks


def check_public_api():
    variables = _read(os.path.join(HERE, "variables.css"))
    missing_tokens = [t for t in REQUIRED_TOKENS if not re.search(rf"{re.escape(t)}\s*:", variables)]
    assert not missing_tokens, f"variables.css missing tokens: {missing_tokens}"

    # THE CHECK THAT KEEPS LIGHT MODE READABLE.
    #
    # A token declared in one theme block and forgotten in another does not
    # throw, warn, or render red — the declaration that uses it is simply
    # DROPPED, and the element keeps whatever it inherited. On a `color:` that
    # is white text on a white card: invisible, and invisible in a way no
    # amount of clicking around in the theme you happen to be developing in
    # will show you. So the three blocks must declare exactly the same set.
    blocks = _theme_blocks(variables)
    reference = THEME_BLOCKS[0]
    expected = set(blocks[reference])
    assert expected, f"the `{reference}` block declares no tokens"
    for sel, tokens in blocks.items():
        if sel == reference:
            continue
        missing = sorted(expected - set(tokens))
        extra = sorted(set(tokens) - expected)
        assert not missing, (
            f"theme `{sel}` is missing {missing} — a token defined in only some "
            f"themes silently drops the declarations that use it (white-on-white)"
        )
        assert not extra, (
            f"theme `{sel}` declares {extra}, which `{reference}` does not; "
            f"add them to every theme block or to none"
        )

    # Every REQUIRED_TOKEN must be in all three, not merely somewhere in the
    # file — the regex above would be satisfied by one definition.
    for sel, tokens in blocks.items():
        missing = [t for t in REQUIRED_TOKENS if t != "--drawer-width" and t not in tokens]
        assert not missing, f"theme `{sel}` missing required tokens: {missing}"

    # ── The Account swatches may not drift from the palettes they depict ──
    #
    # components.css's `[data-theme-preview="…"]` rules are the one place in
    # this folder allowed to hardcode colour, and for a good reason: each
    # miniature shows a theme the viewer is NOT in, so var(--bg) would resolve
    # to the ACTIVE theme and paint all three previews identically. The cost of
    # that exemption is that the literals are a hand-copy of variables.css, and
    # a hand-copy silently goes stale — the picker would keep advertising a
    # colour the theme no longer uses, which is worse than no preview because
    # it is confidently wrong.
    #
    # The check is deliberately loose about ROLE: it does not demand that the
    # swatch's card equal `--card` (the blue miniature draws its card with
    # `--surface`, which reads better at 62px). It demands only that every
    # colour in a preview rule still EXISTS somewhere in the theme block it
    # claims to preview. Change a palette and the stale literal fails here.
    components = _read(os.path.join(HERE, "components.css"))
    by_theme = {
        "blue": blocks[THEME_BLOCKS[0]],
        "dark": blocks[THEME_BLOCKS[1]],
        "light": blocks[THEME_BLOCKS[2]],
    }
    stale = []
    for theme, literal in re.findall(
        r'\[data-theme-preview="([a-z]+)"\][^{]*\{([^}]*)\}', components
    ):
        palette = by_theme.get(theme)
        assert palette is not None, (
            f'components.css previews a theme `{theme}` that variables.css does '
            f"not define"
        )
        live = {v.strip().lower() for v in palette.values()}
        for hexval in re.findall(r"#[0-9a-fA-F]{3,8}\b", literal):
            if hexval.lower() not in live:
                stale.append(f"{theme}: {hexval}")
    assert not stale, (
        "Account theme swatch(es) hardcode a colour their theme no longer uses "
        f"— re-copy from variables.css: {sorted(set(stale))}"
    )


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

    # ── NO LIGHT-ON-DARK LITERALS ANYWHERE ────────────────────────────────
    # The whole-app rule that came in with the themes. `#fff` and
    # `rgba(255,255,255,α)` were written when there was exactly one palette
    # and "white" meant "the brightest thing on a navy card". Under the light
    # theme the same literal is white on white — and unlike a broken selector
    # it renders perfectly, it is just invisible, so it survives every test
    # that does not involve a human looking at that specific element in that
    # specific theme.
    #
    # Instead: var(--white) for strongest text (near-BLACK in light),
    # var(--on-accent) for text on an --accent fill, var(--img-mat) for the
    # white matting behind a transparent PNG, or the tint ladder
    # `rgb(var(--tint-rgb) / calc(α * var(--tint-k)))` for translucent film.
    #
    # Scope is every stylesheet the app loads, including the ones outside this
    # folder — the token contract is this folder's public API, and the files
    # most likely to reintroduce a literal are the feature CSS in ../practice
    # and ../targeted-practice, which have no token check of their own.
    #
    # Two exemptions, both narrow and both mechanical rather than by filename:
    #   - a rule whose selector carries [data-theme-preview=…]. Those are the
    #     Account tab's theme swatches: miniatures of themes the viewer is NOT
    #     in, so var(--bg) would resolve to the ACTIVE theme and render all
    #     three previews identically. Literals are the only correct answer.
    #   - a declaration marked /* graph-legend */. The Knowledge Graph legend
    #     has to match node colours the cytoscape canvas paints from JS, which
    #     are a fixed data scale, not chrome.
    # Every spelling of white, not just the two that happened to be in the tree
    # when this check was written. The space-separated `rgb(255 255 255 / a)`
    # form matters most: the tint ladder this migration introduced is written
    # `rgb(var(--tint-rgb) / calc(...))`, so that is now the syntax an author
    # reaches for by imitation — and `rgb(255 255 255 / 0.06)` would have sailed
    # straight past a regex that only knew the comma form. #fff8 / #ffffff80 are
    # the same literal with the alpha folded into the hex.
    light_literal_re = re.compile(
        r"""
          \#fff[0-9a-f]?\b            # #fff, #fff8
        | \#ffffff([0-9a-f]{2})?\b    # #ffffff, #ffffff80
        | rgba?\(                     # rgb()/rgba(), comma OR space separated,
            \s*255\s*[,\s]\s*         # with an optional /alpha
            255\s*[,\s]\s*
            255\s*[,\s/)]
        """,
        re.IGNORECASE | re.VERBOSE,
    )
    css_roots = [HERE, os.path.join(HERE, ".."), os.path.normpath(os.path.join(HERE, "..", "practice")),
                 os.path.normpath(os.path.join(HERE, "..", "targeted-practice"))]
    scanned = set()
    offenders = []
    for root in css_roots:
        for dirpath, _dirs, fnames in os.walk(root):
            # variables.css IS the palette — it is where the literals belong.
            # Vendored/third-party trees are not ours to retheme.
            if any(part in dirpath for part in ("arena-book", "content", "node_modules", "archive")):
                continue
            if root != HERE and os.path.normpath(dirpath) != os.path.normpath(root):
                continue
            for fname in sorted(fnames):
                if not fname.endswith(".css"):
                    continue
                path = os.path.join(dirpath, fname)
                real = os.path.realpath(path)
                if real in scanned or fname == "variables.css":
                    continue
                scanned.add(real)
                selector = ""
                for lineno, line in enumerate(_read(path).split("\n"), 1):
                    stripped = line.strip()
                    if stripped.endswith("{"):
                        selector = stripped[:-1].strip()
                    elif "{" in stripped:
                        selector = stripped.split("{", 1)[0].strip()
                    if not light_literal_re.search(line):
                        continue
                    if "[data-theme-preview=" in selector or "graph-legend" in line:
                        continue
                    rel = os.path.relpath(path, HERE)
                    offenders.append(f"{rel}:{lineno}: {stripped}")
    assert not offenders, (
        "light-on-dark colour literal(s) — invisible in the light theme. Use "
        "var(--white) / var(--on-accent) / var(--img-mat), or the tint ladder "
        "rgb(var(--tint-rgb) / calc(A * var(--tint-k))):\n  " + "\n  ".join(offenders)
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
        "xp.css",
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
