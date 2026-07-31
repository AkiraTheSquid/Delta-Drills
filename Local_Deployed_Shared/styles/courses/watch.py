"""watch.py — health checks for styles/courses

Pure-CSS folder. The parent `../watch.py` already validates existence,
brace balance, link order in index.html, and the no-hex-colors rule
across all five fragments here. This script focuses on the *intent* of
the split — each fragment owns a specific selector family and must not
drift into another fragment's territory.
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

FRAGMENTS = ("page.css", "forkgate.css", "detail.css", "modal.css", "responsive.css")


def _read(name):
    with open(os.path.join(HERE, name), "r", encoding="utf-8") as f:
        return f.read()


# ── Import checks ──────────────────────────────
# CSS has no module system — "import" here means each fragment exists
# and is non-empty. Parent watch already does file-exists, but we want
# this to fail loudly if a fragment is wiped to zero bytes by a bad edit.
def check_imports():
    for f in FRAGMENTS:
        path = os.path.join(HERE, f)
        assert os.path.isfile(path), f"missing fragment: {f}"
        assert os.path.getsize(path) > 0, f"empty fragment: {f}"


# ── Public API checks ─────────────────────────
# The "public API" of each fragment is the selector family it owns,
# referenced by courses.js. Verify each fragment defines at least one
# selector from its assigned family so the JS doesn't render unstyled.
def check_public_api():
    expected_selectors = {
        "page.css": (".courses-page",),
        "forkgate.css": (".fork-gate", ".fork-gate-input", ".fork-gate-submit"),
        "detail.css": (".course-hero", ".course-chapter", ".course-source-link"),
        "modal.css": (".chapter-modal", ".section-item", ".section-number"),
    }
    for fname, selectors in expected_selectors.items():
        text = _read(fname)
        for sel in selectors:
            assert sel in text, f"{fname} missing required selector: {sel}"


# ── Invariant checks ──────────────────────────
# Each fragment owns its prefix family. Drift = a fragment defining a
# selector that belongs in another file. We check this by ensuring no
# fragment defines a class block from another fragment's family root.
def check_invariants():
    block_re = re.compile(r"\.([a-z][\w-]*)\s*[\{\:,]", re.IGNORECASE)

    fragment_owners = {
        "page.css": (re.compile(r"^courses-page$"),),
        "forkgate.css": (re.compile(r"^fork-gate"),),
        "detail.css": (re.compile(r"^courses-detail-view$|^course-(article|hero|intro|chapters|chapter|sources|source-link)"),),
        "modal.css": (re.compile(r"^(chapter-modal|section-)"),),
    }

    for fname, owners in fragment_owners.items():
        text = _read(fname)
        text_no_comments = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        defined = set(block_re.findall(text_no_comments))
        for cls in defined:
            owned_here = any(p.match(cls) for p in owners)
            owned_elsewhere = any(
                p.match(cls)
                for other_fname, other_owners in fragment_owners.items()
                if other_fname != fname
                for p in other_owners
            )
            assert not (owned_elsewhere and not owned_here), (
                f"{fname} defines `.{cls}` which belongs in another fragment "
                f"per the courses-CSS split contract"
            )

    # responsive.css is allowed to override selectors from any fragment, so
    # we only assert it contains a max-width media query (its reason to exist).
    responsive = _read("responsive.css")
    assert "@media (max-width:" in responsive, (
        "responsive.css must contain a max-width media query — that is the "
        "fragment's sole purpose"
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
