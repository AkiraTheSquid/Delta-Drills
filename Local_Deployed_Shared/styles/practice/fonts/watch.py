"""watch.py — health checks for fonts

The failure this folder exists to prevent is a SILENT one. A missing or corrupt
face does not error: the browser falls through the stack to Palatino and the
page still looks like a serif page, so a heading set in the wrong font reads as
a styling opinion rather than as a broken asset. Every check here turns one of
those silent fallbacks into a failed run.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHEET = os.path.join(HERE, "..", "arena-notebook.css")

# The faces, and the weight/style each one is declared for. Sizes are the real
# upstream ones; the floor is what separates a font from a 404 page saved under
# a .woff name (GitHub Pages answers a missing path with 200 + HTML).
FACES = {
    "et-book-roman-line-figures.woff": ("400", "normal"),
    "et-book-bold-line-figures.woff": ("700", "normal"),
    "et-book-display-italic-old-style-figures.woff": ("400", "italic"),
}
MIN_BYTES = 20_000


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def check_imports():
    """Every declared face is on disk, and is actually a WOFF."""
    for name in FACES:
        path = os.path.join(HERE, name)
        assert os.path.exists(path), f"missing font file: {name}"
        size = os.path.getsize(path)
        # A saved 404 page is ~9 KB of HTML and passes an existence check.
        assert size >= MIN_BYTES, f"{name} is only {size} bytes — a saved error page?"
        with open(path, "rb") as f:
            magic = f.read(4)
        assert magic == b"wOFF", f"{name} is not a WOFF (magic {magic!r})"

    assert os.path.exists(os.path.join(HERE, "LICENSE")), (
        "ET Book's MIT LICENSE must travel with the fonts it covers"
    )


def check_public_api():
    """The stylesheet declares all three faces, against the names on disk."""
    css = _read(SHEET)
    blocks = re.findall(r"@font-face\s*\{(.*?)\}", css, re.DOTALL)
    declared = {}
    for body in blocks:
        url = re.search(r'url\("fonts/([^"]+)"\)', body)
        if not url:
            continue
        weight = re.search(r"font-weight:\s*([^;]+);", body)
        style = re.search(r"font-style:\s*([^;]+);", body)
        family = re.search(r"font-family:\s*([^;]+);", body)
        assert family and family.group(1).strip() == '"ETBookRoman"', (
            "the family must stay ETBookRoman — it is LessWrong's own name for "
            "this face, which is what makes their measurements comparable to ours"
        )
        declared[url.group(1)] = (
            weight.group(1).strip() if weight else None,
            style.group(1).strip() if style else None,
        )

    for name, expected in FACES.items():
        assert name in declared, f"arena-notebook.css never declares {name}"
        assert declared[name] == expected, (
            f"{name} is declared {declared[name]}, expected {expected} — a face "
            f"declared at the wrong weight is loaded and then never selected"
        )

    for name in declared:
        assert name in FACES, f"arena-notebook.css points at {name}, which is not in this folder"


def check_invariants():
    """No commercial face, and no font fetched from someone else's account."""
    css = _read(SHEET)
    # Comments FIRST. The stylesheet documents at length why it must not reach
    # for Typekit or warnock-pro, so scanning the raw text finds the warning and
    # reports it as the violation it is warning about.
    live = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)

    # 🔴 The mistake that would work locally and fail in production, while
    # spending LessWrong's Adobe licence on the way.
    assert "typekit" not in live.lower(), (
        "arena-notebook.css links a Typekit stylesheet — that is LessWrong's own "
        "font kit, not ours. Self-host, or fall back through --lw-serif."
    )
    assert "warnock" not in live.lower(), (
        "warnock-pro is a commercial Adobe face we have no licence to serve"
    )

    # The fallback chain is what carries the body text, so losing it is losing
    # the design on every machine without ET Book cached.
    chain = re.search(r"--lw-serif:\s*([^;]+);", live)
    assert chain, "arena-notebook.css no longer defines --lw-serif"
    for expected in ("Palatino", "Georgia", "serif"):
        assert expected in chain.group(1), f"--lw-serif dropped {expected}"

    assert "--lw-header:" in live and "ETBookRoman" in live, (
        "--lw-header must still put ETBookRoman in front of --lw-serif"
    )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
