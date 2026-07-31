"""watch.py — health checks for extension

Chrome reports a broken unpacked extension as one opaque line in
chrome://extensions, usually after you have already clicked Load unpacked. These
checks catch the failures that produce that line, from the shell.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

REQUIRED_PERMISSIONS = {"sidePanel", "tabs", "storage"}


def _manifest():
    with open(os.path.join(HERE, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def check_imports():
    """The manifest parses and declares MV3."""
    m = _manifest()
    assert m.get("manifest_version") == 3, "manifest_version must be 3"
    assert m.get("version"), "manifest needs a version"


def check_public_api():
    """Every file the manifest points at exists.

    A dangling path is the single most common reason Load unpacked fails.
    """
    m = _manifest()
    refs = [m["background"]["service_worker"], m["side_panel"]["default_path"]]
    refs += [j for cs in m.get("content_scripts", []) for j in cs.get("js", [])]
    missing = [r for r in refs if not os.path.exists(os.path.join(HERE, r))]
    assert not missing, f"manifest references missing files: {missing}"

    perms = set(m.get("permissions", []))
    lost = REQUIRED_PERMISSIONS - perms
    assert not lost, f"missing permissions: {sorted(lost)}"


def check_invariants():
    """The two rules that are easy to break by accident.

    1. MV3's CSP blocks remote script, so an external <script src> or a
       stylesheet from a CDN turns into a blank panel with a console error.
    2. No mastery math in the client. Prerequisite gating, BKT, FIRe and decay
       are backend-owned; a threshold constant appearing here would mean two
       sources of truth for what a learner knows.
    """
    for page in ("panel.html", "app.html"):
        with open(os.path.join(HERE, "panel", page), encoding="utf-8") as fh:
            markup = fh.read()
        remote = re.findall(r'(?:src|href)\s*=\s*"(?:https?:)?//[^"]+"', markup)
        assert not remote, f"remote asset in {page} (MV3 CSP blocks it): {remote}"

    banned = re.compile(
        r"\b(P_TRANSIT|P_GUESS|P_SLIP|UNLOCK_THRESHOLD|MASTERY_THRESHOLD|HALF_LIFE_DAYS)\b"
    )
    for root, _dirs, files in os.walk(HERE):
        if "node_modules" in root:
            continue
        for name in files:
            if not name.endswith(".js"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as fh:
                hit = banned.search(fh.read())
            assert not hit, (
                f"{os.path.relpath(path, HERE)} names a backend mastery constant "
                f"({hit.group(0)}) — that math stays server-side"
            )


def check_no_shadowed_globals():
    """panel.html loads four classic scripts into ONE global lexical scope.

    `api.js` declares `const api`; `navigate.js` declares `const slugKc`. If
    `panel.js` then writes `const {api} = window.DD`, that is a redeclaration and
    the entire file throws a SyntaxError at parse time — no handlers wired, no
    view activated, so the panel renders as a blank page with a dead ⚙. It looks
    like a CSS or manifest fault and is neither, and `node --check` on the files
    individually cannot see it. Hence a check.
    """
    with open(os.path.join(HERE, "panel", "panel.js"), encoding="utf-8") as fh:
        src = fh.read()
    for name in ("api", "tab", "notebooks", "slugKc", "whoIsOpen", "jumpTo",
                 "paintNotebook", "renderNotebookList"):
        bare = re.search(rf"^\s*{name}\s*[,}}]", src, re.M)
        assert not bare, (
            f"panel.js destructures `{name}` under its own name — that redeclares "
            f"the const api.js/navigate.js already made and kills the whole file. "
            f"Alias it (`{name}: dd{name[0].upper()}{name[1:]}`)."
        )


def check_side_panel_shell():
    """The default panel is the shell that hosts the real web app.

    The app is a remote page, and MV3 will not accept a URL as
    `side_panel.default_path`, so the panel has to be a local page holding an
    iframe. If the default_path ever swings back to `panel.html`, the student
    gets the hand-built Colab UI instead of the app and nobody will guess why.
    """
    m = _manifest()
    assert m["side_panel"]["default_path"] == "panel/app.html", (
        "side_panel.default_path must be panel/app.html — the shell that frames "
        "the deployed web app"
    )
    with open(os.path.join(HERE, "panel", "app.js"), encoding="utf-8") as fh:
        src = fh.read()
    assert "delta-drills.vercel.app" in src, "app.js lost its default app address"
    # The frame is another origin; reaching into it cannot work and would mean
    # the shell had quietly become a second copy of the app.
    for reach in ("contentDocument", "contentWindow"):
        assert reach not in src, f"app.js touches {reach} — the frame is cross-origin"


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants,
              check_no_shadowed_globals, check_side_panel_shell]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print("PASS extension")
