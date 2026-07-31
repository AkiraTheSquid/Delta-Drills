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

REQUIRED_PERMISSIONS = {"sidePanel", "tabs", "storage", "system.display"}

# The deploy this extension is built around. The panel frames it and the toolbar
# button opens it — see check_side_panel_shell and check_study_layout.
COLAB_APP_ORIGIN = "https://delta-drills-colab.vercel.app"


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
    # Scripts and stylesheets only. A remote <iframe src> is not an asset the
    # page CSP governs — it is a nested browsing context, and in app.html it is
    # the entire point of the file.
    remote_script = re.compile(r'<script[^>]*\ssrc\s*=\s*"(?:https?:)?//', re.I)
    remote_style = re.compile(r'<link[^>]*\shref\s*=\s*"(?:https?:)?//', re.I)
    for page in ("panel.html", "app.html"):
        with open(os.path.join(HERE, "panel", page), encoding="utf-8") as fh:
            markup = fh.read()
        for what, rx in (("script", remote_script), ("stylesheet", remote_style)):
            assert not rx.search(markup), (
                f"remote {what} in {page} — MV3's CSP blocks it and the page "
                f"renders blank"
            )

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
    """The side panel is the live website, framed — and nothing else.

    MV3 will not accept a URL as `side_panel.default_path`, so the panel has to
    be a local page holding the site in an iframe. Everything about that page is
    load-bearing and easy to erode:

    * If `default_path` swings back to `panel.html`, the student gets the
      hand-built Colab UI instead of the site and nobody will guess why.
    * Without `identity-credentials-get` in the frame's `allow`, Sign in with
      Google renders and then fails — FedCM is denied inside a cross-origin
      frame unless the embedder grants it via Permissions Policy.
    * Any script in this page is the start of a second front end. There is not
      supposed to be one; the site already has tabs, auth and navigation.
    """
    m = _manifest()
    assert m["side_panel"]["default_path"] == "panel/app.html", (
        "side_panel.default_path must be panel/app.html — the page that frames "
        "the live site"
    )
    with open(os.path.join(HERE, "panel", "app.html"), encoding="utf-8") as fh:
        markup = fh.read()

    frame = re.search(r"<iframe\b[^>]*>", markup, re.I | re.S)
    assert frame, "app.html has no <iframe> — the panel IS the website"
    frame = frame.group(0)

    assert COLAB_APP_ORIGIN in frame, (
        f"app.html must frame {COLAB_APP_ORIGIN} — the Colab edition. The normal "
        f"deploy solves in an in-page editor, which is not what an extension "
        f"that sits beside a notebook is for"
    )
    assert "identity-credentials-get" in frame, (
        "the frame needs allow=\"identity-credentials-get …\" or Sign in with "
        "Google fails inside the panel"
    )
    assert not re.search(r"<script\b", markup, re.I), (
        "app.html grew a script — the panel is meant to be the site, not a "
        "front end wrapped around it"
    )


def check_study_layout():
    """The toolbar button tiles two windows — app left, Colab right.

    Four things make that work, and three of them fail silently:

    * `openPanelOnActionClick` MUST be false. With the default, Chrome consumes
      the toolbar click to open the side panel and `action.onClicked` never
      fires — the button simply stops doing anything new, with no error.
    * `action.onClicked` has to be registered at the top level of the worker.
      The service worker is torn down constantly; a listener added inside a
      promise callback is not there when the event arrives.
    * The app pane must be the Colab edition. Tiling the normal deploy next to
      Colab puts an in-page editor beside a notebook — two places to solve the
      same drill.
    * A stray Colab tab has to be moved into the right-hand window. "Open in
      Colab" is a plain link, so without the move the notebook opens beside the
      app on the LEFT and the layout is gone one click in.
    """
    with open(os.path.join(HERE, "background.js"), encoding="utf-8") as fh:
        src = fh.read()

    assert "openPanelOnActionClick: false" in src, (
        "background.js must set openPanelOnActionClick:false — otherwise Chrome "
        "eats the toolbar click for the side panel and action.onClicked never "
        "fires"
    )
    assert re.search(r"^chrome\.action\.onClicked\.addListener", src, re.M), (
        "chrome.action.onClicked must be registered at the top level of the "
        "worker, or it is missing whenever the worker has been torn down"
    )
    assert COLAB_APP_ORIGIN in src, (
        f"the app pane must open {COLAB_APP_ORIGIN} — the edition that routes a "
        f"drill to its notebook"
    )
    assert "chrome.tabs.move" in src, (
        "nothing moves a stray Colab tab into the notebook window — the first "
        "'Open in Colab' click will land the notebook on the left and undo the "
        "layout"
    )
    # The Colab pane receives moved tabs, and Chrome refuses to move a tab into
    # a popup. Only the app pane may be one.
    popups = re.findall(r'type:\s*"popup"', src)
    assert len(popups) == 1, (
        f'expected exactly one type:"popup" window (the app pane); found '
        f"{len(popups)}. The Colab pane must be a normal window or tabs cannot "
        f"be moved into it"
    )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants,
              check_no_shadowed_globals, check_side_panel_shell,
              check_study_layout]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print("PASS extension")
