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
    * Any script beyond `app.js` is the start of a second front end. There is
      not supposed to be one; the site already has tabs, auth and navigation.
      `app.js` is the sole exception and earns it by not being a front end: it
      renders nothing and only forwards the site's "open this notebook" to
      `chrome.tabs`, which a cross-origin frame cannot do for itself. Anything
      inline, or a second file, is the drift this check exists to catch.
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
    scripts = re.findall(r"<script\b([^>]*)>", markup, re.I)
    for attrs in scripts:
        src = re.search(r'src\s*=\s*"([^"]+)"', attrs, re.I)
        assert src, (
            "app.html has an inline script — MV3's CSP blocks it anyway, and the "
            "panel is meant to be the site, not a front end wrapped around it"
        )
        assert src.group(1) == "app.js", (
            f"app.html loads {src.group(1)} — the only script this page may load "
            f"is app.js, the notebook-opening bridge"
        )

    # The bridge drives tab navigation from a `message` event, which any frame
    # can fire. Without both halves of the sender check plus a URL allowlist,
    # this page is an open redirect holding the extension's `tabs` permission.
    with open(os.path.join(HERE, "panel", "app.js"), encoding="utf-8") as fh:
        bridge = fh.read()
    assert "event.origin !== APP_ORIGIN" in bridge, (
        "app.js must reject messages from any origin but the framed site"
    )
    assert "event.source !== frame.contentWindow" in bridge, (
        "app.js must reject messages from anything but the frame it embeds — "
        "origin alone lets any same-origin popup drive the tab"
    )
    assert "colab\\.research\\.google\\.com" in bridge, (
        "app.js must refuse to navigate anywhere but Colab"
    )

    # The second message the bridge forwards. It only unhides a cell, so it is
    # not the open-redirect risk the URL one is — but a payload that stopped
    # being a bare problem number would be reaching into the page with whatever
    # the content script does with it.
    assert "dd:reveal-solution" in bridge, (
        "app.js no longer forwards dd:reveal-solution — the notebook's answer "
        "cell would stay hidden after the learner said how it went, with no way "
        "to open it but the toggle"
    )
    with open(os.path.join(HERE, "content", "colab_focus.js"), encoding="utf-8") as fh:
        focus = fh.read()
    assert "/^\\d+$/.test(n)" in focus, (
        "colab_focus.js must validate the revealed problem as a number before "
        "using it — the panel forwards whatever the page sent"
    )


def check_button_opens_the_panel():
    """The toolbar button opens the SIDE PANEL, and the worker does nothing else.

    On 2026-07-31 this briefly opened two tiled `chrome.windows` instead — app
    left, Colab right — to satisfy "app on the left". It worked and it was
    wrong: separate windows are not a side pane, they pop out of the browser
    rather than splitting it, and the user has to manage two windows for one
    task. The panel is the design.

    Which SIDE the panel sits on is a Chrome setting (Settings → Appearance →
    "Side panel position"), and its width is a drag. Neither is reachable from
    an extension — `chrome.sidePanel` has `setOptions` and `setPanelBehavior`
    and nothing else. So a worker that starts creating or moving windows is
    always re-implementing a preference that already exists, and this check is
    what stops that from creeping back.
    """
    with open(os.path.join(HERE, "background.js"), encoding="utf-8") as fh:
        src = fh.read()

    assert "openPanelOnActionClick: true" in src, (
        "background.js must set openPanelOnActionClick:true — that is what makes "
        "the toolbar button open the side panel"
    )
    for api in ("chrome.windows.create", "chrome.windows.update", "chrome.tabs.move"):
        assert api not in src, (
            f"background.js calls {api} — the button opens the side panel, it "
            f"does not arrange windows. Panel side and width are Chrome "
            f"settings, not extension behaviour"
        )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants,
              check_no_shadowed_globals, check_side_panel_shell,
              check_button_opens_the_panel]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    print("PASS extension")
