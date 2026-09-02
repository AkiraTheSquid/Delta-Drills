"""watch.py — health checks for groups

The Groups tab. Every check here guards a boundary that has no other guard:
the surface is three classic scripts with no build step, so nothing but this
file notices when a global stops being published, when a fetch grows a
fallback, or when the public directory starts shipping names.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, '../..'))

# In load order. Nothing here reads another of them at LOAD time — every
# cross-file reference is a `window.DD…` lookup inside a function — but the
# order is asserted anyway, because a reader deciding where to add the eighth
# file should be told there is an order rather than left to guess.
FILES = (
    "groups_store.js",
    "groups_checklist_doc.js",
    "groups_checklist.js",
    "groups_day.js",
    "groups_join.js",
    "groups_lane.js",
    "groups_view.js",
)


def _read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _src(name):
    return _read(os.path.join(HERE, name))


def _code(source):
    """`source` with its comments taken out.

    These files are commented at least as heavily as they are written, and the
    comments NAME the things the checks below look for — the confirm() rule is
    written down three lines above the code that keeps it. Scanning the raw
    text made this file fail on its own documentation.

    Crude on purpose: it will also blank a `//` inside a string literal, which
    costs nothing here because every check downstream is a presence test for a
    call, not a parse.
    """
    without_block = re.sub(r"/\*.*?\*/", " ", source, flags=re.S)
    return re.sub(r"//[^\n]*", " ", without_block)


def check_imports():
    """The three files exist and each publishes the global the others read.

    They are classic scripts: the only linkage between them is a name on
    `window`, so a rename that looks local is a blank page.
    """
    for name in FILES:
        assert os.path.exists(os.path.join(HERE, name)), f"{name} is missing"

    published = {
        "groups_store.js": "window.DDGroupStore",
        "groups_checklist_doc.js": "window.DDChecklistDoc",
        "groups_checklist.js": "window.DDChecklist",
        "groups_day.js": "window.DDGroupsDay",
        "groups_join.js": "window.DDGroupsJoin",
        "groups_lane.js": "window.DDGroupsLane",
        "groups_view.js": "window.DDGroups",
    }
    for name, global_name in published.items():
        assert f"{global_name} =" in _src(name), (
            f"{name} no longer publishes {global_name} — the other two read it "
            "off window and there is no module graph to catch this"
        )


def check_public_api():
    """index.html loads all three, in order, and app.js still wires the page.

    Loading is not optional and the ORDER is not either: app.js's boot
    switchTab can route straight to #page-groups off an invite link, so the
    store has to exist by then.
    """
    index = _read(os.path.join(SHARED, "index.html"))
    positions = []
    for name in FILES:
        needle = f'src="groups/{name}'
        assert needle in index, f"index.html stopped loading groups/{name}"
        positions.append(index.index(needle))
    assert positions == sorted(positions), (
        "the groups scripts are out of order in index.html — groups_store.js "
        "must load before groups_view.js reads it at boot"
    )

    # 🔴 THE TIPTAP BUNDLE MUST BE THERE, AND MUST NOT BE A SCRIPT TAG.
    # `groups_checklist.js` dynamic-imports it on the first mount, which is the
    # whole reason a 400 KB editor can live on a tab most visitors never open.
    # A <script> tag here would load it for everybody, on every page view.
    bundle = os.path.join(SHARED, "vendor", "tiptap", "tiptap.bundle.esm.js")
    assert os.path.exists(bundle), (
        "vendor/tiptap/tiptap.bundle.esm.js is gone — the checklist column "
        "would draw 'the checklist editor could not load' for everybody. It is "
        "committed on purpose: this app has no bundler, and a gitignored file "
        "under Local_Deployed_Shared/ is silently absent from the deploy"
    )
    # 🔴 AND THE SPECIFIER CLIMBS OUT OF THIS FOLDER. A dynamic import()
    # resolves against the SCRIPT's URL, not the document's — the opposite of
    # every other URL in this app. `./vendor/...` from groups/ asks for
    # /groups/vendor/... and 404s, which the page reports as "the checklist
    # editor could not load" while everything else on the tab works.
    assert '"../vendor/tiptap/tiptap.bundle.esm.js"' in _src("groups_checklist.js"), (
        "groups_checklist.js's bundle specifier is no longer ../vendor/... — a "
        "dynamic import() resolves against THIS FILE's URL, so anything that "
        "does not climb out of groups/ 404s"
    )

    assert "vendor/tiptap" not in index, (
        "index.html loads the Tiptap bundle with a script tag. It is imported "
        "on first use by groups_checklist.js so that only people who open the "
        "Groups tab pay the 400 KB"
    )

    # 🔴 AND ALL THREE BEFORE app.js. This is the one that actually bit: app.js
    # runs its boot switchTab while it parses, and that call asks the store
    # whether the address bar carries an invite token. Loaded after app.js the
    # store is undefined at that moment, so `?group=<token>` opens the Learner
    # Home and the invite silently does nothing — every other part of the
    # feature works, which is why it took a browser to find.
    app_at = index.find('src="app.js')
    assert app_at != -1, "index.html stopped loading app.js"
    assert max(positions) < app_at, (
        "the groups scripts moved BELOW app.js in index.html — app.js's boot "
        "switchTab reads DDGroupStore.inviteFromLocation(), so an invite link "
        "would stop landing on the Groups tab, with no error anywhere"
    )

    assert 'id="page-groups"' in index, (
        "#page-groups is gone; switchTab falls back to Practice for a name with "
        "no page, so the menu row would look like a dead button"
    )
    assert 'id="groups-root"' in index, "#groups-root is the only mount point"
    assert 'data-goto-tab="groups"' in index, (
        "the account-menu row is gone. It is the REAL entry point — basic mode "
        "is the default and hides the tab strip entirely"
    )
    assert 'href="styles/groups.css' in index, "index.html stopped loading styles/groups.css"

    app_js = _read(os.path.join(SHARED, "app.js"))
    assert "window.DDGroups?.refresh()" in app_js, (
        "app.js stopped refreshing the roster on arrival — the page would show "
        "whatever the last visit left behind"
    )
    # 🔴 And tearing the editor down on the way OUT. The teardown is what
    # flushes the save debounce; without it, typing a line and immediately
    # clicking another tab loses the line with no error anywhere.
    assert "window.DDGroups?.suspend?.()" in app_js, (
        "app.js stopped suspending the Groups tab on leave — the checklist "
        "editor's half-second save debounce would never be flushed, so the "
        "last line typed before switching tabs is silently discarded"
    )
    assert "window.DDIdentity" in app_js, (
        "app.js stopped publishing DDIdentity; groups_store.js reads it to "
        "decide whether to make the call at all"
    )
    advanced = re.search(r"const advancedOnlyTabs = \[[^\]]*\]", app_js)
    assert advanced and '"groups"' not in advanced.group(0), (
        "groups landed in advancedOnlyTabs — switchTab REFUSES to route to one "
        "in basic mode, which is the default, so the menu row would silently "
        "open Practice instead"
    )


def check_invariants():
    """The rules this feature is built around, each with no other guard."""
    store = _src("groups_store.js")
    join = _src("groups_join.js")
    view = _src("groups_view.js")
    lane = _src("groups_lane.js")
    editor = _src("groups_checklist.js")
    day = _src("groups_day.js")

    # 🔴 No second readiness scale. The map lives in practice/placement-results.js
    # and mirrors _mastery_from_theta in backend/app/diagnostic.py; practice's
    # own watcher fails on drift, but only for the copies it knows about.
    for constant in ("DIFF_FLOOR", "DIFF_SPAN", "SEED_MASTERY"):
        assert constant not in lane, (
            f"{constant} was re-declared in groups_lane.js. Use "
            "PlacementResults.readiness()/.band() — a second copy of that map "
            "is how two screens end up quoting different numbers for one learner"
        )
    assert "results.readiness(" in lane and "results.band(" in lane, (
        "groups_lane.js stopped scoring areas through PlacementResults"
    )

    # 🔴 The public directory never learns a name.
    row = join.split("buildDirectoryRow")[1].split("const renderDirectory")[0]
    assert "display_name" not in row, (
        "a directory row started reading display_name. /groups/public answers "
        "initials only; a fallback here would work on the roster and silently "
        "do nothing on the directory, which is how the boundary stops being real"
    )

    # 🔴 One consent gate in front of every way in. Joining publishes this
    # learner's mastery to everyone in the group and cannot be undone by
    # clicking again.
    for call in ("createGroup", "joinGroup", "joinPublicGroup"):
        hits = list(re.finditer(rf"store\(\)\.{call}\(", join))
        assert hits, f"groups_join.js no longer calls {call}"
        for hit in hits:
            window = join[max(0, hit.start() - 300):hit.start()]
            assert "ask(" in window, (
                f"a {call} call in groups_join.js is not behind ask() — the "
                "consent gate is the feature, and the door added later is the "
                "one that forgets to ask"
            )

    # 🔴 No bare-fetch fallback. On Vercel the SPA rewrite answers a relative
    # /api/... with 200 text/html, so a fallback renders the signed-out reading
    # for a signed-in learner with nothing in the console.
    assert "window.fetch" not in store and "|| fetch" not in store, (
        "groups_store.js grew a bare-fetch fallback — see the apiFetch note in "
        "app.js and this folder's README"
    )

    # 🔴 A native confirm blocks the page and this repo's browser checks cannot
    # dismiss one.
    for name, source in (
        ("groups_join.js", _code(join)),
        ("groups_view.js", _code(view)),
        ("groups_lane.js", _code(lane)),
        ("groups_day.js", _code(day)),
    ):
        assert not re.search(r"[^.\w]confirm\(", source), (
            f"{name} uses confirm(); the listing toggle's two-click arm is the "
            "pattern here"
        )

    # 🔴 The token is a capability: it must leave the address bar once used.
    assert "clearInviteFromLocation" in store and "clearInviteFromLocation()" in join, (
        "the invite token is no longer cleared out of the URL after a join — a "
        "token left in the bar gets pasted, bookmarked and screenshotted"
    )

    # The unprobed honesty rule, carried over from placement-results.js.
    assert "placement-area--unprobed" in lane and "not probed" in lane, (
        "groups_lane.js stopped dimming unprobed areas. An unprobed theta is a "
        "prior wearing a percentage, and here it is being read ABOUT SOMEBODY ELSE"
    )

    # ── THE CHECKLIST COLUMN ────────────────────────────────────────────────

    # 🔴 ONE LIVE EDITOR ON THE PAGE, AND IT IS YOURS. Twelve members would be
    # twelve ProseMirror node-view registries and eleven contenteditables
    # nobody may type into; everybody else's list is the read-only renderer.
    assert "DDChecklist.mount(" in lane, "groups_lane.js stopped mounting the editor"
    mounts = _code(lane).count(".mount(")
    assert mounts == 1, (
        f"groups_lane.js mounts the checklist editor {mounts} times. There is "
        "exactly one editable column on this page — the row marked is-you — "
        "and every other member is drawn by DDChecklistDoc.renderStored"
    )
    assert "renderStored(" in lane, (
        "groups_lane.js stopped drawing other members' checklists with the "
        "read-only renderer"
    )

    # 🔴 A LOAD MUST NEVER SAVE. Tiptap 3 takes an OPTIONS OBJECT; a positional
    # `false` is the Tiptap 2 signature, is silently ignored, and defaults
    # emitUpdate to TRUE — so every setContent would fire onUpdate and write
    # back the row it had just read. Delta Note shipped exactly that bug.
    for hit in re.finditer(r"setContent\(([^;]*?)\)\s*;", _code(editor)):
        assert "emitUpdate: false" in hit.group(1), (
            "a setContent in groups_checklist.js does not pass "
            "{ emitUpdate: false } — opening a day would save the day it just "
            "read, and a positional `false` does NOT work in Tiptap 3"
        )

    # 🔴 THE TEARDOWN FLUSHES. Changing the day destroys the editor, and the
    # debounce is half a second: without the flush the last words typed before
    # a click on the next-day arrow are the ones that are lost.
    destroy = editor.split("destroy: ()")[1][:400]
    assert "save()" in destroy, (
        "groups_checklist.js's destroy() no longer flushes the pending save — "
        "the last edit before a day change is discarded silently"
    )

    # 🔴 A DAY KEY IS LOCAL. toISOString() is UTC: west of Greenwich it names
    # YESTERDAY all evening, so a checklist is written under one key and read
    # back under another. It does not throw; the list just comes back empty.
    assert "toISOString" not in _code(day), (
        "groups_day.js reached for toISOString(). A day key is the LOCAL "
        "calendar date — build it from getFullYear/getMonth/getDate"
    )

    # 🔴 A SAVE NAMES THE DAY IT WAS TYPED ON, NOT THE DAY ON SCREEN. The
    # editor's teardown flushes its pending save, and the teardown happens
    # BECAUSE the day just changed — so an onSave reading the module's live
    # `day` files the last sentence typed on Wednesday under Tuesday, over the
    # top of whatever was there. Nothing throws and nothing is lost, which is
    # why it took a browser and a diff of the database to see.
    assert "const paintedDay = day;" in view and "saveDay(paintedDay," in view, (
        "groups_view.js stopped capturing the painted day for the checklist "
        "save. `onSave: (text) => saveDay(day, text)` reads the day AT SAVE "
        "TIME, and the teardown flush fires after the day has already moved"
    )

    # 🔴 A FAILED GROUP READ IS NOT AN EMPTY ONE. Spelling both `null` drew
    # the create/join card for a member of a group during an outage.
    my_group = store.split("async readMyGroup(")[1][:600]
    assert "return undefined;" in my_group, (
        "groups_store.js's readMyGroup went back to answering null on a "
        "failed read. `null` means you are in no group"
    )
    assert "group === undefined" in view, (
        "groups_view.js stopped telling a failed group read apart from an "
        "empty one"
    )
    assert "seq !== fetchSeq || !active" in view, (
        "groups_view.js's roster read stopped checking the tab is still open. "
        "It calls onGroup(), which starts a day read, which paints, which mounts"
    )

    # 🔴 THE THIRD STATE HAS TO REACH A SCREEN READER. A native checkbox has
    # two states; an X'd item announced as "not checked" is the same words as
    # an untouched one.
    doc = _src("groups_checklist_doc.js")
    assert "indeterminate" in doc and "aria-checked" in doc, (
        "groups_checklist_doc.js stopped announcing the X state. A tri-state "
        "checkbox is `indeterminate` plus aria-checked=mixed"
    )
    for name, source in (("groups_checklist_doc.js", doc), ("groups_checklist.js", editor)):
        assert "paintCheckboxState(" in source, (
            f"{name} stopped painting its checkbox through the shared helper — "
            "the two columns would announce the same document differently"
        )

    # 🔴 NO PLACEHOLDER IN THE EDITOR. ProseMirror draws a placeholder from the
    # empty paragraph's ::before, and inside a task item that paragraph sits
    # BESIDE the checkbox — the prompt landed on top of the box.
    assert "data-placeholder" not in editor, (
        "groups_checklist.js put a placeholder back on the editor. It overlaps "
        "the checkbox of an empty task item"
    )

    # 🔴 ONE ISLAND FOR EVERYBODY. The day picker is the board's first row and
    # every member is a row beneath it — not a control above a stack of cards.
    assert 'el("div", "dd-board")' in view and "dd-board-day" in view, (
        "groups_view.js stopped building the single board. The day row and the "
        "member rows share one panel"
    )

    # 🔴 A FAILED READ IS NOT A SLOW READ. Both arrive with no entries, and
    # spelling them the same way leaves "Reading this day…" on screen forever
    # with nothing in flight behind it.
    assert 'dayState = answer === null ? "failed" : "ready"' in view, (
        "groups_view.js stopped distinguishing a failed day read from a "
        "pending one. `entries === null` alone cannot say which"
    )
    assert 'dayState === "failed"' in lane, (
        "groups_lane.js stopped rendering the failed-read state. A column "
        "that says 'Reading this day…' with no request in flight is a lie"
    )

    # 🔴 A SAVE THAT FAILED HAS TO SAY SO. The write is debounced, so the
    # person has already looked away; `void`ing the answer turns a 413 or a
    # dead connection into a save that never happened and never said so.
    assert "return store().saveDay(" in view, (
        "groups_view.js stopped returning the save. Its answer is what the "
        "column reads to show 'Not saved: …'"
    )
    assert "dd-checklist-status" in lane and "Not saved:" in lane, (
        "groups_lane.js stopped surfacing a failed checklist save"
    )

    # 🔴 LEAVING THE TAB MUST STOP A READ THAT IS ALREADY IN FLIGHT. A late
    # answer repaints, and a repaint MOUNTS AN EDITOR — into a hidden page,
    # with nothing left to destroy it.
    assert "active = false;" in view and "!root || !active" in view, (
        "groups_view.js stopped guarding its repaint on the tab still being "
        "open. suspend() tears the editor down; the in-flight read does not "
        "know that and will mount another one"
    )

    # 🔴 ONE WRITE AT A TIME, AND NO READ ACROSS A PENDING WRITE. Each save
    # sends the WHOLE document: two in flight together let the network decide
    # which text survives, and a read that overtakes the teardown flush hands
    # the editor a stale document it then saves back over the newer one.
    assert "dayWrites = run" in store and "await settled(dayWrites)" in store, (
        "groups_store.js stopped serialising the day writes. saveDay must "
        "chain onto dayWrites and readDay must wait for it"
    )

    # 🔴 THE WRITE ENDPOINT TAKES NO TARGET. The row is keyed by the
    # authenticated user; a member_id parameter would let anybody in the group
    # rewrite anybody else's day, and a group is joined by anyone with a link.
    save_day = store.split("async saveDay(")[1][:900]
    assert "member" not in save_day, (
        "groups_store.js's saveDay grew a member parameter. Your own row is "
        "the only checklist that endpoint may write"
    )


if __name__ == '__main__':
    checks = [check_imports, check_public_api, check_invariants]
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
