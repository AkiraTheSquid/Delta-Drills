"""watch.py — health checks for vendor/tiptap

The vendored Tiptap engine is a pre-built JS ESM bundle, not importable Python.
These checks verify the bundle is present, non-trivial, and still exposes the
named exports the web UI depends on, plus the single-bundle invariant.
Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import sys
import os

HERE = os.path.dirname(__file__)
BUNDLE = os.path.join(HERE, 'tiptap.bundle.esm.js')
ENTRY = os.path.join(HERE, 'entry.mjs')

# The public surface the app imports. If a rebuild drops one of these, every
# consumer (brain-dump, future outliner) breaks — fail loudly here instead.
REQUIRED_EXPORTS = ['Editor', 'StarterKit', 'TaskList', 'TaskItem', 'Node', 'Mark', 'Extension']


# ── Import checks ──────────────────────────────
# Bundle is JS; "import" here means the files exist and are readable.
def check_imports():
    for path in (BUNDLE, ENTRY):
        assert os.path.isfile(path), f"missing vendored file: {os.path.basename(path)}"
    size = os.path.getsize(BUNDLE)
    # A real Tiptap+ProseMirror bundle is hundreds of KB; a truncated/empty
    # build would be a silent breakage.
    assert size > 100_000, f"tiptap.bundle.esm.js suspiciously small ({size} bytes)"


# ── Public API checks ─────────────────────────
# Verify the bundle still re-exports every name the web UI relies on.
def check_public_api():
    with open(BUNDLE, encoding='utf-8') as fh:
        text = fh.read()
    # esbuild emits a final `export{...}` block aliasing internal names.
    export_block = text.rsplit('export{', 1)[-1] if 'export{' in text else ''
    for name in REQUIRED_EXPORTS:
        assert f' as {name}' in export_block or f'{name},' in export_block or f'{name}}}' in export_block, \
            f"bundle no longer exports `{name}` (rebuild dropped it?)"


# ── Invariant checks ──────────────────────────
# Single-bundle rule: nothing else may load a second Tiptap/ProseMirror copy.
def check_invariants():
    webroot = os.path.abspath(os.path.join(HERE, '..', '..'))
    index = os.path.join(webroot, 'index.html')
    if os.path.isfile(index):
        with open(index, encoding='utf-8') as fh:
            html = fh.read()
        # A CDN <script> for tiptap/prosemirror would duplicate the singleton.
        for bad in ('cdn.jsdelivr.net/npm/@tiptap', 'unpkg.com/@tiptap',
                    'cdn.jsdelivr.net/npm/prosemirror', 'unpkg.com/prosemirror'):
            assert bad not in html, f"index.html loads a 2nd Tiptap/ProseMirror copy ({bad}) — breaks the singleton"
    # entry.mjs is the source of truth for the export surface.
    with open(ENTRY, encoding='utf-8') as fh:
        entry = fh.read()
    assert '@tiptap/core' in entry, "entry.mjs no longer sources @tiptap/core"

    # 🔴 THE BUNDLE IS IMPORTED, NEVER SCRIPT-TAGGED. It is 400 KB and its one
    # consumer is a tab most visitors never open, so groups/groups_checklist.js
    # dynamic-imports it on the first mount. A <script> tag in index.html would
    # load it for everybody, on every page view, and nothing would look wrong.
    if os.path.isfile(index):
        with open(index, encoding='utf-8') as fh:
            html = fh.read()
        assert 'vendor/tiptap' not in html, (
            "index.html loads the Tiptap bundle directly. It is imported on "
            "first use by groups/groups_checklist.js so only the Groups tab pays"
        )

    # And the consumer is still there. A vendored 400 KB blob with no importer
    # is dead weight nobody would think to delete.
    consumer = os.path.join(webroot, 'groups', 'groups_checklist.js')
    assert os.path.isfile(consumer), "groups/groups_checklist.js is gone"
    with open(consumer, encoding='utf-8') as fh:
        assert 'vendor/tiptap/tiptap.bundle.esm.js' in fh.read(), (
            "groups/groups_checklist.js no longer imports this bundle — either "
            "the import moved, or this folder is now unused"
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
    print("PASS vendor/tiptap")
