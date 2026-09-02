# vendor/tiptap

Vendored [Tiptap](https://tiptap.dev) (MIT) editor engine, shipped as a single
pre-built ESM bundle. This app has no bundler — `Local_Deployed_Shared/` is
classic scripts served statically — so Tiptap is built once and committed.

Copied verbatim from Delta Note's `shared/web-components/vendor/tiptap/`
(2026-09-02), bundle byte-for-byte identical. That is deliberate: the two apps
store the same `{v, doc}` checklist document, and one engine at one version is
what keeps a document written in one readable in the other.

## Purpose

The Groups tab's per-day checklist: a task list whose checkbox cycles three
states (open → checked → X). See `../../groups/groups_checklist.js`, which
extends `TaskItem` with the third state, and `../../groups/groups_checklist_doc.js`,
which owns the stored document and the read-only renderer.

## Key Files

- `tiptap.bundle.esm.js` — the build output. A single deduped ESM module.
  Named exports: `Editor`, `Node`, `Mark`, `Extension`, `StarterKit`,
  `TaskList`, `TaskItem`.
- `entry.mjs` — the bundle's source entry (what esbuild is fed). The
  authoritative list of what the bundle exposes; edit it, then rebuild.
- `LICENSE` — Tiptap MIT license.

## Invariants

- **🔴 ONE BUNDLE, ALWAYS.** Tiptap is built on ProseMirror, and ProseMirror's
  state/model is a SINGLETON contract: two separately-bundled copies of
  `prosemirror-state` / `prosemirror-model` break the editor at runtime with
  opaque errors. Never load a second Tiptap or ProseMirror alongside this one —
  no CDN `<script>`, no second vendored copy. New extensions go into
  `entry.mjs` and a rebuild, not into an import at the app layer. `watch.py`
  fails if `index.html` grows a CDN tag for either.
- **🔴 IT IS IMPORTED ON FIRST USE, NEVER SCRIPT-TAGGED.** 400 KB for a tab
  most visitors never open. `groups_checklist.js` calls `import()` on the first
  mount and caches the promise module-wide, so the whole roster shares one
  load and everybody else pays nothing. `watch.py` fails if `vendor/tiptap`
  ever appears in `index.html`.
- **🔴 THE IMPORT SPECIFIER CLIMBS OUT OF `groups/`.** A dynamic `import()`
  resolves against the SCRIPT's URL, not the document's — the opposite of every
  other URL in this app. From `groups/groups_checklist.js` that means
  `../vendor/tiptap/…`; `./vendor/…` asks for `/groups/vendor/…` and 404s,
  which the page reports as "the checklist editor could not load" while every
  other part of the tab works. Written the wrong way first and caught in the
  browser; `groups/watch.py` pins it now.
- **🔴 IT MUST STAY TRACKED BY GIT.** The deploy script syncs
  `Local_Deployed_Shared/` and runs `git add -A`, so a gitignored file under it
  is silently absent from production — the 2026-08-06 crosswalk incident. This
  folder is committed on purpose, minified blob and all.

## Pinned version

Tiptap **3.27.1** (`@tiptap/core`, `@tiptap/starter-kit`,
`@tiptap/extension-list`). `StarterKit` brings paragraph/heading/bold/italic/
lists; `TaskList` + `TaskItem` come from `@tiptap/extension-list` and are added
explicitly in `entry.mjs` because StarterKit does not register them by default.

🔴 Tiptap **3**, not 2, and the difference bites: `setContent(content, false)`
is the Tiptap 2 signature. Tiptap 3 takes an OPTIONS OBJECT, silently ignores a
stray boolean, and defaults `emitUpdate` to TRUE — so a positional `false`
reads as suppressed and emits anyway, and a plain LOAD saves the row it has
just read. Always `setContent(content, { emitUpdate: false })`.

The Placeholder extension is deliberately NOT bundled: its v3 viewport plugin
relies on layout APIs that cannot be smoke-verified headless. The empty-state
placeholder is done in the app layer instead — a class toggled from
`editor.isEmpty` plus `::before { content: attr(data-placeholder) }` in
`../../styles/groups.css`.

## Rebuild (reproduce / upgrade)

```sh
mkdir tiptap-build && cd tiptap-build && npm init -y
npm i @tiptap/core @tiptap/pm @tiptap/starter-kit esbuild
#   (TaskList/TaskItem live in @tiptap/extension-list, a starter-kit dep)
# copy entry.mjs from this folder into the build dir, then:
npx esbuild entry.mjs --bundle --format=esm --platform=browser \
  --target=es2020 --minify --legal-comments=none \
  --outfile=tiptap.bundle.esm.js
# copy tiptap.bundle.esm.js back here; update the pinned version above.
```

Then open the Groups tab in a browser and type into your own column: the
bundle has no test here, and `watch.py` can only check that the exports still
exist and the file is not truncated.

## Consumers

- `../../groups/groups_checklist.js` — the Groups tab's day checklist. The only
  one today.

## Recent Changes

- 2026-09-02: Folder created. Tiptap 3.27.1 copied from Delta Note so the
  Groups tab's member rows can carry that app's three-state checkbox list.
