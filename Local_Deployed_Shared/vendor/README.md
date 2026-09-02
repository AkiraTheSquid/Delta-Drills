# vendor

## Purpose

Third-party code the app ships **as files it serves itself**, rather than
pulling from a CDN at runtime. One subfolder per library, each with its own
README, LICENSE and `watch.py`.

There is exactly one tenant today: `tiptap/`, the editor engine behind the
Groups tab's three-state checklists.

Vendoring rather than linking is a deliberate trade. A CDN `<script>` is a
second origin that has to be up, has to be reachable from wherever a learner
is, and has to agree with the version this code was written against. The cost
of vendoring is a large binary in the repo and a manual upgrade; the benefit is
that the app has no runtime dependency on anybody else's uptime, and a checkout
of this repo at any commit is the app as it ran.

## Owns

- The vendored library files themselves, byte for byte as they were copied.
- Each library's LICENSE, kept beside the code it covers.
- The per-library `watch.py` that asserts the copy is intact and still reached
  the way its consumers expect.

## Does NOT own

- **How the library is used.** Every consumer lives in the folder that needs
  it — `groups/groups_checklist.js` imports the Tiptap bundle; nothing in here
  knows what a checklist is.
- **The build.** There is no build step in this app. Whatever lands here is
  already the artefact the browser loads; nothing here is compiled, bundled or
  minified on the way out.
- **`package.json` / `node_modules`.** This app is classic scripts served
  statically. A vendored file is not an npm dependency and must not become one
  by being listed as one.

## Key Files

- `tiptap/` — Tiptap 3 as a single ESM bundle (~400 KB), copied from Delta
  Note's `shared/web-components/js/vendor/tiptap/` so the two apps write the
  same `{v, doc}` checklist document. Read `tiptap/README.md` before touching
  it: the one-bundle rule there is a correctness constraint, not tidiness.

## Data & External Dependencies

None at runtime. Nothing in this folder makes a request, reads storage or
touches the backend. The files are static assets served by the same origin as
the rest of `Local_Deployed_Shared/`.

The *upstream* dependency is a human one: each subfolder's README records where
its copy came from and what version it is, because there is no lockfile to ask.

## How It Works (Flow)

1. A consumer decides it needs the library — for Tiptap, that is the first time
   somebody opens the Groups tab with a checklist of their own to edit.
2. It reaches the file by a **relative path from the consuming script**, and
   loads it with a dynamic `import()`.
3. The promise is cached by the consumer, so one page load fetches one copy no
   matter how many rows want it.

Nothing in this folder is loaded at boot, and nothing is referenced from
`index.html`. A visitor who never opens the tab never pays for it.

## Invariants & Constraints

- 🔴 **These files stay git-tracked.** The deploy script commits with
  `git add -A` and pushes what is in the repo; a vendored file caught by a
  `.gitignore` rule (`*.bundle.js`, `vendor/`, a size rule) is silently absent
  in production while every local check passes. That failure mode has already
  cost this project once — see the crosswalk incident of 2026-08-06.
- 🔴 **One copy of a library, ever.** Two copies of an editor engine is not
  redundancy, it is two module registries and two prototype chains for
  documents that must be interchangeable. If a second consumer appears, it
  imports the same file.
- 🔴 **A vendored file is not edited.** Fix a bug by extending the library from
  the consumer's side (a custom extension, a wrapper), never by patching the
  copy — a patched bundle looks identical to an unpatched one and the next
  upgrade silently drops the fix. If a patch is genuinely unavoidable, it goes
  in the subfolder's README under its own heading, with the diff.
- 🔴 **The specifier is relative to the SCRIPT, not the page.** A dynamic
  `import()` resolves against the URL of the file that ran it — the opposite of
  `<script src>`, `fetch()` and every href this app writes. A consumer in
  `groups/` reaches this folder as `../vendor/...`.
- Every subfolder carries the upstream LICENSE. Tiptap 3 is MIT; keeping the
  licence beside the code is what makes redistributing it legitimate.

## Extension Points

Adding a library: make a subfolder, copy the artefact and its LICENSE in
unmodified, write a README saying **where it came from, which version, why it
is vendored and who imports it**, and a `watch.py` that fails if the file is
missing, truncated, duplicated on a CDN, or no longer imported by the consumer
it was added for. Then confirm `git check-ignore -v <path>` says nothing.

Upgrading one: replace the artefact, update the version in its README, and run
that folder's watcher plus the consumer's. There is no lockfile to tell you
what changed, so the watcher's export list is the contract.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **A vendored file gitignored out of production** — `PREVENTED`
  - When it happens: a broad ignore rule (`*.bundle.js`, `vendor/`, anything
    size-based) is added for unrelated reasons.
  - Symptom: works locally, 404 in production — and because the SPA rewrite
    answers 200 with `text/html`, a `curl` of the URL looks fine. Only the
    content type gives it away.
  - Prevention: `tiptap/watch.py` asserts the bundle is present and large;
    check `git check-ignore -v` when adding anything here.

- **Two copies of an editor engine** — `PREVENTED`
  - When it happens: a second feature wants Tiptap and adds a CDN tag "just for
    that page".
  - Symptom: documents that look right and fail to round-trip; ProseMirror
    state/model must be a singleton.
  - Prevention: the subfolder watcher fails if `index.html` script-tags the
    bundle or if a CDN URL for the same library appears in the app.

## Recent Changes
- 2026-09-02: Folder created for the Groups tab's checklists. First and only
  tenant: `tiptap/`, copied from Delta Note so both apps read and write the
  same stored document.
