# content

## Purpose
The Colab DOM adapter. Everything in this extension that knows what Colab's
markup looks like lives here, so that when Google changes it there is exactly one
file to fix.

## Owns
- Reporting **which notebook this tab is**, and whether it has finished
  mounting — the panel needs both before it can decide to switch notebooks.
- Finding a cell — by our own stable anchor, or by matching rendered text.
- Making it actually visible: expanding collapsed sections, scrolling the right
  container, and confirming the landing rather than assuming it.
- Reading a cell's rendered output for the assertion pre-select.

## Does NOT own
- What to navigate to, or when — including which notebook. The panel decides;
  this answers `dd:identify` and `dd:goto`. It never navigates the tab itself.
- Notebook content or anchors — those are minted by
  `scripts/generate_colab_notebooks.py`.
- Any network call. This script never talks to the backend.

## Key Files
- `colab.js`: the whole adapter. Registers a `chrome.runtime.onMessage` listener
  for `dd:ping`, `dd:identify`, `dd:goto` and `dd:read-output`, injected at
  `document_idle` on `https://colab.research.google.com/*`.

## Data & External Dependencies
Colab's DOM, and nothing else. The selectors this depends on, all verified
against a live notebook:

| Selector | What it is |
|---|---|
| `colab-scroller#notebook-main` | the real scroll container — **not** `window` |
| `div.cell` | one notebook cell |
| `.focused` | the active cell |
| `md-icon-button.header-section-toggle` | a section's expand/collapse control; its `aria-label` reads "Expand"/"Collapse" for the *current* state |
| `cell-<metadata.id>` | the DOM id Colab derives from an nbformat 4.5 cell id |

## How It Works (Flow)
0. `identify()` — reports `{lessonId, ready, url, cells}`. Identity comes from
   the `cell-dd-lesson-*` anchor, then the `dd:dd-lesson-…` comment marker, then
   the `DD_LESSON_ID = "…"` line in the setup cell. That last one is the
   reliable route: it is plain rendered text, so it survives Colab dropping cell
   ids. `ready` is about the DOM, not identity — a notebook mid-mount has no
   cells, and calling that "the wrong notebook" would loop the panel through a
   pointless re-navigation.
1. `findCell({anchor, text})` — tries `cell-<anchor>`, then the `<!-- dd:… -->`
   marker in the markdown body, then a case-insensitive text match.
2. `expandAbove(cell)` — walks backwards through the cell list from the target and
   clicks every toggle labelled "Expand". Walking the whole prefix, not just the
   nearest toggle, is what handles nested sections.
3. `goto` re-resolves the cell after expanding (expansion re-renders and detaches
   the node), scrolls, waits out the animation, and retries once if the cell is
   still not visible.
4. `readOutput` returns the output text plus `errored`.

## Invariants & Constraints
- **Anchors are unreliable on upstream ARENA notebooks.** All 458 ARENA_5.0
  notebooks are nbformat 4.2 with no `metadata.id`, so Colab mints fresh DOM ids
  on every load. The text fallback is not a nicety — it is the only thing that
  works there. Never remove it.
- **Expand before scrolling.** A cell inside a collapsed section has zero height,
  so `scrollIntoView` silently succeeds and nothing moves.
- **Re-resolve after expanding.** The captured node is likely detached.
- **A clean run is not a pass.** `readOutput` reports `errored` as authoritative
  but never infers success — most cells print something either way, and guessing
  would feed the ladder a fabricated attempt.
- Never `alert`/`confirm`/`prompt`: a modal dialog blocks the page and the
  extension stops receiving messages entirely.
- Keep this script side-effect free until a message arrives. It runs on every
  Colab page the student opens, including ones that have nothing to do with us.

## Extension Points
- New capability → a `case "dd:x"` in the `onMessage` switch. Return `true` from
  the listener for anything async, or the response channel closes early.
- Selector drift → the four constants at the top of `colab.js`.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **No listener in an already-open tab** — `ACTIVE`
  - When it happens: the extension is loaded or reloaded while Colab is already open.
  - Symptom: `chrome.tabs.sendMessage` throws; the panel shows "Reload the Colab tab".
  - Root cause: content scripts inject at navigation, not retroactively.
  - Prevention/fix: reload the tab. The panel already distinguishes this
    (`no-receiver`) from "no Colab tab at all" so the message is actionable.

- **Silent no-op scroll into a collapsed section** — `RESOLVED`
  - When it happens: the target sits under a collapsed header.
  - Symptom: scroll reports success, nothing moves, `visible:false`.
  - Root cause: zero-height target.
  - Prevention/fix: `expandAbove` + re-resolve + the `isVisible` confirmation.
    Do not "simplify" `goto` by dropping the second `scrollToCell` — a lazily
    mounted editor can change heights mid-scroll.

- **Selector rot** — `ACTIVE`
  - When it happens: Colab ships a markup change.
  - Symptom: every jump reports `not-found`.
  - Root cause: these are undocumented internals with no stability guarantee.
  - Prevention/fix: `watch.py` asserts all four selectors are still referenced,
    which catches an accidental deletion but not an upstream rename. Confirm
    against a live notebook after any Colab UI change.

## Invariants added with `dd:identify`
- **Report a fragmentless URL.** Colab appends `#scrollTo=…` as the student
  moves around, and the panel stores this URL to reopen the notebook later — a
  stored fragment would reopen mid-notebook forever after.
- **`ready` and `lessonId` are separate answers.** "Not loaded yet" and "loaded,
  and it is a different notebook" lead to opposite decisions in the panel.

## Recent Changes
- 2026-07-31: Added `dd:identify` — notebook identity and mount state, so the
  panel can switch notebooks between problems.
- 2026-07-31: Initial build — `dd:ping` / `dd:goto` / `dd:read-output`.
  Pass 1 of `docs/spec-colab-its-surface.md`.
