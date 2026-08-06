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
- `colab_focus.js` + `colab_dd.css`: **show one problem, and skin the page.** Two
  independent toggles in a small floating panel (bottom-right, collapsible),
  persisted in `chrome.storage.local` under `dd_colab_view`:
  - *Only this problem* — hides every cell that is not part of the problem the
    URL points at. The target comes from the fragment the app's "Open in Colab ↗"
    already builds (`#scrollTo=dd-q123`), so nothing has to message anything.
    "Part of the problem" is decided by the number in `dd-q<n>`, which is why a
    stage-2 problem's worked example (`dd-q<n>-example`, minted by the generator
    after the PROBLEM it scaffolds) is in focus alongside it for free.
  - *Delta Drills theme* — the dark skin, Colab's header/toolbars/left pane/Gemini
    spark hidden. Adapted from Seth's hand-written CSS; its `a {display:none}`
    rule is deliberately dropped (it hid links inside the problem prose too).

  - *Gemini autocomplete* — **off by default**, and the one row where unticked
    is the working state. Colab ships "Show AI-powered inline completions" ON
    (its own switch is Settings → AI Assistance), and on one of our notebooks
    what it completes is the answer, in grey, ahead of the caret. Suppressed
    only on a page carrying `dd-` anchors, so other people's notebooks keep it.

  Every CSS rule is scoped under `html.dd-theme` / `html.dd-focus` /
  `html.dd-no-ai`, which only this script adds — all off is a stock Colab page.
  The toggle panel's own styling is the one unscoped exception, because it is
  the way back out.
- `colab_no_ai.js`: **the Gemini off switch**, and the only file here that runs
  in the page's own world (`"world": "MAIN"` in the manifest). It has to:
  an isolated content script cannot see `window.monaco`, and this works by
  setting `inlineSuggest: {enabled: false}` on every editor — hooking
  `onDidCreateEditor` for cells mounted later and `onDidChangeConfiguration` for
  Colab re-applying its own options, with a slow sweep behind both. MAIN world
  means no `chrome.*` at all, so it holds no policy: `colab_focus.js` dispatches
  `dd:gemini-off` / `dd:gemini-on` (the flag is in the event *name* — a
  `CustomEvent`'s `detail` is minted in the sending world), and it starts
  suppressed so that a slow `chrome.storage` read cannot leave a window in which
  the editor is live and completing. It gives back only what it took, tracked in
  a `WeakSet`, or our toggle would switch on a feature the user had disabled in
  Colab's own settings.
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
- **Hiding Gemini's shadow text in CSS alone is worse than not hiding it.**
  Monaco's Tab handler accepts the suggestion in the MODEL, not the one on
  screen, so a `display: none` leaves Tab pasting in an answer the learner never
  saw. The `html.dd-no-ai` rules are a backstop for the frames before Monaco
  loads; the real switch is `colab_no_ai.js`. Both are driven by the same class
  on the same pass so there is no state where the text is hidden but still live.
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
- 2026-08-06 (**a lesson is a focus target, not just a problem**): `colab_focus.js`.
  The app routes its teaching step to `#scrollTo=dd-kp-<slug>`, and focus only
  understood `dd-q<n>`. So the fragment resolved to nothing, `sticky` kept the
  PREVIOUS problem, and the concept the learner had just been sent to stayed
  `dd-out-of-focus` — scrolled to and `display: none`. Nothing moved on screen,
  which reads as a dead link rather than as focus being wrong. Targets are now
  groups of two kinds: `q:<n>` from the anchor itself, and `kp:<slug>` for a RUN
  of cells, since the prose under a concept header carries minted ids that name
  nothing — membership comes from DOM order, which is how the generator emits
  them. A `dd-q…` cell always wins its own group, or routing to a drill would
  unfold the whole lesson around it. `watch.py::check_a_lesson_is_a_focus_target`
  runs both shipped patterns and asserts that ordering.
- 2026-08-04 (**Gemini's shadow text no longer writes the answer**): Colab ships
  "Show AI-powered inline completions" ON (Settings → AI Assistance, its own key
  is `inlineCompletions`), so a learner routed to a problem the ladder says they
  cannot do yet was shown the solution in grey ahead of their caret. New
  `colab_no_ai.js` sets `inlineSuggest: {enabled: false}` on every Monaco editor
  — verified live against a real notebook: suppressed on load, restored on
  `dd:gemini-on`, re-clamped when Colab re-applies its own options, and applied
  to an editor created afterwards.
  - 🔴 **CSS alone is worse than nothing here.** Monaco's Tab handler accepts the
    suggestion in the MODEL, so hiding `.ghost-text-decoration` leaves Tab
    pasting in an answer the learner never saw. The `html.dd-no-ai` rules exist
    only as a backstop for the frames before Monaco loads, and they are driven by
    the same class, set on the same pass that dispatches the event — hidden but
    still live is the one state that must not exist.
  - 🔴 **It cannot be an ordinary content script.** The isolated world has no
    `window.monaco`, so this one is declared `"world": "MAIN"` — where there is
    no `chrome.*` at all. Hence the split: `colab_focus.js` owns the policy and
    dispatches `dd:gemini-off` / `dd:gemini-on`, with the flag in the event NAME
    because a `CustomEvent`'s `detail` is minted in the sending world.
  - Starts suppressed and is corrected afterwards, because the storage read is
    async: the other order leaves a window in which the editor is live and
    completing. Restores only editors it actually disabled (a `WeakSet`), or the
    toggle would switch on a feature the user had turned off in Colab itself.
  - Gated on the page carrying `dd-` anchors — other people's notebooks keep
    their completions. `watch.py::check_gemini_suppression_is_ours_only` pins
    that, the `"world": "MAIN"` key and the event names; both new assertions were
    negative-tested by removing the thing they guard.
  - 🟡 Found while writing the test: an exception inside `apply` is swallowed by
    the boot try/catch, which then runs `start()` a second time and leaves two
    MutationObservers and two hashchange listeners on a page that looks fine.
    `pushGemini` therefore never throws. Anything else added to `apply` owes the
    same care.
- 2026-08-03 (**the worked example is in focus with its problem, and the grouping needed no change to make it so**): stage 2 is now a pair — a solved example, then the same move on different specifics. `scripts/generate_colab_notebooks.py` emits the example as cells directly above the problem's header, anchored `dd-q<problem>-example`. `problemOf`'s trailing-boundary regex already sorts that into the problem's group, so focus keeps the pair on screen together and the learner finds the example by scrolling up. `watch.py::check_stage_two_pair_survives_focus` runs the shipped pattern against `dd-q481-example` / `dd-q481-example-code` and pins the boundary in the other direction too. **Do not "be explicit about the suffixes" in that regex** — narrowing it to the known `-hints|-code|-check|-solution` set would drop every example out of focus silently, leaving a bare problem that looks exactly like a problem.
  - The one thing that DID have to change here is telling the two apart. In focus mode the pair is the only content on screen: a prompt, a code cell, a prompt, a code cell, previously styled identically. `colab_focus.js` now tags example cells `.dd-example` and `colab_dd.css` gives them a left rule and a dimmed surface under `html.dd-theme` — the same reference-not-work vocabulary the app uses in `styles/practice/ladder.css`. Deliberately not collapsed: the whole request is that scrolling up shows the solved problem, so it has to be legible without a click.
  - Verified without the extension, which cannot be side-loaded on this machine: the real `colab_focus.js` and `colab_dd.css` were run against a synthetic DOM built from the generated cell ids. Focused on 481 the visible set is example → example-code → problem → code → check (solution still hidden until `reveal`); focused on its neighbour 482, which has no authored pair, it is problem → code → check exactly as before, with 481's example correctly gone.
- 2026-08-02 (later): **Scrolling no longer loses the problem, and the setup
  cell is off screen.** Two independent things the notebook was doing to the
  learner.
  - *Focus followed the fragment, and Colab owns the fragment.* Colab rewrites
    `#scrollTo=` to whatever cell is at the top of the viewport as you scroll —
    its own deep-linking feature. `apply()` read that straight, so scrolling up
    to re-read the checker (`dd-checker`) or down past the last cell of the
    problem left the fragment naming a cell that belongs to no problem, focus
    switched off, and the whole notebook unfolded under the cursor. The
    fragment is now a way to CHANGE the target, never to clear it: `focusTarget`
    keeps the last problem that resolved and only replaces it when a fragment
    names a different one. The panel routes by rewriting the fragment, so the
    one path that must keep working is exactly the one that still does.
  - *The setup cell was three dead variables at the top of every lesson.* It
    carried `DD_TOKEN` and a backend URL for a completion beacon that does not
    exist on this route and cannot — the panel learns a result by reading the
    line `dd_check` prints. `scripts/generate_colab_notebooks.py` now emits only
    `DD_LESSON_ID`, and `html.dd-theme .cell.dd-setup-cell` hides what is left:
    `colab.js`'s `identify` reads that id off the rendered text, and text in a
    `display: none` cell is still text. It stays in `ALWAYS_VISIBLE` — that is
    focus's business, and un-theming brings the cell back.
- 2026-08-02: **The theme hides Colab's left rail, not just its drawer.** The
  rule was `colab-left-pane.colab-left-pane-open` — and that class is only on
  the element while the drawer is EXPANDED. Closed, which is how a notebook
  opens, the element keeps rendering its 48px icon strip
  (`.colab-left-pane-nib`: contents, find, snippets, secrets, files), so the
  rule hid the pane in the one case the learner had opened it deliberately and
  left the strip on screen the rest of the time. Now `colab-left-pane` outright.
  Verified against a live notebook: the cell list reflows to the full width and
  all 668 cells stay rendered.
- 2026-07-31: **The answer stays hidden until the learner has answered.**
  `colab_focus.js` tags every `dd-q<n>-solution` cell `.dd-solution`,
  `colab_dd.css` hides it under the new `html.dd-hide-solutions` root class, and
  the panel's verdict click unhides exactly one — `dd:reveal-solution` →
  `colab.js`'s switch → `window.__ddFocus.reveal(n)`. Asked for as "then and
  only then it shows you the solution … below what you typed". A collapsed
  notebook cell was not enough: it still printed "💡 Solution — Problem 480"
  under the code you were trying to write.
  - **Why it cannot live in the notebook.** Colab renders every cell's output in
    a sandboxed iframe, so no CSS or JS a cell emits can reach a sibling cell.
    Hiding one cell from another is only possible from a content script.
  - The problem number is re-validated here (`/^\d+$/`) even though the panel
    already checked it — the panel forwards what the framed page sent.
  - **Running the check is the other way in.** `dd_check` prints its verdict as
    plain stdout, which Colab renders as text in THIS document — so
    `colab_focus.js` reads the same line the learner reads, opens that
    problem's answer, and forwards `dd:check-result` to the panel, which hands
    it to the app as the verdict the learner would otherwise have clicked. That
    is the whole reporting channel: a cell's rich output is sandboxed away from
    the page, and a beacon would need a token pasted into the notebook.
    `scripts/watch.py` grades the printed wording against the pattern here, so
    the two cannot drift apart.
    - Reported per problem, deduped on the exact line, and **the first pass
      only records what is already on screen** — a notebook reopened with its
      saved outputs would otherwise replay every grade in it on load, and
      unlock answers to problems the learner has not looked at.
    - `textContent`, not `innerText`: this runs on every mutation of a notebook
      that mutates constantly. The cell's source is in there too, which is why
      the pattern requires the printed prefix — `dd_check(480)` cannot match.
  - Unlocks are per page-load, never persisted: reopening a notebook is how you
    get a clean run at a problem, and a remembered unlock hands you the answer
    before you start. **There is no "show solutions" toggle**, and there was one
    for about an hour — a switch that turns the exercise off is not a setting.
    Run the check.
  - `dd-checker` joined `dd-setup` as always-visible. It defines `dd_check`, so
    focus mode hiding it made every check cell below it a NameError — which
    reads as broken starter code, not as a missing prerequisite.
  - `check_css_is_opt_in` now allows `html.dd-hide-solutions` as a third scope,
    and requires every rule under it to also name `.dd-solution`: it is the one
    scope that is ON by default, so that pairing is what keeps it from touching
    a notebook that has nothing to do with Delta Drills.
- 2026-07-31: **Focus mode + the Delta Drills skin** (`colab_focus.js`,
  `colab_dd.css`). Hooking onto individual cells needed nothing new in the
  notebooks: the generator has minted `dd-setup` / `dd-lesson-<id>` /
  `dd-kp-<slug>` / `dd-q<n>[-hints|-code]` since the panel needed them for
  jumping, and Colab renders each as the DOM id `cell-<id>`.
  - 🔴 **Focus hides nothing unless the target resolved to real cells.** Blank is
    what a failed load looks like, and on ARENA's 458 un-id'd notebooks no match
    is the DEFAULT case. `dd-setup` is exempt in all cases — hide it and the
    answer cell dies on `NameError`, which reads as broken starter code.
  - Group membership matches `dd-q<n>` with a trailing boundary. A bare prefix
    puts `dd-q12` and `dd-q123` in one group and shows two problems at once.
  - Re-applies on `hashchange` (opening the next problem changes only the
    fragment, which is not a navigation) and on a debounced MutationObserver
    (Colab mounts cells long after load).
  - Covered by `This-Directory-Only/scripts/test_colab_focus.mjs` — 13 assertions
    against a stub DOM, because the real page is behind a Google login that the
    browser checks cannot reach. `watch.py` gained
    `check_focus_cannot_blank_the_notebook` and `check_css_is_opt_in`; both were
    negative-tested by reintroducing the fault.
- 2026-07-31: Added `dd:identify` — notebook identity and mount state, so the
  panel can switch notebooks between problems.
- 2026-07-31: Initial build — `dd:ping` / `dd:goto` / `dd:read-output`.
  Pass 1 of `docs/spec-colab-its-surface.md`.
