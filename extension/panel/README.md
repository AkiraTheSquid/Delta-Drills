# panel

## Purpose
The side panel itself. Two of them, and the distinction matters:

- **`app.html` — the default, and the whole panel.** A `<style>` block and one
  `<iframe>` over `https://delta-drills.vercel.app/`. No script. The student sees
  the live site, with the site's own tabs and the site's own Google sign-in. MV3
  will not take a URL as `side_panel.default_path`, which is the only reason a
  local page exists at all.
- **`panel.html` + `api.js` + `navigate.js` + `panel.js` — the hand-built tutor
  UI.** A narrow one-problem-at-a-time surface that drives Colab notebooks
  directly. Works, but nothing points at it; open it by URL or repoint
  `default_path`.

The rest of this file is about the second one unless it says otherwise.

## Owns
- The view state machine: `connect → loading → gate → problem → settled`, plus `error`.
- The countdown, its expiry-means-wrong rule, and the settle window.
- The HTTP client for the Delta Drills backend, and the stored base URL + JWT.
- The bridge to the Colab tab (`chrome.tabs.sendMessage`).

## Does NOT own
- Any mastery decision — see `../README.md`. `nextQuestion()` *is* the scheduler;
  this folder renders its answer.
- Colab DOM knowledge. Cell lookup, section expansion, notebook identity and
  output reading live in `../content/colab.js`; the panel only names an anchor,
  a fallback string, and a URL.
- The `question → notebook` map. That is compiled by
  `scripts/generate_colab_notebooks.py` alongside the notebooks themselves, so
  the map and the anchors can never drift apart.

## Key Files

**`app.html`** — no scripts, so none of the load-order hazards below apply. Two
things in it are load-bearing and neither is obvious:

- `allow="identity-credentials-get; …"` on the frame. Sign in with Google
  negotiates over FedCM, and FedCM inside a cross-origin frame is denied unless
  the embedder delegates it through Permissions Policy. Drop the attribute and
  the button still renders — it just stops working.
- The absence of everything else. A toolbar, a settings sheet or a "which app
  address" input is a second front end growing back, and it competes with the
  site's own navigation. An earlier pass had all three; they were removed.

**The tutor UI.** Load order is a dependency chain, not a style choice — each
file destructures the previous one's global at its top level, **under an alias**:

1. `notebook-index.js`: **generated**, `window.DD_NOTEBOOKS`. The
   `question → lesson → file` map. Never hand-edit; rerun
   `scripts/generate_colab_notebooks.py`.
2. `api.js`: `window.DD = {api, tab, notebooks, store, ApiError}`. `api` wraps
   the backend, `tab` wraps talking to (and navigating) the Colab tab,
   `notebooks` resolves which notebook a question lives in and what URL opens
   it. Nothing here renders.
3. `navigate.js`: `window.DDNav`. Switching notebooks and finding cells —
   `jumpTo`, `ensureNotebook`, the notebook row and the Settings list.
4. `panel.js`: the view state machine and all event wiring. Imports as
   `api: ddApi`, `slugKc: navSlugKc` and so on — **never under the original
   name**. These are classic scripts sharing one global scope, so a bare
   `const {api} = window.DD` redeclares api.js's own `const api` and kills the
   entire file at parse time, leaving a blank panel.

Plus `panel.html` (every view as a `<section id="view-…">`, toggled by class)
and `panel.css` (dark, sized for a ~360px Chrome side panel).

## Data & External Dependencies
- `NextQuestionResponse` from `backend/app/practice_schemas.py`. The fields this
  panel actually reads: `question_id`, `question_text`, `subtopic`, `topic`,
  `starter_code`, `hint`, `lesson_gate[]`, `ladder_stage`, `ladder_kc`,
  `ladder_kc_title`, `ladder_estimate{n, correct, p, ci, worked_seen}`.
- `chrome.storage.local` keys: `dd_base`, `dd_token`, `dd_nb_repo` (the
  `owner/repo[@branch]/[path]` used to compute Colab URLs) and `dd_nb_urls`
  (the `lesson_id → URL` map learned from tabs that actually opened).
- `window.DD_NOTEBOOKS` from `notebook-index.js`.
- No libraries. Plain scripts, no modules, no bundler.

## How It Works (Flow)
1. `boot()` loads the stored base + token. No token → the connect view.
2. `advance()` calls `/next-question`. A non-empty `lesson_gate` routes to
   `renderGate`, otherwise `renderProblem` and the countdown starts. Both paint
   the notebook the target lives in, then fill in `open` / `switch` once
   `dd:identify` comes back from the tab.
3. `goToCurrentCell()` hands the target to `jumpTo`, which switches notebooks
   first if it has to and then sends `dd:goto` with `dd-q<question_id>` plus a
   `Problem <id>` text fallback.
4. `grade(correct)` posts `/submit-local-eval`, shows the verdict, and starts the
   settle countdown. `Undo` posts `/override` with the opposite value.
5. Settle reaches zero → `advance()`.

## Invariants & Constraints
- **`state.graded` guards `grade()`.** The button click and the timer expiry can
  fire on the same problem; two `/submit-local-eval` posts would log two attempts
  against the ladder for one question.
- **`clearTimers()` before every view change.** A leaked interval keeps counting
  behind the new view and will auto-grade a problem the student is not on.
- `api.js` must not touch the DOM, and neither `panel.js` nor `navigate.js` may
  call `fetch`. That split is what keeps the backend contract in one readable file.
- **Never inline a Colab URL.** Everything goes through `notebooks.urlFor`, or
  the repo setting and the remembered-URL map stop being the source of truth for
  where a notebook is.
- **Navigate, then wait, then jump.** Colab mounts a notebook well after the
  content script starts answering, so a jump issued straight after navigation
  finds nothing and reports a missing anchor.
- The render sequence (`state.nav`) guards the notebook badge: identifying a tab
  is a round trip and the student can be on the next problem by the time it
  returns.
- 401 from any call means sign out and return to connect — never retry silently.
- Rung scaffolding renders **here**, never by mutating notebook cells. The
  notebook stays an artifact the student can fork and keep.

## Extension Points
- New endpoint → a method on `api` in `api.js`.
- New view → a `<section id="view-x">` in `panel.html` plus `"x"` in `VIEWS`.
- New way of locating a notebook → a lookup in `notebooks.forQuestion`, or a
  new branch in `notebooks.urlFor`.
- A view that needs the notebook row → add `#<prefix>-nb-title` and
  `#<prefix>-nb-state` to `panel.html` and call `paintNotebook(prefix, …)`;
  `watch.py` checks the pair exists.
- Configurable timer → `DEFAULT_SECONDS` / `SETTLE_SECONDS` at the top of
  `panel.js`; both are placeholders pending real numbers from the bank.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Panel state is lost when the panel closes** — `ACTIVE`
  - When it happens: the student closes the side panel mid-problem.
  - Symptom: reopening starts a fresh `/next-question`; the in-flight countdown is gone.
  - Root cause: the panel is a page, not a worker — closing it tears down all state.
  - Prevention/fix: acceptable for now (an abandoned problem logs nothing). If it
    needs to survive, the timer deadline has to move into the service worker,
    not into `chrome.storage`, or a closed panel will still auto-grade.

- **Timer keeps running across a view switch** — `RESOLVED`
  - When it happens: any transition out of the problem view.
  - Symptom: a problem the student already left gets auto-marked wrong.
  - Root cause: `setInterval` handles are not tied to view lifecycle.
  - Prevention/fix: every path that changes view calls `clearTimers()` first.
    Keep that true when adding a view.

- **A switch lands on the wrong notebook** — `ACTIVE`
  - When it happens: the repo setting points somewhere the files are not, or a
    remembered URL was learned from an unrelated notebook.
  - Symptom: "That link opened `<other>`, not `<target>`."
  - Root cause: `urlFor` prefers a remembered URL over a computed one, and a
    remembered URL is only as good as the tab it was learned from.
  - Prevention/fix: **Forget remembered notebook links** in ⚙ clears the map;
    the Settings list shows `remembered` vs `repo` per lesson so you can tell
    which route is in play before clearing anything.

- **A blank panel with a dead ⚙** — `RESOLVED`
  - Cause: a bare (unaliased) destructure at the top of `panel.js`. See Key
    Files. `node --check` per file does not catch it; concatenate the four
    scripts and check that, which is what `../watch.py` approximates.

## Recent Changes
- 2026-07-31: The panel is the live site. `app.html` reduced to one framed
  `<iframe>`; the toolbar, settings sheet and fallback screen that wrapped it —
  and `app.js`/`app.css` with them — are deleted. Added
  `identity-credentials-get` so Google sign-in works in the frame. Verified at
  400px against the live deploy.
- 2026-07-31: Fixed the parse-time redeclaration that rendered `panel.html`
  blank; every import in `panel.js` is now aliased.
- 2026-07-31: Cross-notebook transitions — `notebooks` resolver in `api.js`,
  routing split out into `navigate.js`, the 📓 row in the problem and gate
  views, and the Settings notebook list.
- 2026-07-31: Initial build. Connect / gate / problem / settle flow against the
  live backend. Pass 1 of `docs/spec-colab-its-surface.md`.
