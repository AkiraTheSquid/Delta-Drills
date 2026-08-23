# Delta Drills

This repo is set up with two git worktrees:

- Local dev: `/home/stellar-thread/Applications/Delta-Drills-Local` (branch: `main`)
- Production deploy: `/home/stellar-thread/Applications/Delta-Drills-Deployed` (branch: `deploy`)

The Vercel production branch is `deploy` and the public URL is:

- https://delta-drills.vercel.app

Direct page links use the ordinary app with global navigation hidden. Paths are
case-insensitive; `/Knowledge-Graph` is the portfolio-facing default. Other
routes: `/Why-This-App`, `/Courses`, `/Practice`, `/Notebooks`,
`/Targeted-Practice`, `/Account`, and `/Split-Tool`. Each solo page keeps a faint
"Open full app" exit in its top-left corner. (`/How-It-Works` was removed with
the page it served on 2026-08-22 and now falls through to the full app.)

## Local development

1) Backend (FastAPI):

```
cd /home/stellar-thread/Applications/Delta-Drills-Local/backend
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2) Frontend (static):

```
cd /home/stellar-thread/Applications/Delta-Drills-Local
python3 serve.py 5173
```

3) Open the UI:

- http://localhost:5173/

The API base defaults to `http://localhost:8000` automatically when running on localhost — no manual configuration needed.

## Practice (adaptive learning)

- The Practice tab is available to everyone. A visitor with no account is given
  one automatically by `guest-session.js` — see **Guest sessions** below.
- Practice uses the backend algorithm via `/api/practice/*`.
- Progress is persisted per user on the backend at `backend/user_data/`.

## Guest sessions

Nobody has to sign in. On first load `guest-session.js` mints an account
(`POST /auth/signup`, a random `guest-<hex>@guest.delta-drills.app` address and
a random password) and keeps the credentials in that browser's `localStorage`.

This exists because the interesting half of the app is backend-only. The
placement diagnostic, the lessons, the stage ladder and the BKT student model
are all guarded by `practiceMode === "backend"` in `practice/`, and
`getPracticeMode()` answers `"local"` for anyone without a token — so "guest"
used to mean a static drill pool with no placement, no lessons and no mastery.
Giving the visitor an account was a much smaller change than porting four
backend modules into the browser and then keeping two copies of the pedagogy in
step.

Consequences worth knowing:

- **`authToken` no longer means "a person signed in."** It means "this session
  can call the backend". Identity questions — the guest banner, the topbar
  email, the Account tab, which tab the app lands on — go through `isSignedIn()`
  in `app.js`. Capability questions still ask `authToken`.
- **The guest credentials are never deleted, not even by Log out.** Signing in
  with Google and then signing out returns you to the same guest account with
  the progress it already had.
- **Progress is per browser, not per person, and signing in does NOT carry it
  over.** Google sign-in mints a different backend account and there is no merge
  or link endpoint, so what someone did as a guest stays with the guest account.
  The banner and the landing page say so in those words; they used to promise
  "continue with Google to keep it across devices", which was not true. Closing
  the gap properly is a backend change — a guest-to-user claim endpoint — and is
  the open follow-up here.
- **A transient backend failure must not orphan a guest.** `postAuth` reports the
  HTTP status, and only a **401** on `/auth/login` — the backend saying these
  credentials are not an account — mints a replacement. A 429, a 5xx or an
  offline browser leaves the stored credentials alone and falls back to local
  mode for that load. Overwriting them on any failure is how one bad minute
  costs a learner every attempt they have ever made.
- **Failure is not fatal.** If the backend cannot be reached nothing is stored,
  the mode stays `"local"`, and the app behaves as it did before this existed.
- **This is not a security boundary.** Signup was already open to anyone; this
  only skips the form. It does mean one row per browser that visits.

## Notes + fixes added

- Root UI (`/home/stellar-thread/Applications/Delta-Drills-Local/index.html`) now includes the Practice tab/page and loads `frontend/practice.css`, `frontend/practice.js`, and Pyodide. Previously the root UI did not include the Practice page, so the tab never appeared when serving from the repo root.
- Practice is auth-gated in the root UI switch logic, matching the rest of the app.
- `python3 -m http.server` is required on this system (plain `python` may not exist).
- 2026-04-29: added auth-gated `Courses` tab (`data-tab="courses"` → `#page-courses`) between Course and Papers. Page hosts two mutually-exclusive views inside the container — `#courses-list-view` (search input + filtered results) and `#courses-detail-view` (article view rendered on click). `courses.js` owns navigation: clicking a course card swaps to the detail view, the in-article "Back to courses" button swaps back. Seeded with one course — ARENA Curriculum — including a hero block, intro paragraph, and per-chapter cards (Fundamentals, Transformer Interpretability, RL, LLM Evaluations, Alignment Science, Capstone) each with its illustration sourced from `images.squarespace-cdn.com`. Styling lives at `styles/courses.css`.
- 2026-04-29: added per-course "Include course for study?" Yes/No toggle. Renders on the right side of each list card and at the top of the detail article (above the hero). State persists in `localStorage` under `delta_drills_courses_include` as `{ courseId: "yes" | "no" }`. List and detail controls mirror each other via `syncIncludeControls()`; clicks on the toggle do not propagate to the card-open handler.
- 2026-04-29: courses detail — chapter cards with a `sections` array become clickable and open a pure-information modal listing the chapter's sections (number badge + title + description). Backdrop/×/Esc all close. Per-chapter accent color is passed via inline `--section-number-color` on the modal root. Currently only Chapter 1 (Transformer Interpretability) has section data populated; other chapters render as static rows until their data lands. `openChapterModal()` / `closeChapterModal()` live in `courses.js`; styles in `styles/courses/`.
- 2026-04-29: courses CSS reorganized — the single `styles/courses.css` was split into five focused fragments under `styles/courses/` (`list.css`, `include.css`, `detail.css`, `modal.css`, `responsive.css`) so each file stays small. `index.html` now links all five between `styles/arena.css` and `styles/responsive.css`. Per-fragment selector ownership is enforced by `styles/courses/watch.py`.
- 2026-04-29: chapter-sections modal data filled out for the remaining ARENA chapters. Each chapter now has its own per-chapter accent color sourced from learn.arena.education: Ch0 Fundamentals `#DC2626`, Ch1 Interpretability `#D97706`, Ch2 RL `#059669`, Ch3 Evals `#2563EB`, Ch4 Alignment Science `#4F46E5`. Capstone Project still has no `sections` array (it's an open-ended project, not section-based) and renders as a static row. `watch.py` invariants now lock one representative section title plus the hex color per chapter.
- 2026-07-31: Courses tab simplified to ARENA-only and rewired to upstream Colab. There is exactly one course, so the list view is gone: `#courses-list-view`, `#courses-search`, `#courses-results`, the "Include course for study?" toggle (and its `delta_drills_courses_include` storage key), and the "Back to courses" button were all deleted. `#courses-detail-view` is now the whole tab and renders on load. The hero carries a `.course-sources` row linking Callum McDougall's repo (`github.com/callummcdougall/ARENA_3.0`), `arena.education`, and the published curriculum book.
- 2026-07-31: Courses section links now open Callum McDougall's original Colab notebooks instead of the local `arena-book/` HTML mirror. `notebookPathForBookUrl()` maps each `/arena-book/<path>.html` to `<path>.ipynb` and hands it to `colabUpstreamHref()` from `stats/predicted-links.js`, which is the app-wide helper that swaps the repo owner to the student's fork when `account_github_username` is set. All 31 section URLs were verified to resolve against links already present in the book mirror. New `courses-fork-gate.js` intercepts the *first* Colab click and offers the one-time setup: fork `ARENA_3.0`, then enter a GitHub username. Saving writes the same `account_github_username` key the Account tab uses, fills in that field if present, and fires `courses:github-owner-changed` so already-rendered links repoint without a re-render. Skipping sets `delta_drills_courses_fork_prompted` so the dialog asks only once. A username alone grants no write access — the fork click is what duplicates the notebooks, so the dialog walks the student through it rather than pretending to do it server-side. Fork existence is checked against the GitHub API advisory-only: 404 warns but still allows "use it anyway", and a network failure never blocks opening the notebook.
- 2026-07-31: Statistics tab removed. The nav is now How It Works / Knowledge Graph / Account / Courses / Practice / Targeted Practice. The tab's priority, learning-rate and predicted-course-score tables were built on assumptions the learner had no way to check, so they read as authoritative without being verifiable — cognitive load without signal. Deleted: the tab button, `stats.js`, and `stats/{stats-dom,dom,data,render,graph,predicted-data,predicted,init}.js`; `styles/stats.css` trimmed to the shared `.stats-bar*` rules. Three files under `stats/` were KEPT because other tabs read their globals — `weights.js` (practice adaptive weighting), `predicted-links.js` (`colabUpstreamHref`, used by Courses / Practice / Targeted Practice), and `predicted-prereqs-temp.js` (ARENA unlock gating); see `stats/README.md`. Backend endpoints (`/api/practice/subtopics`, weight sync) were left in place. Side effect handled: the tab's Advanced table was the only UI for disabling a topic/subtopic, so `practice/questions.js` now ignores saved disable flags when they would empty the question bank rather than dead-ending a learner who can no longer undo them.
- 2026-08-07: **The app is called ARENA Delta Drills, there is a "Why This App Exists" tab, and every tab and feature carries a ⓘ.** The topbar logo and `<title>` both lead with ARENA now, and the Courses tab opens with a `.courses-about` block stating plainly what the app is for — one curriculum, ARENA, drilled for — with a line that names which of the two deploys you are on (`.cae-app` / `.cae-colab`, switched by `html.dd-colab-edition`; the two Vercel projects ship the same bytes, so this has to be a CSS switch). New leftmost tab `data-tab="why-this-app"` → `#page-why-this-app`; **its copy is deliberately unwritten** — the page ships as a `.wta-placeholder` reading "Coming soon" so the tab and route exist, and the argument lands in a later pass. It carries neither `.auth-only` nor `.guest-only`, so it is the one tab visible in both states without a `guestVisibleTabs` entry.
  The ⓘ system is three new files: `infotips-registry.js` (copy), `infotips.js` (behaviour), `styles/infotips.css` (the dot + the panel). To give anything an explanation, put `data-dd-info="<key>"` on it and write the key in the registry — placement defaults to inside the element, `data-dd-info-place="after"|"before"` puts the dot beside it, and a `<button>`/`<a>` anchor is forced to "after" because a button cannot nest a button. Dots are re-derived from a debounced MutationObserver, not injected once, because `targeted-practice-dom.js` and `arena-unlock-dom.js` write their pages at runtime and things like the stage dots are `innerHTML`-replaced on every question. Two traps are worth knowing before editing it: the **tab** dots are hand-written in `index.html`, not injected, because `app.js` captures `.auth-only`/`.guest-only` into static NodeLists at eval time and a dot injected later would stay visible beside a hidden tab; and `dotFor()` matches on the anchor's **key**, not merely on "is a dot" — `.kg2-controls` holds one anchor and contains another, so class-only matching found the wrong dot, minted a second, and since minting mutates the DOM the observer re-scanned and did it again, reaching ~1800 dots before it was caught. A third: an anchor can **move**, and its old dot does not follow. `concept-graph/kc-colab-route.js` relocates `#kg-colab-link` and `#kg-maximize` out of the info aside into `.kg2-controls` on the Colab edition, which stranded their dots in the aside while a fresh pair appeared in the new parent — caught on the live Colab deploy, not locally, because that route only fires on that host. Each scan now sweeps generated dots no anchor still claims, which also covers an anchor that is removed outright. `watch.py`'s new `check_infotips` asserts the anchor set and the registry key set are equal in both directions, since a key with no anchor and an anchor with no key both fail silently.
  Two pre-existing bugs surfaced and were fixed on the way: `.topbar-auth[hidden]` did nothing (a class rule's `display: flex` beats the UA `[hidden]`), so the green "signed in" dot had been showing to guests; and with the indicator genuinely hidden, `space-between` slid the whole tab strip to the right edge, so `.tabs` now takes `margin-right: auto` and `.tab` takes `white-space: nowrap`.
- 2026-04-29: chapter-modal visual pass — modal background switched to `--bg` (less saturated than `--card`), section-row bubbles removed (rows blend into the modal and pick up `--surface` only on hover), section number lost its bordered rectangle, scrollbar themed to match the dark UI, and overall row density tightened.
- 2026-08-18: **the app stopped looking like a website squeezed into a side panel.** Two changes, both aimed at the Chrome extension, which frames this site at ~300-400px (`extension/panel/app.html`) and so lives permanently in the layout everything here was least tested at.
  **Dark scrollbars** (`styles/scrollbars.css`, new, loaded with the base layer). Every scroller in a `#1a1a2e` app was drawing the platform's white track — a bright stripe down the right edge and another across the bottom, and in a 300px panel that bar is a visible fraction of the UI. `color-scheme: dark` on `:root` plus a full `::-webkit-scrollbar` skin. Both are needed and they are not duplicates: `color-scheme` reaches scrollbars CSS cannot (and keeps native controls consistent with the hand-darkened inputs in `components.css`), while Chrome **ignores** `color-scheme` for any scroller a `::-webkit-scrollbar` rule matches, so the skin has to be complete on its own — including `::-webkit-scrollbar-corner`, or a white square is left where the vertical and horizontal bars meet.
  **The tab strip became a hamburger menu below 900px** (`nav-drawer.js` + `styles/nav-drawer.css`, both new). Seven tabs that refuse to wrap were a sideways-scrolling sliver at panel width: two tabs visible, five off the edge, which reads as an app with no navigation. Now the topbar keeps one clean 56px row (hamburger, title, account) and the tabs are a vertical list in an off-canvas drawer. 🔴 **The strip is MOVED, never cloned** — `nav-drawer.js` does `drawer.appendChild(nav)` on the live `<nav class="tabs">` and puts it back at a comment-node marker on the way to a wide viewport. `app.js` captures `.tab`, `.auth-only` and `.guest-only` into static NodeLists at eval time and drives clicks, the `.active` highlight and guest/auth visibility off them; a copy is in none of those lists, so its tabs would not switch pages and would show Account and Split Tool to a signed-out visitor — and nothing would throw. `watch.py`'s `check_nav_drawer` fails on `cloneNode`, on the script loading before `app.js`, and on the 900px breakpoint drifting from the one in `styles/practice/layout.css`; `styles/watch.py` fails if `#nav-drawer` ever grows its own `.tabs` in the markup. In the drawer the strip lays out as a two-column grid so each tab and its ⓘ share a row without touching the markup — which works only because a tab and its dot are always hidden together. Also removed: the old `≤600px` rule that broke `.topbar` into a column. With the tabs gone it left an empty second row, and the taller header quietly invalidated `.practice-container`'s `height: calc(100vh - 56px - 40px)`.
  Two keyboard bugs were found and fixed on the way out, both reproduced in a real browser at 380px rather than reasoned about (`nav-drawer.js?v=3`). **The focus trap leaked.** It intercepted Tab only at the two edges of its ring, and the ring starts at the toggle — which is outside the drawer and *before* it in the DOM, so a forward Tab off the toggle was on nobody's edge, the browser handled it, and focus landed on the account controls behind the scrim. Only a signed-in visitor could see it: signed out, `.topbar-auth` carries `[hidden]` and there is nothing between the two elements to land on. Every Tab is now placed by index (`(index + step + ring.length) % ring.length`), which cannot leak because the browser never chooses; `watch.py` pins it. **A breakpoint change dropped focus on `<body>`**, so the next Tab restarted the page from the top. `appendChild` on an element containing the focused node blurs it, so "was the caret ours?" has to be read *before* the move — asking afterwards always answers no and no rescue ever runs. Crossing to wide now lands the caret on the active tab in the restored strip, crossing to narrow lands it on the toggle. What remains, and is not fixed: the page behind an open drawer is pointer-blocked by the scrim but never made `inert`, so a screen reader's virtual cursor can still reach it. Keyboard is contained; assistive-tech isolation would want a real modal boundary (`<dialog>`, or app-shell inert management) and is a larger change than this pass.

## UI layout note

- The root UI is now the primary interface (served at `/home/stellar-thread/Applications/Delta-Drills-Local/index.html`).
- The previous `frontend/` UI has been moved into `/home/stellar-thread/Applications/Delta-Drills-Local-Backups/ui-legacy-20260215-110811/frontend/` as a snapshot.

## Troubleshooting Practice tab

- Make sure you are serving `http://localhost:5173/` from `/home/stellar-thread/Applications/Delta-Drills-Local`.
- If the tab still does not show, hard refresh and confirm you are logged in.
- For the adaptive algorithm, the backend must be running on port 8000.

## Production deploy workflow

Make changes locally in the `main` worktree, then sync to the deploy worktree:

```
cd /home/stellar-thread/Applications/Delta-Drills-Local
./scripts/sync-deploy.sh
```

Review and push production:

```
cd /home/stellar-thread/Applications/Delta-Drills-Deployed
git status
# commit any deploy-only changes if needed
# then push deploy

git push origin deploy
```

## Sync back from deploy to main

If you made deploy-only changes and want to bring them back to local:

```
cd /home/stellar-thread/Applications/Delta-Drills-Local
./scripts/sync-local.sh
```

## Notes

- The backend is not deployed on Vercel. Only the static frontend is.
- `http://localhost:8000/` returns 404 by design. Use `/health` for checks.
- Keep deploy-only tweaks in the deploy worktree to avoid polluting local dev.

## Recent Changes

- **2026-08-22 — the app has levels, and the progress bar IS the topbar seam.**
  `xp.js` + `styles/xp.css` are new. The topbar carries a `Level N` chip between
  the logo and the tab strip, and the 1px `--border` line under `.topbar` is now
  a progress bar: `.dd-xp-seam` is absolutely positioned over it and fills from
  the left as the learner works. 🔴 **The bar renders no numeral** — no count, no
  percent, no `340 / 500`. The only number on screen is the level; the hover
  `title` on the chip is the one place the raw XP is available. That is the whole
  design brief, so a future "just show the XP on the bar" is a regression, not a
  feature.
  - **Placement is `bottom: -2px`, and that is not arbitrary.** `.topbar` is
    `box-sizing: border-box` at 56px with a 1px bottom border, so an absolutely
    positioned child is laid against a **55px** padding box — and `.tab.active`
    already owns its last 2px for the accent underline. At `bottom: 0` or even
    `-1px` the seam clips that underline. `-2px` puts the bar on the border row
    itself plus 1px over the page.
  - **Everything the learner enters pays, and it is wired at ONE place.** Every
    recording path in this app ends at a `PracticeAPI` method, so the awards are
    a wrapper block at the bottom of `practice/api.js` rather than award() calls
    sprinkled through six handlers in `events.js`, `colab_mode.js` and
    `diagnostic-page.js` — the next handler to be added gets XP for free. A MISS
    pays too (10 vs 25): the placement test is built out of misses, and a bar
    that only moved on a correct answer would charge the learner for using the
    feature that finds their level. Anything outside that chokepoint
    (`targeted-practice.js`, the lesson gate in `practice/lessons.js`) dispatches
    `delta:xp` instead, so no caller needs a load-order relationship with
    `xp.js`. A generic `input` tick (1 XP, throttled to 15s) covers typing.
  - **State is localStorage keyed by `auth_email`**, exactly like
    `practice/storage.js` — a guest keeps their levels, a signed-in account keeps
    theirs, and `delta:auth-state-changed` re-reads the store. No merge, because
    there is no merge anywhere else in this app and this would be the only place
    progress silently changed owner.
  - Two bugs found before shipping, both in a real browser. **The level-up snap
    painted a stale percentage**: `render()` captured `pct` when the award was
    made, so a second award landing during the 560ms hold was overwritten by the
    first award's remainder — which was 0%, so the bar emptied. It now re-reads
    `state` at paint time, and a plain repaint defers to a pending snap.
    **The typing tick was dead for the first 15 seconds of every page load**
    (codex caught this one): `performance.now()` counts from navigation, so a `0`
    throttle sentinel means "you already earned one". It is `-Infinity` now.

- **2026-08-22 — the app has three themes, and the Account page picks between
  them.** `theme.js` is new and owns the whole switch: it stamps
  `<html data-theme="blue|dark|light">` **synchronously from `<head>`, above the
  stylesheet links**, so a light-theme user never sees a frame of the dark
  palette, and on `DOMContentLoaded` it renders the radio group into
  `#account-theme-options`. The choice lives in `localStorage["dd_theme"]`, so
  it follows the browser rather than the account — no backend, no migration,
  and it works signed out. `window.DDTheme` exposes `get`/`set`/`themes`, and a
  `delta:theme-changed` event fires on set for anything that paints its own
  canvas.
  - **`blue` is the palette the app has always had and is the default**, both
    as `:root` and as `:root[data-theme="blue"]`; a visitor who never opens the
    picker sees exactly what shipped before. `dark` is Colab-style neutral
    grey, `light` is the same app on white. Every token is defined three times
    in `styles/variables.css` and `styles/watch.py` fails the folder if the
    three blocks ever disagree — a token defined in only some themes drops the
    declarations that use it, which is how white-on-white ships.
  - **The Account markup is two additions and nothing else**: the script tag in
    `<head>`, and a `.account-theme` block between `.account-mode` and
    `#account-form`. It sits outside the form on purpose, exactly like the
    advanced-mode toggle — it applies on change and has no Save. The three
    options are rendered by `theme.js` rather than written in the markup, so
    `variables.css` and that file's `THEMES` list stay the only two places a
    theme is named. `infotips-registry.js` gained the `account-theme` copy;
    `watch.py`'s `check_infotips` fails on an anchor with no entry, which is
    how the missing one was caught.
  - **The white-on-white problem was made loud rather than avoided.**
    `styles/watch.py` now bans `#fff`/`#ffffff`/`rgba(255,255,255,…)` in every
    stylesheet under `styles/`, `practice/` and `targeted-practice/`, with two
    mechanical exemptions: the `[data-theme-preview=…]` swatches on the Account
    page (miniatures of themes the viewer is *not* in, so `var()` would draw
    all three identically) and lines marked `/* graph-legend */` (they must
    match Cytoscape node colours painted from JS). Verified in a real browser
    across every page with transitions disabled: **light 0 AA failures, dark 0,
    blue 9** — and all nine blue findings pre-date this work, which is the
    evidence the default palette did not move. 🔴 Disabling transitions is not
    optional when auditing: `.tab` and `.dd-info` animate `color`, and sampling
    mid-transition reported white-on-white at ratio 1.0 for elements that were
    fine.

- **2026-08-22 — the How It Works tab was deleted, and guests stopped getting a
  cut-down app.** Three changes that belong together, because the landing page
  now promises a diagnostic and a student model that a signed-out visitor could
  not previously reach.
  - **How It Works is gone** — tab, ⓘ, `<main id="page-how-it-works">`, its
    `/How-It-Works` solo route, and both `watch.py` tab lists. Its content was
    the long Math-Academy-style mechanism essay; the argument for the app lives
    on **Why this app exists** instead. `styles/how-it-works.css` was NOT
    deleted or renamed: it also owns every `.kg2-*` Knowledge Graph rule, and
    several `concept-graph/` modules and READMEs name it by that path. Its three
    `#page-how-it-works` selectors were repointed at `#page-why-this-app`, which
    inherits the `.hiw-*` type scale.
  - **Why this app exists is the landing page** and carries real copy: what the
    app is (a Khan Academy / LeetCode for the ARENA curriculum, drilling the
    PyTorch prerequisites), the "machine learning to teach machine learning"
    claim, and the six-step loop it runs. `app.js` lands a visitor here only
    when there is no stored token — i.e. their FIRST visit. A returning guest
    has a token and lands on Practice, because they have already read this.
  - **`guest-session.js` gives a signed-out visitor a real backend session**
    (see **Guest sessions** above), so the diagnostic, the lessons and the
    mastery model work with no sign-in. Verified end to end against the Fly
    backend from a cleared browser: guest provisioned, `practiceMode` came up
    `"backend"`, Practice served a real question, the placement diagnostic
    started and its progress survived a reload.
