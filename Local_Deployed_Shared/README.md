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

- **The topbar is chrome and nothing else** (2026-08-23). No brand line, no version tag, no signed-in email: the tab strip owns the right edge, the level pill sits on the left beside the hamburger, and `.logo` survives only as an empty flexible spacer that `nav-drawer.js` needs below 900px. The bar's height is `--dd-topbar-h` in `styles/layout.css` — 🔴 set it there, not as a literal, because `#page-practice` and `.nav-drawer-head` size themselves against it.
- **What is in the bar, left to right** (2026-08-27): the hamburger, the **level pill**, the tab strip (advanced mode only), the **concept pill** — a third of the viewport wide, naming the concept under test and filling as the learner works — and the account control. The **session clock hangs UNDER the bar as a notch**, a direct child of `<header class="topbar">` positioned against it. 🔴 Two of these are progress bars that fill; they are told apart by size and label, not by hue, so do not give them the same gradient. 🔴 `.topbar-side` must keep `min-width: auto`: both side tracks are `1fr`, and waiving their automatic minimum lets the middle cell size them below their contents, which overlap instead of shrinking (neither can — both are `nowrap`).
- ⚠️ **`.footer` is styled and never rendered.** `styles/layout.css` has a fixed 40px `.footer` rule and no `<footer>` element exists anywhere in `index.html`. `#page-practice` used to subtract 40px for it, which drew a strip of bare page background under the practice tab; that reserve is gone. The rule is left in place — deleting it is a separate call — but nothing should size itself against it.

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

- **2026-08-27 (later) — the concept pill stopped flickering: it is down when
  there is no question on screen, and a resume no longer loses the concept.**
  Changed: `concept-pill.js`, `practice/timer.js`, `watch_concept_pill.py`.

  Seth: "whenever I first joined the page the top bar wasn't there and then it
  took like a second later before the top bar appeared. And then once I pressed
  the button to continue practice, the top bar completely disappears … the top
  progress bar only appears again after going to the next problem." Two
  separate causes, one visible symptom.

  🔴 **Resuming a paused session threw the ladder fields away.**
  `timer.js:_restoreSavedQuestion` rebuilt the saved question with
  `buildPracticeQuestionFromBank`, and the BANK has no `ladder_kc` /
  `ladder_stage` / `ladder_kc_title` — those are per-served-question, from the
  backend queue. So `renderQuestion` got a question with no concept on it,
  `LadderUI.decorate` found no kc, and `StageLadder.hide()` took the concept
  off the screen for the whole of the resumed question: the topbar chip, the
  ladder card, and the heading (which fell back to the subtopic) together. It
  came back at the next question because that one is served by the queue. The
  fix prefers `hydrateSavedPracticeQuestionFromBank` — the function built for
  exactly this pair, which spreads the saved question first and then overwrites
  every artifact field from the bank, so the bank stays authoritative for the
  question and only the fields it has no opinion about survive. Gated on the
  saved question being THIS question, and it falls back to the plain build.

  🔴 **The chip drew whether or not a question was on screen.** The ladder
  publishes per QUESTION and knows nothing about tabs or the idle screen, and
  it renders once in the background at load (timer.js `start()`: "nothing about
  the one rendered in the background at init is recorded") — which is the
  second-late pop-in on a cold load, for a question nobody had asked for. It
  also stayed up across the Notebooks and Account tabs. `concept-pill.js` now
  reads the same two facts `styles/practice/timer.css` uses to hide the ladder
  card itself: `#page-practice.hidden` (another tab) and
  `#page-practice.session-idle` (the practice page with no question on it,
  which a lesson page correctly clears). A CSS rule cannot do this — the chip
  is in the topbar, a sibling of every page, so no selector rooted at
  `#page-practice` reaches it. A MutationObserver on that one class is the
  other half: a pause and a tab switch change nothing about the READING, so the
  ladder never fires for either.

  🔑 `hidden` is also the class the chip puts on ITSELF, so the guards are
  scoped to `_onScreen` and to the observer block rather than to the file.
  11 new mutations, all biting.

  ⚠️ Found while fixing this, NOT fixed: `practice/kc-practice.js` builds its
  queue items with the same bank builder, so the single-KC lesson ladder has
  the same gap. It supplies its own `StageLadder.show`, so it does not show the
  same symptom; it is still a second place where a rebuilt question has no
  ladder fields.

- **2026-08-27 — the concept under test moved into the topbar as a bar that
  fills, and the session clock went back to being a notch.** New:
  `concept-pill.js`, `styles/concept-pill.css`, `watch_concept_pill.py`.
  Changed: `index.html`, `practice/stage-ladder.js`, `styles/layout.css`,
  `styles/practice/notch-menu.css`.

  Seth: the concept "should be on the top bar instead of on the left … a pill
  at the top that gets filled up as you make progress … and it won't have the
  other complicated thing where it has the different stages", and "the timer is
  below the top tab and the timer is like kind of a notch". On the first build
  the pill was the level pill's size and the note back was "much bigger …
  probably one-third of the screen for the top bar should be devoted to the
  progress bar", without growing the bar's height.

  - **The pill is the level pill's twin** — one grid cell, a gradient fill, and
    the label drawn TWICE (base colours off the fill, `--on-accent` clipped to
    the filled width). See `styles/xp.css` for why one text colour cannot work.
    It is `flex: 0 1 33vw` with `min-width: 0`, 34px tall in a 44px bar, and it
    is the item that gives way when advanced mode's tab strip shares the cell.
  - 🔴 **It computes nothing.** `practice/stage-ladder.js` fires
    `dd-concept-progress` at the end of every render (and from `hide()`), and
    the pill draws the `pct` it is handed. The ladder's arithmetic — Wilson
    lower bound against the promotion threshold, or the promotion streak, over
    four rungs — stays in one place. A second implementation would be a second
    answer to "how far into this concept am I", which is what the ladder exists
    to have ended.
  - 🔴 **`pct: null` is not 0.** No reading (an unknown rung, a KC-less item)
    draws an empty dashed track and says so in the tooltip; 0% is a claim about
    the learner. A "skip the write if it has not changed" cache on `pct` cannot
    work for the same reason — `null` is both a legal value and the cache's
    reset state, and it collapsed on exactly the case it must not skip. Caught
    in the browser, not in review.
  - 🔴 **The fill tops out at 75%**, inherited from `_overall()` unrescaled:
    reaching the Solo rung is not being done with the concept.
  - **The heading card in the left panel is `display: none` from 621px up**
    (`body .question-number-row` — out-specifying `question.css` rather than
    out-ordering it), so the concept is named exactly once at every width. The
    card comes back under 620px, where the pill is dropped rather than shrunk
    into an ellipsis. The card and the four-rung ladder inside it are HIDDEN,
    not deleted: `practice/ui.js` still writes `#question-number`,
    `stage-ladder.js` still reads it back, and the ladder's render is what feeds
    the pill.
  - **The notch hangs off the topbar**, not off `.practice-container` the way it
    did before 2026-08-24 — so it is still app chrome that never re-parents and
    never travels with the workspace. Bottom-only corners, no top border, the
    bar's own border shared. Every id is unchanged, so `practice/timer.js`,
    `practice/placement-timer.js` and `practice/notch-menu.js` are untouched.
  - 🔴 **`.topbar-side` lost `min-width: 0`.** That waived the two `1fr` side
    tracks' automatic minimum, so they were free to size below their contents —
    which then overlapped rather than shrinking. Invisible until the middle cell
    got big: measured at 1440px in advanced mode, the left track collapsed to
    62px with the level pill hanging past its edge and under the tab strip.
  - ⚠️ **`watch_concept_pill.py` is not in `Local_Deployed_Shared/watch.py`'s
    list yet** — that file belongs to another live session. It needs
    `from watch_concept_pill import check_concept_pill` and the entry in
    `__main__`, or it only runs when someone runs it directly.
  - 🪦 Considered and NOT done: folding `styles/xp.css` and
    `styles/concept-pill.css` into one filled-pill primitive. They genuinely
    duplicate the fill, the clip, the two label layers and the transitions, and
    codex flagged it — but the two pills are being tuned independently right
    now and a shared contract would freeze that. Worth doing once the concept
    pill's size settles.

- **2026-08-27 — the "Review the graph" door reviews the graph the app teaches
  from.** `instructor-review.js`, `index.html`, `styles/instructor-review.css`,
  `concept-graph/lesson-graph.js`, `watch_instructor_graph.py` (new),
  `This-Directory-Only/backend/app/practice/graph_feedback_router.py`.
  Seth: it "is not utilizing the other graph that's significantly better… the
  one that's interactive that whenever you click on it it displays the lesson",
  and the door should make it "cover the whole screen, or not the whole screen,
  still leaving like the top bar". Also, explicitly, the **PyTorch** graph —
  "not the whole entire graph that's part of the other part that's connected to
  the entire curriculum… the hierarchical graph".
  - **It was drawing the dead graph.** The door built its OWN cytoscape over
    `concept-graph/graph-viz.json` — the ARENA 205-atom export, force-directed,
    no lessons behind its nodes, and **not wired into `index.html` anywhere
    else** since `lesson-graph.js` superseded it. So an instructor was flagging
    the sequencing of a curriculum no learner is served from. It now hosts the
    63-KC lesson graph: dagre, `rankDir: BT`, prerequisites underneath what they
    unlock.
  - 🔴 **Hosted, not copied.** `.kg-container.kg2` is MOVED into `#ir-kg-frame`
    and moved back — the same borrow `concept-graph/why-graph.js` makes for the
    landing page's maximise, and for the same reason: the lesson pane, the
    learner-model dock and the Practice hand-off all arrive for free and cannot
    drift, because it IS the Knowledge Graph tab. The element that moves is the
    `.kg2` **container**, never the `.kg2-wrap` inside it — `fitWrap()` looks the
    wrap up as `.kg2 .kg2-wrap` and the graph loses its height if the wrap is
    what travels.
  - 🔴 **Two borrowers, one graph.** `hostKg` takes it only when it is still
    `.closest("#page-knowledge-graph")`, because each borrower remembers its own
    home and whichever releases second finds nothing to put back — the tab is
    then permanently empty with nothing raised. Release covers all three exits:
    `show()` (the back button), the instructor-mode flag dropping, and a
    `MutationObserver` on the page's own `hidden` class, which is the only
    signal that `switchTab` navigated away (app.js fires no tab event).
  - 🔴 **A deferred host can outlive the view.** `openGraph` defers `hostKg` by a
    frame so cytoscape has a sized container, and a frame is long enough to
    leave in — a stale callback would then park the graph inside a hidden page.
    `hostKg` re-checks that the page and the view are still showing.
  - **Taps: bubbles stay the learner's, edges become the instructor's.**
    `lesson-graph.js` gained exactly one export, `window.deltaConceptGraphCy()`,
    and this file binds the EDGE taps that file has no handler for. A bubble tap
    is NOT stolen for a form — it opens the lesson, which is what an instructor
    is here to read — and the tapped concept is offered to a `Flag "<name>"`
    button instead. Handlers are bound once for the life of the page and gated
    on `hosting()`, so they are inert while a learner has the same graph.
  - **The flag card floats mid-left, not in a corner.** Both left corners are
    already taken by the hosted graph's own furniture: the no-data notice at the
    top, the mastery legend riding on the learner-model dock at the bottom. The
    dock is opaque at `z-index: 20` and cut the Send button clean off.
  - **The log now says which graph an id names** — `graph: "lesson-kc"`, a
    closed `Literal` beside `"arena-atom"`. `kc_registry.json` ids and ARENA atom
    ids do not overlap, and a mixed append-only log with no namespace cannot be
    read a month later. Optional, so every entry already on disk stays valid.
    ⚠️ **The backend half is not deployed** — the frontend works either way (an
    older backend ignores the extra field), but the namespace is only persisted
    once `graph_feedback_router.py` ships.
  - Verified in a browser, not by eye: overlay pinned at `top: 44px` full width,
    63 nodes / 81 edges hosted, bubble tap opens the lesson AND names the flag
    button, edge tap opens the edge form, the two-tap missing-edge flow names
    both concepts by label, and the graph returns home on all three exits —
    including straight from the hosted view to the Knowledge Graph tab, which
    renders at 866×665 with its own fit. `watch_instructor_graph.py` carries 13
    guards; **7 mutations were introduced one at a time and all 7 were caught**.
    Cache-bust: `instructor-review.js?v=2`, `instructor-review.css?v=2`,
    `lesson-graph.js?v=26`.

- **2026-08-27 — a feedback note is as tall as what you wrote in it.**
  `autogrow.js` (new), `index.html`, `instructor-review.js`,
  `styles/instructor-review.css`, `styles/practice/feedback.css`.
  Seth, in his words: the feedback box "doesn't have an enlarged text box in
  order to improve the questions with deeper, more in-depth feedback", it
  should expand "automatically" as you type "rather than forcing you to
  scroll down or increase the size of the text box", and it is "extremely
  wide" — make it "more like the 'why this app exists' and 'how this app
  works' pages in terms of its formatting for the width and text size".
  - **`autogrow.js` is the whole behaviour, wired by attribute.** Any
    `textarea[data-autogrow]` grows on `input`; the listener is DELEGATED on
    `document`, so the `.ir-q-note` that `instructor-review.js` builds per
    question card needs no registration. Three notes are on it today:
    `.ir-q-note`, `#ir-form-note`, `#problem-feedback-note`. Each keeps its
    own `min-height` in CSS — that is the empty box's size, a per-surface
    design choice, and it also means the script never needs a floor.
  - 🔴 **The stylesheets DEPEND on the script being loaded.** All three notes
    are now `resize: none; overflow-y: hidden`. With no script to size them a
    long note is clipped, with no scrollbar and no drag handle to escape
    with — strictly worse than the one-line box this replaced. The
    `<script src="autogrow.js">` tag and the CSS must ship together.
  - 🔴 **`input` alone is not enough, and `scrollHeight` excludes the
    border.** A note can hold text it never got a keystroke for — a restored
    value, or a box that was `display: none` at load and is shown later — and
    measuring one while hidden reads 0 and writes a height nothing corrects.
    So a `ResizeObserver` re-measures on any WIDTH change (0 → rendered, and
    re-wraps), `focusin` catches the rest, and height changes are ignored
    because those are ours and reacting to them is a loop. The border is
    added back on every measure, cached per element: the app is border-box,
    so `height = scrollHeight` lands 2px short and shows a permanent hairline
    scrollbar (same fix, same reason as `practice/notebook-editor.js`).
  - **The one case it cannot see:** assigning `note.value = …` from script
    fires no `input` and changes no width. A surface that prefills a note
    must call `DDAutoGrow.grow(note)` — which is what `openForm` does after
    it clears `#ir-form-note`.
  - **Width and type follow the explainer pages.** `.ir-head`, `#ir-chooser`
    and `#ir-questions` take the same 985px measure
    `#page-learn-about-app .container` sets; the question text is 18px/1.7
    and the note 17px/1.6 on a 68ch measure. `.ir-container` stays 1100px
    and the measure sits on those three children instead, because `#ir-graph`
    is `position: fixed` full-bleed and wants every pixel.
  - **The practice note's expand-on-focus is gone** (`min-height: 38px` →
    `160px` on `:focus`). An expand-on-focus box still stops at a height
    someone guessed, and a `min-height` that changes on focus fights a
    measured height — the box would shrink back the moment it blurred.

- **2026-08-24 — the top-right account menu, the timer in the topbar, and one
  Learner Home.** `index.html`, `app.js`, `account-menu.js` (new),
  `styles/account-menu.css` (new), `styles/layout.css`,
  `styles/practice/{notch-menu,diagnostic}.css`,
  `practice/{diagnostic-page,init,placement-results}.js`, `solo-route.js`,
  `watch.py`, `styles/watch.py`, `practice/watch*.py`.
  Seth, in his words: "put the cog icon in the top right rather than the
  middle", "put the timer in the top middle bar rather than as the notch",
  "instead of a vague cog ... making it clear that it is clickable", "it should
  say 'Account and Settings'", "'Learner home' should stand out significantly
  relative to everything else", and "the diagnostic and practice should be
  combined into one tab, with it being called 'Learner Home'".
  - 🔴 **The cog was not mis-styled, it was mis-placed by CSS Grid.** `.topbar`
    is a three-column grid and the middle child was the tab strip, which basic
    mode sets to `display: none`. A `display: none` grid item takes **no cell**,
    so auto-placement slid the right-hand side into column 2 and the cog landed
    at x≈700 on a 1440 viewport — dead centre. All three cells now carry an
    explicit `grid-column`, which is what actually fixes it; no amount of
    `justify-content` on the right cell would have. Measured after:
    `L [18,660] M [660,770] R [770,1412]` on a 1430px bar, mid centre 715 =
    page centre 715.
  - **The clock moved into that middle cell.** `#practice-notch` was
    `position: absolute` with a `translateX(-50%)` and a downward-hanging
    notch shape; it is an ordinary flex child of `.topbar-mid` now with a pill
    radius, so it sits beside the tab strip in advanced mode and alone in basic
    mode. Its menu is unchanged apart from re-anchoring to `calc(100% + 10px)`.
  - **The cog is a labelled control.** `#topbar-account` is an avatar glyph, the
    words **Account and Settings** and a caret; `#account-menu` is five rows,
    each with a 16px icon, with **Learner home** first, bold, tinted, and cut
    off from the rest by a separator. 🔴 **The label is deliberately not the
    email** — an address answers "who am I signed in as", asked once a session,
    and says nothing about what the button does. On a phone the words drop and
    the avatar + caret stay.
  - **Why a menu at all:** basic mode has no tab strip, so before this the cog
    was the ONLY control that left the practice screen and it went to exactly
    one place. "Why this app exists" was reachable from the welcome fork once,
    on a first visit, and never again — which is what Seth hit as "when you
    click the left option to read about Delta Drills, make it such that it
    actually works". The clicks always worked; the page was a dead end.
    `data-lab-open` on the two explainer rows opens the matching `<details>`
    after the switch, exclusively, on the next frame (before layout exists,
    `scrollIntoView` scrolls to the top of the document instead).
  - **One tab.** `#page-diagnostic` is **deleted**. `#diagnostic-overview` is a
    card inside `.session-setup`; `#diagnostic-workspace-host` is the last child
    of `#page-practice` and the workspace moves between it and
    `#practice-workspace-home` as a probe comes and goes. The whole Practice-tab
    LOCK went with it — `setPracticeLock`, `LOCK_WHY`, and app.js's matching
    redirect — because one tab cannot be locked against itself. 🔴 The merge
    created a specificity collision: `#page-practice.session-idle
    .practice-split { display: none }` outranks `.diagnostic-workspace-host
    .practice-split`, so a probe rendered as a blank idle screen; settled with
    two-ID selectors in `styles/practice/diagnostic.css`, not a JS class dance.
  - **The area bars are on the idle screen now.** Seth: "it should display the
    information about einops, numpy, and einsum to be learned". No hard-coded
    list was needed — `/api/practice/diagnostic/status` returns all three areas
    with `probes: 0` from the first call on a new account, so
    `PlacementResults.renderAreas` is public and called on **every** status
    read rather than only on a completed placement.
  - 🔴 **Two real bugs fell out of it, and neither threw anything.**
    (1) `diagnostic-page.js` refreshed at parse time, and it parses BEFORE
    `practice/init.js` runs `detectPracticeMode()` —
    `PracticeAPI.diagnosticStatus()` returns **null**, not a failure, while the
    mode is still its `"local"` default, and `render(null)` paints "Sign in to
    take the placement test." with every button and both area lists hidden. It
    did not bite while the placement had its own page, because that page is
    hidden at load and the boot guard never fired; pointing the guard at
    `#page-practice` made the broken call the first thing every visitor saw, on
    a backend that was answering 200 the whole time. init.js announces
    `delta:practice-mode-ready` and the boot block waits for it.
    (2) `apiFetch` is a top-level `const`, so it was never a window property,
    and `concept-graph/{kc_lattice_read,lesson-graph}.js` both guard
    `window.apiFetch || fetch` — falling back to a **relative** URL that never
    reaches the backend. Locally a boot 404; on Vercel the SPA rewrite answers
    `/api/practice/kc-lattice` with 200 text/html, `res.ok` is true,
    `res.json()` throws, the catch writes null, and the knowledge graph and the
    "Why this app exists" map render the guest/offline reading for a signed-in
    learner. `window.apiFetch = apiFetch` publishes it; the two boot 404s are
    gone and the dial went from "0 of 63 concepts" to "23 of 63 concepts
    measured directly" on the same account. Same shape as the `.vercelignore`
    runtime-fetch trap: check the content type, not the code.
  - **Criticked** (codex, `gpt-5.6-sol`): five findings, four real and fixed.
    (1) The boot fallback latched — a `DDGuest.ensure()` slower than 8s would
    have let the timer paint the signed-out copy and then DROPPED the real
    mode-ready event; only the event closes the door now. (2) A
    `#page-practice #diagnostic-workspace-host .stage-ladder { display: block }`
    line I had added on spec outranked BOTH `.stage-ladder.hidden` and
    `body.dd-basic-mode .stage-ladder` (each `(0,2,0)`) and forced an empty
    advanced-only widget onto basic-mode probes — deleted, with a tombstone;
    measured `display: none` on a live probe after. (3) `role="menu"` promised
    arrow keys the menu did not have: focus now lands on the first row, Up/Down
    wrap, Home/End jump, Escape returns focus to the trigger — all six checked
    with real key events. (4) `watch.py` computed the `#account-menu` slice and
    then matched labels against the WHOLE document; scoped to the slice, whose
    terminator turned out never to have matched — mutation-tested by renaming a
    row, which now fails the suite. **Disputed:** codex read the menu as staying
    open across tab switches; it missed the document-level click-away listener.
    Checked on a real 1440px tab strip — clicking Notebooks switched the page
    AND closed the menu with `aria-expanded="false"`.
  - Verified in a real browser against the live Fly backend: a full six-probe
    placement driven end to end on the merged tab (workspace hosted and
    released, results unhidden in place with no tab switch, bars moving from
    28%/not-probed to 2%/2-probed, start button relabelled "Retake the
    placement test"), plus 390px (no horizontal overflow, menu inside the
    viewport), light and blue themes, and a clean console.

- **2026-08-23 — one front door: a two-arrow fork, one "Learn about the App"
  tab, and no tab strip in basic mode.** `index.html`, `app.js`,
  `solo-route.js`, `concept-graph/why-graph.js`, `practice/diagnostic-page.js`,
  `styles/learn-about.css` (new), `styles/{how-it-works,why-map}.css`,
  `watch.py`, `styles/watch.py`.
  Seth's ask, in his words: two arrows and "literally nothing else for the user
  experience"; the left one says it is optional, the right one goes straight to
  the placement test; "after taking the placement diagnostic, it needs to take
  the learner to the practice"; and "when you enable the advanced mode it should
  show you all the other tabs that it currently has, so you actually get the
  tabs back".
  - **The two explainer tabs became one.** `Why This App Exists` +
    `How to use it` → `#page-learn-about-app`, tab `learn-about-app`, label
    **Learn about the App**. What is unconditionally on screen is one heading
    and one paragraph — what Delta Drills is. Everything else is a native
    `<details>`: *The three markers, in full* and *How the app works* (the old
    "How to use it" body, renamed on Seth's instruction), both closed on
    arrival. **The concept map moved into *How the app works*** — it is the
    mechanism, not the argument. 🔴 That put it inside a CLOSED disclosure, so
    `why-graph.js::whenVisible` now listens for the `<details>` `toggle` event
    as well as the page's class list; the page can be perfectly visible with the
    map still unrendered, and the page-class observer alone leaves it on
    "Loading the map…" forever. `/why-this-app` and `/how-to-use` still resolve
    — both are legacy slugs pointing at the merged page in `solo-route.js`.
  - **`#page-welcome` is a page with no tab.** Two `[data-goto-tab]` arms and
    nothing else: no pitch, no CTA, and no guest banner (that lives outside
    every `.page`, so `app.js` stamps `body.dd-welcome` and the rule is in
    `learn-about.css`). `app.js` lands a first-time visitor here instead of on
    the old pitch page. There is deliberately **no route back to it** — a fork
    is a thing you pass through once.
  - 🔴 **Basic mode has no tab strip at all, and basic mode is the default.**
    `body.dd-basic-mode` hides `.tabs`, the copy `nav-drawer.js` parks in the
    drawer, and `.nav-toggle`. Navigation is the fork, the topbar cog
    (→ Account), and the app moving the learner on by itself. **Advanced mode
    is the whole way back** — every tab returns unchanged, because this is a
    display rule and nothing is unmounted. Because the cog is then the only
    control that leaves Practice, the Account page carries `.dd-basic-nav`
    ("← Back to practice" / "Learn about the App"), hidden in advanced mode.
  - **A finished placement hands the learner to Practice** (`diagnostic-page.js`
    `render()`), and stops there — "after it takes them to the practice they
    just continue studying that". 🔴 It keys on the **transition**, not the
    state: `render()` runs on every `delta:practice-state-changed` and on every
    tab entry, so keying on "is complete" alone would drag a learner off the
    Placement tab every time they opened it months later. **Basic mode only** —
    advanced mode still has the strip, so `#diagnostic-results` stays readable
    there for as long as the learner wants it. Nothing is lost in basic mode
    either: `practice/readiness.js` is the single writer of that figure and the
    Practice idle dial prints the same reading, caption and detail line.
  - `watch.py::check_front_door` asserts the lot — one tab and no trace of the
    two it replaced, two disclosures with the lead paragraph OUTSIDE them, the
    map inside *How the app works*, the toggle listener in `why-graph.js`, the
    fork's two arms and the word "optional", `#page-welcome` never becoming a
    tab, all three basic-mode strip rules, the **link order** of
    `learn-about.css` after `nav-drawer.css`, the Account escape row, and the
    hand-off to Practice. Every one of those fails silently rather than
    raising.
  - **Criticked, and it found a real one.** Codex is answering again — the
    08-27 usage wall other sessions hit today has lifted. 🔴 **A tab name that
    matches no page blanks the app**: `switchTab` hides every `.page` whose id
    is not `page-<name>`, so a stale `dd_recovered_tab` — the sessionStorage
    key `guest-session.js` writes before a recovery reload — carrying
    `why-this-app` across a deploy would have left a topbar over nothing, with
    no error anywhere. The URL aliases in `solo-route.js` cannot help; that
    name never goes near a pathname. Fixed BOTH ways in `app.js`: `renamedTabs`
    maps the two retired names onto the merged page, and a general existence
    check falls back to the same pair the boot call chooses between, so the
    next rename and every `[data-goto-tab]` typo land softly too. Both are
    asserted in `check_front_door`, and both were reproduced in the browser
    before and after. Codex's second finding is real but is **not this diff**:
    `renderLength`'s fallback chip says "N questions" when `min_probes` is
    missing, and `budget` is a ceiling — it should read "up to N". That line
    belongs to the placement-readiness session's uncommitted work; flagged in
    `collab`, not edited here.
  - Verified in a real browser: the fork at 1854px and 390px with no scrollbar
    either way,
    the disclosures and the map drawing on open, the right arm landing on the
    Placement tab, the cog landing on Account and the escape row landing back on
    Practice, advanced mode restoring the strip, and a stubbed placement going
    active → complete switching to Practice once and NOT bouncing a re-visit.

- **2026-08-23 — the topbar is a three-column grid, and 28 ⓘ dots are gone.**
  `index.html`, `app.js`, `nav-drawer.js`, `infotips-registry.js`,
  `styles/{layout,nav-drawer,infotips,xp}.css`, `watch.py`.
  🔴 `.topbar` is `grid-template-columns: 1fr auto 1fr`: two EQUAL side columns
  with the tab strip in the `auto` column between them, which is the only way
  it is actually centred on the viewport — `justify-content: space-between`
  centres the middle child only when the outer two happen to match, and they
  never do. Measured at 0.00px error signed out and with all ten tabs showing.
  Source order IS the layout (level pill → strip → cog), and `.logo`, the empty
  spacer that outlived the brand line, is deleted: an extra child in a
  three-column grid displaces the centred strip into a side column.
  The right edge is one cog (`#topbar-cog`), wired through app.js's existing
  `[data-goto-tab]` hook. Drawer mode (<900px) puts the row back to flex,
  because `nav-drawer.js` MOVES the strip out and a grid with one child gone
  drops the right-hand side into the middle.
  **The ⓘ cull**: 51 dots → 23. All ten tab dots plus 18 controls whose own
  label already said what they did (Show hint, Submit, Code Editor, Run,
  Solution, Tutor, the guest banner, …). What stayed is the jargon — mastery
  tiers, the ladder rungs, the ARENA unlock, the graph legend, the placement
  clock, the developer fields. 🔴 Deleting the tab dots is also what
  straightened the active underline: `.tab`'s `border-bottom` spans the label,
  and the eye had been reading label+dot as the tab.
  `watch.py::check_infotips` was inverted to assert the dots stay gone.

- **2026-08-23 — the landing story is now two tabs.** `index.html`,
  `infotips-registry.js`, `styles/how-it-works.css`. "Why This App Exists"
  carries Seth's value-add essay (personalized/AI-paced, no decision fatigue,
  expertise reversal effect) in his own words; a new guest-visible
  **How to use it** tab to its right holds the practical path (placement test
  → live on Practice → handoff to ARENA) plus the six-step loop grid, which
  moved there from the why page. The 3x2 `.steps-grid` override in
  `how-it-works.css` is scoped to `#page-how-to-use` now. Tab id
  `how-to-use`, infotip key `tab.how-to-use`.

- **2026-08-23 — a 401 is three different things, and only one of them means
  "sign in".** `app.js`, `guest-session.js`, `practice/diagnostic-page.js`.
  `apiFetch` recovers an expired GUEST token in place — it holds the password —
  and retries the one request that failed, so a token dying under a graded
  submit is a hiccup instead of "Submit did nothing". Three rules keep that
  from becoming its own bug, all three found by codex on the first cut:
  - 🔴 **A replay is only safe while the identity has not moved.** The retry
    fires when `authToken` changed mid-flight, and a second sign-in changes
    `authToken` too. `setAuthState` does NOT reload on guest → person (the
    guest already held a token, so `wasAuthed` is true), so the request's POST
    body — a graded answer — could be replayed against a different account.
    `apiFetch` now captures `authEmail` beside the token and replays only while
    they still match.
  - 🔴 **The same race runs backwards inside `refreshSilently`**, which checked
    `isGuestSession()` before awaiting `/auth/login` and adopted the result
    after. Signing in during that await put the guest token back on top and set
    `GUEST_ACTIVE_KEY` again — sign in, be a guest. A login whose identity moved
    underneath it is now dropped.
  - 🔴 **A failed silent re-login is not proof of being signed out.** It also
    declines to run while the backend is unreachable and during its 30s
    cooldown, and the caller sees the original 401 either way. `DDGuest.
    canRecover()` (guest session + credentials still in this browser) separates
    them, so a learner mid-placement gets outage copy and a retry instead of a
    stuck "Sign in to take the placement test". **403 is untouched** — a fresh
    token for the same account cannot fix an authorization refusal.

- **2026-08-22 — the app has levels, and the `Level N` pill IS the progress
  bar.** `xp.js` + `styles/xp.css` are new. The topbar carries a `Level N` chip
  between the logo and the tab strip, and that chip colours in from the left as
  the learner works — one object for both facts, which level and how far into
  it. 🔴 **Nothing renders a numeral for the progress** — no count, no percent,
  no `340 / 500`. The only number on screen is the level; the hover `title` on
  the chip is the one place the raw XP is available. That is the whole design
  brief, so a future "just show the XP on the bar" is a regression, not a
  feature.
  - **It was the topbar seam first, for about four hours.** The bar started as a
    2px line laid over `.topbar`'s bottom border (`.dd-xp-seam`), which is a
    genuinely elegant place to put it and almost invisible in practice — a
    hairline only reads as progress if you already know to look for it. The pill
    replaced it the same day; `.dd-xp-seam*` is gone from every file, and the
    `bottom: -2px` measurement that made it clear `.tab.active`'s underline went
    with it.
  - **One custom property drives the whole pill.** `xp.js` writes
    `--dd-xp-pct` on `.dd-level` and nothing else; the fill's width and the clip
    on the on-accent text layer both read it, so they cannot drift apart the way
    two separate JS writes eventually would. 🔴 The label is drawn **twice**,
    stacked in one grid cell — a normal copy and an `--on-accent` copy clipped
    to the filled width — because no single text colour is legible on both a
    near-transparent chip and a saturated indigo→cyan fill in all three themes.
    The padding lives on the text layers, not on `.dd-level`: put it on the pill
    and the clip edge and the fill edge separate by exactly that padding.
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
    first award's remainder — which was 0%, so the pill emptied. It now re-reads
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
