# extension

## Purpose
The Chrome MV3 **side panel** that puts Delta Drills beside the Colab notebook
the student is working in. Toolbar click opens the panel; the notebook fills the
rest of the window beside it.

**Which side the panel is on, and how wide it is, are CHROME settings — not this
extension's.** `chrome.sidePanel` has `setOptions` and `setPanelBehavior` and
nothing about position or width; there is no API to ask for. The student sets it
once:

- **Left instead of right:** Settings → Appearance → *Side panel position* →
  "Show on left" (or right-click the panel and pick it there).
- **Wider:** drag the panel's inner edge. Chrome caps how far it goes, so a true
  50/50 split may not be reachable.

⚠️ **Do not "solve" either of those in code.** A pass on 2026-07-31 made the
toolbar button tile two `chrome.windows` — app left, notebook right — to get the
app onto the left. It worked and it was wrong: separate windows are not a side
pane, they pop out of the browser instead of splitting it, and it left the
student managing two windows for one task. Reverted; `watch.py` fails on
`chrome.windows.create` in the worker so it cannot creep back.

**The panel shows the live website.** Not a copy of it, not a client of its API —
`https://delta-drills-colab.vercel.app/` itself: its tab bar, its Sign in with
Google, its practice loop, its knowledge graph. There is no panel front end, and
there is not supposed to be one.

That address is the **Colab edition** — a second Vercel project serving the same
code as `delta-drills.vercel.app`, where a drill routes to its published lesson
notebook instead of the in-page editor
(`Local_Deployed_Shared/practice/colab_mode.js`). Same backend, same Supabase,
same bank: practice here moves the same mastery record. Framing the normal deploy
would put an in-page editor next to a notebook — two places to solve one drill —
so `watch.py` fails if the address drifts back.

`panel/app.html` is the entire thing: a `<style>` block and one `<iframe>`. No
script.

A separate hand-built tutor UI also lives here (`panel/panel.html`) — a narrow
one-problem-at-a-time surface that drives Colab notebooks directly. It is not
wired to anything. Open it deliberately at
`chrome-extension://<id>/panel/panel.html`, or point `side_panel.default_path`
at it.

## Install (unpacked)
No build step, no npm, no store listing.

1. `chrome://extensions` → turn on **Developer mode** (top right).
2. **Load unpacked** → select this `extension/` folder.
3. Pin "Delta Drills — ARENA tutor" and click it. The site loads in the panel.
4. Sign in with Google, exactly as on the site.

**Signing in here is separate from signing in on the website.** The site runs as
a cross-site frame under a `chrome-extension://` top level, so Chrome gives it
its own partitioned storage. It persists; it just does not carry over from a
normal tab, and vice versa — the panel starts in GUEST. If the panel stays empty
or sign-in never sticks, the usual cause is Chrome's *Block third-party cookies*
setting denying the frame its storage — allow site data for
`delta-drills.vercel.app`.

After editing any file, hit the reload arrow on the extension card. **If a Colab
tab was already open, reload that tab too** — content scripts inject at
navigation, so an existing tab has no listener for the new instance.

Local backend: open Settings (⚙) and set the base URL to
`http://127.0.0.1:8000`. It is already in `host_permissions`.

**Notebooks.** Settings also carries a *Notebook repo*, which is how the panel
computes a Colab URL for a notebook the student has not opened yet. It reads
`owner/repo[@branch]/[path]` and ships pointing at the repo that
`scripts/publish_colab_notebooks.sh` publishes, so a fresh install can already
open all nine. Publish your own with:

```bash
scripts/publish_colab_notebooks.sh            # creates <you>/arena-book-colab
```

then paste the `Panel ⚙:` line it prints. The Settings list shows every lesson
and how the panel would reach it — `remembered` (a URL proven to work),
`repo` (computed, untried) or `no link`.

Preflight from the shell (catches the errors that otherwise appear only after
clicking Load unpacked):

```bash
python3 extension/watch.py && python3 extension/panel/watch.py && python3 extension/content/watch.py
```

## Owns
- The student's moment-to-moment workflow: which problem is on screen, the
  countdown, the two self-grade buttons, the undo window before auto-advance.
- Navigation into a Colab notebook — resolving **which** notebook holds the
  problem, switching the tab to it when it is not the open one, finding the
  cell, expanding the collapsed sections above it, confirming it landed.
- Storage of the backend base URL, the session JWT, the notebook repo, and the
  learned `lesson → URL` map (`chrome.storage.local`).

## Does NOT own
- **Any mastery decision.** Prerequisite gating, BKT posteriors, FIRe credit
  propagation, forgetting-curve decay and the expertise-reversal ladder all run
  in `This-Directory-Only/backend/app/`. This folder moves JSON; it never scores,
  gates, or schedules. If a decision looks like it belongs here, it belongs there.
- Notebook content, or the `question → notebook` map. Both are generated by
  `scripts/generate_colab_notebooks.py` from `Local_Deployed_Shared/lessons/`;
  `panel/notebook-index.js` and its two sibling parts (`-questions`,
  `-concepts`) are its output and must never be hand-edited.
- Code execution and grading — the student's own Colab kernel runs the cell.

## Key Files
- `manifest.json`: MV3 declaration. `sidePanel` + `tabs` + `storage`, host
  permissions for Colab and the backend (plus localhost for dev).
  `side_panel.default_path` is `panel/app.html` — the framed site, not the
  tutor UI.
- `background.js`: service worker. Its only real job is
  `sidePanel.setPanelBehavior({openPanelOnActionClick: true})`, which cannot be
  declared in the manifest. It must not grow window management — see Purpose.
- `panel/`: the UI, the backend client, notebook routing and the generated
  notebook index. See `panel/README.md`.
- `content/`: the Colab DOM adapter. See `content/README.md`.

## Data & External Dependencies
- Delta Drills backend (`https://delta-drills-backend.fly.dev`) —
  `/auth/login`, `/api/practice/next-question`, `/submit-local-eval`,
  `/override`, `/exposure`, `/worked-seen`, `/kc-lattice`. The backend sets
  `allow_origins=["*"]`, so no CORS change is needed for the extension origin.
- `NextQuestionResponse` is the contract that matters
  (`backend/app/practice_schemas.py`) — notably `ladder_stage`, `ladder_kc`,
  `ladder_estimate`, `starter_code`, `hint` and `lesson_gate`.
- No third-party libraries, no build step. Everything ships as written.

## How It Works (Flow)
1. Toolbar click opens the panel. It reads the stored token; with none, it shows
   the connect view and posts to `/auth/login`.
2. `GET /api/practice/next-question` returns the whole scheduling decision in one
   call — subtopic, KC, difficulty target, ladder rung, rung-appropriate starter.
3. A non-empty `lesson_gate` means the learner has never been taught this KC, so
   the panel routes to the lesson first and posts `/exposure` + `/worked-seen`.
4. Otherwise the problem renders with a countdown, alongside the notebook it
   lives in and whether that notebook is `open` or needs a `switch`.
   **Go to the cell** asks the tab which notebook it is (`dd:identify`),
   navigates it to the right one if they differ, waits for Colab to finish
   mounting, then scrolls to `dd-q<question_id>`.
5. **Mark correct** / **Mark wrong** — or timer expiry, which auto-marks wrong —
   posts `/submit-local-eval`. That endpoint runs the identical downstream chain
   as server-side grading.
6. A settle window offers `/override` as Undo, then the loop returns to step 2.

## Invariants & Constraints
- **No mastery math in JavaScript, ever.** Duplicating a threshold here means two
  sources of truth for what a learner knows.
- MV3 CSP forbids remote script. Everything must ship in the folder — no CDN.
- The panel and the content script cannot call each other directly; they talk via
  `chrome.tabs.sendMessage`.
- A missing content script is the *normal* case (no Colab tab, or a tab that
  loaded before the extension). It must degrade to a message, never an exception.
- Grading is guarded by `state.graded` — a click and a timer expiry can race, and
  double-submitting would log two attempts against the ladder for one problem.
- **Switching notebooks is the normal case, not an error case.** The tutor picks
  weakest-first across every subtopic and one lesson is one subtopic, so
  consecutive problems routinely live in different notebooks. Any code path that
  assumes "the open notebook is the right notebook" is wrong.
- **A URL that worked outranks a URL we computed.** `notebooks.urls` is written
  every time a tab successfully identifies itself, and wins over the repo
  setting — that is what lets a student who uploaded to Drive, or who forked to
  a differently-named repo, keep working without configuring anything.

## Extension Points
- New backend call → add a method to `panel/api.js`, nothing else.
- New Colab DOM capability → add a `dd:` message case in `content/colab.js`.
- New panel screen → add a `<section id="view-…">` and its name to `VIEWS`.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Stale content script after a reload** — `ACTIVE`
  - When it happens: the extension is reloaded while a Colab tab is already open.
  - Symptom: **Go to the cell** does nothing; the panel reports "Reload the Colab tab".
  - Root cause: content scripts are injected at navigation, so an already-open tab
    has no listener for the new extension instance.
  - Prevention/fix: reload the Colab tab after reloading the extension. The panel
    detects this specific case (`no-receiver`) and says so rather than failing silently.

- **Anchors do not exist on ARENA's own notebooks** — `ACTIVE`
  - When it happens: navigating inside any of the 458 upstream ARENA_5.0 notebooks.
  - Symptom: `dd-q…` lookups miss.
  - Root cause: those files are nbformat 4.2 with no `metadata.id`, so Colab mints
    fresh DOM ids on every load.
  - Prevention/fix: `findCell` falls back to matching rendered text. Keep that
    fallback working — generated notebooks are the only ones with stable ids.

- **Silent scroll into a collapsed section** — `RESOLVED`
  - When it happens: the target cell sits under a collapsed section header.
  - Symptom: the scroll appears to succeed and nothing moves.
  - Root cause: the cell has zero height, so `scrollIntoView` is a no-op.
  - Prevention/fix: `expandAbove` clicks every preceding
    `md-icon-button.header-section-toggle` labelled "Expand" before scrolling, and
    `goto` re-resolves the cell afterwards because expanding re-renders it.

- **A stale `notebook-index*.js` sends the panel to the wrong notebook** — `ACTIVE`
  - When it happens: lessons change and only the notebooks get regenerated.
  - Symptom: a jump switches notebooks and then reports the cell is not there.
  - Root cause: the index is a compile-time snapshot of `question → lesson`.
  - Prevention/fix: `generate_colab_notebooks.py` always writes both, and
    `publish_colab_notebooks.sh` regenerates before pushing. Never edit the
    index by hand. `panel/watch.py` fails if the file is missing entirely, but
    it cannot tell a stale index from a current one.

- **Panel renders blank with a dead ⚙** — `RESOLVED`
  - When it happens: any edit to `panel/panel.js` that pulls a name out of
    `window.DD` or `window.DDNav` under that name.
  - Symptom: an empty panel. No view, no handlers, and the cog does nothing —
    which reads as a CSS or manifest fault and is neither.
  - Root cause: `panel.html` loads four **classic** scripts, so they share one
    global lexical scope. `const {api} = window.DD` redeclares the `const api`
    that `api.js` already made, and the whole file dies at parse time.
    `node --check` per file cannot see it.
  - Prevention/fix: alias every import (`api: ddApi`, `slugKc: navSlugKc`).
    `watch.py`'s `check_no_shadowed_globals` fails on a bare one; to check by
    hand, `cat` the four files together and `node --check /dev/stdin`.
    `app.html` is immune — it loads exactly one script.

- **Sign in with Google fails inside the panel** — `RESOLVED`
  - When it happens: the frame's `allow` attribute loses
    `identity-credentials-get`.
  - Symptom: the Google button renders normally and then the flow dies.
  - Root cause: GSI negotiates over FedCM, and FedCM in a cross-origin frame is
    denied unless the embedder delegates it through Permissions Policy.
  - Prevention/fix: keep `allow="identity-credentials-get; …"` on the iframe.
    `watch.py` fails without it.

## Recent Changes
- 2026-07-31 (later): **Reverted the tiled-windows experiment.** The button opens
  the side panel again and `background.js` is back to one call; `system.display`
  dropped. Panel side and width are Chrome settings and always were — putting the
  panel on the left is Settings → Appearance → *Side panel position*. `watch.py`'s
  `check_study_layout` became `check_button_opens_the_panel`, which now fails on
  any `chrome.windows.*` / `chrome.tabs.move` call in the worker.
- 2026-07-31: **The panel frames the Colab edition.**
  `delta-drills-colab.vercel.app` — a second Vercel project serving the same code
  with notebook routing on, so a drill opens in its lesson notebook instead of the
  in-page editor. `watch.py` fails if the address drifts back to the normal app.
- 2026-07-31: **The panel is the website.** `panel/app.html` is one `<iframe>`
  over `https://delta-drills.vercel.app/`, with no script and no panel chrome —
  an earlier pass wrapped it in a toolbar and a settings sheet, which was a
  second front end by another name and is gone. Verified live at 400px: the site
  renders edge to edge with its own tab bar, Pyodide loads, the practice engine
  loads, 499 questions load, and the `accounts.google.com/gsi/button` frame
  mounts inside it. The deploy sends neither `X-Frame-Options` nor
  `frame-ancestors`, so framing needs nothing from the server.
- 2026-07-31: Fixed the blank-panel SyntaxError in `panel/panel.js` (see above).
- 2026-07-31: Cross-notebook transitions. `dd:identify` reports which notebook a
  tab holds; the panel resolves the target from a generated index, switches the
  tab and waits for Colab to mount before jumping. Notebook routing split into
  `panel/navigate.js`. Added `scripts/publish_colab_notebooks.sh`.
- 2026-07-31: Initial build — MV3 panel, Colab content script, backend client.
  Pass 1 of `docs/spec-colab-its-surface.md`.
