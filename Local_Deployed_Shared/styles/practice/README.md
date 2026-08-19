# styles/practice

## Purpose
- Stylesheets for the Practice tab (the adaptive-queue question page), split out of the former monolithic `practice.css` (972 LOC) into per-concern files.

## Owns
- All CSS for the practice page shell, session setup/status, question display, feedback/rating UI, code editor/output, and the tab's smaller extras.

## Does NOT own
- Progress-bar styles (`../../practice/bars.css`), ARENA unlock interstitial styles (`../../practice/arena-unlock.css`, `arena-unlock-timer.css`), Targeted Practice styles (`../../targeted-practice/targeted-practice.css`).
- Global utilities — `.hidden`, buttons, variables live in `../components.css`, `../base.css`, `../variables.css`.

## Key Files
- `layout.css`: page container, left/right split panels, responsive stacking under 900px.
- `timer.css`: rigid-session UI — the pre-session setup panel (question count, answer/review time), the in-session status row (progress, phase, strict countdown), and the `session-idle` page-state rules.
- `question.css`: question number/text, imported-helpers pills + detail, target-image visual, meta chips, cold-start badge, prose/code-block split.
- `feedback.css`: submit/skip/don't-know row, result badge, felt-difficulty rating buttons, problem quality flags, missed-fact row, failed-tests block.
- `editor.css`: code editor, Run button, output area + output visual, solution section, AI explanation.
- `misc.css`: hint/answer aids, practice-mode intro, self-report row, placement entry button, torch Colab notice, mode-demotion notice, topbar auth indicator.
- `difficulty-bar.css`: **the one difficulty readout on the practice page** (`#target-difficulty`, mounted under the concept strip; behaviour in `../../practice/difficulty-bar.js` + `bars.js`). Full bleed, 32px track, `Old → New` at 26/38px, and two named regions drawn ON the track: a hatched red zone below the support floor and a green one from the next rung's threshold up. Stacking matters and is set here — zones sit above the fill and the green/red band (translucent, so their labels survive a bar running through them) and below the white markers and the accent tick, which are the actual readings. The track internals still come from `../../practice/bars.css`; this file only re-sizes the track and adds what is drawn on top of it.
- `notebook-view.css`: **the Notebooks tab**, which is not the practice page at all — it lives here because it is a practice-surface stylesheet and shares the tokens. Styles the notebook list, the sticky toolbar (Back / title / jump-to / Restart session), the banner the view uses to say the session restarted or that the learner is signed out, and the cell: a `[n]` execution counter, an editable source block, its output, and the run/failed/stale borders. Solutions and hints are `<details>` — closed is the default and the styling must not make an open one look like an ordinary cell. Scoped under `.nbv-*`; nothing here is shared with `editor.css`, because a notebook cell and the practice editor look alike and are not the same thing.
- `colab-edition.css`: the Colab edition's tutor rail, and its review step (a verdict opens the solution; everything else in the feedback panel is hidden there). Everything is scoped under `html.dd-colab-edition` (set by `practice/colab_mode.js` on the `delta-drills-colab` deploy) and is inert on the normal app. Strips the prompt, the editor and the worked example — they are in the notebook beside the panel — and re-stacks the concept strip for panel widths. `html.dd-no-notebook` (set by `ui.js`) turns all of it back off for the ~75 questions with no published notebook, which would otherwise get an empty rail.

## Data & External Dependencies
- CSS custom properties from `../variables.css` (`--border`, `--surface`, `--muted`, `--accent`, …).
- Markup lives in `../../index.html` (`#page-practice`); class toggling driven by `../../practice/*.js`.

## How It Works (Flow)
1. `index.html` links all six files (after `../components.css`, so same-specificity practice rules win).
2. `#page-practice.session-idle` (set in markup, toggled by `practice/timer.js`) shows the setup panel and hides `.practice-split` **and `.concept-topbar`**; starting a session flips it.

## Invariants & Constraints
- Load order: these files must come AFTER `styles/components.css` — several rules rely on overriding it by order.
- Any rule that sets `display` on an element JS toggles with `.hidden` must re-assert `.selector.hidden { display: none; }` when its selector outweighs the global `.hidden` class (see `#practice-submit-area.hidden` in `feedback.css`).
- Keep selectors class-based; ID selectors beat `.hidden` and have caused real hide-failures.

## Extension Points
- New practice-tab UI: add to the file matching its concern (or a new file + `<link>` in `index.html` with a `?v=` param). Bump the file's `?v=` on every edit — stale-cache trap.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **Page-state rules that name only `.practice-split`** — `RESOLVED`
  - When it happens: a new element is added to `.practice-container` as a SIBLING of `.practice-split` rather than a child of it.
  - Symptom: the setup screen shows a strip of leftover question state (concept name, rung, difficulty) for a question that is not on screen.
  - Root cause: `#page-practice.session-idle .practice-split { display: none }` hides the split's subtree only. `.concept-topbar` was deliberately lifted out of the split so it could span both panels, which also lifted it out of that rule.
  - Prevention/fix: any new direct child of `.practice-container` that belongs to the question view needs its own `#page-practice.session-idle` rule. `watch.py` asserts the topbar's.
  - Status: RESOLVED (2026-07-30).

- **ID selector defeats `.hidden`** — `RESOLVED`
  - When it happens: a rule like `#practice-submit-area { display: flex }` styles an element JS hides via `classList.add("hidden")`.
  - Symptom: element never disappears.
  - Root cause: ID specificity (1,0,0) beats the global `.hidden` class (0,1,0).
  - Prevention/fix: re-assert `#the-id.hidden { display: none; }` next to the rule, or use a class selector.
  - Status: RESOLVED (2026-07-12) for `#practice-submit-area`; the rule stands for new code.

## Recent Changes
- 2026-08-19: Added `notebook-view.css` for the new Notebooks tab (see Key Files). Linked from `index.html` alongside the other practice stylesheets; `practice/watch_notebook.py` asserts both the link and the file, since a missing stylesheet here fails as an unstyled 656-cell wall of text rather than as an error.
- 2026-08-18: **One difficulty bar, and it is `difficulty-bar.css`.** The page had two — `concept-topbar.css`'s `.concept-topbar-diff-bar` (96px, in the strip) and `bars.css`'s `.target-difficulty` card down in `.question-meta-row` — showing one quantity in two visual languages. The diff and estimate blocks are deleted from `concept-topbar.css` (the strip is now the concept name plus the rung dots), and `colab-edition.css` loses the overrides that sized them plus the `.question-meta-row` rule that placed the card; all it keeps for the rail is the bar's padding. The 2026-08-04 and 2026-08-03 entries below describe those deleted rules and are kept as history, not as current behaviour.
- 2026-08-07: **`misc.css` — `.topbar-auth[hidden] { display: none }`.** `app.js#updateAuthIndicators` hides the signed-in indicator with the `hidden` *attribute*, but `.topbar-auth { display: flex }` is a class rule and beats the UA's `[hidden] { display: none }` — so the green "signed in" dot had been showing to signed-out guests, which is the exact confusion the indicator was added to remove. Link tag `?v=4`. Found when the new ⓘ next to it turned up in a guest session; the same fix is why `styles/layout.css` had to pin `.tabs` with `margin-right: auto` (with the indicator genuinely gone, `space-between` slid the whole tab strip to the right edge).
- 2026-08-04: **The felt-difficulty buttons stack on the Colab edition, and `#feedback-help` is no longer hidden there.** Three buttons on one row is a 1600px page's layout; in a 290px side panel each gets ~85px and "Way too easy" wraps to three lines inside a box sized for one. `colab-edition.css` sets `.feedback-buttons { flex-direction: column }` with `flex: 0 0 auto` on the buttons — verified legible with no truncation at 290/340/410px. `#feedback-help` comes back (smaller type) because the rating now moves where the next problem is pitched and that consequence is written down nowhere else; three buttons with no stated effect read as a survey. `.next-problem-btn` shares the container but never shares a line — `showNextProblemButton` hides the three.
- 2026-08-04: **The scaffold rail staggers its labels, and the difficulty bar shows the move the last answer made.** `colab-edition.css` takes `.stage-dot-label` out of flow (absolutely positioned, `nth-child(even)` below the dot and the rest above) so four labels can overlap in width without colliding and the dots sit closer together; first/last anchor to their edge instead of centring, so nothing runs off the panel. Verified at 290/340/410/520px. `concept-topbar.css` gains `.concept-topbar-diff-delta` (green `is-gain` / red `is-loss`, animating width) and `.concept-topbar-diff-from`, a hairline at the value the answer started from — the fill stops at the LOWER of the two values so the delta span owns the contested stretch, which is what lets one rule serve both directions: on a gain it grows outwards, on a loss it collapses back. `colab-edition.css` widens that track to 10px and turns the fill blue, so the strip's two bars stop reading as the same quantity.
- 2026-08-03: **`.question-meta-row` is back on the Colab edition, and the 2026-07-31 entry below is now wrong about it.** It was hidden there because a verdict recorded an attempt and learned nothing from it — there was no old → new to draw, so the bar sat on whatever the previous question had left on it. `recordLocalEval` now returns the step it caused (offline: sampled either side of the engine call; backend: from `/submit-local-eval`, which had never been finalizing its attempts at all), so the rail draws difficulty climbing inside the current stage — green when an answer earned it, red when one cost it. `#ewma-accuracy` stays hidden on purpose, and that is not the same call: the concept strip three rows up already carries a correct/total and an interval for the concept, and a second percentage computed a different way reads as the panel disagreeing with itself in a 400px column. `concept-topbar.css` also stopped dropping every scaffold-rung label below 700px — every side panel is below 700px, so on that deploy the ladder had always rendered as four unlabelled circles; the narrow breakpoint now keeps the active rung's label and `colab-edition.css` restores all four.
- 2026-07-31: The Colab edition's review step. `colab-edition.css?v=3` stopped hiding `.solution-section` — it lives inside `#practice-feedback-area`, which stays hidden until a verdict, so answering now opens the reference solution under the buttons that recorded it. The rest of that panel is hidden there instead (`#feedback-help`, `#ewma-accuracy`, `.question-meta-row`, `#ai-explanation-section`, `#override-row`): all of it describes a grade the tutor did not compute on this deploy, stacked above the answer the learner just asked for.
  Also fixed three more live instances of the invariant below, found by auditing every `.hidden` element's computed display on a running page: `.override-row` and `.missed-fact-row` (`feedback.css?v=2`) and `.output-visual-canvas` (`editor.css?v=2`) all outranked the global `.hidden` by load order. These were **not** Colab-only — on the normal app every correct answer showed "Is this a mistake? I got it right", and "I missed one specific thing" showed after correct answers too. `question.css` had carried the matching `.question-visual-canvas.hidden` pair since it was written, which is why the same bug in `editor.css` looked deliberate.
- 2026-07-31: Added `colab-edition.css` (see Key Files). Also fixed a live instance of the invariant above: `layout.css`'s `.practice-right` outweighed `.hidden` by load order, so the panel ignored `classList.add("hidden")` and the editor stayed on screen through torch routing — `.practice-left.hidden, .practice-right.hidden` now re-assert it. Never visible on the normal deploy because no bank question carries a notebook path; found the moment the Colab edition started routing.
- 2026-07-30: `timer.css` hides `.concept-topbar` in the `session-idle` state. The topbar is a sibling of `.practice-split`, so the existing idle rule never reached it and the setup screen displayed the paused session's concept strip. `watch.py` now asserts the new rule.
- 2026-07-12: Folder created — `practice.css` split into layout/timer/question/feedback/editor/misc. `timer.css` rewritten from the old timed-mode toggle to the rigid session setup/status UI. Added `#practice-submit-area.hidden` specificity fix.
