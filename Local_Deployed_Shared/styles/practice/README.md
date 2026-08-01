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
- `colab-edition.css`: the Colab edition's tutor rail. Everything is scoped under `html.dd-colab-edition` (set by `practice/colab_mode.js` on the `delta-drills-colab` deploy) and is inert on the normal app. Strips the prompt, the editor and the worked example — they are in the notebook beside the panel — and re-stacks the concept strip for panel widths. `html.dd-no-notebook` (set by `ui.js`) turns all of it back off for the ~75 questions with no published notebook, which would otherwise get an empty rail.

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
- 2026-07-31: Added `colab-edition.css` (see Key Files). Also fixed a live instance of the invariant above: `layout.css`'s `.practice-right` outweighed `.hidden` by load order, so the panel ignored `classList.add("hidden")` and the editor stayed on screen through torch routing — `.practice-left.hidden, .practice-right.hidden` now re-assert it. Never visible on the normal deploy because no bank question carries a notebook path; found the moment the Colab edition started routing.
- 2026-07-30: `timer.css` hides `.concept-topbar` in the `session-idle` state. The topbar is a sibling of `.practice-split`, so the existing idle rule never reached it and the setup screen displayed the paused session's concept strip. `watch.py` now asserts the new rule.
- 2026-07-12: Folder created — `practice.css` split into layout/timer/question/feedback/editor/misc. `timer.css` rewritten from the old timed-mode toggle to the rigid session setup/status UI. Added `#practice-submit-area.hidden` specificity fix.
