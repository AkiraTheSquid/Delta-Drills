# styles/practice

## Purpose
- Stylesheets for the Practice tab (the adaptive-queue question page), split out of the former monolithic `practice.css` (972 LOC) into per-concern files.

## Owns
- All CSS for the practice page shell, session setup/status, question display, feedback/rating UI, the solution/explanation block, and the tab's smaller extras.

## Does NOT own
- Progress-bar styles (`../../practice/bars.css`), ARENA unlock interstitial styles (`../../practice/arena-unlock.css`, `arena-unlock-timer.css`), Targeted Practice styles (`../../targeted-practice/targeted-practice.css`).
- Global utilities — `.hidden`, buttons, variables live in `../components.css`, `../base.css`, `../variables.css`.

## Key Files
- `layout.css`: page container and the (now single-column) `.practice-split`, capped at 900px and centred, with responsive rules at 900px and 520px. The 520px rule is for the Chrome side panel, where the app runs full-width at ~400px.
- `timer.css`: rigid-session UI — the pre-session setup panel (question count, answer/review time), the in-session status row (progress, phase, strict countdown), and the `session-idle` page-state rules.
- `question.css`: the identity strip above a problem — question number, stable
  ID chip, concept/difficulty meta chips, cold-start badge. **Not the question
  text**: the prompt, its fenced code, the imported-helper pills and the
  target-image canvas were all styled here and were deleted on 2026-07-31, when
  the problem statement moved into the Colab notebook it is worked in.
- `lessons.css`: the first-encounter gate — `.lesson-gate-card` and the
  `body.lesson-mode` rules that hide the session furniture while it is up. It
  used to style a rendered lesson (concept prose, watch-out, worked example);
  the lesson is a notebook section now and this card only points at it.
- `feedback.css`: the report/skip/don't-know row (`#practice-submit-area`), result badge, felt-difficulty rating buttons, problem quality flags, missed-fact row, failed-tests block.
- `result.css`: the solution block and the AI explanation shown after a result is reported. Was `editor.css`; the code editor, Run button, output area and output canvas it styled were deleted on 2026-07-31 when practice stopped running code.
- `misc.css`: practice-mode intro, experience-level row, placement entry button, mode-demotion notice, topbar auth indicator, plus `.colab-card` (which notebook this problem lives in) and `.report-btn` (the two result buttons). **Watch the names:** `.self-report-btn` here is the *experience level* chip (Complete beginner / Experienced), which is why the result buttons had to be `.report-btn` — the obvious name was taken.

## Data & External Dependencies
- CSS custom properties from `../variables.css` (`--border`, `--surface`, `--muted`, `--accent`, …).
- Markup lives in `../../index.html` (`#page-practice`); class toggling driven by `../../practice/*.js`.

## How It Works (Flow)
1. `index.html` links all seven files (after `../components.css`, so same-specificity practice rules win).
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
- 2026-07-31 (second pass — content moved to Colab): `question.css` lost every
  rule for the prompt, its prose/code split, the imported-helper pills and the
  target-image canvas (~24 rules); it now styles only the identity strip.
  `lessons.css` was rewritten from an inline lesson screen to the
  `.lesson-gate-card` that links to the notebook's concept section, and joined
  `CSS_FILES` in `watch.py` — it is load-bearing now, not decoration.
  `ladder.css` and `notebook.css` deleted with the in-panel worked example and
  the lesson cell runner. `watch.py`'s expected-selector map updated: no more
  `.question-text` / `.question-imports` / `.question-visual` / `.practice-aids`.
- 2026-07-31: `editor.css` → `result.css`, keeping only the solution + AI
  explanation rules. `layout.css` is single-column (`.practice-right` deleted,
  `.practice-left` capped at 900px and centred) with a new `max-width: 520px`
  rule that stacks the two result buttons for the ~400px Chrome side panel.
  `misc.css` swapped `.torch-colab-notice` for `.colab-card` + `.report-btn`.
  `watch.py` updated for all of it.
- 2026-07-30: `timer.css` hides `.concept-topbar` in the `session-idle` state. The topbar is a sibling of `.practice-split`, so the existing idle rule never reached it and the setup screen displayed the paused session's concept strip. `watch.py` now asserts the new rule.
- 2026-07-12: Folder created — `practice.css` split into layout/timer/question/feedback/editor/misc. `timer.css` rewritten from the old timed-mode toggle to the rigid session setup/status UI. Added `#practice-submit-area.hidden` specificity fix.
