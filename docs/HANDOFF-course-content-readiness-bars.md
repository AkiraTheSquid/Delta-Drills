# Handoff — Course content tab: per-section readiness bars

**Status:** partially built, uncommitted. 2026-08-09.
**Read this before grepping.** It exists because the session that produced it
spent most of a context window rediscovering facts that were already written
down — in `graphify-out/`, in file header comments, and in folder READMEs.

---

## Start here, in this order

1. **`graphify query "<your question>"`** — from the repo root. The Python
   backend is **96% covered** by the graph and answers in ~700 tokens what
   takes a dozen greps. One query returned the entire LKT prediction path
   (`engine_bridge.predict()` L249, `logistic_engine.predict()` L576,
   `Prediction` L548, `attempt_log.record_attempt()` L238).
2. **Read the file header comment** of anything the query points at. This repo
   puts its real design rationale there, not in commit messages.
   `extension/content/colab.js`'s header already documented the Colab
   anchor-resolution rules that the session went to a live browser to confirm.
3. **Then** grep — only for what the first two missed.

### The frontend blind spot — FIXED 2026-08-09, and why it matters

The graph used to cover the Python backend at 96% but `Local_Deployed_Shared`
JS at only 55%, with `lesson-graph.js`, `prereq_subtopics.js`, `kc_interval.js`
and every other file in `concept-graph/` **silently absent**. Querying the
graph about them returned nothing, which reads as "this does not exist" — it is
what sent one whole session back to grep.

🔑 **Root cause: `graphify` honours `.gitignore`, exactly.** Measured: 39 of 87
JS files gitignored, 39 missing from the graph, **0 gitignored files present,
0 non-ignored files missing**. Not a parser bug, not the `#1666` empty-cache
warning (a red herring about JSON files), and not the graphify version —
upgrading 0.9.9 → 0.9.37 changed nothing.

`Local_Deployed_Shared/concept-graph/` was ignored wholesale while **13 of its
16 files were already force-added**, so the rule was a dead letter that only
did harm. Fixed by ignoring the three generated blobs by name
(`graph-viz.json`, `graph-audit.html`, `iter5_v2.js`) instead of the folder.
JS coverage **55% → 65%**; `graphify query "mastery bar render confidence
interval"` now returns `mastery-bar.js render() L66` directly.

🔴 **`docs/` is still ignored the same way** (`.gitignore:71`, 138 files
force-added, 12 still untracked). Same trap, not yet fixed — this handoff file
itself needed `git add -f`. If you add anything under `docs/`, force-add it or
it will not exist for anyone else.

⚠️ **The general rule this teaches:** in this repo, a gitignored folder that
contains shipping code is a triple hazard — invisible to git, invisible to the
deploy image, and invisible to the code graph. Two live examples were found
untracked-but-shipping on 2026-08-09: `kc_interval.js` and
`kc_crosswalk_mastery.js`, both loaded by `index.html:933-934`. Now tracked.

---

## What is done (uncommitted, 4 modified + 1 added)

- **Rename `Courses` → `Course content`** — `index.html` tab label + info
  `aria-label`, `infotips-registry.js` title. Internal keys
  (`data-tab="courses"`, `data-dd-info="tab.courses"`, `styles/courses/*`)
  deliberately untouched: they are routing, not copy.
- **`concept-graph/mastery-bar.js` (new)** — the mastery bar's markup, lifted
  out of `lesson-graph.js`'s private `_masteryBar` so the Course content tab
  can draw the same bar. `window.DeltaMasteryBar.render({value, ci, measured,
  gates, unknownColor})`. Verified **5/5 byte-identical** to the original
  across mid-range, inferred-wide-band, `NaN`, tight-interval-floor, and
  no-band cases. See that folder's README for the full entry.

🔴 **`concept-graph/` is gitignored wholesale (`.gitignore:87`).** Every file
in it that ships was force-added. `mastery-bar.js` needed `git add -f` or it
would have 404'd in production while working locally — `window.DeltaMasteryBar`
undefined, `_masteryBar` throws, Knowledge Graph dock breaks. **Any new file in
that folder has the same trap.** This is the same failure shape as the
2026-08-06 crosswalk incident recorded in `concept-graph/README.md`.

---

## What is left

### 1. Expose the LKT prediction over HTTP (the actual blocker)

Nothing serves `predict()` — it is computed during scoring and prioritization
only. Needs a read-only router returning, per KC:

```python
# engine_bridge.predict(user_state, kc, difficulty_score=..., stage=...) -> E.Prediction
# Prediction fields: p, p_mean, logit_mean, logit_var, contributions
value = pred.p                                    # uncertainty-attenuated P(correct)
ci    = (sigmoid(pred.logit_mean - z * sqrt(pred.logit_var)),
         sigmoid(pred.logit_mean + z * sqrt(pred.logit_var)))
```

The interval is **asymmetric in probability space and comes straight from the
model** — no Wilson approximation for this bar. `engine_bridge.mastery()` is
P(correct) on a median-difficulty SOLO item, the natural per-KC reading.

### 2. Wire the section bar

`courses.js` → `buildSectionItem`, insert after `info.appendChild(d)`:

```js
window.DeltaMasteryBar.render({ value, ci, measured })   // gates OMITTED — see below
```

⚠️ Section paths need a **`content/ARENA_5.0-main/` prefix** to match
`window.ARENA_EXERCISES_BY_NOTEBOOK` keys. `notebookPathForBookUrl()` returns
the path without it.

🔴 **Leave `gates` off for a section bar.** 85% / 95% are *one concept's*
thresholds. A section is an aggregate over many exercises and has no such gate;
drawing the tick labels a threshold that does not exist there.

🔴 **Aggregating intervals across a section's KCs:** average the per-KC
*bounds*. Do **not** treat KCs as independent and shrink variance by `1/n` —
that draws a tight band the model never claimed.

---

## Decisions already made — do not relitigate

- **Bar shows true `P(correct)`**, not `P(known)`, and not a rescale. Seth
  chose this explicitly.
- **Do NOT apply BKT's guess/slip emission to a non-BKT quantity.**
  `bkt_mastery.py:43` has `P_GUESS=0.20`, `P_SLIP=0.10`, so
  `P(correct) = 0.20 + 0.70·L` — but that is only valid on a BKT posterior `L`.
  The subtopic `p` reachable via the ARENA prereq map is **EWMA accuracy**,
  already a P(correct); transforming it double-counts the guess floor.
- **The LKT engine is wired and runs *beside* BKT, deliberately** — see
  `engine_bridge.py`'s docstring. BKT still owns the atom posteriors, the
  unlock lattice and the Statistics panel. Reasons: engine weights are v0 and
  unfitted, and a miscalibrated engine gating the lattice could lock a learner
  out of the course.
- **Glicko is not a pending migration.** `logistic_engine.py:18-28` — the
  additive-logistic model *generalises* Elo/Glicko; line 66: *"it reduces
  exactly to Glicko's shape"*. Design doc:
  `ITS-procedural-AI-SYNC/glicko-vs-lkt-mastery-engine.md`. Replacing BKT with
  Glicko would be a step **down** from what is already there.
- **The open question worth answering first:** `attempt_log` is now being
  written, so a Brier / reliability curve is finally computable. That check —
  not more UI — is what decides whether the engine can take over the unlock
  lattice too.

---

## Deferred to a later slice (Seth's call)

Timed ARENA attempts: press start → countdown from ARENA's own suggested time →
expiry auto-submits as wrong → model updates.

Already verified as feasible:
- **381 of 483 ARENA exercises (78%)** publish `You should spend up to N-M
  minutes` in the *same markdown cell* as the `### Exercise -` heading. The 102
  without cluster in `0.2_CNNs_&_ResNets` and `0.4_Backprop`. Fallback:
  `DEFAULT_TARGET_SECONDS = 240`.
- The timer already exists and already reports what is needed —
  `arena-unlock.js:365` calls `ArenaUnlockTimer.resetForExercise()`, and L412
  POSTs `elapsed_seconds` + `target_seconds` to `/api/practice/arena-rating`.
- Two deltas only: source `targetSeconds` from ARENA's text instead of the
  hand-set catalog value, and change over-target from *stuck-hint banner* to
  *auto-submit as wrong*.

## Extension notes (verified live, 2026-08-09)

The Chrome extension already does what a feasibility question might ask about:
`extension/content/colab_dd.css` injects via the manifest `css` field (the
CSP-safe channel), and `colab_focus.js` already hides/shows cells
(`html.dd-focus`) and themes the page (`html.dd-theme`).

- **Theme is not gated on `dd-` anchors** — it already applies on upstream
  ARENA notebooks today.
- **Focus degrades correctly there**: `live = Boolean(target) && inFocus > 0`,
  and when false it strips `dd-out-of-focus` so the page cannot go blank.
- **Cell anchors:** your generated notebooks carry `metadata.id`, which Colab
  promotes to `<div id="cell-dd-q224-worked">`. Upstream ARENA is nbformat 4.2
  with **0 of 87** cells having ids, so Colab mints random ones
  (`cell--Hxe1Kmvm0p5`) — heading text is the only stable key there, and
  `arena/exercises.js` already carries those titles.
  ⚠️ Normalize backticks: the JSON has ``implement `make_rays_1d` ``, the DOM
  renders `implement make_rays_1d`.
- ❌ **HTML comments do not survive.** The `<!-- dd:dd-q224 -->` markers are
  stripped by Colab's sanitizer (3264 comment nodes on the page, all Lit
  framework markers, zero `dd:`). The `metadata.id` is what does the work.

All of the above is already in `extension/content/colab.js`'s header comment.
Read it before re-deriving it.
