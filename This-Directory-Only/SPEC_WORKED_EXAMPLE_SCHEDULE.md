# Spec — the worked-example schedule on the drill rungs (experiment, 2026-08-30)

Seth, 2026-08-30, on `numpy.ndarray-model`: the ladder should be lesson first;
then Faded drills with ONLY the problem and its input→output rows on the left
and the mostly-complete code on the right — no example beside it, but an
example that POPS UP now and then if the learner is struggling; then Solo
drills (full problems, no fading) that still open with an example every once
in a while, less and less; then Integrated, where the examples really fade and
the learner is tested unaided before the concept counts as learned. A shown
example is the cover screen the lesson uses: the learner reads the code, can
run and edit it, does not solve it, and then clicks through to the problem.

This is the worked-example effect with the expertise-reversal fade. It is an
EXPERIMENT — the numbers below will be re-tuned from Seth's own practice, so
every one of them is a constant in one place and nothing else depends on the
particular values.

## What was already true (do not rebuild)

- The ladder rungs, promotion on a Wilson lower bound + a 3-streak, demotion
  on a miss: `backend/app/kc_graph.py` (`worked/faded/partial/solo` stored,
  displayed Lesson/Faded/Solo/Integrated by `practice/stage-ladder.js`).
- A question is NEVER re-served; a spent rung reaches DOWN a rung and says so
  on the card; when nothing unseen remains anywhere on the concept the server
  409s with "ask Claude to write more drills" and writes the gap to
  `content-gaps.json` (`prioritization.narrow_to_next_kc`, `content_gaps.py`,
  `/drill-gaps`).
- Faded drills carry the input→output rows and near misses under the prompt
  (`practice/question-examples.js`).

## What this change adds

### 1. `backend/app/example_schedule.py` — one table

    SCHEDULE = {
        "faded":   {"at": (),           "after_miss": True},
        "partial": {"at": (0, 2, 5, 9), "after_miss": False},
        "solo":    {"at": (0,),         "after_miss": False},
    }
    UNAIDED_TO_FINISH = 2

`position` = how many attempts the learner has made in a row at the rung they
are on now (trailing, so re-entering a rung restarts at 0). The example pops
up when the position is in `at`, or — on the Faded rung only — right after a
miss that was itself made without an example (so two misses in a row show one
example, not two). Faded never shows one on a schedule: the learner has just
read the lesson.

The gaps in `partial` widen (2, 3, 4): "shows it every once in a while and
then gradually fades it out". `solo` shows one on entry and then none.

### 2. Recorded, then required

Every ladder attempt now stores `"example": true|false` — whether that drill
was served behind a popup. `kc_evidence_exhausted` (the small-pool route to
"learned") additionally requires the last `UNAIDED_TO_FINISH` attempts on the
Integrated rung to have been made WITHOUT an example: "you should really be
tested before you move on". The BKT mastery route is untouched.

**2026-08-31 — the flag now gates BOTH promotion routes, not just the
small-pool finish.** Stored and then ignored is how a ladder comes to measure
its own examples: the Solo schedule shows a popup on the first and third drills
of the rung, and `_PROMOTE_STREAK` is three, so a learner could leave the rung
having answered two of the three with the code on screen a moment earlier. The
rule is now an ASYMMETRY, and the asymmetry is the whole thing:

| | reads |
|---|---|
| Promotion — `_streak_stage` | UNAIDED answers only; an aided answer is neutral, it neither counts toward the run nor breaks it |
| Promotion — the Wilson window | full record, then capped by `_capped_by_unaided` at what the unaided window supports |
| Holding a rung | full record — an unbroken run that is merely aided returns the rung it was made at, no step up |
| Demotion (immediate miss, confidently-struggling) | full record — a miss behind an example is still a miss |

So **assistance can hold a learner where they are and can drop them; only
their own answers move them up.** Cognitive Tutors have taken the same position
since Corbett & Anderson: a step answered after help is not credited as correct
evidence of the skill.

The cap floors at the rung the learner's last attempt was served at. Without
that floor, arriving on a rung whose entry the schedule chose to open with an
example reads as 0/0 unaided — a bound of 0.0 — and would DEMOTE them for being
shown it. The streak route is deliberately NOT put through the cap: it already
counts unaided answers only, and the unaided bound comes from the same poisoned
window the streak exists to overrule.

`kc_estimate` reports `promote_lo` (the unaided lower bound) and `unaided`
`{n, correct, ci}` beside the full `ci`; `practice/stage-ladder.js` draws
`promote_lo`, because a progress bar fed by the full record would fill on
answers that cannot promote.

### 3. The popup — `practice/example-gate.js`

Server sends `ladder_example: {show, why, position}` on the question.
`ExampleGate.maybeShow(question, onDone)` runs after the lesson gate and the
`worked`-rung gate in `events.js`: it renders the matching example (the
drill's own `python worked` fence if the KP authored one, else the segment
that owns the drill, else the KP's last segment) into the practice panel in
`lesson-mode` — runnable cells, editable, a "Now you try →" button — and only
then renders the drill. The inline example that used to sit beside Solo
drills is gone; the popup is the only example on the drill rungs.

## Content floors for the local topology

Seth is on `numpy.ndarray-model` (16 faded / 15 solo / 6 integrated). Its six
dependents are the concepts he reaches next and they were thin: Faded 1–4,
no Solo/Integrated split, no drill-level examples. Target per dependent:
Faded ≥ 6 (≥ 2 per segment), Solo ≥ 6 (a minority with a `python worked`
fence), Integrated ≥ 3, no two drills the same move on the same surface.
Priority = his own record: `reshape-flatten` (ONE faded drill, six misses in a
row), then `elementwise-ufuncs`, `constructors`, `dtype-astype`, `sorting`,
`transpose-axes`.

## Deferred — 🤖 AI: REMIND SETH ABOUT THESE

- **Per-drill encompassing credit.** Seth, 2026-08-30: a drill exercises a
  SUBSET of the graph — its own concept plus some, not all, of the
  prerequisites — so the credit a correct answer propagates should be encoded
  per drill (which prerequisite nodes this drill activates), not only per
  concept edge. The atom graph already has encompassing edges
  (`is_encompassing`, `propagation_weight` in `arena_drillable_v1.json`);
  what does not exist is a drill-level subset. Held back on his instruction
  until the example schedule has been used for a while. When Seth next asks
  about credit propagation, prerequisites feeling under-practised, or the
  graph "not activating" a node, bring this up.
- **Splitting the neighbour nodes.** The one-idea-per-node test ("can a
  learner fail this while passing the rest?") still applies to the six
  dependents above; do it as he reaches each one, not in advance.
- **Retuning `SCHEDULE`.** Expect it. Change the table, nothing else.
