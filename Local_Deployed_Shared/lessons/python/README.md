# python — lesson py-0, the floor of the graph

## Purpose
- The seven knowledge points a learner needs *before* any library exists for
  them: what `=` does, the everyday types, lists and tuples, indexing from
  zero, calling a function, writing one with `def`/`return`, and what a dot
  means (import, attribute, method).
- It exists because `numpy.ndarray-model` — the first concept of the course —
  already assumed all of it. A beginner opening the app was handed a drill
  reading "write a function `solve(rows)`" with nothing behind it.

## Owns
- The seven `python.*` KCs and their KP pages, and nothing above them.
- The claim that `numpy.ndarray-model` has prerequisites at all: its
  `prereqs` in `kc_registry.json` point here.
- Drill ids 568–609 (42 of them), Topic `Python` / Subtopic `Getting started`.

## Does NOT own
- Anything tensor-shaped. The moment a page needs `torch`, it belongs in
  `../numpy/`.
- Loops, comprehensions, classes, files, exceptions-as-control-flow. Explicitly
  out of scope: the floor is what the FIRST numpy lesson assumes, not a Python
  course.
- The ladder machinery, fading, and question selection — `scripts/` and
  `This-Directory-Only/backend/app/`.

## Key Files
- `kp-values-and-names.md` → `kp-types-and-conversion.md` →
  `kp-lists-and-tuples.md` → `kp-indexing.md`, and
  `kp-calling-functions.md` → `kp-defining-functions.md` /
  `kp-dots-and-imports.md`. That is also the prerequisite order.
- `watch.py`: the four invariants below, executable.
- The registry entry and lesson row live in `../kc_registry.json`; the hover
  definitions in `../glossary.js`.

## Data & External Dependencies
- `../kc_registry.json` (KC ids, titles, prereqs, the `py-0` lesson row).
- `../glossary.js` — `kcLesson` must carry every KC here or the jargon popup
  loses its "Taught in" line; `watch_jargon.py` fails the build on that.
- `../qmatrix_tags.json`, built from these pages' frontmatter by
  `scripts/build_qmatrix.py` (its `EASY_TOPICS` includes `"Python"`).
- The drills themselves: `This-Directory-Only/csv files of problems/curated_additions.csv`
  plus hand-authored payloads in `This-Directory-Only/chatgpt/curated_overrides.jsonl`.
- No torch, no numpy, no einops — see the invariant below.

## How It Works (Flow)
1. `kc_registry.json` puts `python.values-and-names` at the root, so a brand-new
   account's frontier is exactly that one concept.
2. `compile_lessons.py` turns these pages into `lessons_structured.json`;
   `build_qmatrix.py` derives the question→KC tags from the frontmatter.
3. The backend serves the Lesson rung first (`kc_graph._stage_from` returns
   `worked` while `worked_seen == 0`), then the faded drills from this folder.
4. Clearing `python.indexing`, `python.defining-functions` and
   `python.dots-and-imports` unlocks `numpy.ndarray-model`.

## Invariants & Constraints
- 🔴 **Every construct a py-0 drill uses must be taught by one of the seven
  concepts, at or before the drill's own.** This folder is the FLOOR, so a
  drill reaching past it has nothing earlier to be taught by — that is what
  the three ACTIVE cases below are. Enforced by `watch.py` via
  `scripts/guard_checks.py` scoped to `python.`, as a ratchet against
  `scripts/solution_prereq_baseline.json`. The ARENA-grounding half runs too
  and should stay silent here: the floor teaches no library at all, and a
  page declaring torch symbols has stopped being the floor.
- **Exactly one root.** `python.values-and-names` has no prereqs and every
  other KC in the registry is reachable from it.
- **Nothing here imports a library.** Checked over the page bodies; a page that
  reaches for torch or numpy is teaching above its own level.
- **`numpy.ndarray-model`'s prereqs are all `python.*`.** If a numpy concept
  ever appears there, the floor has stopped being the floor.
- 🔴 **No python ATOM may gate the numpy course.** The concept-graph atoms in
  `arena_drillable_v1.json` are a separate mechanism from these registry
  prereqs, and `bkt_mastery.item_is_unlocked` requires *every* gating
  prerequisite atom to be ready. Wiring a python atom under
  `tensor-wraps-ndarray` would lock every existing account out of numpy until
  it cleared 42 brand-new drills. `is_hard_gate` is **not read at runtime**;
  the only escape hatch is `NON_GATING_ATOMS`.

## Extension Points
- Adding a concept: append the KC to `../kc_registry.json` (with `lesson:
  "py-0"`), add a `kcLesson` row to `../glossary.js`, write `kp-<name>.md` in
  the same shape as the others, append drills to `curated_additions.csv` (LAST
  file in `CSV_SOURCES` — ids are positional), then re-run
  `compile_lessons.py`, `build_qmatrix.py`, `export_questions_json.py`,
  `export_kc_atom_crosswalk.py` and `audit_question_bank.py --gate`.
- Authoring drills: write the answer, then *execute* it to derive every
  expected value and every near-miss output. Hand-computing them produced seven
  defects in the first pass of this folder.

## Known Issues, Recurring Bugs, and Pain Points (and How to Prevent Them)

- **The library-free floor gets parked out of serving** — `RESOLVED`
  - When it happens: any time these drills are added or re-tagged.
  - Symptom: a brand-new account gets **no question at all** —
    `/api/practice/next-question` 404s "No questions available", the frontier
    reads `['python.values-and-names']`, and every other KC is locked.
  - Root cause: `lessons.torch_only_serving()` parks any question whose code
    does not import torch, to stop an un-converted numpy drill following a
    torch lesson. These drills import nothing *on purpose*, so they were
    parked — and a concept with no servable questions can never become
    learned, so the lock never lifts.
  - Prevention/fix: `lessons.is_prelibrary(question_id)` exempts questions
    whose every target KC belongs to a `Python`-topic lesson. Verify with a
    cold-start `select_next_subtopic(UserPracticeState(...))` before shipping
    any change to the registry root.
  - Status: `RESOLVED` 2026-08-28.

- **Drills here use syntax this floor never teaches** — `ACTIVE`
  - When it happens: authoring a drill for a py-0 concept and reaching for the
    idiomatic Python spelling.
  - Symptom: a learner who has never coded is handed `kind in ("int", "float")`
    (q578), `tuple(len(row) for row in rows)` plus `set(...)` (q585), or an
    `if`/`else` (q603) — membership, a generator expression, a set, and a
    conditional, none of which is a KC here. This is the same class of failure
    the whole py-0 floor exists to fix, one level down.
  - Root cause: the seven concepts cover names, types, sequences, indexing and
    functions. They do NOT cover comparison, booleans, `in`, iteration or
    branching, and nothing checks a drill's body against the concepts the graph
    has actually reached by that point.
  - Prevention/fix: either rewrite those three using only taught constructs, or
    add the missing KCs (comparison + `True`/`False` at minimum) — deliberately
    NOT loops, per the brief for this folder. Found by codex 2026-08-28; left
    open because the ndarray split took priority.
  - Status: `ACTIVE` — q578, q585, q603.

- **A new atom silently locks the whole course** — `ACTIVE`
  - When it happens: adding these concepts to the concept graph as atoms.
  - Symptom: existing accounts stop being served numpy questions entirely.
  - Root cause: prerequisite atoms are ALL-of gates at serve time.
  - Prevention/fix: keep the ordering in `kc_registry` prereqs only;
    `check_no_python_atom_gates_the_numpy_floor` in `watch.py` fails the build
    if a python atom becomes a prerequisite of a numpy one.
  - Status: `ACTIVE` — the check is the only thing standing between this and a
    production outage.

## Recent Changes
- 2026-08-29: `watch.py` gained the two standing content guards. The three
  drills recorded ACTIVE below (q578, q585, q603) are now MECHANISED —
  they sit in the recorded baseline rather than being remembered in prose,
  and a fourth drill of the same kind is refused at the point it is written.
  Fixing those three should SHRINK the baseline; that is the intended
  direction and needs no re-record.
- 2026-08-28: Folder created. Seven KPs, 42 drills (ids 568–609), the registry
  rewire that gives `numpy.ndarray-model` prerequisites for the first time, and
  `watch.py`'s four invariants. `lessons.is_prelibrary` added on the backend so
  the dialect gate stops parking library-free drills.
