# Splitting the blob nodes: 37 concepts → ~45

*Written 2026-08-31. This plan has been derived, lost to a context compaction,
and re-derived at least twice. That is the reason it is a file: the measurement
is cheap to redo but the **decision** — what the test for a node boundary is,
and why the obvious test is wrong — costs a long conversation each time.*

---

## The problem, in the app's own terms

The graph has 37 concepts. Between them they declare **144 symbols** in
`new_syntax`, and the distribution is not flat. Early nodes carry one or two
new things; a handful of later ones carry eight, nine, ten — and those same
nodes were never cut into segments and never got drills written for them.

Both mastery models estimate **one number per concept**. BKT keeps a single
`p(known)` per KC; the logistic engine keeps a single ability per KC. The
lattice gates on that number. So a node declaring ten symbols is making one
claim covering all ten, and the learner is promoted past all ten on evidence
from whichever two or three the drills happen to touch.

`scripts/audit_symbol_coverage.py` measures exactly that gap. As of
2026-08-31, **51 of 144 declared symbols are drilled fewer than twice on their
own concept, and 19 are drilled zero times.** The debt is concentrated:

| KC | symbols | segments | drills | drills/symbol | under floor | at zero |
|---|---:|---:|---:|---:|---:|---:|
| `numpy.stack-concat-interleave` | 10 | 1 | 5 | 0.5 | **10** | 3 |
| `numpy.random-generator` | 10 | 1 | 3 | 0.3 | **7** | 4 |
| `python.dots-and-imports` | 9 | 1 | 6 | 0.7 | 5 | 1 |
| `python.types-and-conversion` | 6 | 1 | 6 | 1.0 | 4 | 2 |
| `numpy.slicing-views` | 6 | 1 | 8 | 1.3 | 3 | 0 |
| `python.calling-functions` | 6 | 1 | 6 | 1.0 | 0 | 0 |

Compare the nodes that have had the full treatment: `numpy.ndarray-model` —
11 symbols, 3 segments, 37 drills, **zero** under the floor.

Two different problems live in that table and they need to be named separately:

* **Under-drilling.** Ten symbols and three drills. Splitting the node does not
  create a single drill; it only makes the shortage honest, by turning one
  unbacked claim into three smaller claims that are each visibly unbacked.
* **Over-claiming.** One number standing behind ten independent skills. This is
  what splitting fixes, and it cannot be fixed by writing drills.

A split is worth doing when the second problem is real. Where only the first is
real, write drills and leave the node alone.

---

## The test for a node boundary

The tempting rule is **one function per node**. It is wrong, and the graph
already contains the counter-example: **7 of the 11 `einops.*` nodes declare
zero `new_syntax` symbols at all.** They teach a *pattern* — `pooling`,
`grids-montage`, `split-axes` — spelled with `rearrange`, which an earlier node
introduced. Under a one-function rule those seven nodes should not exist, and
they are among the most useful in the course, because ARENA is full of exactly
those patterns and nothing else teaches them.

The rule that survives contact with both halves of the graph:

> **Can a learner fail this while succeeding at the rest of the node?**
> If yes, it is a separate concept. If no, it is one concept with several
> spellings.

That is the same question the mastery model is implicitly asking, which is why
it is the right one: independently-failable skills need independent estimates,
or the estimate is an average of things that were never averaged.

Worked examples of the test:

* `torch.Generator` + `manual_seed` (making a reproducible stream) vs
  `torch.randn(..., generator=g)` (threading it into a call). A learner can seed
  correctly and never thread it, and does — that is the classic bug. **Split.**
* `torch.rand` vs `torch.randn` vs `torch.randint` vs `torch.randperm`. Four
  spellings of "draw from a distribution", same call shape, and someone who can
  do one can do the others from the docs. **One node.**
* `int()` / `float()` / `bool()` / `str()` — four spellings of one idea.
  **One node**, which is why `python.types-and-conversion` stays whole.
* `torch.stack` vs `torch.cat`. The single most-confused pair in the course:
  one adds an axis, the other does not. Failing one while passing the other is
  the *normal* case. **Split.**

---

## The plan: 37 → ~45

Eight new nodes, three symbol evictions, one merge.

```
numpy.random-generator          -> 3   the four samplers (rand/randn/randint/randperm)
                                       seeding your own stream (DONE 2026-08-31)
                                       using a stream you were handed
numpy.stack-concat-interleave   -> 2   stack vs cat (does it add an axis?)
                                       hstack/vstack/column_stack conventions
                                  evict torch.empty + torch.empty#dtype -> numpy.constructors
                                  evict Tensor.ravel                    -> numpy.reshape-flatten
numpy.slicing-views             -> 2   slice syntax + multi-axis indexing
                                       reversal / flip / rot90
                                  merge its views half INTO numpy.views-and-copies
python.dots-and-imports         -> 2   attribute and method access (the dot)
                                       import namespaces (import, import-as, module members)
python.calling-functions        -> 2   calling + positional arguments
                                       keyword arguments
python.types-and-conversion     -> 1   LEAVE — one idea, four spellings
```

**Evictions matter as much as the splits.** `torch.empty` is a *constructor*
sitting on the stacking node because one stacking drill happened to allocate
with it; `Tensor.ravel` is *flattening*. Each is a symbol whose node cannot
drill it because the node is not about it. Moving the symbol to the node that
already drills the idea fixes the coverage gap without writing anything.

**Ordering, and one dissent from it.** Seth approved `numpy.random-generator`
first, on the reasoning that it is the largest unbacked mastery claim and gates
nothing downstream (it has **zero dependents** in the registry, verified
2026-08-31), so a bad split there cannot cascade. The gating argument still
holds and it is the right first cut. But on the fresh measurement
`numpy.stack-concat-interleave` is the worse node — all ten of its symbols are
under the floor against random-generator's seven, and it is the one carrying
the stack/cat confusion. It should be second, not later.

`python.calling-functions` is last: it is the only blob node that already
**passes** the coverage floor, so the case for splitting it rests purely on the
independently-failable test, not on missing evidence.

---

## What the first split changed about the plan

Two things the plan did not anticipate, both worth carrying to the next one.

**The teaching order can be forced by the symbols, not by the pedagogy.** The
plan said seeding first, then the samplers, then threading — which reads well
and is impossible. Whichever of "seed your own generator" and "thread a
generator into a call" comes first needs the other's symbols: you cannot
demonstrate a seeded stream without drawing from it, and you cannot draw from
it without `generator=`. The prerequisite guard says so out loud, and it is
right. So the node that OWNS the symbols has to come first, and the split
landed as samplers → seeding → threading.

**A node can be worth having with no symbols of its own.** Once seeding owned
`torch.Generator`, `Tensor.manual_seed` and all four `generator=` kwargs,
threading had nothing left to declare — and it is still the most valuable of
the three. It teaches the discipline: *use the generator you were handed; do
not make one, do not reseed it*. Both mistakes run fine and return plausible
numbers, which is exactly the kind of failure that deserves its own mastery
estimate. It is the same shape as the seven `einops.*` pattern nodes. When a
split leaves one half with no symbols, that is a signal the half is about a
practice rather than an API — not a signal to abandon the split.

A consequence for the coverage guard: a node declaring nothing has nothing to
check, so the guard is silent on exactly the nodes where the drills carry the
whole burden. Their floor is the rung floor (Faded ≥ 2, Solo ≥ 6,
Integrated ≥ 3), which nothing currently enforces per node.

## What a split actually costs

Not a registry edit. Every one of these, or the split is worse than the blob:

1. **Registry entry** in `Local_Deployed_Shared/lessons/kc_registry.json` —
   `id`, `lesson`, `title`, `syntax`, `prereqs`.
2. **Prereq edges**, both directions. The new nodes chain to each other in
   teaching order, the first inherits the old node's prereqs, and anything that
   depended on the old id must be re-pointed. Leaving an id nothing depends on
   and nothing reaches is how a node becomes unreachable in the lattice.
3. **A KP page per node**, with the full four-rung ladder — Lesson, Faded,
   Solo, Integrated — per `Local_Deployed_Shared/lessons/AUTHORING.md`. A new
   node with one rung is a node the learner gets stuck on.
4. **Drill re-tagging** in `qmatrix_tags.json`, and almost always **new
   drills**: the split divides the existing drills between the new nodes, so a
   node that had three now has one. The coverage floor is 2 per symbol.
5. **All three guard baselines** re-recorded — prereqs, ARENA grounding, symbol
   coverage — because the KC ids in their keys change. Re-recording is the
   mechanism for admitting debt on purpose; say in the commit what moved and
   why, and check the counts moved the direction you expected.
6. **Everything else that names the id.** The registry is the least of it. The
   2026-08-31 split had to touch the glossary (`lessons/glossary.js`, both the
   term's `kc` and the lesson map), the retirement manifest and its archive
   watcher, `concept-graph/kc_atom_crosswalk.json` and `kc_difficulty.json`
   (regenerate — a registry KC missing from the crosswalk fails its watcher),
   and `backend/app/data/question_atom_tags.jsonl`, because a brand-new node
   whose drills are all brand-new has no atom tags and therefore no crosswalk
   entry at all. The watchers found every one of these; none was predicted.

   One more that NO watcher finds, because it lives in generated Colab
   artifacts: `scripts/generate_colab_notebooks.py` writes the old concept id
   into `lessons/colab_notebooks.json`, `lessons/notebooks/manifest.json` and
   `extension/panel/notebook-index-concepts.js`. After a split those hold a
   dead key and no key for the new nodes, so the in-app Notebooks tab and the
   Colab fork have nothing to open for them. Regenerate as part of the split —
   it was NOT done on 2026-08-31 because the Colab lane was already failing its
   own preflight for unrelated reasons and another session was mid-deploy.
7. **Seth's stored `kc_ladder` still names the old id.** It loads and resolves
   (verified during the 2026-08-30 ARENA cut), but his progress on the old node
   does not transfer to the new ones. Splitting a node he is standing on resets
   him on it. Read his position off Fly before cutting — the procedure is in
   `CLAUDE.md`, and the rule is one concept at a time.

---

## Where this stands

- [x] Measured, and the measurement is now standing: `audit_symbol_coverage.py`,
      wired into all five watchers via `scripts/guard_checks.py`.
- [x] Plan written down (this file).
- [x] `numpy.random-generator` → 3, **done 2026-08-31**. `numpy.random-samplers`
      → `numpy.random-seeding` → `numpy.random-threading`, 32 new drills
      (676–707), every rung floor met (Faded ≥ 2, Solo ≥ 6, Integrated ≥ 3) and
      **zero symbols under the coverage floor on all three nodes**. The old page
      is archived and recorded under a new `split_kcs` key in the retirement
      manifest — a split is not a retirement, and filing it as one would tell
      the next reader that random numbers left the course.
- [ ] `numpy.stack-concat-interleave` → 2, plus the three evictions.
- [ ] `numpy.slicing-views` → 2, plus the merge into `views-and-copies`.
- [ ] `python.dots-and-imports` → 2.
- [ ] `python.calling-functions` → 2.
- [ ] **Nothing in the course teaches `*args` or `None`.** Found while recording
      the 2026-08-31 prereq baseline: `syntax.star-args` is used, untaught, by
      drills on 138 concepts and `syntax.none` by drills on 352 — the two
      largest single entries in the prerequisite debt, and every new drill that
      writes `def f(*shape)` or `generator=None` adds to them. That is a missing
      node in the python course, not a per-drill defect, and no split will
      close it.
